"""Turn a patch clustering into "which clusters look like a lesion".

0003（control欠損クラスタの抽出）で書いた解析を、クラスタリング手法から切り離した
もの。0006でk-means以外の手法（Leiden・DBSCAN）を試すときに、解析側を完全に同じに
保たないと手法の比較にならないため、ここに寄せている。

どの関数もpatchではなく**スライド単位**で効果を測る。同一スライドのpatchは独立では
ないので、patch数のまま検定するとnが数百万になり、どんな些細な差でも有意になる。
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score


def occupancy_matrix(
    slide_index: np.ndarray, assignments: np.ndarray, n_slides: int, n_clusters: int
) -> np.ndarray:
    """Fraction of each slide's patches that fall in each cluster.

    Rows sum to 1. This per-slide normalization is what makes the downstream
    group comparison a comparison of n=slides rather than n=patches.
    """
    counts = np.zeros((n_slides, n_clusters), dtype=np.float64)
    np.add.at(counts, (slide_index, assignments), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    return counts / np.clip(totals, 1, None)


def control_group_by_dose(manifest: pd.DataFrame) -> np.ndarray:
    """Default reference group: untreated animals."""
    if not manifest["dose_level"].isin(["Control", "Low", "Middle", "High"]).all():
        raise ValueError("Unknown dose labels: audit metadata and exclude unknown slides before evaluation")
    if "pathology_label_status" in manifest and manifest["pathology_label_status"].eq("unknown").any():
        raise ValueError("Unknown pathology coverage: filter eligible slides before evaluation")
    return (manifest["dose_level"] == "Control").to_numpy()


def control_group_by_finding(manifest: pd.DataFrame) -> np.ndarray:
    """Reference group: slides with no recorded finding, treated or not.

    dose_levelで切ると、投与されたが所見が出なかったスライド（tier1コーパスでは
    投与群2,570枚中1,323枚）が陽性側に入り、逆に自然発生病変のあるcontrol
    （871枚中40枚）が陰性側に入る。「病変の形態を拾う」のが目的ならこちらが
    直接の対比になる。
    """
    control_group_by_dose(manifest)  # Also reject historical manifests with missing metadata.
    if manifest["has_finding"].isna().any():
        raise ValueError("Unknown pathology labels cannot serve as negative controls")
    return ~manifest["has_finding"].to_numpy().astype(bool)


def cluster_statistics(
    occ: np.ndarray,
    manifest: pd.DataFrame,
    *,
    presence_eps: float,
    min_treated_presence: float,
    logger: logging.Logger,
    reference_selector=control_group_by_dose,
) -> pd.DataFrame:
    """Per-cluster comparison of occupancy between a reference group and the rest.

    The test is Mann-Whitney U over *slides* (not patches): patches within a
    slide are not independent, so testing on patches would put n in the
    millions and make any trivial difference significant.

    `reference_selector` decides what counts as the reference ("control") side.
    Default is dose_level == "Control"; control_group_by_finding() switches it
    to "no finding recorded". Column names keep the control_/treated_ prefixes
    either way.
    """
    is_control = reference_selector(manifest)
    ctrl = occ[is_control]
    trt = occ[~is_control]

    rows = []
    for c in range(occ.shape[1]):
        ctrl_c, trt_c = ctrl[:, c], trt[:, c]
        both = np.concatenate([ctrl_c, trt_c])
        if both.max() == both.min():
            # 全スライドで占有率が同一（多くは両群とも0）。mannwhitneyuはこの入力で
            # 例外を投げるので、差が無いものとして扱う。
            p = 1.0
        else:
            _, p = mannwhitneyu(trt_c, ctrl_c, alternative="greater")

        present_treated = trt_c > presence_eps
        # 「controlに含まれない」の操作的定義。control側の裾(95%点)が閾値未満で、
        # かつ投与側の一定割合以上のスライドに出ていること。
        control_absent = (
            np.quantile(ctrl_c, 0.95) < presence_eps
            and present_treated.mean() >= min_treated_presence
        )

        sub = manifest.loc[~is_control].loc[present_treated]
        rows.append(
            {
                "cluster_id": c,
                "control_median": float(np.median(ctrl_c)),
                "treated_median": float(np.median(trt_c)),
                "control_q95": float(np.quantile(ctrl_c, 0.95)),
                "treated_presence_frac": float(present_treated.mean()),
                "control_presence_frac": float((ctrl_c > presence_eps).mean()),
                "p_value": float(p),
                "control_absent": bool(control_absent),
                # 単一スライド・単一化合物に集中しているクラスタは、所見ではなく
                # 染色ムラやペンマークなどのアーチファクトである可能性が高い。
                "n_treated_slides": int(present_treated.sum()),
                "n_compounds": int(sub["compound_name"].nunique()),
                "n_experiments": int(sub["exp_id"].nunique()),
            }
        )

    stats = pd.DataFrame(rows)
    finite = stats["p_value"].to_numpy()
    stats["q_value"] = false_discovery_control(np.clip(finite, 0, 1), method="bh")
    logger.info(
        f"clusters: {len(stats)}, control_absent={int(stats['control_absent'].sum())}, "
        f"q<0.05={int((stats['q_value'] < 0.05).sum())}"
    )
    return stats


def select_candidates(stats: pd.DataFrame, *, q_threshold: float, min_compounds: int) -> np.ndarray:
    """Cluster ids meeting enrichment criteria; morphology requires visual review."""
    mask = (
        stats["control_absent"]
        & (stats["q_value"] < q_threshold)
        & (stats["n_compounds"] >= min_compounds)
    )
    return stats.loc[mask, "cluster_id"].to_numpy()


def finding_associations(
    occ: np.ndarray,
    manifest: pd.DataFrame,
    candidates: np.ndarray,
    *,
    presence_eps: float,
    max_finding_types: int,
) -> pd.DataFrame:
    """Fisher exact test of "slide carries cluster c" against "slide has finding f".

    This is the step that turns a control-absent cluster into a *named* visual
    token: a cluster that co-occurs with Necrosis and nothing else is a
    necrosis token. Fisher rather than chi-square because many findings appear
    in only a handful of slides.
    """
    exploded = manifest.explode("finding_types")
    top_findings = (
        exploded["finding_types"].dropna().value_counts().head(max_finding_types).index.tolist()
    )
    # スライド×所見の有無を先に作る。クラスタごとに作り直すと candidates × findings 回
    # 同じものを計算することになる。
    finding_matrix = {
        finding: manifest["finding_types"].apply(lambda fs: finding in list(fs)).to_numpy()
        for finding in top_findings
    }

    rows = []
    for c in candidates:
        has_cluster = occ[:, c] > presence_eps
        for finding in top_findings:
            has_finding = finding_matrix[finding]
            table = [
                [int((has_cluster & has_finding).sum()), int((has_cluster & ~has_finding).sum())],
                [int((~has_cluster & has_finding).sum()), int((~has_cluster & ~has_finding).sum())],
            ]
            odds, p = fisher_exact(table, alternative="greater")
            rows.append(
                {
                    "cluster_id": int(c),
                    "finding_type": finding,
                    "n_slides_both": table[0][0],
                    "n_slides_cluster_only": table[0][1],
                    "n_slides_finding_only": table[1][0],
                    "odds_ratio": float(odds),
                    "p_value": float(p),
                }
            )

    assoc = pd.DataFrame(rows)
    if not assoc.empty:
        assoc["q_value"] = false_discovery_control(
            np.clip(assoc["p_value"].to_numpy(), 0, 1), method="bh"
        )
    return assoc


def leave_one_compound_out_auroc(
    occ: np.ndarray,
    manifest: pd.DataFrame,
    *,
    presence_eps: float,
    min_treated_presence: float,
    q_threshold: float,
    min_compounds: int,
    logger: logging.Logger,
    reference_selector=control_group_by_dose,
) -> tuple[float | None, pd.DataFrame]:
    """AUROC of a cluster-based slide score, with cluster selection held out.

    Selecting "control-absent" clusters uses the control/treated labels, so
    scoring the same slides with those clusters is circular and will look
    excellent by construction. Here the selection is refitted on seven
    compounds and applied to the eighth, which is also the question that
    matters in practice: does a cluster found on known compounds transfer to
    a compound the selection never saw?
    """
    scores = np.full(len(manifest), np.nan)
    per_compound = []

    for compound in sorted(manifest["compound_name"].unique()):
        held_out = (manifest["compound_name"] == compound).to_numpy()
        train_manifest = manifest.loc[~held_out].reset_index(drop=True)
        train_stats = cluster_statistics(
            occ[~held_out],
            train_manifest,
            presence_eps=presence_eps,
            min_treated_presence=min_treated_presence,
            logger=logger,
            reference_selector=reference_selector,
        )
        selected = select_candidates(
            train_stats, q_threshold=q_threshold, min_compounds=min_compounds
        )
        if len(selected) == 0:
            logger.warning(f"LOCO {compound}: no cluster selected on the other compounds")
            scores[held_out] = 0.0
        else:
            scores[held_out] = occ[np.ix_(held_out, selected)].sum(axis=1)

        sub = manifest.loc[held_out]
        y = sub["has_finding"].astype(int).to_numpy()
        entry = {
            "compound_name": compound,
            "n_selected_clusters": int(len(selected)),
            "n_slides": int(held_out.sum()),
            "n_with_finding": int(y.sum()),
        }
        if len(np.unique(y)) == 2:
            entry["auroc"] = float(roc_auc_score(y, scores[held_out]))
        per_compound.append(entry)

    y_all = manifest["has_finding"].astype(int).to_numpy()
    pooled = (
        float(roc_auc_score(y_all, scores)) if len(np.unique(y_all)) == 2 else None
    )
    return pooled, pd.DataFrame(per_compound)
