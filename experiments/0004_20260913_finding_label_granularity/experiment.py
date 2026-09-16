import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

from scipy.stats import false_discovery_control, mannwhitneyu
from sklearn.metrics import silhouette_score

BASE_MANIFEST_COLUMNS = [
    "slide_id",
    "svs_path",
    "sacrifice_period",
    "compound_name",
    "dose",
    "dose_unit",
    "dose_level",
    "exp_id",
    "group_id",
    "individual_id",
    "sex_type",
    "strain_type",
]


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(exp_dir: Path, filename: str = "config.yml") -> dict:
    """Load a config file from the experiment directory.

    run_slurm.sh は RUN_COMMAND で常に --config を渡しているので、それを実際に
    使えるようにしておく。同じコードを別の入力（0005の染色正規化版の特徴量など）で
    回すときに、元の config.yml を書き換えずに別ファイルを指せる。
    """
    config_path = Path(filename)
    if not config_path.is_absolute():
        config_path = exp_dir / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    """Define CLI args for all variable dimensions (used in GRID_ARGS / RUN_COMMAND)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")
    return parser.parse_args()


def occupancy_from_assignments(
    assignments: pd.DataFrame, column: str, slide_ids: list[str]
) -> np.ndarray:
    """Slide × cluster occupancy fractions, rows ordered like slide_ids."""
    table = pd.crosstab(assignments["slide_id"], assignments[column])
    table = table.reindex(index=slide_ids, fill_value=0)
    counts = table.to_numpy(dtype=np.float64)
    return counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)


def topography_of(finding_sites, finding: str) -> str | None:
    """Pull the topography recorded for one finding name on a slide.

    finding_sites entries look like "Hypertrophy @ Centrilobular". A slide with
    the same finding at two sites is ambiguous for this analysis and is dropped
    by returning None.
    """
    hits = [s.split(" @ ", 1)[1] for s in list(finding_sites) if s.split(" @ ", 1)[0] == finding]
    uniq = sorted(set(hits))
    return uniq[0] if len(uniq) == 1 else None


def enrichment_test(
    occ: np.ndarray, positive: np.ndarray, logger: logging.Logger
) -> pd.DataFrame:
    """Per-cluster Mann-Whitney of occupancy, positive slides vs the rest.

    Slide-level again (see 0003): patches inside a slide are not independent.
    """
    rows = []
    for c in range(occ.shape[1]):
        pos, neg = occ[positive, c], occ[~positive, c]
        both = np.concatenate([pos, neg])
        if both.max() == both.min():
            p = 1.0
        else:
            _, p = mannwhitneyu(pos, neg, alternative="greater")
        rows.append(
            {
                "cluster_id": c,
                "positive_median": float(np.median(pos)),
                "negative_median": float(np.median(neg)),
                "positive_mean": float(pos.mean()),
                "negative_mean": float(neg.mean()),
                "p_value": float(p),
            }
        )
    out = pd.DataFrame(rows)
    out["q_value"] = false_discovery_control(np.clip(out["p_value"].to_numpy(), 0, 1), method="bh")
    return out


def label_separation(
    profiles: np.ndarray, labels: np.ndarray, rng: np.random.Generator, n_permutations: int
) -> dict:
    """How well a labelling separates slides in occupancy space.

    Silhouette on cosine distance between slide occupancy profiles. A finer
    labelling can score higher for purely combinatorial reasons, so the score
    is reported against a null built by shuffling the same labels.
    """
    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) < 2 or (counts < 2).any():
        return {"n_groups": int(len(uniq)), "silhouette": None, "note": "群が2つ未満、または1枚しかない群がある"}

    observed = float(silhouette_score(profiles, labels, metric="cosine"))
    null = np.array(
        [
            silhouette_score(profiles, rng.permutation(labels), metric="cosine")
            for _ in range(n_permutations)
        ]
    )
    return {
        "n_groups": int(len(uniq)),
        "group_sizes": {str(u): int(c) for u, c in zip(uniq, counts)},
        "silhouette": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        # 並べ替え帰無分布のうち観測値以上になった割合
        "p_permutation": float((null >= observed).mean()),
    }


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.tggates_metadata import attach_pathology_findings

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()

    config = load_config(Path(__file__).parent, args.config)
    seed: int = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    # 0003と同じ理由でconfigから取る（正規化版の結果を別ランとして残すため）
    variant_key = config.get("variant_key", "label_granularity")

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    source_dir = project_root / config["source_run_dir"]
    assignments = pd.read_parquet(source_dir / "cluster_assignments.parquet")
    assignments["slide_id"] = assignments["slide_id"].astype(str)

    # 0003のmanifestには finding_types しか無いので、部位付きで作り直す。
    # 0001側のmanifestは590枚で再生成されているため、ここでは使わない。
    base = pd.read_parquet(source_dir / "slide_manifest.parquet")
    base["slide_id"] = base["slide_id"].astype(str)
    manifest = attach_pathology_findings(
        base[BASE_MANIFEST_COLUMNS].copy(),
        dataset_dir / config["pathology_csv"],
        organ=config.get("organ", "Liver"),
    ).sort_values("slide_id").reset_index(drop=True)
    slide_ids = manifest["slide_id"].tolist()
    logger.info(f"manifest: {len(manifest)} slides, {int(manifest['has_finding'].sum())} with findings")

    k = config["k"]
    column = f"cluster_k{k}"
    occ = occupancy_from_assignments(assignments, column, slide_ids)
    logger.info(f"occupancy: {occ.shape} (k={k})")

    is_treated = (manifest["dose_level"] != "Control").to_numpy()

    # ── 所見名ごとに、部位が何種類あるか ──────────────────────────────────────
    site_rows = []
    for row in manifest.itertuples(index=False):
        for site in row.finding_sites:
            finding, topography = site.split(" @ ", 1)
            site_rows.append(
                {
                    "slide_id": row.slide_id,
                    "compound_name": row.compound_name,
                    "finding_type": finding,
                    "topography": topography,
                }
            )
    sites = pd.DataFrame(site_rows)
    per_finding = sites.groupby("finding_type").agg(
        n_slides=("slide_id", "nunique"),
        n_topography=("topography", "nunique"),
        n_compounds=("compound_name", "nunique"),
    )
    logger.info("所見名ごとの部位数:\n" + per_finding.sort_values("n_slides", ascending=False).head(12).to_string())

    # ── H1/H2: 部位を揃えたら化合物をまたいで形態が一致するか ────────────────
    target_finding = config["target_finding"]
    target_site = config["target_topography"]

    topo = manifest["finding_sites"].apply(lambda fs: topography_of(fs, target_finding))
    has_target_finding = topo.notna().to_numpy()
    is_target_site = (topo == target_site).to_numpy()

    logger.info(
        f"{target_finding}: {int(has_target_finding.sum())}枚 / "
        f"部位内訳 {topo.dropna().value_counts().to_dict()}"
    )
    site_by_compound = (
        manifest.assign(topography=topo)
        .dropna(subset=["topography"])
        .groupby(["compound_name", "topography"])
        .size()
        .rename("n_slides")
        .reset_index()
    )
    logger.info("化合物 × 部位:\n" + site_by_compound.to_string(index=False))

    # H1: 対象部位の陽性スライドで富化しているクラスタ（対照は他の投与スライド）
    positive = is_target_site & is_treated
    comparison = is_treated & ~is_target_site
    subset = positive | comparison
    stats = enrichment_test(occ[subset], positive[subset], logger)
    stats = stats.sort_values("q_value")
    top = stats[stats["q_value"] < config["q_threshold"]]
    logger.info(f"H1: {target_finding} @ {target_site} で富化したクラスタ {len(top)}個 (q<{config['q_threshold']})")

    # 富化クラスタを、化合物ごと・部位ごとの占有率に展開して H1/H2 を読む
    detail_rows = []
    for cluster_id in top["cluster_id"].head(config["n_report_clusters"]):
        for compound in sorted(manifest["compound_name"].unique()):
            for dose_group, mask in [
                ("treated", is_treated),
                ("control", ~is_treated),
            ]:
                sel = (manifest["compound_name"] == compound).to_numpy() & mask
                if not sel.any():
                    continue
                detail_rows.append(
                    {
                        "cluster_id": int(cluster_id),
                        "compound_name": compound,
                        "group": dose_group,
                        "topography": (
                            topo[sel].dropna().mode().iloc[0]
                            if topo[sel].notna().any()
                            else None
                        ),
                        "n_slides": int(sel.sum()),
                        "mean_occupancy": float(occ[sel, cluster_id].mean()),
                        "n_slides_present": int((occ[sel, cluster_id] > config["presence_eps"]).sum()),
                    }
                )
    detail = pd.DataFrame(detail_rows)
    for cluster_id, grp in detail.groupby("cluster_id"):
        treated_only = grp[grp["group"] == "treated"].sort_values("mean_occupancy", ascending=False)
        logger.info(
            f"\ncluster {cluster_id} の化合物別占有率(投与群):\n"
            + treated_only[["compound_name", "topography", "n_slides", "n_slides_present", "mean_occupancy"]]
            .to_string(index=False)
        )

    # ── 所見名 vs 所見名@部位 のどちらが形態を説明するか ─────────────────────
    n_perm = config["n_permutations"]
    comparisons = {}
    target_slides = np.flatnonzero(has_target_finding & is_treated)
    if len(target_slides) >= 4:
        profiles = occ[target_slides]
        topo_labels = topo.to_numpy()[target_slides]
        compound_labels = manifest["compound_name"].to_numpy()[target_slides]
        comparisons["topography"] = label_separation(profiles, topo_labels, rng, n_perm)
        comparisons["compound"] = label_separation(profiles, compound_labels, rng, n_perm)
        logger.info(
            f"\n{target_finding} 陽性スライド {len(target_slides)}枚 の分離度 (silhouette, cosine):"
        )
        for name, res in comparisons.items():
            logger.info(f"  {name}: {res}")

    # ── Save results ──────────────────────────────────────────────────────────
    manifest.to_parquet(run_dir / "slide_manifest_with_sites.parquet", index=False)
    stats.to_parquet(run_dir / "site_cluster_stats.parquet", index=False)
    detail.to_parquet(run_dir / "cluster_compound_detail.parquet", index=False)
    per_finding.reset_index().to_parquet(run_dir / "finding_topography_counts.parquet", index=False)

    results = {
        "k": k,
        "n_slides": len(manifest),
        "target_finding": target_finding,
        "target_topography": target_site,
        "n_target_finding_slides": int(has_target_finding.sum()),
        "topography_counts": {str(kk): int(v) for kk, v in topo.dropna().value_counts().items()},
        "n_enriched_clusters": int(len(top)),
        "enriched_clusters": [int(c) for c in top["cluster_id"].head(config["n_report_clusters"])],
        "label_separation": comparisons,
        "caveat": (
            "部位と化合物はこのデータでほぼ交絡している。交絡を免れて検証できるのは、"
            "同一部位を複数化合物が共有しているケース（Hypertrophy @ Centrilobular の"
            "phenobarbital / promethazine / thioacetamide）だけ。"
        ),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
