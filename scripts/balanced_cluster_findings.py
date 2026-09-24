"""Which findings do exp0009's candidate clusters correspond to, and does the
answer differ between clusters shared across many compounds and clusters that
are one compound's signature?

Runs on the saved cluster_assignments.parquet with the thresholds that
threshold_sweep.csv picked out (min_treated_presence=0.0, min_compounds off),
so no clustering or propagation re-runs.

「複数化合物にまたがるクラスタが共通の所見に対応しているか」が主眼なので、
Fisher検定の結果を化合物シェア(patch量ベース)と並べて出す。

Usage: python scripts/balanced_cluster_findings.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reanalyze_cluster_thresholds import (  # noqa: E402
    MANIFEST,
    ROOT,
    compound_balance,
    load_assignments,
)

K_VALUES = [1000, 2000]
MIN_TREATED_PRESENCE = 0.0
MIN_COMPOUNDS = 1
Q_THRESHOLD = 0.05
PRESENCE_EPS = 0.001
MAX_FINDING_TYPES = 15
BALANCED_TOP1_MAX = 0.5
SPECIFIC_TOP1_MIN = 0.8


class _QuietLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def compound_shares(
    slide_index: np.ndarray, assignments: np.ndarray, manifest: pd.DataFrame,
    n_clusters: int, cluster_ids: list[int], top_n: int = 4
) -> dict[int, str]:
    """"compound(share) compound(share) ..." per cluster, patch量ベース."""
    is_control = (manifest["dose_level"] == "Control").to_numpy()
    compounds = sorted(manifest.loc[~is_control, "compound_name"].unique())
    code_of = {c: i for i, c in enumerate(compounds)}
    slide_compound = np.full(len(manifest), -1, dtype=np.int32)
    for i, (comp, ctrl) in enumerate(zip(manifest["compound_name"], is_control)):
        if not ctrl:
            slide_compound[i] = code_of[comp]

    patch_compound = slide_compound[slide_index]
    keep = patch_compound >= 0
    flat = patch_compound[keep].astype(np.int64) * n_clusters + assignments[keep]
    counts = np.bincount(flat, minlength=len(compounds) * n_clusters)
    counts = counts.reshape(len(compounds), n_clusters).astype(np.float64)

    out = {}
    for c in cluster_ids:
        col = counts[:, c]
        total = col.sum()
        if total == 0:
            out[c] = ""
            continue
        order = np.argsort(col)[::-1][:top_n]
        out[c] = " ".join(f"{compounds[i]}({col[i] / total:.0%})" for i in order if col[i] > 0)
    return out


def main() -> None:
    from lib.cluster_analysis import (
        cluster_statistics,
        finding_associations,
        occupancy_matrix,
        select_candidates,
    )

    manifest = pd.read_parquet(MANIFEST)
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)
    quiet = _QuietLogger()

    for k in K_VALUES:
        slide_index, assignments = load_assignments(k, manifest)
        n_clusters = int(assignments.max()) + 1
        occ = occupancy_matrix(slide_index, assignments, len(manifest), n_clusters)
        bal = compound_balance(slide_index, assignments, manifest, n_clusters)

        stats = cluster_statistics(
            occ, manifest, presence_eps=PRESENCE_EPS,
            min_treated_presence=MIN_TREATED_PRESENCE, logger=quiet,
        )
        cand = select_candidates(stats, q_threshold=Q_THRESHOLD, min_compounds=MIN_COMPOUNDS)
        assoc = finding_associations(
            occ, manifest, cand, presence_eps=PRESENCE_EPS, max_finding_types=MAX_FINDING_TYPES
        )
        assoc = assoc[assoc["q_value"] < Q_THRESHOLD].merge(bal, on="cluster_id")

        print(f"\n{'=' * 100}\nk={k}: 候補 {len(cand)} クラスタ / 所見と有意に紐づいた組 {len(assoc)}")
        if assoc.empty:
            continue

        assoc["group"] = np.where(
            assoc["top1_share"] < BALANCED_TOP1_MAX, "balanced",
            np.where(assoc["top1_share"] >= SPECIFIC_TOP1_MIN, "compound-specific", "mid"),
        )

        print(f"\n-- 所見ごと: 何クラスタが紐づいたか（balanced = 最多化合物<50%）--")
        pivot = (
            assoc.groupby(["finding_type", "group"])["cluster_id"].nunique()
            .unstack(fill_value=0)
        )
        pivot["total"] = pivot.sum(axis=1)
        print(pivot.sort_values("total", ascending=False).to_string())

        bal_assoc = assoc[assoc["group"] == "balanced"]
        if bal_assoc.empty:
            print("\n-- balancedクラスタに有意な所見の紐づきなし --")
            continue

        ids = sorted(bal_assoc["cluster_id"].unique().tolist())
        shares = compound_shares(slide_index, assignments, manifest, n_clusters, ids)
        best = bal_assoc.sort_values("q_value").groupby("cluster_id").head(2).copy()
        best["compounds(patchシェア)"] = best["cluster_id"].map(shares)
        best = best.sort_values(["top1_share", "q_value"])

        print(f"\n-- balancedクラスタの所見（クラスタあたり上位2件）--")
        cols = ["cluster_id", "finding_type", "odds_ratio", "q_value", "n_slides_both",
                "effective_n_compounds", "compounds(patchシェア)"]
        with pd.option_context("display.max_colwidth", 70):
            print(best[cols].to_string(index=False))

        assoc.to_csv(ROOT / f"findings_k{k}.csv", index=False)
        print(f"\nwrote {ROOT / f'findings_k{k}.csv'}")


if __name__ == "__main__":
    main()
