"""Re-run exp0009's候補選定 on already-saved cluster assignments, sweeping the
two analysis-only thresholds (min_compounds / min_treated_presence) without
redoing clustering or label propagation.

Both thresholds only enter select_candidates / cluster_statistics, so the
saved cluster_assignments.parquet is enough — the ~12min/k propagate_labels
pass does not have to run again for a threshold change.

Also reports **how evenly each cluster's treated patches are spread across
compounds**. 0003のplan.md (2026-09-16) の記録通り `n_compounds` はスライド数
しか見ておらず、他化合物が数十patch寄与するだけで2化合物と数えてしまう。
ここではpatch量ベースのシェアを見て、本当に複数化合物にまたがるクラスタ
（＝化合物固有の署名ではなく共通の形態トークンでありうるもの）を拾う。

Usage: python scripts/reanalyze_cluster_thresholds.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ROOT = PROJECT_ROOT / "outputs/0009_20260919_tier1_corpus_clustering"
MANIFEST = ROOT / "manifest_tier1_corpus.parquet"

# k -> どのスイープpartにassignmentsが入っているか
PART_FOR_K = {k: f"kmeans_k_sweep_part{(k - 100) // 500}" for k in range(100, 2001, 100)}

K_VALUES = [100, 500, 1000, 2000]
MIN_TREATED_PRESENCE = [0.0, 0.01, 0.02, 0.05, 0.1]
MIN_COMPOUNDS = [1, 2]
Q_THRESHOLD = 0.05
PRESENCE_EPS = 0.001
BALANCED_TOP1_MAX = 0.5  # 最多化合物のpatchシェアがこれ未満なら「偏りが小さい」


class _QuietLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def load_assignments(k: int, manifest: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    path = ROOT / PART_FOR_K[k] / "cluster_assignments.parquet"
    col = f"cluster_kmeans_k{k}"
    tbl = pq.read_table(path, columns=["slide_id", col])
    assignments = tbl.column(col).to_numpy().astype(np.int32)
    pos = {s: i for i, s in enumerate(manifest["slide_id"])}

    # slide_idはcategoryで保存されている。55M行を文字列に展開すると遅いので、
    # カテゴリ辞書(3,441個)だけを引いてコード配列をindexに変換する。
    s = tbl.column("slide_id").to_pandas()
    if isinstance(s.dtype, pd.CategoricalDtype):
        cat_to_pos = np.array([pos[str(c)] for c in s.cat.categories], dtype=np.int32)
        slide_index = cat_to_pos[s.cat.codes.to_numpy()]
    else:
        slide_index = np.array([pos[str(v)] for v in s], dtype=np.int32)
    return slide_index, assignments


def compound_balance(
    slide_index: np.ndarray, assignments: np.ndarray, manifest: pd.DataFrame, n_clusters: int
) -> pd.DataFrame:
    """Per-cluster share of treated patches contributed by each compound."""
    is_control = (manifest["dose_level"] == "Control").to_numpy()
    compounds = sorted(manifest.loc[~is_control, "compound_name"].unique())
    code_of = {c: i for i, c in enumerate(compounds)}
    slide_compound = np.full(len(manifest), -1, dtype=np.int32)
    for i, (comp, ctrl) in enumerate(zip(manifest["compound_name"], is_control)):
        if not ctrl:
            slide_compound[i] = code_of[comp]

    patch_compound = slide_compound[slide_index]
    keep = patch_compound >= 0  # 投与群patchのみ
    flat = patch_compound[keep].astype(np.int64) * n_clusters + assignments[keep]
    counts = np.bincount(flat, minlength=len(compounds) * n_clusters)
    counts = counts.reshape(len(compounds), n_clusters).astype(np.float64)

    totals = counts.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = counts / np.where(totals > 0, totals, 1)
    top1 = shares.max(axis=0)
    # 逆シンプソン指数: シェアが均等なら化合物数、1化合物独占なら1に近づく
    effective_n = 1.0 / np.clip((shares**2).sum(axis=0), 1e-12, None)
    top_compound = [compounds[i] for i in shares.argmax(axis=0)]
    return pd.DataFrame(
        {
            "cluster_id": np.arange(n_clusters),
            "n_treated_patches": totals.astype(np.int64),
            "top1_compound": top_compound,
            "top1_share": top1,
            "effective_n_compounds": effective_n,
        }
    )


def main() -> None:
    from lib.cluster_analysis import (
        cluster_statistics,
        leave_one_compound_out_auroc,
        occupancy_matrix,
        select_candidates,
    )

    manifest = pd.read_parquet(MANIFEST)
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)
    quiet = _QuietLogger()

    grid_rows, balance_rows = [], []
    for k in K_VALUES:
        slide_index, assignments = load_assignments(k, manifest)
        n_clusters = int(assignments.max()) + 1
        occ = occupancy_matrix(slide_index, assignments, len(manifest), n_clusters)
        bal = compound_balance(slide_index, assignments, manifest, n_clusters)

        for mtp in MIN_TREATED_PRESENCE:
            stats = cluster_statistics(
                occ, manifest, presence_eps=PRESENCE_EPS, min_treated_presence=mtp, logger=quiet
            )
            merged = stats.merge(bal, on="cluster_id")
            for mc in MIN_COMPOUNDS:
                cand = select_candidates(stats, q_threshold=Q_THRESHOLD, min_compounds=mc)
                sel = merged[merged["cluster_id"].isin(cand)]
                row = {
                    "k": k,
                    "min_treated_presence": mtp,
                    "min_compounds": mc,
                    "n_candidates": len(cand),
                    "balanced(top1<0.5)": int((sel["top1_share"] < BALANCED_TOP1_MAX).sum()),
                    "median_top1_share": round(float(sel["top1_share"].median()), 3) if len(sel) else None,
                    "median_eff_compounds": round(float(sel["effective_n_compounds"].median()), 2) if len(sel) else None,
                    "median_n_experiments": float(sel["n_experiments"].median()) if len(sel) else None,
                }
                if mc == 1:  # LOCOはfoldごとに再計算が走るので代表条件だけ
                    pooled, _ = leave_one_compound_out_auroc(
                        occ, manifest, presence_eps=PRESENCE_EPS, min_treated_presence=mtp,
                        q_threshold=Q_THRESHOLD, min_compounds=mc, logger=quiet,
                    )
                    row["auroc_loco"] = round(pooled, 4) if pooled is not None else None
                grid_rows.append(row)

                if mc == 1 and len(sel):
                    top = sel.nsmallest(5, "top1_share").copy()
                    top.insert(0, "min_treated_presence", mtp)
                    top.insert(0, "k", k)
                    balance_rows.append(top)
        print(f"k={k} done", flush=True)

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(ROOT / "threshold_sweep.csv", index=False)
    print("\n=== 閾値グリッド ===")
    print(grid.to_string(index=False))

    if balance_rows:
        bal_df = pd.concat(balance_rows, ignore_index=True)
        bal_df.to_csv(ROOT / "balanced_clusters.csv", index=False)
        print("\n=== 最も化合物の偏りが小さい候補クラスタ（min_compounds撤廃時、上位5個/条件）===")
        cols = ["k", "min_treated_presence", "cluster_id", "top1_compound", "top1_share",
                "effective_n_compounds", "n_compounds", "n_treated_slides", "n_experiments", "q_value"]
        print(bal_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
