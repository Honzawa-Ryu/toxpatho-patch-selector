"""Compare two clusterings on how much of their structure is batch vs treatment.

0004で使った3つの読みを、任意の0003ランに対してまとめて出す。染色正規化の前後を
並べるために書いたが、入力を変えれば他の比較（別のk、590枚版など）にも使える。

測るもの:

1. **バッチの上限**: 病変が一切無いcontrolスライドだけを、由来する実験でラベル付け
   したときの分離度。ここが高いなら、それは病変ではなく染色・スキャン条件。
2. **処置の信号**: 投与群とcontrolの分離度。1が下がっても2まで下がるなら、
   正規化がバッチと一緒に病変の手がかりまで消している。
3. **H1**: 同じ所見名・同じ部位を共有する化合物が、共通のクラスタを持つか。
   各化合物を「自群のcontrol」と比べて富化クラスタを取る（同一実験内比較）。
   化合物をまたいだ投与群同士の比較は、control基準が無いためバッチを拾うので使わない。

Usage:
    python scripts/compare_batch_effect.py \
        --runs outputs/0003_.../cluster_mining outputs/0003_.../cluster_mining_macenko \
        --labels 正規化なし 正規化あり --k 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs/0004_20260913_finding_label_granularity/label_granularity/slide_manifest_with_sites.parquet"
)
# 同じ所見名・同じ部位（Hypertrophy @ Centrilobular）を共有する化合物。
# 部位と化合物の交絡を免れて「形態が化合物をまたぐか」を問えるのはここだけ。
SHARED_SITE_COMPOUNDS = ["phenobarbital", "promethazine", "thioacetamide"]


def occupancy(assignments: pd.DataFrame, column: str, slide_ids: list[str]) -> np.ndarray:
    table = pd.crosstab(assignments["slide_id"], assignments[column])
    table = table.reindex(index=slide_ids, fill_value=0)
    counts = table.to_numpy(dtype=np.float64)
    return counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)


def separation(
    profiles: np.ndarray, labels: np.ndarray, rng: np.random.Generator, n_perm: int
) -> dict | None:
    """Silhouette against a label-shuffled null, so group counts cannot inflate it."""
    uniq, counts = np.unique(labels, return_counts=True)
    if len(uniq) < 2 or (counts < 2).any():
        return None
    observed = float(silhouette_score(profiles, labels, metric="cosine"))
    null = np.array(
        [silhouette_score(profiles, rng.permutation(labels), metric="cosine") for _ in range(n_perm)]
    )
    return {
        "silhouette": round(observed, 3),
        "null_mean": round(float(null.mean()), 3),
        "z": round(float((observed - null.mean()) / null.std()), 1),
    }


def enriched_within_compound(
    occ: np.ndarray, manifest: pd.DataFrame, compound: str, top: int
) -> list[int]:
    """Clusters where this compound's treated slides exceed its own controls."""
    is_c = (manifest["compound_name"] == compound).to_numpy()
    high = is_c & (manifest["dose_level"] != "Control").to_numpy()
    ctrl = is_c & (manifest["dose_level"] == "Control").to_numpy()
    rows = []
    for c in range(occ.shape[1]):
        a, b = occ[high, c], occ[ctrl, c]
        if np.ptp(np.concatenate([a, b])) == 0:
            continue
        _, p = mannwhitneyu(a, b, alternative="greater")
        rows.append({"cluster_id": c, "diff": a.mean() - b.mean(), "p": p})
    ranked = pd.DataFrame(rows).sort_values("diff", ascending=False).head(top)
    return ranked["cluster_id"].tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--top-clusters", type=int, default=4)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if len(args.runs) != len(args.labels):
        raise SystemExit("--runs と --labels の数を揃えてください")

    manifest = pd.read_parquet(args.manifest)
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)
    slide_ids = manifest["slide_id"].tolist()
    is_control = (manifest["dose_level"] == "Control").to_numpy()

    for run, label in zip(args.runs, args.labels):
        rng = np.random.default_rng(args.seed)
        path = run if run.is_absolute() else PROJECT_ROOT / run
        assignments = pd.read_parquet(path / "cluster_assignments.parquet")
        assignments["slide_id"] = assignments["slide_id"].astype(str)
        occ = occupancy(assignments, f"cluster_k{args.k}", slide_ids)

        print(f"\n{'=' * 68}\n{label}  (k={args.k})  {path.name}\n{'=' * 68}")

        batch = separation(
            occ[is_control], manifest["compound_name"].to_numpy()[is_control], rng, args.n_permutations
        )
        print(f"1. バッチの上限（control {int(is_control.sum())}枚を実験で）: {batch}")

        treat = separation(
            occ, np.where(is_control, "control", "treated"), rng, args.n_permutations
        )
        print(f"2. 処置の信号（全219枚を投与/control で）    : {treat}")

        enriched = {
            c: enriched_within_compound(occ, manifest, c, args.top_clusters)
            for c in SHARED_SITE_COMPOUNDS
        }
        print(f"3. H1: Hypertrophy @ Centrilobular を共有する3化合物の富化クラスタ")
        for c, ids in enriched.items():
            print(f"     {c:15s}: {ids}")
        pairs = [
            (a, b)
            for i, a in enumerate(SHARED_SITE_COMPOUNDS)
            for b in SHARED_SITE_COMPOUNDS[i + 1 :]
        ]
        for a, b in pairs:
            shared = sorted(set(enriched[a]) & set(enriched[b]))
            print(f"     {a} ∩ {b}: {shared if shared else '[] （共通なし）'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
