"""Two structural questions about exp0009's cluster↔finding correspondence:

  Q2. 1つのクラスタが2つ以上の所見に「同程度」に対応している例はあるか。
      TG-GATEsは1スライドに複数所見を併記するので、単に両方有意なだけでは
      「ラベルがいつも一緒に付いてくる」だけかもしれない。コーパス全体での
      2所見の共起度(Jaccard)を並べて、それ以上の対応かを見る。

  Q3. 1つの所見が「大きく異なる」複数クラスタに割れている例はあるか。
      同一ラベル下に別機序・別形態が同居しているなら、そのクラスタ同士は
      embedding空間上でも離れているはず。64次元PCA空間でのcentroid距離を
      測り、全クラスタ間距離の分布と比べて相対化する。
      脂肪変性(Degeneration, fatty)は個別に詳細を出す。

Usage: python scripts/cluster_finding_structure.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reanalyze_cluster_thresholds import MANIFEST, ROOT, load_assignments  # noqa: E402

K = 1000
MIN_TREATED_PRESENCE = 0.0
Q_THRESHOLD = 0.05
PRESENCE_EPS = 0.001
N_FINDINGS = 36
CENTROID_SAMPLE = 3_000_000  # centroid推定に使うpatch数


class _Q:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def cluster_centroids(assignments: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """64次元PCA空間でのクラスタ重心。全55Mを読むのは重いのでpatchを抽出して推定。"""
    emb = np.load(ROOT / "embedding_cache" / "embeddings.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(emb), size=min(CENTROID_SAMPLE, len(emb)), replace=False))
    sample = np.asarray(emb[idx], dtype=np.float64)
    lab = assignments[idx]

    dim = sample.shape[1]
    sums = np.zeros((n_clusters, dim))
    np.add.at(sums, lab, sample)
    counts = np.bincount(lab, minlength=n_clusters).astype(np.float64)
    with np.errstate(invalid="ignore"):
        cent = sums / np.clip(counts, 1, None)[:, None]
    cent[counts == 0] = np.nan
    return cent


def compound_share_strings(
    slide_index, assignments, manifest, n_clusters, ids, top_n=3
) -> dict[int, str]:
    from scripts.balanced_cluster_findings import compound_shares

    return compound_shares(slide_index, assignments, manifest, n_clusters, ids, top_n=top_n)


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

    slide_index, assignments = load_assignments(K, manifest)
    n_clusters = int(assignments.max()) + 1
    occ = occupancy_matrix(slide_index, assignments, len(manifest), n_clusters)
    stats = cluster_statistics(
        occ, manifest, presence_eps=PRESENCE_EPS,
        min_treated_presence=MIN_TREATED_PRESENCE, logger=_Q(),
    )
    cand = select_candidates(stats, q_threshold=Q_THRESHOLD, min_compounds=1)
    assoc = finding_associations(
        occ, manifest, cand, presence_eps=PRESENCE_EPS, max_finding_types=N_FINDINGS
    )
    sig = assoc[assoc["q_value"] < Q_THRESHOLD].copy()
    print(f"k={K} / 候補 {len(cand)}クラスタ / 有意な(クラスタ,所見)組 {len(sig)}\n")

    # ── Q2: 1クラスタ × 2所見が同程度か ────────────────────────────────────
    fmat = {
        f: manifest["finding_types"].apply(lambda fs, f=f: f in list(fs)).to_numpy()
        for f in sig["finding_type"].unique()
    }

    def jaccard(f1, f2):
        a, b = fmat[f1], fmat[f2]
        u = (a | b).sum()
        return float((a & b).sum() / u) if u else 0.0

    rows = []
    for cid, g in sig.groupby("cluster_id"):
        g = g.sort_values("odds_ratio", ascending=False)
        if len(g) < 2:
            continue
        t1, t2 = g.iloc[0], g.iloc[1]
        if not np.isfinite(t1["odds_ratio"]) or not np.isfinite(t2["odds_ratio"]):
            continue
        ratio = t1["odds_ratio"] / max(t2["odds_ratio"], 1e-9)
        rows.append({
            "cluster_id": cid,
            "所見1": t1["finding_type"], "OR1": round(t1["odds_ratio"], 1), "共起枚数1": int(t1["n_slides_both"]),
            "所見2": t2["finding_type"], "OR2": round(t2["odds_ratio"], 1), "共起枚数2": int(t2["n_slides_both"]),
            "OR比": round(ratio, 2),
            "2所見のJaccard": round(jaccard(t1["finding_type"], t2["finding_type"]), 3),
        })
    pair = pd.DataFrame(rows)
    comparable = pair[
        (pair["OR比"] < 2.0) & (pair["OR2"] >= 10) & (pair["共起枚数2"] >= 10)
    ].sort_values("OR比")
    print("=== Q2: 上位2所見が同程度(OR比<2・OR2>=10・共起10枚以上)のクラスタ ===")
    print(f"{len(comparable)} / {len(pair)} クラスタが該当\n")
    with pd.option_context("display.width", 220):
        print(comparable.head(20).to_string(index=False))

    # ── Q3: 1所見が離れた複数クラスタに割れているか ──────────────────────
    cent = cluster_centroids(assignments, n_clusters)
    valid = ~np.isnan(cent).any(axis=1)
    cl = np.where(valid)[0]
    d_all = np.linalg.norm(cent[cl][:, None, :] - cent[cl][None, :, :], axis=-1)
    iu = np.triu_indices(len(cl), k=1)
    ref_med = float(np.median(d_all[iu]))
    print(f"\n\n=== Q3: 所見ごとのクラスタ散らばり（全クラスタ間距離の中央値 {ref_med:.2f} を1.0として相対化）===")

    pos = {c: i for i, c in enumerate(cl)}
    frows = []
    for f, g in sig.groupby("finding_type"):
        ids = [c for c in g["cluster_id"].unique() if c in pos]
        if len(ids) < 2:
            continue
        sub = np.array([pos[c] for c in ids])
        dd = d_all[np.ix_(sub, sub)]
        iu2 = np.triu_indices(len(ids), k=1)
        frows.append({
            "finding_type": f,
            "クラスタ数": len(ids),
            "重心間距離_中央値": round(float(np.median(dd[iu2])) / ref_med, 2),
            "重心間距離_最大": round(float(dd[iu2].max()) / ref_med, 2),
        })
    fdf = pd.DataFrame(frows).sort_values("重心間距離_中央値", ascending=False)
    print(fdf.to_string(index=False))

    # ── 脂肪変性の詳細 ─────────────────────────────────────────────────
    target = "Degeneration, fatty"
    fat = sig[sig["finding_type"] == target].sort_values("odds_ratio", ascending=False)
    print(f"\n\n=== 「{target}」に紐づくクラスタの内訳 ===")
    if fat.empty:
        print("該当なし")
        return
    ids = fat["cluster_id"].tolist()
    shares = compound_share_strings(slide_index, assignments, manifest, n_clusters, ids)
    fat = fat.assign(**{"compounds(patchシェア)": fat["cluster_id"].map(shares)})
    cols = ["cluster_id", "odds_ratio", "q_value", "n_slides_both", "compounds(patchシェア)"]
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(fat[cols].to_string(index=False))

    sub = np.array([pos[c] for c in ids if c in pos])
    if len(sub) >= 2:
        dd = d_all[np.ix_(sub, sub)] / ref_med
        print(f"\n重心間距離（相対値、全クラスタ中央値=1.0）:")
        print(pd.DataFrame(dd.round(2), index=[c for c in ids if c in pos],
                           columns=[c for c in ids if c in pos]).to_string())

    # 各クラスタが実際にどのスライドを取っているか（重なりを見る）
    print("\n各クラスタが保有する脂肪変性スライドの重なり(Jaccard):")
    has_fat = fmat[target]
    sets = {c: set(np.where((occ[:, c] > PRESENCE_EPS) & has_fat)[0]) for c in ids}
    mat = pd.DataFrame(
        [[round(len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1), 2) for b in ids] for a in ids],
        index=ids, columns=ids,
    )
    print(mat.to_string())

    fat.to_csv(ROOT / "fatty_degeneration_clusters.csv", index=False)
    fdf.to_csv(ROOT / "finding_cluster_spread.csv", index=False)
    print(f"\nwrote {ROOT / 'fatty_degeneration_clusters.csv'}, {ROOT / 'finding_cluster_spread.csv'}")


if __name__ == "__main__":
    main()
