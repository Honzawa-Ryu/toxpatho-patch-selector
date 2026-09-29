"""Do "how many slides / how many compounds contributed patches" actually
predict whether a candidate cluster is a real lesion token?

ユーザー観察: cluster 88 は組織辺縁の白背景が大半で異物の写り込みもあるのに、
15化合物・多スライドにまたがる。辺縁や背景は全スライドに存在するので、
多様性指標は「ユビキタスなアーチファクト」でこそ高くなりうる。仮説が
成り立つかを実測する。

指標:
  n_slides / n_compounds : クラスタにpatchを供給したスライド数・化合物数
      （occupancy閾値ではなくpatch帰属そのものから数える）
  effective_n_compounds  : 逆シンプソン。patch量シェアの偏りを見る
  tissue_frac            : patchに占める組織の割合（1 - 白背景率）。
      辺縁・背景クラスタを直接捕まえるための指標

Usage: python scripts/cluster_quality_metrics.py [--n-probe 24]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reanalyze_cluster_thresholds import MANIFEST, PART_FOR_K, ROOT, load_assignments  # noqa: E402

PRESENCE_EPS = 0.001
Q_THRESHOLD = 0.05
N_FINDINGS = 36
WHITE_THRESHOLD = 220  # RGB平均がこれ以上なら背景扱い


class _Q:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _tissue_fraction(job: tuple) -> tuple[int, float, float, float]:
    """クラスタの代表patchにおける組織割合。(cluster_id, 単純版平均, 最小, 連結版平均)

    単純版: 白い画素を全て背景とみなす。脂肪滴・空胞・類洞の内腔まで背景に
        数えてしまうので、白く抜ける病変を持つクラスタを不当に低く評価する。
    連結版: **画像の縁から連結している白領域だけ**を背景（スライドガラス）と
        みなす。組織に囲まれた白抜け（脂肪滴・空胞）は組織側に数える。
    """
    cluster_id, rows = job
    import h5py
    import openslide
    from scipy import ndimage

    fracs = []
    fracs_cc = []
    for slide_id, grp in pd.DataFrame(rows).groupby("slide_id"):
        try:
            with h5py.File(ROOT / "features_tier1_corpus" / f"{slide_id}.h5", "r") as h:
                a = h["coords"].attrs
                size = int(a.get("patch_size_level0", a.get("patch_size", 256)))
            slide = openslide.OpenSlide(grp["svs_path"].iloc[0])
        except Exception:
            continue
        for r in grp.itertuples():
            try:
                tile = slide.read_region((int(r.x), int(r.y)), 0, (size, size)).convert("RGB")
                arr = np.asarray(tile.resize((128, 128)), dtype=np.float32).mean(axis=2)
                white = arr >= WHITE_THRESHOLD
                fracs.append(float((~white).mean()))

                # 縁に接する白領域のみ背景とみなす
                lab, n = ndimage.label(white)
                if n == 0:
                    fracs_cc.append(1.0)
                else:
                    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
                    border.discard(0)
                    bg = np.isin(lab, list(border)) if border else np.zeros_like(white)
                    fracs_cc.append(float((~bg).mean()))
            except Exception:
                pass
        slide.close()
    if not fracs:
        return cluster_id, float("nan"), float("nan"), float("nan")
    return cluster_id, float(np.mean(fracs)), float(np.min(fracs)), float(np.mean(fracs_cc))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--n-probe", type=int, default=24)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    K = args.k

    import pyarrow.parquet as pq
    from scipy.stats import spearmanr

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
        occ, manifest, presence_eps=PRESENCE_EPS, min_treated_presence=0.0, logger=_Q()
    )
    cand = select_candidates(stats, q_threshold=Q_THRESHOLD, min_compounds=1)
    assoc = finding_associations(
        occ, manifest, cand, presence_eps=PRESENCE_EPS, max_finding_types=N_FINDINGS
    )
    sig = assoc[assoc["q_value"] < Q_THRESHOLD]
    print(f"k={K}: 候補{len(cand)}クラスタ", flush=True)

    # ── patch帰属ベースの多様性（occupancy閾値を使わない）──────────────────
    compound_of = manifest["compound_name"].to_numpy()
    patch_slide = slide_index
    patch_compound_code = pd.Categorical(compound_of[patch_slide]).codes
    n_comp_total = len(pd.Categorical(compound_of).categories)

    rows = []
    for c in cand:
        m = assignments == c
        slides = np.unique(patch_slide[m])
        comps = np.unique(patch_compound_code[m])
        counts = np.bincount(patch_compound_code[m], minlength=n_comp_total).astype(float)
        share = counts / counts.sum()
        rows.append({
            "cluster_id": int(c),
            "n_patches": int(m.sum()),
            "n_slides": len(slides),
            "n_compounds": len(comps),
            "effective_n_compounds": float(1.0 / np.clip((share ** 2).sum(), 1e-12, None)),
        })
    div = pd.DataFrame(rows)

    # ── 組織割合（辺縁・背景クラスタの検出）────────────────────────────────
    col = f"cluster_kmeans_k{K}"
    tbl = pq.read_table(ROOT / PART_FOR_K[K] / "cluster_assignments.parquet", columns=["x", "y", col])
    xs, ys = tbl.column("x").to_numpy(), tbl.column("y").to_numpy()
    labels = tbl.column(col).to_numpy()
    svs_of = dict(zip(manifest["slide_id"], manifest["svs_path"]))
    slide_of = manifest["slide_id"].to_numpy()

    rng = np.random.default_rng(42)
    order = np.argsort(labels, kind="stable")
    sl = labels[order]
    starts = np.searchsorted(sl, np.arange(n_clusters), "left")
    ends = np.searchsorted(sl, np.arange(n_clusters), "right")

    jobs = []
    for c in cand:
        pos = order[starts[c]:ends[c]]
        pick = pos[rng.choice(len(pos), size=min(args.n_probe, len(pos)), replace=False)]
        sids = slide_of[slide_index[pick]]
        jobs.append((int(c), [{"slide_id": s, "svs_path": svs_of[s], "x": int(x), "y": int(y)}
                              for s, x, y in zip(sids, xs[pick], ys[pick])]))

    tis = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for cid, mean_f, min_f, mean_cc in ex.map(_tissue_fraction, jobs):
            tis.append({"cluster_id": cid, "tissue_frac_mean": mean_f,
                        "tissue_frac_min": min_f, "tissue_frac_cc": mean_cc})
    tissue = pd.DataFrame(tis)

    # ── まとめ ────────────────────────────────────────────────────────────
    grade_path = ROOT / f"grade_response_k{K}.csv"
    df = div.merge(tissue, on="cluster_id").merge(
        stats[["cluster_id", "n_experiments"]], on="cluster_id"
    )
    df["max_OR"] = df["cluster_id"].map(
        sig.groupby("cluster_id")["odds_ratio"].max().replace([np.inf], np.nan)
    )
    df["n_findings"] = df["cluster_id"].map(sig.groupby("cluster_id")["finding_type"].nunique())
    if grade_path.exists():
        gr = pd.read_csv(grade_path)
        if "k" not in gr or not gr["k"].eq(K).all():
            raise ValueError("Grade metrics must match the requested k")
        ok = gr[(gr["rho"] > 0) & (gr["rho_q"] < 0.05)]
        df["grade_passed"] = df["cluster_id"].isin(ok["cluster_id"]).astype(int)
        df["best_rho"] = df["cluster_id"].map(gr.groupby("cluster_id")["rho"].max())
    df.insert(0, "k", K)
    df.to_csv(ROOT / f"cluster_quality_metrics_k{K}.csv", index=False)

    print("\n=== 候補158クラスタの指標分布 ===")
    print(df[["n_patches", "n_slides", "n_compounds", "effective_n_compounds",
              "n_experiments", "tissue_frac_mean"]].describe().round(3).to_string())

    print("\n=== 指標同士の順位相関（spearman）===")
    cols = ["n_slides", "n_compounds", "effective_n_compounds", "n_experiments",
            "tissue_frac_mean", "max_OR", "n_findings"]
    if "grade_passed" in df:
        cols += ["grade_passed", "best_rho"]
    sub = df[cols].dropna()
    corr = pd.DataFrame(
        [[spearmanr(sub[a], sub[b]).statistic for b in cols] for a in cols],
        index=cols, columns=cols,
    )
    print(corr.round(2).to_string())

    print("\n=== n_slides 五分位ごとの平均 ===")
    df["n_slides_bin"] = pd.qcut(df["n_slides"], 5, labels=["最小", "小", "中", "大", "最大"])
    agg_cols = ["n_slides", "n_compounds", "tissue_frac_mean", "max_OR", "n_findings"]
    if "grade_passed" in df:
        agg_cols.append("grade_passed")
    print(df.groupby("n_slides_bin", observed=True)[agg_cols].mean().round(3).to_string())

    print("\n=== 組織割合が低い（＝辺縁・背景が多い）クラスタ 上位15 ===")
    show = ["cluster_id", "tissue_frac_mean", "n_slides", "n_compounds",
            "effective_n_compounds", "n_experiments", "n_findings"]
    if "grade_passed" in df:
        show.append("grade_passed")
    print(df.nsmallest(15, "tissue_frac_mean")[show].round(3).to_string(index=False))

    print(f"\nwrote {ROOT / f'cluster_quality_metrics_k{K}.csv'}")


if __name__ == "__main__":
    main()
