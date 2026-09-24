"""Contact sheets for exp0009's candidate clusters, filtered by tissue fraction
and **ranked by how well each cluster corresponds to a finding**.

対応の強さは保有内率 = P(所見 | そのクラスタを保有するスライド) で測る。
ORは「背景と比べた濃縮度」なので、保有スライドの大半が別物でも非保有側に
その所見が皆無なら無限大になりうる（cluster 110のEdemaがこれ）。
「このクラスタがあるスライドは、その所見を持っているか」を直接見る保有内率の
ほうが、対応の強さの指標として素直。

主所見もq値最小ではなく保有内率最大で決める。ファイル名に順位・保有内率・
保有枚数を入れてあるので、ディレクトリ内でソートすれば対応の強い順に並ぶ。

組織割合は cluster_quality_metrics.csv の tissue_frac_cc（画像の縁から連結する
白領域のみを背景とみなす版）を使う。単純な白画素率だと脂肪滴・空胞を背景に
数えてしまい、脂肪変性クラスタを不当に落とす。

Usage:
    python scripts/exemplars_ranked.py --min-tissue-frac 0.85 --n 36
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exemplars_by_finding import _build_sheet, _safe  # noqa: E402
from scripts.reanalyze_cluster_thresholds import MANIFEST, PART_FOR_K, ROOT, load_assignments  # noqa: E402


class _Q:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--grid-cols", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-tissue-frac", type=float, default=0.85)
    ap.add_argument("--q-threshold", type=float, default=0.05)
    ap.add_argument("--min-treated-presence", type=float, default=0.0)
    ap.add_argument("--presence-eps", type=float, default=0.001)
    ap.add_argument("--out-name", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dup-threshold", type=float, default=0.9,
                    help="この保有内率を複数所見で超えるクラスタにDUPフラグを立てる")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    from lib.cluster_analysis import (
        cluster_statistics,
        finding_associations,
        occupancy_matrix,
        select_candidates,
    )
    from scripts.balanced_cluster_findings import compound_shares

    manifest = pd.read_parquet(MANIFEST)
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)

    slide_index, assignments = load_assignments(args.k, manifest)
    n_clusters = int(assignments.max()) + 1
    occ = occupancy_matrix(slide_index, assignments, len(manifest), n_clusters)
    stats = cluster_statistics(
        occ, manifest, presence_eps=args.presence_eps,
        min_treated_presence=args.min_treated_presence, logger=_Q(),
    )
    cand = select_candidates(stats, q_threshold=args.q_threshold, min_compounds=1)
    print(f"k={args.k}: 候補 {len(cand)}クラスタ", flush=True)

    # ── 組織割合フィルタ ──────────────────────────────────────────────────
    qm_path = ROOT / f"cluster_quality_metrics_k{args.k}.csv"
    if not qm_path.exists():
        qm_path = ROOT / "cluster_quality_metrics.csv"
    dropped = []
    if args.min_tissue_frac > 0 and qm_path.exists():
        qm = pd.read_csv(qm_path).set_index("cluster_id")
        tf = qm["tissue_frac_cc"] if "tissue_frac_cc" in qm else qm["tissue_frac_mean"]
        dropped = [int(c) for c in cand if tf.get(c, 1.0) < args.min_tissue_frac]
        cand = np.array([c for c in cand if tf.get(c, 1.0) >= args.min_tissue_frac])
        print(f"組織割合 >= {args.min_tissue_frac}: {len(cand)}クラスタ "
              f"(除外 {len(dropped)}個: {sorted(dropped)})", flush=True)

    # ── 所見との対応（保有内率）────────────────────────────────────────────
    counts = manifest.explode("finding_types")["finding_types"].dropna().value_counts()
    n_test = int((counts >= 5).sum())
    assoc = finding_associations(
        occ, manifest, cand, presence_eps=args.presence_eps, max_finding_types=n_test
    )
    sig = assoc[assoc["q_value"] < args.q_threshold].copy()

    fmat = {f: manifest["finding_types"].apply(lambda fs, f=f: f in list(fs)).to_numpy()
            for f in sig["finding_type"].unique()}
    base = {f: float(fmat[f].mean()) for f in fmat}

    n_carrying = {int(c): int((occ[:, c] > args.presence_eps).sum()) for c in cand}
    prec, lift = [], []
    for r in sig.itertuples():
        n = n_carrying[int(r.cluster_id)]
        p = r.n_slides_both / n if n else 0.0
        prec.append(p)
        lift.append(p / base[r.finding_type] if base[r.finding_type] > 0 else 0.0)
    sig["保有内率"] = prec
    sig["lift"] = lift
    sig["n_carrying"] = sig["cluster_id"].map(n_carrying)

    # 複数所見で高い保有内率を示すクラスタにフラグを立てる。
    # 1パッチに複数所見が同居するのは病理的に正当な場合もあるので
    # （CCl4の脂肪変性＋炎症細胞浸潤など）、除外はせず目視確認に回す。
    n_high = (sig[sig["保有内率"] >= args.dup_threshold]
              .groupby("cluster_id")["finding_type"].nunique())

    # 主所見＝保有内率が最大のもの（同率ならq値が小さい方）
    primary = (sig.sort_values(["保有内率", "q_value"], ascending=[False, True])
               .groupby("cluster_id").head(1).set_index("cluster_id"))
    ranked = primary.sort_values(["保有内率", "n_carrying"], ascending=[False, False])
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    print(f"保有内率の中央値 {ranked['保有内率'].median():.3f} / "
          f"0.8以上 {(ranked['保有内率'] >= 0.8).sum()}個 / "
          f"0.5未満 {(ranked['保有内率'] < 0.5).sum()}個", flush=True)

    shares = compound_shares(slide_index, assignments, manifest, n_clusters, list(cand), top_n=3)

    # ── patch座標 ─────────────────────────────────────────────────────────
    col = f"cluster_kmeans_k{args.k}"
    tbl = pq.read_table(ROOT / PART_FOR_K[args.k] / "cluster_assignments.parquet",
                        columns=["x", "y", col])
    xs, ys = tbl.column("x").to_numpy(), tbl.column("y").to_numpy()
    labels = tbl.column(col).to_numpy()
    svs_of = dict(zip(manifest["slide_id"], manifest["svs_path"]))
    slide_of = manifest["slide_id"].to_numpy()

    out_root = ROOT / (args.out_name or f"exemplars_ranked_k{args.k}")
    rng = np.random.default_rng(args.seed)
    order = np.argsort(labels, kind="stable")
    sl = labels[order]
    starts = np.searchsorted(sl, np.arange(n_clusters), "left")
    ends = np.searchsorted(sl, np.arange(n_clusters), "right")

    jobs, index_rows = [], []
    for cid, row in ranked.iterrows():
        cid = int(cid)
        pos = order[starts[cid]:ends[cid]]
        if len(pos) == 0:
            continue
        # 目的が「所見と対応づいたパッチ集」なので、主所見を持つスライド由来の
        # patchだけを見せる。クラスタ全体から一様に抜くと、保有内率が低い
        # クラスタほど「その所見を持たないスライド」のpatchが混ざる。
        on_finding = fmat[row["finding_type"]][slide_index[pos]]
        pool = pos[on_finding] if on_finding.any() else pos
        pick = pool[rng.choice(len(pool), size=min(args.n, len(pool)), replace=False)]
        sids = slide_of[slide_index[pick]]
        rows = [{"slide_id": s, "svs_path": svs_of[s], "x": int(x), "y": int(y)}
                for s, x, y in zip(sids, xs[pick], ys[pick])]

        g = sig[sig["cluster_id"] == cid].sort_values("保有内率", ascending=False)
        top_comp = shares.get(cid, "").split("(")[0] or "unknown"
        dup = int(n_high.get(cid, 0))
        dup_tag = f"_DUP{dup}" if dup >= 2 else ""
        fname = (f"r{int(row['rank']):03d}_p{row['保有内率']:.2f}_n{int(row['n_carrying'])}"
                 f"{dup_tag}_cluster_{cid:05d}_{_safe(top_comp)}.png")
        out_path = out_root / _safe(row["finding_type"]) / fname

        caption = (f"#{int(row['rank'])} cluster {cid} | {row['finding_type']} "
                   f"保有内率 {row['保有内率']:.0%} ({int(row['n_slides_both'])}/{int(row['n_carrying'])}枚) "
                   f"lift={row['lift']:.1f} OR={row['odds_ratio']:.0f}")
        if dup >= 2:
            caption += f" [DUP: {dup}所見が保有内率{args.dup_threshold:.0%}以上]"
        caption += f"\n{shares.get(cid, '')}"
        if len(g) > 1:
            others = ", ".join(f"{r.finding_type}({r.保有内率:.0%})" for r in g.iloc[1:4].itertuples())
            caption += f"\n他: {others}"

        jobs.append((cid, rows, caption, out_path, args.grid_cols))
        index_rows.append({
            "rank": int(row["rank"]), "cluster_id": cid,
            "primary_finding": row["finding_type"], "保有内率": round(row["保有内率"], 3),
            "n_slides_both": int(row["n_slides_both"]), "n_carrying": int(row["n_carrying"]),
            "lift": round(row["lift"], 2), "odds_ratio": row["odds_ratio"],
            "n_findings": len(g), "n_high_prec_findings": dup,
            "compounds": shares.get(cid, ""),
            "path": str(out_path.relative_to(ROOT)),
        })

    done, made = 0, {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_build_sheet, j) for j in jobs]
        for fut in as_completed(futs):
            cid, path = fut.result()
            if path:
                made[cid] = path
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} クラスタ完了", flush=True)

    # 副次的な対応所見にもsymlinkを張る（実体は主所見側のみ）
    n_links = 0
    for row in index_rows:
        cid = row["cluster_id"]
        if cid not in made:
            continue
        real = Path(made[cid])
        for f in sig[sig["cluster_id"] == cid]["finding_type"]:
            if f == row["primary_finding"]:
                continue
            link = out_root / _safe(f) / real.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists() and not link.is_symlink():
                link.symlink_to(real)
                n_links += 1

    idx = pd.DataFrame(index_rows).sort_values("rank")
    idx.to_csv(out_root / "index.csv", index=False)
    if dropped:
        pd.Series(sorted(dropped), name="dropped_cluster_id").to_csv(
            out_root / "dropped_by_tissue_frac.csv", index=False)

    print(f"\nシート {len(made)}枚 / symlink {n_links}本 -> {out_root}")
    print("\n=== 保有内率 上位15 ===")
    print(idx.head(15)[["rank", "cluster_id", "primary_finding", "保有内率",
                        "n_carrying", "lift", "n_findings"]].to_string(index=False))
    print("\n=== 保有内率 下位10 ===")
    print(idx.tail(10)[["rank", "cluster_id", "primary_finding", "保有内率",
                        "n_carrying", "lift", "n_findings"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
