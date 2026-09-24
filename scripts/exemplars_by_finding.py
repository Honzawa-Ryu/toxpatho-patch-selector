"""Contact sheets for every candidate cluster of exp0009, filed under the
finding each cluster corresponds to.

make_cluster_exemplars.py は0003向けでクラスタIDを手で指定する前提だった
（列名も `cluster_k{k}`）。候補が158個あるとそれでは回らないので、候補の
列挙・所見との対応づけ・出力先の振り分けまで一括で行う。

出力構成:
    exemplars_by_finding/
        <所見名>/cluster_00200_OR679_carbon-tetrachloride.png   ← 主所見の実体
        <別の所見>/cluster_00200_...png                          ← symlink
        index.csv

患者由来の形態が均質かを見たいので、重心近傍ではなく一様ランダムに抜く
（make_cluster_exemplars.pyと同じ理由）。各シートには上部に
クラスタID・所見・OR・化合物シェアのキャプションを載せる。

Usage: python scripts/exemplars_by_finding.py [--k 1000] [--n 36] [--workers 12]
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

from scripts.reanalyze_cluster_thresholds import MANIFEST, ROOT, load_assignments  # noqa: E402

FEATURES_DIR = ROOT / "features_tier1_corpus"
TILE = 256
CAPTION_H = 54


class _Q:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name).strip("_")


def _build_sheet(job: tuple) -> tuple[int, str | None]:
    """1クラスタ分のコンタクトシート。ワーカープロセスで実行される。"""
    cluster_id, rows, caption, out_path, grid_cols = job
    import h5py
    import openslide
    from PIL import Image, ImageDraw

    tiles = []
    for slide_id, grp in pd.DataFrame(rows).groupby("slide_id"):
        try:
            with h5py.File(FEATURES_DIR / f"{slide_id}.h5", "r") as h:
                attrs = h["coords"].attrs
                size = int(attrs.get("patch_size_level0", attrs.get("patch_size", 256)))
            slide = openslide.OpenSlide(grp["svs_path"].iloc[0])
        except Exception as e:  # スライド1枚の失敗で全体を止めない
            print(f"cluster {cluster_id}: {slide_id} skipped ({e})", flush=True)
            continue
        for r in grp.itertuples():
            try:
                tile = slide.read_region((int(r.x), int(r.y)), 0, (size, size)).convert("RGB")
                tiles.append(tile.resize((TILE, TILE)))
            except Exception:
                pass
        slide.close()

    if not tiles:
        return cluster_id, None

    n_rows = (len(tiles) + grid_cols - 1) // grid_cols
    sheet = Image.new("RGB", (grid_cols * TILE, n_rows * TILE + CAPTION_H), "white")
    draw = ImageDraw.Draw(sheet)
    for i, line in enumerate(caption.split("\n")[:3]):
        draw.text((8, 6 + i * 16), line, fill="black")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % grid_cols) * TILE, CAPTION_H + (i // grid_cols) * TILE))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return cluster_id, str(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--grid-cols", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-treated-presence", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="先頭N個だけ処理（動作確認用）")
    ap.add_argument("--clusters", type=int, nargs="+", default=None, help="指定クラスタのみ処理")
    ap.add_argument("--out-name", type=str, default=None, help="出力ディレクトリ名")
    args = ap.parse_args()

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
        occ, manifest, presence_eps=0.001,
        min_treated_presence=args.min_treated_presence, logger=_Q(),
    )
    cand = select_candidates(stats, q_threshold=0.05, min_compounds=1)
    assoc = finding_associations(occ, manifest, cand, presence_eps=0.001, max_finding_types=36)
    sig = assoc[assoc["q_value"] < 0.05].copy()
    shares = compound_shares(slide_index, assignments, manifest, n_clusters, list(cand), top_n=3)
    print(f"k={args.k}: 候補{len(cand)}クラスタ / 有意な(クラスタ,所見)組 {len(sig)}", flush=True)

    # patch座標はcluster_assignments.parquetから取る（x,yはlevel-0画素）
    import pyarrow.parquet as pq
    from scripts.reanalyze_cluster_thresholds import PART_FOR_K

    col = f"cluster_kmeans_k{args.k}"
    tbl = pq.read_table(ROOT / PART_FOR_K[args.k] / "cluster_assignments.parquet",
                        columns=["x", "y", col])
    xs = tbl.column("x").to_numpy()
    ys = tbl.column("y").to_numpy()
    labels = tbl.column(col).to_numpy()
    svs_of = dict(zip(manifest["slide_id"], manifest["svs_path"]))
    slide_of = manifest["slide_id"].to_numpy()

    out_root = ROOT / (args.out_name or f"exemplars_by_finding_k{args.k}")
    if args.clusters:
        cand = np.array([c for c in cand if int(c) in set(args.clusters)])
        print(f"指定クラスタに絞り込み: {len(cand)}個", flush=True)
    rng = np.random.default_rng(args.seed)
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    starts = np.searchsorted(sorted_labels, np.arange(n_clusters), side="left")
    ends = np.searchsorted(sorted_labels, np.arange(n_clusters), side="right")

    jobs, index_rows = [], []
    for cid in (cand[: args.limit] if args.limit else cand):
        member_pos = order[starts[cid]:ends[cid]]
        if len(member_pos) == 0:
            continue
        pick = member_pos[rng.choice(len(member_pos), size=min(args.n, len(member_pos)), replace=False)]
        sids = slide_of[slide_index[pick]]
        rows = [{"slide_id": s, "svs_path": svs_of[s], "x": int(x), "y": int(y)}
                for s, x, y in zip(sids, xs[pick], ys[pick])]

        g = sig[sig["cluster_id"] == cid].sort_values("q_value")
        primary = g.iloc[0]["finding_type"] if len(g) else "_no_finding"
        or_val = g.iloc[0]["odds_ratio"] if len(g) else float("nan")
        top_comp = shares.get(cid, "").split("(")[0] or "unknown"
        fname = f"cluster_{cid:05d}_OR{'inf' if not np.isfinite(or_val) else int(or_val)}_{_safe(top_comp)}.png"
        out_path = out_root / _safe(primary) / fname

        caption = (
            f"k={args.k} cluster {cid} | {primary} (OR={or_val:.1f}, q={g.iloc[0]['q_value']:.1e})"
            if len(g) else f"k={args.k} cluster {cid} | 所見との有意な対応なし"
        )
        caption += f"\n{shares.get(cid, '')}"
        if len(g) > 1:
            others = ", ".join(f"{r.finding_type}(OR{r.odds_ratio:.0f})" for r in g.iloc[1:4].itertuples())
            caption += f"\n他: {others}"

        jobs.append((cid, rows, caption, out_path, args.grid_cols))
        index_rows.append({
            "cluster_id": cid, "primary_finding": primary,
            "n_findings": len(g), "odds_ratio": or_val,
            "all_findings": "; ".join(g["finding_type"].tolist()),
            "compounds": shares.get(cid, ""), "path": str(out_path.relative_to(ROOT)),
        })

    done = 0
    made: dict[int, str] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_build_sheet, j) for j in jobs]
        for fut in as_completed(futs):
            cid, path = fut.result()
            if path:
                made[cid] = path
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} クラスタ完了", flush=True)

    # 副次的に対応する所見のディレクトリにはsymlinkを張る（実体は主所見側のみ）
    n_links = 0
    for row in index_rows:
        cid = row["cluster_id"]
        if cid not in made:
            continue
        real = Path(made[cid])
        for f in row["all_findings"].split("; ")[1:]:
            link = out_root / _safe(f) / real.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists() and not link.is_symlink():
                link.symlink_to(real)
                n_links += 1

    idx = pd.DataFrame(index_rows).sort_values(["primary_finding", "odds_ratio"], ascending=[True, False])
    idx.to_csv(out_root / "index.csv", index=False)
    print(f"\nシート {len(made)}枚 / symlink {n_links}本 -> {out_root}")
    print("\n所見ごとのシート数（主所見ベース）:")
    print(idx["primary_finding"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
