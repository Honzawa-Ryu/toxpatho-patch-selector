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

import h5py
import openslide
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


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


def load_config(exp_dir: Path) -> dict:
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TILE, COLS = 256, 6
# 一貫性の段階: (下限, 色)。上から順に判定する
BAND_COLORS = [(0.9, "#2166ac"), (0.7, "#92c5de"), (0.5, "#fdae61"), (0.0, "#d73027")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", choices=["layer", "q_vote", "vote_steps"], default="layer")
    parser.add_argument("--layer", help="--analysis layer のときに必須")
    return parser.parse_args()


def vote_top_finding(cluster_dir: Path, labels: list[str], stab_n: int, with_q: bool = False):
    """各patchについて、最も多くのrunで候補になった所見とその割合（全run数に対する）を返す。

    あわせて、一貫性（最多所見の回数 / 何らかの所見の候補になった回数）と、
    2番目に多い所見とその割合（全run数に対する）も返す。

    with_q=True のときは、そのpatchが最多所見の候補クラスタに入ったrunでの
    log10(q値)・クラスタ大きさの中央値も返す（入らなかったrunは除く）。
    """
    cands = {r: pd.read_csv(cluster_dir / f"eval_{r}" / "candidates.csv").dropna(subset=["finding_type"]) for r in labels}
    findings = sorted(set().union(*[set(c.finding_type) for c in cands.values()]))
    fi = {f: i for i, f in enumerate(findings)}
    votes = np.zeros((stab_n, len(findings)), np.uint8)
    if with_q:
        f_runs = np.full((stab_n, len(labels)), -1, np.int16)
        q_runs = np.full((stab_n, len(labels)), np.nan, np.float32)
        size_runs = np.full((stab_n, len(labels)), np.nan, np.float32)
    for j, r in enumerate(labels):
        lab = np.load(cluster_dir / f"eval_{r}" / "stability_labels.npy")
        c = cands[r]
        table = np.full(int(lab.max()) + 1, -1, np.int16)
        table[c.cluster_id.to_numpy()] = [fi[f] for f in c.finding_type]
        f = table[lab]
        hit = np.flatnonzero(f >= 0)
        votes[hit, f[hit]] += 1
        if with_q:
            q_table = np.full(len(table), np.nan, np.float32)
            q_table[c.cluster_id.to_numpy()] = np.log10(c.q_value.to_numpy())  # qは1e-134まで出るのでfloat32に入れる前にlogにする
            size_table = np.full(len(table), np.nan, np.float32)
            size_table[c.cluster_id.to_numpy()] = c.n_patches.to_numpy()
            f_runs[:, j], q_runs[:, j], size_runs[:, j] = f, q_table[lab], size_table[lab]
    top_f, top_frac = votes.argmax(1), votes.max(1) / len(labels)
    n_any = votes.sum(1)
    with np.errstate(all="ignore"):
        consistency = np.where(n_any > 0, votes.max(1) / n_any, np.nan)
    votes[np.arange(stab_n), top_f] = 0
    second_f, second_frac = votes.argmax(1), votes.max(1) / len(labels)
    base = (top_f, top_frac, findings, consistency, second_f, second_frac)
    if not with_q:
        return base
    own = f_runs == top_f[:, None]
    q_runs[~own], size_runs[~own] = np.nan, np.nan
    with np.errstate(all="ignore"):
        median_log_q, median_size = np.nanmedian(q_runs, axis=1), np.nanmedian(size_runs, axis=1)
    return (*base, median_log_q, median_size)


def sample_by_slide(pool: np.ndarray, slide_of: np.ndarray, rng: np.random.Generator, n: int, cap: int) -> np.ndarray:
    """スライドを一様に選んでからpatchを取る（大きなスライドに偏らせない）。"""
    order = np.argsort(slide_of[pool], kind="stable")
    slides, starts, counts = np.unique(slide_of[pool][order], return_index=True, return_counts=True)
    options = {int(s): rng.choice(pool[order[a:a + c]], size=min(cap, c), replace=False)
               for s, a, c in zip(slides, starts, counts)}
    chosen: list[int] = []
    for rnd in range(cap):
        for s in rng.permutation(slides):
            if rnd < len(options[int(s)]):
                chosen.append(int(options[int(s)][rnd]))
                if len(chosen) == n:
                    return np.asarray(chosen)
    return np.asarray(chosen)


def tissue_fraction(tile: Image.Image) -> float:
    white = np.asarray(tile.resize((128, 128)), dtype=np.float32).mean(axis=2) >= 220
    comp, _ = ndimage.label(white)
    border = np.unique(np.r_[comp[0], comp[-1], comp[:, 0], comp[:, -1]])
    return float(1 - np.isin(comp, border[border != 0]).mean())


def draw_sheet(title: str, tiles: list[dict], path: Path) -> None:
    rows = (len(tiles) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * TILE, 34 + rows * (TILE + 20)), "white")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 13)
    d.text((6, 8), title, fill="black", font=font)
    for j, t in enumerate(tiles):
        if t is None:
            continue
        x, y = (j % COLS) * TILE, 34 + (j // COLS) * (TILE + 20)
        sheet.paste(t["image"], (x, y + 18))
        if "band" in t:  # 見出しの帯を色分けする（vote_stepsでは一貫性）
            d.rectangle((x, y, x + TILE - 1, y + 17), fill=t["band"])
        text_color = "white" if t.get("band") == BAND_COLORS[0][1] else "black"
        name_len = 12 if "tag" in t else 16
        d.text((x + 2, y + 2), f"{t.get('tag', '')}{t['slide_id']} {t['dose'][:1]} {t['compound'][:name_len]}", fill=text_color, font=font)
        d.rectangle((x, y + 18, x + TILE - 1, y + 18 + TILE - 1), outline="#1a9850" if t["carries"] else "#d73027", width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def extract_tiles(picks: np.ndarray, finding: str, ctx: dict, feature_dir: Path, extra: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    """選んだpatchをWSIから切り出し、シート用の画像とtiles.csv用の行を返す。extraは行に足す列。"""
    tiles, tile_rows, wsis = [], [], {}
    for p in picks:
        sid = ctx["cache_ids"][ctx["slide_of"][p]]
        m = ctx["manifest"].loc[sid]
        with h5py.File(feature_dir / f"{sid}.h5", "r") as h:
            size = int(h["coords"].attrs["patch_size_level0"])
        if sid not in wsis:
            wsis[sid] = openslide.OpenSlide(str(m["svs_path"]))
        x, y = int(ctx["xy"][p, 0]), int(ctx["xy"][p, 1])
        img = wsis[sid].read_region((x, y), 0, (size, size)).convert("RGB").resize((TILE, TILE))
        tiles.append({"image": img, "slide_id": sid, "dose": str(m["dose_level"]), "compound": str(m["compound_name"]),
                      "carries": bool(finding in m["finding_types"])})
        tile_rows.append({"finding": finding, "slide_id": sid, "x": x, "y": y, "patch_size_level0": size,
                          "compound": str(m["compound_name"]), "dose": str(m["dose_level"]), "slide_carries_finding": tiles[-1]["carries"],
                          **{k: v[p].item() for k, v in extra.items()}, "tissue_fraction": tissue_fraction(img)})
    for w in wsis.values():
        w.close()
    return tiles, tile_rows


def load_context(project_root: Path, config: dict, with_q: bool) -> dict:
    """投票結果と、1M標本patchのスライド・座標・メタデータをまとめて読む。"""
    cdir = project_root / config["cluster_exp_dir"]
    summary = pd.read_csv(cdir / "summary" / "summary.csv")
    labels = [r for r in summary.loc[summary.method != "reference", "label"]
              if (cdir / f"eval_{r}" / "stability_labels.npy").exists()]
    stab_idx = np.load(cdir / "prepare_reference" / "stability_idx.npy")
    ctx = dict(zip(["top_f", "top_frac", "findings", "consistency", "second_f", "second_frac",
                    "median_log_q", "median_size"],
                   vote_top_finding(cdir, labels, len(stab_idx), with_q)))
    ctx["n_runs"] = len(labels)
    cache = project_root / config["embedding_cache_dir"]
    ctx["cache_ids"] = np.load(cache / "slide_ids.npy", allow_pickle=False).astype(str)
    ctx["slide_of"] = np.load(cache / "slide_index.npy")[stab_idx]
    ctx["xy"] = np.load(cache / "xy.npy")[stab_idx]
    manifest = pd.read_parquet(project_root / config["manifest_parquet"])
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    ctx["manifest"] = manifest.set_index("slide_id").loc[ctx["cache_ids"]]
    ctx["known"] = set(pd.read_parquet(project_root / config["eval_manifest_parquet"])["slide_id"].astype(str))
    ctx["in_eval"] = np.isin(ctx["cache_ids"], list(ctx["known"]))[ctx["slide_of"]]
    return ctx


def pool_stats(pool: np.ndarray, finding: str, ctx: dict) -> dict:
    """patch集合の大きさ・化合物の偏り・所見ありスライドに入っている割合。"""
    slides = ctx["slide_of"][pool]
    carries = ctx["manifest"]["finding_types"].map(lambda fs: finding in fs).to_numpy()
    cs = pd.Series(ctx["manifest"]["compound_name"].to_numpy()[slides]).value_counts()
    return {"n_patches": len(pool), "n_slides": len(np.unique(slides)), "n_compounds": len(cs),
            "top_compound": cs.index[0], "top_compound_share": cs.iloc[0] / len(pool),
            "patch_in_carrying_slide": float(carries[slides].mean())}


def run_layer(args, config: dict, ctx: dict, run_dir: Path, feature_dir: Path, logger: logging.Logger) -> None:
    lo, hi = config["layers"][args.layer]
    top_f, top_frac, findings = ctx["top_f"], ctx["top_frac"], ctx["findings"]
    slide_of, manifest, cache_ids, in_eval = ctx["slide_of"], ctx["manifest"], ctx["cache_ids"], ctx["in_eval"]
    member = in_eval & (top_frac >= lo) & (top_frac < hi)
    logger.info(f"layer members: {int(member.sum())} patches ({member.mean():.4f} of sample; eval slides only)")

    # 所見ごとの閾値検討用: 閾値を下げたときに各所見で何が増えるか（層に依存しない全所見の表）
    table = []
    for fi, finding in enumerate(findings):
        for t in config["table_thresholds"]:
            pool = np.flatnonzero(in_eval & (top_f == fi) & (top_frac >= t))
            if len(pool):
                table.append({"finding": finding, "threshold": t, **pool_stats(pool, finding, ctx)})
    pd.DataFrame(table).to_csv(run_dir / "finding_threshold_table.csv", index=False)

    rng = np.random.default_rng(config["seed"])
    rows, tile_rows = [], []
    for fi, finding in enumerate(findings):
        pool = np.flatnonzero(member & (top_f == fi))
        if len(pool) < config["min_patches_per_finding"]:
            continue
        carries = manifest["finding_types"].map(lambda fs: finding in fs).to_numpy()
        treated = (manifest["dose_level"] != "Control").to_numpy()
        rows.append({"finding": finding, **pool_stats(pool, finding, ctx),
                     "patch_in_treated_slide": float(treated[slide_of[pool]].mean()),
                     "slide_prevalence_known": float(carries[np.isin(cache_ids, list(ctx["known"]))].mean())})
        picks = sample_by_slide(pool, slide_of, rng, config["tiles_per_finding"], config["max_per_slide"])
        tiles, trs = extract_tiles(picks, finding, ctx, feature_dir, {"vote_fraction": top_frac})
        tile_rows += trs
        r = rows[-1]
        title = (f"{args.layer} [{lo},{hi}) {finding} | {r['n_patches']} sample patches / {r['n_slides']} slides / {r['n_compounds']} compounds"
                 f" | in finding-carrying slide {r['patch_in_carrying_slide']:.2f} (slide prevalence {r['slide_prevalence_known']:.2f})"
                 f" | green=slide has finding, red=not")
        safe = "".join(c if c.isalnum() else "_" for c in finding)
        draw_sheet(title, tiles, run_dir / "sheets" / f"{safe}.jpg")
        logger.info(f"{finding}: {len(pool)} patches, {len(tiles)} tiles")

    pd.DataFrame(rows).sort_values("n_patches", ascending=False).to_csv(run_dir / "layer_summary.csv", index=False)
    pd.DataFrame(tile_rows).to_csv(run_dir / "tiles.csv", index=False)
    (run_dir / "results.json").write_text(json.dumps(
        {"n_runs": ctx["n_runs"], "layer": [lo, hi], "n_members": int(member.sum()), "n_findings_sheeted": len(rows)},
        indent=2, ensure_ascii=False))


def run_q_vote(config: dict, ctx: dict, run_dir: Path, feature_dir: Path, logger: logging.Logger) -> None:
    """投票率と、候補クラスタのq値（run間の中央値）の関係を所見ごとに見る。"""
    from scipy.stats import spearmanr

    qc = config["q_vote"]
    top_f, top_frac, findings = ctx["top_f"], ctx["top_frac"], ctx["findings"]
    log_q, size = ctx["median_log_q"], ctx["median_size"]
    slide_of, manifest = ctx["slide_of"], ctx["manifest"]
    base = ctx["in_eval"] & (top_frac >= qc["min_vote_corr"])
    logger.info(f"correlation pool: {int(base.sum())} patches with vote >= {qc['min_vote_corr']} (eval slides only)")

    # patch単位で、同じスライドのpatchは独立でないため、スライド単位（中央値）の相関も併記する
    def corr_row(finding: str, idx: np.ndarray) -> dict:
        per_slide = pd.DataFrame({"s": slide_of[idx], "v": top_frac[idx], "q": log_q[idx], "n": size[idx]}).groupby("s").median()
        return {"finding": finding, "n_patches": len(idx), "n_slides": len(per_slide),
                "rho_vote_logq": spearmanr(top_frac[idx], log_q[idx])[0],
                "rho_vote_size": spearmanr(top_frac[idx], size[idx])[0],
                "rho_logq_size": spearmanr(log_q[idx], size[idx])[0],
                "rho_vote_logq_slide": spearmanr(per_slide.v, per_slide.q)[0] if len(per_slide) > 2 else np.nan}

    rows = [corr_row("ALL", np.flatnonzero(base))]
    for fi, finding in enumerate(findings):
        idx = np.flatnonzero(base & (top_f == fi))
        if len(idx) >= config["min_patches_per_finding"]:
            rows.append(corr_row(finding, idx))
    pd.DataFrame(rows).to_csv(run_dir / "q_vote_correlation.csv", index=False)

    # 投票率の区間ごとのq値・クラスタ大きさの分布（所見をまたいだ全体）
    bins = qc["vote_bins"]
    b = pd.cut(top_frac[base], bins, right=False)
    pd.DataFrame({"vote_bin": b, "log10_q": log_q[base], "cluster_size": size[base]}).groupby("vote_bin", observed=True).describe(
        percentiles=[0.25, 0.5, 0.75]).to_csv(run_dir / "q_by_vote_bin.csv")

    # 目視用: 投票率が一定以上のpatchのうち、q値の低い側と高い側の四分位からそれぞれ並べる
    rng = np.random.default_rng(config["seed"])
    sheet_pool = ctx["in_eval"] & (top_frac >= qc["min_vote_sheet"])
    tile_rows, n = [], qc["tiles_per_side"]
    extra = {"vote_fraction": top_frac, "median_log10_q": log_q, "median_cluster_size": size}
    for fi, finding in enumerate(findings):
        pool = np.flatnonzero(sheet_pool & (top_f == fi))
        if len(pool) < config["min_patches_per_finding"]:
            continue
        lo_cut, hi_cut = np.quantile(log_q[pool], [0.25, 0.75])
        sides = {"low_q": pool[log_q[pool] <= lo_cut], "high_q": pool[log_q[pool] >= hi_cut]}
        tiles = []
        for side, sp in sides.items():
            picks = sample_by_slide(sp, slide_of, rng, n, config["max_per_slide"])
            t, trs = extract_tiles(picks, finding, ctx, feature_dir, extra)
            tiles += t + [None] * (n - len(t))  # 片側が足りなくても上下の段がずれないように空ける
            tile_rows += [{**tr, "q_side": side} for tr in trs]
        title = (f"vote>={qc['min_vote_sheet']} {finding} | top {n // COLS} rows: lowest-q quartile (log10q<={lo_cut:.1f})"
                 f" | bottom {n // COLS} rows: highest-q quartile (log10q>={hi_cut:.1f}) | {len(pool)} patches | green=slide has finding")
        safe = "".join(c if c.isalnum() else "_" for c in finding)
        draw_sheet(title, tiles, run_dir / "sheets" / f"{safe}.jpg")
        logger.info(f"{finding}: {len(pool)} patches, q quartiles {lo_cut:.1f} / {hi_cut:.1f}")
    pd.DataFrame(tile_rows).to_csv(run_dir / "tiles.csv", index=False)
    (run_dir / "results.json").write_text(json.dumps(
        {"n_runs": ctx["n_runs"], "n_corr_pool": int(base.sum()), "n_sheet_pool": int(sheet_pool.sum())}, indent=2))


def run_vote_steps(config: dict, ctx: dict, run_dir: Path, feature_dir: Path, logger: logging.Logger) -> None:
    """所見ごとに投票率を区間に分け、区間ごと・閾値以上の累積の集計と、区間を1段ずつ並べたシートを作る。"""
    vc = config["vote_steps"]
    edges, n = vc["edges"], vc["tiles_per_bin"]
    top_f, top_frac, findings, in_eval = ctx["top_f"], ctx["top_frac"], ctx["findings"], ctx["in_eval"]
    cons, second_frac = ctx["consistency"], ctx["second_frac"]
    second_name = np.where(second_frac > 0, np.asarray(findings)[ctx["second_f"]], "")
    extra = {"vote_fraction": top_frac, "consistency": cons, "second_finding": second_name, "second_vote_fraction": second_frac}
    rng = np.random.default_rng(config["seed"])
    table, tile_rows = [], []
    for fi, finding in enumerate(findings):
        own = in_eval & (top_f == fi) & (top_frac >= edges[0])  # 票のないpatchはargmaxで先頭の所見になるので除く
        if (own & (top_frac >= vc["min_vote_for_finding"])).sum() < config["min_patches_per_finding"]:
            continue
        tiles = []
        for lo, hi in reversed(list(zip(edges[:-1], edges[1:]))):  # 投票率の高い区間から上に並べる
            pool = np.flatnonzero(own & (top_frac >= lo) & (top_frac < hi))
            cum = np.flatnonzero(own & (top_frac >= lo))
            row = {"finding": finding, "bin_lo": lo, "bin_hi": hi}
            row.update({f"bin_{k}": v for k, v in pool_stats(pool, finding, ctx).items()} if len(pool) else {"bin_n_patches": 0})
            row.update({f"cum_{k}": v for k, v in pool_stats(cum, finding, ctx).items()} if len(cum) else {"cum_n_patches": 0})
            row["bin_median_consistency"] = float(np.median(cons[pool])) if len(pool) else np.nan
            table.append(row)
            picks = sample_by_slide(pool, ctx["slide_of"], rng, n, config["max_per_slide"]) if len(pool) else np.array([], int)
            t, trs = extract_tiles(picks, finding, ctx, feature_dir, extra)
            tiles += [{**x, "tag": f"{lo:.2g} c{cons[p]:.2f} ", "band": next(c for b, c in BAND_COLORS if cons[p] >= b)}
                      for x, p in zip(t, picks)] + [None] * (n - len(t))  # 足りない区間は空けて段をそろえる
            tile_rows += [{**tr, "bin_lo": lo} for tr in trs]
        title = (f"{finding} | vote-fraction bins, top={edges[-2]:.1f}-1.0 ... bottom={edges[0]:.2f}-{edges[1]:.1f}, "
                 f"{n} tiles ({n // COLS} rows) per bin | label: bin lower edge, c=consistency | "
                 f"header band c>=0.9 dark blue / >=0.7 light blue / >=0.5 orange / <0.5 red | border green=slide has finding")
        safe = "".join(c if c.isalnum() else "_" for c in finding)
        draw_sheet(title, tiles, run_dir / "sheets" / f"{safe}.jpg")
        logger.info(f"{finding}: {int(own.sum())} patches with vote >= {edges[0]}")
    pd.DataFrame(table).to_csv(run_dir / "vote_step_table.csv", index=False)
    pd.DataFrame(tile_rows).to_csv(run_dir / "tiles.csv", index=False)
    (run_dir / "results.json").write_text(json.dumps(
        {"n_runs": ctx["n_runs"], "n_findings_sheeted": int(pd.DataFrame(table).finding.nunique())}, indent=2))


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    args = parse_args()
    config = load_config(Path(__file__).parent)
    if args.analysis == "layer" and args.layer not in config["layers"]:
        sys.exit(f"--layer must be one of {list(config['layers'])}")
    variant = args.layer if args.analysis == "layer" else args.analysis
    run_dir = get_run_dir(project_root, __file__, variant, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    meta = {"layer": args.layer, "lo": config["layers"][args.layer][0], "hi": config["layers"][args.layer][1]} \
        if args.analysis == "layer" else {"analysis": args.analysis, **config[args.analysis]}
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant, **meta)
    logger.info(f"Starting: {exp_name} / {variant}")

    ctx = load_context(project_root, config, with_q=args.analysis == "q_vote")
    logger.info(f"votes from {ctx['n_runs']} runs over {len(ctx['top_f'])} sample patches, {len(ctx['findings'])} findings")
    feature_dir = project_root / config["feature_dir"]
    if args.analysis == "layer":
        run_layer(args, config, ctx, run_dir, feature_dir, logger)
    elif args.analysis == "q_vote":
        run_q_vote(config, ctx, run_dir, feature_dir, logger)
    else:
        run_vote_steps(config, ctx, run_dir, feature_dir, logger)
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
