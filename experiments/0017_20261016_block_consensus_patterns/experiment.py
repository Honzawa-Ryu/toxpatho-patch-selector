import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd
import torch

import h5py
import openslide
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")   # run_slurm.shが渡す。読むのは常に実験ディレクトリのconfig.yml
    return parser.parse_args()


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TILE, COLS, HEAD = 192, 8, 30


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def load_subsets(slides: pd.DataFrame, features_dir: Path, cfg: dict, logger):
    """各slideのh5を1回だけ読み、PCA用・学習プール・評価用のpatchを互いに重ならないよう抜き出す（1536次元、L2正規化）。"""
    out = {k: [] for k in ["pca", "train", "eval"]}
    rows = {k: [] for k in out}
    eval_xy = []
    n_need = cfg["pca_patches_per_slide"] + cfg["train_pool_per_slide"] + cfg["eval_patches_per_slide"]
    for i, sid in enumerate(slides.slide_id):
        with h5py.File(features_dir / f"{sid}.h5", "r") as h:
            n = h["features"].shape[0]
            pick = np.sort(np.random.default_rng([cfg["seed"], int(sid)]).choice(n, size=min(n_need, n), replace=False))
            f = l2(h["features"][pick].astype(np.float32))
            xy = h["coords"][:][pick]
        perm = np.random.default_rng([cfg["seed"], int(sid), 1]).permutation(len(pick))
        a, b = cfg["pca_patches_per_slide"], cfg["pca_patches_per_slide"] + cfg["train_pool_per_slide"]
        for key, sel in [("pca", perm[:a]), ("train", perm[a:b]), ("eval", perm[b:])]:
            out[key].append(f[sel])
            rows[key].append(np.full(len(sel), i))
            if key == "eval":
                eval_xy.append(xy[sel])
        if (i + 1) % 100 == 0:
            logger.info(f"  read {i + 1}/{len(slides)} slides")
    return ({k: np.concatenate(v) for k, v in out.items()}, {k: np.concatenate(v) for k, v in rows.items()},
            np.concatenate(eval_xy))


def tissue_fraction(slides: pd.DataFrame, row: np.ndarray, xy: np.ndarray, size0: dict, cfg: dict) -> np.ndarray:
    """評価用patchの組織割合（白に近くない画素の割合）。低解像度の層から読む。"""
    frac = np.zeros(len(row))
    for i, sl in enumerate(slides.itertuples()):
        idx = np.flatnonzero(row == i)
        w = openslide.OpenSlide(sl.svs_path)
        level = w.get_best_level_for_downsample(cfg["tissue_level_downsample"])
        ds = w.level_downsamples[level]
        px = max(8, int(round(size0[sl.slide_id] / ds)))
        for j in idx:
            rgb = np.asarray(w.read_region((int(xy[j, 0]), int(xy[j, 1])), level, (px, px)).convert("RGB"), dtype=np.float32)
            frac[j] = (rgb.mean(axis=2) < cfg["tissue_white_threshold"]).mean()
        w.close()
    return frac


def assign(x: torch.Tensor, centers: np.ndarray) -> np.ndarray:
    return torch.cdist(x, torch.from_numpy(centers).to(x.device)).argmin(dim=1).cpu().numpy()


def cores_from_edges(n: int, pairs: np.ndarray, rate: np.ndarray, tau: float, method: str, row: np.ndarray,
                     cfg: dict) -> np.ndarray:
    """共所属率tau以上の辺だけ残してまとまりを取る。小さいもの・slideが少ないものは未割当（-1）。"""
    from lib.clustering import leiden_on_edges
    keep = rate >= tau
    p, w = pairs[keep], rate[keep]
    if method == "components":
        g = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n))
        _, lab = connected_components(g, directed=False)
    else:
        lab = leiden_on_edges(n, p, w, cfg["leiden_resolution"], cfg["seed"])
    out = np.full(n, -1)
    nxt = 0
    for c in np.unique(lab):
        m = lab == c
        if m.sum() >= cfg["min_core_patches"] and len(np.unique(row[m])) >= cfg["min_core_slides"]:
            out[m] = nxt
            nxt += 1
    return out


def best_jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """aの各コアについて、bのコアとの最大Jaccard（patch集合）。"""
    res = []
    for c in range(a.max() + 1):
        ma = a == c
        js = [((ma & (b == d)).sum() / (ma | (b == d)).sum()) for d in np.unique(b[ma]) if d >= 0]
        res.append(max(js) if js else 0.0)
    return np.asarray(res)


def core_table(core: np.ndarray, row: np.ndarray, slides: pd.DataFrame, centrality: np.ndarray,
               base_rate: pd.Series) -> pd.DataFrame:
    recs = []
    for c in range(core.max() + 1):
        m = core == c
        s = slides.iloc[np.unique(row[m])]
        patch_slides = slides.iloc[row[m]]
        treated = s[s.dose_level != "Control"]
        rec = {"core": c, "n_patches": int(m.sum()), "n_slides": len(s), "n_exp": s.exp_id.nunique(),
               "top_exp_share": float(patch_slides.compound_name.value_counts(normalize=True).iloc[0]),
               "top_exp": patch_slides.compound_name.value_counts().index[0],
               "ctrl_patch_frac": float((patch_slides.dose_level == "Control").mean()),
               "mean_centrality": float(centrality[m].mean())}
        for d in ["Control", "Low", "Middle", "High"]:
            rec[f"patches_{d}"] = int((patch_slides.dose_level == d).sum())
        for t in ["4 day", "8 day", "15 day", "29 day"]:
            rec[f"patches_{t.replace(' ', '')}"] = int((patch_slides.sacrifice_period == t).sum())
        for sc, n in patch_slides.scanner.value_counts().items():
            rec[f"patches_{sc}"] = int(n)
        # 所見: 投与群のslideのうちその所見を持つ割合を、13試験の投与群全体の割合（base）と比べる
        if len(treated):
            f = treated.finding_types.explode().dropna().value_counts() / len(treated)
            lift = (f / base_rate.reindex(f.index)).sort_values(ascending=False)
            top = f.sort_values(ascending=False).head(3)
            rec["treated_slides"] = len(treated)
            rec["treated_with_finding_frac"] = float(treated.has_finding.mean())
            rec["top_findings"] = "; ".join(f"{k} {v:.2f}" for k, v in top.items())
            rec["top_lift_findings"] = "; ".join(f"{k} x{v:.1f}" for k, v in lift.head(3).items() if f[k] >= 0.3)
        recs.append(rec)
    return pd.DataFrame(recs)


def lift_text(r) -> str:
    v = getattr(r, "top_lift_findings", "")
    return v if isinstance(v, str) else ""


def draw_sheet(title: str, rows: list[tuple[str, list[dict]]], path: Path) -> None:
    font = ImageFont.truetype(FONT, 12)
    h = TILE + HEAD
    sheet = Image.new("RGB", (COLS * TILE, 34 + sum((len(t) + COLS - 1) // COLS for _, t in rows) * h + 20 * len(rows)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), title, fill="black", font=ImageFont.truetype(FONT, 15))
    y = 34
    for header, tiles in rows:
        d.text((4, y + 2), header, fill="#b03a00", font=font)
        y += 20
        for j, t in enumerate(tiles):
            x, yy = (j % COLS) * TILE, y + (j // COLS) * h
            d.text((x + 3, yy + 2), t["l1"], fill="black", font=font)
            d.text((x + 3, yy + 15), t["l2"], fill="#555555", font=font)
            sheet.paste(t["image"], (x, yy + HEAD))
        y += ((len(tiles) + COLS - 1) // COLS) * h
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def read_tiles(picks, row, xy, slides, size0, labeler):
    tiles, wsis = [], {}
    for p in picks:
        sl = slides.iloc[row[p]]
        if sl.slide_id not in wsis:
            wsis[sl.slide_id] = openslide.OpenSlide(sl.svs_path)
        s = int(size0[sl.slide_id])
        img = wsis[sl.slide_id].read_region((int(xy[p, 0]), int(xy[p, 1])), 0, (s, s)).convert("RGB").resize((TILE, TILE))
        l1, l2_ = labeler(sl, p)
        tiles.append({"image": img, "l1": l1, "l2": l2_})
    for w in wsis.values():
        w.close()
    return tiles


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.clustering import _undirected_pairs, kmeans_lloyd, knn_graph
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from sklearn.decomposition import PCA

    parse_args()
    exp_name = os.environ["EXP_NAME"]
    cfg = load_config(Path(__file__).parent)
    variant_key = "pilot"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, config=cfg)
    logger.info(f"Starting: {exp_name} / {variant_key}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 対象slide ---
    m = pd.read_parquet(project_root / cfg["manifest_parquet"])
    m["slide_id"] = m.slide_id.astype(str)
    slides = m[m.exp_id.isin(cfg["exp_ids"])].sort_values(["exp_id", "dose", "sacrifice_period", "individual_id"]).reset_index(drop=True)
    scan = pd.read_csv(project_root / "experiments/0010_20260925_pathology_benchmark_subset/label_audit/slide_scanner_meta.csv",
                       dtype={"slide_id": str})[["slide_id", "aperio.ScanScope ID"]].rename(columns={"aperio.ScanScope ID": "scanner"})
    slides = slides.merge(scan, on="slide_id", how="left", validate="one_to_one")
    features_dir = project_root / cfg["features_dir"]
    missing = [s for s in slides.slide_id if not (features_dir / f"{s}.h5").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} slides lack features (0016 full phase): {missing[:5]}")
    treated = slides[slides.dose_level != "Control"]
    base_rate = treated.finding_types.explode().dropna().value_counts() / len(treated)
    logger.info(f"{len(slides)} slides ({(slides.dose_level == 'Control').sum()} Control), {slides.exp_id.nunique()} exp")

    # --- patchの抜き出しとPCA ---
    t0 = time.time()
    feats, rows, eval_xy = load_subsets(slides, features_dir, cfg, logger)
    is_ctrl = (slides.dose_level == "Control").to_numpy()
    pc = feats["pca"]
    ctrl_rows, trt_rows = np.flatnonzero(is_ctrl[rows["pca"]]), np.flatnonzero(~is_ctrl[rows["pca"]])
    trt_rows = np.random.default_rng(cfg["seed"]).choice(trt_rows, size=min(len(ctrl_rows), len(trt_rows)), replace=False)
    pca = PCA(n_components=cfg["pca_components"], svd_solver="randomized", random_state=cfg["seed"])
    pca.fit(pc[np.r_[ctrl_rows, trt_rows]])
    train = pca.transform(feats["train"]).astype(np.float32)
    ev = pca.transform(feats["eval"]).astype(np.float32)
    del feats
    logger.info(f"PCA{cfg['pca_components']} (Control {len(ctrl_rows)} + treated {len(trt_rows)} patches): "
                f"explained {pca.explained_variance_ratio_.sum():.3f}; train pool {len(train):,}, eval {len(ev):,}; {time.time() - t0:.0f}s")

    size0 = {}
    for sid in slides.slide_id:
        with h5py.File(features_dir / f"{sid}.h5", "r") as h:
            size0[sid] = int(h["coords"].attrs.get("patch_size_level0", 256))
    t0 = time.time()
    tfrac = tissue_fraction(slides, rows["eval"], eval_xy, size0, cfg)
    keep = tfrac >= cfg["min_tissue_frac"]
    logger.info(f"tissue fraction: {(~keep).sum():,} / {len(keep):,} eval patches below {cfg['min_tissue_frac']} excluded; {time.time() - t0:.0f}s")
    ev, erow, exy, tfrac_kept = ev[keep], rows["eval"][keep], eval_xy[keep], tfrac[keep]

    # --- 複数runのk-means ---
    x_ev = torch.from_numpy(ev).to(device)
    trow = rows["train"]
    labels, run_meta, centers = [], [], {}
    for k in cfg["ks"]:
        for s in cfg["sample_seeds"]:
            rng = np.random.default_rng([cfg["seed"], s])
            sel = np.concatenate([rng.choice(np.flatnonzero(trow == i), size=cfg["train_patches_per_slide"], replace=False)
                                  for i in range(len(slides)) if (trow == i).sum() >= cfg["train_patches_per_slide"]])
            for init in cfg["init_seeds"]:
                t0 = time.time()
                _, c, n_iter = kmeans_lloyd(train[sel], k, init, device)
                labels.append(assign(x_ev, c))
                key = f"k{k:03d}_s{s}_i{init}"
                centers[key] = c
                run_meta.append({"run": key, "k": k, "sample_seed": s, "init_seed": init, "n_iter": n_iter,
                                 "seconds": round(time.time() - t0, 1)})
        logger.info(f"k={k}: {len(cfg['sample_seeds']) * len(cfg['init_seeds'])} runs done")
    labels = np.stack(labels)                                    # (runs, eval)
    run_meta = pd.DataFrame(run_meta)
    np.savez_compressed(run_dir / "centers.npz", **centers)
    np.save(run_dir / "eval_labels.npy", labels)
    run_meta.to_csv(run_dir / "runs.csv", index=False)

    # --- 共所属グラフ ---
    t0 = time.time()
    nn, _ = knn_graph(ev, cfg["knn"], device)
    pairs = _undirected_pairs(nn)
    co = labels[:, pairs[:, 0]] == labels[:, pairs[:, 1]]    # (runs, edges)
    rate = co.mean(0)
    rate_k = {k: co[run_meta.k.to_numpy() == k].mean(0) for k in cfg["ks"]}
    logger.info(f"kNN graph: {len(pairs):,} edges; co-assignment rate median {np.median(rate):.2f}; {time.time() - t0:.0f}s")
    pd.DataFrame({"a": pairs[:, 0], "b": pairs[:, 1], "rate": rate, **{f"rate_k{k}": v for k, v in rate_k.items()}}) \
        .to_parquet(run_dir / "edges.parquet", index=False)

    # --- コア（tau × 方法）と安定性 ---
    n = len(ev)
    summary, assignments = [], {}
    for tau in cfg["taus"]:
        for method in ["components", "leiden"]:
            core = cores_from_edges(n, pairs, rate, tau, method, erow, cfg)
            assignments[f"{method}_tau{tau}"] = core
            sizes = np.bincount(core[core >= 0]) if (core >= 0).any() else np.array([])
            summary.append({"tau": tau, "method": method, "n_cores": int(core.max() + 1), "assigned_frac": float((core >= 0).mean()),
                            "largest_core_frac": float(sizes.max() / n) if len(sizes) else 0.0,
                            "median_core_patches": float(np.median(sizes)) if len(sizes) else 0.0})
            logger.info(f"tau={tau} {method}: {summary[-1]}")
    stab = []
    rng = np.random.default_rng(cfg["seed"])
    for rep in range(cfg["n_split_repeats"]):
        perm = rng.permutation(len(labels))
        halves = [perm[: len(perm) // 2], perm[len(perm) // 2:]]
        for method in ["components", "leiden"]:
            full = assignments[f"{method}_tau{cfg['sheet_tau']}"]
            for hi, hidx in enumerate(halves):
                r = (labels[hidx][:, pairs[:, 0]] == labels[hidx][:, pairs[:, 1]]).mean(0)
                half = cores_from_edges(n, pairs, r, cfg["sheet_tau"], method, erow, cfg)
                bj = best_jaccard(full, half)
                stab += [{"repeat": rep, "half": hi, "method": method, "core": c, "best_jaccard": float(v)} for c, v in enumerate(bj)]
    stab = pd.DataFrame(stab)
    stab.to_csv(run_dir / "core_stability.csv", index=False)
    pd.DataFrame(summary).to_csv(run_dir / "core_summary.csv", index=False)

    ev_table = pd.DataFrame({"slide_id": slides.slide_id.to_numpy()[erow], "x": exy[:, 0], "y": exy[:, 1], "tissue_frac": tfrac_kept,
                             **{k: v for k, v in assignments.items()}})
    ev_table.to_parquet(run_dir / "eval_patches.parquet", index=False)

    # --- コアの中身と画像シート（sheet_tau） ---
    for method in ["components", "leiden"]:
        core = assignments[f"{method}_tau{cfg['sheet_tau']}"]
        same = core[pairs[:, 0]] == core[pairs[:, 1]]
        c_sum = np.bincount(pairs[same, 0], weights=rate[same], minlength=n) + np.bincount(pairs[same, 1], weights=rate[same], minlength=n)
        c_n = np.bincount(pairs[same, 0], minlength=n) + np.bincount(pairs[same, 1], minlength=n)
        centrality = np.where(c_n > 0, c_sum / np.maximum(c_n, 1), 0.0)
        tab = core_table(core, erow, slides, centrality, base_rate) if core.max() >= 0 else pd.DataFrame()
        if len(tab):
            st = stab[stab.method == method].groupby("core").best_jaccard.median()
            tab["split_half_jaccard_median"] = tab.core.map(st)
            tab = tab.sort_values("n_patches", ascending=False)
        tab.to_csv(run_dir / f"cores_{method}_tau{cfg['sheet_tau']}.csv", index=False)
        sheet_rows = []
        for r in tab.head(cfg["sheet_max_cores"]).itertuples():
            members = np.flatnonzero(core == r.core)
            members = members[np.argsort(-centrality[members])]
            picks, seen = [], set()
            for p in members:   # 中心性の高い順に、slideが重ならないよう選ぶ
                if erow[p] not in seen:
                    picks.append(p)
                    seen.add(erow[p])
                if len(picks) == cfg["sheet_tiles"]:
                    break
            tiles = read_tiles(picks, erow, exy, slides, size0,
                               lambda sl, p: (f"e{sl.exp_id} {sl.compound_name[:13]} {sl.dose_level[0]}",
                                              f"{sl.sacrifice_period} {sl.scanner} t{tfrac_kept[p]:.2f}"))
            header = (f"core {r.core}: {r.n_patches} patches / {r.n_slides} slides / {r.n_exp} exp | Ctrl {r.ctrl_patch_frac:.2f} | "
                      f"top exp {r.top_exp[:16]} {r.top_exp_share:.2f} | split-half J {r.split_half_jaccard_median:.2f} | "
                      f"{lift_text(r)}")
            sheet_rows.append((header, tiles))
        draw_sheet(f"0017 cores ({method}, tau={cfg['sheet_tau']}), sorted by size; tiles by centrality, one per slide; "
                   f"label: exp, compound, dose (C/L/M/H), time, scanner, tissue fraction", sheet_rows,
                   run_dir / "sheets" / f"cores_{method}_tau{cfg['sheet_tau']}.jpg")
        un = np.flatnonzero(core < 0)
        pick = np.random.default_rng(cfg["seed"]).choice(un, size=min(48, len(un)), replace=False) if len(un) else []
        tiles = read_tiles(pick, erow, exy, slides, size0,
                           lambda sl, p: (f"e{sl.exp_id} {sl.compound_name[:13]} {sl.dose_level[0]}", f"{sl.scanner} t{tfrac_kept[p]:.2f}"))
        draw_sheet(f"0017 unassigned patches ({method}, tau={cfg['sheet_tau']}), random 48 of {len(un):,}",
                   [("unassigned", tiles)], run_dir / "sheets" / f"unassigned_{method}_tau{cfg['sheet_tau']}.jpg")

    (run_dir / "results.json").write_text(json.dumps(
        {"n_slides": len(slides), "n_eval_patches": int(n), "n_eval_excluded_blank": int((~keep).sum()),
         "n_runs": len(labels), "n_edges": int(len(pairs)), "summary": summary}, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
