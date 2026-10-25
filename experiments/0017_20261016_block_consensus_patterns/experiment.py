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
    parser.add_argument("--variant", default="pilot")       # config.ymlのvariantsのキー（対象の試験）
    return parser.parse_args()


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TILE, COLS, HEAD = 192, 8, 30


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _read_one(args):
    """1枚分: h5を丸ごと読んでから必要な行を抜く（h5pyの飛び飛びの読み出しは非常に遅いため、2026-10-21変更）。"""
    path, sid, cfg = args
    a, b = cfg["pca_patches_per_slide"], cfg["pca_patches_per_slide"] + cfg["train_pool_per_slide"]
    n_need = b + cfg["eval_patches_per_slide"]
    with h5py.File(path, "r") as h:
        feats = h["features"][:]
        coords = h["coords"][:]
    n = len(feats)
    pick = np.sort(np.random.default_rng([cfg["seed"], int(sid)]).choice(n, size=min(n_need, n), replace=False))
    f = l2(feats[pick].astype(np.float32))
    xy = coords[pick]
    perm = np.random.default_rng([cfg["seed"], int(sid), 1]).permutation(len(pick))
    return f[perm[:a]], f[perm[a:b]], f[perm[b:]], xy[perm[b:]]


def load_subsets(slides: pd.DataFrame, features_dir, cfg: dict, logger):
    """各slideのh5を1回だけ読み、PCA用・学習プール・評価用のpatchを互いに重ならないよう抜き出す（1536次元、L2正規化）。
    slideごとの読み込みは複数プロセスで並列に行う（cfg["io_workers"]）。結果はslideの順に並ぶ。"""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [((features_dir(sid) if callable(features_dir) else features_dir / f"{sid}.h5"), sid, cfg) for sid in slides.slide_id]
    out = {k: [] for k in ["pca", "train", "eval"]}
    rows = {k: [] for k in out}
    eval_xy = []
    with ProcessPoolExecutor(max_workers=cfg.get("io_workers", 1)) as ex:
        for i, (fp, ft, fe, xy) in enumerate(ex.map(_read_one, jobs, chunksize=4)):
            for key, arr in [("pca", fp), ("train", ft), ("eval", fe)]:
                out[key].append(arr)
                rows[key].append(np.full(len(arr), i))
            eval_xy.append(xy)
            if (i + 1) % 200 == 0:
                logger.info(f"  read {i + 1}/{len(slides)} slides")
    return ({k: np.concatenate(v) for k, v in out.items()}, {k: np.concatenate(v) for k, v in rows.items()},
            np.concatenate(eval_xy))


def _tissue_one(args):
    svs, xy, size0, cfg = args
    w = openslide.OpenSlide(svs)
    level = w.get_best_level_for_downsample(cfg["tissue_level_downsample"])
    ds = w.level_downsamples[level]
    px = max(8, int(round(size0 / ds)))
    out = []
    for x, y in xy:
        rgb = np.asarray(w.read_region((int(x), int(y)), level, (px, px)).convert("RGB"), dtype=np.float32)
        out.append((rgb.mean(axis=2) < cfg["tissue_white_threshold"]).mean())
    w.close()
    return np.asarray(out)


def tissue_fraction(slides: pd.DataFrame, row: np.ndarray, xy: np.ndarray, size0: dict, cfg: dict, logger=None) -> np.ndarray:
    """評価用patchの組織割合（白に近くない画素の割合）。低解像度の層から読む。slideごとに複数プロセスで並列。"""
    from concurrent.futures import ProcessPoolExecutor
    frac = np.zeros(len(row))
    idxs = [np.flatnonzero(row == i) for i in range(len(slides))]
    jobs = [(sl.svs_path, xy[ix], size0[sl.slide_id], cfg) for sl, ix in zip(slides.itertuples(), idxs)]
    with ProcessPoolExecutor(max_workers=cfg.get("io_workers", 1)) as ex:
        for i, (ix, v) in enumerate(zip(idxs, ex.map(_tissue_one, jobs, chunksize=4))):
            frac[ix] = v
            if logger and (i + 1) % 500 == 0:
                logger.info(f"  tissue fraction {i + 1}/{len(slides)} slides")
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
            # 30%以上のslideに出ている所見の中から倍率の高い順に選ぶ（2026-10-17修正: 以前は倍率上位3つを先に取ってから
            # 30%で絞っていたため、まれな所見が上位を占めると30%超の所見があっても表示されなかった）
            common = f[f >= 0.3]
            lift = (common / base_rate.reindex(common.index)).sort_values(ascending=False)
            top = f.sort_values(ascending=False).head(3)
            rec["treated_slides"] = len(treated)
            rec["treated_with_finding_frac"] = float(treated.has_finding.mean())
            rec["top_findings"] = "; ".join(f"{k} {v:.2f}" for k, v in top.items())
            rec["top_lift_findings"] = "; ".join(f"{k} x{v:.1f}" for k, v in lift.head(3).items())
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


def per_k_consensus(labels, run_meta, pairs, rate_k, row, slides, base_rate, cfg, tau, logger=None):
    """kごとに分けてコンセンサスのコアを作る（2026-10-18方針: kを混ぜない）。

    各kについて、そのkのrunだけの共所属率でコアを作り、そのkのrunを半分ずつに分けて一致度を測る。
    さらに細かいkのコアが1つ粗いkのどのコアにどれだけ入るか（包含率）をたどり、全粒度で入れ子になるかを判定する。
    Returns: cores {k: (n,) array}, tables {k: DataFrame}, chains DataFrame（最も細かいkのコアごと）
    """
    a, b = pairs[:, 0], pairs[:, 1]
    n = len(row)
    ks = sorted(rate_k)
    rng = np.random.default_rng(cfg["seed"])
    cores, tables = {}, {}
    for k in ks:
        idx = np.flatnonzero(run_meta.k.to_numpy() == k)
        rate = rate_k[k]
        core = cores_from_edges(n, pairs, rate, tau, "leiden", row, cfg)
        js = []
        for _ in range(cfg["n_split_repeats"]):
            p = rng.permutation(idx)
            for h in (p[: len(p) // 2], p[len(p) // 2:]):
                r = (labels[h][:, a] == labels[h][:, b]).mean(0)
                js.append(best_jaccard(core, cores_from_edges(n, pairs, r, tau, "leiden", row, cfg)))
        same = (core[a] == core[b]) & (core[a] >= 0)
        cs = np.bincount(a[same], weights=rate[same], minlength=n) + np.bincount(b[same], weights=rate[same], minlength=n)
        cn = np.bincount(a[same], minlength=n) + np.bincount(b[same], minlength=n)
        tab = core_table(core, row, slides, np.where(cn > 0, cs / np.maximum(cn, 1), 0.0), base_rate)
        tab["split_half_J"] = tab.core.map(dict(enumerate(np.median(np.stack(js), axis=0)))) if len(tab) else []
        cores[k], tables[k] = core, tab
        if logger:
            lo = tab.ctrl_patch_frac < 0.05
            logger.info(f"per-k k={k}: {len(tab)} cores, assigned {(core >= 0).mean():.2f}, J median {tab.split_half_J.median():.2f}, "
                        f"Control<5% {int(lo.sum())} (J {tab[lo].split_half_J.median():.2f})")
    parent = {}
    for fine, coarse in zip(ks[::-1][:-1], ks[::-1][1:]):
        recs = []
        for c in range(cores[fine].max() + 1):
            pc = pd.Series(cores[coarse][cores[fine] == c]).value_counts(normalize=True).drop(-1, errors="ignore")
            recs.append({"core": c, "parent": int(pc.index[0]) if len(pc) else -1, "contained": float(pc.iloc[0]) if len(pc) else 0.0})
        parent[(fine, coarse)] = pd.DataFrame(recs).set_index("core")
    chains = []
    for r in tables[ks[-1]].itertuples():
        rec = {f"core_k{ks[-1]}": r.core, "n_patches": r.n_patches, "ctrl_frac": r.ctrl_patch_frac, "top_exp": r.top_exp,
               "top_exp_share": r.top_exp_share, "findings": lift_text(r), f"J_k{ks[-1]}": r.split_half_J}
        cur, ok = r.core, True
        for fine, coarse in zip(ks[::-1][:-1], ks[::-1][1:]):
            q = parent[(fine, coarse)].loc[cur] if cur >= 0 else None
            par, cont = (int(q.parent), float(q.contained)) if q is not None else (-1, 0.0)
            rec.update({f"parent_k{coarse}": par, f"contained_in_k{coarse}": cont,
                        f"J_k{coarse}": tables[coarse].set_index("core").split_half_J.get(par, np.nan)})
            ok = ok and cont >= cfg.get("nest_contained_min", 0.7)
            cur = par
        rec["nested_all_levels"] = ok
        chains.append(rec)
    return cores, tables, pd.DataFrame(chains)


def select_slides(cfg: dict, project_root: Path) -> pd.DataFrame:
    """対象slideの表（行の順がpatchのslide番号になる）。可視化スクリプトからも使う。"""
    m = pd.read_parquet(project_root / cfg["manifest_parquet"])
    m["slide_id"] = m.slide_id.astype(str)
    if "pathology_label_status" in m:   # 所見不明・同一個体の重複は除く（0010と同じ規則）
        m = m[m.pathology_label_status != "unknown"]
    if "duplicate_animal_image" in m:
        m = m[~m.duplicate_animal_image]
    slides = m[m.exp_id.isin(cfg["exp_ids"])].sort_values(["exp_id", "dose", "sacrifice_period", "individual_id"]).reset_index(drop=True)
    if "scanner" not in slides:
        scan = pd.read_csv(project_root / "experiments/0010_20260925_pathology_benchmark_subset/label_audit/slide_scanner_meta.csv",
                           dtype={"slide_id": str})[["slide_id", "aperio.ScanScope ID"]].rename(columns={"aperio.ScanScope ID": "scanner"})
        slides = slides.merge(scan, on="slide_id", how="left", validate="one_to_one")
    if cfg.get("qc_tiers"):   # tggate-wsi-qcの目録の区分で絞る（2026-10-22: ぼやけ・縞がクラスタリングに効くため）
        q = pd.read_csv(cfg["qc_catalog"], dtype={"slide_id": str}, usecols=["slide_id", "tier"])
        slides = slides[slides.slide_id.isin(q[q.tier.isin(cfg["qc_tiers"])].slide_id)].reset_index(drop=True)
    return slides


def derive_cache(src: Path, src_slides: pd.DataFrame, slides: pd.DataFrame, cache: Path) -> None:
    """別variantのキャッシュから、slidesに含まれるslideの分だけ抜き出す。
    patchの抜き方はslideごとに決まっている（_read_oneのseedはslide_id）ので、読み直した場合と同じになる。"""
    new_idx = pd.Series(np.arange(len(slides)), index=slides.slide_id)
    remap = src_slides.slide_id.map(new_idx).to_numpy()          # 元のslide番号 → 新しい番号（含まれなければNaN）
    z = np.load(src / "subsets.npz")
    out = {}
    for k in ["pca", "train", "eval"]:
        r = remap[z[f"rows_{k}"]]
        m = ~np.isnan(r)
        out[k], out[f"rows_{k}"] = z[k][m], r[m].astype(np.int64)
        if k == "eval":
            out["eval_xy"] = z["eval_xy"][m]
            np.save(cache / "tissue_frac.npy", np.load(src / "tissue_frac.npy")[m])
    np.savez(cache / "subsets.npz", **out)


def cache_dir(cfg: dict, variant_key: str, project_root: Path) -> Path:
    """抜き出したpatchのキャッシュ。cache_variantがあればそのvariantのものを共有する（slideの組が同じとき）。"""
    return project_root / "outputs/0017_20261016_block_consensus_patterns" / f"{cfg.get('cache_variant', variant_key)}_cache"


def center_by_slide(feats: dict, rows: dict, n_slides: int):
    """各slideの平均（抜き出した全patch: PCA用・学習用・評価用）を引き、slide内の相対的な方向だけを残す（2026-10-23）。
    Returns: (平均を引いた特徴量, slideごとの平均 (n_slides, d))"""
    from scipy.sparse import csr_matrix
    keys = list(feats)
    r = np.concatenate([rows[k] for k in keys])
    ind = csr_matrix((np.ones(len(r), dtype=np.float32), (r, np.arange(len(r)))), shape=(n_slides, len(r)))
    sums = np.asarray(ind @ np.concatenate([feats[k] for k in keys]))
    means = (sums / np.bincount(r, minlength=n_slides)[:, None]).astype(np.float32)
    return {k: feats[k] - means[rows[k]] for k in keys}, means


def load_cached(cfg: dict, variant_key: str, project_root: Path, slides: pd.DataFrame) -> dict:
    """キャッシュを読み、center_per_slideなら平均を引く。Returns: feats, rows, eval_xy, means（引かないならNone）"""
    z = np.load(cache_dir(cfg, variant_key, project_root) / "subsets.npz")
    feats = {k: z[k] for k in ["pca", "train", "eval"]}
    rows = {k: z[f"rows_{k}"] for k in ["pca", "train", "eval"]}
    means = None
    if cfg.get("center_per_slide"):
        feats, means = center_by_slide(feats, rows, len(slides))
    return {"feats": feats, "rows": rows, "eval_xy": z["eval_xy"], "means": means}


def fit_pca(cfg: dict, slides: pd.DataFrame, feats_pca: np.ndarray, rows_pca: np.ndarray):
    """PCAはControlと投与群から同数のpatchで学習する（0003の注意）。"""
    from sklearn.decomposition import PCA
    is_ctrl = (slides.dose_level == "Control").to_numpy()
    c, t = np.flatnonzero(is_ctrl[rows_pca]), np.flatnonzero(~is_ctrl[rows_pca])
    t = np.random.default_rng(cfg["seed"]).choice(t, size=min(len(c), len(t)), replace=False)
    return PCA(n_components=cfg["pca_components"], svd_solver="randomized", random_state=cfg["seed"]).fit(feats_pca[np.r_[c, t]])


def cached_pca(cfg: dict, variant_key: str, project_root: Path, slides: pd.DataFrame):
    """main()と同じ手順・seedでPCAを学習し直す。Returns: (fitted PCA, load_cachedの結果)"""
    d = load_cached(cfg, variant_key, project_root, slides)
    return fit_pca(cfg, slides, d["feats"]["pca"], d["rows"]["pca"]), d


def cached_eval_embedding(cfg: dict, variant_key: str, project_root: Path):
    """main()のキャッシュから、main()と同じ手順でPCAを当て直す（可視化・0018から使う）。
    組織割合で除いた後の評価用patchを返すので、順番は eval_patches.parquet と同じ。
    Returns: slides, ev (PCA後), feats (1536次元。center_per_slideなら平均を引いたもの), row, xy"""
    slides = select_slides(cfg, project_root)
    pca, d = cached_pca(cfg, variant_key, project_root, slides)
    keep = np.load(cache_dir(cfg, variant_key, project_root) / "tissue_frac.npy") >= cfg["min_tissue_frac"]
    feats = d["feats"]["eval"][keep]
    return slides, pca.transform(feats).astype(np.float32), feats, d["rows"]["eval"][keep], d["eval_xy"][keep]


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.clustering import _undirected_pairs, kmeans_lloyd, knn_graph
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    args = parse_args()
    exp_name = os.environ["EXP_NAME"]
    cfg = load_config(Path(__file__).parent)
    variant_key = args.variant
    cfg.update(cfg.get("variants", {}).get(variant_key, {}))   # 試験の組だけ差し替える
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, config=cfg)
    logger.info(f"Starting: {exp_name} / {variant_key}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slides = select_slides(cfg, project_root)
    # 特徴量は全patch版を優先し、なければ（追加Controlなど）2,048 patch版を使う
    fdirs = [project_root / d for d in cfg.get("features_dirs", [cfg["features_dir"]])]

    def features_dir(sid):
        for d in fdirs:
            if (d / f"{sid}.h5").exists():
                return d / f"{sid}.h5"
        return None
    missing = [s for s in slides.slide_id if features_dir(s) is None]
    if missing:
        raise FileNotFoundError(f"{len(missing)} slides lack features (0016 full phase): {missing[:5]}")
    treated = slides[slides.dose_level != "Control"]
    base_rate = treated.finding_types.explode().dropna().value_counts() / len(treated)
    logger.info(f"{len(slides)} slides ({(slides.dose_level == 'Control').sum()} Control), {slides.exp_id.nunique()} exp")

    # --- patchの抜き出しとPCA ---（時間がかかるので結果を保存し、やり直し時は読み込む。2026-10-21）
    t0 = time.time()
    cache = cache_dir(cfg, variant_key, project_root)
    cache.mkdir(parents=True, exist_ok=True)
    if not (cache / "subsets.npz").exists() and cfg.get("derive_cache_from"):
        derive_cache(project_root / "outputs" / exp_name / f"{cfg['derive_cache_from']}_cache",
                     select_slides({**cfg, "qc_tiers": None}, project_root), slides, cache)
        logger.info(f"derived subsets from {cfg['derive_cache_from']}_cache")
    if (cache / "subsets.npz").exists():
        z = np.load(cache / "subsets.npz")
        feats = {k: z[k] for k in ["pca", "train", "eval"]}
        rows = {k: z[f"rows_{k}"] for k in ["pca", "train", "eval"]}
        eval_xy = z["eval_xy"]
        logger.info(f"loaded cached subsets from {cache}")
    else:
        feats, rows, eval_xy = load_subsets(slides, features_dir, cfg, logger)
        np.savez(cache / "subsets.npz", eval_xy=eval_xy, **feats, **{f"rows_{k}": v for k, v in rows.items()})
    if cfg.get("center_per_slide"):
        feats, means = center_by_slide(feats, rows, len(slides))
        np.save(run_dir / "slide_means.npy", means)
        logger.info("subtracted per-slide means (slide-relative directions only)")
    is_ctrl = (slides.dose_level == "Control").to_numpy()
    ctrl_rows = np.flatnonzero(is_ctrl[rows["pca"]])
    trt_rows = np.flatnonzero(~is_ctrl[rows["pca"]])[: len(ctrl_rows)]   # 件数の記録用（学習はfit_pcaで同数を無作為抽出）
    pca = fit_pca(cfg, slides, feats["pca"], rows["pca"])
    train = pca.transform(feats["train"]).astype(np.float32)
    ev = pca.transform(feats["eval"]).astype(np.float32)
    del feats
    logger.info(f"PCA{cfg['pca_components']} (Control {len(ctrl_rows)} + treated {len(trt_rows)} patches): "
                f"explained {pca.explained_variance_ratio_.sum():.3f}; train pool {len(train):,}, eval {len(ev):,}; {time.time() - t0:.0f}s")

    size0 = {}
    for sid in slides.slide_id:
        with h5py.File(features_dir(sid), "r") as h:
            size0[sid] = int(h["coords"].attrs.get("patch_size_level0", 256))
    t0 = time.time()
    if (cache / "tissue_frac.npy").exists():
        tfrac = np.load(cache / "tissue_frac.npy")
    else:
        tfrac = tissue_fraction(slides, rows["eval"], eval_xy, size0, cfg, logger)
        np.save(cache / "tissue_frac.npy", tfrac)
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
        for method in cfg.get("mixed_methods", ["components", "leiden"]):
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
        for method in cfg.get("mixed_methods", ["components", "leiden"]):
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
    for method in cfg.get("mixed_methods", ["components", "leiden"]):
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

    # --- kごとのコンセンサス（2026-10-18方針）と、全粒度で入れ子かつControlが少ないコアの画像シート ---
    if cfg.get("per_k", False):
        t0 = time.time()
        tau = cfg["sheet_tau"]
        cores_k, tables_k, chains = per_k_consensus(labels, run_meta, pairs, rate_k, erow, slides, base_rate, cfg, tau, logger)
        out = run_dir / "per_k"
        out.mkdir(exist_ok=True)
        for k, t in tables_k.items():
            t.sort_values("n_patches", ascending=False).to_csv(out / f"cores_k{k}_tau{tau:g}.csv", index=False)
        ev_table.assign(**{f"core_k{k}": c for k, c in cores_k.items()})[["slide_id", "x", "y"] + [f"core_k{k}" for k in cores_k]] \
            .to_parquet(out / f"assignments_tau{tau:g}.parquet", index=False)
        chains.to_csv(out / f"chains_tau{tau:g}.csv", index=False)
        kf = max(cores_k)
        core, rf = cores_k[kf], rate_k[kf]
        same = (core[pairs[:, 0]] == core[pairs[:, 1]]) & (core[pairs[:, 0]] >= 0)
        c_sum = np.bincount(pairs[same, 0], weights=rf[same], minlength=n) + np.bincount(pairs[same, 1], weights=rf[same], minlength=n)
        c_n = np.bincount(pairs[same, 0], minlength=n) + np.bincount(pairs[same, 1], minlength=n)
        centrality = np.where(c_n > 0, c_sum / np.maximum(c_n, 1), 0.0)
        keep_c = chains[chains.nested_all_levels & (chains.ctrl_frac < 0.05)][f"core_k{kf}"]
        sel = tables_k[kf][tables_k[kf].core.isin(keep_c)].sort_values(["top_exp_share", "n_patches"], ascending=[True, False])
        for f0 in range(0, len(sel), 10):
            sheet_rows = []
            for r in sel.iloc[f0:f0 + 10].itertuples():
                members = np.flatnonzero(core == r.core)
                members = members[np.argsort(-centrality[members])]
                picks, seen = [], set()
                for p in members:
                    if erow[p] not in seen:
                        picks.append(p)
                        seen.add(erow[p])
                    if len(picks) == cfg["sheet_tiles"]:
                        break
                tiles = read_tiles(picks, erow, exy, slides, size0,
                                   lambda sl, p: (f"e{sl.exp_id} {sl.compound_name[:13]} {sl.dose_level[0]}",
                                                  f"{sl.sacrifice_period} {sl.scanner} {'F' if sl.has_finding else '-'}"))
                exps = slides.iloc[erow[members]].compound_name.value_counts(normalize=True)
                header = (f"core {r.core}: {r.n_patches} patches / {r.n_slides} slides / {r.n_exp} exp | Ctrl {r.ctrl_patch_frac:.2f} | "
                          f"J {r.split_half_J:.2f} | {lift_text(r)} | " + ", ".join(f"{k_[:12]} {v:.0%}" for k_, v in exps.head(4).items()))
                sheet_rows.append((header, tiles))
            draw_sheet(f"0017 {variant_key}: k={kf} cores nested across k={sorted(cores_k)} with Control < 5% (tau={tau}), "
                       f"multi-experiment first. F = slide has a recorded finding", sheet_rows,
                       run_dir / "sheets" / f"per_k{kf}_nested_ctrl_lt0.05" / f"cores_{f0 // 10 + 1:02d}.jpg")
        logger.info(f"per-k done: {int(chains.nested_all_levels.sum())} nested, {len(sel)} nested with Control<5%; {time.time() - t0:.0f}s")

    (run_dir / "results.json").write_text(json.dumps(
        {"n_slides": len(slides), "n_eval_patches": int(n), "n_eval_excluded_blank": int((~keep).sum()),
         "n_runs": len(labels), "n_edges": int(len(pairs)), "summary": summary}, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
