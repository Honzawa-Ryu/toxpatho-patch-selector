import argparse
import importlib.util
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

import h5py
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    parser.add_argument("--variant", default="v1")
    parser.add_argument("--mode", default="build", choices=["build", "context", "context_build", "context_explore", "lobular_probe"])   # context: 隣接patchの情報を足してプロトタイプに割り当て直す
    parser.add_argument("--source", default="v3_maps")                             # context: 表現のモデルとプロトタイプを読む出力
    return parser.parse_args()


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 140})
WSI_DIRS = ["/workspace/andre01/honzawa/02-playground/tggate-wsi-qc/data/wsi", "/workspace/andre01/honzawa/02-playground/tggate-bench/data/wsi_all",
            "/workspace/andre01/honzawa/02-playground/toxpatho-perturbation-bench/data/raw_wsi",
            "/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/data/TGGATEs/WSI/Liver/4_day",
            "/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/data/TGGATEs/WSI/Liver/8_day",
            "/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/data/TGGATEs/WSI/Liver/15_day",
            "/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/data/TGGATEs/WSI/Liver/29_day"]   # 手元のsvsの置き場所


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   # 0020の並列読み込みで関数を子プロセスに渡すため（fork後の子でも見つかるように）
    spec.loader.exec_module(mod)
    return mod


def tgg_patches(ex17, cfg17, variant, project_root, skip_ids):
    """0017のキャッシュから評価用patch（組織5割以上）を取り、slide内の順位（キャッシュの約350 patch/枚が基準）を付ける。"""
    from scipy.stats import norm
    slides = ex17.select_slides(cfg17, project_root)
    d = ex17.load_cached(cfg17, variant, project_root, slides)
    keep = np.load(ex17.cache_dir(cfg17, variant, project_root) / "tissue_frac.npy") >= cfg17["min_tissue_frac"]
    ev, row, xy = d["feats"]["eval"][keep], d["rows"]["eval"][keep], d["eval_xy"][keep]
    allf = np.concatenate([d["feats"][k] for k in ["pca", "train", "eval"]])
    allr = np.concatenate([d["rows"][k] for k in ["pca", "train", "eval"]])
    rank = np.empty_like(ev)
    for s in np.unique(row):
        ref = np.sort(allf[allr == s], axis=0)
        m = row == s
        for j in range(ev.shape[1]):
            rank[m, j] = np.searchsorted(ref[:, j], ev[m, j])
        rank[m] = norm.ppf((rank[m] + 0.5) / len(ref))
    use = ~np.isin(slides.slide_id.to_numpy()[row], list(skip_ids))   # アノテーションのslideと重なるものは除く
    return slides, ev[use], rank[use].astype(np.float32), row[use], xy[use]


def transform(models, rank, name, cfg):
    """slide内の順位（1536次元）を、保存したモデルで各表現に写す。"""
    K = models["pre"].transform(rank).astype(np.float32)
    if name == "unsup":
        return K[:, :cfg["pca_components"]]
    Lw = models["lda"].transform(K).astype(np.float32)
    if name == "weak":
        return Lw
    Q = models["Q"]
    R = models["rpca"].transform(K - (K @ Q) @ Q.T).astype(np.float32)
    return np.c_[Lw / models["scale"][0], cfg["combo_resid_weight"] * R / models["scale"][1]].astype(np.float32)


def draw_slide_maps(cfg, slides, models, run_dir, project_root, logger):
    """選んだslideの全patchを、slide内の順位 → 表現 → 最も近いプロトタイプ、で色分けする。
    左: 元画像、中: プロトタイプの色、右: Bayerのポリゴン（アノテーションがあるslideのみ）。"""
    import colorsys
    import openslide
    from PIL import Image, ImageDraw, ImageFont
    from scipy.stats import norm
    name = cfg["map_representation"]
    cen = torch.from_numpy(np.load(run_dir / f"centroids_{name}.npy"))
    tab = pd.read_csv(run_dir / f"clusters_{name}.csv").set_index("cluster")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    # 色: Bayerのラベルに対応するもの（純度0.5以上、実質以外）は濃く、実質寄りは淡く
    struct = {k for k, r in tab.iterrows() if isinstance(r.top_label, str) and r.top_label != "parenchyma" and r.top_label_purity >= .5}
    pal = {}
    for i, k in enumerate(tab.index):
        h = (i * .61803398875) % 1
        pal[k] = tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, .85, .9)) if k in struct else tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h, .35, .85))

    def label_of(k):
        r = tab.loc[k]
        lab = f"{r.top_label} {r.top_label_purity:.0%}" if isinstance(r.top_label, str) else "-"
        return f"proto {k}: {lab} | Ctrl {r.ctrl_share:.0%} | {r.n_exp} exp"
    ann_dir = project_root / "data/by-tggates-annotations/annotations/geojson"
    out = run_dir / "figures" / f"slide_maps_{name}"
    out.mkdir(parents=True, exist_ok=True)
    from shapely.geometry import shape
    for sid in cfg["map_slides"]:
        r = slides[slides.slide_id == str(sid)]
        if r.empty:
            logger.info(f"map: {sid} not in the dataset, skipped")
            continue
        r = r.iloc[0]
        with h5py.File(r.h5, "r") as h:
            f = h["features"][:].astype(np.float32)
            xy = h["coords"][:]
            size0 = int(h["coords"].attrs.get("patch_size_level0", 256))
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        srt = np.sort(f, axis=0)
        rk = np.empty_like(f)
        for j in range(f.shape[1]):
            rk[:, j] = np.searchsorted(srt[:, j], f[:, j])
        rk = norm.ppf((rk + 0.5) / len(f)).astype(np.float32)
        Z = torch.from_numpy(transform(models, rk, name, cfg))
        cl = torch.cdist(Z, cen).argmin(1).numpy()
        w = openslide.OpenSlide(r.svs_path)
        th = w.get_thumbnail((cfg["map_thumb"], cfg["map_thumb"])).convert("RGB")
        sc = th.width / w.dimensions[0]
        w.close()
        over = Image.new("RGBA", th.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        b = max(2, round(size0 * sc))
        for (x, y), k in zip(xy, cl):
            d.rectangle([x * sc, y * sc, x * sc + b, y * sc + b], fill=(*pal[k], 230))
        mapped = Image.alpha_composite(th.convert("L").convert("RGBA"), over).convert("RGB")
        panels = [th, mapped]
        gj = ann_dir / f"{sid}.svs.geojson"
        if gj.exists():
            ann = th.copy()
            da = ImageDraw.Draw(ann)
            for ft in json.loads(gj.read_text())["features"]:
                lab = ft["properties"]["classification"]["name"]
                if lab == "Parenchyma":
                    continue
                g = shape(ft["geometry"])
                col = (220, 0, 0) if "ecrosis" in lab or "poptosis" in lab else (0, 90, 220) if lab in ("Central vein", "Portal tract", "Large vessels") else (230, 0, 200) if lab == "Mitosis" else (0, 150, 60)
                for part in getattr(g, "geoms", [g]):
                    pts = [(px * sc, py * sc) for px, py in part.exterior.coords]
                    if len(pts) >= 3:
                        da.line(pts + [pts[0]], fill=col, width=2)
            panels.append(ann)
        cnt = pd.Series(cl).value_counts(normalize=True)
        shown = [k for k in cnt.index if cnt[k] >= .01]
        W = sum(p.width for p in panels) + 10 * (len(panels) - 1)
        canvas = Image.new("RGB", (W, th.height + 40 + 18 * (len(shown) + 2)), "white")
        x0 = 0
        for p in panels:
            canvas.paste(p, (x0, 30))
            x0 += p.width + 10
        cd = ImageDraw.Draw(canvas)
        cd.text((6, 6), f"{sid}  e{r.exp} {r.compound}  {r.dose}  {r.period}  | prototypes ({name}, k={len(tab)}): vivid = matches a Bayer label (purity>=50%), "
                        f"pale = parenchyma-like" + ("  | right: Bayer polygons (red necrosis/apoptosis, blue vessels/portal, magenta mitosis, green other)" if gj.exists() else ""),
                fill="black", font=font)
        y = th.height + 36
        for i, k in enumerate(shown):
            yy = y + 18 * i
            cd.rectangle([6, yy + 2, 20, yy + 14], fill=pal[k])
            cd.text((26, yy), f"{cnt[k]:.0%}  {label_of(k)}", fill="black", font=font)
        canvas.save(out / f"{sid}_{str(r.compound).replace(' ', '_')[:20]}_{r.dose}.jpg", quality=88)
        logger.info(f"map {sid} done")


def grid_neighbors(xy, size0):
    """同じslideの8近傍（patchの格子上で隣り合うもの）。Returns: 隣接の組 (m, 2)"""
    g = np.round(xy / size0).astype(np.int64)
    key = {(a, b): i for i, (a, b) in enumerate(g)}
    pairs = []
    for i, (a, b) in enumerate(g):
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                if (da or db) and (a + da, b + db) in key:
                    pairs.append((i, key[(a + da, b + db)]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def contextualize(Z, pairs, hops, alpha):
    """学習なしの空間グラフ上の伝播: z <- (1-alpha) z + alpha * 隣接の平均 を hops 回（ProtoGTXの空間的な文脈化の、学習なしの近似）。"""
    n = len(Z)
    deg = np.bincount(pairs[:, 0], minlength=n).astype(np.float32)
    out = Z.copy()
    for _ in range(hops):
        agg = np.zeros_like(out)
        np.add.at(agg, pairs[:, 0], out[pairs[:, 1]])
        nb = agg / np.maximum(deg, 1)[:, None]
        out = np.where(deg[:, None] > 0, (1 - alpha) * out + alpha * nb, out)
    return out.astype(np.float32)


def run_context(cfg, args, project_root, run_dir, logger, device):
    """0021のプロトタイプ（固定の参照バンク）に、隣接patchの情報を足した表現で割り当て直し、Bayerのラベルとの対応と空間的なまとまりを比べる。
    アノテーションのある全slideの全patchを使う。"""
    import pickle
    from sklearn.metrics import normalized_mutual_info_score
    src = project_root / "outputs" / os.environ["EXP_NAME"] / args.source
    models = pickle.loads((src / "models.pkl").read_bytes())
    name = cfg["map_representation"]
    cen = torch.from_numpy(np.load(src / f"centroids_{name}.npy")).to(device)
    ex20 = load_module(project_root / "experiments" / cfg["exp0020"] / "experiment.py", "exp0020")
    cfg20 = {**ex20.load_config(project_root / "experiments" / cfg["exp0020"]), "random_per_slide": 10 ** 9}   # 全patch
    prio = cfg20["label_priority"]
    ann = project_root / cfg20["annotation_dir"]
    fdir = project_root / cfg20["features_dir"]
    ids = [s_["filename"].split(".")[0] for s_ in json.loads((ann / "slides.json").read_text())["slides"]]
    ids = [i for i in ids if (fdir / f"{i}.h5").exists()]
    settings = [(h, a) for h in cfg["context_hops"] for a in (cfg["context_alphas"] if h else [0.0])]
    labs, assigns, coh = [], {st: [] for st in settings}, {st: [0, 0] for st in settings}
    keep_maps = {}
    t0 = time.time()
    for k, sid in enumerate(ids):
        r = ex20._load_slide((sid, fdir / f"{sid}.h5", ann / "annotations/geojson" / f"{sid}.svs.geojson", cfg20))
        with h5py.File(fdir / f"{sid}.h5", "r") as h:
            size0 = int(h["coords"].attrs.get("patch_size_level0", 256))
        Z = transform(models, r["rank"], name, cfg)
        pairs = grid_neighbors(r["xy"], size0)
        labs.append(r["label"])
        for st in settings:
            Zc = contextualize(Z, pairs, *st) if st[0] else Z
            cl = torch.cdist(torch.from_numpy(Zc).to(device), cen).argmin(1).cpu().numpy()
            assigns[st].append(cl)
            if len(pairs):
                coh[st][0] += int((cl[pairs[:, 0]] == cl[pairs[:, 1]]).sum())
                coh[st][1] += len(pairs)
            if sid in {str(x) for x in cfg["map_slides"]}:
                keep_maps.setdefault(sid, {"xy": r["xy"], "size0": size0})[st] = cl
        if (k + 1) % 20 == 0:
            logger.info(f"  context: {k + 1}/{len(ids)} slides; {time.time() - t0:.0f}s")
    lab = np.concatenate(labs)
    m = lab >= 0
    rows = []
    tabs = {}
    for st in settings:
        cl = np.concatenate(assigns[st])
        rec = {"hops": st[0], "alpha": st[1], "nmi_label": normalized_mutual_info_score(lab[m], cl[m]),
               "neighbor_same_prototype": coh[st][0] / max(coh[st][1], 1)}
        ct = pd.crosstab(lab[m], cl[m])
        for c in ct.index:
            top = ct.loc[c].idxmax()
            rec[f"conc_{prio[c]}"] = ct.loc[c, top] / ct.loc[c].sum()
            rec[f"purity_{prio[c]}"] = ct.loc[c, top] / ct[top].sum()
        rows.append(rec)
        tabs[st] = ct
        logger.info(f"context hops={st[0]} alpha={st[1]}: NMI {rec['nmi_label']:.3f}, neighbor same prototype {rec['neighbor_same_prototype']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "context_metrics.csv", index=False)
    # 図: 指標
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    labels_show = [prio[c] for c in sorted(np.unique(lab[m])) if (lab == c).sum() >= 200]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    xl = [f"h{h}/a{a:g}" for h, a in settings]
    axes[0].plot(xl, res.nmi_label, "o-", label="NMI with Bayer labels")
    axes[0].plot(xl, res.neighbor_same_prototype, "s-", label="adjacent patches in the same prototype")
    axes[0].legend(frameon=False)
    axes[0].set_title("overall")
    for ax, kind in [(axes[1], "conc"), (axes[2], "purity")]:
        for lb in labels_show:
            ax.plot(xl, res[f"{kind}_{lb}"], "o-", label=lb)
        ax.set_title({"conc": "share of the label's patches in its top prototype", "purity": "purity of the label in that prototype"}[kind])
        ax.legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"C1. Adding neighbourhood context (graph smoothing on 8-neighbours) before assigning to the fixed {name} prototypes; "
                 f"{len(ids)} annotated slides, {m.sum():,} labeled patches", fontsize=11)
    fig.savefig(out / "C1_context_metrics.png")
    plt.close(fig)
    # 地図: 文脈なしと、いくつかの設定を並べる
    draw_context_maps(cfg, keep_maps, settings, src, run_dir, project_root, logger)
    (run_dir / "results.json").write_text(json.dumps({"metrics": res.to_dict(orient="records")}, indent=2, ensure_ascii=False, default=float))


def run_context_build(cfg, args, project_root, run_dir, logger, device):
    """隣接patchの情報を足した表現で、プロトタイプを作り直す（自分の表現と周りの平均を連結）。
    アノテーションのある全slideの全patch。k-meansは無作為の一部で学習し、全patchに割り当てる。"""
    import pickle
    from sklearn.metrics import normalized_mutual_info_score
    from lib.clustering import kmeans_lloyd
    src = project_root / "outputs" / os.environ["EXP_NAME"] / args.source
    models = pickle.loads((src / "models.pkl").read_bytes())
    name = cfg["map_representation"]
    ex17 = load_module(project_root / "experiments" / cfg["exp0017"] / "experiment.py", "exp0017")
    ex20 = load_module(project_root / "experiments" / cfg["exp0020"] / "experiment.py", "exp0020")
    cfg20 = {**ex20.load_config(project_root / "experiments" / cfg["exp0020"]), "random_per_slide": 10 ** 9}
    prio = cfg20["label_priority"]
    ann = project_root / cfg20["annotation_dir"]
    fdir = project_root / cfg20["features_dir"]
    slides = pd.read_csv(src / "slides.csv", dtype={"slide_id": str})
    slides = slides[slides.source == "annotated"].reset_index(drop=True)
    hb = cfg["cb"]
    feats = {v: [] for v in hb["variants"]}
    labs, rows, xys, pairs_all, size0, zs = [], [], [], [], {}, []
    t0 = time.time()
    off = 0
    for i, sid in enumerate(slides.slide_id):
        r = ex20._load_slide((sid, fdir / f"{sid}.h5", ann / "annotations/geojson" / f"{sid}.svs.geojson", cfg20))
        with h5py.File(fdir / f"{sid}.h5", "r") as h:
            size0[sid] = int(h["coords"].attrs.get("patch_size_level0", 256))
        Z = transform(models, r["rank"], name, cfg)
        zs.append(Z)
        pr = grid_neighbors(r["xy"], size0[sid])
        ctx = {h_: contextualize(Z, pr, h_, hb["alpha"]) for h_ in {hh for v in hb["variants"].values() for hh in v["hops"]}}
        for v, spec in hb["variants"].items():
            feats[v].append(np.concatenate([Z * spec["self_weight"]] + [ctx[h_] * spec["ctx_weight"] for h_ in spec["hops"]], axis=1).astype(np.float32))
        labs.append(r["label"])
        rows.append(np.full(len(Z), i))
        xys.append(r["xy"])
        pairs_all.append(pr + off)
        off += len(Z)
        if (i + 1) % 20 == 0:
            logger.info(f"  context_build: {i + 1}/{len(slides)} slides; {time.time() - t0:.0f}s")
    lab, row, xy, pairs = np.concatenate(labs), np.concatenate(rows), np.concatenate(xys), np.concatenate(pairs_all)
    np.savez(run_dir / "context_cache.npz", Z=np.concatenate(zs), lab=lab, row=row, xy=xy, pairs=pairs,
             size0=np.array([size0[x] for x in slides.slide_id]))   # context_explore で読み直す（読み込み13分を省く）
    lab = np.where(np.isin(lab, [c for c in np.unique(lab) if c >= 0 and (lab == c).sum() >= cfg20["min_label_patches"]]), lab, -1)
    is_ctrl = (slides.dose.to_numpy() == "Control")[row]
    exp = slides.exp.astype(str).to_numpy()[row]
    logger.info(f"{len(slides)} slides, {len(lab):,} patches ({(lab >= 0).sum():,} labeled); {time.time() - t0:.0f}s")
    rng = np.random.default_rng(cfg["seed"])
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    summary, assigns = [], {}
    m = lab >= 0
    for v, X in feats.items():
        X = np.concatenate(X)
        fit = rng.choice(len(X), size=min(hb["fit_points"], len(X)), replace=False)
        best = None
        for sd in cfg["kmeans_seeds"]:
            _, c, _ = kmeans_lloyd(np.ascontiguousarray(X[fit]), cfg["k"], sd, device)
            cl_f = torch.cdist(torch.from_numpy(X[fit]).to(device), torch.from_numpy(c).to(device)).argmin(1).cpu().numpy()
            inert = float(((X[fit] - c[cl_f]) ** 2).sum())
            if best is None or inert < best[0]:
                best = (inert, c)
        cen = torch.from_numpy(best[1]).to(device)
        cl = np.concatenate([torch.cdist(torch.from_numpy(x).to(device), cen).argmin(1).cpu().numpy() for x in np.array_split(X, 50)])
        dist = np.linalg.norm(X - best[1][cl], axis=1)
        assigns[v] = cl
        rec = {"variant": v, "dim": X.shape[1], "nmi_label": normalized_mutual_info_score(lab[m], cl[m]),
               "neighbor_same_prototype": float((cl[pairs[:, 0]] == cl[pairs[:, 1]]).mean())}
        ct = pd.crosstab(lab[m], cl[m])
        for c in ct.index:
            top = ct.loc[c].idxmax()
            rec[f"conc_{prio[c]}"], rec[f"purity_{prio[c]}"] = ct.loc[c, top] / ct.loc[c].sum(), ct.loc[c, top] / ct[top].sum()
        summary.append(rec)
        logger.info(f"{v}: NMI {rec['nmi_label']:.3f}, neighbour same {rec['neighbor_same_prototype']:.3f}; "
                    + ", ".join(f"{prio[c]} {rec[f'conc_{prio[c]}']:.2f}/{rec[f'purity_{prio[c]}']:.2f}" for c in ct.index))
        # クラスタ表と画像シート
        recs = []
        for k in range(cfg["k"]):
            mk = cl == k
            e = pd.Series(exp[mk]).value_counts(normalize=True)
            lm = mk & m
            rr = {"cluster": k, "n": int(mk.sum()), "n_slides": int(len(np.unique(row[mk]))), "n_exp": int(len(e)),
                  "top_exp_compound": slides.compound[slides.exp.astype(str) == e.index[0]].iloc[0] if len(e) else "", "top_exp_share": float(e.iloc[0]) if len(e) else np.nan,
                  "ctrl_share": float(is_ctrl[mk].mean()) if mk.any() else np.nan, "n_labeled": int(lm.sum())}
            if lm.sum():
                lc = pd.Series(lab[lm]).value_counts(normalize=True)
                rr["top_label"], rr["top_label_purity"] = prio[lc.index[0]], float(lc.iloc[0])
            recs.append(rr)
        tab = pd.DataFrame(recs)
        tab.to_csv(run_dir / f"clusters_{v}.csv", index=False)
        order = tab.sort_values(["top_label", "n"], ascending=[True, False]).cluster.to_numpy()
        sheet_rows = []
        for k in order:
            mem = np.flatnonzero(cl == k)
            mem = mem[np.argsort(dist[mem])]
            picks, seen = [], set()
            for p_ in mem:
                if row[p_] not in seen:
                    picks.append(p_)
                    seen.add(row[p_])
                if len(picks) == cfg["sheet_tiles"]:
                    break
            tiles = ex17.read_tiles(picks, row, xy, slides, size0,
                                    lambda s_, p_: (f"e{s_.exp} {str(s_.compound)[:13]} {str(s_.dose)[0]}", f"{s_.period} {prio[lab[p_]] if lab[p_] >= 0 else ''}"))
            r_ = tab.set_index("cluster").loc[k]
            head = (f"proto {k}: {r_.n} p / {r_.n_slides} sl / {r_.n_exp} exp | top {str(r_.top_exp_compound)[:16]} {r_.top_exp_share:.0%} | Ctrl {r_.ctrl_share:.0%} | "
                    f"labeled {r_.n_labeled}" + (f": {r_.top_label} {r_.top_label_purity:.0%}" if isinstance(r_.get("top_label"), str) else ""))
            sheet_rows.append((head, tiles))
        for f0 in range(0, len(sheet_rows), 10):
            ex17.draw_sheet(f"0021 {v} prototypes (k={cfg['k']}, with neighbourhood context): tiles nearest the centroid, one per slide",
                            sheet_rows[f0:f0 + 10], out / "sheets" / v / f"prototypes_{f0 // 10 + 1:02d}.jpg")
        # 地図
        draw_cluster_maps(cfg, slides, cl, row, xy, size0, tab, v, out / f"maps_{v}", project_root, logger)
    pd.DataFrame(summary).to_csv(run_dir / "context_build_metrics.csv", index=False)
    (run_dir / "results.json").write_text(json.dumps({"metrics": summary}, indent=2, ensure_ascii=False, default=float))


def lobular_position(slides, row, xy, size0, project_root, max_dist):
    """各patchの小葉内の位置 = d_cv / (d_cv + d_pt)（0: 中心静脈側、1: 門脈域側）。Bayerの中心静脈・門脈域ポリゴンからの距離（level 0 px）。
    ポリゴンは網羅的ではないので、両方が max_dist 以内にあるpatchだけ（それ以外はNaN）。"""
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree
    pos = np.full(len(row), np.nan)
    for si, sid in enumerate(slides.slide_id):
        gj = project_root / "data/by-tggates-annotations/annotations/geojson" / f"{sid}.svs.geojson"
        feats = json.loads(gj.read_text())["features"]
        cv = [shape(f["geometry"]) for f in feats if f["properties"]["classification"]["name"] == "Central vein"]
        pt = [shape(f["geometry"]) for f in feats if f["properties"]["classification"]["name"] == "Portal tract"]
        if not cv or not pt:
            continue
        mk = np.flatnonzero(row == si)
        pts = [Point(x + size0[sid] / 2, y + size0[sid] / 2) for x, y in xy[mk]]
        tc, tp = STRtree(cv), STRtree(pt)
        dc = np.array([cv[tc.nearest(q)].distance(q) for q in pts])
        dp = np.array([pt[tp.nearest(q)].distance(q) for q in pts])
        ok = (dc < max_dist) & (dp < max_dist) & ((dc + dp) > 0)
        pos[mk[ok]] = dc[ok] / (dc[ok] + dp[ok])
    return pos


def run_lobular_probe(cfg, args, project_root, run_dir, logger, device):
    """表現（自分・周り）から小葉内の位置を線形に当てられるか（slideごとに分けた交差検証、リッジ回帰）。
    当てられなければ、プロトタイプに小葉構造が出ないのは表現にその情報がないため。"""
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    src = project_root / "outputs" / os.environ["EXP_NAME"] / args.source
    c = np.load(src / "context_cache.npz")
    Z, row, xy, pairs = c["Z"], c["row"], c["xy"], c["pairs"]
    slides = pd.read_csv(project_root / "outputs" / os.environ["EXP_NAME"] / cfg["ce"]["slides_from"] / "slides.csv", dtype={"slide_id": str})
    slides = slides[slides.source == "annotated"].reset_index(drop=True)
    size0 = dict(zip(slides.slide_id, c["size0"].tolist()))
    pos = lobular_position(slides, row, xy, size0, project_root, cfg["ce"]["lobular_max_dist"])
    pm = np.flatnonzero(np.isfinite(pos))
    ctx = {h: contextualize(Z, pairs, h, cfg["ce"]["alpha"])[pm] for h in (2, 4, 8, 16)}
    inputs = {"self": Z[pm], "ctx2": ctx[2], "ctx4": ctx[4], "ctx8": ctx[8], "ctx16": ctx[16], "self+ctx4+ctx16": np.c_[Z[pm], ctx[4], ctx[16]]}
    y, g = pos[pm], row[pm]
    rows = []
    for name, X in inputs.items():
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            pred[te] = RidgeCV(alphas=np.logspace(-2, 3, 8)).fit(X[tr], y[tr]).predict(X[te])
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({"input": name, "r2_slide_out": float(r2), "spearman": float(spearmanr(y, pred)[0]), "n": int(len(y)), "n_slides": int(len(np.unique(g)))})
        logger.info(f"lobular probe {name}: R2 {r2:.3f}, Spearman {rows[-1]['spearman']:.3f}")
    pd.DataFrame(rows).to_csv(run_dir / "lobular_probe.csv", index=False)
    # 全patchで小葉内の位置を予測して地図に描く（slideごとに分けた学習: そのslideを学習から外したモデルで予測）
    import openslide
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib.cm as mcm
    full = {h: contextualize(Z, pairs, h, cfg["ce"]["alpha"]) for h in (4, 16)}
    Xall = np.c_[Z, full[4], full[16]].astype(np.float32)
    Xtr = inputs["self+ctx4+ctx16"]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    out = run_dir / "figures" / "lobular_maps"
    out.mkdir(parents=True, exist_ok=True)
    cmap = mcm.get_cmap("coolwarm")
    for sid in [str(x) for x in cfg["ce"]["map_slides"]] + [str(x) for x in cfg["ce"].get("lobular_extra_maps", [])]:
        hit = np.flatnonzero(slides.slide_id.to_numpy() == sid)
        if not len(hit):
            continue
        si = hit[0]
        model = RidgeCV(alphas=np.logspace(-2, 3, 8)).fit(Xtr[g != si], y[g != si])
        mk = row == si
        pr = model.predict(Xall[mk])
        r = slides.iloc[si]
        w = openslide.OpenSlide(r.svs_path)
        th = w.get_thumbnail((cfg["map_thumb"], cfg["map_thumb"])).convert("RGB")
        sc = th.width / w.dimensions[0]
        w.close()
        over = Image.new("RGBA", th.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        b = max(2, round(size0[sid] * sc))
        lo, hi = np.percentile(pr, [5, 95])
        for (x0, y0), v in zip(xy[mk], pr):
            c_ = cmap(float(np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)))
            d.rectangle([x0 * sc, y0 * sc, x0 * sc + b, y0 * sc + b], fill=tuple(int(255 * q) for q in c_[:3]) + (220,))
        mapped = Image.alpha_composite(th.convert("L").convert("RGBA"), over).convert("RGB")
        canvas = Image.new("RGB", (th.width * 2 + 10, th.height + 34), "white")
        canvas.paste(th, (0, 30))
        canvas.paste(mapped, (th.width + 10, 30))
        ImageDraw.Draw(canvas).text((6, 6), f"{sid} e{r.exp} {r.compound} {r.dose} {r.period} | predicted lobular position (blue = central-vein side, red = portal side; "
                                            f"slide held out of training; colour range = 5-95th percentile of this slide)", fill="black", font=font)
        canvas.save(out / f"{sid}_{str(r.compound).replace(' ', '_')[:20]}_{r.dose}.jpg", quality=88)
        # 拡大: Bayerの中心静脈・門脈域が多い窓で、予測と実際の血管の位置を重ねる
        from shapely.geometry import shape
        gj = project_root / "data/by-tggates-annotations/annotations/geojson" / f"{sid}.svs.geojson"
        polys = [(f_["properties"]["classification"]["name"], shape(f_["geometry"])) for f_ in json.loads(gj.read_text())["features"]
                 if f_["properties"]["classification"]["name"] in ("Central vein", "Portal tract")]
        if polys:
            cxy = np.array([[g_.centroid.x, g_.centroid.y] for _, g_ in polys])
            win = cfg["ce"]["lobular_zoom_px"]
            dens = [((np.abs(cxy - q) < win / 2).all(1)).sum() for q in cxy]
            cx, cy = cxy[int(np.argmax(dens))]
            x0w, y0w = int(cx - win / 2), int(cy - win / 2)
            w = openslide.OpenSlide(r.svs_path)
            lv = w.get_best_level_for_downsample(win / 1600)
            ds = w.level_downsamples[lv]
            img = w.read_region((max(x0w, 0), max(y0w, 0)), lv, (int(win / ds), int(win / ds))).convert("RGB")
            w.close()
            ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
            d2 = ImageDraw.Draw(ov)
            s0 = size0[sid]
            inwin = (xy[mk, 0] >= x0w - s0) & (xy[mk, 0] < x0w + win) & (xy[mk, 1] >= y0w - s0) & (xy[mk, 1] < y0w + win)
            for (px, py), v in zip(xy[mk][inwin], pr[inwin]):
                c_ = cmap(float(np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)))
                d2.rectangle([(px - max(x0w, 0)) / ds, (py - max(y0w, 0)) / ds, (px - max(x0w, 0) + s0) / ds, (py - max(y0w, 0) + s0) / ds],
                             fill=tuple(int(255 * q) for q in c_[:3]) + (110,))
            for lab_, g_ in polys:
                for part in getattr(g_, "geoms", [g_]):
                    pts = [((qx - max(x0w, 0)) / ds, (qy - max(y0w, 0)) / ds) for qx, qy in part.exterior.coords]
                    if len(pts) >= 3:
                        d2.line(pts + [pts[0]], fill=(0, 0, 160, 255) if lab_ == "Central vein" else (160, 0, 0, 255), width=4)
            zoom = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
            cz = Image.new("RGB", (img.width * 2 + 10, img.height + 34), "white")
            cz.paste(img, (0, 30))
            cz.paste(zoom, (img.width + 10, 30))
            ImageDraw.Draw(cz).text((6, 6), f"{sid} zoom ({win * float(r.get('mpp', 0.5) if hasattr(r, 'get') else 0.5):.0f} um approx): predicted position (blue CV side, red portal side) "
                                            f"with Bayer outlines (dark blue = central vein, dark red = portal tract)", fill="black", font=font)
            cz.save(out / f"zoom_{sid}_{str(r.compound).replace(' ', '_')[:20]}.jpg", quality=88)
        logger.info(f"lobular map {sid} done")
    (run_dir / "results.json").write_text(json.dumps({"lobular_probe": rows}, indent=2, ensure_ascii=False, default=float))


def run_context_explore(cfg, args, project_root, run_dir, logger, device):
    """context_buildのキャッシュから、周りの情報の効かせ方をいろいろ試す（重みの掃引と、粗い領域→細かいプロトタイプの2段階）。"""
    from sklearn.metrics import normalized_mutual_info_score
    from lib.clustering import kmeans_lloyd
    src = project_root / "outputs" / os.environ["EXP_NAME"] / args.source
    c = np.load(src / "context_cache.npz")
    Z, lab, row, xy, pairs = c["Z"], c["lab"], c["row"], c["xy"], c["pairs"]
    src0 = project_root / "outputs" / os.environ["EXP_NAME"] / cfg["ce"]["slides_from"]
    slides = pd.read_csv(src0 / "slides.csv", dtype={"slide_id": str})
    slides = slides[slides.source == "annotated"].reset_index(drop=True)
    size0 = dict(zip(slides.slide_id, c["size0"].tolist()))
    ex20 = load_module(project_root / "experiments" / cfg["exp0020"] / "experiment.py", "exp0020")
    cfg20 = ex20.load_config(project_root / "experiments" / cfg["exp0020"])
    prio = cfg20["label_priority"]
    lab = np.where(np.isin(lab, [k for k in np.unique(lab) if k >= 0 and (lab == k).sum() >= cfg20["min_label_patches"]]), lab, -1)
    m = lab >= 0
    is_ctrl = (slides.dose.to_numpy() == "Control")[row]
    exp = slides.exp.astype(str).to_numpy()[row]
    hp = cfg["ce"]
    rng = np.random.default_rng(cfg["seed"])
    t0 = time.time()
    pos = lobular_position(slides, row, xy, size0, project_root, hp["lobular_max_dist"])
    pm = np.isfinite(pos)
    pbin = np.digitize(pos, [1 / 3, 2 / 3])
    logger.info(f"lobular position: {pm.sum():,} patches near both an annotated central vein and portal tract; {time.time() - t0:.0f}s")
    ctx = {}
    # 伝播はslideをまたがない（pairsは同じslide内だけ）ので、全体で一度に計算できる
    for h_ in sorted({h for v in hp["variants"].values() for h in v.get("hops", []) + v.get("coarse_hops", [])}):
        ctx[h_] = contextualize(Z, pairs, h_, hp["alpha"])
        logger.info(f"context hops {h_} done")

    def kmeans_assign(X, k, seeds):
        fit = rng.choice(len(X), size=min(hp["fit_points"], len(X)), replace=False)
        best = None
        for sd in seeds:
            _, cc, _ = kmeans_lloyd(np.ascontiguousarray(X[fit]), k, sd, device)
            cl_f = torch.cdist(torch.from_numpy(X[fit]).to(device), torch.from_numpy(cc).to(device)).argmin(1).cpu().numpy()
            inert = float(((X[fit] - cc[cl_f]) ** 2).sum())
            if best is None or inert < best[0]:
                best = (inert, cc)
        cen = torch.from_numpy(best[1]).to(device)
        return np.concatenate([torch.cdist(torch.from_numpy(x).to(device), cen).argmin(1).cpu().numpy() for x in np.array_split(X, 50)])

    lob = None
    if any(spec["kind"] == "lobular" for spec in hp["variants"].values()):
        # 小葉の軸: 自分＋周り4回＋周り16回から、Bayerの中心静脈・門脈域で決めた位置をリッジ回帰（全アノテーションslideで学習）
        from sklearn.linear_model import RidgeCV
        for h_ in (4, 16):
            if h_ not in ctx:
                ctx[h_] = contextualize(Z, pairs, h_, hp["alpha"])
        Xl = np.c_[Z, ctx[4], ctx[16]].astype(np.float32)
        lob = RidgeCV(alphas=np.logspace(-2, 3, 8)).fit(Xl[pm], pos[pm]).predict(Xl).astype(np.float32)
        lob = ((lob - lob.mean()) / lob.std()).astype(np.float32)
        np.save(run_dir / "lobular_axis.npy", lob)
        logger.info("lobular axis fitted")
    summary = []
    out = run_dir / "figures"
    for v, spec in hp["variants"].items():
        t0 = time.time()
        if spec["kind"] == "lobular":
            parts = [Z * spec["self_weight"]] + [ctx[h] * spec["ctx_weight"] for h in spec["hops"]]
            scale = np.sqrt((Z ** 2).sum(1).mean())   # 自分の表現の大きさに合わせる
            X = np.concatenate(parts + [(lob * scale * spec["lobular_weight"])[:, None]], axis=1).astype(np.float32)
            cl = kmeans_assign(X, cfg["k"], cfg["kmeans_seeds"])
        elif spec["kind"] == "concat":
            X = np.concatenate([Z * spec["self_weight"]] + [ctx[h] * spec["ctx_weight"] for h in spec["hops"]], axis=1).astype(np.float32)
            cl = kmeans_assign(X, cfg["k"], cfg["kmeans_seeds"])
        else:   # 2段階: 周りの情報だけで粗い領域 → 領域ごとに自分の表現で細かく
            Xc = np.concatenate([ctx[h] for h in spec["coarse_hops"]], axis=1).astype(np.float32)
            region = kmeans_assign(Xc, spec["coarse_k"], cfg["kmeans_seeds"])
            cl = np.full(len(Z), -1)
            nxt = 0
            sizes = np.bincount(region, minlength=spec["coarse_k"])
            for g in range(spec["coarse_k"]):
                mg = np.flatnonzero(region == g)
                kg = max(1, int(round(spec["fine_total"] * sizes[g] / sizes.sum())))
                kg = min(kg, max(1, len(mg) // 2000))
                sub = kmeans_assign(Z[mg], kg, cfg["kmeans_seeds"][:2]) if kg > 1 else np.zeros(len(mg), dtype=int)
                cl[mg] = sub + nxt
                nxt += kg
            np.save(run_dir / f"region_{v}.npy", region)
            rtab = pd.DataFrame([{"cluster": g, "n": int((region == g).sum()), "n_exp": int(len(np.unique(exp[region == g]))),
                                  "ctrl_share": float(is_ctrl[region == g].mean()), "mean_lobular_pos": float(np.nanmean(pos[region == g])),
                                  **({"top_label": prio[pd.Series(lab[(region == g) & m]).value_counts().index[0]],
                                      "top_label_purity": float(pd.Series(lab[(region == g) & m]).value_counts(normalize=True).iloc[0])}
                                     if ((region == g) & m).any() else {"top_label": np.nan, "top_label_purity": np.nan})} for g in range(spec["coarse_k"])])
            rtab.to_csv(run_dir / f"regions_{v}.csv", index=False)
            draw_cluster_maps({**cfg, "map_slides": hp["map_slides"]}, slides, region, row, xy, size0, rtab, v + "_regions", out / f"maps_{v}_regions", project_root, logger)
        k_ = int(cl.max()) + 1
        rec = {"variant": v, "n_prototypes": k_, "nmi_label": normalized_mutual_info_score(lab[m], cl[m]),
               "nmi_lobular_pos": normalized_mutual_info_score(pbin[pm], cl[pm]),
               "neighbor_same_prototype": float((cl[pairs[:, 0]] == cl[pairs[:, 1]]).mean())}
        ct = pd.crosstab(lab[m], cl[m])
        for cc in ct.index:
            top = ct.loc[cc].idxmax()
            rec[f"conc_{prio[cc]}"], rec[f"purity_{prio[cc]}"] = ct.loc[cc, top] / ct.loc[cc].sum(), ct.loc[cc, top] / ct[top].sum()
        summary.append(rec)
        logger.info(f"{v}: {k_} prototypes, NMI {rec['nmi_label']:.3f}, NMI lobular {rec['nmi_lobular_pos']:.3f}, neighbour same {rec['neighbor_same_prototype']:.3f}; "
                    + ", ".join(f"{prio[cc]} {rec[f'conc_{prio[cc]}']:.2f}/{rec[f'purity_{prio[cc]}']:.2f}" for cc in ct.index) + f"; {time.time() - t0:.0f}s")
        recs = []
        for k in range(k_):
            mk = cl == k
            lm = mk & m
            rr = {"cluster": k, "n": int(mk.sum()), "n_exp": int(len(np.unique(exp[mk]))), "ctrl_share": float(is_ctrl[mk].mean()) if mk.any() else np.nan,
                  "mean_lobular_pos": float(np.nanmean(pos[mk])) if (mk & pm).sum() >= 50 else np.nan}
            if lm.sum():
                lc = pd.Series(lab[lm]).value_counts(normalize=True)
                rr["top_label"], rr["top_label_purity"] = prio[lc.index[0]], float(lc.iloc[0])
            else:
                rr["top_label"], rr["top_label_purity"] = np.nan, np.nan
            recs.append(rr)
        tab = pd.DataFrame(recs)
        tab.to_csv(run_dir / f"clusters_{v}.csv", index=False)
        np.save(run_dir / f"assign_{v}.npy", cl)
        draw_cluster_maps({**cfg, "map_slides": hp["map_slides"]}, slides, cl, row, xy, size0, tab, v, out / f"maps_{v}", project_root, logger)
    res = pd.DataFrame(summary)
    res.to_csv(run_dir / "context_explore_metrics.csv", index=False)
    (run_dir / "results.json").write_text(json.dumps({"metrics": summary}, indent=2, ensure_ascii=False, default=float))


def draw_cluster_maps(cfg, slides, cl, row, xy, size0, tab, v, out, project_root, logger):
    """全patchの割り当て（cl）を、選んだslideに描く（元画像 / プロトタイプ / Bayerのポリゴン）。"""
    import colorsys
    import openslide
    from PIL import Image, ImageDraw, ImageFont
    from shapely.geometry import shape
    out.mkdir(parents=True, exist_ok=True)
    tab = tab.set_index("cluster")
    struct = {k for k, r in tab.iterrows() if isinstance(r.top_label, str) and r.top_label != "parenchyma" and r.top_label_purity >= .5}
    pal = {k: tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * .61803398875) % 1, .85 if k in struct else .35, .9 if k in struct else .85))
           for i, k in enumerate(tab.index)}
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    for sid in [str(x) for x in cfg["map_slides"]]:
        hit = np.flatnonzero(slides.slide_id.to_numpy() == sid)
        if not len(hit):
            continue
        si = hit[0]
        r = slides.iloc[si]
        mk = row == si
        w = openslide.OpenSlide(r.svs_path)
        th = w.get_thumbnail((cfg["map_thumb"], cfg["map_thumb"])).convert("RGB")
        sc = th.width / w.dimensions[0]
        w.close()
        over = Image.new("RGBA", th.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        b = max(2, round(size0[sid] * sc))
        for (x, y), k in zip(xy[mk], cl[mk]):
            d.rectangle([x * sc, y * sc, x * sc + b, y * sc + b], fill=(*pal[k], 230))
        mapped = Image.alpha_composite(th.convert("L").convert("RGBA"), over).convert("RGB")
        annimg = th.copy()
        da = ImageDraw.Draw(annimg)
        gj = project_root / "data/by-tggates-annotations/annotations/geojson" / f"{sid}.svs.geojson"
        for ft in json.loads(gj.read_text())["features"]:
            lab_ = ft["properties"]["classification"]["name"]
            if lab_ == "Parenchyma":
                continue
            g = shape(ft["geometry"])
            col = (220, 0, 0) if "ecrosis" in lab_ or "poptosis" in lab_ else (0, 90, 220) if lab_ in ("Central vein", "Portal tract", "Large vessels") else (230, 0, 200) if lab_ == "Mitosis" else (0, 150, 60)
            for part in getattr(g, "geoms", [g]):
                pts = [(px * sc, py * sc) for px, py in part.exterior.coords]
                if len(pts) >= 3:
                    da.line(pts + [pts[0]], fill=col, width=2)
        cnt = pd.Series(cl[mk]).value_counts(normalize=True)
        shown = [k for k in cnt.index if cnt[k] >= .01]
        canvas = Image.new("RGB", (th.width * 3 + 20, th.height + 40 + 18 * (len(shown) + 1)), "white")
        for j, pnl in enumerate([th, mapped, annimg]):
            canvas.paste(pnl, (j * (th.width + 10), 30))
        cd = ImageDraw.Draw(canvas)
        cd.text((6, 6), f"{sid}  e{r.exp} {r.compound}  {r.dose}  {r.period} | {v} prototypes (vivid = Bayer-label prototypes) | right: Bayer polygons "
                        f"(red necrosis/apoptosis, blue vessels/portal, magenta mitosis, green other)", fill="black", font=font)
        for i, k in enumerate(shown):
            yy = th.height + 36 + 18 * i
            rr = tab.loc[k]
            cd.rectangle([6, yy + 2, 20, yy + 14], fill=pal[k])
            lb = f"{rr.top_label} {rr.top_label_purity:.0%}" if isinstance(rr.top_label, str) else "-"
            cd.text((26, yy), f"{cnt[k]:.0%}  proto {k}: {lb} | Ctrl {rr.ctrl_share:.0%} | {rr.n_exp} exp", fill="black", font=font)
        canvas.save(out / f"{sid}_{str(r.compound).replace(' ', '_')[:20]}_{r.dose}.jpg", quality=88)


def draw_context_maps(cfg, keep_maps, settings, src, run_dir, project_root, logger):
    import colorsys
    import openslide
    from PIL import Image, ImageDraw, ImageFont
    tab = pd.read_csv(src / f"clusters_{cfg['map_representation']}.csv").set_index("cluster")
    struct = {k for k, r in tab.iterrows() if isinstance(r.top_label, str) and r.top_label != "parenchyma" and r.top_label_purity >= .5}
    pal = {k: tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * .61803398875) % 1, .85 if k in struct else .35, .9 if k in struct else .85))
           for i, k in enumerate(tab.index)}
    slides = pd.read_csv(src / "slides.csv", dtype={"slide_id": str})
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    show = [st for st in settings if st in cfg["context_map_settings_tuples"]]
    out = run_dir / "figures" / "context_maps"
    out.mkdir(parents=True, exist_ok=True)
    for sid, dd in keep_maps.items():
        r = slides[slides.slide_id == sid].iloc[0]
        w = openslide.OpenSlide(r.svs_path)
        th = w.get_thumbnail((cfg["context_map_thumb"], cfg["context_map_thumb"])).convert("RGB")
        sc = th.width / w.dimensions[0]
        w.close()
        panels = [("original", th)]
        b = max(2, round(dd["size0"] * sc))
        for st in show:
            over = Image.new("RGBA", th.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(over)
            for (x, y), k in zip(dd["xy"], dd[st]):
                d.rectangle([x * sc, y * sc, x * sc + b, y * sc + b], fill=(*pal[k], 230))
            panels.append((f"hops {st[0]}, alpha {st[1]:g}" if st[0] else "no context", Image.alpha_composite(th.convert("L").convert("RGBA"), over).convert("RGB")))
        W = sum(p.width for _, p in panels) + 10 * (len(panels) - 1)
        canvas = Image.new("RGB", (W, th.height + 50), "white")
        cd = ImageDraw.Draw(canvas)
        x0 = 0
        for title, p in panels:
            canvas.paste(p, (x0, 46))
            cd.text((x0 + 4, 26), title, fill="black", font=font)
            x0 += p.width + 10
        cd.text((6, 4), f"{sid}  e{r.exp} {r.compound}  {r.dose}  {r.period}  | same prototype colours as v3_maps (vivid = Bayer-label prototypes, pale = parenchyma-like)",
                fill="black", font=font)
        canvas.save(out / f"{sid}_{str(r.compound).replace(' ', '_')[:20]}_{r.dose}.jpg", quality=88)
        logger.info(f"context map {sid} done")


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.clustering import kmeans_lloyd
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    args = parse_args()
    exp_name = os.environ["EXP_NAME"]
    cfg = load_config(Path(__file__).parent)
    run_dir = get_run_dir(project_root, __file__, args.variant, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=args.variant, config=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting: {exp_name} / {args.variant}")
    if args.mode == "lobular_probe":
        run_lobular_probe(cfg, args, project_root, run_dir, logger, device)
        complete_run(run_dir)
        logger.info("Done.")
        return
    if args.mode == "context_explore":
        run_context_explore(cfg, args, project_root, run_dir, logger, device)
        complete_run(run_dir)
        logger.info("Done.")
        return
    if args.mode == "context_build":
        run_context_build(cfg, args, project_root, run_dir, logger, device)
        complete_run(run_dir)
        logger.info("Done.")
        return
    if args.mode == "context":
        cfg["context_map_settings_tuples"] = [tuple(x) for x in cfg["context_map_settings"]]
        run_context(cfg, args, project_root, run_dir, logger, device)
        complete_run(run_dir)
        logger.info("Done.")
        return

    # --- データ ---
    t0 = time.time()
    ex20 = load_module(project_root / "experiments" / cfg["exp0020"] / "experiment.py", "exp0020")
    cfg20 = ex20.load_config(project_root / "experiments" / cfg["exp0020"])
    d20, meta20 = ex20.build_dataset(cfg20, project_root, logger)
    prio = cfg20["label_priority"]
    counts = pd.Series(d20["label"][d20["label"] >= 0]).value_counts()
    keep_lab = [c for c in counts.index if counts[c] >= cfg20["min_label_patches"]]
    lab20 = np.where(np.isin(d20["label"], keep_lab), d20["label"], -1)
    ex17 = load_module(project_root / "experiments" / cfg["exp0017"] / "experiment.py", "exp0017")
    cfg17 = ex17.load_config(project_root / "experiments" / cfg["exp0017"])
    cfg17.update(cfg17["variants"][cfg["tgg_variant"]])
    sl17, raw17, rank17, row17, xy17 = tgg_patches(ex17, cfg17, cfg["tgg_variant"], project_root, set(meta20.slide_id))
    logger.info(f"annotated: {len(meta20)} slides, {len(lab20):,} patches; {cfg['tgg_variant']}: {len(np.unique(row17))} slides, {len(row17):,} patches; {time.time() - t0:.0f}s")

    # 1つのslide表にまとめる（svsの場所、試験、用量、所見）
    q = pd.read_csv(cfg20["qc_catalog"], dtype={"slide_id": str, "EXP_ID": str})
    q = q.set_index("slide_id")

    def svs(sid):
        for dd in WSI_DIRS:
            p = Path(dd) / f"{sid}.svs"
            if p.exists():
                return str(p)
        return None
    a = meta20[["slide_id"]].assign(exp=meta20.EXP_ID.astype(str), compound=meta20.COMPOUND_NAME, dose=meta20.DOSE_LEVEL,
                                    period=meta20.SACRIFICE_PERIOD, svs_path=meta20.slide_id.map(svs), source="annotated",
                                    h5=[str(project_root / cfg20["features_dir"] / f"{s}.h5") for s in meta20.slide_id], has_finding=np.nan)
    used17 = np.unique(row17)
    b = sl17.iloc[used17]
    fdirs = [project_root / x for x in cfg17["features_dirs"]]

    def h5of(sid):
        for f in fdirs:
            if (f / f"{sid}.h5").exists():
                return str(f / f"{sid}.h5")
    b = pd.DataFrame({"slide_id": b.slide_id.to_numpy(), "exp": b.exp_id.astype(str).str.zfill(4).to_numpy(), "compound": b.compound_name.to_numpy(),
                      "dose": b.dose_level.to_numpy(), "period": b.sacrifice_period.to_numpy(), "svs_path": b.svs_path.to_numpy(),
                      "source": cfg["tgg_variant"], "h5": [h5of(s) for s in b.slide_id], "has_finding": b.has_finding.astype(float).to_numpy()})
    slides = pd.concat([a, b], ignore_index=True)
    remap17 = {old: len(a) + i for i, old in enumerate(used17)}
    row = np.r_[d20["slide"], np.array([remap17[r] for r in row17])]
    xy = np.r_[d20["xy"], xy17]
    raw = np.r_[d20["X"], raw17]
    rank = np.r_[d20["rank"], rank17]
    lab = np.r_[lab20, np.full(len(row17), -1)]
    is_ctrl = (slides.dose.to_numpy() == "Control")[row]
    exp = slides.exp.to_numpy()[row]
    size0 = {}
    for r in slides.itertuples():
        with h5py.File(r.h5, "r") as h:
            size0[r.slide_id] = int(h["coords"].attrs.get("patch_size_level0", 256))
    slides.to_csv(run_dir / "slides.csv", index=False)

    # --- 表現 ---
    pre = PCA(n_components=cfg["pre_pca"], svd_solver="randomized", random_state=cfg["seed"]).fit(rank)
    K = pre.transform(rank).astype(np.float32)
    reps = {}
    if "unsup" in cfg["representations"]:
        reps["unsup"] = K[:, :cfg["pca_components"]]
    if "weak" in cfg["representations"]:
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(K[lab >= 0], lab[lab >= 0])
        reps["weak"] = lda.transform(K).astype(np.float32)
    if "combo" in cfg["representations"]:
        # weakのLDA方向（ラベルで学んだ）と、その方向を除いた残りのPCA（ラベルにない変化）を、分散をそろえて並べる
        Wl = lda.scalings_[:, :lda.transform(K[:2]).shape[1]]
        Q, _ = np.linalg.qr(Wl)
        resid = K - (K @ Q) @ Q.T
        rpca = PCA(n_components=cfg["combo_resid_dim"], random_state=cfg["seed"]).fit(resid)
        R = rpca.transform(resid).astype(np.float32)
        Lw = reps["weak"]
        scale = (Lw.std(0).mean(), R.std(0).mean())
        reps["combo"] = np.c_[Lw / scale[0], cfg["combo_resid_weight"] * R / scale[1]].astype(np.float32)
    models = {"pre": pre, "lda": lda if "weak" in cfg["representations"] else None,
              "Q": Q if "combo" in cfg["representations"] else None, "rpca": rpca if "combo" in cfg["representations"] else None,
              "scale": scale if "combo" in cfg["representations"] else None}
    import pickle
    (run_dir / "models.pkl").write_bytes(pickle.dumps(models))
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    summary = []
    rng = np.random.default_rng(cfg["seed"])
    for name, Z in reps.items():
        t0 = time.time()
        best = None
        for s in cfg["kmeans_seeds"]:
            cl, c, _ = kmeans_lloyd(np.ascontiguousarray(Z), cfg["k"], s, device)
            inert = float(((Z - c[cl]) ** 2).sum())
            if best is None or inert < best[0]:
                best = (inert, cl, c)
        _, cl, cen = best
        dist = np.linalg.norm(Z - cen[cl], axis=1)
        # クラスタごとのまとめ
        recs = []
        for k in range(cfg["k"]):
            m = cl == k
            e = pd.Series(exp[m]).value_counts(normalize=True)
            lm = m & (lab >= 0)
            rec = {"cluster": k, "n": int(m.sum()), "n_slides": int(len(np.unique(row[m]))), "n_exp": int(len(e)), "top_exp": e.index[0],
                   "top_exp_compound": slides.compound[slides.exp == e.index[0]].iloc[0], "top_exp_share": float(e.iloc[0]),
                   "ctrl_share": float(is_ctrl[m].mean()), "annotated_share": float((slides.source.to_numpy()[row[m]] == "annotated").mean()),
                   "n_labeled": int(lm.sum())}
            if lm.sum():
                lc = pd.Series(lab[lm]).value_counts(normalize=True)
                rec["top_label"], rec["top_label_purity"] = prio[lc.index[0]], float(lc.iloc[0])
            recs.append(rec)
        tab = pd.DataFrame(recs)
        # ラベル × クラスタ: そのラベルのpatchがどのクラスタに入ったか（集中度）
        lc = pd.crosstab(lab[lab >= 0], cl[lab >= 0]).reindex(columns=range(cfg["k"]), fill_value=0)
        lc.index = [prio[i] for i in lc.index]
        conc = lc.div(lc.sum(1), axis=0)
        pur = lc.div(lc.sum(0).replace(0, np.nan), axis=1)
        tab.to_csv(run_dir / f"clusters_{name}.csv", index=False)
        conc.to_csv(run_dir / f"label_concentration_{name}.csv")
        from sklearn.metrics import normalized_mutual_info_score
        summary.append({"representation": name, "dim": Z.shape[1], "nmi_label": normalized_mutual_info_score(lab[lab >= 0], cl[lab >= 0]),
                        "median_n_exp": float(tab.n_exp.median()), "median_top_exp_share": float(tab.top_exp_share.median()),
                        "clusters_ctrl_lt5": int((tab.ctrl_share < .05).sum())})
        logger.info(f"{name}: {summary[-1]}; {time.time() - t0:.0f}s")

        # 図: ラベル × クラスタ（集中度）と、クラスタの性質
        order = tab.sort_values(["top_label", "n"], ascending=[True, False]).cluster.to_numpy() if "top_label" in tab else np.arange(cfg["k"])
        fig, axes = plt.subplots(2, 1, figsize=(0.42 * cfg["k"] + 4, 9), gridspec_kw={"height_ratios": [1.3, 1]}, sharex=True)
        ax = axes[0]
        v = conc[order].to_numpy()
        ax.imshow(v, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        for (i, j), x in np.ndenumerate(v):
            if x >= .1:
                p_ = pur[order].to_numpy()[i, j]
                ax.text(j, i, f"{x:.0%}"[:-1] + f"\n{p_:.0%}"[:-1], ha="center", va="center", fontsize=5.5, color="white" if x > .6 else "black")
        ax.set_yticks(range(len(conc)), [f"{c} ({lc.loc[c].sum()})" for c in conc.index], fontsize=8)
        ax.set_title(f"{name}: share of each Bayer label's patches in each cluster (top number, %); purity of that label in the cluster (bottom, %)", fontsize=10)
        ax = axes[1]
        t = tab.set_index("cluster").loc[order]
        x = np.arange(len(order))
        ax.bar(x - .2, t.top_exp_share, width=.4, color="#eb6834", label="top experiment share")
        ax.bar(x + .2, t.ctrl_share, width=.4, color="#9aa3ad", label="Control share")
        for i, r in enumerate(t.itertuples()):
            ax.text(i, max(r.top_exp_share, r.ctrl_share) + .02, f"{r.n_exp}", ha="center", fontsize=6.5)
        ax.set_xticks(x, [str(c) for c in order], fontsize=7.5)
        ax.set_xlabel("cluster (number above bars = experiments)")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(out / f"P1_labels_and_clusters_{name}.png")
        plt.close(fig)

        # プロトタイプの画像シート（重心に近い順、別々のslide）
        rows_ = []
        for k in order:
            members = np.flatnonzero(cl == k)
            members = members[np.argsort(dist[members])]
            picks, seen = [], set()
            for p in members:
                if row[p] not in seen:
                    picks.append(p)
                    seen.add(row[p])
                if len(picks) == cfg["sheet_tiles"]:
                    break
            tiles = ex17.read_tiles(picks, row, xy, slides, size0,
                                    lambda s_, p: (f"e{s_.exp} {str(s_.compound)[:13]} {str(s_.dose)[0]}",
                                                   f"{s_.period} {prio[lab[p]] if lab[p] >= 0 else ''}"))
            r = tab.set_index("cluster").loc[k]
            head = (f"cluster {k}: {r.n} p / {r.n_slides} sl / {r.n_exp} exp | top exp {str(r.top_exp_compound)[:16]} {r.top_exp_share:.0%} | "
                    f"Ctrl {r.ctrl_share:.0%} | labeled {r.n_labeled}" + (f": {r.top_label} {r.top_label_purity:.0%}" if isinstance(r.get("top_label"), str) else ""))
            rows_.append((head, tiles))
        for f0 in range(0, len(rows_), 10):
            ex17.draw_sheet(f"0021 {name} prototypes (k={cfg['k']}): tiles nearest the centroid, one per slide; label line 2 = Bayer label if annotated",
                            rows_[f0:f0 + 10], out / "sheets" / name / f"prototypes_{f0 // 10 + 1:02d}.jpg")
        np.save(run_dir / f"centroids_{name}.npy", cen)
        np.save(run_dir / f"assign_{name}.npy", cl)
        logger.info(f"{name}: sheets done; {time.time() - t0:.0f}s")

    # t-SNE（両方の表現をクラスタとラベルで）
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits
    li = np.flatnonzero(lab >= 0)
    sub = np.r_[rng.choice(li, size=min(8000, len(li)), replace=False),
                rng.choice(np.flatnonzero(lab < 0), size=cfg["tsne_points"] - min(8000, len(li)), replace=False)]
    import colorsys
    fig, axes = plt.subplots(len(reps), 3, figsize=(27, 9 * len(reps)), squeeze=False)
    for i, (name, Z) in enumerate(reps.items()):
        with threadpool_limits(limits=8):
            p2 = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=0, n_jobs=8).fit_transform(Z[sub])
        cl = np.load(run_dir / f"assign_{name}.npy")[sub]
        pal = np.asarray([colorsys.hsv_to_rgb((j * .61803398875) % 1, .7, .85) for j in range(cfg["k"])])
        axes[i, 0].scatter(*p2.T, c=pal[cl], s=1.2, linewidths=0, rasterized=True)
        for j in range(cfg["k"]):
            if (cl == j).any():
                axes[i, 0].text(*np.median(p2[cl == j], axis=0), str(j), fontsize=7, ha="center", va="center",
                                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=.7))
        axes[i, 0].set_title(f"{name}: clusters", fontsize=11)
        ls = lab[sub]
        axes[i, 1].scatter(*p2[ls < 0].T, color="#dde1e5", s=1, linewidths=0, rasterized=True)
        for j, c in enumerate(sorted(keep_lab)):
            mm = ls == c
            axes[i, 1].scatter(*p2[mm].T, color=colorsys.hsv_to_rgb((j * .61803398875) % 1, .8, .85), s=2, linewidths=0, rasterized=True, label=prio[c])
        axes[i, 1].legend(markerscale=6, frameon=False, fontsize=7)
        axes[i, 1].set_title(f"{name}: Bayer labels", fontsize=11)
        ex_ = exp[sub]
        uex = np.unique(ex_)
        epal = {e: colorsys.hsv_to_rgb((j * .61803398875) % 1, .6, .8) for j, e in enumerate(uex)}
        axes[i, 2].scatter(*p2.T, c=[epal[e] for e in ex_], s=1, linewidths=0, rasterized=True)
        axes[i, 2].set_title(f"{name}: experiment ({len(uex)} colors)", fontsize=11)
        for a_ in axes[i]:
            a_.set_xticks([])
            a_.set_yticks([])
    fig.savefig(out / "P2_tsne.png")
    plt.close(fig)
    if cfg.get("map_slides"):
        draw_slide_maps(cfg, slides, models, run_dir, project_root, logger)
    pd.DataFrame(summary).to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "results.json").write_text(json.dumps({"summary": summary}, indent=2, ensure_ascii=False, default=float))
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
