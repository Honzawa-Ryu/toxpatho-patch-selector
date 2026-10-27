import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


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
    """Define CLI args for all variable dimensions (used in GRID_ARGS / RUN_COMMAND)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")   # run_slurm.shが渡す。読むのは常に実験ディレクトリのconfig.yml
    parser.add_argument("--variant", default="v1")
    return parser.parse_args()


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 150})
_W = {}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def fit_combo(K, lab_c, cfg, seed):
    """230枚で、LDA（Bayerのラベル）＋LDAを除いた残りのPCAを学び、写す関数を返す（0025のoof_comboの全体版）。"""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    m = lab_c >= 0
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(K[m].astype(np.float32), lab_c[m])
    L = int(lab_c.max())
    Q, _ = np.linalg.qr(lda.scalings_[:, :L])
    sub = np.random.default_rng(seed).choice(len(K), size=min(300000, len(K)), replace=False)
    Ks = K[sub].astype(np.float32)
    rpca = PCA(n_components=cfg["combo_resid_dim"], random_state=seed).fit(Ks - (Ks @ Q) @ Q.T)
    s0 = lda.transform(Ks)[:, :L].std(0).mean()
    s1 = rpca.transform(Ks - (Ks @ Q) @ Q.T).std(0).mean()

    def tf(Kb):
        Kb = Kb.astype(np.float32)
        return np.c_[lda.transform(Kb)[:, :L] / s0, cfg["combo_resid_weight"] * rpca.transform(Kb - (Kb @ Q) @ Q.T) / s1].astype(np.float32)
    return tf


def _init_worker(pre_path):
    _W["pre"] = pickle.loads(Path(pre_path).read_bytes())["pre"]


def _load_target(h5_path):
    with h5py.File(h5_path, "r") as h:
        f = h["features"][:].astype(np.float32)
        xy = h["coords"][:]
        size0 = int(h["coords"].attrs.get("patch_size_level0", 256))
    rk = sys.modules["exp0023"].slide_rank(f)
    return {"K": _W["pre"].transform(rk).astype(np.float16), "xy": xy, "size0": size0,
            "pairs": sys.modules["exp0021"].grid_neighbors(xy, size0)}


def _raw_sample(h5_path, n=300):
    with h5py.File(h5_path, "r") as h:
        f = h["features"][:].astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f[np.random.default_rng(len(f)).choice(len(f), size=min(n, len(f)), replace=False)]


def _init_raw(pca_bytes):
    _W["raw_pca"] = pickle.loads(pca_bytes)


def _raw_transform(h5_path):
    with h5py.File(h5_path, "r") as h:
        f = h["features"][:].astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return _W["raw_pca"].transform(f).astype(np.float32)


def target_slides(cfg, project_root):
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[q.EXP_ID.notna() & (q.EXP_ID != "")]
    q = q[q.EXP_ID.astype(int).isin(cfg["target_exps"])].copy()
    q["h5"] = [str(project_root / cfg["features_dir"] / f"{s}.h5") for s in q.slide_id]
    q = q[[Path(x).exists() for x in q.h5]].reset_index(drop=True)
    q["exp"] = q.EXP_ID.astype(int)
    q["svs_path"] = [str(project_root / "data/TGGATEs/WSI/Liver" / str(p).replace(" ", "_") / f"{s}.svs") for s, p in zip(q.slide_id, q.SACRIFICE_PERIOD)]
    return q.rename(columns={"COMPOUND_NAME": "compound", "DOSE_LEVEL": "dose", "SACRIFICE_PERIOD": "period"})


def fat_grades(cfg, project_root, slides):
    p = pd.read_csv(project_root / cfg["pathology_csv"], dtype=str)
    p = p[(p.ORGAN == "Liver") & (p.SP_FLG.str.lower() != "true") & p.FINDING_TYPE.isin(cfg["fat_findings"])]
    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
    sk = slides[["slide_id"] + key].copy()
    for c in key:
        sk[c] = pd.to_numeric(sk[c], errors="coerce")
        p[c] = pd.to_numeric(p[c], errors="coerce")
    t = sk.merge(p, on=key)
    t["g"] = t.GRADE_TYPE.map(cfg["grade_map"]).fillna(1)
    G = t.pivot_table(index="slide_id", columns="FINDING_TYPE", values="g", aggfunc="max").reindex(slides.slide_id).fillna(0)
    for f in cfg["fat_findings"]:
        if f not in G:
            G[f] = 0.0
    topo = t.groupby(["slide_id", "FINDING_TYPE"]).TOPOGRAPHY_TYPE.first().unstack()
    return G[cfg["fat_findings"]], topo


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()

    # Build variant_key from all variable dimensions so each config gets its own directory.
    # Include every arg that changes the result (model, seed, prompt, ...).
    # Example:
    #   model_short = args.model.replace("/", "-")
    #   variant_key = f"{model_short}__{args.seed}"
    variant_key = args.variant

    # Initialize output directory (exits immediately if already completed).
    # Written on OUTPUT_ROOT (scratch) when set; scripts/slurm_entry.sh
    # rsyncs it back to project_root/outputs/ at job end.
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    from concurrent.futures import ProcessPoolExecutor
    from scipy.stats import spearmanr
    from sklearn.metrics import homogeneity_score
    from lib.clustering import kmeans_lloyd
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = config
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex17 = load_module(project_root / "experiments/0017_20261016_block_consensus_patterns/experiment.py", "exp0017")
    ex25 = load_module(project_root / "experiments" / cfg["exp0025"] / "experiment.py", "exp0025")
    cfg20 = yaml.safe_load(open(project_root / "experiments" / cfg["exp0020"] / "config.yml"))
    prio = cfg20["label_priority"]

    # 1. 230枚（Bayer）と脂肪変性の試験のslide
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Ka, laba, pairs_a = c["K"], c["lab"].astype(np.int64), c["pairs"]
    keep = [k for k in np.unique(laba) if k >= 0 and (laba == k).sum() >= cfg20["min_label_patches"]]
    laba = np.where(np.isin(laba, keep), laba, -1)
    lab_c = np.where(laba >= 0, np.searchsorted(keep, np.maximum(laba, 0)), -1)
    annot_ids = pd.read_csv(src / "slides.csv", dtype=str).slide_id.tolist()
    slides = target_slides(cfg, project_root)
    G, topo = fat_grades(cfg, project_root, slides)
    slides.drop(columns=["h5"]).to_csv(run_dir / "slides.csv", index=False)
    G.to_csv(run_dir / "slide_fat_grades.csv")
    logger.info(f"target: {len(slides)} slides " + json.dumps({str(k): int(v) for k, v in slides.groupby("exp").size().items()}) +
                f"; fatty degeneration > 0: {int((G['Degeneration, fatty'] > 0).sum())}")
    parts = []
    pre_path = project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl"
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init_worker, initargs=(str(pre_path),)) as ex:
        for i, r in enumerate(ex.map(_load_target, slides.h5)):
            parts.append(r)
    off = np.cumsum([0] + [len(p_["K"]) for p_ in parts])
    Kt = np.concatenate([p_["K"] for p_ in parts])
    row = np.concatenate([np.full(len(p_["K"]), i) for i, p_ in enumerate(parts)])
    xy = np.concatenate([p_["xy"] for p_ in parts])
    size0 = {sid: p_["size0"] for sid, p_ in zip(slides.slide_id, parts)}
    pairs_t = np.concatenate([p_["pairs"] + off[i] for i, p_ in enumerate(parts)])
    na = len(Ka)
    logger.info(f"annotated {na:,} + target {len(Kt):,} patches; {time.time() - t0:.0f}s")

    # 2. 表現（2026-10-27ユーザー: LDAでラベルのない変化がつぶされている可能性。表現を並べて比べる）
    #  combo           : 0025の土台（slide内の順位 → PCA256 → LDA＋残りのPCA16、重み0.5）
    #  combo_bilateral : combo＋境界を保つ伝播（0025の最良）
    #  combo_strong    : 残りのPCAを32次元・重み1.0（ラベルのない変化を強く残す）
    #  unsup           : slide内の順位 → PCA の上位64次元（LDAなし）
    #  raw             : 以前（0003）の方式。L2 → PCA64、slide内の順位もLDAもなし
    reps_ab = {}
    need = set(cfg["representations"])
    if need & {"combo", "combo_bilateral"} or any(r_.startswith("unsup_lda") for r_ in need):
        tf = fit_combo(Ka, lab_c, cfg, cfg["seed"])
        reps_ab["combo"] = np.concatenate([tf(b_) for b_ in np.array_split(np.concatenate([Ka, Kt]), 40)])
    if "combo_strong" in need:
        tf2 = fit_combo(Ka, lab_c, {**cfg, "combo_resid_dim": 32, "combo_resid_weight": 1.0}, cfg["seed"])
        reps_ab["combo_strong"] = np.concatenate([tf2(b_) for b_ in np.array_split(np.concatenate([Ka, Kt]), 40)])
    if "unsup" in need or any(r_.startswith("unsup_lda") for r_ in need):
        reps_ab["unsup"] = np.concatenate([Ka[:, :64], Kt[:, :64]]).astype(np.float32)
    del Ka, Kt
    if "raw" in need:
        from sklearn.decomposition import PCA
        h5s = [str(project_root / cfg["features_dir"] / f"{s_}.h5") for s_ in annot_ids] + slides.h5.tolist()
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
            samp = np.concatenate(list(ex.map(_raw_sample, h5s)))
        pca = PCA(n_components=64, random_state=cfg["seed"]).fit(samp)
        with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init_raw, initargs=(pickle.dumps(pca),)) as ex:
            reps_ab["raw"] = np.concatenate(list(ex.map(_raw_transform, h5s))).astype(np.float32)
        assert len(reps_ab["raw"]) == na + len(row), "raw: patch count mismatch with cache"
    logger.info("representations: " + ", ".join(f"{k_} {v.shape[1]}d" for k_, v in reps_ab.items()) + f"; {time.time() - t0:.0f}s")
    Gr = ex25.Graph(np.concatenate([pairs_a, pairs_t + na]), na + len(row), device)
    cx = cfg["context"]

    L_lda = len(keep) - 1

    def get_X(rep):
        if rep.startswith("unsup_lda"):
            # 2026-10-27ユーザー: LDAでつぶした表現だけでなく、元の表現も残る塩梅を探す。unsup（PCA64）にLDAの方向をβ倍で並べる
            beta = float(rep[len("unsup_lda"):])
            U = reps_ab["unsup"]
            Lw = reps_ab["combo"][:, :L_lda]
            Lw = Lw * (U.std(0).mean() / Lw.std(0).mean())
            return torch.from_numpy(np.concatenate([U, beta * Lw], axis=1).astype(np.float32)).to(device)
        if rep == "combo_bilateral":
            Z = torch.from_numpy(reps_ab["combo"]).to(device)
            w = Gr.edge_weights(Z, cx["sigma_q"])
            return torch.cat([Z, cx["w"] * Gr.propagate(Z, cx["hops"], cx["alpha"], w)], 1)
        return torch.from_numpy(reps_ab[rep]).to(device)
    Zt = np.zeros(len(row))   # 対象のpatch数だけ使う
    rng = np.random.default_rng(cfg["seed"])
    is_ctrl = (slides.dose == "Control").to_numpy()
    fat = G["Degeneration, fatty"].to_numpy()
    cmpd = np.where(slides.exp.isin([66, 67]), "CCl4", "cholesterol")
    out = run_dir / "figures"
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    (out / "maps").mkdir(exist_ok=True)
    summary = {}
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    summ_rows = []

    def consensus(X, k, group):
        """コンセンサスk-means（0025 v4）。学習点（230枚と脂肪変性の試験から層別）とseedを変えてn回、重心を集めてk個にまとめ直す。
        groupでseedの組を変える（2組のARIでseed依存性を測る）。"""
        cents = []
        for m_ in range(cfg["consensus_runs"]):
            rr = np.random.default_rng([cfg["seed"], group, m_])
            fit = np.r_[rr.choice(na, size=min(cfg["fit_annot"], na), replace=False), na + rr.choice(len(Zt), size=min(cfg["fit_target"], len(Zt)), replace=False)]
            _, cc, _ = kmeans_lloyd(X[torch.from_numpy(fit).to(device)].cpu().numpy(), k, 1000 * (group + 1) + m_, device)
            cents.append(cc)
        _, meta, _ = kmeans_lloyd(np.concatenate(cents), k, cfg["seed"] + group, device)
        cen = torch.from_numpy(meta).to(device)
        return torch.cat([torch.cdist(x, cen).argmin(1) for x in X.split(65536)]).cpu().numpy(), meta

    def evaluate(rep, kb, cut, cl, ari):
        k = int(cl.max()) + 1
        tag = f"{rep}_k{kb}" + (f"_cut{cut}" if cut != kb else "")
        cla, clt = cl[:na], cl[na:]
        np.save(run_dir / f"assign_target_{tag}.npy", clt.astype(np.int16))
        hom = homogeneity_score(laba[laba >= 0], cla[laba >= 0])
        # slideごとの割合と、同じ試験のControlとの差
        sh = np.zeros((len(slides), k))
        np.add.at(sh, (row, clt), 1)
        sh /= sh.sum(1, keepdims=True)
        d = np.full_like(sh, np.nan)
        for e, idx in slides.groupby("exp").indices.items():
            cidx = idx[is_ctrl[idx]]
            if len(cidx) >= cfg["min_ctrl"]:
                d[idx] = sh[idx] - sh[cidx].mean(0)
        rows_ = []
        for j in range(k):
            r = {"prototype": j, "n_target": int((clt == j).sum()), "n_annot": int((cla == j).sum()),
                 "bayer_top_label": prio[pd.Series(laba[(cla == j) & (laba >= 0)]).value_counts().index[0]] if ((cla == j) & (laba >= 0)).any() else ""}
            mt = clt == j
            # 純度（slideの記録で見る）: このプロトタイプの対象のpatchのうち、脂肪変性ありのslideから来た割合（化合物ごと）
            for cname in ("CCl4", "cholesterol"):
                cm = cmpd == cname
                pm = mt & cm[row]
                r[f"frac_from_fatty_{cname}"] = float((fat[row[pm]] > 0).mean()) if pm.any() else np.nan
                r[f"n_{cname}"] = int(pm.sum())
                ok = cm & ~np.isnan(d[:, 0])
                r[f"rho_{cname}"] = spearmanr(fat[ok], d[ok, j])[0] if len(np.unique(fat[ok])) > 1 else np.nan
                r[f"delta_pos_{cname}"] = float(np.nanmean(d[ok & (fat > 0), j])) if (ok & (fat > 0)).any() else np.nan
            rows_.append(r)
        tab = pd.DataFrame(rows_)
        tab.to_csv(run_dir / f"prototypes_{tag}.csv", index=False)
        nmi = normalized_mutual_info_score(laba[laba >= 0], cla[laba >= 0])
        summary[tag] = {"homogeneity_bayer": hom, "nmi_bayer": nmi}
        sr = {"rep": rep, "k_kmeans": kb, "n_prototypes": k, "ari_two_consensus": ari, "homogeneity_bayer": hom, "nmi_bayer": nmi}
        for cname in ("CCl4", "cholesterol"):
            big = tab[tab[f"n_{cname}"] >= cfg["min_proto_patches"]]
            b_ = big.sort_values(f"rho_{cname}", ascending=False).iloc[0] if len(big) else None
            sr[f"best_rho_{cname}"] = float(b_[f"rho_{cname}"]) if b_ is not None else np.nan
            sr[f"best_rho_purity_{cname}"] = float(b_[f"frac_from_fatty_{cname}"]) if b_ is not None else np.nan
            sr[f"max_purity_{cname}"] = float(big[f"frac_from_fatty_{cname}"].max()) if len(big) else np.nan
            sr[f"n_protos_purity90_{cname}"] = int((big[f"frac_from_fatty_{cname}"] >= 0.9).sum())
        summ_rows.append(sr)
        pd.DataFrame(summ_rows).to_csv(run_dir / "summary.csv", index=False)
        logger.info(f"{tag}: homogeneity on Bayer labels {hom:.3f}")
        for cname in ("CCl4", "cholesterol") if (kb, cut) in [tuple(x_) for x_ in cfg["sheet_at"]] else ():
            tt = tab.sort_values(f"rho_{cname}", ascending=False).head(cfg["top_prototypes"])
            logger.info(f"  {cname} top: " + "; ".join(f"p{int(x.prototype)} rho {x[f'rho_{cname}']:.2f} purity {x[f'frac_from_fatty_{cname}']:.2f} (Bayer {x.bayer_top_label})" for _, x in tt.iterrows()))
            # 画像: 程度の高いslideから、そのプロトタイプのpatchを1枚ずつ
            sheet = []
            for _, x in tt.iterrows():
                j = int(x.prototype)
                cand_sl = np.flatnonzero((cmpd == cname) & (fat > 0))
                cand_sl = cand_sl[np.argsort(-fat[cand_sl], kind="stable")]
                picks = []
                for si in cand_sl:
                    cand = np.flatnonzero((row == si) & (clt == j))
                    if len(cand):
                        picks.append(int(rng.choice(cand)))
                    if len(picks) == cfg["sheet_tiles"]:
                        break
                if picks:
                    tiles = ex17.read_tiles(picks, row, xy, slides, size0,
                                            lambda s_, p_: (f"e{s_.exp} {str(s_.compound)[:13]} {str(s_.dose)[0]}", f"{s_.period} fatty {int(fat[row[p_]])}"))
                    sheet.append((f"{tag} prototype {j} | {cname}: rho {x[f'rho_{cname}']:.2f}, share from fatty slides {x[f'frac_from_fatty_{cname}']:.2f} | Bayer: {x.bayer_top_label}", tiles))
            for f0 in range(0, len(sheet), 3):
                ex17.draw_sheet(f"0026 {tag} {cname}: prototypes rising with fatty degeneration (highest grade first, one per slide)",
                                sheet[f0:f0 + 3], out / "sheets" / f"{tag}_{cname}_{f0 // 3 + 1:02d}.jpg")
        # 図: 程度ごとの平均割合（上位のプロトタイプ）
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for ax, cname in zip(axes, ("CCl4", "cholesterol")):
            tt = tab.sort_values(f"rho_{cname}", ascending=False).head(cfg["top_prototypes"])
            cm = cmpd == cname
            for _, x in tt.iterrows():
                j = int(x.prototype)
                gs = sorted(np.unique(fat[cm]))
                ax.plot(gs, [np.nanmean(sh[cm & (fat == g_), j]) for g_ in gs], marker="o", label=f"p{j} ({x.bayer_top_label})")
            ax.set_xticks(range(5), ["0", "min", "sli", "mod", "sev"])
            ax.set_xlabel("recorded fatty degeneration grade")
            ax.set_ylabel("share of the slide's patches")
            ax.set_title(f"{cname}: top {cfg['top_prototypes']} prototypes ({tag})")
            ax.legend(frameon=False, fontsize=7)
        fig.savefig(out / f"F1_share_by_grade_{tag}.png")
        plt.close(fig)
        # 地図: 化合物ごとに最も程度の高いslideと、Control
        picks_sl = []
        for cname in ("CCl4", "cholesterol"):
            cm = np.flatnonzero((cmpd == cname) & (fat > 0))
            if len(cm):
                picks_sl.append(int(cm[np.argmax(fat[cm])]))
            cc_ = np.flatnonzero((cmpd == cname) & is_ctrl)
            if len(cc_):
                picks_sl.append(int(cc_[0]))
        if [kb, cut] == cfg["map_at"] and rep in cfg["map_reps"]:
            import colorsys
            import openslide
            from PIL import Image, ImageDraw
            hot = set()
            for cname in ("CCl4", "cholesterol"):
                hot |= set(tab.sort_values(f"rho_{cname}", ascending=False).head(cfg["top_prototypes"]).prototype.astype(int))
            pal = {j: tuple(int(255 * v) for v in colorsys.hsv_to_rgb((i * 0.618) % 1, 0.85, 0.95)) for i, j in enumerate(sorted(hot))}
            for si in picks_sl:
                r_ = slides.iloc[si]
                wsl = openslide.OpenSlide(r_.svs_path)
                th = wsl.get_thumbnail((cfg["map_thumb"], cfg["map_thumb"])).convert("RGB")
                scl = th.width / wsl.dimensions[0]
                b = max(2, round(size0[r_.slide_id] * scl))
                ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
                dr = ImageDraw.Draw(ov)
                for (x0, y0), j in zip(xy[row == si], clt[row == si]):
                    dr.rectangle([x0 * scl, y0 * scl, x0 * scl + b, y0 * scl + b], fill=(pal[j] + (235,)) if j in pal else (200, 200, 200, 90))
                mp = Image.alpha_composite(th.convert("L").convert("RGBA"), ov).convert("RGB")
                canvas = Image.new("RGB", (th.width * 2 + 10, th.height + 30), "white")
                canvas.paste(th, (0, 30))
                canvas.paste(mp, (th.width + 10, 30))
                ImageDraw.Draw(canvas).text((6, 6), f"{r_.slide_id} e{r_.exp} {r_.compound} {r_.dose} {r_.period} | fatty grade {int(fat[si])} | coloured: prototypes rising with fatty degeneration " +
                                            ", ".join(f"p{j}" for j in sorted(hot)), fill="black")
                canvas.save(out / "maps" / f"{tag}_{r_.slide_id}.jpg", quality=88)
        logger.info(f"{tag} done (ARI {ari:.3f}); {time.time() - t0:.0f}s")

    for rep in cfg["representations"]:
        torch.cuda.empty_cache()
        X = get_X(rep)
        for kb in cfg["ks_big"]:
            clA, cenA = consensus(X, kb, 0)
            clB, _ = consensus(X, kb, 1)
            sub = np.random.default_rng(0).choice(len(clA), size=min(300000, len(clA)), replace=False)
            ari = float(adjusted_rand_score(clA[sub], clB[sub]))
            # 重心を階層化（Ward）して、いくつかの数で切る。kを1つに決めず、後から粒度を選べるようにする（2026-10-27ユーザー了承）
            Zl = linkage(cenA.astype(np.float64), method="ward")
            np.save(run_dir / f"linkage_{rep}_k{kb}.npy", Zl)
            np.save(run_dir / f"centroids_{rep}_k{kb}.npy", cenA)
            for cut in [c_ for c_ in cfg["cuts"] if c_ < kb] + [kb]:
                lab_cut = np.arange(kb) if cut == kb else fcluster(Zl, cut, criterion="maxclust") - 1
                evaluate(rep, kb, cut, lab_cut[clA], ari)
        del X

    results: dict = {"n_target_slides": len(slides), "n_target_patches": int(len(Zt)), "summary": summary}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()