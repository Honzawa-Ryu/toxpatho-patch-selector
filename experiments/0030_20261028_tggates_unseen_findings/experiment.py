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

# --- Deep learning (uncomment if needed) ---
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset

# --- Visualization (uncomment if needed) ---
# import matplotlib.pyplot as plt
# import seaborn as sns


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


def _load_sub(h5_path):
    """0027の_load_ctrlと同じ間引き（乱数はpatch数で決まる）で、特徴（PCA256）に加えて座標とpatch_size_level0も返す。"""
    import h5py
    W = sys.modules["exp0027"]._W
    with h5py.File(h5_path, "r") as h:
        f = h["features"][:].astype(np.float32)
        xy = h["coords"][:]
        s0 = int(h["coords"].attrs["patch_size_level0"])
    rk = sys.modules["exp0023"].slide_rank(f)
    idx = np.arange(len(rk))
    if len(rk) > W["n"]:
        idx = np.random.default_rng(len(rk)).choice(len(rk), size=W["n"], replace=False)
    return W["pre"].transform(rk[idx]).astype(np.float32), xy[idx], s0


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
    import torch
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = {**config, **config.get("variants", {}).get(args.variant, {})}
    t0 = time.time()
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    ex23 = load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex26 = load_module(project_root / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(project_root / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    cfg20 = yaml.safe_load(open(project_root / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    run26 = project_root / cfg.get("proto_root", "outputs/" + cfg["exp0026"])   # プロトタイプの置き場。既定は0026

    # 1. 0026 tree_v3と同じLDA（Bayerの230枚、脂肪滴系の21枚のラベルを除く）
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Ka, laba = c["K"], c["lab"].astype(np.int64)
    keep = [k for k in np.unique(laba) if k >= 0 and (laba == k).sum() >= cfg20["min_label_patches"]]
    laba = np.where(np.isin(laba, keep), laba, -1)
    lab_c = np.where(laba >= 0, np.searchsorted(keep, np.maximum(laba, 0)), -1)
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    ex_ids = pd.read_csv(project_root / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab_c = np.where(np.isin(bayer.slide_id.to_numpy()[c["row"]], ex_ids), -1, lab_c)
    tf = ex26.fit_combo(Ka, lab_c, cfg, cfg["seed"])
    L = len(keep) - 1
    del Ka

    # 2. 対象のslide: 対象の所見がある試験の全slide（Bayerの230枚と脂肪変性の3試験は除く）
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[q.EXP_ID.notna() & ~q.slide_id.isin(bayer.slide_id)]
    q = q[~q.EXP_ID.astype(int).isin(cfg["exclude_exps"])].copy()
    def h5_of(sid):
        for d_ in cfg["features_dirs"]:
            if (project_root / d_ / f"{sid}.h5").exists():
                return str(project_root / d_ / f"{sid}.h5")
        return ""
    q["h5"] = [h5_of(x) for x in q.slide_id]
    q = q[q.h5 != ""]
    pth = pd.read_csv(project_root / cfg["pathology_csv"], dtype=str)
    pth = pth[(pth.ORGAN == "Liver") & (pth.SP_FLG.str.lower() != "true") & pth.FINDING_TYPE.isin(list(cfg["findings"]))]
    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
    for k_ in key:
        q[k_] = pd.to_numeric(q[k_], errors="coerce")
        pth[k_] = pd.to_numeric(pth[k_], errors="coerce")
    pth["g"] = pth.GRADE_TYPE.map(cfg["grade_map"]).fillna(1)
    pos = q.merge(pth, on=key)
    exps = sorted(set(pos.EXP_ID))
    sl = q[q.EXP_ID.isin(exps)].reset_index(drop=True)
    G = pos.pivot_table(index="slide_id", columns="FINDING_TYPE", values="g", aggfunc="max").reindex(sl.slide_id).fillna(0)
    G = G.reindex(columns=list(cfg["findings"]), fill_value=0)
    sl["is_ctrl"] = sl.DOSE_LEVEL == "Control"
    sl["bayer_exp"] = sl.EXP_ID.isin(set(bayer.EXP_ID.dropna().astype(int)))
    logger.info(f"slides: {len(sl)} in {len(exps)} experiments ({sl.COMPOUND_NAME.nunique()} compounds), controls {int(sl.is_ctrl.sum())}, "
                f"in Bayer experiments {int(sl.bayer_exp.sum())}; positives per finding: {(G > 0).sum().to_dict()}; {time.time() - t0:.0f}s")

    # 3. 各slideを2,048 patchに間引き、slide内の順位 → PCA256 → U（64）・LDA（L）
    pre_path = str(project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl")
    pre = pickle.loads(Path(pre_path).read_bytes())["pre"]
    sl_run = [run for run, rep, k in cfg["maps"].values() if rep == "sphering_lda"]
    sc = json.loads((run26 / sl_run[0] / "scale_sphering_lda.json").read_text()) if sl_run else None
    resume = bool(cfg.get("resume")) and (run_dir / "assignments.npz").exists()
    feat_from = project_root / cfg["features_from"] if cfg.get("features_from") else run_dir / "features_sub.npz"
    if resume:
        # 途中で落ちた実行の割り当て（row, 座標, 各表現のプロトタイプ番号）を読み、特徴の読み直しを飛ばす（地図の部分の修正の再実行用）
        import h5py
        z = np.load(run_dir / "assignments.npz")
        row, XY = z["row"], z["xy"]
        size0 = {}
        for sid, h5p in zip(sl.slide_id, sl.h5):
            with h5py.File(h5p, "r") as h:
                size0[sid] = int(h["coords"].attrs["patch_size_level0"])
        expn = sl.EXP_ID.to_numpy()[row]
        cm = sl.is_ctrl.to_numpy()[row]
        ex27._init_ctrl(pre_path, cfg["patches_per_slide"])   # 地図の白色化で対照群を読み直すため
        logger.info(f"resume: assignments for {len(row):,} patches loaded; {time.time() - t0:.0f}s")
    else:
        if feat_from.exists():
            # 2026-10-29: 以前の実行で保存した間引き特徴（slide内の順位 → PCA256 の上位64と、LDAの13次元は固定なので再利用できる）
            z = np.load(feat_from, allow_pickle=True)
            U, Lp, row, XY = z["U"].astype(np.float32), z["Lp"].astype(np.float32), z["row"], z["xy"]
            size0 = json.loads(str(z["size0"]))
            assert (z["slide_ids"] == sl.slide_id.to_numpy()).all(), "features_from: slide list differs"
            logger.info(f"features loaded from {feat_from}; {time.time() - t0:.0f}s")
        else:
            with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
                Us, Ls, XYs, size0 = [], [], [], {}
                for sid, (K, xy, s0) in zip(sl.slide_id, ex.map(_load_sub, sl.h5, chunksize=4)):
                    Us.append(K[:, :64])
                    Ls.append(tf(K)[:, :L].astype(np.float32))
                    XYs.append(xy)
                    size0[sid] = s0
            row = np.concatenate([np.full(len(u), i) for i, u in enumerate(Us)])
            U, Lp, XY = np.concatenate(Us), np.concatenate(Ls), np.concatenate(XYs)
            del Us, Ls, XYs
            np.savez_compressed(run_dir / "features_sub.npz", U=U.astype(np.float16), Lp=Lp.astype(np.float16), row=row, xy=XY, size0=json.dumps(size0), slide_ids=sl.slide_id.to_numpy().astype(str))
        expn = sl.EXP_ID.to_numpy()[row]
        cm = sl.is_ctrl.to_numpy()[row]
        e_fix = np.where(np.isin(expn, expn[cm]), expn, -999)   # 対照群のない試験は全試験の対照群でまとめて白色化
        Sw = ex27.sphering(U, e_fix, np.r_[U[cm], U[cm]], np.r_[expn[cm], np.full(int(cm.sum()), -999)])
        rep_X = {"sphering_lda": lambda: np.concatenate([Sw, sc["lda_beta"] * sc["lda_scale"] * Lp], 1), "sphering": lambda: Sw, "lda": lambda: Lp, "unsup": lambda: U, "unsup13": lambda: U[:, :L]}
        logger.info(f"patches: {len(U):,}; {time.time() - t0:.0f}s")

    # 4. プロトタイプに割り当て、slideごとの構成比（同じ試験の対照群の平均を引く）
    names, shares, deltas, cls = {}, {}, {}, {}
    for mname, (run, rep, k) in cfg["maps"].items():
        if resume:
            cl = z[f"cl_{mname}"].astype(np.int64)
        else:
            cen = torch.from_numpy(np.load(run26 / run / f"centroids_{rep}_k{k}.npy").astype(np.float32))
            X = torch.from_numpy(rep_X[rep]().astype(np.float32))
            cl = torch.cat([torch.cdist(x, cen).argmin(1) for x in X.split(65536)]).numpy()
        cls[mname] = cl
        sh = np.zeros((len(sl), k))
        np.add.at(sh, (row, cl), 1)
        sh /= sh.sum(1, keepdims=True)
        ctrl_mean = pd.DataFrame(sh[sl.is_ctrl.to_numpy()]).groupby(sl.EXP_ID.to_numpy()[sl.is_ctrl.to_numpy()]).mean()
        base = ctrl_mean.reindex(sl.EXP_ID).to_numpy()
        base = np.where(np.isnan(base), sh[sl.is_ctrl.to_numpy()].mean(0), base)
        shares[mname], deltas[mname] = sh, sh - base
        names[mname] = pd.read_csv(project_root / "outputs" / cfg["exp0029"] / cfg["names_from"] / f"prototype_names_{mname}.csv").set_index("prototype").name.fillna("").reindex(range(k)).to_numpy()
        np.save(run_dir / f"shares_{mname}.npy", sh)
    sl.drop(columns=["h5"]).assign(**{f"grade__{f}": G[f].to_numpy() for f in G}).to_csv(run_dir / "slides.csv", index=False)
    logger.info(f"assigned; {time.time() - t0:.0f}s")

    # 5. 所見の分類（化合物単位の分割）: (a) TG-GATEsの記録で読み出しを学ぶロジスティック回帰、(b) 学習なし: Bayerのラベルの名前が付いたプロトタイプの割合
    comp = sl.COMPOUND_NAME.to_numpy()
    rows = []
    for f, blab in cfg["findings"].items():
        y = (G[f].to_numpy() > 0).astype(int)
        npos_c = len(set(comp[y == 1]))
        for mname in cfg["maps"]:
            D_ = deltas[mname]
            oof = np.full(len(sl), np.nan)
            for tr, te in GroupKFold(n_splits=min(cfg["cv_splits"], npos_c)).split(D_, y, comp):
                if y[tr].sum() == 0:
                    continue
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
                oof[te] = clf.fit(D_[tr], y[tr]).predict_proba(D_[te])[:, 1]
            ok = ~np.isnan(oof)
            r = {"finding": f, "bayer_label": blab or "(none)", "map": mname, "n_pos": int(y.sum()), "n_pos_compounds": npos_c,
                 "AUC_readout": roc_auc_score(y[ok], oof[ok]) if len(set(y[ok])) > 1 else np.nan}
            for part, pm in [("bayer_exp", sl.bayer_exp.to_numpy()), ("new_exp", ~sl.bayer_exp.to_numpy())]:
                m_ = ok & pm
                r[f"AUC_readout_{part}"] = roc_auc_score(y[m_], oof[m_]) if len(set(y[m_])) > 1 else np.nan
            if blab:
                gs = deltas[mname][:, names[mname] == blab].sum(1)
                r["n_named_protos"] = int((names[mname] == blab).sum())
                r["AUC_named_noTGtraining"] = roc_auc_score(y, gs) if (names[mname] == blab).any() else np.nan
                r["rho_named_vs_grade"] = spearmanr(G[f].to_numpy(), gs)[0] if (names[mname] == blab).any() else np.nan
            # プロトタイプ: 記録ありslideで最も増えるもの（対照群との差の平均）と、その名前
            j = int(np.argmax(D_[y == 1].mean(0)))
            r.update({"top_proto": j, "top_proto_name": names[mname][j], "top_proto_delta_pos": float(D_[y == 1, j].mean()), "top_proto_delta_neg": float(D_[(y == 0) & ~sl.is_ctrl.to_numpy(), j].mean())})
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "finding_classification.csv", index=False)
    pd.set_option("display.width", 250)
    logger.info("\n" + res.pivot_table(index="finding", columns="map", values="AUC_readout").round(3).to_string())
    logger.info("named prototypes, no TG-GATEs training:\n" + res.dropna(subset=["AUC_named_noTGtraining"]).pivot_table(index="finding", columns="map", values="AUC_named_noTGtraining").round(3).to_string())

    # 6. 図
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    order = list(cfg["findings"])
    mk = {"D_unsup": ("o", "#999999"), "C_lda": ("s", "#009E73"), "B_sphering": ("^", "#0072B2"), "A_sphering_lda": ("D", "#D55E00")}
    for ax, col, ttl in [(axes[0], "AUC_readout", "readout trained on TG-GATEs records (compound-held-out CV)"), (axes[1], "AUC_named_noTGtraining", "no TG-GATEs training: prototypes named by Bayer labels")]:
        for mname, (mk_, cc) in mk.items():
            v = res[res["map"] == mname].set_index("finding")[col].reindex(order) if col in res else None
            if v is None:
                continue
            ax.scatter(v.to_numpy(), range(len(order)), marker=mk_, color=cc, s=36, label=mname, zorder=3)
        ax.axvline(0.5, color="0.75", lw=0.8)
        ax.set_xlim(0.3, 1.0)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("AUROC (finding recorded vs not)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(order)), [f"{f} [{cfg['findings'][f] or 'no Bayer label'}]" for f in order], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"TG-GATEs slides unseen in training ({len(sl)} slides, {len(exps)} experiments)", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "F1_finding_auc.png", dpi=150)
    plt.close(fig)

    if not resume:
        np.savez_compressed(run_dir / "assignments.npz", row=row, xy=XY, **{f"cl_{m_}": cls[m_].astype(np.int16) for m_ in cls})

    # 7. 地図と各プロトタイプの見た目（2026-10-29ユーザー）。選んだslideは全patchで割り当て、プロトタイプの見た目は全slideの間引きpatchから
    mt = cfg.get("maps_tiles")
    if mt:
        import glob
        import h5py
        import openslide
        from PIL import Image, ImageDraw, ImageFont
        ex17 = load_module(project_root / "experiments/0017_20261016_block_consensus_patterns/experiment.py", "exp0017")
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        def svs_of(r):
            cand = project_root / "data/TGGATEs/WSI/Liver" / str(r.SACRIFICE_PERIOD).replace(" ", "_") / f"{r.slide_id}.svs"
            if cand.exists():
                return str(cand)
            g = glob.glob(str(project_root / "data/TGGATEs/WSI" / "**" / f"{r.slide_id}.svs"), recursive=True)
            return g[0] if g else ""
        sl["svs_path"] = [svs_of(r) for r in sl.itertuples()]
        logger.info(f"svs found for {int((sl.svs_path != '').sum())} / {len(sl)} slides")
        rgb = {g: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for g, h in mt["colors"].items()}
        # 7a. slideの選び方: 所見ごとに記録ありを最大 n_per_finding 枚（新しい試験を優先、程度の高い順）＋その試験の対照群1枚
        pick = []
        for f, b in cfg["findings"].items():
            if not b:
                continue
            cand = sl[(G[f].to_numpy() > 0) & (sl.svs_path != "")].copy()
            cand["g"] = G[f].to_numpy()[cand.index]
            cand = cand.sort_values(["bayer_exp", "g"], ascending=[True, False])
            seen_c = set()
            for r in cand.itertuples():
                if r.COMPOUND_NAME in seen_c:
                    continue
                pick.append((r.Index, f))
                seen_c.add(r.COMPOUND_NAME)
                if len(seen_c) >= mt["n_per_finding"]:
                    break
        for i, f in list(pick):
            cs = sl[(sl.EXP_ID == sl.EXP_ID[i]) & sl.is_ctrl & (sl.svs_path != "")]
            if len(cs) and not any(j == cs.index[0] for j, _ in pick):
                pick.append((cs.index[0], "control"))
        logger.info(f"maps: {len(pick)} slides")
        mout = run_dir / "maps"
        mout.mkdir(exist_ok=True)
        cen_t = {m_: torch.from_numpy(np.load(run26 / cfg["maps"][m_][0] / f"centroids_{cfg['maps'][m_][1]}_k{cfg['maps'][m_][2]}.npy").astype(np.float32)) for m_ in cfg["maps"]}
        grp = {}
        for m_, (run, rep, k) in cfg["maps"].items():
            t_ = pd.read_csv(run26 / run / "tree" / f"{rep}_k{k}" / "prototypes_in_tree_order.csv").set_index("prototype")
            grp[m_] = t_.group_30.reindex(range(k)).to_numpy()
        import colorsys
        pal = [tuple(int(255 * c_) for c_ in colorsys.hsv_to_rgb((i_ * 0.618) % 1, 0.65 + 0.3 * (i_ % 2), 0.95 - 0.25 * ((i_ // 2) % 2))) for i_ in range(31)]
        pool_cache = {}

        def ctrl_U(e):
            """試験eの対照群のU（64次元）。e=None は全試験の対照群から40枚（再開時は読み直す）"""
            if not resume:
                m_ = cm if e is None else cm & (expn == e)
                return U[m_]
            if e not in pool_cache:
                hs = sl.h5[sl.is_ctrl & (sl.EXP_ID == e)] if e is not None else sl.h5[sl.is_ctrl].sample(min(40, int(sl.is_ctrl.sum())), random_state=cfg["seed"])
                pool_cache[e] = np.concatenate([_load_sub(h)[0][:, :64] for h in hs]) if len(hs) else np.zeros((0, 64), np.float32)
            return pool_cache[e]

        mrows = []
        for i, why in pick:
            r = sl.loc[i]
            with h5py.File(r.h5, "r") as h:
                f_ = h["features"][:].astype(np.float32)
                xy_f = h["coords"][:]
                s0 = int(h["coords"].attrs["patch_size_level0"])
            Kf = pre.transform(ex23.slide_rank(f_)).astype(np.float32)
            Uf, Lf = Kf[:, :64], tf(Kf)[:, :L].astype(np.float32)
            e = r.EXP_ID
            Uc = ctrl_U(e)
            Sf = ex27.sphering(Uf, np.full(len(Uf), e), Uc, np.full(len(Uc), e)) if len(Uc) else ex27.sphering(Uf, np.full(len(Uf), -999), ctrl_U(None), np.full(len(ctrl_U(None)), -999))
            Xf = {"sphering_lda": (lambda: np.concatenate([Sf, sc["lda_beta"] * sc["lda_scale"] * Lf], 1)), "sphering": (lambda: Sf), "lda": (lambda: Lf), "unsup": (lambda: Uf), "unsup13": (lambda: Uf[:, :L])}
            w = openslide.OpenSlide(r.svs_path)
            th = w.get_thumbnail((mt["thumb"], mt["thumb"])).convert("RGB")
            scl = th.width / w.dimensions[0]
            w.close()
            b_ = max(2, round(s0 * scl))
            rec = {"slide_id": r.slide_id, "why": why, "compound": r.COMPOUND_NAME, "exp": e, "dose": r.DOSE_LEVEL, "period": r.SACRIFICE_PERIOD, "bayer_exp": bool(r.bayer_exp),
                   "records": "; ".join(f"{f}={int(G[f].iloc[i])}" for f in G if G[f].iloc[i] > 0), "n_patches": len(Uf)}
            for mode in ("named", "groups"):
                panels = [th]
                for m_, (run, rep, k) in cfg["maps"].items():
                    X_ = torch.from_numpy(Xf[rep]().astype(np.float32))
                    cl_ = torch.cat([torch.cdist(x, cen_t[m_]).argmin(1) for x in X_.split(65536)]).numpy()
                    ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
                    d = ImageDraw.Draw(ov)
                    if mode == "named":
                        nm_ = names[m_][cl_]
                        for (x0, y0), g_ in zip(xy_f, nm_):
                            if g_ in rgb:
                                d.rectangle([x0 * scl, y0 * scl, x0 * scl + b_, y0 * scl + b_], fill=rgb[g_] + (235,))
                        panels.append(Image.alpha_composite(th.convert("L").convert("RGBA"), ov).convert("RGB"))
                        for g_ in rgb:
                            rec[f"{m_}__{g_}"] = float((nm_ == g_).mean())
                    else:
                        for (x0, y0), g_ in zip(xy_f, grp[m_][cl_]):
                            d.rectangle([x0 * scl, y0 * scl, x0 * scl + b_, y0 * scl + b_], fill=pal[int(g_)] + (255,))
                        panels.append(Image.alpha_composite(th.convert("RGBA"), ov).convert("RGB"))
                canvas = Image.new("RGB", (th.width * len(panels) + 10 * (len(panels) - 1), th.height + 30), "white")
                dd = ImageDraw.Draw(canvas)
                for j, (pn, ttl) in enumerate(zip(panels, ["original"] + [m_.split("_")[0] + (" (30 groups)" if mode == "groups" else "") for m_ in cfg["maps"]])):
                    canvas.paste(pn, (j * (th.width + 10), 30))
                    dd.text((j * (th.width + 10) + 4, 6), ttl, fill="black", font=font)
                canvas.save(mout / f"{r.slide_id}_{mode}.jpg", quality=85)
            mrows.append(rec)
        pd.DataFrame(mrows).to_csv(run_dir / "map_slides.csv", index=False)
        logger.info(f"maps done; {time.time() - t0:.0f}s")
        # 7b. 各プロトタイプの見た目: 全slideの間引きpatchから、プロトタイプごとに別々のslideの tiles_per_proto 枚（投与群を優先）
        tout = run_dir / "tiles"
        tout.mkdir(exist_ok=True)
        rng = np.random.default_rng(cfg["seed"])
        ok_s = (sl.svs_path != "").to_numpy()
        sdf = sl[["slide_id", "svs_path", "COMPOUND_NAME", "DOSE_LEVEL", "SACRIFICE_PERIOD", "EXP_ID"]].copy()
        lbl = lambda s_, q_: (f"{s_.slide_id} e{s_.EXP_ID}", f"{str(s_.COMPOUND_NAME)[:14]} {str(s_.DOSE_LEVEL)[0]} {s_.SACRIFICE_PERIOD}")
        for m_, (run, rep, k) in cfg["maps"].items():
            cl_ = cls[m_]
            tab = pd.read_csv(project_root / "outputs" / cfg["exp0029"] / cfg["names_from"] / f"prototype_names_{m_}.csv").set_index("prototype")
            for part in range(0, k, mt["protos_per_sheet"]):
                rows_t = []
                for j in range(part, min(k, part + mt["protos_per_sheet"])):
                    idx = np.where((cl_ == j) & ok_s[row])[0]
                    if not len(idx):
                        continue
                    treated = idx[~ctrl_row[idx]] if (ctrl_row := sl.is_ctrl.to_numpy()[row])[idx].any() else idx
                    order_ = rng.permutation(treated if len(treated) >= mt["tiles_per_proto"] else idx)
                    picks, seen = [], set()
                    for q in order_:
                        if row[q] in seen:
                            continue
                        picks.append(int(q)); seen.add(row[q])
                        if len(picks) >= mt["tiles_per_proto"]:
                            break
                    nm_ = tab.name.get(j, "")
                    hdr = f"p{j:02d}  name: {nm_ if isinstance(nm_, str) and nm_ else '-'}   Bayer top: {tab.top_label.get(j, '')} ({tab.n_patches_bayer.get(j, 0):,} patches)   unseen patches: {len(idx):,} in {len(np.unique(row[idx]))} slides"
                    rows_t.append((hdr, ex17.read_tiles(picks, row, XY, sdf, size0, lbl)))
                ex17.draw_sheet(f"{m_}: prototypes {part}-{min(k, part + mt['protos_per_sheet']) - 1} on TG-GATEs slides unseen in training", rows_t, tout / f"{m_}_p{part:03d}.jpg")
            logger.info(f"tiles {m_}; {time.time() - t0:.0f}s")

    results: dict = {"n_slides": len(sl), "n_experiments": len(exps), "mean_AUC_readout": res.groupby("map").AUC_readout.mean().to_dict(),
                     "mean_AUC_named": res.groupby("map").AUC_named_noTGtraining.mean().to_dict() if "AUC_named_noTGtraining" in res else {}}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()