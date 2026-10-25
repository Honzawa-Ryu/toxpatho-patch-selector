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
    parser.add_argument("--variant", default="lobular_v1")
    parser.add_argument("--mode", default="tgg", choices=["tgg", "external"])   # external: TG-GATEsだけで学習した層を外部（0022）に当てる
    return parser.parse_args()


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 140})


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_external(cfg, project_root, run_dir, logger, ex21, pre, X, pos, pm, src):
    """TG-GATEsのアノテーション132枚だけで学習した層（小葉の位置の回帰、0021の形態プロトタイプ）を、外部の78枚にそのまま当てる。
    外部では何も学習・調整しない。"""
    from scipy.stats import norm
    from sklearn.linear_model import RidgeCV
    import torch
    model = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(X[pm], pos[pm])
    logger.info(f"lobular model fitted on {pm.sum():,} TG-GATEs patches (alpha {model.alpha_:.3g})")
    models21 = pickle.loads((src / "models.pkl").read_bytes())
    cen = torch.from_numpy(np.load(src / "centroids_combo.npy"))
    ptab = pd.read_csv(src / "clusters_combo.csv").set_index("cluster")
    ext = project_root / "outputs" / cfg["exp0022"] / "default"
    sl = pd.read_csv(ext / "slides.csv", dtype={"slide_id": str})
    fdir = project_root / "outputs" / cfg["exp0022"] / "features_uni_v2_macenko"
    fr = pd.read_csv(Path(cfg["bench_root"]) / cfg["external_findings"], encoding="utf-8-sig", dtype=str)
    fr["grade"] = pd.to_numeric(fr.grade_ordinal_within_source, errors="coerce").fillna(0)
    recs, keep = [], {}
    for r in sl.itertuples():
        key = r.slide_id.replace(" ", "_")
        with h5py.File(fdir / f"{key}.h5", "r") as h:
            f = h["features"][:].astype(np.float32)
            xy = h["coords"][:]
            size0 = int(h["coords"].attrs["patch_size_level0"])
        f /= np.linalg.norm(f, axis=1, keepdims=True)
        srt = np.sort(f, axis=0)
        rk = np.empty_like(f)
        for j in range(f.shape[1]):
            rk[:, j] = np.searchsorted(srt[:, j], f[:, j])
        rk = norm.ppf((rk + 0.5) / len(f)).astype(np.float32)
        K = pre.transform(rk).astype(np.float32)[:, :cfg["dim"]]
        pr = ex21.grid_neighbors(xy, size0)
        Xe = np.concatenate([K] + [ex21.contextualize(K, pr, h_, cfg["alpha"]) for h_ in cfg["hops"]], axis=1)
        lob = model.predict(Xe)
        rel = pd.Series(lob).rank(pct=True).to_numpy()
        Z = torch.from_numpy(ex21.transform(models21, rk, "combo", {"pca_components": 64, "combo_resid_weight": cfg["combo_resid_weight"]}))
        cl = torch.cdist(Z, cen).argmin(1).numpy()
        f_ = fr[(fr.study == r.study) & (fr.label_unit == str(r.label_unit))]
        pos_f = f_[f_.grade > 0]
        rec = {"slide_id": r.slide_id, "provider": r.provider, "study": r.study, "label_unit": r.label_unit, "compound": r.compound, "dose": r.dose_raw,
               "control_status": r.control_status, "n_patches": len(f),
               "findings": "; ".join(f"{a} ({g:g})" for a, g in zip(pos_f.finding_raw, pos_f.grade)),
               "centrilobular_necrosis": float(pos_f[pos_f.finding_raw.str.contains("ecrosis") & pos_f.finding_raw.str.contains("entrilobular")].grade.max() if len(pos_f) else 0) if len(pos_f) else 0.0,
               "centrilobular_hypertrophy": float(pos_f[pos_f.finding_raw.str.contains("ypertrophy") & pos_f.finding_raw.str.contains("entrilobular")].grade.max()) if len(pos_f[pos_f.finding_raw.str.contains("ypertrophy") & pos_f.finding_raw.str.contains("entrilobular")]) else 0.0}
        for name, ids in cfg["lesion_prototypes"].items():
            mk = np.isin(cl, ids)
            rec[f"share_{name}"] = float(mk.mean())
            rec[f"relpos_{name}"] = float(np.median(rel[mk])) if mk.sum() >= 20 else np.nan
        recs.append(rec)
        if r.slide_id in [str(x) for x in cfg["external_maps"]]:
            keep[r.slide_id] = (r, xy, size0, lob, cl)
        logger.info(f"{key}: {len(f)} patches; " + ", ".join(f"{k} {v:.3f}" for k, v in rec.items() if k.startswith("share_")))
    t = pd.DataFrame(recs)
    t.to_csv(run_dir / "external_layers.csv", index=False)
    # 図: 壊死プロトタイプの割合と、その小葉内の位置（記録の中心静脈側の壊死の程度ごと）
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in [(axes[0], "share_necrosis"), (axes[1], "relpos_necrosis")]:
        for g, c in [(0, "#9aa3ad"), (1, "#a8cbf2"), (2, "#5b9be6"), (3, "#2a78d6"), (4, "#1f4f8f")]:
            v = t[t.centrilobular_necrosis == g][col].dropna()
            if len(v):
                ax.scatter(np.full(len(v), g) + np.random.default_rng(0).uniform(-.15, .15, len(v)), v, color=c, s=25)
                ax.text(g, ax.get_ylim()[1] if False else np.nanmax(t[col]) * 1.02 if col.startswith("share") else 1.02, f"n={len(v)}", ha="center", fontsize=8)
        ax.set_xlabel("recorded centrilobular necrosis grade (within-company scale; 0 = not recorded)")
        ax.set_ylabel({"share_necrosis": "share of patches in TG-GATEs necrosis prototypes", "relpos_necrosis": "their predicted lobular position (rank in slide; 0 = central)"}[col])
        if col.startswith("relpos"):
            ax.axhline(.5, color="#cccccc")
            ax.set_ylim(0, 1.05)
    fig.suptitle("X1. External (Eisai, Shionogi): TG-GATEs-trained layers applied as is (no external training)", fontsize=11)
    fig.savefig(out / "X1_external_necrosis.png")
    plt.close(fig)
    # 地図
    import openslide
    from PIL import Image, ImageDraw, ImageFont
    import colorsys
    import matplotlib.cm as mcm
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    struct = {k for k, rr in ptab.iterrows() if isinstance(rr.top_label, str) and rr.top_label != "parenchyma" and rr.top_label_purity >= .5}
    pal = {k: tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * .61803398875) % 1, .85 if k in struct else .35, .9 if k in struct else .85)) for i, k in enumerate(ptab.index)}
    cmap = mcm.get_cmap("coolwarm")
    for sid, (r, xy, size0, lob, cl) in keep.items():
        w = openslide.OpenSlide(str(Path(cfg["bench_root"]) / r.path))
        th = w.get_thumbnail((1200, 1200)).convert("RGB")
        sc = th.width / w.dimensions[0]
        w.close()
        b = max(2, round(size0 * sc))
        lo, hi = np.percentile(lob, [5, 95])
        panels = [th]
        for kind in ("proto", "lob"):
            ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            for (x0, y0), k, v in zip(xy, cl, lob):
                col = pal[k] if kind == "proto" else tuple(int(255 * q) for q in cmap(float(np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)))[:3])
                d.rectangle([x0 * sc, y0 * sc, x0 * sc + b, y0 * sc + b], fill=col + (220,))
            panels.append(Image.alpha_composite(th.convert("L").convert("RGBA"), ov).convert("RGB"))
        canvas = Image.new("RGB", (th.width * 3 + 20, th.height + 34), "white")
        for j, pn in enumerate(panels):
            canvas.paste(pn, (j * (th.width + 10), 30))
        rr = t[t.slide_id == sid].iloc[0]
        ImageDraw.Draw(canvas).text((6, 6), f"{sid} {r.provider} {r.study} {r.compound} {r.dose_raw} | middle: TG-GATEs prototypes (vivid = Bayer-label prototypes) | right: lobular position "
                                            f"(blue central, red portal) | findings: {rr.findings[:120]}", fill="black", font=font)
        canvas.save(out / f"external_{sid.replace(' ', '_')}.jpg", quality=88)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    args = parse_args()
    exp_name = os.environ["EXP_NAME"]
    cfg = load_config(Path(__file__).parent)
    run_dir = get_run_dir(project_root, __file__, args.variant, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=args.variant, config=cfg)
    ex21 = load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    ex20 = load_module(project_root / "experiments" / cfg["exp0020"] / "experiment.py", "exp0020")
    cfg20 = {**ex20.load_config(project_root / "experiments" / cfg["exp0020"]), "random_per_slide": 10 ** 9}
    prio = cfg20["label_priority"]
    src = project_root / "outputs" / cfg["exp0021"] / cfg["models_from"]
    pre = pickle.loads((src / "models.pkl").read_bytes())["pre"]   # 教師なしのPCA（ラベルは使っていない）
    slides = pd.read_csv(src / "slides.csv", dtype={"slide_id": str})
    slides = slides[slides.source == "annotated"].reset_index(drop=True)
    ann = project_root / cfg20["annotation_dir"]
    fdir = project_root / cfg20["features_dir"]

    # --- 表現: slide内の順位 → 教師なしPCA64、その周りの平均（伝播4・16回） ---
    t0 = time.time()
    Xs, labs, rows, xys, pairs_all, size0 = [], [], [], [], [], {}
    off = 0
    for i, sid in enumerate(slides.slide_id):
        r = ex20._load_slide((sid, fdir / f"{sid}.h5", ann / "annotations/geojson" / f"{sid}.svs.geojson", cfg20))
        with h5py.File(fdir / f"{sid}.h5", "r") as h:
            size0[sid] = int(h["coords"].attrs.get("patch_size_level0", 256))
        K = pre.transform(r["rank"]).astype(np.float32)[:, :cfg["dim"]]
        pr = ex21.grid_neighbors(r["xy"], size0[sid])
        Xs.append(np.concatenate([K] + [ex21.contextualize(K, pr, h_, cfg["alpha"]) for h_ in cfg["hops"]], axis=1).astype(np.float32))
        labs.append(r["label"])
        rows.append(np.full(len(K), i))
        xys.append(r["xy"])
        pairs_all.append(pr + off)
        off += len(K)
        if (i + 1) % 20 == 0:
            logger.info(f"  {i + 1}/{len(slides)} slides; {time.time() - t0:.0f}s")
    X, lab, row, xy, pairs = np.concatenate(Xs), np.concatenate(labs), np.concatenate(rows), np.concatenate(xys), np.concatenate(pairs_all)
    del Xs
    pos = ex21.lobular_position(slides, row, xy, size0, project_root, cfg["lobular_max_dist"])
    pm = np.isfinite(pos)
    exp = slides.exp.astype(str).to_numpy()
    logger.info(f"{len(slides)} slides, {len(X):,} patches, dim {X.shape[1]}; position targets {pm.sum():,} patches on "
                f"{len(np.unique(row[pm]))} slides; {time.time() - t0:.0f}s")

    if args.mode == "external":
        run_external(cfg, project_root, run_dir, logger, ex21, pre, X, pos, pm, src)
        complete_run(run_dir)
        logger.info("Done.")
        return
    # --- 小葉の位置: 試験ごとに分けた5分割で学習し、学習に使わない試験の全patchを予測（out-of-fold） ---
    oof = np.full(len(X), np.nan, dtype=np.float32)
    fold_of_slide = np.full(len(slides), -1)
    gkf = GroupKFold(n_splits=cfg["folds"])
    for f, (tr_s, te_s) in enumerate(gkf.split(np.arange(len(slides)), groups=exp)):
        fold_of_slide[te_s] = f
        trm = np.isin(row, tr_s) & pm
        model = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(X[trm], pos[trm])
        tem = np.isin(row, te_s)
        oof[tem] = model.predict(X[tem])
        tm = tem & pm
        r2 = 1 - ((pos[tm] - oof[tm]) ** 2).sum() / ((pos[tm] - pos[tm].mean()) ** 2).sum() if tm.any() else np.nan
        logger.info(f"fold {f}: train {trm.sum():,} target patches, test exps {sorted(set(exp[te_s]))[:6]}..., R2 {r2:.3f}")
    r2_all = 1 - ((pos[pm] - oof[pm]) ** 2).sum() / ((pos[pm] - pos[pm].mean()) ** 2).sum()
    from scipy.stats import spearmanr
    rho = spearmanr(pos[pm], oof[pm])[0]
    logger.info(f"out-of-fold (leave-experiment-out): R2 {r2_all:.3f}, Spearman {rho:.3f}")
    # slide内での相対位置（slideごとの順位、0〜1）: slide全体の偏りを除く
    rel = np.empty_like(oof)
    for si in range(len(slides)):
        mk = row == si
        rel[mk] = (pd.Series(oof[mk]).rank(pct=True)).to_numpy()
    np.savez(run_dir / "lobular_oof.npz", oof=oof, rel=rel, row=row, lab=lab, xy=xy, pos=pos, fold_of_slide=fold_of_slide)

    # --- 現実との突き合わせ: Bayerのラベルの付いたpatchの位置を、化合物ごとに、所見表の分布（TOPOGRAPHY_TYPE）と並べる ---
    p_ = pd.read_csv(project_root / "data/corrected/open_tggates_pathology.csv", dtype=str)
    p_ = p_[p_.ORGAN == "Liver"]
    q = pd.read_csv(cfg20["qc_catalog"], dtype=str).set_index("slide_id")
    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
    sk = q.loc[slides.slide_id, key].reset_index()
    for c in key:
        sk[c] = pd.to_numeric(sk[c])
        p_[c] = pd.to_numeric(p_[c], errors="coerce")
    topo = sk.merge(p_, on=key)
    fmap = {"necrosis": "Necrosis", "single_cell_necrosis": "Single cell necrosis", "mitosis": "Increased mitosis", "inflammation": "Cellular infiltration"}
    recs = []
    for lname, fname in fmap.items():
        li = prio.index(lname)
        for comp in slides.compound.unique():
            sidx = np.flatnonzero(slides.compound.to_numpy() == comp)
            mk = (lab == li) & np.isin(row, sidx)
            if mk.sum() < cfg["min_patches"]:
                continue
            t = topo[(topo.slide_id.isin(slides.slide_id.iloc[sidx])) & (topo.FINDING_TYPE == fname)].TOPOGRAPHY_TYPE.value_counts()
            recs.append({"label": lname, "compound": comp, "n_patches": int(mk.sum()), "n_slides": int(len(np.unique(row[mk]))),
                         "recorded_topography": "; ".join(f"{k} {v}" for k, v in t.items()) if len(t) else "",
                         "main_topography": t.index[0] if len(t) else "", "oof_pos_median": float(np.median(oof[mk])),
                         "rel_pos_median": float(np.median(rel[mk])), "rel_pos_q25": float(np.quantile(rel[mk], .25)), "rel_pos_q75": float(np.quantile(rel[mk], .75))})
    tab = pd.DataFrame(recs).sort_values(["label", "rel_pos_median"])
    tab.to_csv(run_dir / "lesion_positions.csv", index=False)
    logger.info("\n" + tab[["label", "compound", "n_patches", "main_topography", "oof_pos_median", "rel_pos_median"]].to_string(index=False))
    # 図
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    colors = {"Centrilobular": "#2a78d6", "Periportal": "#e34948", "Hepatocyte": "#9aa3ad", "Midlobular": "#1baf7a"}
    labs_show = [l for l in fmap if (tab.label == l).any()]
    fig, axes = plt.subplots(1, len(labs_show), figsize=(5.5 * len(labs_show), 0.32 * tab.groupby("label").size().max() + 2.5), squeeze=False)
    for ax, lname in zip(axes[0], labs_show):
        t = tab[tab.label == lname].reset_index(drop=True)
        li = prio.index(lname)
        for i, r in t.iterrows():
            sidx = np.flatnonzero(slides.compound.to_numpy() == r.compound)
            v = rel[(lab == li) & np.isin(row, sidx)]
            ax.boxplot([v], positions=[i], vert=False, widths=.6, showfliers=False,
                       boxprops=dict(color=colors.get(r.main_topography, "black")), medianprops=dict(color=colors.get(r.main_topography, "black"), lw=2))
        ax.set_yticks(range(len(t)), [f"{c[:22]} ({n}p) [{tp[:12] or '-'}]" for c, n, tp in zip(t.compound, t.n_patches, t.main_topography)], fontsize=8)
        ax.axvline(.5, color="#cccccc", lw=1)
        ax.set_xlim(0, 1)
        ax.set_xlabel("predicted lobular position, rank within slide (0 = central-vein side, 1 = portal side)")
        ax.set_title(f"Bayer '{lname}' patches by compound\n[recorded topography of the finding: blue Centrilobular, red Periportal, grey Hepatocyte]", fontsize=9)
    fig.suptitle(f"R1. Predicted lobular position of lesion patches (model trained without the compound's experiment; OOF R2 {r2_all:.2f}, Spearman {rho:.2f})", fontsize=11)
    fig.savefig(out / "R1_lesion_lobular_position.png")
    plt.close(fig)
    (run_dir / "results.json").write_text(json.dumps({"oof_r2": float(r2_all), "oof_spearman": float(rho), "n_slides": int(len(slides)),
                                                      "lesions": tab.to_dict(orient="records")}, indent=2, ensure_ascii=False, default=float))
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
