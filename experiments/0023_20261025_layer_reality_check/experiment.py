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
    parser.add_argument("--mode", default="tgg", choices=["tgg", "external", "three_layers"])   # external: TG-GATEsだけで学習した層を外部（0022）に当てる
    return parser.parse_args()


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 140})


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def max_grade(pos_f, *words):
    """所見名に words をすべて含む行の最大の程度。該当する行がなければ0（空の最大値はNaNになり、図から落ちるため）。"""
    m = np.ones(len(pos_f), dtype=bool)
    for w in words:
        m &= pos_f.finding_raw.str.contains(w).to_numpy()
    return float(pos_f.grade[m].max()) if m.any() else 0.0


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
               "centrilobular_necrosis": max_grade(pos_f, "ecrosis", "entrilobular"),
               "centrilobular_hypertrophy": max_grade(pos_f, "ypertrophy", "entrilobular")}
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


def slide_rank(f):
    """slide内の順位を正規分布の分位点に置き換える（0020のrank_slideと同じ）。"""
    from scipy.stats import norm
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    srt = np.sort(f, axis=0)
    rk = np.empty_like(f)
    for j in range(f.shape[1]):
        rk[:, j] = np.searchsorted(srt[:, j], f[:, j])
    return norm.ppf((rk + 0.5) / len(f)).astype(np.float32)


def lesion_extent(cl, ids, pairs, n, ex21, hops):
    """病変プロトタイプ（ids）の指示関数を隣接グラフ上で伝播し、各patchの周りに同じ病変がどれだけあるか（0〜1）を返す。"""
    ind = np.isin(cl, ids).astype(np.float32)[:, None]
    return ex21.contextualize(ind, pairs, hops, 0.5)[:, 0] if len(pairs) else ind[:, 0]


def run_three_layers(cfg, project_root, run_dir, logger, ex21, pre, X, pos, pm, src, slides, lab, row, xy, pairs, size0, cl, prio):
    """3つの層（形態・小葉の位置・領域）をpatchごとに付け、TG-GATEs（所見表の程度・分布）と外部（企業の記録）に突き合わせる。
    - 形態: 0021の組み合わせ版のプロトタイプ（Bayerの病変ラベルで学んだLDAを含む。TG-GATEsのアノテーションslideでの評価は楽観的）
    - 小葉の位置: TG-GATEsは試験ごとに分けた5分割の予測（out-of-fold）、外部は全TG-GATEsで学習したモデル
    - 領域: 周りの情報（伝播16回）だけのk-means（TG-GATEsで学習）＋病変の広がり（病変プロトタイプの周りの密度）
    外部では何も学習・調整しない。"""
    import torch
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from lib.clustering import kmeans_lloyd
    hp = cfg["layers"]
    dim = cfg["dim"]
    exp = slides.exp.astype(str).to_numpy()
    les = cfg["lesion_prototypes"]
    # --- 小葉の位置 ---
    oof = np.full(len(X), np.nan, dtype=np.float32)
    for tr_s, te_s in GroupKFold(n_splits=cfg["folds"]).split(np.arange(len(slides)), groups=exp):
        trm = np.isin(row, tr_s) & pm
        tem = np.isin(row, te_s)
        oof[tem] = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(X[trm], pos[trm]).predict(X[tem])
    lob_full = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(X[pm], pos[pm])
    rel = np.empty_like(oof)
    for si in range(len(slides)):
        mk = row == si
        rel[mk] = pd.Series(oof[mk]).rank(pct=True).to_numpy()
    zone = np.digitize(rel, [1 / 3, 2 / 3])   # 0 中心静脈側、1 中間、2 門脈域側
    # --- 領域（周りの情報だけ） ---
    ctx = X[:, 2 * dim:3 * dim]   # 伝播16回
    rng = np.random.default_rng(cfg["seed"])
    fit = rng.choice(len(ctx), size=min(hp["region_fit"], len(ctx)), replace=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, rcen, _ = kmeans_lloyd(np.ascontiguousarray(ctx[fit]), hp["region_k"], cfg["seed"], dev)
    rcen_t = torch.from_numpy(rcen)
    region = np.concatenate([torch.cdist(torch.from_numpy(x), rcen_t).argmin(1).numpy() for x in np.array_split(ctx, 50)])
    ext_nec = lesion_extent(cl, les["necrosis"], pairs, len(cl), ex21, hp["extent_hops"])
    logger.info(f"layers on TG-GATEs: {len(X):,} patches; region k={hp['region_k']}")
    # 領域ごとの中身
    is_ctrl = (slides.dose.to_numpy() == "Control")[row]
    m = lab >= 0
    rtab = []
    for g in range(hp["region_k"]):
        mg = region == g
        rec = {"region": g, "share": float(mg.mean()), "ctrl_share": float(is_ctrl[mg].mean()), "lobular_rel_mean": float(rel[mg].mean()),
               "n_exp": int(len(np.unique(exp[row[mg]])))}
        for name, ids in les.items():
            rec[f"proto_{name}"] = float(np.isin(cl[mg], ids).mean())
        if (mg & m).any():
            lc = pd.Series(lab[mg & m]).value_counts(normalize=True)
            rec["top_label"], rec["top_label_share"] = prio[lc.index[0]], float(lc.iloc[0])
        rtab.append(rec)
    rtab = pd.DataFrame(rtab)
    rtab.to_csv(run_dir / "region_layer.csv", index=False)
    logger.info("\n" + rtab.round(3).to_string(index=False))

    # --- TG-GATEs: slideごとの3層の要約と、所見表（壊死の程度・分布） ---
    p_ = pd.read_csv(project_root / "data/corrected/open_tggates_pathology.csv", dtype=str)
    p_ = p_[(p_.ORGAN == "Liver") & (p_.SP_FLG.str.lower() != "true")]
    q = pd.read_csv(cfg["qc_catalog"], dtype=str).set_index("slide_id")
    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
    sk = q.loc[slides.slide_id, key].reset_index()
    for c in key:
        sk[c] = pd.to_numeric(sk[c])
        p_[c] = pd.to_numeric(p_[c], errors="coerce")
    tp = sk.merge(p_, on=key)
    tp["g"] = tp.GRADE_TYPE.map({"minimal": 1, "slight": 2, "moderate": 3, "severe": 4}).fillna(1)
    nec = tp[tp.FINDING_TYPE == "Necrosis"].groupby("slide_id").agg(grade=("g", "max"), topo=("TOPOGRAPHY_TYPE", lambda v: v.mode().iloc[0] if v.notna().any() else ""))
    trows = []
    for si, sid in enumerate(slides.slide_id):
        mk = row == si
        mn = mk & np.isin(cl, les["necrosis"])
        trows.append({"slide_id": sid, "compound": slides.compound.iloc[si], "dose": slides.dose.iloc[si], "necrosis_grade": float(nec.grade.get(sid, 0)),
                      "necrosis_topo": nec.topo.get(sid, ""), "share_necrosis": float(mn.sum() / mk.sum()),
                      "extent_necrosis": float(np.median(ext_nec[mn])) if mn.sum() >= hp["min_lesion_patches"] else np.nan,
                      "relpos_necrosis": float(np.median(rel[mn])) if mn.sum() >= hp["min_lesion_patches"] else np.nan})
    tt = pd.DataFrame(trows)
    tt.to_csv(run_dir / "tgg_slide_layers.csv", index=False)
    r1 = spearmanr(tt.necrosis_grade, tt.share_necrosis)[0]
    ok = tt.extent_necrosis.notna() & (tt.necrosis_grade > 0)
    r2 = spearmanr(tt.necrosis_grade[ok], tt.extent_necrosis[ok])[0] if ok.sum() > 5 else np.nan
    logger.info(f"TG-GATEs: necrosis share vs grade Spearman {r1:.3f}; extent vs grade (slides with necrosis) {r2:.3f}")

    # --- 外部 ---
    import torch as _t
    models21 = pickle.loads((src / "models.pkl").read_bytes())
    cen = _t.from_numpy(np.load(src / "centroids_combo.npy"))
    ext = project_root / "outputs" / cfg["exp0022"] / "default"
    sl = pd.read_csv(ext / "slides.csv", dtype={"slide_id": str})
    fdir = project_root / "outputs" / cfg["exp0022"] / "features_uni_v2_macenko"
    fr = pd.read_csv(Path(cfg["bench_root"]) / cfg["external_findings"], encoding="utf-8-sig", dtype=str)
    fr["grade"] = pd.to_numeric(fr.grade_ordinal_within_source, errors="coerce").fillna(0)
    erows, keep = [], {}
    for r in sl.itertuples():
        k_ = r.slide_id.replace(" ", "_")
        with h5py.File(fdir / f"{k_}.h5", "r") as h:
            f = h["features"][:].astype(np.float32)
            exy = h["coords"][:]
            s0 = int(h["coords"].attrs["patch_size_level0"])
        rk = slide_rank(f)
        K = pre.transform(rk).astype(np.float32)[:, :dim]
        pr = ex21.grid_neighbors(exy, s0)
        c16 = ex21.contextualize(K, pr, 16, cfg["alpha"])
        Xe = np.concatenate([K, ex21.contextualize(K, pr, 4, cfg["alpha"]), c16], axis=1)
        erel = pd.Series(lob_full.predict(Xe)).rank(pct=True).to_numpy()
        ecl = _t.cdist(_t.from_numpy(ex21.transform(models21, rk, "combo", {"pca_components": 64, "combo_resid_weight": cfg["combo_resid_weight"]})), cen).argmin(1).numpy()
        ereg = _t.cdist(_t.from_numpy(c16), rcen_t).argmin(1).numpy()
        eext = lesion_extent(ecl, les["necrosis"], pr, len(ecl), ex21, hp["extent_hops"])
        mn = np.isin(ecl, les["necrosis"])
        f_ = fr[(fr.study == r.study) & (fr.label_unit == str(r.label_unit)) & (fr.grade > 0)]
        necr = f_[f_.finding_raw.str.contains("ecrosis") & ~f_.finding_raw.str.contains("ingle cell") & ~f_.finding_raw.str.contains("bile")]
        rec = {"slide_id": r.slide_id, "provider": r.provider, "study": r.study, "compound": r.compound, "dose": r.dose_raw,
               "necrosis_grade": float(necr.grade.max()) if len(necr) else 0.0,
               "necrosis_focal": bool(necr.finding_raw.str.contains("ocal").any()), "necrosis_massive": bool(necr.finding_raw.str.contains("assive").any()),
               "necrosis_centrilobular": bool(necr.finding_raw.str.contains("entrilobular").any()),
               "findings": "; ".join(f"{a} ({g:g})" for a, g in zip(f_.finding_raw, f_.grade)),
               "share_necrosis": float(mn.mean()), "extent_necrosis": float(np.median(eext[mn])) if mn.sum() >= hp["min_lesion_patches"] else np.nan,
               "relpos_necrosis": float(np.median(erel[mn])) if mn.sum() >= hp["min_lesion_patches"] else np.nan}
        for g in range(hp["region_k"]):
            rec[f"region_{g}"] = float((ereg == g).mean())
        erows.append(rec)
        if r.slide_id in [str(x) for x in cfg["external_maps"]]:
            keep[r.slide_id] = (r, exy, s0, ecl, np.digitize(erel, [1 / 3, 2 / 3]), ereg)
    et = pd.DataFrame(erows)
    et.to_csv(run_dir / "external_slide_layers.csv", index=False)
    r3 = spearmanr(et.necrosis_grade, et.share_necrosis)[0]
    logger.info(f"external: necrosis share vs grade Spearman {r3:.3f}")
    draw_three_layer_figures(cfg, run_dir, tt, et, rtab, r1, r2, r3)
    # 地図: TG-GATEsの数枚と外部
    tg_keep = {}
    for sid in [str(x) for x in cfg["tg_maps"]]:
        hit = np.flatnonzero(slides.slide_id.to_numpy() == sid)
        if len(hit):
            si = hit[0]
            mk = row == si
            tg_keep[sid] = (slides.iloc[si], xy[mk], size0[sid], cl[mk], zone[mk], region[mk])
    draw_three_layer_maps(cfg, run_dir, tg_keep, keep, project_root, src, hp["region_k"])


def draw_three_layer_figures(cfg, run_dir, tt, et, rtab, r1, r2, r3):
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for rowi, (t, title, rr) in enumerate([(tt, "TG-GATEs (132 annotated slides; record: grade 1-4)", r1), (et, "External (78 slides; within-company grade)", r3)]):
        gr = sorted(t.necrosis_grade.unique())
        ax = axes[rowi, 0]
        for g in gr:
            v = t[t.necrosis_grade == g].share_necrosis
            ax.scatter(np.full(len(v), g) + rng.uniform(-.15, .15, len(v)), v, s=14, alpha=.7, color="#2a78d6")
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel("recorded necrosis grade (0 = none)")
        ax.set_ylabel("share of patches in necrosis prototypes")
        ax.set_title(f"{title}\nmorphology layer: Spearman {rr:.2f}", fontsize=9)
        ax = axes[rowi, 1]
        if rowi == 0:
            for g in gr:
                v = t[(t.necrosis_grade == g)].extent_necrosis.dropna()
                ax.scatter(np.full(len(v), g) + rng.uniform(-.15, .15, len(v)), v, s=14, alpha=.7, color="#1baf7a")
            ax.set_xlabel("recorded necrosis grade")
            ax.set_title(f"region layer (extent around necrosis patches)\nSpearman (necrosis slides) {r2:.2f}", fontsize=9)
        else:
            cats = [("none", (t.necrosis_grade == 0)), ("focal", t.necrosis_focal & ~t.necrosis_massive), ("centrilobular", t.necrosis_centrilobular & ~t.necrosis_massive),
                    ("massive", t.necrosis_massive)]
            for i, (nm, mk) in enumerate(cats):
                v = t[mk].extent_necrosis.dropna()
                ax.scatter(np.full(len(v), i) + rng.uniform(-.15, .15, len(v)), v, s=14, alpha=.7, color="#1baf7a")
                ax.text(i, 1.02, f"n={len(v)}", ha="center", fontsize=8)
            ax.set_xticks(range(len(cats)), [c for c, _ in cats])
            ax.set_ylim(0, 1.08)
            ax.set_title("region layer: extent by recorded pattern", fontsize=9)
        ax.set_ylabel("median neighbourhood necrosis density")
        ax = axes[rowi, 2]
        if rowi == 0:
            for i, tp in enumerate(["Centrilobular", "Hepatocyte", "Midlobular", "Periportal"]):
                v = t[t.necrosis_topo == tp].relpos_necrosis.dropna()
                ax.scatter(np.full(len(v), i) + rng.uniform(-.15, .15, len(v)), v, s=14, alpha=.7, color="#e34948")
                ax.text(i, 1.02, f"n={len(v)}", ha="center", fontsize=8)
            ax.set_xticks(range(4), ["Centrilob.", "Hepatocyte", "Midlob.", "Periportal"])
            ax.set_xlabel("recorded topography of necrosis")
        else:
            cats = [("no CL record", ~t.necrosis_centrilobular & (t.necrosis_grade > 0)), ("centrilobular", t.necrosis_centrilobular), ("none", t.necrosis_grade == 0)]
            for i, (nm, mk) in enumerate(cats):
                v = t[mk].relpos_necrosis.dropna()
                ax.scatter(np.full(len(v), i) + rng.uniform(-.15, .15, len(v)), v, s=14, alpha=.7, color="#e34948")
                ax.text(i, 1.02, f"n={len(v)}", ha="center", fontsize=8)
            ax.set_xticks(range(len(cats)), [c for c, _ in cats])
        ax.axhline(.5, color="#cccccc")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("lobular position of necrosis patches (0 = central)")
        ax.set_title("lobular layer", fontsize=9)
    fig.suptitle("L1. Three layers vs pathology records (morphology / region-extent / lobular position)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "L1_three_layers_vs_records.png")
    plt.close(fig)


def draw_three_layer_maps(cfg, run_dir, tg_keep, ext_keep, project_root, src, region_k):
    """1枚につき4枚並べる: 元画像 / 形態（病変プロトタイプだけ色、他は灰） / 小葉の位置（3帯） / 合成（実質を小葉の帯で淡く塗り、病変を濃く重ねる）。"""
    import colorsys
    import openslide
    from PIL import Image, ImageDraw, ImageFont
    les = cfg["lesion_prototypes"]
    lcol = {"necrosis": (220, 30, 30), "mitosis": (200, 0, 200), "single_cell_necrosis": (240, 150, 0), "dark_hepatocytes": (120, 60, 20),
            "central_vein": (0, 170, 200), "portal_tract": (30, 60, 220)}
    zcol = [(110, 150, 235), (215, 215, 215), (240, 140, 120)]
    rcol = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * .61803398875) % 1, .55, .9)) for i in range(region_k)]
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    out = run_dir / "figures" / "maps"
    out.mkdir(parents=True, exist_ok=True)

    def lesion_name(k):
        for nm, ids in les.items():
            if k in ids:
                return nm
        return None

    def draw(tag, svs, title, xy, s0, cl, zone, region):
        w = openslide.OpenSlide(svs)
        th = w.get_thumbnail((cfg["map_thumb"], cfg["map_thumb"])).convert("RGB")
        sc = th.width / w.dimensions[0]
        w.close()
        b = max(2, round(s0 * sc))
        gray = th.convert("L").convert("RGBA")
        panels = [th]
        for kind in ("morph", "zone", "region", "composite"):
            ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            for (x0, y0), k, z, g in zip(xy, cl, zone, region):
                ln = lesion_name(int(k))
                if kind == "morph":
                    col = lcol[ln] + (235,) if ln else (200, 200, 200, 120)
                elif kind == "zone":
                    col = zcol[z] + (220,)
                elif kind == "region":
                    col = rcol[g] + (220,)
                else:
                    col = (lcol[ln] + (240,)) if ln else (zcol[z] + (200,))
                d.rectangle([x0 * sc, y0 * sc, x0 * sc + b, y0 * sc + b], fill=col)
            panels.append(Image.alpha_composite(gray, ov).convert("RGB"))
        W = sum(p.width for p in panels) + 10 * (len(panels) - 1)
        canvas = Image.new("RGB", (W, th.height + 56), "white")
        cd = ImageDraw.Draw(canvas)
        x = 0
        for p, nm in zip(panels, ["original", "morphology (lesion prototypes)", "lobular zone (blue central / grey mid / red portal)", f"region (k={region_k})", "composite"]):
            canvas.paste(p, (x, 50))
            cd.text((x + 4, 30), nm, fill="black", font=font)
            x += p.width + 10
        cd.text((6, 6), title + " | lesion colours: red necrosis, magenta mitosis, orange single-cell necrosis, brown dark hepatocytes, cyan central vein, blue portal tract",
                fill="black", font=font)
        canvas.save(out / f"{tag}.jpg", quality=88)

    for sid, (r, xy, s0, cl, zone, region) in tg_keep.items():
        draw(f"tgg_{sid}", r.svs_path, f"TG-GATEs {sid} e{r.exp} {r.compound} {r.dose} {r.period} (lobular = out-of-fold by experiment)", xy, s0, cl, zone, region)
    for sid, (r, xy, s0, cl, zone, region) in ext_keep.items():
        draw(f"external_{sid.replace(' ', '_')}", str(Path(cfg["bench_root"]) / r.path), f"External {sid} {r.provider} {r.compound} {r.dose_raw} (no external training)",
             xy, s0, cl, zone, region)


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
    Xs, labs, rows, xys, pairs_all, size0, cls = [], [], [], [], [], {}, []
    import torch
    models21 = pickle.loads((src / "models.pkl").read_bytes())
    cen21 = torch.from_numpy(np.load(src / "centroids_combo.npy"))
    off = 0
    for i, sid in enumerate(slides.slide_id):
        r = ex20._load_slide((sid, fdir / f"{sid}.h5", ann / "annotations/geojson" / f"{sid}.svs.geojson", cfg20))
        with h5py.File(fdir / f"{sid}.h5", "r") as h:
            size0[sid] = int(h["coords"].attrs.get("patch_size_level0", 256))
        K = pre.transform(r["rank"]).astype(np.float32)[:, :cfg["dim"]]
        pr = ex21.grid_neighbors(r["xy"], size0[sid])
        Xs.append(np.concatenate([K] + [ex21.contextualize(K, pr, h_, cfg["alpha"]) for h_ in cfg["hops"]], axis=1).astype(np.float32))
        if args.mode == "three_layers":
            Zc = torch.from_numpy(ex21.transform(models21, r["rank"], "combo", {"pca_components": 64, "combo_resid_weight": cfg["combo_resid_weight"]}))
            cls.append(torch.cdist(Zc, cen21).argmin(1).numpy())
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

    if args.mode == "three_layers":
        run_three_layers(cfg, project_root, run_dir, logger, ex21, pre, X, pos, pm, src, slides, lab, row, xy, pairs, size0, np.concatenate(cls), prio)
        complete_run(run_dir)
        logger.info("Done.")
        return
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
