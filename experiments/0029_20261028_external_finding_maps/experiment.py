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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def name_prototypes(cl, lab, prio, k, nm):
    """Bayerのラベル付きpatchで、各プロトタイプに濃縮されたラベルの名前を付ける（割合 >= min_share か、全体の割合の min_lift 倍以上。複数あれば倍率の最大）。"""
    m = lab >= 0
    cnt = np.zeros((k, len(prio)))
    np.add.at(cnt, (cl[m], lab[m]), 1)
    n_lab = cnt.sum(1, keepdims=True)
    share = cnt / np.maximum(n_lab, 1)
    prev = cnt.sum(0) / cnt.sum()
    lift = share / np.maximum(prev, 1e-12)
    ok = (cnt >= nm["min_n"]) & ((share >= nm["min_share"]) | (lift >= nm["min_lift"]))
    rows = []
    for j in range(k):
        cand = np.where(ok[j])[0]
        best = cand[np.argmax(lift[j, cand])] if len(cand) else -1
        rows.append({"prototype": j, "name": prio[best] if best >= 0 else "", "share": share[j, best] if best >= 0 else np.nan,
                     "lift": lift[j, best] if best >= 0 else np.nan, "n_labeled": int(n_lab[j, 0]),
                     "top_label": prio[int(np.argmax(cnt[j]))] if n_lab[j, 0] else "", "n_patches_bayer": int((cl == j).sum())})
    return pd.DataFrame(rows)


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
    import h5py
    import openslide
    import torch
    from PIL import Image, ImageDraw, ImageFont
    cfg = {**config, **config.get("variants", {}).get(args.variant, {})}
    t0 = time.time()
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    ex23 = load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex26 = load_module(project_root / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(project_root / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    cfg20 = yaml.safe_load(open(project_root / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    prio = cfg20["label_priority"]
    run26 = project_root / "outputs" / cfg["exp0026"]

    # 1. TG-GATEs: 0026 tree_v3と同じLDA（脂肪滴系の21枚のラベルを除く）を学び直す。プロトタイプの名前はBayerのラベルから
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Ka, laba = c["K"], c["lab"].astype(np.int64)
    keep = [k for k in np.unique(laba) if k >= 0 and (laba == k).sum() >= cfg20["min_label_patches"]]
    laba = np.where(np.isin(laba, keep), laba, -1)
    lab_c = np.where(laba >= 0, np.searchsorted(keep, np.maximum(laba, 0)), -1)
    ex_ids = pd.read_csv(project_root / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab_c = np.where(np.isin(pd.read_csv(src / "slides.csv", dtype=str).slide_id.to_numpy()[c["row"]], ex_ids), -1, lab_c)
    tf = ex26.fit_combo(Ka, lab_c, cfg, cfg["seed"])
    L = len(keep) - 1
    na = len(Ka)
    del Ka
    pre = pickle.loads((project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl").read_bytes())["pre"]
    names, cens = {}, {}
    for mname, (run, rep, k) in cfg["maps"].items():
        cl_all = np.load(run26 / run / "tree" / f"{rep}_k{k}" / "assign_all.npy").astype(np.int64)
        tab = name_prototypes(cl_all[:na], laba, prio, k, cfg["naming"])
        tab.to_csv(run_dir / f"prototype_names_{mname}.csv", index=False)
        names[mname] = tab.name.to_numpy()
        cens[mname] = np.load(run26 / run / f"centroids_{rep}_k{k}.npy").astype(np.float32)
        logger.info(f"{mname}: names {tab.name.replace('', '-').value_counts().to_dict()}")
    logger.info(f"TG-GATEs side ready; {time.time() - t0:.0f}s")

    # 2. 外部の全slideを写す（Sphering は外部の各試験の対照群で）
    ext = project_root / "outputs" / cfg["exp0022"]
    sl = pd.read_csv(ext / "default/slides.csv", dtype=str)
    sl["is_ctrl"] = sl.control_status.isin(cfg["control_status"])
    parts = []
    for r in sl.itertuples():
        with h5py.File(ext / "features_uni_v2_macenko" / f"{r.slide_id.replace(' ', '_')}.h5", "r") as h:
            f = h["features"][:].astype(np.float32)
            parts.append({"xy": h["coords"][:], "size0": int(h["coords"].attrs["patch_size_level0"])})
        K = pre.transform(ex23.slide_rank(f)).astype(np.float32)
        parts[-1].update(U=K[:, :64], Lp=tf(K)[:, :L])
    row = np.concatenate([np.full(len(p_["U"]), i) for i, p_ in enumerate(parts)])
    U = np.concatenate([p_["U"] for p_ in parts])
    Lp = np.concatenate([p_["Lp"] for p_ in parts])
    study = sl.study.to_numpy()[row]
    cm = sl.is_ctrl.to_numpy()[row]
    Sw = ex27.sphering(U, study, U[cm], study[cm])
    sl_run = [run for run, rep, k in cfg["maps"].values() if rep == "sphering_lda"]
    sc = json.loads((run26 / sl_run[0] / "scale_sphering_lda.json").read_text()) if sl_run else None

    def as_rep(rep, U_, S_, L_):
        """表現の名前から、外部（またはTG-GATEsの対照群）の特徴を組み立てる"""
        return {"sphering_lda": lambda: np.concatenate([S_, sc["lda_beta"] * sc["lda_scale"] * L_], axis=1), "sphering": lambda: S_,
                "lda": lambda: L_, "unsup": lambda: U_}[rep]().astype(np.float32)
    X = {mname: as_rep(rep, U, Sw, Lp) for mname, (run, rep, k) in cfg["maps"].items()}
    titles = cfg.get("titles", {"A_sphering_lda": "A: Sphering + LDA", "B_sphering": "B: Sphering"})
    cls = {}
    for mname in cfg["maps"]:
        ct = torch.from_numpy(cens[mname])
        cls[mname] = torch.cat([torch.cdist(x, ct).argmin(1) for x in torch.from_numpy(X[mname]).split(65536)]).numpy()
    logger.info(f"external: {len(sl)} slides, {len(U):,} patches assigned; {time.time() - t0:.0f}s")

    # 3. slideの選び方（約40枚）と、各slideの記録
    fr = pd.read_csv(Path(cfg["bench_root"]) / cfg["external_findings"], encoding="utf-8-sig", dtype=str)
    fr["grade"] = pd.to_numeric(fr.grade_ordinal_within_source, errors="coerce").fillna(0)
    pos = fr[fr.grade > 0]
    se = cfg["selection"]
    is_nec = pos.finding_raw.str.contains(se["necrosis_word"]) & ~pos.finding_raw.str.contains("|".join(se["necrosis_exclude"]))
    key = lambda d: set(zip(d.study, d.label_unit))
    nec_u, inf_u = key(pos[is_nec]), key(pos[pos.finding_raw.str.contains(se["inflammation_word"])])
    sl["unit"] = list(zip(sl.study, sl.label_unit))
    sl["necrosis_recorded"] = sl.unit.isin(nec_u)
    sl["inflammation_recorded"] = sl.unit.isin(inf_u)
    sl["records"] = [" / ".join(f"{g.finding_raw} ({g.grade_raw})" for g in pos[(pos.study == r.study) & (pos.label_unit == r.label_unit)].itertuples()) for r in sl.itertuples()]
    ctrl_pick = sl[sl.is_ctrl].sort_values("slide_id").groupby("study").head(se["n_ctrl_per_study"]).index
    sl["role"] = np.select([sl.necrosis_recorded, sl.index.isin(ctrl_pick), sl.inflammation_recorded & ~sl.necrosis_recorded],
                           ["necrosis", "control", "inflammation_only"], "")
    sel = sl[sl.role != ""].copy()
    logger.info(f"selected {len(sel)} slides: {sel.role.value_counts().to_dict()}")

    # 4. 色の付いた領域の割合（全slide）と地図（選んだslide）
    groups = cfg["groups"]
    share_rows = []
    for i, r in enumerate(sl.itertuples()):
        m = row == i
        d = {"slide_id": r.slide_id, "study": r.study, "role": r.role, "is_ctrl": r.is_ctrl, "necrosis_recorded": r.necrosis_recorded, "inflammation_recorded": r.inflammation_recorded, "records": r.records}
        for mname in cfg["maps"]:
            nm_ = names[mname][cls[mname][m]]
            for g in groups:
                d[f"{mname}__{g}"] = float((nm_ == g).mean())
        share_rows.append(d)
    shares = pd.DataFrame(share_rows)
    shares.to_csv(run_dir / "slide_group_shares.csv", index=False)
    from scipy.stats import mannwhitneyu
    auc = []
    for g, rec in [("necrosis", "necrosis_recorded"), ("inflammation", "inflammation_recorded")]:
        for mname in cfg["maps"]:
            v = shares[f"{mname}__{g}"]
            pos, ctl, neg = v[shares[rec]], v[shares.is_ctrl & ~shares[rec]], v[~shares[rec]]
            auc.append({"group": g, "map": mname, "n_pos": len(pos), "median_pos": pos.median(), "median_ctrl": ctl.median(),
                        "AUC_pos_vs_ctrl": mannwhitneyu(pos, ctl).statistic / (len(pos) * len(ctl)), "AUC_pos_vs_all_neg": mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))})
    pd.DataFrame(auc).to_csv(run_dir / "slide_auc.csv", index=False)
    logger.info("slide AUC:\n" + pd.DataFrame(auc).round(3).to_string(index=False))
    logger.info("median share by role:\n" + shares[shares.role != ""].groupby("role")[[c_ for c_ in shares if "__" in c_]].median().round(4).T.to_string())

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    out = run_dir / "maps"
    out.mkdir(exist_ok=True)
    rgb = {g: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for g, h in groups.items()}
    for i in sel.index:
        r = sl.loc[i]
        w = openslide.OpenSlide(str(Path(cfg["bench_root"]) / r.path))
        th = w.get_thumbnail((cfg["thumb"], cfg["thumb"])).convert("RGB")
        s_ = th.width / w.dimensions[0]
        w.close()
        m = row == i
        xy, b = parts[i]["xy"], max(2, round(parts[i]["size0"] * s_))
        panels = [th]
        for mname in cfg["maps"]:
            ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            for (x0, y0), g in zip(xy, names[mname][cls[mname][m]]):
                if g in rgb:
                    d.rectangle([x0 * s_, y0 * s_, x0 * s_ + b, y0 * s_ + b], fill=rgb[g] + (235,))
            panels.append(Image.alpha_composite(th.convert("L").convert("RGBA"), ov).convert("RGB"))
        canvas = Image.new("RGB", (th.width * len(panels) + 10 * (len(panels) - 1), th.height + 30), "white")
        dd = ImageDraw.Draw(canvas)
        for j, (pn, ttl) in enumerate(zip(panels, ["original"] + [titles[m_] + "" for m_ in cfg["maps"]])):
            canvas.paste(pn, (j * (th.width + 10), 30))
            dd.text((j * (th.width + 10) + 4, 6), ttl, fill="black", font=font)
        canvas.save(out / f"{r.slide_id.replace(' ', '_')}.jpg", quality=85)
    sel.drop(columns=["unit"]).to_csv(run_dir / "selected_slides.csv", index=False)

    # 5. 拡大図: 各slideで、壊死・炎症の色（AとBの和）が最も密な窓と、肝臓内の無作為な窓を、元画像・A・Bで並べる
    zc = cfg.get("zoom")
    if zc:
        zout = run_dir / "zoom"
        zout.mkdir(exist_ok=True)
        rng = np.random.default_rng(cfg["seed"])
        W, P = zc["window_patches"], zc["px_per_patch"]
        zrows = []
        for i in sel.index:
            r = sl.loc[i]
            m = row == i
            xy, s0 = parts[i]["xy"], parts[i]["size0"]
            gx, gy = (xy[:, 0] - xy[:, 0].min()) // s0, (xy[:, 1] - xy[:, 1].min()) // s0
            nm = {mname: names[mname][cls[mname][m]] for mname in cfg["maps"]}
            windows = []
            for kind in zc["kinds"]:
                if kind == "random":
                    score = np.ones(len(xy))
                else:
                    score = np.zeros(len(xy))
                    for mname in cfg["maps"]:
                        score += nm[mname] == kind
                grid = np.zeros((gy.max() + W + 1, gx.max() + W + 1))
                np.add.at(grid, (gy, gx), score)
                cs = np.pad(grid.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                dens = cs[W:, W:] - cs[:-W, W:] - cs[W:, :-W] + cs[:-W, :-W]   # 左上が(y, x)の窓の合計
                for (wy, wx) in windows:          # 既に選んだ窓と重ならないように
                    dens[max(0, wy - W + 1):wy + W, max(0, wx - W + 1):wx + W] = -1
                if kind == "random":
                    cand = np.argwhere(dens >= 0.8 * W * W)
                    if not len(cand):
                        continue
                    wy, wx = cand[rng.integers(len(cand))]
                else:
                    if dens.max() < zc["min_hits"]:
                        continue
                    wy, wx = np.unravel_index(np.argmax(dens), dens.shape)
                windows.append((wy, wx))
                x0, y0 = int(xy[:, 0].min() + wx * s0), int(xy[:, 1].min() + wy * s0)
                w = openslide.OpenSlide(str(Path(cfg["bench_root"]) / r.path))
                ds = W * s0 / (W * P)
                lv = w.get_best_level_for_downsample(ds)
                f_ = w.level_downsamples[lv]
                im = w.read_region((x0, y0), lv, (int(W * s0 / f_), int(W * s0 / f_))).convert("RGB").resize((W * P, W * P), Image.LANCZOS)
                w.close()
                inw = (gx >= wx) & (gx < wx + W) & (gy >= wy) & (gy < wy + W)
                panels = [im]
                for mname in cfg["maps"]:
                    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
                    d = ImageDraw.Draw(ov)
                    for qx, qy, g in zip(gx[inw] - wx, gy[inw] - wy, nm[mname][inw]):
                        if g in rgb:
                            d.rectangle([qx * P + 1, qy * P + 1, (qx + 1) * P - 2, (qy + 1) * P - 2], outline=rgb[g] + (255,), width=3, fill=rgb[g] + (zc["alpha"],))
                    panels.append(Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB"))
                canvas = Image.new("RGB", (W * P * len(panels) + 10 * (len(panels) - 1), W * P + 30), "white")
                dd = ImageDraw.Draw(canvas)
                for j, (pn, ttl) in enumerate(zip(panels, ["original"] + [titles[m_] + "" for m_ in cfg["maps"]])):
                    canvas.paste(pn, (j * (W * P + 10), 30))
                    dd.text((j * (W * P + 10) + 4, 6), ttl, fill="black", font=font)
                fn = f"{r.slide_id.replace(' ', '_')}__{kind}.jpg"
                canvas.save(zout / fn, quality=88)
                zrows.append({"slide_id": r.slide_id, "kind": kind, "file": f"zoom/{fn}", "x0": x0, "y0": y0, "size_um": round(W * s0 * float(r.mpp_x), 0),
                              **{f"{mname}__{g}": int((nm[mname][inw] == g).sum()) for mname in cfg["maps"] for g in groups}})
        pd.DataFrame(zrows).to_csv(run_dir / "zoom_windows.csv", index=False)
        logger.info(f"zoom: {len(zrows)} windows; {time.time() - t0:.0f}s")

    # 6. 試験間差の影響（2026-10-28ユーザー: 特に塩野義とエーザイ）。対照群のslideだけで、A/Bそれぞれ
    se_ = cfg.get("study_effect")
    if se_:
        from scipy.spatial.distance import jensenshannon
        import colorsys
        prov = sl.provider.to_numpy()
        ci = np.where(sl.is_ctrl)[0]
        eff = []
        for mname, (run, rep, k) in cfg["maps"].items():
            P_ = np.zeros((len(sl), k))
            np.add.at(P_, (row, cls[mname]), 1)
            P_ /= P_.sum(1, keepdims=True)
            d = {"within_study": [], "between_study_same_provider": [], "between_provider": []}
            for a in range(len(ci)):
                for b in range(a + 1, len(ci)):
                    i_, j_ = ci[a], ci[b]
                    kind = "within_study" if sl.study[i_] == sl.study[j_] else ("between_study_same_provider" if prov[i_] == prov[j_] else "between_provider")
                    d[kind].append(jensenshannon(P_[i_], P_[j_]))
            # 近傍: 対照群のpatchのうち、別の会社・別の試験のpatchが近傍に占める割合（同じslideは除く）
            cmk = np.isin(row, ci)
            Xc = torch.from_numpy(X[mname][cmk])
            rc, pc, stc = row[cmk], prov[row[cmk]], study[cmk]
            q = np.random.default_rng(cfg["seed"]).choice(len(Xc), min(se_["queries"], len(Xc)), replace=False)
            nn_prov, nn_study = [], []
            for qq in np.array_split(q, 20):
                dist = torch.cdist(Xc[qq], Xc)
                dist[torch.from_numpy(rc[None, :] == rc[qq][:, None])] = float("inf")
                nb = dist.topk(se_["knn"], largest=False).indices.numpy()
                nn_prov.append((pc[nb] != pc[qq][:, None]).mean(1))
                nn_study.append((stc[nb] != stc[qq][:, None]).mean(1))
            # 会社に偏ったプロトタイプ: 会社ごとの対照群patch数で割った割合で、片方が90%以上
            cnt = pd.crosstab(cls[mname][cmk], pc)
            frac = cnt / cnt.sum(0)
            spec = (frac.max(1) / frac.sum(1)) >= se_["provider_specific"]
            eff.append({"map": mname, **{f"JS_{k_}": float(np.mean(v)) for k_, v in d.items()},
                        "knn_other_provider": float(np.concatenate(nn_prov).mean()), "knn_other_study": float(np.concatenate(nn_study).mean()),
                        "expected_other_provider_if_mixed": float(1 - ((pd.Series(pc).value_counts() / len(pc)) ** 2).sum()),
                        "share_ctrl_patches_in_provider_specific_protos": float(cnt[spec].sum().sum() / cnt.sum().sum()), "n_provider_specific_protos": int(spec.sum())})
            pd.DataFrame(P_, index=sl.slide_id).to_csv(run_dir / f"proto_shares_{mname}.csv")
        eff = pd.DataFrame(eff)
        eff.to_csv(run_dir / "study_effect.csv", index=False)
        logger.info("study effect (controls):\n" + eff.round(3).T.to_string())

        # 全プロトタイプの色分け（系統樹を30に切ったまとまりで色。AとBの色は対応しない）
        grp = {}
        for mname, (run, rep, k) in cfg["maps"].items():
            t_ = pd.read_csv(run26 / run / "tree" / f"{rep}_k{k}" / "prototypes_in_tree_order.csv").set_index("prototype")
            grp[mname] = t_.group_30.reindex(range(k)).to_numpy()
        pal = [tuple(int(255 * c_) for c_ in colorsys.hsv_to_rgb((i_ * 0.618) % 1, 0.65 + 0.3 * (i_ % 2), 0.95 - 0.25 * ((i_ // 2) % 2))) for i_ in range(31)]
        aout = run_dir / "maps_all"
        aout.mkdir(exist_ok=True)
        for i in np.r_[ci, sel.index[~sel.index.isin(ci)]]:
            r = sl.loc[i]
            w = openslide.OpenSlide(str(Path(cfg["bench_root"]) / r.path))
            th = w.get_thumbnail((cfg["thumb"], cfg["thumb"])).convert("RGB")
            s_ = th.width / w.dimensions[0]
            w.close()
            m = row == i
            xy, b = parts[i]["xy"], max(2, round(parts[i]["size0"] * s_))
            panels = [th]
            for mname in cfg["maps"]:
                ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
                d = ImageDraw.Draw(ov)
                for (x0, y0), g in zip(xy, grp[mname][cls[mname][m]]):
                    d.rectangle([x0 * s_, y0 * s_, x0 * s_ + b, y0 * s_ + b], fill=pal[int(g)] + (255,))
                panels.append(Image.alpha_composite(th.convert("RGBA"), ov).convert("RGB"))
            canvas = Image.new("RGB", (th.width * len(panels) + 10 * (len(panels) - 1), th.height + 30), "white")
            dd = ImageDraw.Draw(canvas)
            for j, (pn, ttl) in enumerate(zip(panels, ["original"] + [titles[m_] + " (30 groups)" for m_ in cfg["maps"]])):
                canvas.paste(pn, (j * (th.width + 10), 30))
                dd.text((j * (th.width + 10) + 4, 6), ttl, fill="black", font=font)
            canvas.save(aout / f"{r.slide_id.replace(' ', '_')}.jpg", quality=85)
        logger.info(f"all-prototype maps; {time.time() - t0:.0f}s")

    # 7. 壊死と名付いたプロトタイプに入った外部のpatch（A/B、壊死の記録ありslideと、それ以外のslideから）
    nt = cfg.get("necrosis_tiles")
    if nt:
        ex17 = load_module(project_root / "experiments/0017_20261016_block_consensus_patterns/experiment.py", "exp0017")
        sdf = sl.assign(svs_path=[str(Path(cfg["bench_root"]) / p_) for p_ in sl.path])
        size0 = {r.slide_id: parts[i]["size0"] for i, r in enumerate(sl.itertuples())}
        xy_all = np.concatenate([p_["xy"] for p_ in parts])
        rng = np.random.default_rng(cfg["seed"])
        lbl = lambda s_, q_: (f"{s_.provider[:4]} {s_.slide_id}", f"{'nec rec' if s_.necrosis_recorded else ('ctrl' if s_.is_ctrl else 'no nec rec')}")
        rows_t, nrec = [], []
        for mname in cfg["maps"]:
            hit = names[mname][cls[mname]] == "necrosis"
            for part, mask in [("壊死の記録ありslide", sl.necrosis_recorded.to_numpy()[row]), ("壊死の記録なしslide（対照群を含む）", ~sl.necrosis_recorded.to_numpy()[row])]:
                idx = np.where(hit & mask)[0]
                pick = []
                for i_ in rng.permutation(np.unique(row[idx])):
                    cand = idx[row[idx] == i_]
                    pick += list(rng.choice(cand, min(nt["per_slide"], len(cand)), replace=False))
                    if len(pick) >= nt["n"]:
                        break
                nrec.append({"map": mname, "part": part, "n_patches": len(idx), "n_slides": len(np.unique(row[idx]))})
                if pick:
                    rows_t.append((f"{mname}: {part}（壊死と名付いたプロトタイプのpatch {len(idx):,}、{len(np.unique(row[idx]))} slide）", ex17.read_tiles(pick[:nt['n']], row, xy_all, sdf, size0, lbl)))
        ex17.draw_sheet("External patches in prototypes named necrosis (A: Sphering + LDA, B: Sphering)", rows_t, run_dir / "necrosis_tiles.jpg")
        pd.DataFrame(nrec).to_csv(run_dir / "necrosis_tiles_counts.csv", index=False)
        logger.info(f"necrosis tiles: {nrec}; {time.time() - t0:.0f}s")

    # 8. TG-GATEsと外部の差（2026-10-28ユーザー: 差が大きいイメージ。これを超えられるか）。対照群どうしで、補正なし（PCA64）・B・Aを比べる
    dg = cfg.get("domain_gap")
    if dg:
        from concurrent.futures import ProcessPoolExecutor
        from scipy.spatial.distance import jensenshannon
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(cfg["seed"])
        # TG-GATEsの対照群: Bayerの集合の試験から無作為に dg["tg_exps"] 試験、各試験 dg["tg_per_exp"] 枚、各slide dg["patches"] patch
        q = pd.read_csv(dg["qc_catalog"], dtype=str)
        q = q[(q.DOSE_LEVEL == "Control") & q.EXP_ID.notna()]
        bayer_exps = set(pd.read_csv(src / "slides.csv", dtype=str).EXP_ID.dropna())
        def h5_of(sid):
            for d_ in dg["features_dirs"]:
                if (project_root / d_ / f"{sid}.h5").exists():
                    return str(project_root / d_ / f"{sid}.h5")
            return ""
        q = q[q.EXP_ID.isin(bayer_exps)].copy()
        q["h5"] = [h5_of(x) for x in q.slide_id]
        q = q[q.h5 != ""]
        ok_exps = [e for e, n in q.EXP_ID.value_counts().items() if n >= dg["tg_per_exp"]]
        exps = rng.choice(sorted(ok_exps), dg["tg_exps"], replace=False)
        qt = pd.concat([q[q.EXP_ID == e].sample(dg["tg_per_exp"], random_state=cfg["seed"]) for e in exps])
        with ProcessPoolExecutor(max_workers=cfg.get("workers", 6), initializer=ex27._init_ctrl, initargs=(str(project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl"), dg["patches"])) as ex:
            Kt = list(ex.map(ex27._load_ctrl, qt.h5))
        st_t = np.concatenate([np.full(len(k_), f"TG e{e}") for k_, e in zip(Kt, qt.EXP_ID)])
        Kt = np.concatenate(Kt)
        Ut, Lt = Kt[:, :64], tf(Kt)[:, :L]
        # TG-GATEs側の白色化: 各試験のControl（ここでは選んだ対照群そのもの）で。0026と同じ手順
        St = ex27.sphering(Ut, st_t, Ut, st_t)
        # 外部の対照群: 各slideから同じ数のpatch
        ce = np.concatenate([rng.choice(np.where(row == i)[0], min(dg["patches"], int((row == i).sum())), replace=False) for i in np.where(sl.is_ctrl)[0]])
        st_e = np.array([f"{sl.provider[i][:4]} {sl.study[i]}" for i in row[ce]])
        dom = np.r_[np.zeros(len(Kt), int), np.ones(len(ce), int)]
        stud = np.r_[st_t, st_e]
        R = {mname: np.r_[as_rep(rep, Ut, St, Lt), X[mname][ce]] for mname, (run, rep, k) in cfg["maps"].items()}
        if not any(rep == "unsup" for run, rep, k in cfg["maps"].values()):
            R = {"none (PCA64)": np.r_[Ut, U[ce]], **R}
        gap = []
        for rname, Xr in R.items():
            Xr = torch.from_numpy(Xr.astype(np.float32))
            qi = rng.choice(len(Xr), min(dg["queries"], len(Xr)), replace=False)
            od, os_ = [], []
            for qq in np.array_split(qi, 20):
                d_ = torch.cdist(Xr[qq], Xr)
                d_[torch.from_numpy(stud[None, :] == stud[qq][:, None])] = float("inf")   # 同じ試験は除く
                nb = d_.topk(dg["knn"], largest=False).indices.numpy()
                od.append((dom[nb] != dom[qq][:, None]).mean(1))
                os_.append(dom[qq])
            od, os_ = np.concatenate(od), np.concatenate(os_)
            rec = {"representation": rname, "knn_other_domain_from_TG": float(od[os_ == 0].mean()), "knn_other_domain_from_external": float(od[os_ == 1].mean()),
                   "expected_if_mixed_from_TG": float((dom == 1).mean()), "expected_if_mixed_from_external": float((dom == 0).mean())}
            mk = rname if rname in cfg["maps"] else None
            if mk:
                mins = [torch.cdist(x, torch.from_numpy(cens[mk])).min(1) for x in Xr.split(65536)]
                dmin, a = torch.cat([m_.values for m_ in mins]).numpy(), torch.cat([m_.indices for m_ in mins]).numpy()
                rec["median_dist_to_centroid_TG"] = float(np.median(dmin[dom == 0]))
                rec["median_dist_to_centroid_external"] = float(np.median(dmin[dom == 1]))
                Pm = {s_: np.bincount(a[stud == s_], minlength=len(cens[mk])) / (stud == s_).sum() for s_ in np.unique(stud)}
                ss = list(Pm)
                dd = {"TG_vs_TG": [], "ext_vs_ext": [], "TG_vs_ext": []}
                for i_ in range(len(ss)):
                    for j_ in range(i_ + 1, len(ss)):
                        k_ = "TG_vs_TG" if ss[i_].startswith("TG") and ss[j_].startswith("TG") else ("ext_vs_ext" if not ss[i_].startswith("TG") and not ss[j_].startswith("TG") else "TG_vs_ext")
                        dd[k_].append(jensenshannon(Pm[ss[i_]], Pm[ss[j_]]))
                rec.update({f"JS_study_{k_}": float(np.mean(v)) for k_, v in dd.items()})
            gap.append(rec)
            # 図: t-SNE（各試験から同じ数）
            pick = np.concatenate([rng.choice(np.where(stud == s_)[0], dg["tsne_per_study"], replace=False) for s_ in np.unique(stud)])
            Y = TSNE(perplexity=30, init="pca", random_state=cfg["seed"]).fit_transform(Xr.numpy()[pick])
            fig, ax = plt.subplots(figsize=(6.5, 6))
            cols = {"TG": "#999999", "shio": "#0072B2", "eisa": "#D55E00"}
            for s_ in np.unique(stud[pick]):
                mm = stud[pick] == s_
                ax.scatter(Y[mm, 0], Y[mm, 1], s=3, alpha=.5, color=cols[s_.split()[0]], label=None)
            for k_, c_ in cols.items():
                ax.scatter([], [], color=c_, s=20, label={"TG": "TG-GATEs controls", "shio": "Shionogi controls", "eisa": "Eisai controls"}[k_])
            ax.legend(frameon=False, loc="best", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{rname}: t-SNE of control patches ({dg['tsne_per_study']} per study)", fontsize=9)
            fig.savefig(run_dir / f"domain_tsne_{rname.split()[0]}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        gap = pd.DataFrame(gap)
        gap.to_csv(run_dir / "domain_gap.csv", index=False)
        qt.to_csv(run_dir / "domain_gap_tg_controls.csv", index=False)
        logger.info(f"TG controls: {len(qt)} slides from experiments {list(exps)}\n" + gap.round(3).T.to_string())
    logger.info(f"maps: {len(sel)}; {time.time() - t0:.0f}s")

    results: dict = {"n_slides": len(sl), "n_selected": len(sel), "roles": sel.role.value_counts().to_dict(),
                     "named": {k_: pd.Series(v).replace("", "-").value_counts().to_dict() for k_, v in names.items()}}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()