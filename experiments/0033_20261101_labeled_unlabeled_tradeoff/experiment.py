"""0033: ラベルあり所見（①）とラベルなし所見（②）を両立する表現を選ぶ。設計は config.yml の冒頭と plan.md。

同じ土台（slide内の順位 → 0021のPCA256）から7つの表現を作り、3つの評価を1つの表に並べる:
  (a) heldout : 0031式。Bayerの230枚を化合物単位で5分割、分割内で LDA → コンセンサスk-means → 名前付け、評価側のpatchをポリゴンと照合
  (b) unseen  : 0030式。学習に使わないTG-GATEsのslideで、プロトタイプの構成比（同じ試験の対照群との差）から所見記録の有無を読み出す
  (c) fatty   : 0026式。脂肪変性の3試験で、プロトタイプごとに脂肪変性ありslide由来の割合（純度）と程度との相関
(b)(c)のプロトタイプはBayerの230枚＋脂肪変性の3試験で作る。
"""
import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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
    return parser.parse_args()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


COL = {"D_unsup": "#0072B2", "D13_unsup13": "#56B4E9", "C_lda": "#D55E00", "B_sphering": "#009E73", "A_sphering_lda": "#E69F00", "E_unsup13_lda": "#CC79A7", "F_combo": "#999999"}   # Okabe-Ito


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")
    args = parse_args()
    variant_key = args.variant
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    config = load_config(Path(__file__).parent)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)
    logger.info(f"Starting: {exp_name} / {variant_key}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    import torch
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cfg = {**config, **config.get("variants", {}).get(args.variant, {})}
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex25 = load_module(project_root / "experiments" / cfg["exp0025"] / "experiment.py", "exp0025")
    ex26 = load_module(project_root / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(project_root / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    ex29 = load_module(project_root / "experiments" / cfg["exp0029"] / "experiment.py", "exp0029")
    ex30 = load_module(project_root / "experiments" / cfg["exp0030"] / "experiment.py", "exp0030")
    ex31 = load_module(project_root / "experiments" / cfg["exp0031"] / "experiment.py", "exp0031")
    cfg20 = yaml.safe_load(open(project_root / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    prio = cfg20["label_priority"]
    pre_path = str(project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl")
    reps_all, k = cfg["reps"], cfg["k"]

    def h5_of(sid):
        for d_ in cfg["features_dirs"]:
            if (project_root / d_ / f"{sid}.h5").exists():
                return str(project_root / d_ / f"{sid}.h5")
        return ""

    # 1. Bayerの230枚（0025 cache: slide内の順位 → PCA256）。脂肪滴系の21枚のラベルは学習・名前付け・評価のどれにも使わない（0031と同じ）
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    K, laba, row_a = c["K"], c["lab"].astype(np.int64), c["row"]
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    keep = [k_ for k_ in np.unique(laba) if k_ >= 0 and (laba == k_).sum() >= cfg20["min_label_patches"]]
    names_l = [prio[k_] for k_ in keep]
    nlab, L = len(keep), len(keep) - 1
    lab = np.where(np.isin(laba, keep), np.searchsorted(keep, np.maximum(laba, 0)), -1)
    ex_ids = pd.read_csv(project_root / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab = np.where(np.isin(bayer.slide_id.to_numpy()[row_a], ex_ids), -1, lab)
    laba = np.where(lab >= 0, laba, -1)
    exp_a = bayer.EXP_ID.fillna("-1").astype(int).to_numpy()[row_a]
    comp_s = bayer.COMPOUND_NAME.fillna("unknown_" + bayer.slide_id).to_numpy()
    U = K[:, :64].astype(np.float32)
    logger.info(f"Bayer: {len(bayer)} slides, {len(K):,} patches, labeled {(lab >= 0).sum():,} in {nlab} labels; compounds {len(set(comp_s))}; {time.time() - t0:.0f}s")

    # 2. 脂肪変性の3試験（各slide 2,048 patch、0026 target_slides）と、白色化用の対照群（Bayerの試験 ∪ 脂肪変性の試験）
    fc = cfg["fatty"]
    tgt = ex26.target_slides({**cfg, **fc, "features_dir": cfg["features_dirs"][0]}, project_root)
    tgt = tgt[~tgt.slide_id.isin(bayer.slide_id)].reset_index(drop=True)
    G_fat, _ = ex26.fat_grades({**cfg, **fc}, project_root, tgt)
    fat = G_fat["Degeneration, fatty"].to_numpy()
    cmpd = np.where(tgt.compound.str.contains("tetrachloride", case=False), "CCl4", np.where(tgt.compound.str.contains("cholesterol", case=False), "cholesterol", "other"))
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts = list(ex.map(ex30._load_sub, tgt.h5, chunksize=4))
    Kt = np.concatenate([p_[0] for p_ in parts]).astype(np.float32)
    row_t = np.concatenate([np.full(len(p_[0]), i) for i, p_ in enumerate(parts)])
    del parts
    exp_t = tgt.exp.to_numpy()[row_t]
    tgt.drop(columns=["h5"]).assign(fat_grade=fat, compound_group=cmpd).to_csv(run_dir / "fatty_slides.csv", index=False)
    logger.info(f"fatty: {len(tgt)} slides {dict(zip(*np.unique(cmpd, return_counts=True)))}, {len(Kt):,} patches, fatty degeneration > 0: {int((fat > 0).sum())}; {time.time() - t0:.0f}s")
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[q.EXP_ID.notna() & (q.EXP_ID != "")]
    exps_ab = set(exp_a) | set(fc["target_exps"])
    qc = q[(q.DOSE_LEVEL == "Control") & q.EXP_ID.astype(int).isin(exps_ab)].copy()
    qc["h5"] = [h5_of(x) for x in qc.slide_id]
    qc = qc[qc.h5 != ""]
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts_c = list(ex.map(ex27._load_ctrl, qc.h5, chunksize=4))
    Kc = np.concatenate(parts_c)[:, :64].astype(np.float32)
    exp_c = np.concatenate([np.full(len(p_), int(e)) for p_, e in zip(parts_c, qc.EXP_ID)])
    del parts_c
    logger.info(f"controls for sphering: {len(qc)} slides in {len(set(exp_c))} experiments, {len(Kc):,} patches; {time.time() - t0:.0f}s")

    def sphere(U_, e_, Kc_, ec_):
        """試験ごとに、その試験の対照群で白色化。対照群のない試験は全試験の対照群をまとめて使う（0026・0031と同じ）"""
        e_fix = np.where(np.isin(e_, ec_), e_, -999)
        return ex27.sphering(U_, e_fix, np.r_[Kc_, Kc_], np.r_[ec_, np.full(len(ec_), -999)]).astype(np.float32)

    def rep_of(r_, U_, Sw, Fc, L_, sc):
        """表現を1つ作る。Fc = fit_comboの出力 [LDA/s0 ‖ 0.5×残差PCA/s1]。sc = A・Eでのスケール（LDAの大きさを元の表現にそろえる）"""
        Lw = Fc[:, :L_]
        return {"D_unsup": lambda: U_, "D13_unsup13": lambda: U_[:, :L_], "C_lda": lambda: Lw, "B_sphering": lambda: Sw,
                "A_sphering_lda": lambda: np.concatenate([Sw, cfg["lda_beta"] * sc["A"] * Lw], 1),
                "E_unsup13_lda": lambda: np.concatenate([U_[:, :L_], cfg["lda_beta"] * sc["E"] * Lw], 1), "F_combo": lambda: Fc}[r_]().astype(np.float32)

    def scales_of(U_, Sw, Fc, L_):
        Lw = Fc[:, :L_]
        return {"A": float(Sw.std(0).mean() / Lw.std(0).mean()), "E": float(U_[:, :L_].std(0).mean() / Lw.std(0).mean())}

    def apply_tf(tf, K_):
        return np.concatenate([tf(b_) for b_ in np.array_split(K_, max(1, len(K_) // 200000))]).astype(np.float32)

    def assign(cen, A):
        return torch.cat([torch.cdist(torch.from_numpy(x).to(device), cen).argmin(1) for x in np.array_split(A, max(1, len(A) // 65536))]).cpu().numpy()

    Sw_a = sphere(U, exp_a, Kc, exp_c)
    Sw_t = sphere(Kt[:, :64], exp_t, Kc, exp_c)

    # 3. (a) held-out: 化合物単位の5分割。分割内で LDA → コンセンサスk-means → 名前付け（0031と同じ）
    hc = cfg["heldout"]
    pred = {r_: np.full(len(K), -1, dtype=np.int64) for r_ in reps_all}
    for fi, (tr_s, te_s) in enumerate(GroupKFold(n_splits=hc["folds"]).split(np.zeros(len(bayer)), groups=comp_s)):
        tr, te = np.isin(row_a, tr_s), np.isin(row_a, te_s)
        tf_f = ex26.fit_combo(K[tr], lab[tr], cfg, cfg["seed"] + fi)
        L_f = len(np.unique(lab[tr][lab[tr] >= 0])) - 1
        Fc_f = apply_tf(tf_f, K)
        sc_f = scales_of(U, Sw_a, Fc_f, L_f)
        tr_idx = torch.from_numpy(np.where(tr)[0]).to(device)
        for r_ in reps_all:
            X = torch.from_numpy(rep_of(r_, U, Sw_a, Fc_f, L_f, sc_f)).to(device)
            _, cen = ex25.consensus_kmeans(X[tr_idx], k, hc["consensus_runs"], hc["fit_points"], cfg["seed"] + fi, device)
            cl = torch.cat([torch.cdist(x, cen).argmin(1) for x in X.split(65536)]).cpu().numpy()
            nm = ex31.name_prototypes(cl[tr], lab[tr], nlab, k, cfg["naming"])
            pred[r_][te] = nm[cl[te]]
            del X
        del Fc_f
        logger.info(f"heldout fold {fi}: train {len(tr_s)} / test {len(te_s)} slides ({len(set(comp_s[te_s]))} compounds); {time.time() - t0:.0f}s")
    ev = lab >= 0
    y, cp = lab[ev], comp_s[row_a][ev]
    M = {r_: ex31.label_metrics(y, pred[r_][ev], nlab) for r_ in reps_all}
    comps = np.unique(cp)
    idx_by_c = {c_: np.where(cp == c_)[0] for c_ in comps}
    rng = np.random.default_rng(cfg["seed"])
    nonpar = [i for i, n in enumerate(names_l) if n != "parenchyma"]
    Bh = {r_: [] for r_ in reps_all}
    for _ in range(hc["bootstrap"]):
        ii = np.concatenate([idx_by_c[c_] for c_ in rng.choice(comps, len(comps))])
        for r_ in reps_all:
            Bh[r_].append(np.nanmean(ex31.label_metrics(y[ii], pred[r_][ev][ii], nlab)[nonpar, 2]))
    Bh = {r_: np.array(v) for r_, v in Bh.items()}
    held = pd.DataFrame([{"rep": r_, "macro_F1_nonparenchyma": float(np.nanmean(M[r_][nonpar, 2])), "lo": float(np.nanpercentile(Bh[r_], 2.5)), "hi": float(np.nanpercentile(Bh[r_], 97.5))} for r_ in reps_all])
    pd.DataFrame([{"rep": r_, "label": names_l[l], "n_test": int((y == l).sum()), "recall": M[r_][l, 0], "precision": M[r_][l, 1], "F1": M[r_][l, 2]} for r_ in reps_all for l in range(nlab)]).to_csv(run_dir / "heldout_label_metrics.csv", index=False)
    np.savez_compressed(run_dir / "heldout_pred.npz", **{r_: pred[r_].astype(np.int8) for r_ in reps_all}, ev=ev)
    del pred

    # 4. (b)(c)のプロトタイプ: Bayerの230枚＋脂肪変性の3試験。LDAは230枚の全ラベル（21枚除く）で
    tf = ex26.fit_combo(K, lab, cfg, cfg["seed"])
    Fc_a, Fc_t = apply_tf(tf, K), apply_tf(tf, Kt)
    sc = scales_of(U, Sw_a, Fc_a, L)
    (run_dir / "scales.json").write_text(json.dumps(sc))
    pc = cfg["proto"]
    fa = rng.choice(len(K), min(pc["fit_annot"], len(K)), replace=False)
    ft = rng.choice(len(Kt), min(pc["fit_target"], len(Kt)), replace=False)
    is_ctrl_t = tgt.dose.eq("Control").to_numpy()
    cens, names, cl_t, fat_rows, fat_summ = {}, {}, {}, [], []
    for r_ in reps_all:
        Xa, Xt = rep_of(r_, U, Sw_a, Fc_a, L, sc), rep_of(r_, Kt[:, :64], Sw_t, Fc_t, L, sc)
        _, cen = ex25.consensus_kmeans(torch.from_numpy(np.r_[Xa[fa], Xt[ft]]).to(device), k, pc["consensus_runs"], pc["fit_points"], cfg["seed"], device)
        cens[r_] = cen
        np.save(run_dir / f"centroids_{r_}_k{k}.npy", cen.cpu().numpy().astype(np.float32))
        cl_a = assign(cen, Xa)
        cl_t[r_] = assign(cen, Xt)
        nm = ex29.name_prototypes(cl_a, laba, prio, k, cfg["naming"])
        nm.to_csv(run_dir / f"prototype_names_{r_}.csv", index=False)
        names[r_] = nm.set_index("prototype").name.fillna("").reindex(range(k)).to_numpy()
        # (c) 脂肪変性: slideごとの構成比と、同じ試験の対照群との差（0026と同じ）
        sh = np.zeros((len(tgt), k))
        np.add.at(sh, (row_t, cl_t[r_]), 1)
        sh /= sh.sum(1, keepdims=True)
        d = np.full_like(sh, np.nan)
        for e, idx in tgt.groupby("exp").indices.items():
            cidx = idx[is_ctrl_t[idx]]
            if len(cidx) >= fc["min_ctrl"]:
                d[idx] = sh[idx] - sh[cidx].mean(0)
        tab = []
        for j in range(k):
            r = {"rep": r_, "prototype": j, "name": names[r_][j], "n_target": int((cl_t[r_] == j).sum())}
            for cname in ("CCl4", "cholesterol"):
                cm = cmpd == cname
                pm = (cl_t[r_] == j) & cm[row_t]
                r[f"n_{cname}"] = int(pm.sum())
                r[f"frac_from_fatty_{cname}"] = float((fat[row_t[pm]] > 0).mean()) if pm.any() else np.nan
                ok = cm & ~np.isnan(d[:, 0])
                r[f"rho_{cname}"] = spearmanr(fat[ok], d[ok, j])[0] if len(np.unique(fat[ok])) > 1 else np.nan
            tab.append(r)
        tab = pd.DataFrame(tab)
        fat_rows.append(tab)
        sr = {"rep": r_}
        for cname in ("CCl4", "cholesterol"):
            big = tab[tab[f"n_{cname}"] >= fc["min_proto_patches"]]
            sr[f"n_protos_purity90_{cname}"] = int((big[f"frac_from_fatty_{cname}"] >= 0.9).sum())
            sr[f"max_purity_{cname}"] = float(big[f"frac_from_fatty_{cname}"].max()) if len(big) else np.nan
            sr[f"best_rho_{cname}"] = float(big[f"rho_{cname}"].max()) if len(big) else np.nan
        fat_summ.append(sr)
        logger.info(f"prototypes {r_}: named {int((names[r_] != '').sum())}; fatty {sr}; {time.time() - t0:.0f}s")
        del Xa, Xt
    pd.concat(fat_rows).to_csv(run_dir / "fatty_prototypes.csv", index=False)
    fat_summ = pd.DataFrame(fat_summ)
    fat_summ.to_csv(run_dir / "fatty_summary.csv", index=False)
    del K, Kt, Fc_a, Fc_t, Sw_a, Sw_t, U

    # 5. (b) 学習に使わないTG-GATEsのslide（0030と同じ選び方）: 各slide 2,048 patch → PCA256。保存して次回から読み直さない
    uc = cfg["unseen"]
    q2 = q[~q.slide_id.isin(bayer.slide_id) & ~q.EXP_ID.astype(int).isin(uc["exclude_exps"])].copy()
    q2["h5"] = [h5_of(x) for x in q2.slide_id]
    q2 = q2[q2.h5 != ""]
    pth = pd.read_csv(project_root / cfg["pathology_csv"], dtype=str)
    pth = pth[(pth.ORGAN == "Liver") & (pth.SP_FLG.str.lower() != "true") & pth.FINDING_TYPE.isin(list(uc["findings"]))]
    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
    for k_ in key:
        q2[k_] = pd.to_numeric(q2[k_], errors="coerce")
        pth[k_] = pd.to_numeric(pth[k_], errors="coerce")
    pth["g"] = pth.GRADE_TYPE.map(uc["grade_map"]).fillna(1)
    pos = q2.merge(pth, on=key)
    sl = q2[q2.EXP_ID.isin(sorted(set(pos.EXP_ID)))].reset_index(drop=True)
    G = pos.pivot_table(index="slide_id", columns="FINDING_TYPE", values="g", aggfunc="max").reindex(sl.slide_id).fillna(0).reindex(columns=list(uc["findings"]), fill_value=0)
    sl["is_ctrl"] = sl.DOSE_LEVEL == "Control"
    sl["bayer_exp"] = sl.EXP_ID.isin(set(bayer.EXP_ID.dropna().astype(int)))
    logger.info(f"unseen: {len(sl)} slides in {sl.EXP_ID.nunique()} experiments, controls {int(sl.is_ctrl.sum())}; positives {(G > 0).sum().to_dict()}; {time.time() - t0:.0f}s")
    feat_from = project_root / uc["features_from"] if uc.get("features_from") else run_dir / "features_unseen.npz"
    if feat_from.exists():
        z = np.load(feat_from, allow_pickle=True)
        assert (z["slide_ids"] == sl.slide_id.to_numpy().astype(str)).all(), "features_from: slide list differs"
        Ku, row_u = z["K"].astype(np.float32), z["row"]
        logger.info(f"unseen features loaded from {feat_from}; {time.time() - t0:.0f}s")
    else:
        with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
            parts = list(ex.map(ex30._load_sub, sl.h5, chunksize=4))
        Ku = np.concatenate([p_[0] for p_ in parts]).astype(np.float32)
        row_u = np.concatenate([np.full(len(p_[0]), i) for i, p_ in enumerate(parts)])
        xy_u = np.concatenate([p_[1] for p_ in parts])
        del parts
        np.savez(run_dir / "features_unseen.npz", K=Ku.astype(np.float16), row=row_u, xy=xy_u, slide_ids=sl.slide_id.to_numpy().astype(str))
        logger.info(f"unseen patches: {len(Ku):,}, saved; {time.time() - t0:.0f}s")
    exp_u = sl.EXP_ID.to_numpy()[row_u]
    cm_u = sl.is_ctrl.to_numpy()[row_u]
    Uu = np.ascontiguousarray(Ku[:, :64])   # 複製（Kuの参照を残さず解放するため）
    Sw_u = sphere(Uu, exp_u, Uu[cm_u], exp_u[cm_u])
    Fc_u = apply_tf(tf, Ku)
    del Ku
    sl.drop(columns=["h5"]).assign(**{f"grade__{f}": G[f].to_numpy() for f in G}).to_csv(run_dir / "unseen_slides.csv", index=False)
    deltas = {}
    for r_ in reps_all:
        cl = assign(cens[r_], rep_of(r_, Uu, Sw_u, Fc_u, L, sc))
        sh = np.zeros((len(sl), k))
        np.add.at(sh, (row_u, cl), 1)
        sh /= sh.sum(1, keepdims=True)
        ctrl_mean = pd.DataFrame(sh[sl.is_ctrl.to_numpy()]).groupby(sl.EXP_ID.to_numpy()[sl.is_ctrl.to_numpy()]).mean()
        base = ctrl_mean.reindex(sl.EXP_ID).to_numpy()
        deltas[r_] = sh - np.where(np.isnan(base), sh[sl.is_ctrl.to_numpy()].mean(0), base)
        np.save(run_dir / f"unseen_shares_{r_}.npy", sh.astype(np.float32))
    del Uu, Sw_u, Fc_u
    logger.info(f"unseen assigned; {time.time() - t0:.0f}s")

    # 6. (b) 所見の読み出し（化合物単位のCV）と、名前付きプロトタイプ（学習なし）。化合物を復元抽出した区間
    comp_u = sl.COMPOUND_NAME.to_numpy()
    comps_u = np.unique(comp_u)
    idx_u = {c_: np.where(comp_u == c_)[0] for c_ in comps_u}
    boots = [np.concatenate([idx_u[c_] for c_ in rng.choice(comps_u, len(comps_u))]) for _ in range(uc["bootstrap"])]
    unlabeled = [f for f, b in uc["findings"].items() if not b]
    labeled = [f for f, b in uc["findings"].items() if b]
    rows, Bu = [], {r_: {"unlabeled_readout": [], "labeled_readout": [], "labeled_named": []} for r_ in reps_all}
    oofs = {}
    for f, blab in uc["findings"].items():
        yv = (G[f].to_numpy() > 0).astype(int)
        npos_c = len(set(comp_u[yv == 1]))
        for r_ in reps_all:
            D_ = deltas[r_]
            oof = np.full(len(sl), np.nan)
            for tr, te in GroupKFold(n_splits=min(uc["cv_splits"], npos_c)).split(D_, yv, comp_u):
                if yv[tr].sum() == 0:
                    continue
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
                oof[te] = clf.fit(D_[tr], yv[tr]).predict_proba(D_[te])[:, 1]
            ok = ~np.isnan(oof)
            r = {"finding": f, "bayer_label": blab or "(none)", "rep": r_, "n_pos": int(yv.sum()), "n_pos_compounds": npos_c,
                 "AUC_readout": roc_auc_score(yv[ok], oof[ok]) if len(set(yv[ok])) > 1 else np.nan}
            for part, pm in [("bayer_exp", sl.bayer_exp.to_numpy()), ("new_exp", ~sl.bayer_exp.to_numpy())]:
                m_ = ok & pm
                r[f"AUC_readout_{part}"] = roc_auc_score(yv[m_], oof[m_]) if len(set(yv[m_])) > 1 else np.nan
            gs = D_[:, names[r_] == blab].sum(1) if blab and (names[r_] == blab).any() else None
            if blab:
                r["n_named_protos"] = int((names[r_] == blab).sum())
                r["AUC_named_noTGtraining"] = roc_auc_score(yv, gs) if gs is not None else np.nan
            oofs[(f, r_)] = (oof, gs)
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "unseen_finding_classification.csv", index=False)

    def auc_b(yv, s, ii):
        m_ = ~np.isnan(s[ii])
        return roc_auc_score(yv[ii][m_], s[ii][m_]) if len(set(yv[ii][m_])) > 1 else np.nan

    for r_ in reps_all:
        for ii in boots:
            Bu[r_]["unlabeled_readout"].append(np.nanmean([auc_b((G[f].to_numpy() > 0).astype(int), oofs[(f, r_)][0], ii) for f in unlabeled]))
            Bu[r_]["labeled_readout"].append(np.nanmean([auc_b((G[f].to_numpy() > 0).astype(int), oofs[(f, r_)][0], ii) for f in labeled]))
            Bu[r_]["labeled_named"].append(np.nanmean([auc_b((G[f].to_numpy() > 0).astype(int), oofs[(f, r_)][1], ii) if oofs[(f, r_)][1] is not None else np.nan for f in labeled]))
    Bu = {r_: {k_: np.array(v) for k_, v in d_.items()} for r_, d_ in Bu.items()}
    summ = []
    for r_ in reps_all:
        s_ = {"rep": r_, "heldout_macro_F1": float(held.set_index("rep").loc[r_, "macro_F1_nonparenchyma"]), "heldout_F1_lo": float(held.set_index("rep").loc[r_, "lo"]), "heldout_F1_hi": float(held.set_index("rep").loc[r_, "hi"])}
        for k_ in ("unlabeled_readout", "labeled_readout", "labeled_named"):
            sub = res[res.finding.isin(unlabeled if k_ == "unlabeled_readout" else labeled) & (res.rep == r_)]
            s_[f"mean_AUC_{k_}"] = float(sub["AUC_named_noTGtraining" if k_ == "labeled_named" else "AUC_readout"].mean())
            s_[f"mean_AUC_{k_}_lo"], s_[f"mean_AUC_{k_}_hi"] = float(np.nanpercentile(Bu[r_][k_], 2.5)), float(np.nanpercentile(Bu[r_][k_], 97.5))
        s_.update({k_: v for k_, v in fat_summ.set_index("rep").loc[r_].items()})
        summ.append(s_)
    summ = pd.DataFrame(summ)
    summ.to_csv(run_dir / "summary.csv", index=False)
    diffs = []
    for ref in ("C_lda", "B_sphering"):
        for r_ in reps_all:
            if r_ == ref:
                continue
            for k_, (pa, pb) in {"heldout_macro_F1": (Bh[r_], Bh[ref]), "unlabeled_readout": (Bu[r_]["unlabeled_readout"], Bu[ref]["unlabeled_readout"]), "labeled_named": (Bu[r_]["labeled_named"], Bu[ref]["labeled_named"])}.items():
                dd = pa - pb
                diffs.append({"comparison": f"{r_} - {ref}", "metric": k_, "diff": float(np.nanmean(dd)), "lo": float(np.nanpercentile(dd, 2.5)), "hi": float(np.nanpercentile(dd, 97.5)), "p_le0": float(np.nanmean(dd <= 0))})
    pd.DataFrame(diffs).to_csv(run_dir / "differences.csv", index=False)
    pd.set_option("display.width", 250)
    logger.info("\n" + summ.round(3).to_string(index=False))
    logger.info("unseen AUC_readout:\n" + res.pivot_table(index="finding", columns="rep", values="AUC_readout").round(3)[reps_all].to_string())

    # 7. 図: ラベルあり（held-out F1）× ラベルなし（読み出しAUC）の両立、脂肪変性の純度
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    for _, s_ in summ.iterrows():
        ax.errorbar(s_.heldout_macro_F1, s_.mean_AUC_unlabeled_readout, xerr=[[s_.heldout_macro_F1 - s_.heldout_F1_lo], [s_.heldout_F1_hi - s_.heldout_macro_F1]],
                    yerr=[[s_.mean_AUC_unlabeled_readout - s_.mean_AUC_unlabeled_readout_lo], [s_.mean_AUC_unlabeled_readout_hi - s_.mean_AUC_unlabeled_readout]], fmt="o", color=COL[s_.rep], ms=7, lw=1, capsize=0, label=s_.rep)
    ax.set_xlabel("labeled findings: held-out macro F1 (Bayer polygons, compounds held out)")
    ax.set_ylabel("unlabeled findings: mean readout AUC (5 findings, unseen TG-GATEs)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("labeled vs unlabeled (95% CI, compounds resampled)", fontsize=10)
    xs = np.arange(len(reps_all))
    axes[1].bar(xs - 0.2, summ.n_protos_purity90_CCl4, width=0.4, color=[COL[r] for r in reps_all], label="CCl4")
    axes[1].bar(xs + 0.2, summ.n_protos_purity90_cholesterol, width=0.4, color=[COL[r] for r in reps_all], alpha=0.5, label="cholesterol diet")
    axes[1].set_title("fatty degeneration: prototypes with purity >= 0.9", fontsize=10)
    axes[1].legend(fontsize=8, frameon=False)
    axes[2].bar(xs - 0.2, summ.mean_AUC_labeled_named, width=0.4, color=[COL[r] for r in reps_all], label="named prototypes (no TG training)")
    axes[2].bar(xs + 0.2, summ.mean_AUC_labeled_readout, width=0.4, color=[COL[r] for r in reps_all], alpha=0.5, label="readout")
    axes[2].set_ylim(0.4, 1)
    axes[2].set_title("labeled findings on unseen TG-GATEs: mean AUC (5 findings)", fontsize=10)
    axes[2].legend(fontsize=8, frameon=False)
    for a_ in axes[1:]:
        a_.set_xticks(xs)
        a_.set_xticklabels(reps_all, rotation=25, ha="right", fontsize=8)
    for a_ in axes:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(run_dir / "F1_tradeoff.png", dpi=150)
    plt.close(fig)

    results = {"n_bayer_slides": int(len(bayer)), "n_fatty_slides": int(len(tgt)), "n_unseen_slides": int(len(sl)), "labels": names_l, "scales": sc,
               "summary": summ.set_index("rep").round(4).to_dict(orient="index")}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info(f"Done. {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
