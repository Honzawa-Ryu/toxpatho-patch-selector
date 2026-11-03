"""0036: 局所プロトタイプ・ヒストグラムで空間文脈を調べる（第 1 段階、解析のみ）。設計は config.yml の冒頭と plan.md。

1. 0035 と同じ手順で A128 を作り直し（白色化と LDA の重みを 0035 と一致させるため全 3,737 枚を使う）、
   Bayer 230 枚の全 patch と脂肪変性 3 試験の全 patch を、0035 の split_record の重心に割り当てる（細 876、粗 = k=100 の親）。
2. slide ごとにグリッドの近傍（1 周目 8 個、2 周目 16 個）を引く。比較用に slide 内で割当をシャッフルした版も作る。
3. 解析: (a) 共起行列と観測/期待（シャッフル）の比、(b) プロトタイプ内の文脈の分かれ方（粗のヒストグラムで k-means、実際とシャッフルの R²）、
   (c) 中心静脈・門脈域のプロトタイプまでの距離と小葉内の位置の指標、脂肪変性の純プロトタイプ（CCl4 と cholesterol 食）で比べる、(d) 地図と近傍のモザイク。
"""
import argparse
import importlib.util
import json
import logging
import os
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


FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#999999"}
RING1 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
RING2 = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if max(abs(dx), abs(dy)) == 2]


def grid_neighbors(xy, s0, offsets):
    """slide 内のグリッド近傍。xy は level-0 の左上座標、s0 は patch の一辺（level-0 画素）。戻り値 (n, len(offsets)) の局所番号、なければ -1。"""
    g = np.round((xy - xy.min(0)) / s0).astype(np.int64)
    W = int(g[:, 1].max()) + 5
    key = (g[:, 0] + 2) * W + (g[:, 1] + 2)
    order = np.argsort(key)
    sk = key[order]
    out = np.full((len(xy), len(offsets)), -1, dtype=np.int64)
    for k, (dx, dy) in enumerate(offsets):
        k2 = (g[:, 0] + 2 + dx) * W + (g[:, 1] + 2 + dy)
        pos = np.clip(np.searchsorted(sk, k2), 0, len(sk) - 1)
        out[:, k] = np.where(sk[pos] == k2, order[pos], -1)
    resid = float(np.abs((xy - xy.min(0)) / s0 - g).max())
    return out, resid


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")
    args = parse_args()
    run_dir = get_run_dir(project_root, __file__, args.variant, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    config = load_config(Path(__file__).parent)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=args.variant)
    logger.info(f"Starting: {exp_name} / {args.variant}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    import torch
    cfg = {**config, **config.get("variants", {}).get(args.variant, {})}
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(cfg["seed"])
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex26 = load_module(project_root / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(project_root / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    ex30 = load_module(project_root / "experiments" / cfg["exp0030"] / "experiment.py", "exp0030")
    cfg35 = yaml.safe_load(open(project_root / "experiments" / cfg["exp0035"] / "config.yml"))
    cfg20 = yaml.safe_load(open(project_root / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    R35 = project_root / "outputs" / cfg["exp0035"] / cfg["exp0035_run"]
    TREE = R35 / f"tree_{cfg['rep']}_{cfg['method']}"
    pre_path = str(project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl")

    def h5_of(sid):
        for d_ in cfg["features_dirs"]:
            if (project_root / d_ / f"{sid}.h5").exists():
                return str(project_root / d_ / f"{sid}.h5")
        return ""

    # 1. 0035 と同じ並びの全 patch で A128 を作り直す（白色化と LDA の重み fac を 0035 と一致させるため）
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Kb, laba, row_b, xy_b, size0_b = c["K"], c["lab"].astype(np.int64), c["row"], c["xy"], c["size0"]
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    keep = [k_ for k_ in np.unique(laba) if k_ >= 0 and (laba == k_).sum() >= cfg20["min_label_patches"]]
    lab = np.where(np.isin(laba, keep), np.searchsorted(keep, np.maximum(laba, 0)), -1)
    ex_ids = pd.read_csv(project_root / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab = np.where(np.isin(bayer.slide_id.to_numpy()[row_b], ex_ids), -1, lab)
    nb = len(Kb)
    tgt = ex26.target_slides({**cfg, "target_exps": cfg["fatty_exps"], "features_dir": cfg["features_dirs"][0]}, project_root)
    tgt = tgt[~tgt.slide_id.isin(bayer.slide_id)].reset_index(drop=True)
    # 2026-11-04: メモリを抑えるため、全 patch をまとめて持たず、集合ごとに白色化・LDA をかけて標準偏差の和だけを集める（0035 と同じ値になる）
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        Kt_s = np.concatenate([p_[0] for p_ in ex.map(ex30._load_sub, tgt.h5, chunksize=4)]).astype(np.float16)
    S = pd.read_csv(R35 / "slides.csv", dtype={"slide_id": str})
    nt_s = len(Kt_s)
    assert nt_s == len(tgt) * cfg["patches_per_slide"]
    exp_b = bayer.EXP_ID.fillna("-1").astype(int).to_numpy()[row_b]
    exp_ts = np.repeat(tgt.exp.to_numpy(), cfg["patches_per_slide"])
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[q.EXP_ID.notna() & (q.EXP_ID != "")].copy()
    qc = q[(q.DOSE_LEVEL == "Control") & q.EXP_ID.astype(int).isin(set(exp_b) | set(exp_ts))].copy()
    qc["h5"] = [h5_of(x) for x in qc.slide_id]
    qc = qc[qc.h5 != ""]
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts_c = list(ex.map(ex27._load_ctrl, qc.h5, chunksize=4))
    Kc = np.concatenate(parts_c)[:, :128].astype(np.float32)
    exp_c = np.concatenate([np.full(len(p_), int(e)) for p_, e in zip(parts_c, qc.EXP_ID)])
    del parts_c

    def sphere_known(U, e):
        e_fix = np.where(np.isin(e, exp_c), e, -999)
        return ex27.sphering(U, e_fix, np.r_[Kc, Kc], np.r_[exp_c, np.full(len(exp_c), -999)]).astype(np.float32)

    tf = ex26.fit_combo(Kb, lab, cfg, cfg["seed"])
    L = len(keep) - 1

    def lda(K_):
        return np.concatenate([tf(b_)[:, :L] for b_ in np.array_split(K_, max(1, len(K_) // 200000))]).astype(np.float32)
    acc = {"n": 0, "s_sw": 0.0, "q_sw": 0.0, "s_l": 0.0, "q_l": 0.0}

    def add(Sw_, Lw_):
        acc["n"] += len(Sw_)
        acc["s_sw"] = acc["s_sw"] + Sw_.astype(np.float64).sum(0); acc["q_sw"] = acc["q_sw"] + (Sw_.astype(np.float64) ** 2).sum(0)
        acc["s_l"] = acc["s_l"] + Lw_.astype(np.float64).sum(0); acc["q_l"] = acc["q_l"] + (Lw_.astype(np.float64) ** 2).sum(0)
    Sw_b, Lw_b = sphere_known(Kb[:, :128].astype(np.float32), exp_b), lda(Kb)
    add(Sw_b, Lw_b)
    add(sphere_known(Kt_s[:, :128].astype(np.float32), exp_ts), lda(Kt_s))
    del Kt_s
    z = np.load(project_root / cfg35["unseen_features_from"], allow_pickle=True)
    Ku, row_u = z["K"], z["row"]
    del z
    sl_u = S[S.set == "unseen"].reset_index(drop=True)
    exp_u = sl_u.exp.to_numpy()[row_u]
    cm = sl_u.is_ctrl.to_numpy()[row_u]
    Uu = Ku[:, :128].astype(np.float32)
    e_fix = np.where(np.isin(exp_u, exp_u[cm]), exp_u, -999)
    Sw_u = ex27.sphering(Uu, e_fix, np.r_[Uu[cm], Uu[cm]], np.r_[exp_u[cm], np.full(int(cm.sum()), -999)])
    del Uu
    add(Sw_u, lda(Ku))
    del Sw_u, Ku
    N = acc["n"]
    sd = lambda s, q_: np.sqrt(np.maximum(q_ / N - (s / N) ** 2, 0)).mean()
    fac = float(sd(acc["s_sw"], acc["q_sw"]) / sd(acc["s_l"], acc["q_l"]))
    Xb = np.concatenate([Sw_b, cfg["lda_beta"] * fac * Lw_b], 1).astype(np.float32)
    del Kb
    del Sw_b, Lw_b
    (run_dir / "a128_scale.json").write_text(json.dumps({"fac": fac, "lda_beta": cfg["lda_beta"]}))
    logger.info(f"A128 rebuilt for {N:,} patches; fac {fac:.4f}; {time.time() - t0:.0f}s")

    # 2. 割り当て: 0035 の split_record の重心（細）と k=100 の親（粗）
    cen = torch.from_numpy(np.load(TREE / "centroids.npy").astype(np.float32)).to(device)
    Kf = cen.shape[0]
    par = pd.read_csv(TREE / "a_history" / "leaves.csv").set_index("prototype").parent_k100.reindex(range(Kf)).to_numpy().astype(np.int64)
    Kc_ = int(par.max()) + 1

    def assign(X):
        return torch.cat([torch.cdist(torch.from_numpy(x).to(device), cen).argmin(1) for x in np.array_split(X, max(1, len(X) // 65536))]).cpu().numpy()
    qb = assign(Xb)
    keep_X = bool(cfg.get("lobular"))   # v3: 小葉内の位置の回帰に表現を使う
    Xb = Xb.astype(np.float16) if keep_X else None
    Xt_parts = []
    cl35 = np.load(R35 / f"assign_{cfg['rep']}_{cfg['method']}.npy").astype(np.int64)[:nb]
    agree = float((qb == cl35).mean())
    logger.info(f"Bayer: nearest-centroid vs 0035 assignment agreement {agree:.3f}; {time.time() - t0:.0f}s")
    # 脂肪変性の 3 試験: 全 patch を同じ手順で写して割り当てる
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex26._init_worker, initargs=(pre_path,)) as ex:
        parts = list(ex.map(ex26._load_target, tgt.h5, chunksize=2))
    qt, xy_t, row_t = [], [], []
    for i, p_ in enumerate(parts):
        K_ = p_["K"].astype(np.float32)
        e = int(tgt.exp[i])
        Sw_ = sphere_known(K_[:, :128], np.full(len(K_), e))
        X_ = np.concatenate([Sw_, cfg["lda_beta"] * fac * tf(K_)[:, :L]], 1).astype(np.float32)
        qt.append(assign(X_))
        if keep_X:
            Xt_parts.append(X_.astype(np.float16))
        xy_t.append(p_["xy"])
        row_t.append(np.full(len(K_), i))
    size0_t = np.array([p_["size0"] for p_ in parts])
    del parts
    qt, xy_t, row_t = np.concatenate(qt), np.concatenate(xy_t), np.concatenate(row_t)
    logger.info(f"fatty: {len(tgt)} slides, {len(qt):,} full patches assigned; {time.time() - t0:.0f}s")

    # 3. 解析する集合（Bayer の全 patch ＋ 脂肪変性の全 patch）
    sl = pd.concat([bayer.assign(set="bayer", exp=bayer.EXP_ID.fillna("-1").astype(int), svs=bayer.svs_path, size0=size0_b, is_ctrl=bayer.DOSE_LEVEL.eq("Control"), compound=bayer.COMPOUND_NAME)[["slide_id", "set", "exp", "svs", "size0", "is_ctrl", "compound"]],
                    tgt.assign(set="fatty", svs=tgt.svs_path, size0=size0_t, is_ctrl=tgt.dose.eq("Control"))[["slide_id", "set", "exp", "svs", "size0", "is_ctrl", "compound"]]], ignore_index=True)
    G, topo = ex26.fat_grades({**cfg, "fat_findings": ["Degeneration, fatty", "Vacuolization, cytoplasmic", "Deposit, lipid", "Degeneration, vacuolar"], "pathology_csv": cfg35["pathology_csv"], "grade_map": cfg35["grade_map"]},
                              project_root, pd.concat([bayer.rename(columns={"EXP_ID": "EXP_ID"}), tgt], ignore_index=True)[["slide_id"]].merge(q[["slide_id", "EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]], on="slide_id", how="left"))
    sl["fatty_grade"] = G["Degeneration, fatty"].reindex(sl.slide_id).fillna(0).to_numpy()
    sl["fatty_topo"] = (topo["Degeneration, fatty"].reindex(sl.slide_id).fillna("").to_numpy() if "Degeneration, fatty" in topo else "")
    sl["cmpd"] = np.where(sl.compound.fillna("").str.contains("tetrachloride", case=False), "CCl4", np.where(sl.compound.fillna("").str.contains("cholesterol", case=False), "cholesterol", "other"))
    qf = np.r_[qb, qt]
    row = np.r_[row_b, row_t + len(bayer)]
    xy = np.r_[xy_b, xy_t]
    n_all = len(qf)
    qc_ = par[qf]
    sl.to_csv(run_dir / "slides.csv", index=False)
    logger.info(f"analysis set: {len(sl)} slides, {n_all:,} patches; {time.time() - t0:.0f}s")

    # 4. 近傍とシャッフル（slide ごと）
    nb1 = np.full((n_all, 8), -1, dtype=np.int64)
    nb2 = np.full((n_all, 16), -1, dtype=np.int64)
    q_sh = np.empty_like(qf)
    resid = []
    starts = np.r_[0, np.cumsum(np.bincount(row, minlength=len(sl)))]
    assert (np.diff(row) >= 0).all(), "patches must be grouped by slide"
    for i in range(len(sl)):
        a, b = starts[i], starts[i + 1]
        if b - a == 0:
            continue
        n1, r1 = grid_neighbors(xy[a:b], int(sl.size0[i]), RING1)
        n2, _ = grid_neighbors(xy[a:b], int(sl.size0[i]), RING2)
        nb1[a:b] = np.where(n1 >= 0, n1 + a, -1)
        nb2[a:b] = np.where(n2 >= 0, n2 + a, -1)
        q_sh[a:b] = rng.permutation(qf[a:b])
        resid.append(r1)
    sup1 = (nb1 >= 0).sum(1)
    sup2 = (nb2 >= 0).sum(1)
    valid = sup1 >= cfg["min_support_ring1"]
    pd.DataFrame({"set": sl.set.to_numpy()[row], "support_ring1": sup1, "support_ring2": sup2}).groupby(["set", "support_ring1"]).size().rename("n").reset_index().to_csv(run_dir / "neighbor_support.csv", index=False)
    np.savez(run_dir / "arrays.npz", qf=qf.astype(np.int16), q_sh=q_sh.astype(np.int16), row=row.astype(np.int32), xy=xy.astype(np.int32), nb1=nb1.astype(np.int32), nb2=nb2.astype(np.int32))   # 2026-11-04: 解析を作り直しなしで回すため
    logger.info(f"neighbors: max grid residual {max(resid):.3f} patch; ring1 support 8 for {np.mean(sup1 == 8):.3f}, >= {cfg['min_support_ring1']} for {valid.mean():.3f}; {time.time() - t0:.0f}s")

    # 5. (a) 共起行列（1 周目）: 実際とシャッフル。観測/期待
    def cooc(lbl, nbm, K):
        ci = np.repeat(np.arange(n_all), nbm.shape[1])
        nj = nbm.ravel()
        ok = (nj >= 0) & valid[ci]
        return np.bincount(lbl[ci[ok]] * K + lbl[nj[ok]], minlength=K * K).reshape(K, K).astype(np.float64)
    out = {}
    for nm_, lab_, K_ in [("fine", qf, Kf), ("coarse", qc_, Kc_)]:
        lab_sh = q_sh if nm_ == "fine" else par[q_sh]
        Mr, Ms = cooc(lab_, nb1, K_), cooc(lab_sh, nb1, K_)
        np.save(run_dir / f"cooc_ring1_{nm_}_real.npy", Mr)
        np.save(run_dir / f"cooc_ring1_{nm_}_shuffled.npy", Ms)
        Pr = Mr / np.maximum(Mr.sum(1, keepdims=True), 1)
        Ps = Ms / np.maximum(Ms.sum(1, keepdims=True), 1)
        out[nm_] = (Mr, Ms, Pr, Ps)
    Mr, Ms, Pr, Ps = out["fine"]
    n_f = np.bincount(qf[valid], minlength=Kf)
    self_tab = pd.DataFrame({"prototype": np.arange(Kf), "parent_k100": par, "n_valid": n_f, "self_share_real": np.diag(Pr), "self_share_shuffled": np.diag(Ps),
                             "same_parent_share_real": [Pr[a, par == par[a]].sum() for a in range(Kf)], "same_parent_share_shuffled": [Ps[a, par == par[a]].sum() for a in range(Kf)]})
    logger.info(f"cooc: median self share real {self_tab.self_share_real[n_f >= 300].median():.3f} vs shuffled {self_tab.self_share_shuffled[n_f >= 300].median():.3f}; {time.time() - t0:.0f}s")

    from sklearn.metrics import roc_auc_score
    # 6. (c) 中心静脈・門脈域までの距離（patch 単位）と小葉内の位置の指標 z = d_cv / (d_cv + d_pt)（0 = 中心静脈側、1 = 門脈域側）
    from scipy.spatial import cKDTree
    names = pd.read_csv(R35 / f"prototype_names_{cfg['rep']}_{cfg['method']}.csv").set_index("prototype").bayer_name.reindex(range(Kf)).fillna("").to_numpy()
    zc = cfg["zonation"]
    is_cv, is_pt = names == zc["cv_name"], names == zc["pt_name"]
    logger.info(f"zonation landmarks: {int(is_cv.sum())} central-vein prototypes {list(np.where(is_cv)[0])}, {int(is_pt.sum())} portal-tract prototypes {list(np.where(is_pt)[0])}")

    def landmark_masks(lbl):
        """patch ごとの目印（中心静脈・門脈域）。min_cluster > 0 なら、同じ種類の目印が 1 周目に min_cluster 個以上ある patch だけ（血管の断面のまとまり）"""
        out_m = []
        for msk in (is_cv, is_pt):
            m_ = msk[lbl]
            if zc.get("min_cluster", 0) > 0:
                okn = nb1 >= 0
                cnt = (okn & m_[np.where(okn, nb1, 0)]).sum(1)
                m_ = m_ & (cnt >= zc["min_cluster"])
            out_m.append(m_)
        return out_m

    def zonation(lbl):
        m_cv, m_pt = landmark_masks(lbl)
        d_cv, d_pt = np.full(n_all, np.nan), np.full(n_all, np.nan)
        for i in range(len(sl)):
            a, b = starts[i], starts[i + 1]
            g = (xy[a:b] - xy[a:b].min(0)) / int(sl.size0[i])
            for mm, d in ((m_cv, d_cv), (m_pt, d_pt)):
                own = mm[a:b]
                pts = g[own]
                if len(pts) >= 3:
                    dd, _ = cKDTree(pts).query(g, k=2)
                    d[a:b] = np.where(own, dd[:, 1], dd[:, 0])   # 自分自身が目印のときは次に近い目印
        d_cv[d_cv > zc["max_dist_patches"]] = np.nan
        d_pt[d_pt > zc["max_dist_patches"]] = np.nan
        return d_cv, d_pt, d_cv / (d_cv + d_pt)
    d_cv, d_pt, zr = zonation(qf)
    _, _, zs = zonation(q_sh)
    logger.info(f"zonation: index defined for {np.isfinite(zr).mean():.3f} of patches; {time.time() - t0:.0f}s")

    # v3: 小葉内の位置を回帰で推定（0023 の考え方）。正解 = Bayer の中心静脈・門脈域ポリゴンからの距離の比、特徴 = A128 ＋ 1 周目・2 周目の平均。
    #     試験ごとに分けた 5 分割で Bayer を評価し、全 Bayer で学んだモデルで脂肪変性の slide を予測する。slide 内の順位（0 = 中心静脈側）にして使う
    lab_b_names = np.array([cfg20["label_priority"][k] if k >= 0 else "" for k in laba])
    zl = None
    if cfg.get("lobular"):
        from scipy.stats import spearmanr
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import GroupKFold
        ex21 = sys.modules["exp0021"]
        Xall = np.concatenate([Xb] + Xt_parts)
        del Xb, Xt_parts

        # v5: 周囲の平均から、脂肪変性・空胞化の記録名が付いたプロトタイプの patch を除く（周りも脂肪変性なので、その見た目を読んでいないかの確かめ）
        excl_rec = cfg["lobular"].get("exclude_neighbor_records", [])
        rec_names = pd.read_csv(R35 / f"prototype_names_{cfg['rep']}_{cfg['method']}.csv").set_index("prototype").record_name.reindex(range(Kf)).fillna("").to_numpy()
        excl_p = np.isin(qf, np.where(np.isin(rec_names, excl_rec))[0]) if excl_rec else np.zeros(n_all, bool)
        nb_support = np.zeros(n_all, dtype=np.int32)

        def nbmean(nbm):
            out_ = np.zeros_like(Xall)
            for s in range(0, n_all, 500000):
                nn = nbm[s:s + 500000]
                ok = (nn >= 0) & ~excl_p[np.where(nn >= 0, nn, 0)]
                nb_support[s:s + 500000] += ok.sum(1)
                acc_ = np.zeros((len(nn), Xall.shape[1]), dtype=np.float32)
                for k in range(nn.shape[1]):
                    acc_ += np.where(ok[:, k:k + 1], Xall[np.where(ok[:, k], nn[:, k], 0)].astype(np.float32), 0)
                out_[s:s + 500000] = (acc_ / np.maximum(ok.sum(1, keepdims=True), 1)).astype(np.float16)
            return out_
        feats_ = cfg["lobular"].get("features", ["self", "ring1", "ring2"])   # v4: 中心の patch を除いた版（周囲だけ）
        Fm = ([Xall] if "self" in feats_ else []) + ([nbmean(nb1)] if "ring1" in feats_ else []) + ([nbmean(nb2)] if "ring2" in feats_ else [])
        pos = ex21.lobular_position(bayer, row_b, xy_b, dict(zip(bayer.slide_id, size0_b)), project_root, cfg["lobular"]["max_dist"])
        pm = np.isfinite(pos)
        Ftr = np.concatenate([f_[:nb][pm] for f_ in Fm], 1).astype(np.float32)
        exp_tr = sl.exp.to_numpy()[row_b[pm]]
        oof = np.full(pm.sum(), np.nan)
        for tr, te in GroupKFold(n_splits=cfg["lobular"]["folds"]).split(Ftr, groups=exp_tr):
            oof[te] = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(Ftr[tr], pos[pm][tr]).predict(Ftr[te])
        r2 = 1 - ((pos[pm] - oof) ** 2).sum() / ((pos[pm] - pos[pm].mean()) ** 2).sum()
        rho = spearmanr(pos[pm], oof)[0]
        model = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(Ftr, pos[pm])
        pred = np.concatenate([model.predict(np.concatenate([f_[s:s + 500000] for f_ in Fm], 1).astype(np.float32)) for s in range(0, n_all, 500000)])
        del Fm, Ftr, Xall
        zl = np.empty(n_all)
        if excl_rec:
            pred = np.where(nb_support >= 3, pred, np.nan)   # 除いた後に周囲が 3 個未満なら位置なし
            logger.info(f"lobular: excluded {excl_p.mean():.3f} of patches as neighbours ({excl_rec}); position undefined for {np.mean(nb_support < 3):.3f}")
        for i in range(len(sl)):
            a, b = starts[i], starts[i + 1]
            if b > a:
                zl[a:b] = pd.Series(pred[a:b]).rank(pct=True).to_numpy()   # NaN は NaN のまま
        # 目印の確かめ（試験を分けた予測ではないので、Bayer は学習に含まれる。参考）
        mk_ = np.isin(lab_b_names, ["central_vein", "portal_tract"])
        mk_ = mk_ & np.isfinite(zl[:nb])
        auc_l = float(roc_auc_score(lab_b_names[mk_] == "portal_tract", zl[:nb][mk_]))
        lob = {"n_target_patches": int(pm.sum()), "n_target_slides": int(len(np.unique(row_b[pm]))), "oof_r2_leave_experiment_out": float(r2), "oof_spearman": float(rho),
               "auc_portal_vs_central_labels_in_sample": auc_l}
        (run_dir / "lobular_model.json").write_text(json.dumps(lob, indent=2))
        np.save(run_dir / "lobular_rank.npy", zl.astype(np.float32))
        logger.info(f"lobular model: {lob}; {time.time() - t0:.0f}s")

    # 純プロトタイプ（0035 の目視で確かめたもの）での比較: 同じ化合物の slide の中で、純プロトタイプの patch とそれ以外の patch の z
    pure = pd.read_csv(R35 / "fatty_figures" / "fatty_pure_prototypes.csv")
    pure = pure[(pure.rep == cfg["rep"]) & (pure.method == cfg["method"])]
    pure_sets = {cn: set(pure.prototype[pure.compound == cn]) for cn in ("CCl4", "cholesterol")}
    cmpd_p = sl.cmpd.to_numpy()[row]
    fat_p = sl.fatty_grade.to_numpy()[row] > 0
    zrows = []
    for cn, ps in pure_sets.items():
        inpure = np.isin(qf, list(ps))
        for grp, msk in [("pure prototypes", inpure), ("other patches, fatty slides", ~inpure)]:
            sel = (cmpd_p == cn) & fat_p & msk
            for kind, zz in [("real", zr), ("shuffled", zs)] + ([("lobular_model", zl)] if zl is not None else []):
                ok = sel & np.isfinite(zz)
                per_slide = pd.Series(zz[ok]).groupby(row[ok]).mean()
                per_slide = per_slide[pd.Series(ok).groupby(row).sum().reindex(per_slide.index) >= 50]
                bs = [per_slide.sample(len(per_slide), replace=True, random_state=s).mean() for s in range(cfg["bootstrap"])] if len(per_slide) > 1 else [np.nan]
                zrows.append({"compound": cn, "group": grp, "kind": kind, "n_patches": int(ok.sum()), "n_slides": int(len(per_slide)), "z_mean_of_slides": float(per_slide.mean()) if len(per_slide) else np.nan,
                              "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)), "z_median_patches": float(np.median(zz[ok])) if ok.any() else np.nan})
    ztab = pd.DataFrame(zrows)
    # slide ごとの差（純プロトタイプ − 同じ slide のその他）: 記録どおりの向き（CCl4 は負、cholesterol 食は正）の slide の割合
    prow = []
    for kind, zz in [("real", zr)] + ([("lobular_model", zl)] if zl is not None else []):
        for cn, ps in pure_sets.items():
            inpure = np.isin(qf, list(ps))
            base_ = (cmpd_p == cn) & fat_p & np.isfinite(zz)
            a_ = pd.Series(zz[base_ & inpure]).groupby(row[base_ & inpure]).agg(["mean", "size"])
            b_ = pd.Series(zz[base_ & ~inpure]).groupby(row[base_ & ~inpure]).mean()
            a_ = a_[a_["size"] >= 50]
            d_ = (a_["mean"] - b_.reindex(a_.index)).dropna()
            want = -1 if cn == "CCl4" else 1
            bs = [d_.sample(len(d_), replace=True, random_state=s).mean() for s in range(cfg["bootstrap"])] if len(d_) > 1 else [np.nan]
            prow.append({"kind": kind, "compound": cn, "n_slides": int(len(d_)), "mean_diff": float(d_.mean()) if len(d_) else np.nan,
                         "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)), "share_slides_expected_direction": float((np.sign(d_) == want).mean()) if len(d_) else np.nan})
    ptab = pd.DataFrame(prow)
    ptab.to_csv(run_dir / "zonation_paired_by_slide.csv", index=False)
    logger.info("zonation paired by slide (pure - other):\n" + ptab.round(3).to_string(index=False))
    ztab.to_csv(run_dir / "zonation_fatty.csv", index=False)
    logger.info("zonation of fatty pure prototypes:\n" + ztab.round(3).to_string(index=False))
    # 目印そのものの確かめ: Bayer の中心静脈・門脈域のラベルが付いた patch の z（目印は同じラベルで名前を付けたプロトタイプなので循環を含む）
    zcheck = pd.DataFrame([{"bayer_label": nm_, "n": int(np.isfinite(zr[:nb][lab_b_names == nm_]).sum()), "z_median": float(np.nanmedian(zr[:nb][lab_b_names == nm_]))} for nm_ in ["central_vein", "portal_tract", "parenchyma", "necrosis", "inflammation"]])
    mk_ = np.isin(lab_b_names, ["central_vein", "portal_tract"]) & np.isfinite(zr[:nb])
    auc_lm = float(roc_auc_score(lab_b_names[mk_] == "portal_tract", zr[:nb][mk_])) if mk_.any() else np.nan
    zcheck = pd.concat([zcheck, pd.DataFrame([{"bayer_label": "AUC portal vs central (z)", "n": int(mk_.sum()), "z_median": auc_lm}])], ignore_index=True)
    zcheck.to_csv(run_dir / "zonation_landmark_check.csv", index=False)
    logger.info("zonation landmark check:\n" + zcheck.round(3).to_string(index=False))

    # 7. (b) プロトタイプ内の文脈の分かれ方: 粗（k=100）の 1・2 周目のヒストグラム（√）で k-means（k_sub）。実際とシャッフルの R²、子の正味の slide 枚数
    cs = cfg["context_split"]
    from lib.clustering import kmeans_lloyd

    def hist_coarse(idx, lbl_c, own_parent=None):
        H = np.zeros((len(idx), 2 * Kc_), dtype=np.float32)
        for off, nbm in ((0, nb1), (Kc_, nb2)):
            nn = nbm[idx]
            ok = nn >= 0
            rr = np.repeat(np.arange(len(idx)), nn.shape[1])[ok.ravel()]
            np.add.at(H, (rr, off + lbl_c[nn[ok]]), 1)
            if cs.get("exclude_own_parent") and own_parent is not None:
                H[:, off + own_parent] = 0   # 周囲の「自分の親」を数えない（病変の中心か縁かだけで分かれるのを避ける）
            H[:, off:off + Kc_] /= np.maximum(H[:, off:off + Kc_].sum(1, keepdims=True), 1)
        return np.sqrt(H)

    def r2_split(H, seed):
        lab_k, cen_k, _ = kmeans_lloyd(H, cs["k_sub"], seed, device)
        lab_k = np.asarray(lab_k)
        tot = ((H - H.mean(0)) ** 2).sum()
        within = sum(((H[lab_k == k] - H[lab_k == k].mean(0)) ** 2).sum() for k in range(cs["k_sub"]) if (lab_k == k).any())
        return 1 - within / max(tot, 1e-9), lab_k
    srows, sub_assign = [], {}
    qc_sh = par[q_sh]
    w_slide = 1.0 / np.maximum(np.bincount(row, minlength=len(sl)), 1)
    for a in range(Kf):
        idx = np.where((qf == a) & valid)[0]
        if len(idx) < cs["min_patches"]:
            continue
        if len(idx) > cs["max_sample"]:
            idx = np.sort(rng.choice(idx, cs["max_sample"], replace=False))
        r2r, lab_k = r2_split(hist_coarse(idx, qc_, par[a]), cfg["seed"] + a)
        r2s, _ = r2_split(hist_coarse(idx, qc_sh, par[a]), cfg["seed"] + a)
        effs, sizes, exps_ = [], [], []
        for k in range(cs["k_sub"]):
            rk = row[idx[lab_k == k]]
            if len(rk) == 0:
                effs.append(0); sizes.append(0); exps_.append(0)
                continue
            w = np.bincount(rk, minlength=len(sl)) * w_slide
            p = w / w.sum()
            effs.append(1 / (p ** 2).sum())
            sizes.append(len(rk) / len(idx))
            pe = pd.Series(w).groupby(sl.exp.to_numpy()).sum()
            pe = pe / pe.sum()
            exps_.append(1 / (pe ** 2).sum())
        sub_assign[a] = (idx, lab_k)
        srows.append({"prototype": a, "parent_k100": int(par[a]), "n_valid": int(n_f[a]), "r2_real": r2r, "r2_shuffled": r2s, "r2_gain": r2r - r2s,
                      "min_sub_size": float(min(sizes)), "min_sub_eff_slides": float(min(effs)), "min_sub_eff_experiments": float(min(exps_)),
                      "record_name": "", "bayer_name": names[a]})
    stab = pd.DataFrame(srows)
    rec = pd.read_csv(R35 / f"prototype_names_{cfg['rep']}_{cfg['method']}.csv").set_index("prototype").record_name
    stab["record_name"] = rec.reindex(stab.prototype).fillna("").to_numpy()
    stab["candidate"] = (stab.r2_gain >= cs.get("min_r2_gain", 0)) & (stab.min_sub_size >= 0.1) & (stab.min_sub_eff_slides >= cs["min_eff_slides_sub"]) & (stab.min_sub_eff_experiments >= cs.get("min_eff_experiments_sub", 0))
    stab = stab.merge(self_tab[["prototype", "self_share_real", "self_share_shuffled"]], on="prototype")
    stab.sort_values("r2_gain", ascending=False).to_csv(run_dir / "context_split.csv", index=False)
    logger.info(f"context split: {len(stab)} prototypes analysed, median R2 real {stab.r2_real.median():.3f} vs shuffled {stab.r2_shuffled.median():.3f}, candidates {int(stab.candidate.sum())}; {time.time() - t0:.0f}s")

    # ── 図 ──────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from scipy.cluster.hierarchy import dendrogram, linkage
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)

    # F1: 近傍の有効数（集合ごと）
    sup_tab = pd.read_csv(run_dir / "neighbor_support.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.2), layout="constrained")
    for k_, (st, col, mk) in enumerate([("bayer", OI["blue"], "o"), ("fatty", OI["orange"], "s")]):
        t_ = sup_tab[sup_tab.set == st].set_index("support_ring1").n.reindex(range(9)).fillna(0)
        ax.plot(t_.index, t_ / t_.sum(), marker=mk, color=col, label={"bayer": "Bayer の 230 枚", "fatty": "脂肪変性の 3 試験 188 枚"}[st])
    ax.axvline(cfg["min_support_ring1"] - 0.5, color="black", lw=0.8, ls="--")
    ax.text(cfg["min_support_ring1"] - 0.4, 0.5, f"解析に使う（{cfg['min_support_ring1']} 個以上）", fontsize=8)
    ax.set(xlabel="1 周目（3×3）の有効な近傍の数", ylabel="patch の割合", title="近傍の有効数（組織の端・血管腔で欠ける）")
    ax.legend(frameon=False)
    fig.savefig(figs / "F1_neighbor_support.png", dpi=200)
    plt.close(fig)

    # F2: 粗（k=100）の共起の観測/期待（log2）。行・列は観測/期待のプロファイルで並べ替え
    Mr_c, Ms_c, Pr_c, Ps_c = out["coarse"]
    ok_c = Mr_c.sum(1) >= 2000
    E = np.log2((Pr_c[np.ix_(ok_c, ok_c)] + 1e-4) / (Ps_c[np.ix_(ok_c, ok_c)] + 1e-4))
    ordc = dendrogram(linkage(E, method="average", metric="correlation"), no_plot=True)["leaves"]
    ids_c = np.where(ok_c)[0][ordc]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6), layout="constrained", gridspec_kw={"width_ratios": [3, 2]})
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    im = axes[0].imshow(np.clip(E[np.ix_(ordc, ordc)], -3, 3), cmap="RdBu_r", norm=norm, interpolation="nearest")
    axes[0].set(title=f"粗プロトタイプ（k=100 の親、{len(ids_c)} 個）の 3×3 近傍の共起: 観測/期待（log2）\n期待 = slide 内でシャッフルした割当。赤 = 隣り合いやすい", xlabel="近傍のプロトタイプ（並べ替え後）", ylabel="中心のプロトタイプ（並べ替え後）")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], shrink=0.7, label="log2（観測 / 期待）")
    ok_f = n_f >= 300
    axes[1].scatter(self_tab.self_share_shuffled[ok_f], self_tab.self_share_real[ok_f], s=10, color=OI["blue"], alpha=0.6, label="細プロトタイプ（有効な patch 300 以上）")
    lim = [0, max(self_tab.self_share_real[ok_f].max(), 0.1) * 1.05]
    axes[1].plot(lim, lim, color="black", lw=0.8, ls="--", label="差なし（y = x）")
    axes[1].set(xlim=lim, ylim=lim, xlabel="シャッフル後: 3×3 近傍のうち同じプロトタイプの割合", ylabel="実際: 3×3 近傍のうち同じプロトタイプの割合",
                title="プロトタイプは空間的に固まるか\n（線より上 = 実際の方が隣に同じものが多い）")
    axes[1].legend(frameon=False, loc="upper left")
    fig.savefig(figs / "F2_cooccurrence.png", dpi=170)
    plt.close(fig)

    # F3: 文脈の分かれ方（R² 実際 vs シャッフル）
    fig, ax = plt.subplots(figsize=(7, 5.6), layout="constrained")
    grp_col = {"Degeneration, fatty": OI["orange"], "Necrosis": OI["vermillion"], "Single cell necrosis": OI["vermillion"], "Cellular infiltration": OI["blue"]}
    for cand, mk, a_ in ((False, "o", 0.35), (True, "D", 0.9)):
        t_ = stab[stab.candidate == cand]
        ax.scatter(t_.r2_shuffled, t_.r2_real, s=14 if not cand else 26, marker=mk, alpha=a_, c=[grp_col.get(r_, OI["gray"]) for r_ in t_.record_name], edgecolors="none",
                   label="候補（子が 10% 以上・正味の slide 5 枚以上）" if cand else "その他")
    lim = [0, max(stab.r2_real.max(), stab.r2_shuffled.max()) * 1.05]
    ax.plot(lim, lim, color="black", lw=0.8, ls="--")
    for _, r_ in stab[stab.candidate].nlargest(cs["top_n_mosaic"], "r2_gain").iterrows():
        ax.annotate(f"p{int(r_.prototype):04d}", (r_.r2_shuffled, r_.r2_real), fontsize=7, xytext=(3, 3), textcoords="offset points")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=OI["gray"], label="候補（子が 10% 以上・正味の slide 5 枚以上）"), Line2D([], [], marker="o", ls="", color=OI["gray"], alpha=0.4, label="その他"),
                       Line2D([], [], marker="s", ls="", color=OI["orange"], label="記録名: 脂肪変性"), Line2D([], [], marker="s", ls="", color=OI["vermillion"], label="記録名: 壊死・単細胞壊死"),
                       Line2D([], [], marker="s", ls="", color=OI["blue"], label="記録名: 細胞浸潤")], frameon=False, fontsize=8, loc="upper left")
    ax.set(xlim=lim, ylim=lim, xlabel="シャッフル後の近傍で分けたときの R²", ylabel="実際の近傍で分けたときの R²",
           title=f"同じプロトタイプの中に、周囲の違う群があるか（細プロトタイプ {len(stab)} 個、各 {cs['k_sub']} 群に分割）\n線より上 = 実際の周囲の方がはっきり分かれる")
    fig.savefig(figs / "F3_context_split.png", dpi=180)
    plt.close(fig)

    # F4: 小葉内の位置: 脂肪変性の純プロトタイプ vs 同じ slide のその他。指標ごとに（目印の z、目印をシャッフルした z、回帰による slide 内の順位）。slide 単位の平均と 95% 区間
    kinds = [("real", "目印までの距離の比 z", OI["blue"], "o"), ("shuffled", "目印をシャッフルした z（対照）", OI["gray"], "s")] + ([("lobular_model", "回帰による小葉内の位置（slide 内の順位）", OI["vermillion"], "D")] if zl is not None else [])
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ypos, labels_ = [], []
    yy = 0
    seen_k = set()
    for cn in ("CCl4", "cholesterol"):
        for grp in ("pure prototypes", "other patches, fatty slides"):
            y_start = yy
            for kind, klab, col, mk in kinds:
                r_ = ztab[(ztab.compound == cn) & (ztab.group == grp) & (ztab.kind == kind)].iloc[0]
                if np.isfinite(r_.z_mean_of_slides):
                    ax.errorbar(r_.z_mean_of_slides, yy, xerr=[[r_.z_mean_of_slides - r_.lo], [r_.hi - r_.z_mean_of_slides]], fmt=mk, color=col, ms=6, capsize=3,
                                label=klab if kind not in seen_k else None)
                    seen_k.add(kind)
                yy += 0.3
            n_s = int(ztab[(ztab.compound == cn) & (ztab.group == grp) & (ztab.kind == kinds[-1][0])].iloc[0].n_slides)
            labels_.append(f"{cn}: {'純プロトタイプ' if grp == 'pure prototypes' else 'その他の patch'}（{n_s} 枚）")
            ypos.append((y_start + yy - 0.3) / 2)
            yy += 0.5
    ax.set_yticks(ypos, labels_)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="小葉内の位置（0 = 中心静脈側、1 = 門脈域側）",
           title="脂肪変性の純プロトタイプは小葉のどこにあるか（記録: CCl4 は小葉中心性、cholesterol 食は辺縁性）\n点 = slide ごとの平均の平均、線 = slide を単位にした 95% ブートストラップ区間")
    ax.axvline(0.5, color="black", lw=0.6, ls=":")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.savefig(figs / "F4_zonation_fatty.png", dpi=180)
    plt.close(fig)
    logger.info(f"figures F1-F4; {time.time() - t0:.0f}s")

    # F5: 地図（原画像、脂肪変性の純プロトタイプの 5×5 近傍密度、小葉内の位置 z）
    import openslide
    from PIL import Image, ImageDraw, ImageFont
    fj, fjt = ImageFont.truetype(FONT_JP, 26), ImageFont.truetype(FONT_JP, 34)
    pure_all = np.isin(qf, list(pure_sets["CCl4"] | pure_sets["cholesterol"]))
    dens = np.zeros(n_all)
    for nbm in (nb1, nb2):
        okn = nbm >= 0
        dens += np.where(okn, pure_all[np.where(okn, nbm, 0)], 0).sum(1)
    dens = dens / np.maximum(sup1 + sup2, 1)
    pick_s = []
    for cn, g_ in (("CCl4", 3), ("cholesterol", 2), ("cholesterol", 4)):
        c_ = sl[(sl.cmpd == cn) & (sl.fatty_grade == g_) & (sl.set == "fatty")]
        if len(c_):
            pick_s.append(int(c_.index[0]))
    c_ = sl[(sl.set == "fatty") & sl.is_ctrl & (sl.exp == 67)]
    if len(c_):
        pick_s.append(int(c_.index[0]))
    pick_s += [int(sl.index[sl.slide_id == s_][0]) for s_ in cfg["maps"]["extra_slides"] if (sl.slide_id == s_).any()]
    cmap_d = matplotlib.colormaps["Oranges"]
    cmap_z = matplotlib.colormaps["RdBu"]
    for i in pick_s:
        r_ = sl.iloc[i]
        if not Path(str(r_.svs)).exists():
            continue
        a, b = starts[i], starts[i + 1]
        w = openslide.OpenSlide(r_.svs)
        th = w.get_thumbnail((cfg["maps"]["thumb"], cfg["maps"]["thumb"])).convert("RGB")
        scl = th.width / w.dimensions[0]
        w.close()
        x0, y0 = xy[a:b].min(0) * scl
        x1, y1 = (xy[a:b].max(0) + int(r_.size0)) * scl
        box = (int(max(0, x0 - 10)), int(max(0, y0 - 10)), int(min(th.width, x1 + 10)), int(min(th.height, y1 + 10)))
        th = th.crop(box)
        bpx = max(2, round(int(r_.size0) * scl))
        gray = th.convert("L").convert("RGBA")
        panels = [th]
        for vals, cmap_, vmin, vmax in ((dens[a:b], cmap_d, 0, 1), (zr[a:b], cmap_z, 0, 1)):
            ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
            d_ = ImageDraw.Draw(ov)
            for (px, py), v in zip(xy[a:b], vals):
                if not np.isfinite(v):
                    continue
                col = tuple(int(255 * c__) for c__ in cmap_((v - vmin) / (vmax - vmin))[:3])
                X0, Y0 = px * scl - box[0], py * scl - box[1]
                d_.rectangle([X0, Y0, X0 + bpx, Y0 + bpx], fill=col + (220,))
            panels.append(Image.alpha_composite(gray, ov).convert("RGB"))
        hdr, ph = 120, 44
        canvas = Image.new("RGB", (th.width * 3 + 20, hdr + ph + th.height), "white")
        dd = ImageDraw.Draw(canvas)
        dd.text((8, 6), f"slide {r_.slide_id}（{r_.compound}、exp {r_.exp}{'、対照群' if r_.is_ctrl else ''}）　脂肪変性 grade {int(r_.fatty_grade)} {r_.fatty_topo or ''}", fill="black", font=fjt)
        dd.text((8, 52), "中: 周囲 5×5（24 個）のうち脂肪変性の純プロトタイプ（0035 の目視で確かめた 10 個）の割合（白 0 → 濃い橙 1）。", fill="#333333", font=fj)
        dd.text((8, 84), "右: 小葉内の位置の指標 z（赤 = 中心静脈側 0、青 = 門脈域側 1）。目印は Bayer の中心静脈・門脈域の名前が付いたプロトタイプ。色なし = 目印が遠い。", fill="#333333", font=fj)
        for j_, (pn, tt) in enumerate(zip(panels, ["原画像（H&E）", "脂肪変性の純プロトタイプの近傍密度", "小葉内の位置 z"])):
            canvas.paste(pn, (j_ * (th.width + 10), hdr + ph))
            dd.text((j_ * (th.width + 10) + 8, hdr + 6), tt, fill="black", font=fj)
        canvas.save(figs / f"F5_map_slide{r_.slide_id}.jpg", quality=85)
    logger.info(f"F5 maps {len(pick_s)}; {time.time() - t0:.0f}s")

    # F6: 文脈の候補プロトタイプのモザイク（子ごとに、中心 patch と 3×3 の周囲を 4 例）
    top = stab[stab.candidate].nlargest(cs["top_n_mosaic"], "r2_gain")
    T = 240
    for _, r_ in top.iterrows():
        a = int(r_.prototype)
        idx, lab_k = sub_assign[a]
        blocks = []
        for k in range(cs["k_sub"]):
            ik = idx[lab_k == k]
            if len(ik) == 0:
                continue
            Hk = hist_coarse(ik, qc_, par[a]) ** 2
            top_par = np.argsort(-Hk[:, :Kc_].mean(0))[:3]
            desc = "、".join(f"親 {int(t_)}（{Hk[:, t_].mean():.0%}）" for t_ in top_par)
            picks, seen = [], set()
            for p_ in rng.permutation(ik):
                if row[p_] in seen or not Path(str(sl.svs[row[p_]])).exists():
                    continue
                picks.append(int(p_)); seen.add(row[p_])
                if len(picks) >= 4:
                    break
            blocks.append((k, len(ik), desc, picks))
        canvas = Image.new("RGB", (4 * T + 30, 90 + len(blocks) * (T + 60)), "white")
        dd = ImageDraw.Draw(canvas)
        dd.text((8, 6), f"p{a:04d}（記録名: {r_.record_name or 'なし'}、Bayer: {r_.bayer_name or 'なし'}）を周囲の構成で {cs['k_sub']} 群に分けた例", fill="black", font=fj)
        dd.text((8, 40), f"R² 実際 {r_.r2_real:.2f} / シャッフル {r_.r2_shuffled:.2f}。各タイルは中心 patch（枠）と周囲 3×3（約 384 µm 四方）。", fill="#333333", font=ImageFont.truetype(FONT_JP, 20))
        y = 90
        for k, nk, desc, picks in blocks:
            dd.text((8, y + 4), f"群 {k + 1}: {nk} patch。周囲に多い粗プロトタイプ: {desc}", fill="#B34700", font=ImageFont.truetype(FONT_JP, 20))
            for j_, p_ in enumerate(picks):
                s0 = int(sl.size0[row[p_]])
                w = openslide.OpenSlide(sl.svs[row[p_]])
                img = w.read_region((int(xy[p_, 0]) - s0, int(xy[p_, 1]) - s0), 0, (3 * s0, 3 * s0)).convert("RGB").resize((T, T))
                w.close()
                di = ImageDraw.Draw(img)
                di.rectangle([T // 3, T // 3, 2 * T // 3, 2 * T // 3], outline=(0, 0, 0), width=2)
                canvas.paste(img, (8 + j_ * (T + 5), y + 36))
            y += T + 60
        canvas.save(figs / f"F6_context_mosaic_p{a:04d}.jpg", quality=88)
    logger.info(f"F6 mosaics {len(top)}; {time.time() - t0:.0f}s")

    results = {"n_slides": int(len(sl)), "n_patches": int(n_all), "bayer_assignment_agreement": agree, "fac": fac,
               "ring1_full_support_share": float(np.mean(sup1 == 8)), "valid_share": float(valid.mean()),
               "self_share_median_real": float(self_tab.self_share_real[ok_f].median()), "self_share_median_shuffled": float(self_tab.self_share_shuffled[ok_f].median()),
               "context_split_n": int(len(stab)), "context_split_candidates": int(stab.candidate.sum()),
               "r2_median_real": float(stab.r2_real.median()), "r2_median_shuffled": float(stab.r2_shuffled.median()),
               "zonation": ztab.round(4).to_dict(orient="records"), "landmark_check": zcheck.round(4).to_dict(orient="records")}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    self_tab.to_csv(run_dir / "self_cooccurrence.csv", index=False)
    complete_run(run_dir)
    logger.info(f"Done. {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
