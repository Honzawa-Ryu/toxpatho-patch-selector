"""0035: プロトタイプを細かくして純度を上げ、つなぎ直す。設計は config.yml の冒頭と plan.md。

集合は Bayer 230枚（全patch）＋脂肪変性3試験＋0033の未学習3,319枚。表現は A と A128（0033/0034 と同じ作り方）。
方法: flat（k = 100〜1000）/ split（純度の低いクラスタだけ再クラスタリング）/ tree（k=1000 の葉を Ward 法でつなぎ直す。距離だけ、または最多ラベル・最多記録所見が同じときだけ）。
評価: (a) Bayer ラベルの held-out（F1、適合率>=thr のプロトタイプに入る割合）、(b) 未学習 slide の読み出し AUC、(c) 記録の純度（全 slide、脂肪変性を他の所見と一緒に）。
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


COL = {"A_sphering_lda": "#E69F00", "A128_sphering_lda": "#0072B2"}   # Okabe-Ito


# ---------------- クラスタリングの部品 ----------------

def assign(cen, X, device, with_dist=False):
    """重心 cen（torch、device上）に X（numpy）を割り当てる。with_dist で重心までの距離も返す。"""
    import torch
    a, d = [], []
    for x in np.array_split(X, max(1, len(X) // 65536)):
        D = torch.cdist(torch.from_numpy(x).to(device), cen)
        m = D.min(1)
        a.append(m.indices.cpu().numpy())
        if with_dist:
            d.append(m.values.cpu().numpy())
    return (np.concatenate(a), np.concatenate(d)) if with_dist else np.concatenate(a)


def slide_counts(cl, K, slide_of, n_slides):
    """クラスタ × slide の件数（bincount で。np.add.at は 1,000万 patch では遅い）。"""
    return np.bincount(cl.astype(np.int64) * n_slides + slide_of, minlength=K * n_slides).reshape(K, n_slides)


def label_counts(cl_b, lab_b, K, nlab):
    """クラスタ × Bayer のラベルの件数。lab_b は Bayer の patch のラベル（-1 はなし）。"""
    m = lab_b >= 0
    return np.bincount(cl_b[m].astype(np.int64) * nlab + lab_b[m], minlength=K * nlab).reshape(K, nlab)


def record_shares(cs, pos):
    """cs: K × slides の件数、pos: slides × F の記録あり（bool）。所見ごとの share（記録ありslide由来の割合）、基準率、倍率。"""
    n = cs.sum(1).astype(float)
    njf = cs @ pos.astype(np.int64)                     # K × F
    share = njf / np.maximum(n, 1)[:, None]
    base = njf.sum(0) / max(n.sum(), 1)
    return share, base, share / np.maximum(base, 1e-12), njf, n


def label_purity(cnt):
    n = cnt.sum(1)
    return np.where(n > 0, cnt.max(1) / np.maximum(n, 1), np.nan), n


def ward_rejoin(cen, size, key, target):
    """重心と大きさで Ward 法のつなぎ直し（ΔSSE = n_a n_b/(n_a+n_b) |c_a-c_b|^2 が最小の組から）。key が同じ（または -1 = 不明）組だけつなぐ。葉 → クラスタ番号を返す。"""
    K = len(cen)
    c, n, kk = cen.astype(np.float64).copy(), size.astype(np.float64).copy(), key.copy()
    root = np.arange(K)
    active = np.ones(K, bool)

    def row(a):
        d = (n[a] * n / (n[a] + n)) * ((c - c[a]) ** 2).sum(1)
        ok = active & ((kk == kk[a]) | (kk == -1) | (kk[a] == -1))
        d[~ok] = np.inf
        d[a] = np.inf
        return d
    D = np.stack([row(a) for a in range(K)])
    while active.sum() > target:
        i = np.argmin(D)
        a, b = divmod(int(i), K)
        if not np.isfinite(D[a, b]):
            break
        c[a] = (n[a] * c[a] + n[b] * c[b]) / (n[a] + n[b])
        n[a] += n[b]
        if kk[a] == -1:
            kk[a] = kk[b]
        active[b] = False
        root[root == b] = a
        D[b, :] = np.inf
        D[:, b] = np.inf
        D[a, :] = row(a)
        D[:, a] = D[a, :]
    _, mapping = np.unique(root, return_inverse=True)
    return mapping


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
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from lib.clustering import kmeans_lloyd
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
    cfg20 = yaml.safe_load(open(project_root / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    prio = cfg20["label_priority"]
    pre_path = str(project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl")
    reps = cfg["reps"]
    findings = list(cfg["findings"])
    F = len(findings)
    rng = np.random.default_rng(cfg["seed"])

    def h5_of(sid):
        for d_ in cfg["features_dirs"]:
            if (project_root / d_ / f"{sid}.h5").exists():
                return str(project_root / d_ / f"{sid}.h5")
        return ""

    # 1. Bayer の230枚（0025 cache）。21枚のラベルは学習・名前付け・評価のどれにも使わない
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Kb, laba, row_b = c["K"], c["lab"].astype(np.int64), c["row"]
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    keep = [k_ for k_ in np.unique(laba) if k_ >= 0 and (laba == k_).sum() >= cfg20["min_label_patches"]]
    names_l = [prio[k_] for k_ in keep]
    nlab = len(keep)
    lab = np.where(np.isin(laba, keep), np.searchsorted(keep, np.maximum(laba, 0)), -1)
    ex_ids = pd.read_csv(project_root / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab = np.where(np.isin(bayer.slide_id.to_numpy()[row_b], ex_ids), -1, lab)
    laba = np.where(lab >= 0, laba, -1)
    nb = len(Kb)
    comp_b = bayer.COMPOUND_NAME.fillna("unknown_" + bayer.slide_id).to_numpy()
    logger.info(f"Bayer: {len(bayer)} slides, {nb:,} patches, labeled {(lab >= 0).sum():,} in {nlab} labels; {time.time() - t0:.0f}s")

    # 2. 脂肪変性の3試験（0026 target_slides、各 2,048 patch）と、0033 の未学習 3,319枚（保存した PCA256）
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[q.EXP_ID.notna() & (q.EXP_ID != "")].copy()
    tgt = ex26.target_slides({**cfg, "target_exps": cfg["fatty_exps"], "features_dir": cfg["features_dirs"][0]}, project_root)
    tgt = tgt[~tgt.slide_id.isin(bayer.slide_id)].reset_index(drop=True)
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts = list(ex.map(ex30._load_sub, tgt.h5, chunksize=4))
    Kt = np.concatenate([p_[0] for p_ in parts]).astype(np.float16)
    row_t = np.concatenate([np.full(len(p_[0]), i) for i, p_ in enumerate(parts)])
    del parts
    z = np.load(project_root / cfg["unseen_features_from"], allow_pickle=True)
    Ku, row_u, uns_ids = z["K"], z["row"], z["slide_ids"].astype(str)
    del z
    logger.info(f"fatty: {len(tgt)} slides, {len(Kt):,} patches; unseen: {len(uns_ids)} slides, {len(Ku):,} patches; {time.time() - t0:.0f}s")

    # 3. 全 slide の表（qc catalog から）と、11所見の記録（slide × 所見、程度の最大）
    S = pd.concat([pd.DataFrame({"slide_id": bayer.slide_id, "set": "bayer"}), pd.DataFrame({"slide_id": tgt.slide_id, "set": "fatty"}), pd.DataFrame({"slide_id": uns_ids, "set": "unseen"})], ignore_index=True)
    S = S.merge(q[["slide_id", "EXP_ID", "GROUP_ID", "INDIVIDUAL_ID", "COMPOUND_NAME", "DOSE_LEVEL"]], on="slide_id", how="left")
    assert S.EXP_ID.notna().all(), "slides missing from the qc catalog"
    S["exp"] = S.EXP_ID.astype(int)
    S["is_ctrl"] = S.DOSE_LEVEL.eq("Control")
    pth = pd.read_csv(project_root / cfg["pathology_csv"], dtype=str)
    pth = pth[(pth.ORGAN == "Liver") & (pth.SP_FLG.str.lower() != "true") & pth.FINDING_TYPE.isin(findings)].copy()
    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
    for k_ in key:
        S[k_] = pd.to_numeric(S[k_], errors="coerce")
        pth[k_] = pd.to_numeric(pth[k_], errors="coerce")
    pth["g"] = pth.GRADE_TYPE.map(cfg["grade_map"]).fillna(1)
    G = S[["slide_id"] + key].merge(pth, on=key).pivot_table(index="slide_id", columns="FINDING_TYPE", values="g", aggfunc="max").reindex(S.slide_id).fillna(0).reindex(columns=findings, fill_value=0)
    pos = (G.to_numpy() > 0)                                  # slides × F
    off_t, off_u = len(bayer), len(bayer) + len(tgt)
    slide_of = np.r_[row_b, row_t + off_t, row_u + off_u]     # patch → S の行
    N = len(slide_of)
    set_of = np.r_[np.zeros(nb, np.int8), np.ones(len(Kt), np.int8), np.full(len(Ku), 2, np.int8)]
    S.assign(**{f"grade__{f}": G[f].to_numpy() for f in findings}).to_csv(run_dir / "slides.csv", index=False)
    logger.info(f"all: {len(S)} slides in {S.exp.nunique()} experiments, {N:,} patches; positives per finding (slides): {dict(zip(findings, pos.sum(0).tolist()))}; {time.time() - t0:.0f}s")

    # 4. 白色化（64・128次元）。Bayer と脂肪変性の試験は対照群を読み込んで、未学習は集合内の対照群で（0033と同じ）
    exp_p = S.exp.to_numpy()[slide_of]
    qc = q[(q.DOSE_LEVEL == "Control") & q.EXP_ID.astype(int).isin(set(exp_p[set_of < 2]))].copy()
    qc["h5"] = [h5_of(x) for x in qc.slide_id]
    qc = qc[qc.h5 != ""]
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts_c = list(ex.map(ex27._load_ctrl, qc.h5, chunksize=4))
    Kc = np.concatenate(parts_c)[:, :128].astype(np.float32)
    exp_c = np.concatenate([np.full(len(p_), int(e)) for p_, e in zip(parts_c, qc.EXP_ID)])
    del parts_c
    logger.info(f"controls for sphering: {len(qc)} slides in {len(set(exp_c))} experiments; {time.time() - t0:.0f}s")
    K_all = np.concatenate([Kb, Kt, Ku]).astype(np.float16)
    del Kb, Kt, Ku

    def sphere(d):
        U = K_all[:, :d].astype(np.float32)
        out = np.empty_like(U)
        m = set_of < 2
        e_fix = np.where(np.isin(exp_p[m], exp_c), exp_p[m], -999)
        out[m] = ex27.sphering(U[m], e_fix, np.r_[Kc[:, :d], Kc[:, :d]], np.r_[exp_c, np.full(len(exp_c), -999)])
        mu = set_of == 2
        cm = mu & S.is_ctrl.to_numpy()[slide_of]
        e_fix = np.where(np.isin(exp_p[mu], exp_p[cm]), exp_p[mu], -999)
        out[mu] = ex27.sphering(U[mu], e_fix, np.r_[U[cm], U[cm]], np.r_[exp_p[cm], np.full(int(cm.sum()), -999)])
        return out.astype(np.float32)
    Sw = {d: sphere(d) for d in sorted(set(reps.values()))}
    del Kc
    logger.info(f"sphering done ({sorted(Sw)}); {time.time() - t0:.0f}s")

    def lda_all(tr_mask):
        """Bayer の学習側ラベルで LDA を学び、全 patch を写す（次元あたりの標準偏差 1）。"""
        tf = ex26.fit_combo(K_all[:nb][tr_mask], lab[tr_mask], cfg, cfg["seed"])
        L_ = len(np.unique(lab[tr_mask][lab[tr_mask] >= 0])) - 1
        return np.concatenate([tf(b_)[:, :L_] for b_ in np.array_split(K_all, max(1, N // 200000))]).astype(np.float32)

    def rep_X(name, Lw):
        d = reps[name]
        fac = float(Sw[d].std(0).mean() / Lw.std(0).mean())
        return np.concatenate([Sw[d], cfg["lda_beta"] * fac * Lw], 1).astype(np.float32)

    # ---------------- 方法 ----------------
    fl, sp, tr_cfg, ev = cfg["flat"], cfg["split"], cfg["tree"], cfg["eval"]

    def fit_pool(seed):
        """k-means の学習点: Bayer と他の試験から同数（各 fit_points）。コンセンサスの各回はこの中から fit_points を引く。"""
        r_ = np.random.default_rng(seed)
        return np.r_[r_.choice(nb, fl["fit_points"], replace=False), nb + r_.choice(N - nb, fl["fit_points"], replace=False)]

    def flat(X, k, runs, seed):
        pool = torch.from_numpy(X[fit_pool(seed)]).to(device)
        _, cen = ex25.consensus_kmeans(pool, k, runs, fl["fit_points"], seed, device)
        del pool
        cl, dist = assign(cen, X, device, with_dist=True)
        return cl, cen.cpu().numpy(), dist

    def bad_clusters(cl, K, dist, crit, lab_mask):
        """純度が低いクラスタ（分ける対象）。lab_mask: 純度の計算に使える Bayer のラベル（学習側）"""
        cs = slide_counts(cl, K, slide_of, len(S))
        n = cs.sum(1)
        big = n >= sp["min_patches"]
        if crit == "label":
            pur, nl = label_purity(label_counts(cl[:nb], np.where(lab_mask, lab, -1), K, nlab))
            return np.where(big & (nl >= sp["min_labeled"]) & (pur < sp["tau"]))[0]
        if crit == "record":
            share, base, lift, _, _ = record_shares(cs, pos)
            j = lift.argmax(1)
            return np.where(big & (lift[np.arange(K), j] >= sp["min_lift"]) & (share[np.arange(K), j] < sp["tau"]))[0]
        md = np.bincount(cl, weights=dist, minlength=K) / np.maximum(n, 1)
        thr = np.quantile(md[big], 1 - sp["dispersion_top"])
        return np.where(big & (md >= thr))[0]

    def split(X, cl0, dist0, crit, lab_mask, seed):
        cl, dist = cl0.copy(), dist0.copy()
        K = int(cl.max()) + 1
        for r_ in range(sp["rounds"]):
            bad = bad_clusters(cl, K, dist, crit, lab_mask)
            for j in bad:
                idx = np.where(cl == j)[0]
                sub = np.random.default_rng([seed, int(j), r_]).choice(idx, min(sp["fit_points"], len(idx)), replace=False)
                _, cen, _ = kmeans_lloyd(X[sub], sp["k_sub"], seed + int(j), device)
                a, d_ = assign(torch.from_numpy(cen).to(device), X[idx], device, with_dist=True)
                cl[idx] = np.where(a == 0, j, K + a - 1)
                dist[idx] = d_
                K += sp["k_sub"] - 1
            logger.info(f"    split[{crit}] round {r_}: {len(bad)} clusters split -> {K}")
        return cl

    def tree_keys(cl, K, mode, lab_mask):
        if mode == "plain":
            return np.full(K, -1)
        if mode == "guided_label":
            cnt = label_counts(cl[:nb], np.where(lab_mask, lab, -1), K, nlab)
            pur, nl = label_purity(cnt)
            return np.where(nl >= tr_cfg["min_labeled"], cnt.argmax(1), -1)
        share, base, lift, _, _ = record_shares(slide_counts(cl, K, slide_of, len(S)), pos)
        j = lift.argmax(1)
        return np.where(lift[np.arange(K), j] >= tr_cfg["min_lift"], j, F)   # 濃縮なし = F（正常どうしはつなぐ）

    def all_methods(X, lab_mask, seed, runs_of):
        """方法ごとの割り当て（全 patch）を順に返す: (名前, cl)"""
        base = {}
        for k in fl["k_list"]:
            cl, cen, dist = flat(X, k, runs_of(k), seed)
            base[k] = (cl, cen, dist)
            yield f"flat_k{k}", cl
            logger.info(f"  flat k={k}; {time.time() - t0:.0f}s")
        cl0, _, dist0 = base[sp["k_start"]]
        for crit in ["label", "record", "dispersion"]:
            yield f"split_{crit}", split(X, cl0, dist0, crit, lab_mask, seed)
        clL, cenL, _ = base[tr_cfg["k_leaves"]]
        size = np.bincount(clL, minlength=len(cenL))
        for mode in tr_cfg["modes"]:
            key = tree_keys(clL, len(cenL), mode, lab_mask)
            for target in tr_cfg["targets"]:
                mp = ward_rejoin(cenL, size, key, target)
                yield f"tree{tr_cfg['k_leaves']}_{mode}_{target}", mp[clL]
            logger.info(f"  tree {mode}; {time.time() - t0:.0f}s")

    # ---------------- 評価 ----------------
    thr = ev["precision_thr"]
    nm_cfg = cfg["naming"]

    def name_and_precision(cl, K, lab_mask):
        """学習側ラベルで名前付け（0029と同じ規則）と、名前のラベルの学習側適合率。"""
        cnt = label_counts(cl[:nb], np.where(lab_mask, lab, -1), K, nlab)
        n_l = cnt.sum(1, keepdims=True)
        share = cnt / np.maximum(n_l, 1)
        prev = cnt.sum(0) / max(cnt.sum(), 1)
        lift = share / np.maximum(prev, 1e-12)
        ok = (cnt >= nm_cfg["min_n"]) & ((share >= nm_cfg["min_share"]) | (lift >= nm_cfg["min_lift"]))
        name = np.full(K, -1)
        for j in range(K):
            cand = np.where(ok[j])[0]
            if len(cand):
                name[j] = cand[np.argmax(lift[j, cand])]
        prec = np.where(name >= 0, share[np.arange(K), np.maximum(name, 0)], 0.0)
        return name, prec

    # (a) held-out: 化合物単位の5分割。方法ごとに評価側 patch の予測ラベルと「純度の高いプロトタイプに入ったか」を集める
    pred, pure = {}, {}
    ncl = {}
    folds = list(GroupKFold(n_splits=ev["heldout_folds"]).split(np.zeros(len(bayer)), groups=comp_b))
    for fi, (tr_s, te_s) in enumerate(folds):
        tr = np.isin(row_b, tr_s)
        te = ~tr
        Lw = lda_all(tr)
        for rname in reps:
            X = rep_X(rname, Lw)
            for mname, cl in all_methods(X, tr, cfg["seed"] + fi, lambda k: fl["consensus_runs_heldout"]):
                K = int(cl.max()) + 1
                name, prec = name_and_precision(cl, K, tr)
                key_ = (rname, mname)
                if key_ not in pred:
                    pred[key_], pure[key_], ncl[key_] = np.full(nb, -1), np.zeros(nb, bool), []
                pred[key_][te] = name[cl[:nb][te]]
                pure[key_][te] = prec[cl[:nb][te]] >= thr
                ncl[key_].append(K)
            del X
        del Lw
        logger.info(f"heldout fold {fi}: train {len(tr_s)} / test {len(te_s)} slides; {time.time() - t0:.0f}s")
    evm = lab >= 0
    y, cp = lab[evm], comp_b[row_b][evm]
    nonpar = [i for i, n in enumerate(names_l) if n != "parenchyma"]

    def label_table(p, u, ii):
        rows_ = []
        for l in range(nlab):
            t_, pr = y[ii] == l, p[ii] == l
            tp = (t_ & pr).sum()
            rec = tp / max(t_.sum(), 1)
            pre = tp / max(pr.sum(), 1) if pr.any() else 0.0
            cov = (t_ & pr & u[ii]).sum() / max(t_.sum(), 1)
            ppure = (t_ & pr & u[ii]).sum() / max((pr & u[ii]).sum(), 1) if (pr & u[ii]).any() else np.nan
            rows_.append({"label": names_l[l], "n_test": int(t_.sum()), "recall": rec, "precision": pre, "F1": 2 * rec * pre / (rec + pre) if rec + pre > 0 else 0.0, "coverage": cov, "precision_pure": ppure})
        return pd.DataFrame(rows_)
    comps = np.unique(cp)
    idx_by_c = {c_: np.where(cp == c_)[0] for c_ in comps}
    boots = [np.concatenate([idx_by_c[c_] for c_ in rng.choice(comps, len(comps))]) for _ in range(ev["bootstrap"])]
    held_rows, per_label = [], []
    for (rname, mname), p in pred.items():
        u = pure[(rname, mname)]
        p, u = p[evm], u[evm]
        t_ = label_table(p, u, np.arange(len(y)))
        t_.insert(0, "method", mname); t_.insert(0, "rep", rname)
        per_label.append(t_)
        B = []
        for ii in boots:
            tb = label_table(p, u, ii)
            B.append([np.nanmean(tb.F1.to_numpy()[nonpar]), np.nanmean(tb.coverage.to_numpy()[nonpar])])
        B = np.array(B)
        held_rows.append({"rep": rname, "method": mname, "n_clusters": float(np.mean(ncl[(rname, mname)])),
                          "macro_F1": float(np.nanmean(t_.F1.to_numpy()[nonpar])), "macro_F1_lo": float(np.nanpercentile(B[:, 0], 2.5)), "macro_F1_hi": float(np.nanpercentile(B[:, 0], 97.5)),
                          "macro_coverage": float(np.nanmean(t_.coverage.to_numpy()[nonpar])), "macro_coverage_lo": float(np.nanpercentile(B[:, 1], 2.5)), "macro_coverage_hi": float(np.nanpercentile(B[:, 1], 97.5)),
                          "macro_precision_pure": float(np.nanmean(t_.precision_pure.to_numpy()[nonpar]))})
    held = pd.DataFrame(held_rows)
    held.to_csv(run_dir / "heldout_summary.csv", index=False)
    pd.concat(per_label).to_csv(run_dir / "heldout_per_label.csv", index=False)
    pd.set_option("display.width", 250)
    logger.info("held-out:\n" + held.round(3).to_string(index=False))
    del pred, pure

    # (b)(c) 全集合: LDA は Bayer の全ラベル。方法ごとに記録の純度と読み出し AUC
    Lw = lda_all(np.ones(nb, bool))
    us = S.set.eq("unseen").to_numpy()
    sl_u = S[us].reset_index(drop=True)
    comp_u = sl_u.COMPOUND_NAME.to_numpy()
    yG = {f: pos[us, i].astype(int) for i, f in enumerate(findings)}
    unl = [f for f in findings if not cfg["findings"][f] and f != "Degeneration, fatty"]
    labd = [f for f in findings if cfg["findings"][f]]
    rec_rows, read_rows, fat_rows = [], [], []
    cm_t = np.where(S.COMPOUND_NAME.fillna("").str.contains("tetrachloride", case=False), "CCl4", np.where(S.COMPOUND_NAME.fillna("").str.contains("cholesterol", case=False), "cholesterol", "other")).astype(object)
    fat_slide = pos[:, findings.index("Degeneration, fatty")]
    for rname in reps:
        X = rep_X(rname, Lw)
        for mname, cl in all_methods(X, np.ones(nb, bool), cfg["seed"], lambda k: fl["consensus_runs"][k]):
            K = int(cl.max()) + 1
            np.save(run_dir / f"assign_{rname}_{mname}.npy", cl.astype(np.int16))
            nm = ex29.name_prototypes(cl[:nb], laba, prio, K, nm_cfg)
            names = nm.set_index("prototype").name.fillna("").reindex(range(K)).to_numpy()
            cs = slide_counts(cl, K, slide_of, len(S))
            share, base, lift, njf, n = record_shares(cs, pos)
            big = n >= ev["min_proto_patches"]
            for i, f in enumerate(findings):
                sh = np.where(big, share[:, i], -1)
                j = int(sh.argmax())
                hi = big & (share[:, i] >= thr)
                rec_rows.append({"rep": rname, "method": mname, "n_clusters": K, "finding": f, "base_rate": float(base[i]), "best_share": float(share[j, i]), "best_lift": float(lift[j, i]), "best_n": int(n[j]), "best_name": names[j],
                                 "n_protos_share_ge_thr": int(hi.sum()), "coverage": float(njf[hi, i].sum() / max(njf[:, i].sum(), 1))})
            # 脂肪変性を化合物内で見た純度（0026・0033と同じ定義）
            for cname in ("CCl4", "cholesterol"):
                m_ = (cm_t[slide_of] == cname)
                cnt_c = np.bincount(cl[m_], minlength=K)
                cnt_cf = np.bincount(cl[m_ & fat_slide[slide_of]], minlength=K)
                fr = cnt_cf / np.maximum(cnt_c, 1)
                okc = cnt_c >= ev["min_proto_patches"]
                fat_rows.append({"rep": rname, "method": mname, "compound": cname, "n_protos_purity90": int((okc & (fr >= 0.9)).sum()), "max_purity": float(fr[okc].max()) if okc.any() else np.nan,
                                 "coverage_purity90": float(cnt_cf[okc & (fr >= 0.9)].sum() / max(cnt_cf.sum(), 1))})
            # 読み出し（未学習 slide、化合物単位 CV）
            sh_u = cs[:, us].T.astype(float)
            sh_u /= np.maximum(sh_u.sum(1, keepdims=True), 1)
            ctrl = sl_u.is_ctrl.to_numpy()
            cmean = pd.DataFrame(sh_u[ctrl]).groupby(sl_u.exp.to_numpy()[ctrl]).mean().reindex(sl_u.exp).to_numpy()
            D_ = sh_u - np.where(np.isnan(cmean), sh_u[ctrl].mean(0), cmean)
            aucs = {}
            for f in unl + labd:
                yv = yG[f]
                npos_c = len(set(comp_u[yv == 1]))
                oof = np.full(len(sl_u), np.nan)
                for trn, ten in GroupKFold(n_splits=min(ev["cv_splits"], npos_c)).split(D_, yv, comp_u):
                    if yv[trn].sum() == 0:
                        continue
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
                    oof[ten] = clf.fit(D_[trn], yv[trn]).predict_proba(D_[ten])[:, 1]
                ok = ~np.isnan(oof)
                aucs[f] = roc_auc_score(yv[ok], oof[ok]) if len(set(yv[ok])) > 1 else np.nan
            read_rows.append({"rep": rname, "method": mname, "n_clusters": K, "mean_AUC_unlabeled": float(np.nanmean([aucs[f] for f in unl])), "mean_AUC_labeled": float(np.nanmean([aucs[f] for f in labd])), **{f"AUC__{f}": v for f, v in aucs.items()}})
            logger.info(f"{rname} {mname}: K={K}, unlabeled AUC {read_rows[-1]['mean_AUC_unlabeled']:.3f}, fatty share {share[:, findings.index('Degeneration, fatty')][big].max():.2f}; {time.time() - t0:.0f}s")
        del X
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(run_dir / "record_purity.csv", index=False)
    fat = pd.DataFrame(fat_rows)
    fat.to_csv(run_dir / "fatty_within_compound.csv", index=False)
    read = pd.DataFrame(read_rows)
    read.to_csv(run_dir / "readout.csv", index=False)

    # まとめ: 方法ごとに held-out・記録の純度・読み出しを1行に
    piv_cov = rec.pivot_table(index=["rep", "method"], columns="finding", values="coverage")
    piv_sh = rec.pivot_table(index=["rep", "method"], columns="finding", values="best_share")
    summ = held.set_index(["rep", "method"]).join(read.set_index(["rep", "method"])[["mean_AUC_unlabeled", "mean_AUC_labeled"]])
    summ["record_coverage_mean_unlabeled"] = piv_cov[unl + ["Degeneration, fatty"]].mean(1)
    summ["record_coverage_fatty"] = piv_cov["Degeneration, fatty"]
    summ["record_best_share_fatty"] = piv_sh["Degeneration, fatty"]
    summ["record_coverage_mean_labeled"] = piv_cov[labd].mean(1)
    summ = summ.reset_index()
    summ.to_csv(run_dir / "summary.csv", index=False)
    logger.info("summary:\n" + summ.round(3).to_string(index=False))
    logger.info("record coverage (share >= thr):\n" + piv_cov.round(3).to_string())

    # 図: 方法ごとに (held-out coverage, held-out F1, 記録 coverage の平均, 未学習 AUC)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(dict.fromkeys(summ.method))
    xs = np.arange(len(methods))
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    panels = [("macro_coverage", "held-out: share of label patches in prototypes with train precision >= %.1f (macro, 13 labels)" % thr),
              ("macro_F1", "held-out: macro F1 (13 labels)"), ("record_coverage_mean_unlabeled", "records: coverage by prototypes with share >= %.1f (mean of unlabeled findings + fatty)" % thr),
              ("mean_AUC_unlabeled", "unseen slides: mean readout AUC (unlabeled findings)")]
    for ax, (col, ttl) in zip(axes.ravel(), panels):
        for i, rname in enumerate(reps):
            s_ = summ[summ.rep == rname].set_index("method").reindex(methods)
            ax.bar(xs + (i - 0.5) * 0.4, s_[col], width=0.4, color=COL[rname], label=rname)
            if col in ("macro_coverage", "macro_F1"):
                ax.errorbar(xs + (i - 0.5) * 0.4, s_[col], yerr=[s_[col] - s_[col + "_lo"], s_[col + "_hi"] - s_[col]], fmt="none", ecolor="black", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
        ax.set_title(ttl, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(run_dir / "F1_methods.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(reps), figsize=(9 * len(reps), 6))
    for ax, rname in zip(np.atleast_1d(axes), reps):
        hm = piv_cov.loc[rname].reindex(methods)
        im = ax.imshow(hm.to_numpy().T, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(hm.columns))); ax.set_yticklabels(hm.columns, fontsize=8)
        for i_ in range(hm.shape[0]):
            for j_ in range(hm.shape[1]):
                ax.text(i_, j_, f"{hm.iloc[i_, j_]:.2f}", ha="center", va="center", fontsize=6, color="black" if hm.iloc[i_, j_] < 0.6 else "white")
        ax.set_title(f"{rname}: record coverage by prototypes with share >= {thr}", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "F2_record_coverage.png", dpi=150)
    plt.close(fig)

    results = {"n_slides": int(len(S)), "n_patches": int(N), "labels": names_l, "findings": findings, "summary": summ.round(4).to_dict(orient="records")}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info(f"Done. {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
