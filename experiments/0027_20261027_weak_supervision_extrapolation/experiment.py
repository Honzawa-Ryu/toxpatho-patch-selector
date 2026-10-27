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
import torch.nn as nn
import torch.nn.functional as F


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


# ---------------- Control slideの読み込み（slide内の順位 → 0021の教師なしPCA256） ----------------

def _init_ctrl(pre_path, n):
    _W["pre"] = pickle.loads(Path(pre_path).read_bytes())["pre"]
    _W["n"] = n


def _load_ctrl(h5_path):
    with h5py.File(h5_path, "r") as h:
        f = h["features"][:].astype(np.float32)
    rk = sys.modules["exp0023"].slide_rank(f)
    if len(rk) > _W["n"]:
        rk = rk[np.random.default_rng(len(rk)).choice(len(rk), size=_W["n"], replace=False)]
    return _W["pre"].transform(rk).astype(np.float32)


# ---------------- 表現 ----------------

def lda_oof(K, lab_known, exp, folds, ncls):
    """既知のラベルだけで、試験ごとの5分割のLDA。各patchはその試験を学習に入れていないLDAで写す。"""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import GroupKFold
    ue = np.unique(exp)
    Z = np.zeros((len(K), ncls - 1), dtype=np.float32)
    splits = GroupKFold(n_splits=folds).split(ue, groups=ue) if folds > 1 else [(np.arange(len(ue)), np.arange(len(ue)))]
    for tr_e, te_e in splits:
        tr = np.isin(exp, ue[tr_e])
        m = tr & (lab_known >= 0)
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(K[m], lab_known[m])
        te = np.flatnonzero(~tr) if folds > 1 else np.arange(len(K))
        Z[te] = lda.transform(K[te])[:, :ncls - 1]
    return Z


def match_scale(A, B):
    """Bの各次元の大きさの平均を、Aの各次元の大きさの平均にそろえる。"""
    return B * (A.std(0).mean() / max(B.std(0).mean(), 1e-6))


def sphering(K64, exp, Kc64, exp_c, shrink=0.1):
    """試験ごとに、その試験のControlのpatchの平均と共分散で白色化（ZCA）する（Ando 2017のsphering）。"""
    Z = np.zeros_like(K64)
    for e in np.unique(exp):
        c = Kc64[exp_c == e]
        mu = c.mean(0)
        C = np.cov(c - mu, rowvar=False)
        C = (1 - shrink) * C + shrink * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
        w, V = np.linalg.eigh(C)
        W = (V / np.sqrt(np.clip(w, 1e-6, None))) @ V.T
        m = exp == e
        Z[m] = (K64[m] - mu) @ W
    return Z.astype(np.float32)


def tvn(K256, exp, Kc256, exp_c, dim):
    """TVN: 全試験のControlのpatchでPCAを学び（典型的なばらつきの軸）、試験ごとにそのControlの平均・標準偏差で標準化する。"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim, random_state=0).fit(Kc256)
    Z, Zc = pca.transform(K256).astype(np.float32), pca.transform(Kc256).astype(np.float32)
    for e in np.unique(exp):
        c = Zc[exp_c == e]
        m = exp == e
        Z[m] = (Z[m] - c.mean(0)) / (c.std(0) + 1e-6)
    return Z


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.a = a
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.a * g, None


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def fat_eval(Z, n_annot, slides, row, grade, cfg, device, ex25):
    """脂肪変性の試験のpatchも含めてk-means（全patchで学習）し、化合物ごとに:
    best_rho = 脂肪変性の程度と、プロトタイプの割合（同じ試験のControlとの差）の順位相関が最大のプロトタイプの相関、
    purity = そのプロトタイプに入った化合物のpatchのうち脂肪変性ありのslideから来た割合、n90 = 純度90%以上のプロトタイプ数（500 patch以上）。"""
    from scipy.stats import spearmanr
    out = {}
    X = torch.from_numpy(np.ascontiguousarray(Z)).to(device)
    is_ctrl = (slides.dose == "Control").to_numpy()
    cmpd = np.where(slides.exp.isin([66, 67]), "CCl4", "cholesterol")
    for k in cfg["ks"]:
        cl, _ = ex25.kmeans_fit_assign(X, k, cfg["kmeans_seeds"], cfg["fit_points"], np.random.default_rng(0), device)
        clt = cl.cpu().numpy()[n_annot:]
        sh = np.zeros((len(slides), k))
        np.add.at(sh, (row, clt), 1)
        sh /= sh.sum(1, keepdims=True)
        d = np.full_like(sh, np.nan)
        for e, idx in slides.groupby("exp").indices.items():
            c = idx[is_ctrl[idx]]
            if len(c) >= 2:
                d[idx] = sh[idx] - sh[c].mean(0)
        for cname in ("CCl4", "cholesterol"):
            cm = cmpd == cname
            ok = cm & ~np.isnan(d[:, 0])
            rho = np.array([spearmanr(grade[ok], d[ok, j])[0] if d[ok, j].std() > 0 else np.nan for j in range(k)])
            pm = cm[row]
            n_j = np.bincount(clt[pm], minlength=k)
            pur = np.bincount(clt[pm], weights=(grade[row[pm]] > 0).astype(float), minlength=k) / np.maximum(n_j, 1)
            valid = n_j >= 500
            j = np.nanargmax(np.where(valid, rho, np.nan))
            out[f"fat_best_rho_{cname}_k{k}"] = float(rho[j])
            out[f"fat_purity_{cname}_k{k}"] = float(pur[j])
            out[f"fat_n90_{cname}_k{k}"] = int(((pur >= 0.9) & valid & (rho > 0.3)).sum())
    return out


def train_nn(kind, K, lab_known, exp, pairs, folds, ncls, hp, seed, device, logger):
    """gcd: 教師あり対照学習（既知のラベル）＋自己教師の対照学習（隣接patchを正例）。GCD（Vaze 2022）を、patchの増強の代わりに隣接patchで近似。
    ae_adv: 再構成（情報を捨てない）＋既知のラベルの分類＋試験を当てられないようにする敵対学習（scANVI・DANNの要素）。
    どちらも試験ごとの5分割で学び、各patchはその試験を学習に入れていないモデルで写す。"""
    from sklearn.model_selection import GroupKFold
    ue = np.unique(exp)
    Z = np.zeros((len(K), hp["out"]), dtype=np.float32)
    Kt = torch.from_numpy(K).to(device)
    mu, sd = Kt.mean(0), Kt.std(0) + 1e-6
    Kt = (Kt - mu) / sd
    lab_t = torch.from_numpy(lab_known).to(device)
    exp_id = torch.from_numpy(np.searchsorted(ue, exp)).to(device)
    splits = GroupKFold(n_splits=folds).split(ue, groups=ue) if folds > 1 else [(np.arange(len(ue)), np.arange(len(ue)))]
    for f, (tr_e, te_e) in enumerate(splits):
        torch.manual_seed(seed + f)
        g = torch.Generator(device="cpu").manual_seed(seed + f)
        tr = np.isin(exp, ue[tr_e])
        tr_idx = torch.from_numpy(np.flatnonzero(tr)).to(device)
        lab_idx = torch.from_numpy(np.flatnonzero(tr & (lab_known >= 0))).to(device)
        cw = torch.bincount(lab_t[lab_idx], minlength=ncls).float()
        cw = (cw.sum() / cw.clamp_min(1)) / ncls
        enc = mlp(K.shape[1], hp["hidden"], hp["out"]).to(device)
        params = list(enc.parameters())
        if kind == "ae_adv":
            dec = mlp(hp["out"], hp["hidden"], K.shape[1]).to(device)
            cls = nn.Linear(hp["out"], ncls).to(device)
            adv = mlp(hp["out"], hp["hidden"], len(ue)).to(device)
            params += list(dec.parameters()) + list(cls.parameters()) + list(adv.parameters())
        else:
            pr = pairs[tr[pairs[:, 0]] & tr[pairs[:, 1]]]
            pr_t = torch.from_numpy(pr).to(device)
        opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=1e-4)
        steps = hp["epochs"] * max(1, len(tr_idx) // hp["batch"])
        for st in range(steps):
            li = lab_idx[torch.randint(len(lab_idx), (hp["batch"] // 2,), generator=g).to(device)]
            if kind == "gcd":
                # 自己教師: 隣接patchの組（slideをまたがない）を正例、同じバッチの他を負例
                pi = pr_t[torch.randint(len(pr_t), (hp["batch"] // 2,), generator=g).to(device)]
                a = F.normalize(enc(Kt[pi[:, 0]]), dim=1)
                b = F.normalize(enc(Kt[pi[:, 1]]), dim=1)
                logits = a @ b.T / hp["temperature"]
                tgt = torch.arange(len(pi), device=device)
                loss = (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt)) / 2
                # 教師あり対照学習（SupCon）: 同じ既知のラベルを正例
                zl = F.normalize(enc(Kt[li]), dim=1)
                yl = lab_t[li]
                sim = zl @ zl.T / hp["temperature"]
                sim = sim - torch.eye(len(li), device=device) * 1e9
                pos = (yl[:, None] == yl[None, :]).float() - torch.eye(len(li), device=device)
                logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
                sup = -(logp * pos).sum(1) / pos.sum(1).clamp_min(1)
                loss = loss + hp["sup_weight"] * (sup * cw[yl]).mean()
            else:
                bi = tr_idx[torch.randint(len(tr_idx), (hp["batch"],), generator=g).to(device)]
                z = enc(Kt[bi])
                loss = hp["rec_weight"] * F.mse_loss(dec(z), Kt[bi])
                zl = enc(Kt[li])
                loss = loss + hp["cls_weight"] * F.cross_entropy(cls(zl), lab_t[li], weight=cw)
                alpha = min(1.0, st / max(1, steps // 3)) * hp["adv_weight"]
                loss = loss + F.cross_entropy(adv(GradRev.apply(z, alpha)), exp_id[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
        te = torch.from_numpy(np.flatnonzero(~tr) if folds > 1 else np.arange(len(K))).to(device)
        with torch.no_grad():
            Z[te.cpu().numpy()] = torch.cat([enc(Kt[x]) for x in te.split(65536)]).cpu().numpy()
        logger.info(f"    {kind} fold {f}: {steps} steps, last loss {loss.item():.3f}")
    return Z


# ---------------- 評価 ----------------

def evaluate(Z, lab, exp, names, known, heldout, cfg, seed, device, ex25):
    """k-means（kごと、seed2つの平均）でのラベルごとの純度・k80と、試験の混ざり（近傍30個のうち別の試験の割合）、
    伏せたラベルのkNN再現率（同じ試験を除いた近傍の多数決）。"""
    from sklearn.metrics import homogeneity_score
    out = {}
    X = torch.from_numpy(np.ascontiguousarray(Z)).to(device)
    lm = lab >= 0
    for k in cfg["ks"]:
        acc = {}
        for sd in cfg["kmeans_seeds"]:
            cl, _ = ex25.kmeans_fit_assign(X, k, [sd], cfg["fit_points"], np.random.default_rng([seed, sd]), device)
            cl = cl.cpu().numpy()
            C = np.zeros((lab.max() + 1, k))
            np.add.at(C, (lab[lm], cl[lm]), 1)
            pur = C / np.maximum(C.sum(0, keepdims=True), 1)
            for grp, labs_ in (("known", known), ("heldout", heldout)):
                ps, ks_ = [], []
                for nm in labs_:
                    i = names.index(nm)
                    r = C[i]
                    if r.sum() == 0:
                        continue
                    srt = np.sort(r)[::-1].cumsum() / r.sum()
                    ps.append(float((r * pur[i]).sum() / r.sum()))
                    ks_.append(int(np.searchsorted(srt, 0.8) + 1))
                    acc.setdefault(f"purity_{nm}_k{k}", []).append(ps[-1])
                acc.setdefault(f"purity_{grp}_k{k}", []).append(np.mean(ps))
                acc.setdefault(f"k80_{grp}_k{k}", []).append(np.mean(ks_))
            acc.setdefault(f"homogeneity_k{k}", []).append(homogeneity_score(lab[lm], cl[lm]))
        out.update({kk: float(np.mean(v)) for kk, v in acc.items()})
    # 試験の混ざり
    rng = np.random.default_rng(seed)
    q = rng.choice(len(Z), size=min(cfg["mix_queries"], len(Z)), replace=False)
    other = []
    for blk in np.array_split(q, max(1, len(q) // 1024)):
        D = torch.cdist(X[torch.from_numpy(blk).to(device)], X)
        D[torch.arange(len(blk)), torch.from_numpy(blk).to(device)] = float("inf")
        nn_ = D.topk(cfg["knn"], largest=False).indices.cpu().numpy()
        other.append((exp[nn_] != exp[blk, None]).mean(1))
    out["mix_other_exp"] = float(np.concatenate(other).mean())
    # 伏せたラベルのkNN再現率（ラベル付きpatchの中で、同じ試験を除いた近傍15個の多数決）
    li = np.flatnonzero(lm)
    Xl = X[torch.from_numpy(li).to(device)]
    el = torch.from_numpy(exp[li]).to(device)
    pred = np.empty(len(li), dtype=int)
    for s_ in range(0, len(li), 1024):
        D = torch.cdist(Xl[s_:s_ + 1024], Xl)
        D[el[s_:s_ + 1024, None] == el[None, :]] = float("inf")
        nn_ = D.topk(15, largest=False).indices.cpu().numpy()
        v = lab[li][nn_]
        pred[s_:s_ + 1024] = [np.bincount(r).argmax() for r in v]
    y = lab[li]
    for grp, labs_ in (("known", known), ("heldout", heldout)):
        rec = [float((pred[y == names.index(nm)] == names.index(nm)).mean()) for nm in labs_ if (y == names.index(nm)).any()]
        out[f"knn_recall_{grp}"] = float(np.mean(rec))
    return out


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
    cfg = config
    cfg.update(cfg.get("variants", {}).get(args.variant, {}))
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex25 = load_module(project_root / "experiments" / cfg["exp0025"] / "experiment.py", "exp0025")
    cfg20 = yaml.safe_load(open(project_root / "experiments" / cfg["exp0020"] / "config.yml"))
    prio = cfg20["label_priority"]

    # 1. 対象: Bayerの230枚のうち、Controlの特徴量が2枚以上ある試験のslide
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    sl = pd.read_csv(src / "slides.csv", dtype=str)
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    def h5_of(sid):
        for d in cfg["features_dirs"]:
            if (project_root / d / f"{sid}.h5").exists():
                return str(project_root / d / f"{sid}.h5")
        return ""
    qc = q[(q.DOSE_LEVEL == "Control") & q.EXP_ID.isin(sl.EXP_ID)].copy()
    qc["h5"] = [h5_of(x) for x in qc.slide_id]
    qc = qc[qc.h5 != ""]
    good = qc.EXP_ID.value_counts()
    good = set(good[good >= cfg["min_ctrl_slides"]].index)
    keep_sl = np.flatnonzero(sl.EXP_ID.isin(good).to_numpy())
    row0 = c["row"]
    idx = np.flatnonzero(np.isin(row0, keep_sl))
    remap = np.full(len(row0), -1)
    remap[idx] = np.arange(len(idx))
    pairs = c["pairs"]
    pairs = remap[pairs[np.isin(pairs[:, 0], idx) & np.isin(pairs[:, 1], idx)]]
    K = c["K"][idx].astype(np.float32)
    lab = c["lab"].astype(np.int64)[idx]
    exp = sl.EXP_ID.astype(int).to_numpy()[row0[idx]]
    keep = [k for k in np.unique(lab) if k >= 0 and (lab == k).sum() >= cfg20["min_label_patches"]]
    lab = np.where(np.isin(lab, keep), np.searchsorted(keep, np.maximum(lab, 0)), -1)
    names = [prio[k] for k in keep]
    logger.info(f"{len(keep_sl)} annotated slides in {len(good)} experiments with >= {cfg['min_ctrl_slides']} Control slides; {len(K):,} patches, "
                f"{(lab >= 0).sum():,} labeled; labels {names}; {time.time() - t0:.0f}s")
    n_annot = len(K)
    fat_slides = None
    if cfg.get("fatty_exps"):
        # 2026-10-27: 脂肪変性（Bayerにラベルなし＝自然な「伏せた所見」）で外挿性を確かめる。CCl4 66・67、cholesterol 698の全slide（Bayerの集合と重なるslideは除く）
        ex26 = load_module(project_root / "experiments/0026_20261027_fatty_degeneration_prototypes/experiment.py", "exp0026")
        qf = q[q.EXP_ID.notna() & (q.EXP_ID != "")].copy()
        qf = qf[qf.EXP_ID.astype(int).isin(cfg["fatty_exps"]) & ~qf.slide_id.isin(sl.slide_id)].copy()
        qf["h5"] = [h5_of(x) for x in qf.slide_id]
        qf = qf[qf.h5 != ""].reset_index(drop=True)
        qf["exp"] = qf.EXP_ID.astype(int)
        qf = qf.rename(columns={"COMPOUND_NAME": "compound", "DOSE_LEVEL": "dose", "SACRIFICE_PERIOD": "period"})
        pre_path_ = project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl"
        with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init_ctrl, initargs=(str(pre_path_), 10 ** 9)) as ex:
            fparts = list(ex.map(_load_ctrl, qf.h5))
        K = np.concatenate([K] + fparts).astype(np.float32)
        frow = np.concatenate([np.full(len(p_), i) for i, p_ in enumerate(fparts)])
        exp = np.r_[exp, qf.exp.to_numpy()[frow]]
        lab = np.r_[lab, np.full(len(frow), -1)]
        fat_slides, fat_row = qf, frow
        G_fat, _ = ex26.fat_grades({"pathology_csv": cfg["pathology_csv"], "fat_findings": ["Degeneration, fatty"], "grade_map": cfg["grade_map"]}, project_root, qf)
        fat_grade = G_fat["Degeneration, fatty"].to_numpy()
        good = good | {f"{e:04d}" for e in cfg["fatty_exps"]} | {str(e) for e in cfg["fatty_exps"]}
        logger.info(f"fatty experiments: {len(qf)} slides, {len(frow):,} patches; fatty > 0: {int((fat_grade > 0).sum())} slides")
    q2 = q[q.EXP_ID.notna() & (q.EXP_ID != "")]
    qc = q2[(q2.DOSE_LEVEL == "Control") & (q2.EXP_ID.isin(good) | q2.EXP_ID.astype(int).isin(cfg.get("fatty_exps", [])))].copy()
    qc["h5"] = [h5_of(x) for x in qc.slide_id]
    qc = qc[qc.h5 != ""]
    pre_path = project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl"
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init_ctrl, initargs=(str(pre_path), cfg["ctrl_patches_per_slide"])) as ex:
        parts = list(ex.map(_load_ctrl, qc.h5))
    Kc = np.concatenate(parts)
    exp_c = np.concatenate([np.full(len(p_), int(e)) for p_, e in zip(parts, qc.EXP_ID)])
    logger.info(f"Control: {len(qc)} slides, {len(Kc):,} patches; {time.time() - t0:.0f}s")

    # 2. 伏せ方ごとに、各手法の表現を作って評価
    d = cfg["dim"]
    hp = cfg["nn"]
    rows = []
    cache_Z = {}
    for hname, held in cfg["holdouts"].items():
        known = [n for n in names if n not in held]
        lab_known = np.where(np.isin(lab, [names.index(n) for n in known]), lab, -1)
        # 既知のラベルを連番に詰める（LDA・分類器用）
        kmap = {names.index(n): i for i, n in enumerate(known)}
        lab_k = np.array([kmap.get(v, -1) for v in lab_known])
        for meth in cfg["methods"]:
            t1 = time.time()
            if meth == "unsup":
                Z = K[:, :d]
            elif meth == "sphering":
                Z = cache_Z.setdefault("sphering", sphering(K[:, :d], exp, Kc[:, :d], exp_c))
            elif meth == "tvn":
                Z = cache_Z.setdefault("tvn", tvn(K, exp, Kc, exp_c, d))
            elif meth in ("lda", "unsup_lda", "tvn_lda", "sphering_lda"):
                L = cache_Z.get(("lda", hname))
                if L is None:
                    L = cache_Z[("lda", hname)] = lda_oof(K, lab_k, exp, cfg["folds"], len(known))
                if meth == "lda":
                    Z = L
                elif meth == "unsup_lda":
                    Z = np.c_[K[:, :d], cfg["lda_beta"] * match_scale(K[:, :d], L)]
                elif meth == "tvn_lda":
                    T = cache_Z.setdefault("tvn", tvn(K, exp, Kc, exp_c, d))
                    Z = np.c_[T, cfg["lda_beta"] * match_scale(T, L)]
                else:
                    T = cache_Z.setdefault("sphering", sphering(K[:, :d], exp, Kc[:, :d], exp_c))
                    Z = np.c_[T, cfg["lda_beta"] * match_scale(T, L)]
            else:
                Z = train_nn(meth, K, lab_k, exp, pairs, cfg["folds"], len(known), hp, cfg["seed"], device, logger)
            Z = np.ascontiguousarray(Z, dtype=np.float32)
            ev = evaluate(Z[:n_annot], lab[:n_annot], exp[:n_annot], names, known, held, cfg, cfg["seed"], device, ex25)   # Bayerの分だけで評価
            if fat_slides is not None:
                ev.update(fat_eval(Z, n_annot, fat_slides, fat_row, fat_grade, cfg, device, ex25))
            r = {"holdout": hname, "method": meth, "dim": Z.shape[1], **ev, "sec": time.time() - t1}
            rows.append(r)
            pd.DataFrame(rows).to_csv(run_dir / "results.csv", index=False)
            logger.info(f"[{hname}] {meth}: held-out purity k30 {ev['purity_heldout_k30']:.3f} k100 {ev['purity_heldout_k100']:.3f}, "
                        f"known purity k30 {ev['purity_known_k30']:.3f}, held-out kNN recall {ev['knn_recall_heldout']:.3f}, "
                        f"mix {ev['mix_other_exp']:.3f}; {time.time() - t1:.0f}s")

    # 3. 図: 伏せたラベルの純度（外挿性）× 既知のラベルの純度、点の大きさ＝試験の混ざり
    df = pd.DataFrame(rows)
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    pal = dict(zip(cfg["methods"], ["#000000", "#CC79A7", "#D55E00", "#0072B2", "#003F7F", "#56B4E9", "#009E73", "#E69F00", "#999999"]))
    mk = dict(zip(cfg["methods"], "o*sDdvP^X"))
    fig, axes = plt.subplots(len(cfg["holdouts"]), 3, figsize=(15, 4.6 * len(cfg["holdouts"])), squeeze=False)
    for ai, hname in enumerate(cfg["holdouts"]):
        t = df[df.holdout == hname]
        for aj, (xc, yc, xl, yl) in enumerate([("purity_known_k30", "purity_heldout_k30", "known labels: purity (k=30)", "held-out labels: purity (k=30)"),
                                               ("purity_known_k100", "purity_heldout_k100", "known labels: purity (k=100)", "held-out labels: purity (k=100)"),
                                               ("mix_other_exp", "knn_recall_heldout", "batch mixing: 30-NN from another experiment", "held-out labels: kNN recall (other experiments)")]):
            ax = axes[ai, aj]
            for _, r in t.iterrows():
                ax.scatter(r[xc], r[yc], color=pal[r.method], marker=mk[r.method], s=70, label=r.method)
                ax.annotate(r.method, (r[xc], r[yc]), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(f"held out: {', '.join(cfg['holdouts'][hname])}", fontsize=8)
    fig.suptitle("E1. Extrapolation: do labels never used in training still form pure prototypes? (out-of-fold by experiment)", fontsize=10)
    fig.savefig(out / "E1_extrapolation.png")
    plt.close(fig)
    results: dict = {"n_slides": int(len(keep_sl)), "n_experiments": len(good), "n_patches": int(len(K))}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()