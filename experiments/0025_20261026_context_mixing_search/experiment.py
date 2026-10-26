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
    parser.add_argument("--mode", choices=["cache", "search"], required=True)
    parser.add_argument("--variant", default="v1")
    parser.add_argument("--cache_from", default="cache_v1")
    return parser.parse_args()


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 150})
_W = {}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------- cache: 230枚の全patchを、slide内の順位 → 0021の教師なしPCA256 に ----------------

def _init_worker(pre_path, cfg20):
    _W["pre"] = pickle.loads(Path(pre_path).read_bytes())["pre"]
    _W["cfg20"] = cfg20


def _load_one(args):
    sid, h5_path, gj = args
    r = sys.modules["exp0020"]._load_slide((sid, h5_path, gj, _W["cfg20"]))
    with h5py.File(h5_path, "r") as h:
        size0 = int(h["coords"].attrs.get("patch_size_level0", 256))
    K = _W["pre"].transform(r["rank"]).astype(np.float16)
    return {"K": K, "lab": r["label"].astype(np.int16), "xy": r["xy"], "pairs": sys.modules["exp0021"].grid_neighbors(r["xy"], size0), "size0": size0}


def run_cache(cfg, project_root, run_dir, logger):
    from concurrent.futures import ProcessPoolExecutor
    ex20 = load_module(project_root / "experiments" / cfg["exp0020"] / "experiment.py", "exp0020")
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    cfg20 = {**ex20.load_config(project_root / "experiments" / cfg["exp0020"]), "random_per_slide": 10 ** 9}   # 全patch
    ann = project_root / cfg20["annotation_dir"]
    fdir = project_root / cfg20["features_dir"]
    ids = [x["filename"].split(".")[0] for x in json.loads((ann / "slides.json").read_text())["slides"]]
    ids = [i for i in ids if (fdir / f"{i}.h5").exists()]
    q = pd.read_csv(cfg20["qc_catalog"], dtype=str).set_index("slide_id")
    slides = q.loc[ids, ["EXP_ID", "COMPOUND_NAME", "DOSE_LEVEL", "SACRIFICE_PERIOD", "scanscope_id"]].reset_index()
    slides["svs_path"] = [str(project_root / "data/TGGATEs/WSI/Liver" / str(p).replace(" ", "_") / f"{s}.svs") for s, p in zip(slides.slide_id, slides.SACRIFICE_PERIOD)]
    t0 = time.time()
    jobs = [(i, fdir / f"{i}.h5", ann / "annotations/geojson" / f"{i}.svs.geojson") for i in ids]
    parts = []
    src = project_root / "outputs" / cfg["exp0021"] / cfg["models_from"] / "models.pkl"
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=_init_worker, initargs=(str(src), cfg20)) as ex:
        for i, r in enumerate(ex.map(_load_one, jobs)):
            parts.append(r)
            if (i + 1) % 20 == 0:
                logger.info(f"  loaded {i + 1}/{len(jobs)}; {time.time() - t0:.0f}s")
    off = np.cumsum([0] + [len(p["K"]) for p in parts])
    np.savez(run_dir / "cache.npz", K=np.concatenate([p["K"] for p in parts]), lab=np.concatenate([p["lab"] for p in parts]),
             row=np.concatenate([np.full(len(p["K"]), i, dtype=np.int32) for i, p in enumerate(parts)]),
             xy=np.concatenate([p["xy"] for p in parts]), pairs=np.concatenate([p["pairs"] + off[i] for i, p in enumerate(parts)]),
             size0=np.array([p["size0"] for p in parts]))
    slides.to_csv(run_dir / "slides.csv", index=False)
    logger.info(f"{len(slides)} slides, {off[-1]:,} patches; {time.time() - t0:.0f}s")
    return {"n_slides": len(slides), "n_patches": int(off[-1])}


# ---------------- search ----------------

def oof_combo(K, lab, exp, cfg, seed, logger, resid_dim=None, resid_weight=None):
    """combo表現（LDA＋LDAを除いた残りのPCA）を、試験ごとの5分割で作る。各patchは、その試験を学習に入れていない分割のモデルで写す。"""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import GroupKFold
    rd = cfg["combo_resid_dim"] if resid_dim is None else resid_dim
    rw = cfg["combo_resid_weight"] if resid_weight is None else resid_weight
    ue = np.unique(exp)
    L = len(np.unique(lab[lab >= 0])) - 1
    Z = np.zeros((len(K), L + rd), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for f, (tr_e, te_e) in enumerate(GroupKFold(n_splits=cfg["folds"]).split(ue, groups=ue)):
        tr = np.isin(exp, ue[tr_e])
        te = np.flatnonzero(~tr)
        m = tr & (lab >= 0)
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(K[m].astype(np.float32), lab[m])
        Wl = lda.scalings_[:, :L]
        Q, _ = np.linalg.qr(Wl)
        sub = rng.choice(np.flatnonzero(tr), size=min(300000, tr.sum()), replace=False)
        Ks = K[sub].astype(np.float32)
        s0 = lda.transform(Ks)[:, :L].std(0).mean()
        if rd:
            rpca = PCA(n_components=rd, random_state=seed).fit(Ks - (Ks @ Q) @ Q.T)
            s1 = rpca.transform(Ks - (Ks @ Q) @ Q.T).std(0).mean()
        for blk in np.array_split(te, max(1, len(te) // 200000)):
            Kb = K[blk].astype(np.float32)
            parts = [lda.transform(Kb)[:, :L] / s0]
            if rd:
                parts.append(rw * rpca.transform(Kb - (Kb @ Q) @ Q.T) / s1)
            Z[blk] = np.concatenate(parts, axis=1)
        logger.info(f"  fold {f}: {len(ue[tr_e])} train experiments, {len(te):,} patches transformed")
    return Z


class Graph:
    """slide内の8近傍のグラフ（GPU）。伝播はslideをまたがない。"""
    def __init__(self, pairs, n, device):
        self.src = torch.from_numpy(pairs[:, 0]).to(device)
        self.dst = torch.from_numpy(pairs[:, 1]).to(device)
        self.n = n
        self.deg = torch.zeros(n, device=device).index_add_(0, self.src, torch.ones(len(self.src), device=device))

    def mean(self, X, w=None):
        """隣接の（重みつき）平均。隣のないpatchは自分のまま。"""
        if w is None:
            agg = torch.zeros_like(X).index_add_(0, self.src, X[self.dst])
            d = self.deg
        else:
            agg = torch.zeros_like(X).index_add_(0, self.src, X[self.dst] * w[:, None])
            d = torch.zeros(self.n, device=X.device).index_add_(0, self.src, w)
        return torch.where(d[:, None] > 0, agg / d.clamp_min(1e-12)[:, None], X)

    def propagate(self, X, hops, alpha, w=None):
        out = X.clone()
        for _ in range(hops):
            out = (1 - alpha) * out + alpha * self.mean(out, w)
        return out

    def sym(self, X, hops):
        """GCNの対称正規化（自己ループつき）D^-1/2 (A+I) D^-1/2 を hops 回かける（ProtoGTXの伝播から学習する重みとReLUを除いたもの）。"""
        dn = (self.deg + 1).rsqrt()
        out = X
        for _ in range(hops):
            y = out * dn[:, None]
            out = (torch.zeros_like(y).index_add_(0, self.src, y[self.dst]) + y) * dn[:, None]
        return out

    def edge_weights(self, G, sigma_q):
        """境界を保つ伝播の重み: exp(-|g_i - g_j|^2 / s^2)、sは隣接の距離の分位点（sigma_q）。"""
        d2 = ((G[self.src] - G[self.dst]) ** 2).sum(1)
        s2 = torch.quantile(d2[torch.randperm(len(d2), device=d2.device)[:1000000]], sigma_q)
        return torch.exp(-d2 / s2)


def kmeans_fit_assign(X, k, seeds, fit_points, rng, device):
    from lib.clustering import kmeans_lloyd
    fit = rng.choice(len(X), size=min(fit_points, len(X)), replace=False)
    Xf = X[torch.from_numpy(fit).to(device)].cpu().numpy()
    best = None
    for sd in seeds:
        _, c, _ = kmeans_lloyd(Xf, k, sd, device)
        ct = torch.from_numpy(c).to(device)
        inert = float(sum(torch.cdist(x, ct).min(1).values.pow(2).sum() for x in X[torch.from_numpy(fit).to(device)].split(65536)))
        if best is None or inert < best[0]:
            best = (inert, ct)
    cen = best[1]
    cl = torch.cat([torch.cdist(x, cen).argmin(1) for x in X.split(65536)])
    return cl, cen


def soft_smooth(X, cen, graph, beta, iters, tau_q):
    """割り当ての平滑化（MRF的）: 自分の表現での所属確率に、隣接の所属確率の平均を beta 乗でかけて、iters 回くり返す。"""
    D2 = torch.cat([torch.cdist(x, cen).pow(2) for x in X.split(65536)])
    tau = torch.quantile(D2.min(1).values[torch.randperm(len(D2), device=D2.device)[:1000000]], tau_q)
    logP0 = torch.log_softmax(-D2 / tau, dim=1)
    P = logP0.exp()
    for _ in range(iters):
        P = torch.softmax(logP0 + beta * torch.log(graph.mean(P).clamp_min(1e-8)), dim=1)
    return P.argmax(1)


def metrics(cl, lab, exp, row, sel_exps, graph, small, prio):
    from sklearn.metrics import completeness_score, homogeneity_score, normalized_mutual_info_score
    cl = cl.cpu().numpy() if torch.is_tensor(cl) else cl
    out = {"neighbor_same": float((cl[graph.src.cpu().numpy()] == cl[graph.dst.cpu().numpy()]).mean())}
    k = cl.max() + 1
    ne = [len(np.unique(exp[cl == c])) for c in range(k) if (cl == c).any()]
    out["median_exps_per_cluster"] = float(np.median(ne))
    for part, m in (("sel", np.isin(exp, sel_exps) & (lab >= 0)), ("test", ~np.isin(exp, sel_exps) & (lab >= 0))):
        y, c = lab[m], cl[m]
        out[f"nmi_{part}"] = normalized_mutual_info_score(y, c)
        out[f"hom_{part}"] = homogeneity_score(y, c)
        out[f"comp_{part}"] = completeness_score(y, c)
        C = np.zeros((lab.max() + 1, k))
        np.add.at(C, (y, c), 1)
        pur = C / np.maximum(C.sum(0, keepdims=True), 1)
        sp, sk = [], []
        for nm in small:
            i = prio.index(nm)
            r = C[i]
            if r.sum() == 0:
                continue
            srt = np.sort(r)[::-1].cumsum() / r.sum()
            out[f"k80_{nm}_{part}"] = int(np.searchsorted(srt, 0.8) + 1)
            out[f"purity_{nm}_{part}"] = float((r * pur[i]).sum() / r.sum())
            sp.append(out[f"purity_{nm}_{part}"])
            sk.append(out[f"k80_{nm}_{part}"])
        out[f"small_purity_{part}"] = float(np.mean(sp))
        out[f"small_k80_{part}"] = float(np.mean(sk))
    return out


def configs(variant="v1"):
    if variant == "v2":
        return configs_v2()
    if variant == "v3":
        return configs_v3()
    return configs_v1()


def configs_v3():
    """学習する文脈化（ProtoGTXのGCN層に相当）。入力 [自分, 対称正規化の伝播, 境界を保つ伝播] から、
    Bayerのラベルを当てる小さなネットワークを試験ごとの5分割で学び（各patchは自分の試験を学習に入れていない分割のモデル）、
    中間層 h を表現にする。比較用にv2の上位も同じ繰り返しで再評価する。"""
    C = [{"kind": "self"},
         {"kind": "bilateral", "hops": 2, "w": 0.75, "alpha": 0.5, "sigma_q": 0.1},
         {"kind": "bilateral", "hops": 2, "w": 1.0, "alpha": 0.5, "sigma_q": 0.1}]
    for inp in ("self", "self_sym", "self_sym_bil"):
        for hid in (16, 32):
            for out in ("h", "z_h"):
                C.append({"kind": "learned", "inputs": inp, "hidden": hid, "out": out})
    for k in (20, 50):
        C.append({"kind": "self", "k": k})
        C.append({"kind": "bilateral", "hops": 2, "w": 1.0, "alpha": 0.5, "sigma_q": 0.1, "k": k})
    return C


def learned_context(Zt, feats, lab_c, exp, hidden, folds, seed, device, epochs=300):
    """試験ごとの5分割で、[feats] → ReLU(W1) → h → W2 → ラベル、を学ぶ。各patchのhは、その試験を学習に入れていないモデルで出す。"""
    from sklearn.model_selection import GroupKFold
    X = torch.cat(feats, 1)
    ue = np.unique(exp)
    H = torch.zeros(len(X), hidden, device=device)
    lab_t = torch.from_numpy(lab_c).to(device)
    exp_t = torch.from_numpy(exp).to(device)
    ncls = int(lab_c.max()) + 1
    for f, (tr_e, te_e) in enumerate(GroupKFold(n_splits=folds).split(ue, groups=ue)):
        torch.manual_seed(seed + f)
        tr = torch.isin(exp_t, torch.from_numpy(ue[tr_e]).to(device))
        idx = torch.nonzero(tr & (lab_t >= 0)).squeeze(1)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        w = torch.bincount(lab_t[idx], minlength=ncls).float()
        w = (w.sum() / w.clamp_min(1)) / ncls                     # ラベルの数の偏りを均す
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, ncls)).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
        Xi, yi = (X[idx] - mu) / sd, lab_t[idx]
        for _ in range(epochs):
            loss = torch.nn.functional.cross_entropy(net(Xi), yi, weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        te = ~tr
        with torch.no_grad():
            H[te] = net[1](net[0]((X[te] - mu) / sd))
    return H


def configs_v2():
    """v1で有望だったもの（境界を保つ伝播・割り当ての平滑化・弱い線形）の周りを細かく見て、
    対称正規化の伝播（ProtoGTX型）、境界を保つ伝播＋平滑化、土台の表現（残りのPCAの次元と重み）も変える。"""
    C = [{"kind": "self"}]   # 既定の表現（残りのPCA16次元、重み0.5）
    for rep in ({"resid_dim": 16, "resid_weight": 0.25}, {"resid_dim": 16, "resid_weight": 1.0},
                {"resid_dim": 32, "resid_weight": 0.5}, {"resid_dim": 0, "resid_weight": 0.0}):
        C.append({"kind": "self", "rep": rep})
    for h in (1, 2, 3):
        for w in (0.75, 1.0, 1.5):
            for q in (0.1, 0.25):
                C.append({"kind": "bilateral", "hops": h, "w": w, "alpha": 0.5, "sigma_q": q})
    for b in (0.25, 0.5, 0.75):
        for it in (3, 6, 10):
            C.append({"kind": "smooth", "beta": b, "iters": it, "tau_q": 0.5})
    for b in (0.25, 0.5):
        C.append({"kind": "smooth", "beta": b, "iters": 3, "tau_q": 0.25})
        C.append({"kind": "smooth", "beta": b, "iters": 3, "tau_q": 1.0})
    for h in (1, 2, 4):
        for w in (0.5, 1.0):
            C.append({"kind": "sym", "hops": h, "w": w})
    for b in (0.25, 0.5):
        C.append({"kind": "bilateral_smooth", "hops": 2, "w": 1.0, "alpha": 0.5, "sigma_q": 0.25, "beta": b, "iters": 3, "tau_q": 0.5})
    for w in (0.25, 0.5):
        C.append({"kind": "linear", "hops": 2, "w": w, "alpha": 0.5})
    C.append({"kind": "resid_only", "hops": 2, "alpha": 0.5})
    return C


def configs_v1():
    """試す入れ方。kind:
    linear     : [Z, w * 伝播(Z, hops)]
    multi      : [Z, w * 伝播(Z, h1), w * 伝播(Z, h2)]
    bilateral  : [Z, w * 境界を保つ伝播(Z, hops; 重みは自分の表現の近さ)]
    diff       : [Z, w * (伝播(Z, hops) - Z)]（周りとの違い）
    resid_only : LDAの軸は自分のまま、残りのPCAの次元だけ伝播で置き換える
    smooth     : Zでk-means → 割り当てを隣接で平滑化（MRF的）"""
    C = [{"kind": "self"}]
    for h in (1, 2, 4, 8):
        for w in (0.25, 0.5, 1.0, 2.0):
            C.append({"kind": "linear", "hops": h, "w": w, "alpha": 0.5})
    for w in (0.5, 1.0):
        C.append({"kind": "multi", "hops": [1, 4], "w": w, "alpha": 0.5})
        C.append({"kind": "multi", "hops": [2, 8], "w": w, "alpha": 0.5})
    for q in (0.1, 0.25, 0.5):
        for h in (2, 4, 8):
            for w in (0.5, 1.0, 2.0):
                C.append({"kind": "bilateral", "hops": h, "w": w, "alpha": 0.5, "sigma_q": q})
    for h in (2, 4):
        for w in (0.5, 1.0):
            C.append({"kind": "diff", "hops": h, "w": w, "alpha": 0.5})
    for h in (2, 4, 8):
        C.append({"kind": "resid_only", "hops": h, "alpha": 0.5})
    for b in (0.5, 1.0, 2.0, 4.0):
        for it in (1, 3):
            C.append({"kind": "smooth", "beta": b, "iters": it, "tau_q": 0.5})
    return C


def cfg_name(c):
    return c["kind"] + "".join(f"_{k}{v}" for k, v in c.items() if k != "kind").replace(" ", "").replace("'", "")


def run_search(cfg, args, project_root, run_dir, logger, device):
    src = project_root / "outputs" / os.environ["EXP_NAME"] / args.cache_from
    c = np.load(src / "cache.npz")
    K, lab, row, pairs = c["K"], c["lab"].astype(np.int64), c["row"], c["pairs"]
    slides = pd.read_csv(src / "slides.csv", dtype=str)
    cfg20 = yaml.safe_load(open(project_root / "experiments" / cfg["exp0020"] / "config.yml"))
    prio = cfg20["label_priority"]
    keep = [k for k in np.unique(lab) if k >= 0 and (lab == k).sum() >= cfg20["min_label_patches"]]
    lab = np.where(np.isin(lab, keep), lab, -1)
    exp = slides.EXP_ID.astype(int).to_numpy()[row]
    t0 = time.time()
    logger.info(f"{len(slides)} slides, {len(K):,} patches, {(lab >= 0).sum():,} labeled, labels {[prio[k] for k in keep]}")
    lab_c = np.where(lab >= 0, np.searchsorted(keep, np.maximum(lab, 0)), -1)   # LDAのクラスは連番に詰める（0021と同じ12ラベル）
    zcache = {}

    def get_Z(rep):
        """土台の表現（OOFのcombo）。残りのPCAの次元と重みごとに作り、保存する。"""
        key = (rep.get("resid_dim", cfg["combo_resid_dim"]), rep.get("resid_weight", cfg["combo_resid_weight"]))
        if key not in zcache:
            zp = project_root / "outputs" / os.environ["EXP_NAME"] / "z_cache" / f"Z_oof_rd{key[0]}_rw{key[1]}.npy"
            zp.parent.mkdir(exist_ok=True)
            if zp.exists():
                Zn = np.load(zp)
            else:
                Zn = oof_combo(K, lab_c, exp, cfg, cfg["seed"], logger, key[0], key[1])
                np.save(zp, Zn)
            zcache[key] = torch.from_numpy(Zn).to(device)
            logger.info(f"OOF combo resid_dim {key[0]}, weight {key[1]}: {tuple(Zn.shape)}; {time.time() - t0:.0f}s")
        return zcache[key]

    Zt = get_Z({})
    rng = np.random.default_rng(cfg["seed"])
    ue = np.unique(exp)
    sel = rng.choice(ue, size=int(len(ue) * cfg["select_frac"]), replace=False)
    (run_dir / "split.json").write_text(json.dumps({"select": sorted(int(e) for e in sel), "test": sorted(int(e) for e in set(ue) - set(sel))}))
    G = Graph(pairs, len(Zt), device)
    nL = len(keep) - 1
    prop_cache, wcache = {}, {}

    def prop(h, a, q=None):
        key = (h, a, q)
        if key not in prop_cache:
            if q is None:
                prop_cache[key] = G.propagate(Zt, h, a)
            else:
                if q not in wcache:
                    wcache[q] = G.edge_weights(Zt, q)
                prop_cache[key] = G.propagate(Zt, h, a, wcache[q])
        return prop_cache[key]

    n_rep = cfg.get("repeats", 1)
    res, assigns = [], {}
    base = {}
    out_csv = run_dir / "search_runs.csv"
    for cf in configs(args.variant):
        t1 = time.time()
        name = cfg_name(cf)
        kd = cf["kind"]
        Zs = get_Z(cf.get("rep", {}))
        if kd == "self":
            X = Zs
        elif kd == "linear":
            X = torch.cat([Zt, cf["w"] * prop(cf["hops"], cf["alpha"])], 1)
        elif kd == "multi":
            X = torch.cat([Zt] + [cf["w"] * prop(h, cf["alpha"]) for h in cf["hops"]], 1)
        elif kd in ("bilateral", "bilateral_smooth"):
            X = torch.cat([Zt, cf["w"] * prop(cf["hops"], cf["alpha"], cf["sigma_q"])], 1)
        elif kd == "diff":
            X = torch.cat([Zt, cf["w"] * (prop(cf["hops"], cf["alpha"]) - Zt)], 1)
        elif kd == "resid_only":
            X = torch.cat([Zt[:, :nL], prop(cf["hops"], cf["alpha"])[:, nL:]], 1)
        elif kd == "sym":
            X = torch.cat([Zt, cf["w"] * G.sym(Zt, cf["hops"])], 1)
        elif kd == "learned":
            fe = [Zt]
            if "sym" in cf["inputs"]:
                fe.append(G.sym(Zt, 2))
            if "bil" in cf["inputs"]:
                fe.append(prop(2, 0.5, 0.1))
            Hh = learned_context(Zt, fe, lab_c, exp, cf["hidden"], cfg["folds"], cfg["seed"], device)
            Hh = Hh / Hh.std(0).mean().clamp_min(1e-6) * Zt.std(0).mean()   # 大きさを自分の表現にそろえる
            X = Hh if cf["out"] == "h" else torch.cat([Zt, Hh], 1)
        kk_ = cf.get("k", cfg["k"])
        for rp in range(n_rep):
            rr = np.random.default_rng([cfg["seed"], rp])
            seeds = [s_ + 100 * rp for s_ in cfg["kmeans_seeds"]]
            if kd == "smooth":
                cl = soft_smooth(Zt, base[rp][1], G, cf["beta"], cf["iters"], cf["tau_q"])
            else:
                cl, cen = kmeans_fit_assign(X, kk_, seeds, cfg["fit_points"], rr, device)
                if kd == "self" and not cf.get("rep") and "k" not in cf:
                    base[rp] = (cl, cen)
                if kd == "bilateral_smooth":
                    cl = soft_smooth(X, cen, G, cf["beta"], cf["iters"], cf["tau_q"])
            r = {"name": name, "repeat": rp, **{kk: str(v) for kk, v in cf.items()}, **metrics(cl, lab, exp, row, sel, G, cfg["small_labels"], prio)}
            res.append(r)
            if rp == 0:
                assigns[name] = cl.cpu().numpy().astype(np.int16)
        pd.DataFrame(res).to_csv(out_csv, index=False)
        t = pd.DataFrame([x for x in res if x["name"] == name])
        logger.info(f"{name}: NMI sel {t.nmi_sel.mean():.3f}±{t.nmi_sel.std():.3f} (hom {t.hom_sel.mean():.3f}, comp {t.comp_sel.mean():.3f}), "
                    f"small purity {t.small_purity_sel.mean():.2f}, neighbour same {t.neighbor_same.mean():.2f}; {time.time() - t1:.0f}s")
        if len(prop_cache) > 12:
            prop_cache.clear()
    runs = pd.DataFrame(res)
    num = [c_ for c_ in runs.columns if c_ not in ("name", "repeat") and pd.api.types.is_numeric_dtype(runs[c_])]
    df = runs.groupby("name", sort=False)[num].mean()
    sd = runs.groupby("name", sort=False)[num].std().add_suffix("_sd")
    meta = runs.drop_duplicates("name").set_index("name").drop(columns=num + ["repeat"])
    df = pd.concat([meta, df, sd], axis=1).reset_index()
    df.to_csv(run_dir / "search.csv", index=False)
    # 選び方（selectの試験だけで決める。値は繰り返しの平均）:
    #  best_nmi     : NMI最大
    #  best_context : NMIがself（既定の表現）の平均−標準偏差以上の中で、隣のpatchが同じプロトタイプの割合が最大（隣接情報を損なわずに入れる）
    #  best_small   : 小さい構造の純度の平均が最大
    if "k" in df:   # kを変えた行はNMIの大きさが比べられないので、選ぶ対象から外す
        df_all, df = df, df[df["k"].isna()]
    s0 = df[df.name == "self"].iloc[0] if (df.name == "self").any() else df[df.kind == "self"].iloc[0]
    thr = s0.nmi_sel - (s0.nmi_sel_sd if np.isfinite(s0.nmi_sel_sd) else 0.005)
    picks = {"self": s0["name"], "best_nmi": df.sort_values("nmi_sel", ascending=False).iloc[0]["name"],
             "best_context": df[df.nmi_sel >= thr].sort_values("neighbor_same", ascending=False).iloc[0]["name"],
             "best_small": df.sort_values("small_purity_sel", ascending=False).iloc[0]["name"]}
    for kk, nm in picks.items():
        np.save(run_dir / f"assign_{kk}.npy", assigns[nm])
    (run_dir / "picks.json").write_text(json.dumps(picks, indent=1))
    logger.info("picks: " + json.dumps(picks))
    logger.info("\n" + df.set_index("name").loc[list(dict.fromkeys(picks.values())),
                ["nmi_sel", "nmi_sel_sd", "nmi_test", "nmi_test_sd", "hom_test", "comp_test", "small_purity_test", "small_k80_test", "neighbor_same", "median_exps_per_cluster"]].round(3).to_string())
    # 図: 隣のpatchが同じ割合 × NMI（select）、入れ方ごとに色
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    kinds = list(dict.fromkeys(df.kind))
    pal = dict(zip(kinds, ["#000000", "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]))
    mk = dict(zip(kinds, "*osD^vP"))
    for ax, (ycol, ylab) in zip(axes, [("nmi_sel", "NMI with Bayer labels (select experiments)"), ("small_purity_sel", "mean purity of small structures (select)")]):
        for kd in kinds:
            t = df[df.kind == kd]
            ax.scatter(t.neighbor_same, t[ycol], c=pal[kd], marker=mk[kd], s=40 if kd == "self" else 18, label=kd, alpha=.85)
        ax.axhline(s0[ycol], color="0.6", lw=.8, ls="--")
        ax.set_xlabel("adjacent patches in the same prototype")
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("S1. Ways of adding neighbourhood information (230 annotated slides, out-of-fold LDA, k=30)", fontsize=10)
    fig.savefig(out / "S1_tradeoff.png")
    plt.close(fig)
    return {"picks": picks, "n_configs": len(df)}


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
    variant_key = f"{args.mode}_{args.variant}"

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "cache":
        results: dict = run_cache(config, project_root, run_dir, logger)
    else:
        results = run_search(config, args, project_root, run_dir, logger, device)

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()