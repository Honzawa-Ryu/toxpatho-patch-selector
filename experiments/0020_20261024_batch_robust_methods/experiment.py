import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

import h5py
import torch
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
    parser.add_argument("--variant", default="compare")     # 出力の名前（指標・手法を足して回し直すとき）
    parser.add_argument("--mode", default="compare", choices=["compare", "supervised"])   # supervised: Bayerのラベルで学ぶ部分空間（試験ごとの交差検証）
    return parser.parse_args()


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 140})


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---------------- データ: Bayerのアノテーションがあるslideのpatch ----------------

def _load_slide(args):
    """1枚分: 全patchの特徴量を読み、ラベルを付け、ラベル付き全部＋無作為800を返す。
    slide単位の補正（平均・標準偏差・順位・ストライプの平均）は全patchで計算して、選んだpatchに当てる。"""
    sid, h5_path, geojson_path, cfg = args
    from scipy.stats import norm
    from shapely.geometry import box, shape
    from shapely.strtree import STRtree
    with h5py.File(h5_path, "r") as h:
        f = l2(h["features"][:].astype(np.float32))
        xy = h["coords"][:]
        size0 = int(h["coords"].attrs.get("patch_size_level0", 256))
    # ラベル
    feats = json.loads(Path(geojson_path).read_text())["features"]
    lmap, prio = cfg["label_map"], cfg["label_priority"]
    polys, labs = [], []
    for ft in feats:
        lab = lmap.get(ft["properties"]["classification"]["name"])
        if lab is None:
            continue
        g = shape(ft["geometry"])
        polys.append(g if g.is_valid else g.buffer(0))
        labs.append(prio.index(lab))
    label = np.full(len(xy), -1)
    if polys:
        tree = STRtree(polys)
        boxes = [box(x, y, x + size0, y + size0) for x, y in xy]
        pairs = tree.query(boxes, predicate="intersects")   # (2, m): patch, polygon
        best = np.full(len(xy), 10 ** 6)
        for pi, gi in zip(*pairs):
            g = polys[gi]
            inter = boxes[pi].intersection(g).area
            if inter / min(boxes[pi].area, g.area) >= cfg["min_overlap"]:
                best[pi] = min(best[pi], labs[gi])
        label[best < 10 ** 6] = best[best < 10 ** 6]
    rng = np.random.default_rng([cfg["seed"], int(sid)])
    unl = np.flatnonzero(label < 0)
    pick = np.sort(np.r_[np.flatnonzero(label >= 0), rng.choice(unl, size=min(cfg["random_per_slide"], len(unl)), replace=False)])
    # slide単位の補正
    mu, sd = f.mean(0), f.std(0) + 1e-4
    stripe = ((xy[:, 0] + size0 / 2) // cfg["stripe_width"]).astype(int)
    st_mean = {s: f[stripe == s].mean(0) for s in np.unique(stripe[pick])}
    srt = np.sort(f, axis=0)
    fp = f[pick]
    rank = np.empty_like(fp)
    for d in range(f.shape[1]):
        rank[:, d] = np.searchsorted(srt[:, d], fp[:, d], side="left")
    rank = norm.ppf((rank + 0.5) / len(f)).astype(np.float32)
    return {"slide_id": sid, "X": fp, "center": fp - mu, "zscore": (fp - mu) / sd, "rank": rank,
            "stripe_c": fp - np.stack([st_mean[s] for s in stripe[pick]]), "label": label[pick], "xy": xy[pick],
            "stripe": stripe[pick], "n_all": len(f)}


def build_dataset(cfg, project_root, logger):
    from concurrent.futures import ProcessPoolExecutor
    ann = project_root / cfg["annotation_dir"]
    fdir = project_root / cfg["features_dir"]
    ids = [s["filename"].split(".")[0] for s in json.loads((ann / "slides.json").read_text())["slides"]]
    ids = [i for i in ids if (fdir / f"{i}.h5").exists()]
    q = pd.read_csv(cfg["qc_catalog"], dtype={"slide_id": str, "EXP_ID": str})
    meta = q.set_index("slide_id").loc[ids, ["EXP_ID", "COMPOUND_NAME", "DOSE_LEVEL", "SACRIFICE_PERIOD", "scanscope_id"]].reset_index()
    jobs = [(i, fdir / f"{i}.h5", ann / "annotations/geojson" / f"{i}.svs.geojson", cfg) for i in ids]
    parts = []
    with ProcessPoolExecutor(max_workers=cfg.get("workers", 8)) as ex:
        for k, r in enumerate(ex.map(_load_slide, jobs)):
            parts.append(r)
            if (k + 1) % 20 == 0:
                logger.info(f"  loaded {k + 1}/{len(jobs)} slides")
    d = {key: np.concatenate([p[key] for p in parts]) for key in ["X", "center", "zscore", "rank", "stripe_c", "label", "xy", "stripe"]}
    d["slide"] = np.concatenate([np.full(len(p["label"]), i) for i, p in enumerate(parts)])
    meta = meta.assign(n_all=[p["n_all"] for p in parts], n_sampled=[len(p["label"]) for p in parts],
                       n_labeled=[int((p["label"] >= 0).sum()) for p in parts])
    return d, meta


# ---------------- 手法 ----------------

def pca_fit(X, n, seed, fit_rows=None):
    from sklearn.decomposition import PCA
    return PCA(n_components=n, svd_solver="randomized", random_state=seed).fit(X if fit_rows is None else X[fit_rows])


def coral(Z, slide, eps=1e-3):
    """slideごとに共分散を白色化し、全体の共分散で色付けし直す（PCA空間で）。"""
    def msqrt(C, inv):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, eps, None)
        return (V * (w ** (-.5 if inv else .5))) @ V.T
    out = np.empty_like(Z)
    G = np.cov(Z, rowvar=False) + eps * np.eye(Z.shape[1])
    Gs = msqrt(G, False)
    for s in np.unique(slide):
        m = slide == s
        Zs = Z[m] - Z[m].mean(0)
        C = np.cov(Zs, rowvar=False) * (1 - .3) + .3 * G   # 縮小推定（枚数が少ないslideのため）
        out[m] = Zs @ msqrt(C, True) @ Gs + Z.mean(0)
    return out.astype(np.float32)


def harmony(Z, batch, hp, seed, device):
    """Harmony（Korsunsky et al. 2019）の簡略実装: 多様性罰則つきのソフトクラスタリングと、クラスタごとのリッジ回帰による補正を繰り返す。"""
    from lib.clustering import kmeans_lloyd
    Zt = torch.from_numpy(Z).float().to(device)
    b = torch.from_numpy(np.unique(batch, return_inverse=True)[1]).to(device)
    nb = int(b.max()) + 1
    B = torch.nn.functional.one_hot(b, nb).float()
    pr = B.mean(0)
    Zn = torch.nn.functional.normalize(Zt, dim=1)
    _, c, _ = kmeans_lloyd(Zn.cpu().numpy(), hp["n_clusters"], seed, device)
    Y = torch.nn.functional.normalize(torch.from_numpy(c).float().to(device), dim=1)

    R = torch.softmax(-2 * (1 - Zn @ Y.T) / hp["sigma"], dim=1)
    gen = torch.Generator().manual_seed(seed)
    n = len(Zt)
    Zc = Zt.clone()
    for _ in range(hp["max_iter"]):
        for _ in range(hp["max_iter_cluster"]):
            Y = torch.nn.functional.normalize(R.T @ Zn, dim=1)
            O = R.T @ B
            E = R.sum(0)[:, None] * pr[None, :]
            # 元論文と同じく、細胞を5%ずつのブロックに分けて順に更新する（一度に更新すると罰則で振動する）
            for blk in torch.randperm(n, generator=gen).to(device).split(max(1, n // 20)):
                O -= R[blk].T @ B[blk]
                E -= R[blk].sum(0)[:, None] * pr[None, :]
                r = torch.exp(-2 * (1 - Zn[blk] @ Y.T) / hp["sigma"]) * (((E + 1) / (O + 1)) ** hp["theta"])[:, b[blk]].T
                R[blk] = r / r.sum(1, keepdim=True)
                O += R[blk].T @ B[blk]
                E += R[blk].sum(0)[:, None] * pr[None, :]
        Zc = Zt.clone()
        for k in range(R.shape[1]):
            rk = R[:, k]
            ok = B.T @ rk                                    # (nb,)
            A = torch.zeros(nb + 1, nb + 1, device=device)
            A[0, 0] = rk.sum()
            A[0, 1:] = A[1:, 0] = ok
            A[1:, 1:] = torch.diag(ok) + hp["lamb"] * torch.eye(nb, device=device)
            rhs = torch.cat([(rk[:, None] * Zt).sum(0, keepdim=True), B.T @ (rk[:, None] * Zt)])
            W = torch.linalg.solve(A, rhs)
            Zc -= rk[:, None] * (B @ W[1:])
        Zn = torch.nn.functional.normalize(Zc, dim=1)
    return Zc.cpu().numpy()


def mnn_pairs(Z, slide, k, device, chunk=1024):
    """別々のslideの間で、互いにk近傍に入るpatchの組（相互最近傍）。"""
    Zt = torch.from_numpy(np.ascontiguousarray(Z)).float().to(device)
    s = torch.from_numpy(slide).to(device)
    nn = np.empty((len(Z), k), dtype=np.int64)
    for a in range(0, len(Z), chunk):
        D = torch.cdist(Zt[a:a + chunk], Zt)
        D[s[a:a + chunk, None] == s[None, :]] = float("inf")
        nn[a:a + chunk] = D.topk(k, largest=False).indices.cpu().numpy()
    i = np.repeat(np.arange(len(Z)), k)
    j = nn.ravel()
    fwd = set(zip(i.tolist(), j.tolist()))
    keep = np.array([(jj, ii) in fwd for ii, jj in zip(i, j)])
    p = np.stack([i[keep], j[keep]], 1)
    return p[p[:, 0] < p[:, 1]]


def mnn_contrast(X, Zpair, slide, hp, seed, device):
    """相互最近傍の組を正例にした対照学習（InfoNCE）で、X（生の特徴量）から低次元への射影を学ぶ。
    組はslideの平均を引いた空間（Zpair）で探すので、slideのずれを越えて同じ種類のpatchを近づける写像になる。"""
    torch.manual_seed(seed)
    pairs = mnn_pairs(Zpair, slide, hp["mnn_k"], device)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hp["hidden"]), torch.nn.GELU(), torch.nn.Linear(hp["hidden"], hp["out_dim"])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=1e-4)
    Xt = torch.from_numpy(X).float().to(device)
    pt = torch.from_numpy(pairs).to(device)
    for _ in range(hp["epochs"]):
        for blk in torch.randperm(len(pt), device=device).split(hp["batch"]):
            a = torch.nn.functional.normalize(net(Xt[pt[blk, 0]]), dim=1)
            b = torch.nn.functional.normalize(net(Xt[pt[blk, 1]]), dim=1)
            logits = a @ b.T / hp["temperature"]
            t = torch.arange(len(blk), device=device)
            loss = (torch.nn.functional.cross_entropy(logits, t) + torch.nn.functional.cross_entropy(logits.T, t)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        Z = torch.cat([net(x) for x in Xt.split(8192)]).cpu().numpy()
    return Z, len(pairs), loss.item()


def embed(name, d, meta, cfg, device, pca_cache):
    seed, n = cfg["seed"], cfg["pca_components"]
    if name == "raw":
        return d["X"]
    if name in ("pca64", "harmony_slide", "harmony_exp", "bbknn_slide", "coral_slide"):
        if "pca64" not in pca_cache:
            pca_cache["pca64"] = pca_fit(d["X"], n, seed).transform(d["X"]).astype(np.float32)
        Z = pca_cache["pca64"]
        if name in ("pca64", "bbknn_slide"):
            return Z
        if name == "coral_slide":
            return coral(Z, d["slide"])
        batch = d["slide"] if name == "harmony_slide" else meta.EXP_ID.to_numpy()[d["slide"]]
        return harmony(Z, batch, cfg["harmony"], seed, device)
    if name in ("center_harmony", "mnn_contrast", "mnn_contrast_rank"):
        key = "rank" if name == "mnn_contrast_rank" else "center"
        if key + "64" not in pca_cache:
            pca_cache[key + "64"] = pca_fit(d[key], n, seed).transform(d[key]).astype(np.float32)
        if name == "center_harmony":
            return harmony(pca_cache["center64"], d["slide"], cfg["harmony"], seed, device)
        Z, n_pairs, loss = mnn_contrast(d["X"], pca_cache[key + "64"], d["slide"], cfg["contrast"], seed, device)
        pca_cache["mnn_info"] = {"n_pairs": n_pairs, "final_loss": loss}
        return Z
    key = {"center_slide": "center", "zscore_slide": "zscore", "rank_slide": "rank", "center_stripe": "stripe_c"}[name]
    return pca_fit(d[key], n, seed).transform(d[key]).astype(np.float32)


# ---------------- 評価 ----------------

def knn_label(Z, lab, group, k, device, balanced_by=None, chunk=512):
    """ラベル付きpatchだけで、同じgroup（slideや試験）を除いた近傍の多数決。balanced_byを与えると、そのgroupごとに最近傍1つだけを候補にする（BBKNN型）。"""
    idx = np.flatnonzero(lab >= 0)
    Zt = torch.from_numpy(np.ascontiguousarray(Z[idx])).float().to(device)
    g = torch.from_numpy(group[idx]).to(device)
    y = lab[idx]
    pred = np.empty(len(idx), dtype=int)
    nl = int(lab.max()) + 1
    for s in range(0, len(idx), chunk):
        D = torch.cdist(Zt[s:s + chunk], Zt)
        D[g[s:s + chunk, None] == g[None, :]] = float("inf")
        if balanced_by is None:
            nn = D.topk(k, largest=False).indices.cpu().numpy()
        else:
            bb = torch.from_numpy(balanced_by[idx]).to(device)
            nbg = int(bb.max()) + 1
            mins = torch.full((D.shape[0], nbg), float("inf"), device=device)
            mins = mins.scatter_reduce(1, bb[None, :].expand(D.shape[0], -1), D, reduce="amin")
            arg = torch.full((D.shape[0], nbg), -1, dtype=torch.long, device=device)
            hit = (D == mins[:, bb])
            cols = torch.arange(D.shape[1], device=device)[None, :].expand_as(D)
            arg = arg.scatter_reduce(1, bb[None, :].expand(D.shape[0], -1), torch.where(hit, cols, -1), reduce="amax")
            top = mins.topk(k, largest=False).indices
            nn = arg.gather(1, top).cpu().numpy()
        votes = np.zeros((len(nn), nl))
        np.add.at(votes, (np.repeat(np.arange(len(nn)), nn.shape[1]), y[nn].ravel()), 1)
        pred[s:s + len(nn)] = votes.argmax(1)
    return y, pred


def probe_auc(Z, lab, exp, neg_label, hp, seed):
    """各ラベル vs 実質を、試験ごとに分けた交差検証（学習に使わない試験で評価）のロジスティック回帰で見分けるAUC。
    3つ以上の試験に出るラベルだけ。試験をまたいで通用する表現かを測る。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    out = {}
    for c in np.unique(lab[lab >= 0]):
        if c == neg_label:
            continue
        m = (lab == c) | (lab == neg_label)
        if len(np.unique(exp[lab == c])) < 3:
            continue
        X, y, g = Z[m], (lab[m] == c).astype(int), exp[m]
        score = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=min(hp["folds"], len(np.unique(g)))).split(X, y, g):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=hp["C"], max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
            score[te] = clf.decision_function(sc.transform(X[te]))
        ok = np.isfinite(score)
        if len(np.unique(y[ok])) == 2:
            out[c] = roc_auc_score(y[ok], score[ok])
    return out


def batch_mixing(Z, slide, exp, k, n_q, seed, device):
    rng = np.random.default_rng(seed)
    q = rng.choice(len(Z), size=min(n_q, len(Z)), replace=False)
    Zt = torch.from_numpy(np.ascontiguousarray(Z)).float().to(device)
    other_s, other_e = [], []
    for s in range(0, len(q), 1024):
        qq = q[s:s + 1024]
        D = torch.cdist(Zt[qq], Zt)
        D[torch.arange(len(qq)), torch.from_numpy(qq).to(device)] = float("inf")
        nn = D.topk(k, largest=False).indices.cpu().numpy()
        other_s.append((slide[nn] != slide[qq, None]).mean(1))
        other_e.append((exp[nn] != exp[qq, None]).mean(1))
    return float(np.concatenate(other_s).mean()), float(np.concatenate(other_e).mean())


def cluster_scores(Z, lab, k, seed, device):
    from lib.clustering import kmeans_lloyd
    from sklearn.metrics import normalized_mutual_info_score
    cl, _, _ = kmeans_lloyd(np.ascontiguousarray(Z, dtype=np.float32), k, seed, device)
    m = lab >= 0
    nmi = normalized_mutual_info_score(lab[m], cl[m])
    conc = {}
    for c in np.unique(lab[m]):
        v = cl[m][lab[m] == c]
        top = np.bincount(v).argmax()
        conc[c] = ((v == top).mean(), (lab[m][cl[m] == top] == c).mean())   # 最多クラスタへの集中度、そのクラスタの純度
    return nmi, conc, cl


def supervised_cv(d, exp_int, keep, cfg, device, logger):
    """Bayerのラベルで線形判別（LDA、縮小推定）の部分空間を学び、学習に使わない試験で評価する（試験ごとの交差検証）。
    比較に、同じ入力・同じpatchでのPCA（教師なし）も同じ物差しで測る。"""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import balanced_accuracy_score, normalized_mutual_info_score, recall_score
    from sklearn.model_selection import GroupKFold
    from lib.clustering import kmeans_lloyd
    hp = cfg["supervised"]
    lab = d["label"]
    rows, recalls = [], []
    for inp in hp["inputs"]:
        X = {"raw": d["X"], "rank_slide": d["rank"], "center_slide": d["center"]}[inp]
        Xr = PCA(n_components=hp["pre_pca"], svd_solver="randomized", random_state=cfg["seed"]).fit_transform(X).astype(np.float32)
        for space in ["lda", "pca"]:
            ys, ps, nmis, mix = [], [], [], []
            for tr, te in GroupKFold(n_splits=hp["folds"]).split(Xr, groups=exp_int):
                trl = tr[lab[tr] >= 0]
                if space == "lda":
                    proj = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(Xr[trl], lab[trl])
                    Z = proj.transform(Xr[te]).astype(np.float32)
                else:
                    Z = PCA(n_components=len(keep) - 1, random_state=cfg["seed"]).fit(Xr[tr]).transform(Xr[te]).astype(np.float32)
                lt, et = lab[te], exp_int[te]
                if (lt >= 0).sum() < 50 or len(np.unique(et[lt >= 0])) < 2:
                    continue
                y, p = knn_label(Z, lt, et, cfg["knn_label"], device)
                ys.append(y)
                ps.append(p)
                cl, _, _ = kmeans_lloyd(np.ascontiguousarray(Z), len(keep), cfg["seed"], device)
                nmis.append(normalized_mutual_info_score(lt[lt >= 0], cl[lt >= 0]))
                mix.append(batch_mixing(Z, d["slide"][te], et, cfg["knn_batch"], 5000, cfg["seed"], device)[1])
            y, p = np.concatenate(ys), np.concatenate(ps)
            rows.append({"input": inp, "space": space, "bal_acc_exp_out": balanced_accuracy_score(y, p), "nmi_kmeans": float(np.mean(nmis)),
                         "other_exp_nn": float(np.mean(mix))})
            recalls.append(pd.Series(recall_score(y, p, labels=sorted(keep), average=None, zero_division=0),
                                     index=[cfg["label_priority"][c] for c in sorted(keep)], name=f"{inp}/{space}"))
            logger.info(f"supervised {inp}/{space}: " + ", ".join(f"{k} {v:.3f}" for k, v in rows[-1].items() if isinstance(v, float)))
    return pd.DataFrame(rows), pd.DataFrame(recalls)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from sklearn.metrics import balanced_accuracy_score, recall_score

    args = parse_args()
    exp_name = os.environ["EXP_NAME"]
    cfg = load_config(Path(__file__).parent)
    run_dir = get_run_dir(project_root, __file__, args.variant, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=args.variant, config=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting: {exp_name} / {args.variant}; device {device}")

    t0 = time.time()
    d, meta = build_dataset(cfg, project_root, logger)
    prio = cfg["label_priority"]
    counts = pd.Series(d["label"][d["label"] >= 0]).value_counts()
    keep = [c for c in counts.index if counts[c] >= cfg["min_label_patches"]]
    d["label"] = np.where(np.isin(d["label"], keep), d["label"], -1)
    lab_names = {i: prio[i] for i in sorted(keep)}
    meta.to_csv(run_dir / "slides.csv", index=False)
    logger.info(f"{len(meta)} slides, {len(d['label']):,} patches ({(d['label'] >= 0).sum():,} labeled); labels kept: "
                + ", ".join(f"{prio[c]} {counts[c]}" for c in sorted(keep)) + f"; {time.time() - t0:.0f}s")
    exp = meta.EXP_ID.to_numpy()[d["slide"]]
    exp_int = np.unique(exp, return_inverse=True)[1]
    if args.mode == "supervised":
        res, rc = supervised_cv(d, exp_int, keep, cfg, device, logger)
        res.to_csv(run_dir / "supervised_metrics.csv", index=False)
        rc.to_csv(run_dir / "supervised_recall_exp_out.csv")
        (run_dir / "results.json").write_text(json.dumps({"n_slides": int(len(meta)), "n_labeled": int((d["label"] >= 0).sum()),
                                                          "metrics": res.to_dict(orient="records")}, indent=2, ensure_ascii=False, default=float))
        complete_run(run_dir)
        logger.info("Done.")
        return

    rows, recalls, recalls_exp, probes, pca_cache, embs = [], [], [], [], {}, {}
    for name in cfg["methods"]:
        t0 = time.time()
        Z = embed(name, d, meta, cfg, device, pca_cache)
        embs[name] = Z
        rec = {"method": name, "dim": Z.shape[1]}
        bal = d["slide"] if name == "bbknn_slide" else None
        for mode, grp in [("slide", d["slide"]), ("exp", exp_int)]:
            y, p = knn_label(Z, d["label"], grp, cfg["knn_label"], device, balanced_by=bal)
            rec[f"bal_acc_{mode}_out"] = balanced_accuracy_score(y, p)
            r = recall_score(y, p, labels=sorted(keep), average=None, zero_division=0)
            (recalls if mode == "slide" else recalls_exp).append(pd.Series(r, index=[lab_names[c] for c in sorted(keep)], name=name))
        if name != "bbknn_slide":
            aucs = probe_auc(Z, d["label"], exp_int, prio.index("parenchyma"), cfg["probe"], cfg["seed"])
            rec["probe_auc_mean"] = float(np.mean(list(aucs.values()))) if aucs else np.nan
            probes.append(pd.Series({lab_names[c]: v for c, v in aucs.items()}, name=name))
        if name != "bbknn_slide":
            rec["other_slide_nn"], rec["other_exp_nn"] = batch_mixing(Z, d["slide"], exp, cfg["knn_batch"], cfg["batch_queries"], cfg["seed"], device)
            nmi, conc, _ = cluster_scores(Z, d["label"], cfg["kmeans_k"], cfg["seed"], device)
            rec["nmi_kmeans"] = nmi
            for c, (a, b) in conc.items():
                rec[f"conc_{lab_names[c]}"], rec[f"purity_{lab_names[c]}"] = a, b
        rows.append(rec)
        if name.startswith("mnn_contrast"):
            rec.update(pca_cache.get("mnn_info", {}))
        logger.info(f"{name}: " + ", ".join(f"{k} {v:.3f}" for k, v in rec.items() if isinstance(v, float) and not k.startswith(("conc_", "purity_")))
                    + f"; {time.time() - t0:.0f}s")
    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "metrics.csv", index=False)
    rc = pd.DataFrame(recalls)
    rc.to_csv(run_dir / "recall_per_label_slide_out.csv")
    rce = pd.DataFrame(recalls_exp)
    rce.to_csv(run_dir / "recall_per_label_exp_out.csv")
    pr = pd.DataFrame(probes)
    pr.to_csv(run_dir / "probe_auc_leave_experiment_out.csv")

    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    # M1. 混ざり具合 × 区別の保存
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, xcol, ycol in [(axes[0], "other_slide_nn", "bal_acc_slide_out"), (axes[1], "other_exp_nn", "bal_acc_exp_out")]:
        r = res.dropna(subset=[xcol])
        ax.scatter(r[xcol], r[ycol], s=40, color="#2a78d6")
        for t in r.itertuples():
            ax.annotate(t.method, (getattr(t, xcol), getattr(t, ycol)), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"{xcol}: share of {cfg['knn_batch']}-NN from another {'slide' if 'slide' in xcol else 'experiment'} (higher = better mixed)")
        ax.set_ylabel(f"{ycol}: balanced accuracy of label kNN using only other {'slides' if 'slide' in ycol else 'experiments'}")
    b = res[res.method == "bbknn_slide"]
    if len(b):
        axes[0].axhline(b.bal_acc_slide_out.iloc[0], color="#e34948", ls="--", lw=1)
        axes[0].text(axes[0].get_xlim()[0], b.bal_acc_slide_out.iloc[0], " bbknn_slide (graph only)", color="#e34948", fontsize=8, va="bottom")
    fig.suptitle(f"M1. Batch mixing vs label conservation ({len(meta)} annotated slides, {(d['label'] >= 0).sum():,} labeled patches)", fontsize=12)
    fig.savefig(out / "M1_mixing_vs_conservation.png")
    plt.close(fig)
    # M2. ラベルごと: slide外のkNNの再現率、試験外のkNNの再現率、試験外の線形分類のAUC（各ラベル vs 実質）
    fig, axes = plt.subplots(1, 3, figsize=(3 * (0.75 * rc.shape[1] + 2.5), 0.45 * len(rc) + 2.5), sharey=True)
    for ax, t, title, vmin in [(axes[0], rc, "recall, kNN from other slides", 0), (axes[1], rce, "recall, kNN from other experiments", 0),
                               (axes[2], pr, "AUC vs parenchyma, linear probe trained on other experiments", .5)]:
        t = t.reindex(index=rc.index)
        ax.imshow(t.to_numpy(dtype=float), cmap="viridis", vmin=vmin, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(t.to_numpy(dtype=float)):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v < (vmin + 1) / 2 else "black")
        ax.set_xticks(range(t.shape[1]), [f"{c}\n(n={counts[prio.index(c)]})" for c in t.columns], rotation=45, ha="right", fontsize=7.5)
        ax.set_title(title, fontsize=10)
    axes[0].set_yticks(range(len(rc)), rc.index)
    fig.suptitle("M2. Per-label conservation", fontsize=12)
    fig.savefig(out / "M2_recall_per_label.png")
    plt.close(fig)
    # M3. t-SNE（ラベル付きを優先して最大1.2万点）
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits
    rng = np.random.default_rng(cfg["seed"])
    li = np.flatnonzero(d["label"] >= 0)
    li = rng.choice(li, size=min(8000, len(li)), replace=False)
    ui = rng.choice(np.flatnonzero(d["label"] < 0), size=4000, replace=False)
    sub = np.r_[li, ui]
    show = [m for m in cfg["methods"] if m != "bbknn_slide"]
    fig, axes = plt.subplots(2, len(show), figsize=(4.2 * len(show), 8.6), squeeze=False)
    import colorsys
    pal = {c: colorsys.hsv_to_rgb((i * .61803398875) % 1, .8, .85) for i, c in enumerate(sorted(keep))}
    spal = np.asarray([colorsys.hsv_to_rgb((i * .61803398875) % 1, .6, .8) for i in range(len(meta))])
    for j, m in enumerate(show):
        Z = embs[m][sub]
        Zp = pca_fit(Z, min(50, Z.shape[1]), cfg["seed"]).transform(Z) if Z.shape[1] > 50 else Z
        with threadpool_limits(limits=8):
            xy = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", random_state=0, n_jobs=8).fit_transform(Zp)
        ax = axes[0, j]
        lab = d["label"][sub]
        ax.scatter(*xy[lab < 0].T, color="#dde1e5", s=1, linewidths=0, rasterized=True)
        for c in sorted(keep):
            mm = lab == c
            ax.scatter(*xy[mm].T, color=pal[c], s=2, linewidths=0, rasterized=True, label=prio[c])
        ax.set_title(m, fontsize=10)
        ax = axes[1, j]
        ax.scatter(*xy.T, c=spal[d["slide"][sub]], s=1, linewidths=0, rasterized=True)
        for a in axes[:, j]:
            a.set_xticks([])
            a.set_yticks([])
    axes[0, -1].legend(markerscale=6, frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1, .5))
    axes[0, 0].set_ylabel("by label (grey = unlabeled)")
    axes[1, 0].set_ylabel("by slide")
    fig.suptitle("M3. t-SNE per method (top: Bayer label, bottom: slide)", fontsize=12)
    fig.savefig(out / "M3_tsne_per_method.png")
    plt.close(fig)
    np.savez_compressed(run_dir / "dataset_index.npz", slide=d["slide"], label=d["label"], xy=d["xy"], stripe=d["stripe"])
    (run_dir / "results.json").write_text(json.dumps({"n_slides": int(len(meta)), "n_patches": int(len(d["label"])),
                                                      "n_labeled": int((d["label"] >= 0).sum()), "labels": {prio[c]: int(counts[c]) for c in keep},
                                                      "metrics": res.to_dict(orient="records")}, indent=2, ensure_ascii=False, default=float))
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
