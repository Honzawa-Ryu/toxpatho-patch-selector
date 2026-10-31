"""0030 v4の割り当てと特徴から、試験・スキャナがどれだけ混ざったかを C（LDA）・D（64次元）・D13（13次元）と、
順位変換なしの元の特徴（raw: UNI2-h → PCA64）で比べる（2026-10-31ユーザー: 学会要旨で「TG-GATEs内で試験・スキャナが混ざった」と言うための裏付け）。

対照群のslide（857枚、各slideから同じ座標の1,024 patch）で、4表現を同じpatchで:
  knn    : 近傍30個のうち、別の試験・別のスキャナのpatchの割合（同じslideは除く）。完全に混ざったときの期待値（自slide以外の参照の構成）と並べる
  js     : slideごとのプロトタイプ分布のJensen–Shannon距離を、同じ試験 / 別の試験・同じスキャナ / 別の試験・別のスキャナ の組で平均
  spread : プロトタイプごとに、試験の大きさで均した占有率の最大値と、80%を占めるのに要る試験の数（中央値）
  tsne   : 各試験から同数のpatchを描き、試験・スキャナで色分け
全slide（3,319枚、2,048 patch/slide）で、C・D・D13の spread（rawは全slideを読み直さない）。
raw の PCA64 は対照群の20万patchで学習。raw のプロトタイプは同じ対照群patchで k-means（k=100）。C・D・D13 は0032の重心による0030の割り当てをそのまま使う。
スキャナは SS1601 と SS1598（SS1224 は1枚なので除く）。試験とスキャナはほぼ入れ子（43試験のうち2台を使うのは9試験、うち対照群で2台は4試験）。
出力: outputs/0030_20261028_tggates_unseen_findings/<版>/batch_mixing/
usage: python scripts/tggates_unseen_batch_mixing.py [版]  （コンテナ内、Slurm経由、GPU。版の既定は v4）
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_RUN = sys.argv[1] if len(sys.argv) > 1 else "v4"
RUN = ROOT / "outputs/0030_20261028_tggates_unseen_findings" / _RUN
OUT = RUN / "batch_mixing"
_CFG = yaml.safe_load(open(ROOT / "experiments/0030_20261028_tggates_unseen_findings/config.yml"))
_CFG = {**_CFG, **_CFG.get("variants", {}).get(_RUN, {})}
MAPS = list(_CFG["maps"])                       # D_unsup, D13_unsup13, C_lda
REPS = ["raw_pca64"] + MAPS
TITLES = {"raw_pca64": "raw (no rank, PCA64)", **_CFG["titles"]}
SCANNERS = ["SS1601", "SS1598"]
PER_SLIDE, KNN, QUERIES, K, SEED, TSNE_PER_EXP, PCA_FIT = 1024, 30, 20000, 100, 0, 300, 200000
COL = {"raw_pca64": "#999999", "D_unsup": "#0072B2", "D13_unsup13": "#56B4E9", "C_lda": "#D55E00"}   # Okabe-Ito


def _load_raw(args):
    """h5から、指定した座標のpatchの元の特徴（UNI2-h、1536次元）を、指定の順で返す。"""
    import h5py
    h5, xy_want = args
    with h5py.File(h5, "r") as h:
        xy = h["coords"][:]
        f = h["features"][:]
    key = {tuple(p): i for i, p in enumerate(xy.tolist())}
    return f[[key[tuple(p)] for p in xy_want.tolist()]].astype(np.float16)


def knn_mix(X, slide, grp, device, rng):
    """近傍KNN個のうち別のgrpの割合（観測）と、完全に混ざったときの期待値（自slide以外の参照のうち別grpの割合）。"""
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
    st, gt = torch.from_numpy(slide).to(device), torch.from_numpy(grp).to(device)
    q = rng.choice(len(X), min(QUERIES, len(X)), replace=False)
    n_s, n_g = np.bincount(slide), np.bincount(grp)
    obs = []
    for qq in np.array_split(q, max(1, len(q) // 512)):
        qt = torch.from_numpy(qq).to(device)
        D = torch.cdist(Xt[qt], Xt)
        D[st[qt][:, None] == st[None, :]] = float("inf")
        nb = D.topk(KNN, largest=False).indices
        obs.append((gt[nb] != gt[qt][:, None]).float().mean(1).cpu().numpy())
    exp_ = (len(X) - n_g[grp[q]]) / (len(X) - n_s[slide[q]])
    return float(np.concatenate(obs).mean()), float(exp_.mean())


def js_matrix(P):
    """slide × プロトタイプの分布から、全組のJensen–Shannon距離（scipyのjensenshannonと同じ、底e）。"""
    from scipy.special import rel_entr
    n = len(P)
    D = np.zeros((n, n))
    for i in range(n - 1):
        m = (P[i] + P[i + 1:]) / 2
        D[i, i + 1:] = np.sqrt(np.maximum(0.5 * rel_entr(P[i], m).sum(1) + 0.5 * rel_entr(P[i + 1:], m).sum(1), 0))
    return D + D.T


def spread(cl, grp, k):
    """プロトタイプごとに、grpの大きさで均した占有率の最大値と、80%を占めるのに要るgrpの数。"""
    cnt = pd.crosstab(pd.Categorical(cl, categories=range(k)), grp, dropna=False)
    frac = cnt / cnt.sum(0)
    share = (frac.T / frac.sum(1)).T.fillna(0).to_numpy()
    srt = -np.sort(-share, axis=1)
    n80 = (np.cumsum(srt, 1) < 0.8).sum(1) + 1
    return pd.DataFrame({"prototype": range(k), "n_patches": cnt.sum(1).to_numpy(), "top_share": share.max(1), "n_groups_80pct": n80})


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    sl = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str, "EXP_ID": str})
    A = np.load(RUN / "assignments.npz")
    F = np.load(RUN / "features_sub.npz")
    row, xy, U, Lp = A["row"], F["xy"], F["U"], F["Lp"]
    cl_all = {m: A[f"cl_{m}"].astype(np.int64) for m in MAPS}

    def h5_of(sid):
        for d_ in _CFG["features_dirs"]:
            if (ROOT / d_ / f"{sid}.h5").exists():
                return str(ROOT / d_ / f"{sid}.h5")
        raise FileNotFoundError(sid)
    exp_code = pd.factorize(sl.EXP_ID)[0]
    scan_code = np.array([SCANNERS.index(s) if s in SCANNERS else -1 for s in sl.scanscope_id])
    print(f"slides {len(sl)}, patches {len(row):,}; {time.time() - t0:.0f}s", flush=True)

    # 1. 全slideでの spread（C・D・D13）
    rows = []
    for m in MAPS:
        for gname, g in [("experiment", exp_code), ("scanner", scan_code)]:
            ok = g[row] >= 0
            sp = spread(cl_all[m][ok], g[row][ok], K)
            sp.insert(0, "group", gname); sp.insert(0, "rep", m); sp.insert(0, "slides", "all")
            rows.append(sp)
    # 2. 対照群: 各slideから同じ座標の PER_SLIDE patch。rawはh5から読み直す
    ci = np.where(sl.is_ctrl.to_numpy())[0]
    pick = np.concatenate([rng.choice(np.where(row == i)[0], min(PER_SLIDE, int((row == i).sum())), replace=False) for i in ci])
    pick.sort()
    r_c, xy_c = row[pick], xy[pick]
    with ProcessPoolExecutor(max_workers=8) as ex:
        raw = np.concatenate(list(ex.map(_load_raw, [(h5_of(sl.slide_id[i]), xy_c[r_c == i]) for i in ci])))
    print(f"controls: {len(ci)} slides, {len(pick):,} patches, raw loaded; {time.time() - t0:.0f}s", flush=True)
    from sklearn.decomposition import PCA
    fit = rng.choice(len(raw), min(PCA_FIT, len(raw)), replace=False)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=SEED).fit(raw[fit].astype(np.float32))
    X = {"raw_pca64": np.concatenate([pca.transform(b.astype(np.float32)) for b in np.array_split(raw, 20)]).astype(np.float32),
         "D_unsup": U[pick].astype(np.float32), "D13_unsup13": U[pick, :13].astype(np.float32), "C_lda": Lp[pick].astype(np.float32)}
    del raw
    from lib.clustering import kmeans_lloyd
    cl = {m: cl_all[m][pick] for m in MAPS}
    cl["raw_pca64"], _, _ = kmeans_lloyd(X["raw_pca64"], K, SEED, device)
    cl["raw_pca64"] = np.asarray(cl["raw_pca64"]).astype(np.int64)
    print(f"raw PCA64 + k-means; {time.time() - t0:.0f}s", flush=True)

    slide_c = np.searchsorted(ci, r_c)                 # 対照群内のslide番号（0..）
    e_c, s_c = exp_code[r_c], scan_code[r_c]
    sm = s_c >= 0
    e_sl, s_sl = exp_code[ci], scan_code[ci]
    pair_kind = np.where(e_sl[:, None] == e_sl[None, :], "same_experiment",
                         np.where(s_sl[:, None] == s_sl[None, :], "other_experiment_same_scanner", "other_experiment_other_scanner"))
    iu = np.triu_indices(len(ci), 1)
    okp = (s_sl[iu[0]] >= 0) & (s_sl[iu[1]] >= 0)
    summary, js_rows = [], []
    for r in REPS:
        o_e, x_e = knn_mix(X[r], slide_c, e_c, device, rng)
        o_s, x_s = knn_mix(X[r][sm], slide_c[sm], s_c[sm], device, rng)
        P = np.zeros((len(ci), K))
        np.add.at(P, (slide_c, cl[r]), 1)
        P /= P.sum(1, keepdims=True)
        D = js_matrix(P)[iu]
        js = {k_: float(D[okp & (pair_kind[iu] == k_)].mean()) for k_ in ["same_experiment", "other_experiment_same_scanner", "other_experiment_other_scanner"]}
        js_rows += [{"rep": r, "pair": k_, "JS": v, "n_pairs": int((okp & (pair_kind[iu] == k_)).sum())} for k_, v in js.items()]
        for gname, g in [("experiment", e_c), ("scanner", s_c)]:
            sp = spread(cl[r][g >= 0], g[g >= 0], K)
            sp.insert(0, "group", gname); sp.insert(0, "rep", r); sp.insert(0, "slides", "controls")
            rows.append(sp)
        spe = rows[-2]
        summary.append({"rep": r, "knn_other_experiment": o_e, "knn_other_experiment_expected": x_e, "knn_other_scanner": o_s, "knn_other_scanner_expected": x_s,
                        **{f"JS_{k_}": v for k_, v in js.items()},
                        "proto_top_experiment_share_median": float(spe.top_share.median()), "proto_n_experiments_80pct_median": float(spe.n_groups_80pct.median()),
                        "n_protos_top_experiment_share_ge_0.5": int((spe.top_share >= 0.5).sum()),
                        "proto_top_scanner_share_median": float(rows[-1].top_share.median()), "n_protos_top_scanner_share_ge_0.9": int((rows[-1].top_share >= 0.9).sum())})
        print(f"{r}: {summary[-1]}; {time.time() - t0:.0f}s", flush=True)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary_controls.csv", index=False)
    pd.DataFrame(js_rows).to_csv(OUT / "js_distance_controls.csv", index=False)
    sp_all = pd.concat(rows)
    sp_all.to_csv(OUT / "prototype_spread.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))
    print(sp_all.groupby(["slides", "group", "rep"]).agg(top_share_median=("top_share", "median"), n80_median=("n_groups_80pct", "median")).round(3).to_string())

    # 3. 図: 近傍の混ざり（観測 vs 期待）、JS、プロトタイプの試験への偏り、t-SNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    xs = np.arange(len(REPS))
    for ax, key, ttl in [(axes[0], "experiment", "kNN (30): share of neighbours from another experiment"), (axes[1], "scanner", "kNN (30): share of neighbours from another scanner")]:
        ax.bar(xs, summary[f"knn_other_{key}"], color=[COL[r] for r in REPS], width=0.6)
        ax.hlines(summary[f"knn_other_{key}_expected"], xs - 0.35, xs + 0.35, color="black", lw=1.5, label="expected if fully mixed")
        ax.set_ylim(0, 1); ax.set_title(ttl, fontsize=9); ax.legend(fontsize=8, frameon=False)
    w = 0.26
    for j, (k_, lab) in enumerate([("same_experiment", "same experiment"), ("other_experiment_same_scanner", "other exp., same scanner"), ("other_experiment_other_scanner", "other exp., other scanner")]):
        axes[2].bar(xs + (j - 1) * w, summary[f"JS_{k_}"], width=w, color=[COL[r] for r in REPS], alpha=[0.35, 0.65, 1.0][j], label=lab)
    axes[2].set_title("JS distance between control slides' prototype distributions", fontsize=9); axes[2].legend(fontsize=8, frameon=False)
    spc = sp_all[(sp_all.slides == "controls") & (sp_all.group == "experiment")]
    axes[3].boxplot([spc[spc.rep == r].top_share for r in REPS], positions=xs, widths=0.5, showfliers=False)
    axes[3].axhline(1 / len(np.unique(e_sl)), color="black", lw=1, ls="--", label="expected if fully mixed")
    axes[3].set_ylim(0, 1); axes[3].set_title("per prototype: largest experiment's share (size-balanced)", fontsize=9); axes[3].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.set_xticks(xs); ax.set_xticklabels([TITLES[r] for r in REPS], fontsize=8, rotation=15, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"TG-GATEs control slides ({len(ci)} slides, {len(np.unique(e_sl))} experiments, {PER_SLIDE} patches/slide)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "F1_batch_mixing_controls.png", dpi=150)
    plt.close(fig)

    tp = np.concatenate([rng.choice(np.where(e_c == e)[0], min(TSNE_PER_EXP, int((e_c == e).sum())), replace=False) for e in np.unique(e_c)])
    fig, axes = plt.subplots(2, len(REPS), figsize=(4.2 * len(REPS), 8.4))
    cmap = plt.get_cmap("tab20")
    for j, r in enumerate(REPS):
        Y = TSNE(perplexity=30, init="pca", random_state=SEED).fit_transform(X[r][tp])
        for e in np.unique(e_c[tp]):
            mm = e_c[tp] == e
            axes[0, j].scatter(Y[mm, 0], Y[mm, 1], s=2, alpha=0.5, color=cmap(e % 20))
        for si, (s_, c_) in enumerate([(0, "#0072B2"), (1, "#D55E00")]):
            mm = s_c[tp] == si
            axes[1, j].scatter(Y[mm, 0], Y[mm, 1], s=2, alpha=0.4, color=c_, label=SCANNERS[si])
        axes[0, j].set_title(f"{TITLES[r]}\ncolour = experiment ({len(np.unique(e_c))}, colours repeat)", fontsize=9)
        axes[1, j].set_title("colour = scanner", fontsize=9); axes[1, j].legend(fontsize=8, frameon=False, markerscale=5)
        for ax in axes[:, j]:
            ax.set_xticks([]); ax.set_yticks([])
        print(f"t-SNE {r}; {time.time() - t0:.0f}s", flush=True)
    fig.suptitle(f"t-SNE of TG-GATEs control patches ({TSNE_PER_EXP} per experiment)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "F2_tsne_controls.png", dpi=130)
    print(f"done; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
