"""セミナー用: 補正なしではプロトタイプが slide・試験をまたがず、slide 内の順位変換と LDA でまたぐようになることを示す図（2026-11-10 ユーザー）。
計算は 0030 v4 の batch_mixing（scripts/tggates_unseen_batch_mixing.py）が保存した対照群の表現をそのまま使う。新しい特徴量は作らない。

対照群 857 枚（43 試験）、各 slide の同じ 1,024 patch。表現は 3 つ:
  補正なし   : UNI2-h → PCA64、プロトタイプは対照群 patch の k-means（k=100、tggates_unseen_batch_mixing.py と同じ seed で作り直す）
  順位変換   : slide 内の順位変換 → PCA64（D）、プロトタイプは 0032 の重心（Bayer の slide で作ったもの）
  ＋LDA      : D を Bayer のラベルで LDA 13 次元（C）、プロトタイプは 0032 の重心
F1: (a) プロトタイプごとの、最も多い試験の占有率（試験の大きさで均した値）。完全に混ざれば 1/43
    (b) 対照群の slide 2 枚の、プロトタイプ構成の重なり（Σ min(p, q)）。同じ試験 / 別の試験・同じスキャナ / 別の試験・別のスキャナ
    検定（2026-11-10 ユーザー: 有意差に＊）: (a) プロトタイプを単位に Mann-Whitney U（両側、3 比較を Holm）。
    (b) 組は slide を共有して独立でないので、試験ごとに重なりの平均をとり（その試験の slide を含む組の平均）、
        試験を対応にした Wilcoxon 符号順位検定（両側、隣り合う表現の 2 比較 × 3 種類 = 6 比較を Holm）。* p<.05、** p<.01、*** p<.001
F2: 別の試験・別のスキャナ・同じ解剖時点の対照群 2 枚の、全 patch のプロトタイプのマップ（3 表現）。2 枚は、補正なしの重なりが
    その組み合わせ全体の中央値に最も近い組を選ぶ（作為的に選ばないため）。
    色 = そのプロトタイプが 2 枚のどちらに偏るか（左の slide での割合 /（左 + 右））。橙 = 左にだけ、灰 = 両方、青 = 右にだけ
    全 patch の割り当て: 補正なしは PCA64 を tggates_unseen_batch_mixing.py と同じ乱数で作り直し、k-means の重心への最近傍。
    D・C は 0030 と同じ手順（slide 内の順位 → 0021 の PCA256、LDA は 0030 と同じ設定で学び直す）で、0032 の重心への最近傍
F3: F2 の補正なしの行だけ（2026-11-10 ユーザー: スライドをまたがない図は全 patch で）
出力: outputs/0030_20261028_tggates_unseen_findings/v4/batch_mixing/seminar/
usage: PYTHONPATH=.:scripts python scripts/seminar_prototype_mixing_figure.py   （コンテナ内、Slurm経由。CPU、20 分程度）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/0030_20261028_tggates_unseen_findings/v4"
BM = RUN / "batch_mixing"
OUT = BM / "seminar"
PER_SLIDE, K, SEED = 1024, 100, 0                # tggates_unseen_batch_mixing.py と同じ
REPS = [("raw_pca64", "補正なし\n（UNI2-h → PCA64）"), ("D_unsup", "slide 内の順位変換\n（→ PCA64）"), ("C_lda", "順位変換 ＋ LDA\n（13 次元）")]
COL = {"raw_pca64": "#999999", "D_unsup": "#0072B2", "C_lda": "#D55E00"}
PAIR = [("same_exp", "同じ試験"), ("other_exp_same_scanner", "別の試験\n同じスキャナ"), ("other_exp_other_scanner", "別の試験\n別のスキャナ")]
SIDE_COL = ["#0072B2", "#DDDDDD", "#D55E00"]   # 右にだけ ← 両方 → 左にだけ（Okabe-Ito の青と朱、中央は灰）
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
FULL_DIR = "outputs/0016_20261013_tier1_macenko_embeddings/20x_256px_0px_overlap/features_uni_v2_macenko"
WSI_DIRS = [ROOT / "data/TGGATEs/WSI/Liver", Path("/workspace/andre01/honzawa/00-utils/hdd/TGGATEs/WSI/Liver")]


def main():
    from lib.clustering import kmeans_lloyd
    OUT.mkdir(exist_ok=True)
    sl = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str, "EXP_ID": str})
    A, F = np.load(RUN / "assignments.npz"), np.load(RUN / "features_sub.npz")
    R = np.load(BM / "reps_controls.npz")
    row, xy = A["row"], F["xy"]
    # tggates_unseen_batch_mixing.py と同じ乱数の順で、対照群の 1,024 patch を選び直す（座標を得るため）
    rng = np.random.default_rng(SEED)
    ci = np.where(sl.is_ctrl.to_numpy())[0]
    pick = np.concatenate([rng.choice(np.where(row == i)[0], min(PER_SLIDE, int((row == i).sum())), replace=False) for i in ci])
    pick.sort()
    assert np.allclose(F["U"][pick][:50].astype(np.float32), R["X_D_unsup"][:50]), "patch selection does not match reps_controls.npz"
    slide_c, xy_c = R["slide"], xy[pick]
    cl = {"D_unsup": A["cl_D_unsup"][pick].astype(np.int64), "C_lda": A["cl_C_lda"][pick].astype(np.int64)}
    cl["raw_pca64"] = np.asarray(kmeans_lloyd(R["X_raw_pca64"], K, SEED, torch.device("cpu"))[0]).astype(np.int64)
    ns = int(slide_c.max()) + 1
    P = {r: np.bincount(cl[r] * ns + slide_c, minlength=K * ns).reshape(K, ns).T / np.bincount(slide_c)[:, None] for r, _ in REPS}
    exp_sl, scan_sl = sl.EXP_ID.to_numpy()[ci], sl.scanscope_id.to_numpy()[ci]
    period = sl.SACRIFICE_PERIOD.to_numpy()[ci]
    iu = np.triu_indices(ns, 1)
    kind = np.where(exp_sl[iu[0]] == exp_sl[iu[1]], "same_exp", np.where(scan_sl[iu[0]] == scan_sl[iu[1]], "other_exp_same_scanner", "other_exp_other_scanner"))
    ov = {r: np.minimum(P[r][iu[0]], P[r][iu[1]]).sum(1) for r, _ in REPS}
    sp = pd.read_csv(BM / "prototype_spread.csv")
    sp = sp[(sp.slides == "controls") & (sp.group == "experiment")]
    n_exp = len(np.unique(exp_sl))
    rows = []
    for r, lab in REPS:
        s = sp[sp.rep == r]
        rows.append({"rep": r, "top_experiment_share_median": s.top_share.median(), "n_experiments_80pct_median": s.n_groups_80pct.median(),
                     **{f"overlap_median_{k}": float(np.median(ov[r][kind == k])) for k, _ in PAIR}})
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    # 検定
    from scipy.stats import mannwhitneyu, wilcoxon

    def holm(ps):
        """Holm の補正（statsmodels が環境にないため）。"""
        ps = np.asarray(ps, float)
        o = np.argsort(ps)
        adj = np.maximum.accumulate((len(ps) - np.arange(len(ps))) * ps[o])
        out = np.empty_like(ps)
        out[o] = np.minimum(adj, 1)
        return out
    tests = []
    for i1, i2 in [(0, 1), (1, 2), (0, 2)]:
        r1, r2 = REPS[i1][0], REPS[i2][0]
        u = mannwhitneyu(sp[sp.rep == r1].top_share, sp[sp.rep == r2].top_share, alternative="two-sided")
        tests.append({"panel": "a", "kind": "", "rep1": r1, "rep2": r2, "i1": i1, "i2": i2, "stat": u.statistic, "p": u.pvalue,
                      "effect_rank_biserial": 1 - 2 * u.statistic / (100 * 100), "n": 100})
    pa_ = holm([x["p"] for x in tests])
    for x, q in zip(tests, pa_):
        x["p_holm"] = q
    exps = np.unique(exp_sl)
    ex_mean = {}
    for k, _ in PAIR:
        mk = kind == k
        for r, _ in REPS:
            v = []
            for e in exps:
                me = mk & ((exp_sl[iu[0]] == e) | (exp_sl[iu[1]] == e))
                v.append(ov[r][me].mean() if me.any() else np.nan)
            ex_mean[(k, r)] = np.array(v)
    tb = []
    for k, _ in PAIR:
        for i1, i2 in [(0, 1), (1, 2)]:
            x1, x2 = ex_mean[(k, REPS[i1][0])], ex_mean[(k, REPS[i2][0])]
            ok = np.isfinite(x1) & np.isfinite(x2)
            w = wilcoxon(x1[ok], x2[ok], alternative="two-sided")
            tb.append({"panel": "b", "kind": k, "rep1": REPS[i1][0], "rep2": REPS[i2][0], "i1": i1, "i2": i2, "stat": w.statistic, "p": w.pvalue,
                       "median_diff": float(np.median(x2[ok] - x1[ok])), "n": int(ok.sum())})
    for x, q in zip(tb, holm([x["p"] for x in tb])):
        x["p_holm"] = q
    pd.DataFrame(tests + tb).to_csv(OUT / "tests.csv", index=False)
    print(pd.DataFrame(tests + tb)[["panel", "kind", "rep1", "rep2", "n", "p_holm"]].to_string(index=False), flush=True)
    star = lambda q: "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "n.s."

    def bracket(ax, x1, x2, y, txt, h=0.015):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=0.8)
        ax.text((x1 + x2) / 2, y + h, txt, ha="center", va="bottom", fontsize=9)

    # F1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), layout="constrained")
    ax = axes[0]
    jit = np.random.default_rng(1)
    for j, (r, lab) in enumerate(REPS):
        v = sp[sp.rep == r].top_share.to_numpy()
        ax.scatter(j + jit.uniform(-0.18, 0.18, len(v)), v, s=10, color=COL[r], alpha=0.6, edgecolors="none")
        ax.boxplot(v, positions=[j], widths=0.5, showfliers=False, medianprops={"color": "black"}, whiskerprops={"color": "#555555"}, capprops={"color": "#555555"}, boxprops={"color": "#555555"})
    ax.axhline(1 / n_exp, color="black", lw=0.8, ls="--")
    ax.text(2.45, 0.6, f"破線 = 完全に混ざったとき（1/{n_exp}）", ha="right", fontsize=9)
    for x, y in zip(tests, (0.84, 0.84, 0.95)):
        bracket(ax, x["i1"], x["i2"], y, star(x["p_holm"]))
    ax.set(xticks=range(len(REPS)), xticklabels=[f"{l}\n中央値 {sp[sp.rep == r].top_share.median():.2f}" for r, l in REPS], ylim=(0, 1.12), title="(a) プロトタイプ 1 個ごとの、試験への偏り（1 点 = 1 プロトタイプ）")
    ax.set_ylabel("最も多い試験の占有率（試験の大きさで均した値）", labelpad=8)
    ax = axes[1]
    w = 0.26
    for j, (r, lab) in enumerate(REPS):
        for m, (k, kl) in enumerate(PAIR):
            ax.boxplot(ov[r][kind == k], positions=[m + (j - 1) * w], widths=w * 0.85, showfliers=False, patch_artist=True,
                       medianprops={"color": "black"}, boxprops={"facecolor": COL[r], "edgecolor": "#333333"})
    for x in tb:
        m = [k for k, _ in PAIR].index(x["kind"])
        bracket(ax, m + (x["i1"] - 1) * w, m + (x["i2"] - 1) * w, 0.97 + 0.06 * (x["i1"]), star(x["p_holm"]))
    ax.set(xticks=range(len(PAIR)), xticklabels=[kl for _, kl in PAIR], ylim=(0, 1.13), yticks=np.arange(0, 1.01, 0.2), title="(b) 対照群 slide 2 枚の組ごとの、プロトタイプ構成の重なり")
    ax.set_ylabel("2 枚の構成の重なり Σ min(p, q)（1 = 同じ構成）", labelpad=8)
    ax.legend(handles=[matplotlib.patches.Patch(facecolor=COL[r], edgecolor="#333333", label=l.replace("\n", "")) for r, l in REPS], loc="lower left", fontsize=9,
              frameon=True, facecolor="white", edgecolor="#999999", framealpha=1)
    fig.suptitle(f"TG-GATEs の対照群 {ns} 枚（{n_exp} 試験）、各 slide {PER_SLIDE:,} patch、プロトタイプ k = {K}\n"
                 f"(a) Mann-Whitney U、(b) 試験ごとの平均（{n_exp} 試験）で Wilcoxon 符号順位検定、いずれも両側・Holm 補正。* p<.05、** p<.01、*** p<.001", fontsize=10)
    fig.savefig(OUT / "F1_prototype_spread.png", dpi=200)
    plt.close(fig)

    # 2 枚: 別の試験・別のスキャナ・同じ解剖時点で、補正なしの重なりが中央値に最も近い組
    m_ = (kind == "other_exp_other_scanner") & (period[iu[0]] == period[iu[1]])
    cand = np.where(m_)[0]
    best = cand[np.argmin(np.abs(ov["raw_pca64"][cand] - np.median(ov["raw_pca64"][kind == "other_exp_other_scanner"])))]
    a, b = iu[0][best], iu[1][best]
    sids = {s_: sl.slide_id.iloc[ci[s_]] for s_ in (a, b)}

    # 全 patch の割り当て（3 表現）
    import h5py
    import openslide
    import yaml
    from concurrent.futures import ProcessPoolExecutor
    from sklearn.decomposition import PCA
    import tggates_unseen_batch_mixing as bm

    def h5_of(sid):
        for d_ in bm._CFG["features_dirs"]:
            if (ROOT / d_ / f"{sid}.h5").exists():
                return str(ROOT / d_ / f"{sid}.h5")
        raise FileNotFoundError(sid)
    r_c = row[pick]
    with ProcessPoolExecutor(max_workers=8) as ex:
        raw = np.concatenate(list(ex.map(bm._load_raw, [(h5_of(sl.slide_id.iloc[i]), xy_c[r_c == i]) for i in ci])))
    fit = rng.choice(len(raw), min(bm.PCA_FIT, len(raw)), replace=False)
    pca = PCA(n_components=64, svd_solver="randomized", random_state=SEED).fit(raw[fit].astype(np.float32))
    assert np.allclose(pca.transform(raw[:50].astype(np.float32)), R["X_raw_pca64"][:50], atol=1e-2), "PCA does not match reps_controls.npz"
    del raw
    Xr = R["X_raw_pca64"]
    cen = {"raw_pca64": np.stack([Xr[cl["raw_pca64"] == k].mean(0) for k in range(K)])}
    # D・C: 0030 と同じ前処理と LDA
    import importlib.util
    import pickle

    def load_module(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    c30 = yaml.safe_load(open(ROOT / "experiments/0030_20261028_tggates_unseen_findings/config.yml"))
    c30 = {**c30, **c30["variants"]["v4"]}
    load_module(ROOT / "experiments" / c30["exp0021"] / "experiment.py", "exp0021")
    ex23 = load_module(ROOT / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex26 = load_module(ROOT / "experiments" / c30["exp0026"] / "experiment.py", "exp0026")
    cfg20 = yaml.safe_load(open(ROOT / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    src = ROOT / "outputs" / c30["exp0025"] / c30["cache_from"]
    c = np.load(src / "cache.npz")
    Ka, laba = c["K"], c["lab"].astype(np.int64)
    keep = [k for k in np.unique(laba) if k >= 0 and (laba == k).sum() >= cfg20["min_label_patches"]]
    laba = np.where(np.isin(laba, keep), laba, -1)
    lab_c = np.where(laba >= 0, np.searchsorted(keep, np.maximum(laba, 0)), -1)
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    ex_ids = pd.read_csv(ROOT / c30["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab_c = np.where(np.isin(bayer.slide_id.to_numpy()[c["row"]], ex_ids), -1, lab_c)
    tf = ex26.fit_combo(Ka, lab_c, c30, c30["seed"])
    L = len(keep) - 1
    del Ka
    pre = pickle.loads((ROOT / "outputs" / c30["exp0021"] / "v3_maps/models.pkl").read_bytes())["pre"]
    proto = ROOT / c30["proto_root"]
    cen["D_unsup"] = np.load(proto / "centroids_unsup_k100.npy").astype(np.float32)
    cen["C_lda"] = np.load(proto / "centroids_lda_k100.npy").astype(np.float32)

    def nearest(z, c_):
        zt, ct = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)), torch.from_numpy(c_)
        return np.concatenate([torch.cdist(zz, ct).argmin(1).numpy() for zz in torch.split(zt, 4096)])
    full = {}
    for s_ in (a, b):
        with h5py.File(ROOT / FULL_DIR / f"{sids[s_]}.h5", "r") as h:
            f_, c_ = h["features"][:].astype(np.float32), h["coords"][:]
            ps = int(h["coords"].attrs.get("patch_size_level0", 256))
        Kf = pre.transform(ex23.slide_rank(f_)).astype(np.float32)
        lab = {"raw_pca64": nearest(pca.transform(f_), cen["raw_pca64"]), "D_unsup": nearest(Kf[:, :64], cen["D_unsup"]), "C_lda": nearest(tf(Kf)[:, :L], cen["C_lda"])}
        # 確認: 0030 の間引き patch（同じ座標）の割り当てと、全 patch での割り当ての一致率
        key = {tuple(p_): i for i, p_ in enumerate(c_.tolist())}
        sub = np.where(row == ci[s_])[0]
        idx = np.array([key.get(tuple(p_), -1) for p_ in xy[sub].tolist()])
        ok = idx >= 0
        print(sids[s_], len(c_), "patches; agreement with 0030 on", int(ok.sum()), "patches:",
              {r: round(float((lab[r][idx[ok]] == A[f"cl_{r}"][sub][ok]).mean()), 3) for r in ("D_unsup", "C_lda")}, flush=True)
        full[s_] = (c_, ps, lab)

    thumbs = {}
    for s_ in (a, b):
        hit = [p_ for d in WSI_DIRS for p_ in d.glob(f"*/{sids[s_]}.svs")]
        o = openslide.OpenSlide(str(hit[0]))
        th = o.get_thumbnail((1400, 1400)).convert("L")
        thumbs[s_] = (np.asarray(th), th.size[0] / o.dimensions[0])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("side", SIDE_COL)

    def draw(reps, fname, title):
        fig, axes = plt.subplots(len(reps), 2, figsize=(12, 3.9 * len(reps) + 1.0), layout="constrained", squeeze=False)
        for i, (r, lab) in enumerate(reps):
            pa, pb = [np.bincount(full[s_][2][r], minlength=K) / len(full[s_][2][r]) for s_ in (a, b)]
            side = np.where(pa + pb > 0, pa / np.maximum(pa + pb, 1e-12), 0.5)
            for j, s_ in enumerate((a, b)):
                ax = axes[i, j]
                c_, ps, lb = full[s_]
                img, sc = thumbs[s_]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255, alpha=0.25)
                g = (c_ // ps).astype(int)
                H = np.full((g[:, 1].max() + 1, g[:, 0].max() + 1, 4), 0.0)
                H[g[:, 1], g[:, 0]] = cmap(side[lb[r]])
                ax.imshow(H, extent=(0, H.shape[1] * ps * sc, H.shape[0] * ps * sc, 0), interpolation="nearest")
                ax.set(xticks=[], yticks=[], xlim=(0, img.shape[1]), ylim=(img.shape[0], 0))
                for s in ax.spines.values():
                    s.set_visible(False)
                k_ = ci[s_]
                if i == 0:
                    ax.set_title(f"{'左' if j == 0 else '右'}: {sl.COMPOUND_NAME.iloc[k_]} 試験（{sl.EXP_ID.iloc[k_]}）の対照群、{sl.SACRIFICE_PERIOD.iloc[k_]}\n"
                                 f"スキャナ {sl.scanscope_id.iloc[k_]}、slide {sids[s_]}、{len(lb[r]):,} patch", fontsize=10)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=11)
            axes[i, 1].text(1.02, 0.5, f"2 枚の重なり\n{np.minimum(pa, pb).sum():.2f}", transform=axes[i, 1].transAxes, va="center", fontsize=11)
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1)), ax=axes, orientation="horizontal", shrink=0.5, aspect=40, pad=0.02)
        cb.set_ticks([0, 0.5, 1], labels=["右の slide にだけ出る\nプロトタイプ", "両方に同じくらい出る", "左の slide にだけ出る\nプロトタイプ"])
        fig.suptitle(title, fontsize=11)
        fig.savefig(OUT / fname, dpi=200)
        plt.close(fig)
    draw(REPS, "F2_control_maps.png", "同じ「正常」の肝臓 2 枚の全 patch を、プロトタイプがどちらの slide に偏るかで色分け\n"
         "2 枚は、別の試験・別のスキャナ・同じ解剖時点の組のうち、補正なしの重なりが全体の中央値に最も近いもの")
    draw(REPS[:1], "F3_control_maps_raw_fullpatch.png", "補正なし（UNI2-h → PCA64、k = 100）では、同じ「正常」の肝臓 2 枚がほとんど別のプロトタイプで埋まる\n"
         "全 patch を、プロトタイプがどちらの slide に偏るかで色分け")
    print("pair", sids[a], sids[b], {r: round(float(np.minimum(P[r][a], P[r][b]).sum()), 3) for r, _ in REPS}, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
