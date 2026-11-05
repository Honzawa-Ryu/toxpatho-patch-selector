"""対照群の見分けやすさで試験を「似ている／似ていない」にまとめる（2026-11-05ユーザー: 各塊の評価をまず出し、そのあと各試験単位で似てる似てないを出す）。

全 143 試験（0015 の対照群の調査の範囲）の対照群 slide（各 2,048 patch、0016 で抽出済み）を、0035 のプロトタイプに割り当てる:
  B128（白色化128、ラベルなし、0035 v2 の split_record。重心は 0035 の割当から作り直す）、A128（白色化128 ‖ LDA、0035 v1 の split_record）、C（LDA 13 のみ、0032 の k=100）
1. 0015 の塊（cut 0.72）ごとに、塊の中だけで対照群から試験を当てる（scripts/control_study_identification.py の identify と同じ手順）
2. 試験の組ごとに、2 つの試験の対照群をプロトタイプの構成比で見分けられるか（slide 単位の 5 分割、AUC）。偶然の分布は組の中でラベルを入れ替えて作る
3. 見分けやすさ d = max(AUC − 0.5, 0) / 0.5 を距離にして平均連結の系統樹、偶然の 95% 点で切ってまとまりを作る。0015 の塊・スキャナ・投与経路・単回/反復と突き合わせる
出力: outputs/0035_20261102_finer_prototypes_split_rejoin/control_similarity/
usage: python scripts/control_similarity_groups.py   （コンテナ内、Slurm経由。CPU、30〜60 分）
"""
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R35 = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin"
OUT = R35 / "control_similarity"
BLOCKS = ROOT / "experiments/0015_20261012_ctrl_batch_structure/exp_blocks_preview_allctrl_macenko_20261016_cut0.72/exp_blocks.csv"
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
FOLDS, LR_UP, SEED, N_NULL_PAIRS, N_PERM = 5, 1.0, 0, 300, 20
OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pair_auc(Pa, Pb, rng, perm=False):
    """2 つの試験の対照群（Pa, Pb: slide × K）を、学習側で選んだ「a に多い／b に多い」プロトタイプの割合の差で判別した AUC（slide 単位の 5 分割）。"""
    from sklearn.metrics import roc_auc_score
    X = np.r_[Pa, Pb]
    y = np.r_[np.ones(len(Pa)), np.zeros(len(Pb))]
    if perm:
        y = rng.permutation(y)
    fold = rng.permutation(np.arange(len(y)) % FOLDS)
    s = np.full(len(y), np.nan)
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        a_, b_ = tr & (y == 1), tr & (y == 0)
        if not a_.any() or not b_.any() or not te.any():
            continue
        ma, mb = X[a_].mean(0) + 1e-4, X[b_].mean(0) + 1e-4
        lr = np.log2(ma / mb)
        s[te] = X[te][:, lr >= LR_UP].sum(1) - X[te][:, lr <= -LR_UP].sum(1)
    ok = np.isfinite(s)
    return roc_auc_score(y[ok], s[ok]) if len(set(y[ok])) == 2 else np.nan


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin/config.yml"))
    load_module(ROOT / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(ROOT / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex26 = load_module(ROOT / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(ROOT / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    ex30 = load_module(ROOT / "experiments" / cfg["exp0030"] / "experiment.py", "exp0030")
    csi = load_module(ROOT / "scripts/control_study_identification.py", "csi")
    cfg20 = yaml.safe_load(open(ROOT / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    pre_path = str(ROOT / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl")
    rng = np.random.default_rng(SEED)

    def h5_of(sid):
        for d_ in cfg["features_dirs"]:
            if (ROOT / d_ / f"{sid}.h5").exists():
                return str(ROOT / d_ / f"{sid}.h5")
        return ""

    # 0. Bayer の LDA（0035 と同じ）と、B128 の split_record の重心（0035 の割当から、集合ごとに白色化して作り直す）
    src = ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Kb, laba, row_b = c["K"], c["lab"].astype(np.int64), c["row"]
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    keep = [k_ for k_ in np.unique(laba) if k_ >= 0 and (laba == k_).sum() >= cfg20["min_label_patches"]]
    lab = np.where(np.isin(laba, keep), np.searchsorted(keep, np.maximum(laba, 0)), -1)
    ex_ids = pd.read_csv(ROOT / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab = np.where(np.isin(bayer.slide_id.to_numpy()[row_b], ex_ids), -1, lab)
    tf = ex26.fit_combo(Kb, lab, cfg, cfg["seed"])
    L = len(keep) - 1
    S35 = pd.read_csv(R35 / "v1/slides.csv", dtype={"slide_id": str})
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[q.EXP_ID.notna() & (q.EXP_ID != "")].copy()
    q["exp"] = q.EXP_ID.astype(int)
    tgt = ex26.target_slides({**cfg, "target_exps": cfg["fatty_exps"], "features_dir": cfg["features_dirs"][0]}, ROOT)
    tgt = tgt[~tgt.slide_id.isin(bayer.slide_id)].reset_index(drop=True)
    exp_b = bayer.EXP_ID.fillna("-1").astype(int).to_numpy()[row_b]
    exp_ts = np.repeat(tgt.exp.to_numpy(), cfg["patches_per_slide"])
    qc0 = q[(q.DOSE_LEVEL == "Control") & q.exp.isin(set(exp_b) | set(exp_ts))].copy()
    qc0["h5"] = [h5_of(x) for x in qc0.slide_id]
    qc0 = qc0[qc0.h5 != ""]
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts_c = list(ex.map(ex27._load_ctrl, qc0.h5, chunksize=4))
    Kc = np.concatenate(parts_c)[:, :128].astype(np.float32)
    exp_c = np.concatenate([np.full(len(p_), int(e)) for p_, e in zip(parts_c, qc0.EXP_ID)])
    del parts_c

    def sphere_known(U, e):
        e_fix = np.where(np.isin(e, exp_c), e, -999)
        return ex27.sphering(U, e_fix, np.r_[Kc, Kc], np.r_[exp_c, np.full(len(exp_c), -999)]).astype(np.float32)
    clB = np.load(R35 / "v2/assign_B128_sphering_split_record.npy").astype(np.int64)
    KB = int(clB.max()) + 1
    sumB = np.zeros((KB, 128))
    nb = len(Kb)

    def add_sum(cl_, X_):
        return torch.zeros(KB, 128, dtype=torch.float64).index_add_(0, torch.from_numpy(cl_), torch.from_numpy(X_).double()).numpy()
    # 白色化は試験ごとなので、試験の境で分けずにまとめて行ってよい（ex27.sphering は試験ごとに処理する）
    sumB += add_sum(clB[:nb], sphere_known(Kb[:, :128].astype(np.float32), exp_b))
    del Kb
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        Kt_s = np.concatenate([p_[0] for p_ in ex.map(ex30._load_sub, tgt.h5, chunksize=4)]).astype(np.float32)
    nt = len(Kt_s)
    sumB += add_sum(clB[nb:nb + nt], sphere_known(Kt_s[:, :128], exp_ts))
    del Kt_s
    z = np.load(ROOT / cfg["unseen_features_from"], allow_pickle=True)
    Ku, row_u = z["K"], z["row"]
    del z
    sl_u = S35[S35.set == "unseen"].reset_index(drop=True)
    exp_u = sl_u.exp.to_numpy()[row_u]
    cm = sl_u.is_ctrl.to_numpy()[row_u]
    Uu = Ku[:, :128].astype(np.float32)
    del Ku
    e_fix = np.where(np.isin(exp_u, exp_u[cm]), exp_u, -999)
    Sw_u = ex27.sphering(Uu, e_fix, np.r_[Uu[cm], Uu[cm]], np.r_[exp_u[cm], np.full(int(cm.sum()), -999)]).astype(np.float32)
    del Uu
    sumB += add_sum(clB[nb + nt:], Sw_u)
    del Sw_u
    cenB = (sumB / np.maximum(np.bincount(clB, minlength=KB), 1)[:, None]).astype(np.float32)
    np.save(OUT / "centroids_B128_split_record.npy", cenB)
    print(f"B128 centroids rebuilt ({KB})", flush=True)

    # 1. 全 143 試験の対照群（2,048 patch）を割り当てる
    blocks = pd.read_csv(BLOCKS)
    qc = q[(q.DOSE_LEVEL == "Control") & q.exp.isin(blocks.exp_id)].copy()
    qc["h5"] = [h5_of(x) for x in qc.slide_id]
    qc = qc[qc.h5 != ""].reset_index(drop=True)
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts = list(ex.map(ex27._load_ctrl, qc.h5, chunksize=4))
    row = np.concatenate([np.full(len(p_), i) for i, p_ in enumerate(parts)])
    K = np.concatenate(parts).astype(np.float32)
    del parts
    e_ = qc.exp.to_numpy()[row]
    U = K[:, :128]
    Sw = ex27.sphering(U, e_, np.r_[U, U], np.r_[e_, np.full(len(e_), -999)]).astype(np.float32)   # 各試験の対照群そのもので白色化（0035 の未学習の集合と同じ）
    Lw = np.concatenate([tf(b_)[:, :L] for b_ in np.array_split(K, max(1, len(K) // 200000))]).astype(np.float32)
    del K, U
    fac = json.loads((ROOT / "outputs/0036_20261103_local_prototype_context/v5/a128_scale.json").read_text())["fac"]
    reps = {"B128（白色化128、ラベルなし）": (Sw, cenB),
            "A128（白色化128 ‖ LDA）": (np.concatenate([Sw, cfg["lda_beta"] * fac * Lw], 1), np.load(R35 / "v1/tree_A128_sphering_lda_split_record/centroids.npy")),
            "C（LDA 13 のみ）": (Lw, np.load(ROOT / "outputs/0032_20261029_bayer_only_prototypes/v1/centroids_lda_k100.npy"))}
    ns = len(qc)
    Ps = {}
    for name, (X, cen) in reps.items():
        ct = torch.from_numpy(cen.astype(np.float32))
        cl = torch.cat([torch.cdist(torch.from_numpy(x), ct).argmin(1) for x in np.array_split(X.astype(np.float32), max(1, len(X) // 65536))]).numpy()
        Kp = len(cen)
        P = np.bincount(cl * ns + row, minlength=Kp * ns).reshape(Kp, ns).T.astype(float)
        Ps[name] = P / np.maximum(P.sum(1, keepdims=True), 1)
        print(f"assigned {name}", flush=True)
    del Sw, Lw
    meta_i = pd.read_csv(ROOT / "data/corrected/open_tggates_individual.csv", dtype=str)
    meta_i["exp"] = pd.to_numeric(meta_i.EXP_ID, errors="coerce")
    em = meta_i.dropna(subset=["exp"]).groupby("exp").agg(route=("ADMINISTRATION_ROUTE_TYPE", "first"), single_repeat=("SINGLE_REPEAT_TYPE", "first"))
    qc["scan_month"] = pd.to_datetime(qc.scan_dt, errors="coerce").dt.strftime("%Y-%m")
    E = blocks.set_index("exp_id")[["compound", "block"]].copy()
    E["scanner"] = qc.groupby("exp").scanscope_id.agg(lambda s: s.mode().iloc[0])
    E["scan_month"] = qc.groupby("exp").scan_month.agg(lambda s: s.mode().iloc[0] if s.notna().any() else "")
    E["n_ctrl"] = qc.groupby("exp").size()
    E = E.join(em)
    E = E[E.n_ctrl >= 6]
    exps = list(E.index)
    print(f"{len(exps)} experiments, {int(E.n_ctrl.sum())} control slides", flush=True)

    # 2. 塊ごとの評価（塊の中だけで試験を当てる）
    keep_s = qc.exp.isin(exps).to_numpy()
    fold = rng.permutation(np.arange(ns) % FOLDS)
    brow = []
    for name, P in Ps.items():
        for scope, m in [("全試験", keep_s)] + [(f"塊 {b}", keep_s & qc.exp.map(E.block).eq(b).to_numpy()) for b in sorted(E.block.unique())]:
            aucs, acc = csi.identify(P[m], qc.exp.to_numpy()[m], fold[m])
            v = np.array(list(aucs.values()))
            n_e = len(aucs)
            brow.append({"representation": name, "scope": scope, "n_experiments": n_e, "auc_median": float(np.median(v)), "auc_q25": float(np.percentile(v, 25)), "auc_q75": float(np.percentile(v, 75)),
                         "top1_accuracy": acc, "chance": 1 / n_e, "acc_over_chance": acc * n_e})
            print(brow[-1], flush=True)
    BT = pd.DataFrame(brow)
    BT.to_csv(OUT / "block_identification.csv", index=False)

    # 3. 試験の組ごとの見分けやすさ（主に B128）と、偶然の分布
    pair_tabs = {}
    for name in ["B128（白色化128、ラベルなし）", "C（LDA 13 のみ）"]:
        P = Ps[name]
        idx = {e: np.where((qc.exp == e).to_numpy())[0] for e in exps}
        M = np.full((len(exps), len(exps)), np.nan)
        for i in range(len(exps)):
            for j in range(i + 1, len(exps)):
                M[i, j] = M[j, i] = pair_auc(P[idx[exps[i]]], P[idx[exps[j]]], rng)
        np.fill_diagonal(M, 0.5)
        null = []
        for _ in range(N_NULL_PAIRS):
            a, b = rng.choice(len(exps), 2, replace=False)
            null += [pair_auc(P[idx[exps[a]]], P[idx[exps[b]]], rng, perm=True) for _ in range(N_PERM)]
        null = np.array(null)
        thr = float(np.nanpercentile(null, 95))
        pd.DataFrame(M, index=exps, columns=exps).to_csv(OUT / f"pairwise_auc_{name.split('（')[0]}.csv")
        pair_tabs[name] = (M, thr, null)
        iu = np.triu_indices(len(exps), 1)
        print(f"{name}: pairwise AUC median {np.nanmedian(M[iu]):.3f}; null 95% {thr:.3f}; share of pairs <= null95 {np.nanmean(M[iu] <= thr):.3f}", flush=True)

    # 4. 系統樹とまとまり（B128）
    from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import adjusted_rand_score
    M, thr, null = pair_tabs["B128（白色化128、ラベルなし）"]
    D = np.clip((M - 0.5) / 0.5, 0, 1)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    d_thr = max((thr - 0.5) / 0.5, 0)
    grp = fcluster(Z, t=d_thr, criterion="distance")
    E["group_B128"] = grp
    order = dendrogram(Z, no_plot=True)["leaves"]
    E["tree_order"] = pd.Series(np.arange(len(exps)), index=np.array(exps)[order]).reindex(exps).to_numpy()
    gs = pd.Series(grp).value_counts()
    within = []
    for g_ in gs.index:
        ii = np.where(grp == g_)[0]
        if len(ii) > 1:
            sub = M[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)]
            within.append({"group": int(g_), "n_exps": len(ii), "within_auc_mean": float(np.nanmean(sub)), "within_auc_max": float(np.nanmax(sub)),
                           "blocks_0015": E.block.iloc[ii].value_counts().to_dict(), "scanners": E.scanner.iloc[ii].value_counts().to_dict(), "compounds": ", ".join(E.compound.iloc[ii].astype(str)[:12])})
    GT = pd.DataFrame(within).sort_values("n_exps", ascending=False)
    GT.to_csv(OUT / "groups_B128.csv", index=False)
    E.to_csv(OUT / "experiments.csv")
    ari = adjusted_rand_score(E.block, E.group_B128)
    summ = {"n_experiments": len(exps), "null95_auc_B128": thr, "d_threshold": d_thr, "n_groups": int(len(gs)), "n_singletons": int((gs == 1).sum()),
            "group_sizes": gs.sort_values(ascending=False).tolist()[:15], "ari_vs_0015_blocks": float(ari),
            "null95_auc_C": pair_tabs["C（LDA 13 のみ）"][1]}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(summ, flush=True)
    print(GT.head(15).to_string(index=False), flush=True)

    # 図
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    # F1: 塊ごとの評価
    scopes = list(dict.fromkeys(BT.scope))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), layout="constrained")
    cols = {"B128（白色化128、ラベルなし）": "#0072B2", "A128（白色化128 ‖ LDA）": "#D55E00", "C（LDA 13 のみ）": "#009E73"}
    mks = {"B128（白色化128、ラベルなし）": "o", "A128（白色化128 ‖ LDA）": "s", "C（LDA 13 のみ）": "D"}
    for k, (name, col) in enumerate(cols.items()):
        t_ = BT[BT.representation == name].set_index("scope").reindex(scopes)
        x = np.arange(len(scopes)) + (k - 1) * 0.25
        axes[0].errorbar(x, t_.auc_median, yerr=[t_.auc_median - t_.auc_q25, t_.auc_q75 - t_.auc_median], fmt=mks[name], color=col, capsize=3, label=name)
        axes[1].plot(x, t_.acc_over_chance, mks[name], color=col, label=name)
    lab_ = [f"{s}\n（{int(BT[(BT.scope == s)].n_experiments.iloc[0])} 試験）" for s in scopes]
    axes[0].axhline(0.5, color="black", lw=0.6, ls=":")
    axes[0].set(xticks=range(len(scopes)), xticklabels=lab_, ylim=(0.3, 1.0), ylabel="試験ごとの AUC（中央値と四分位）", title="塊の中だけで、対照群から試験を当てられるか（AUC）\n0.5 に近いほど、塊の中の試験は見分けられない")
    axes[1].axhline(1, color="black", lw=0.6, ls=":")
    axes[1].set(xticks=range(len(scopes)), xticklabels=lab_, ylabel="正解率 ÷ 偶然の正解率", title="最もスコアの高い試験を予測にしたときの正解率（偶然の何倍か）\n1 = 偶然と同じ")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(OUT / "F1_block_identification.png", dpi=180)
    plt.close(fig)
    # F2: 試験の組ごとの見分けやすさ（ヒートマップ＋系統樹＋帯）
    fig = plt.figure(figsize=(13, 13.5))
    axd = fig.add_axes([0.16, 0.86, 0.72, 0.1])
    dendrogram(Z, ax=axd, no_labels=True, color_threshold=d_thr, above_threshold_color="#999999")
    axd.axhline(d_thr, color="#D55E00", lw=1, ls="--")
    axd.set_ylabel("見分けやすさ d", fontsize=8)
    axd.set_xticks([])
    axd.set_title(f"試験の組ごとの、対照群の見分けやすさ（B128、{len(exps)} 試験）。赤破線 = 偶然の 95% 点で切る高さ（AUC {thr:.2f}）", fontsize=10)
    axh = fig.add_axes([0.16, 0.12, 0.72, 0.72])
    Mo = M[np.ix_(order, order)]
    im = axh.imshow(Mo, vmin=0.5, vmax=1.0, cmap="Blues", interpolation="nearest")
    axh.set_xticks([]); axh.set_yticks(range(len(exps)))
    axh.set_yticklabels([f"{E.compound.iloc[i][:14]}（{exps[i]}）" for i in order], fontsize=4)
    cax = fig.add_axes([0.9, 0.5, 0.015, 0.3])
    fig.colorbar(im, cax=cax, label="AUC（0.5 = 見分けられない、1 = 完全に見分けられる）")
    strips = [("0015 の塊", E.block.iloc[order].to_numpy(), {b: OI[(int(b) - 1) % 8] for b in E.block.unique()}),
              ("スキャナ", E.scanner.iloc[order].to_numpy(), {"SS1601": "#0072B2", "SS1598": "#D55E00", "SS1224": "#999999"}),
              ("単回 / 反復", E.single_repeat.iloc[order].fillna("?").to_numpy(), {"Single": "#56B4E9", "Repeat": "#CC79A7", "?": "#FFFFFF"}),
              ("投与経路", E.route.iloc[order].fillna("?").to_numpy(), {k: OI[i % 8] for i, k in enumerate(sorted(E.route.fillna("?").unique()))})]
    for k, (nm, vals, cmap_) in enumerate(strips):
        ax_ = fig.add_axes([0.16, 0.1 - 0.02 * (k + 1), 0.72, 0.015])
        ax_.imshow(np.array([[matplotlib.colors.to_rgb(cmap_.get(v, "#FFFFFF")) for v in vals]]), aspect="auto", interpolation="nearest")
        ax_.set_xticks([]); ax_.set_yticks([])
        ax_.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=8)
    handles = [Patch(color=OI[(int(b) - 1) % 8], label=f"塊 {b}") for b in sorted(E.block.unique())] + [Patch(color="#0072B2", label="SS1601"), Patch(color="#D55E00", label="SS1598"),
               Patch(color="#56B4E9", label="単回"), Patch(color="#CC79A7", label="反復")]
    fig.legend(handles=handles, loc="lower center", ncol=10, frameon=False, fontsize=8)
    fig.savefig(OUT / "F2_pairwise_heatmap_B128.png", dpi=170)
    plt.close(fig)
    # F3: 平面配置（多次元尺度法）、色 = 0015 の塊、印 = スキャナ
    from sklearn.manifold import MDS
    Y = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, normalized_stress="auto").fit_transform(D)
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    for b in sorted(E.block.unique()):
        for sc_, mk in (("SS1601", "o"), ("SS1598", "^"), ("SS1224", "x")):
            m_ = (E.block == b).to_numpy() & (E.scanner == sc_).to_numpy()
            if m_.any():
                ax.scatter(Y[m_, 0], Y[m_, 1], color=OI[(int(b) - 1) % 8], marker=mk, s=40, alpha=0.85, edgecolors="none")
    ax.legend(handles=[Patch(color=OI[(int(b) - 1) % 8], label=f"0015 の塊 {b}") for b in sorted(E.block.unique())] +
              [plt.Line2D([], [], marker="o", ls="", color="#555555", label="SS1601"), plt.Line2D([], [], marker="^", ls="", color="#555555", label="SS1598")], frameon=False, fontsize=8, loc="best")
    ax.set(xticks=[], yticks=[], title="試験の配置（対照群の見分けやすさを距離にした多次元尺度法、B128）\n近い = 対照群が見分けにくい。色 = 0015 の塊、印 = スキャナ")
    fig.savefig(OUT / "F3_mds_B128.png", dpi=180)
    print("done", flush=True)


if __name__ == "__main__":
    main()
