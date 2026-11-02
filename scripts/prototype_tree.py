"""0035 のプロトタイプの系統樹（2026-11-02ユーザー: 系統樹で脂肪変性のプロトタイプや p0330・p0647 がどの枝に付くかを見る。切る高さは結果を見て決める。
隣のプロトタイプが微妙なので、作り方 a・b・c もそれぞれ試し、同じ図を作る）。

表現を 0035 と同じ手順で作り直し（A128 = 白色化128 ‖ LDA13、B128 = 白色化128 のみ）、保存した割り当てから各プロトタイプの重心を計算する。系統樹の作り方:
  ward      : 重心の Ward 法（最初の版）
  a_history : 分割の履歴。split_record の葉を、同じ flat k=100 の親どうしで先につなぎ（重心の Ward）、その後で親どうしをつなぐ
  b_knn     : patch の近傍のつながり。各プロトタイプから最大 300 patch を引き、近傍 30 個がどのプロトタイプに入るかの割合を対称化・正規化して距離にし、平均連結
  c_meaning : 意味。記録の 11 所見の割合（log）と Bayer のラベルの構成を標準化し、余弦距離で平均連結
出力（outputs/0035_.../<版>/tree_<rep>_<method>/<作り方>/）:
  leaves.csv、linkage.npy、cut_heights.csv（切る高さごとの群の数）
  F1_tree_all.png  : 全体の系統樹と、葉ごとの帯（記録による名前、対照群由来の割合、最多試験の割合、脂肪変性の純プロトタイプ）
  F2_subtree_*.jpg : 脂肪変性の純プロトタイプと指定の葉を含む枝（葉 36 個以下で最大）。葉ごとに重心に近い patch 4 枚
usage: python scripts/prototype_tree.py <版> <rep> <method> [見たい葉の番号 ...]   例: v1 A128_sphering_lda split_record 330 647
       （コンテナ内、Slurm経由、GPU、メモリ 96GB 程度。記録による名前は scripts/prototype_record_names.py を先に実行）
TODO: 表現の作り直しは 0035 experiment.py の main と重複している（関数に切り出していないため）。
"""
import glob
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin"
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
ROW, DEND_W, TXT_W, N_TILES, MAX_SUB = 128, 360, 620, 4, 36
PURITY, MIN_N = 0.9, 200
# 記録による名前の色（Okabe-Ito を所見の系統でまとめる）。名前なし = 薄い灰
REC_COL = {"Degeneration, fatty": "#E69F00", "Vacuolization, cytoplasmic": "#F0E442", "Necrosis": "#D55E00", "Single cell necrosis": "#D55E00",
           "Cellular infiltration": "#0072B2", "Hematopoiesis, extramedullary": "#0072B2", "Increased mitosis": "#009E73", "Hypertrophy": "#009E73",
           "Proliferation, bile duct": "#56B4E9", "Change, eosinophilic": "#CC79A7", "Degeneration, granular, eosinophilic": "#CC79A7"}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    import torch
    from scipy.cluster.hierarchy import dendrogram, linkage
    run_key, rep, method = sys.argv[1], sys.argv[2], sys.argv[3]
    extra = [int(a) for a in sys.argv[4:]]
    RUN = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin" / run_key
    OUT = RUN / f"tree_{rep}_{method}"
    OUT.mkdir(exist_ok=True)
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    load_module(ROOT / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(ROOT / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex26 = load_module(ROOT / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(ROOT / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    ex30 = load_module(ROOT / "experiments" / cfg["exp0030"] / "experiment.py", "exp0030")
    cfg20 = yaml.safe_load(open(ROOT / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    pre_path = str(ROOT / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl")

    def h5_of(sid):
        for d_ in cfg["features_dirs"]:
            if (ROOT / d_ / f"{sid}.h5").exists():
                return str(ROOT / d_ / f"{sid}.h5")
        return ""

    # 1. 0035 と同じ並びの全 patch: Bayer（cache）→ 脂肪変性 3 試験（各 2,048）→ 未学習 3,319 枚（保存した PCA256）
    src = ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Kb, laba, row_b, xy_b = c["K"], c["lab"].astype(np.int64), c["row"], c["xy"]
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    keep = [k_ for k_ in np.unique(laba) if k_ >= 0 and (laba == k_).sum() >= cfg20["min_label_patches"]]
    lab = np.where(np.isin(laba, keep), np.searchsorted(keep, np.maximum(laba, 0)), -1)
    ex_ids = pd.read_csv(ROOT / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab = np.where(np.isin(bayer.slide_id.to_numpy()[row_b], ex_ids), -1, lab)
    nb = len(Kb)
    tgt = ex26.target_slides({**cfg, "target_exps": cfg["fatty_exps"], "features_dir": cfg["features_dirs"][0]}, ROOT)
    tgt = tgt[~tgt.slide_id.isin(bayer.slide_id)].reset_index(drop=True)
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts = list(ex.map(ex30._load_sub, tgt.h5, chunksize=4))
    Kt = np.concatenate([p_[0] for p_ in parts]).astype(np.float16)
    xy_t = np.concatenate([p_[1] for p_ in parts])
    row_t = np.concatenate([np.full(len(p_[0]), i) for i, p_ in enumerate(parts)])
    del parts
    z = np.load(ROOT / cfg["unseen_features_from"], allow_pickle=True)
    Ku, row_u, xy_u = z["K"], z["row"], z["xy"]
    del z
    S = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str})
    off_t, off_u = len(bayer), len(bayer) + len(tgt)
    slide_of = np.r_[row_b, row_t + off_t, row_u + off_u]
    XY = np.r_[xy_b, xy_t, xy_u]
    K_all = np.concatenate([Kb, Kt, Ku]).astype(np.float16)
    del Kb, Kt, Ku
    N = len(slide_of)
    exp_p = S.exp.to_numpy()[slide_of]
    set_of = np.r_[np.zeros(nb, np.int8), np.ones(len(row_t), np.int8), np.full(len(row_u), 2, np.int8)]
    print(f"patches {N:,}", flush=True)

    # 2. 表現（0035 と同じ）: 白色化 128（Bayer と脂肪変性は対照群を読み込んで、未学習は集合内の対照群で）、必要なら LDA
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[q.EXP_ID.notna() & (q.EXP_ID != "")].copy()
    qc = q[(q.DOSE_LEVEL == "Control") & q.EXP_ID.astype(int).isin(set(exp_p[set_of < 2]))].copy()
    qc["h5"] = [h5_of(x) for x in qc.slide_id]
    qc = qc[qc.h5 != ""]
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts_c = list(ex.map(ex27._load_ctrl, qc.h5, chunksize=4))
    Kc = np.concatenate(parts_c)[:, :128].astype(np.float32)
    exp_c = np.concatenate([np.full(len(p_), int(e)) for p_, e in zip(parts_c, qc.EXP_ID)])
    del parts_c
    U = K_all[:, :128].astype(np.float32)
    Sw = np.empty_like(U)
    m = set_of < 2
    e_fix = np.where(np.isin(exp_p[m], exp_c), exp_p[m], -999)
    Sw[m] = ex27.sphering(U[m], e_fix, np.r_[Kc, Kc], np.r_[exp_c, np.full(len(exp_c), -999)])
    mu = set_of == 2
    cm = mu & S.is_ctrl.to_numpy()[slide_of]
    e_fix = np.where(np.isin(exp_p[mu], exp_p[cm]), exp_p[mu], -999)
    Sw[mu] = ex27.sphering(U[mu], e_fix, np.r_[U[cm], U[cm]], np.r_[exp_p[cm], np.full(int(cm.sum()), -999)])
    del U, Kc
    if rep.startswith("A"):
        tf = ex26.fit_combo(K_all[:nb], lab, cfg, cfg["seed"])
        L = len(keep) - 1
        Lw = np.concatenate([tf(b_)[:, :L] for b_ in np.array_split(K_all, max(1, N // 200000))]).astype(np.float32)
        fac = float(Sw.std(0).mean() / Lw.std(0).mean())
        X = np.concatenate([Sw, cfg["lda_beta"] * fac * Lw], 1).astype(np.float32)
        del Lw
    else:
        X = Sw
    del Sw, K_all
    print(f"representation {rep}: {X.shape}", flush=True)


    # 3. 重心と、各 patch の重心までの距離
    cl = np.load(RUN / f"assign_{rep}_{method}.npy").astype(np.int64)
    assert len(cl) == N
    K = int(cl.max()) + 1
    n = np.bincount(cl, minlength=K)
    cen = torch.zeros(K, X.shape[1], dtype=torch.float64)
    for s in range(0, N, 1_000_000):
        cen.index_add_(0, torch.from_numpy(cl[s:s + 1_000_000]), torch.from_numpy(X[s:s + 1_000_000]).double())
    cen = (cen / torch.from_numpy(np.maximum(n, 1)).double()[:, None]).float()
    dist = np.concatenate([((torch.from_numpy(X[s:s + 1_000_000]) - cen[torch.from_numpy(cl[s:s + 1_000_000])]) ** 2).sum(1).numpy() for s in range(0, N, 1_000_000)])
    cen = cen.numpy()
    np.save(OUT / "centroids.npy", cen)

    # 4. 葉の情報（系統樹の作り方によらない）
    ns = len(S)
    cs = np.bincount(cl * ns + slide_of, minlength=K * ns).reshape(K, ns)
    ctrl = S.is_ctrl.to_numpy()
    exps = S.exp.to_numpy()
    ue, e_idx = np.unique(exps, return_inverse=True)
    ce = np.zeros((K, len(ue)), dtype=np.int64)
    for j_ in range(len(ue)):
        ce[:, j_] = cs[:, e_idx == j_].sum(1)
    srt = -np.sort(-ce, 1)
    n80 = (np.cumsum(srt, 1) < 0.8 * n[:, None]).sum(1) + 1
    fat = S["grade__Degeneration, fatty"].to_numpy() > 0
    comp = S.COMPOUND_NAME.fillna("").to_numpy()
    is_ccl4 = pd.Series(comp).str.contains("tetrachloride", case=False).to_numpy()
    is_chol = pd.Series(comp).str.contains("cholesterol", case=False).to_numpy()
    pur = {}
    for nm_, mc in [("CCl4", is_ccl4), ("cholesterol", is_chol)]:
        nc, ncf = cs[:, mc].sum(1), cs[:, mc & fat].sum(1)
        pur[nm_] = (nc, np.where(nc > 0, ncf / np.maximum(nc, 1), np.nan))
    names_p = RUN / f"prototype_names_{rep}_{method}.csv"
    names = pd.read_csv(names_p).set_index("prototype") if names_p.exists() else pd.DataFrame(index=range(K), columns=["record_name", "bayer_name"])
    leaves = pd.DataFrame({"prototype": np.arange(K), "n_patches": n,
                           "record_name": names.record_name.reindex(range(K)).fillna("").to_numpy(), "bayer_name": names.bayer_name.reindex(range(K)).fillna("").to_numpy(),
                           "share_from_control_slides": cs[:, ctrl].sum(1) / np.maximum(n, 1),
                           "top_experiment": ue[ce.argmax(1)], "top_experiment_compound": [S.COMPOUND_NAME[(exps == ue[a])].iloc[0] for a in ce.argmax(1)],
                           "top_experiment_share": ce.max(1) / np.maximum(n, 1), "n_experiments_80pct": n80,
                           "n_CCl4": pur["CCl4"][0], "fatty_purity_CCl4": pur["CCl4"][1], "n_cholesterol": pur["cholesterol"][0], "fatty_purity_cholesterol": pur["cholesterol"][1]})
    leaves["fatty_pure"] = np.where((leaves.n_CCl4 >= MIN_N) & (leaves.fatty_purity_CCl4 >= PURITY), "CCl4", np.where((leaves.n_cholesterol >= MIN_N) & (leaves.fatty_purity_cholesterol >= PURITY), "cholesterol", ""))

    # 5. 系統樹の作り方（2026-11-02ユーザー: 隣のプロトタイプが微妙。a・b・c をそれぞれ試す）
    from scipy.spatial.distance import squareform
    trees = {}
    # ward: 重心の Ward 法（最初の版）
    trees["ward"] = ("重心の Ward 法（A128 の空間）", "Ward の距離", lambda ids: linkage(cen[ids].astype(np.float64), method="ward"))
    # a: 分割の履歴。split_record の葉は flat k=100 のクラスタを分けたものなので、同じ親の葉どうしを先につなぎ、その後で親どうしをつなぐ（どちらも重心の Ward）
    par_p = RUN / f"assign_{rep}_flat_k100.npy"
    if par_p.exists() and method.startswith("split"):
        c100 = np.load(par_p).astype(np.int64)
        tab = np.bincount(cl * 100 + c100, minlength=K * 100).reshape(K, 100)
        par = tab.argmax(1)
        print(f"split history: leaves nested in their k=100 parent {np.mean(tab.max(1) / np.maximum(tab.sum(1), 1)):.3f}", flush=True)
        leaves["parent_k100"] = par

        def hist_link(ids):
            ids = np.asarray(ids)
            g = par[ids]
            Zs, nid, rep_of_group, sizes = [], len(ids), {}, {}
            node_cen = {i: cen[ids[i]].astype(np.float64) for i in range(len(ids))}
            node_n = {i: float(n[ids[i]]) for i in range(len(ids))}
            node_cnt = {i: 1 for i in range(len(ids))}
            h_max = 0.0
            # 親ごとに Ward（scipy）で中をつなぎ、木を全体の番号に付け替える
            for gv in np.unique(g):
                mem = np.where(g == gv)[0]
                cur = list(mem)
                if len(mem) > 1:
                    zg = linkage(cen[ids[mem]].astype(np.float64), method="ward")
                    loc = {i: int(mem[i]) for i in range(len(mem))}
                    for t_, (a, b, h, c_) in enumerate(zg):
                        A, B = loc[int(a)], loc[int(b)]
                        h_max = max(h_max, h)
                        Zs.append([A, B, h, node_cnt[A] + node_cnt[B]])
                        tot = node_n[A] + node_n[B]
                        node_cen[nid] = (node_n[A] * node_cen[A] + node_n[B] * node_cen[B]) / tot
                        node_n[nid], node_cnt[nid] = tot, node_cnt[A] + node_cnt[B]
                        loc[len(mem) + t_] = nid
                        nid += 1
                    cur = [nid - 1]
                rep_of_group[gv] = cur[0]
            # 親どうし: 親の重心で Ward。高さは親の中の最大より上にずらす（図の高さが逆転しないように）
            roots_ = list(rep_of_group.values())
            if len(roots_) > 1:
                zr = linkage(np.stack([node_cen[r_] for r_ in roots_]), method="ward")
                loc = {i: roots_[i] for i in range(len(roots_))}
                for t_, (a, b, h, c_) in enumerate(zr):
                    A, B = loc[int(a)], loc[int(b)]
                    Zs.append([A, B, h_max + h, node_cnt[A] + node_cnt[B]])
                    node_cnt[nid] = node_cnt[A] + node_cnt[B]
                    loc[len(roots_) + t_] = nid
                    nid += 1
            return np.array(Zs, dtype=np.float64)
        trees["a_history"] = ("分割の履歴（同じ k=100 の親の葉を先につなぎ、親どうしは重心の Ward）", "親の中: Ward の距離 ／ 親どうし: 親の中の最大 + Ward の距離", hist_link)
    # b: patch の近傍のつながり。各プロトタイプから最大 300 patch を引き、プールの中で近傍 30 個がどのプロトタイプに入るか
    rng = np.random.default_rng(0)
    pool = np.concatenate([rng.choice(np.where(cl == j)[0], min(300, int(n[j])), replace=False) for j in range(K) if n[j] > 0])
    Xp = torch.from_numpy(X[pool]).cuda() if torch.cuda.is_available() else torch.from_numpy(X[pool])
    cp = cl[pool]
    P = np.zeros((K, K))
    for s in range(0, len(pool), 2048):
        d_ = torch.cdist(Xp[s:s + 2048], Xp)
        d_[torch.arange(min(2048, len(pool) - s)), torch.arange(s, min(s + 2048, len(pool)))] = float("inf")
        nb_ = d_.topk(30, largest=False).indices.cpu().numpy()
        np.add.at(P, (np.repeat(cp[s:s + 2048], 30), cp[nb_].ravel()), 1)
    del Xp
    P /= np.maximum(P.sum(1, keepdims=True), 1)
    Sym = (P + P.T) / 2
    Dn = 1 - Sym / np.sqrt(np.outer(np.maximum(np.diag(P), 1e-6), np.maximum(np.diag(P), 1e-6)))
    Dn = np.clip(Dn, 0, None)
    np.fill_diagonal(Dn, 0)
    np.save(OUT / "knn_connectivity.npy", P)
    print(f"kNN connectivity: pool {len(pool):,}, mean self-share {np.diag(P).mean():.3f}", flush=True)
    trees["b_knn"] = ("patch の近傍のつながり（各 patch の近傍 30 個がどのプロトタイプに入るか。平均連結）", "1 − 正規化したつながり",
                      lambda ids: linkage(squareform(Dn[np.ix_(ids, ids)], checks=False), method="average"))
    # c: 意味（構成）: 記録の 11 所見の倍率（log）と、Bayer のラベルの構成。標準化して余弦距離、平均連結
    findings = [c_[len("share__"):] for c_ in names.columns if c_.startswith("share__")]
    R = np.log(np.stack([names[f"share__{f}"].reindex(range(K)).fillna(0).to_numpy() for f in findings], 1) + 1e-3)
    labb = np.where(lab >= 0, lab, -1)
    m_ = labb >= 0
    Bc = np.bincount(cl[:nb][m_] * len(keep) + labb[m_], minlength=K * len(keep)).reshape(K, len(keep)).astype(float)
    Bc = Bc / np.maximum(Bc.sum(1, keepdims=True), 1)
    F = np.c_[(R - R.mean(0)) / (R.std(0) + 1e-9), (Bc - Bc.mean(0)) / (Bc.std(0) + 1e-9)]
    trees["c_meaning"] = ("意味（記録の所見の構成と Bayer のラベルの構成。余弦距離、平均連結）", "余弦距離",
                          lambda ids: linkage(F[ids], method="average", metric="cosine"))
    del X

    # 6. 図: 作り方ごとに F1（全体）と F2（部分木）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import openslide
    from PIL import Image, ImageDraw, ImageFont
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_JP).get_name()
    period = q.set_index("slide_id").SACRIFICE_PERIOD.to_dict()
    bayer_svs = bayer.set_index("slide_id").svs_path.to_dict()
    tgt_svs = tgt.set_index("slide_id").svs_path.to_dict()

    def svs_of(sid):
        for p_ in (bayer_svs.get(sid), tgt_svs.get(sid), str(ROOT / "data/TGGATEs/WSI/Liver" / str(period.get(sid, "")).replace(" ", "_") / f"{sid}.svs")):
            if isinstance(p_, str) and Path(p_).exists():
                return p_
        g = glob.glob(str(ROOT / "data/TGGATEs/WSI" / "**" / f"{sid}.svs"), recursive=True)
        return g[0] if g else ""
    svs_cache, s0_cache, tile_cache = {}, {}, {}

    def tiles_of(j):
        if j in tile_cache:
            return tile_cache[j]
        idx = np.where(cl == j)[0]
        idx = idx[np.argsort(dist[idx])]
        picks, seen = [], set()
        for p_ in idx[:5000]:
            sid = S.slide_id[slide_of[p_]]
            if slide_of[p_] in seen:
                continue
            if sid not in svs_cache:
                svs_cache[sid] = svs_of(sid)
            if not svs_cache[sid]:
                continue
            picks.append(int(p_)); seen.add(slide_of[p_])
            if len(picks) >= N_TILES:
                break
        imgs = []
        for p_ in picks:
            sid = S.slide_id[slide_of[p_]]
            if sid not in s0_cache:
                import h5py
                with h5py.File(h5_of(sid), "r") as h:
                    s0_cache[sid] = int(h["coords"].attrs["patch_size_level0"])
            w = openslide.OpenSlide(svs_cache[sid])
            imgs.append(w.read_region((int(XY[p_, 0]), int(XY[p_, 1])), 0, (s0_cache[sid], s0_cache[sid])).convert("RGB").resize((ROW, ROW)))
            w.close()
        tile_cache[j] = imgs
        return imgs

    fj, fjs, fjt = ImageFont.truetype(FONT_JP, 15), ImageFont.truetype(FONT_JP, 12), ImageFont.truetype(FONT_JP, 22)
    targets = list(leaves.prototype[leaves.fatty_pure != ""]) + extra
    all_ids = np.arange(K)
    for tname, (tdesc, ylab, linkf) in trees.items():
        TD = OUT / tname
        TD.mkdir(exist_ok=True)
        Z = linkf(all_ids)
        np.save(TD / "linkage.npy", Z)
        order = dendrogram(Z, no_plot=True)["leaves"]
        lv_all = leaves.assign(tree_order=pd.Series(np.arange(K), index=order).reindex(range(K)).to_numpy())
        lv_all.to_csv(TD / "leaves.csv", index=False)
        hs = np.quantile(Z[:, 2], np.linspace(0.5, 1.0, 26))
        pd.DataFrame({"cut_height": hs, "n_groups": [int((Z[:, 2] > h).sum()) + 1 for h in hs]}).to_csv(TD / "cut_heights.csv", index=False)
        # F1
        fig = plt.figure(figsize=(max(24, K * 0.04), 10))
        ax = fig.add_axes([0.04, 0.42, 0.94, 0.5])
        dendrogram(Z, ax=ax, no_labels=True, color_threshold=0, above_threshold_color="#555555", link_color_func=lambda k: "#555555")
        ax.set_ylabel(ylab)
        ax.set_title(f"プロトタイプの系統樹（{tdesc}）: {rep}、{method}（{K} プロトタイプ、全 3,737 枚・{N:,} patch）。下の帯は葉ごとの情報", fontsize=12)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        lv = leaves.set_index("prototype").loc[order]
        strips = [("記録による名前", np.array([matplotlib.colors.to_rgb(REC_COL.get(r_, "#DDDDDD")) for r_ in lv.record_name])[None]),
                  ("対照群 slide 由来の割合（濃いほど高い）", plt.cm.Greys(lv.share_from_control_slides.to_numpy())[None, :, :3]),
                  ("最多の試験の割合（濃いほど 1 試験に偏る）", plt.cm.Purples(lv.top_experiment_share.to_numpy())[None, :, :3]),
                  ("脂肪変性の純プロトタイプ（橙 CCl4、青 cholesterol 食）", np.array([matplotlib.colors.to_rgb({"CCl4": "#E69F00", "cholesterol": "#0072B2"}.get(f_, "#FFFFFF")) for f_ in lv.fatty_pure])[None])]
        for k_, (ttl, img) in enumerate(strips):
            a_ = fig.add_axes([0.04, 0.34 - 0.075 * k_, 0.94, 0.05])
            a_.imshow(img, aspect="auto", interpolation="nearest")
            a_.set_xticks([]); a_.set_yticks([])
            a_.set_ylabel(ttl, rotation=0, ha="right", va="center", fontsize=9)
        hx = [matplotlib.patches.Patch(color=v, label=k) for k, v in {"脂肪変性": "#E69F00", "空胞化": "#F0E442", "壊死・単細胞壊死": "#D55E00", "細胞浸潤・髄外造血": "#0072B2",
                                                                      "核分裂増加・肥大": "#009E73", "胆管増生": "#56B4E9", "好酸性変化・顆粒状好酸性変性": "#CC79A7", "名前なし": "#DDDDDD"}.items()]
        fig.legend(handles=hx, loc="lower center", ncol=8, frameon=False, fontsize=9, title="記録による名前の色")
        fig.savefig(TD / "F1_tree_all.png", dpi=110)
        plt.close(fig)
        # F2: 対象の葉を含む枝のうち、葉 MAX_SUB 個以下で最大のもの。他の部分木に含まれるものは除く
        members = {i: [i] for i in range(K)}
        parent = {}
        for t_, (a, b, _, _) in enumerate(Z):
            members[K + t_] = members[int(a)] + members[int(b)]
            parent[int(a)] = parent[int(b)] = K + t_
        roots = []
        for j in targets:
            node = j
            while node in parent and len(members[parent[node]]) <= MAX_SUB:
                node = parent[node]
            if node not in roots:
                roots.append(node)
        roots = [r_ for r_ in roots if not any(r_ != o and set(members[r_]) <= set(members[o]) for o in roots)]
        roots = [r_ for r_ in roots if len(members[r_]) > 1]
        for ri, root in enumerate(roots):
            lv_ids = members[root]
            Zs = linkf(np.asarray(lv_ids))
            ordr = [lv_ids[i] for i in dendrogram(Zs, no_plot=True)["leaves"]]
            nl = len(ordr)
            H = nl * ROW
            fig = plt.figure(figsize=(DEND_W / 100, H / 100), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            dendrogram(Zs, ax=ax, orientation="left", no_labels=True, color_threshold=0, link_color_func=lambda k: "#555555")
            ax.set_ylim(0, 10 * nl)
            ax.axis("off")
            fig.canvas.draw()
            dimg = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
            plt.close(fig)
            hdr = 100
            sheet = Image.new("RGB", (DEND_W + TXT_W + N_TILES * ROW, hdr + H), "white")
            sheet.paste(dimg.resize((DEND_W, H)), (0, hdr))
            dd = ImageDraw.Draw(sheet)
            dd.text((8, 6), f"部分木 {ri + 1}（{tdesc}）: {rep}、{method}。葉 {nl} 個", fill="black", font=fjt)
            dd.text((8, 38), "各行が 1 プロトタイプ。左は部分木、中は番号・patch 数・名前・構成、右は重心に近い patch 4 枚（別々の slide、20 倍、約 128 µm 四方）。", fill="#333333", font=fj)
            dd.text((8, 60), "橙の行 = CCl4 の脂肪変性の純プロトタイプ、青の行 = cholesterol 食の純プロトタイプ、緑の枠 = 指定して見たいプロトタイプ（" + ", ".join(f"p{e:04d}" for e in extra) + "）。", fill="#333333", font=fj)
            for i, j in enumerate(ordr):
                y0 = hdr + (nl - 1 - i) * ROW
                r = leaves.loc[j]
                bg = {"CCl4": "#FCEBD0", "cholesterol": "#D6E8F5"}.get(r.fatty_pure, "#FFFFFF" if i % 2 else "#F7F7F7")
                dd.rectangle([DEND_W, y0, DEND_W + TXT_W, y0 + ROW - 1], fill=bg, outline="#2E8B57" if j in extra else None, width=4 if j in extra else 0)
                par_s = f"   k=100 の親 {int(r.parent_k100)}" if "parent_k100" in leaves.columns else ""
                lines_ = [f"p{j:04d}  patch {int(r.n_patches):,}{par_s}",
                          f"記録: {r.record_name or 'なし'}   Bayer: {r.bayer_name or 'なし'}",
                          f"対照群 slide 由来 {r.share_from_control_slides:.0%}   最多試験 exp {int(r.top_experiment)} {str(r.top_experiment_compound)[:16]}（{r.top_experiment_share:.0%}）",
                          f"80% を占める試験数 {int(r.n_experiments_80pct)}   脂肪変性の純度 CCl4 {r.fatty_purity_CCl4:.2f}（{int(r.n_CCl4)}） chol {r.fatty_purity_cholesterol:.2f}（{int(r.n_cholesterol)}）".replace("nan", "-")]
                for k_, t_ in enumerate(lines_):
                    dd.text((DEND_W + 8, y0 + 6 + 29 * k_), t_, fill="black" if k_ == 0 else "#333333", font=fj if k_ == 0 else fjs)
                for k_, img in enumerate(tiles_of(j)):
                    sheet.paste(img, (DEND_W + TXT_W + k_ * ROW, y0))
            sheet.save(TD / f"F2_subtree_{ri + 1:02d}.jpg", quality=85)
        print(f"{tname}: F1 and {len(roots)} subtrees ({[len(members[r_]) for r_ in roots]} leaves)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
