"""0036 の小葉内の位置の推定が、patch の色味（白さ・淡さ）を読んでいないかを確かめる（2026-11-04ユーザー: 深刻な脂肪変性ではまだら模様が見え、平均の色味と相関するはず）。

patch ごとの色味はサムネイル（patch 1 個が約 8 画素）から測る: 平均の R・G・B、白い画素（3 色とも 220 以上）の割合。
  1. 脂肪変性の slide の中で、位置の推定（slide 内の順位）と色味の順位相関（slide ごとの値の分布）
  2. slide ごとに、位置の推定を色味（R・G・B・白の割合）で線形回帰し、その残りで「純プロトタイプ − 同じ slide のその他」の差を出す（0036 と同じ集計）
  3. Bayer の slide で、本当の位置（ポリゴンからの距離の比）と色味の順位相関（色が本来どれだけ位置の情報を持つか）
出力: outputs/0036_.../<版>/color_check/（color_corr_fatty.csv、paired_color_adjusted.csv、color_corr_bayer_truth.csv、F8_color_check.png）
usage: python scripts/lobular_color_check.py [版]   （コンテナ内、Slurm経由。CPU、10〜20 分）
"""
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "outputs/0036_20261103_local_prototype_context" / (sys.argv[1] if len(sys.argv) > 1 else "v5")
R35 = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin/v1"
OUT = RUN / "color_check"
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
DS = 32   # サムネイルの縮小率（patch 256 画素 → 8 画素）
SEED = 0


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def patch_colors(args):
    """1 枚の slide の patch ごとの色味（平均 R・G・B、白い画素の割合）。xy は level-0 の左上、s0 は一辺。"""
    import openslide
    svs, xy, s0 = args
    w = openslide.OpenSlide(svs)
    W, H = w.dimensions
    th = np.asarray(w.get_thumbnail((W // DS, H // DS)).convert("RGB")).astype(np.float32)
    w.close()
    sx, sy = th.shape[1] / W, th.shape[0] / H
    out = np.full((len(xy), 4), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(xy):
        x0, y0 = int(x * sx), int(y * sy)
        x1, y1 = max(x0 + 1, int((x + s0) * sx)), max(y0 + 1, int((y + s0) * sy))
        blk = th[y0:y1, x0:x1].reshape(-1, 3)
        if len(blk):
            out[i, :3] = blk.mean(0)
            out[i, 3] = float((blk.min(1) >= 220).mean())
    return out


def main():
    from scipy.stats import spearmanr
    OUT.mkdir(exist_ok=True)
    a = np.load(RUN / "arrays.npz")
    qf, row, xy = a["qf"].astype(np.int64), a["row"].astype(np.int64), a["xy"]
    zl = np.load(RUN / "lobular_rank.npy").astype(np.float64)
    sl = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str})
    starts = np.r_[0, np.cumsum(np.bincount(row, minlength=len(sl)))]
    pure = pd.read_csv(R35 / "fatty_figures" / "fatty_pure_prototypes.csv")
    pure = pure[(pure.rep == "A128_sphering_lda") & (pure.method == "split_record")]
    pset = {cn: set(pure.prototype[pure.compound == cn]) for cn in ("CCl4", "cholesterol")}
    ok_svs = np.array([Path(str(s)).exists() for s in sl.svs])
    targets = [i for i in range(len(sl)) if ok_svs[i] and starts[i + 1] > starts[i]]
    with ProcessPoolExecutor(max_workers=8) as ex:
        cols = list(ex.map(patch_colors, [(sl.svs[i], xy[starts[i]:starts[i + 1]], int(sl.size0[i])) for i in targets], chunksize=2))
    C = np.full((len(qf), 4), np.nan, dtype=np.float32)
    for i, c in zip(targets, cols):
        C[starts[i]:starts[i + 1]] = c
    np.save(OUT / "patch_colors.npy", C)
    print(f"colors for {len(targets)} slides", flush=True)
    names = ["R", "G", "B", "white"]

    # 1. 脂肪変性の slide の中で、位置の推定と色味の順位相関
    rows = []
    fat_sl = sl.index[(sl.set == "fatty")].to_numpy()
    for i in fat_sl:
        a_, b_ = starts[i], starts[i + 1]
        z_, c_ = zl[a_:b_], C[a_:b_]
        ok = np.isfinite(z_) & np.isfinite(c_).all(1)
        if ok.sum() < 200:
            continue
        r = {"slide_id": sl.slide_id[i], "cmpd": sl.cmpd[i], "fatty_grade": sl.fatty_grade[i], "is_ctrl": bool(sl.is_ctrl[i])}
        for k, nm in enumerate(names):
            r[f"rho_{nm}"] = spearmanr(z_[ok], c_[ok, k])[0]
        r["brightness_rho"] = spearmanr(z_[ok], c_[ok, :3].mean(1))[0]
        rows.append(r)
    cf = pd.DataFrame(rows)
    cf.to_csv(OUT / "color_corr_fatty.csv", index=False)
    print(cf.groupby(["cmpd", cf.fatty_grade > 0]).agg(n=("slide_id", "size"), brightness_rho=("brightness_rho", "median"), white_rho=("rho_white", "median")).round(3).to_string(), flush=True)

    # 2. 色味の効果を除いた残りで、純プロトタイプ − その他（slide ごと）
    cmpd_p = sl.cmpd.to_numpy()[row]
    fat_p = sl.fatty_grade.to_numpy()[row] > 0
    resid = np.full(len(qf), np.nan)
    for i in fat_sl:
        a_, b_ = starts[i], starts[i + 1]
        z_, c_ = zl[a_:b_], C[a_:b_]
        ok = np.isfinite(z_) & np.isfinite(c_).all(1)
        if ok.sum() < 200:
            continue
        Xd = np.c_[np.ones(ok.sum()), c_[ok]]
        beta, *_ = np.linalg.lstsq(Xd, z_[ok], rcond=None)
        rr = np.full(b_ - a_, np.nan)
        rr[ok] = z_[ok] - Xd @ beta
        resid[a_:b_] = rr
    prow = []
    rng = np.random.default_rng(SEED)
    for kind, zz in (("position (as in 0036 v5)", zl), ("position, colour regressed out", resid)):
        for cn, ps in pset.items():
            inpure = np.isin(qf, list(ps))
            base_ = (cmpd_p == cn) & fat_p & np.isfinite(zz)
            a_ = pd.Series(zz[base_ & inpure]).groupby(row[base_ & inpure]).agg(["mean", "size"])
            b_ = pd.Series(zz[base_ & ~inpure]).groupby(row[base_ & ~inpure]).mean()
            a_ = a_[a_["size"] >= 50]
            d_ = (a_["mean"] - b_.reindex(a_.index)).dropna()
            want = -1 if cn == "CCl4" else 1
            bs = [d_.iloc[rng.integers(0, len(d_), len(d_))].mean() for _ in range(2000)] if len(d_) > 1 else [np.nan]
            prow.append({"kind": kind, "compound": cn, "n_slides": int(len(d_)), "mean_diff": float(d_.mean()), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)),
                         "share_slides_expected_direction": float((np.sign(d_) == want).mean())})
    pt = pd.DataFrame(prow)
    pt.to_csv(OUT / "paired_color_adjusted.csv", index=False)
    print(pt.round(3).to_string(index=False), flush=True)

    # 3. Bayer の slide で、本当の位置（ポリゴン）と色味の順位相関
    ex21 = load_module(ROOT / "experiments/0021_20261025_patch_prototypes/experiment.py", "exp0021")
    bay = sl[sl.set == "bayer"].reset_index()
    nb = int(starts[len(bay)])
    pos = ex21.lobular_position(bay.rename(columns={"index": "i"}), row[:nb], xy[:nb], dict(zip(bay.slide_id, bay.size0)), ROOT, 1500)
    trow = []
    for j, r in bay.iterrows():
        a_, b_ = starts[j], starts[j + 1]
        ok = np.isfinite(pos[a_:b_]) & np.isfinite(C[a_:b_]).all(1)
        if ok.sum() < 100:
            continue
        trow.append({"slide_id": r.slide_id, "n": int(ok.sum()), "truth_brightness_rho": spearmanr(pos[a_:b_][ok], C[a_:b_][ok, :3].mean(1))[0],
                     "truth_white_rho": spearmanr(pos[a_:b_][ok], C[a_:b_][ok, 3])[0], "pred_brightness_rho": spearmanr(zl[a_:b_][ok], C[a_:b_][ok, :3].mean(1))[0] if np.isfinite(zl[a_:b_][ok]).all() else np.nan})
    tb = pd.DataFrame(trow)
    tb.to_csv(OUT / "color_corr_bayer_truth.csv", index=False)
    print(f"Bayer: {len(tb)} slides; median rho(true position, brightness) {tb.truth_brightness_rho.median():.3f}, rho(true, white) {tb.truth_white_rho.median():.3f}, rho(predicted, brightness) {tb.pred_brightness_rho.median():.3f}", flush=True)

    # 図
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    ax = axes[0]
    groups = [("Bayer の slide: 本当の位置", tb.truth_brightness_rho, "#999999", "o"), ("Bayer の slide: 推定の位置", tb.pred_brightness_rho, "#56B4E9", "o")]
    for cn, col in (("CCl4", "#E69F00"), ("cholesterol", "#0072B2")):
        for g_, lab_ in ((False, "記録なし（対照・低用量）"), (True, "脂肪変性あり")):
            v = cf[(cf.cmpd == cn) & ((cf.fatty_grade > 0) == g_)].brightness_rho
            groups.append((f"{cn}: {lab_}: 推定の位置", v, col, "D" if g_ else "s"))
    for k, (lab_, v, col, mk) in enumerate(groups):
        v = pd.Series(v).dropna()
        jit = np.random.default_rng(k).uniform(-0.15, 0.15, len(v))
        ax.scatter(v, k + jit, s=10, color=col, marker=mk, alpha=0.6)
        if len(v):
            ax.plot([v.median()] * 2, [k - 0.3, k + 0.3], color="black", lw=1.5)
    ax.set_yticks(range(len(groups)), [f"{g[0]}（{pd.Series(g[1]).notna().sum()} 枚）" for g in groups])
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.invert_yaxis()
    ax.set(xlabel="slide 内の順位相関: 小葉内の位置 × 明るさ（RGB の平均）。正 = 門脈域側ほど明るい", title="位置の推定は色味（明るさ）と相関するか\n点 = slide、黒線 = 中央値")
    ax = axes[1]
    yy = 0
    yl, yt = [], []
    for cn in ("CCl4", "cholesterol"):
        for kind, col, mk in (("position (as in 0036 v5)", "#D55E00", "D"), ("position, colour regressed out", "#009E73", "o")):
            r = pt[(pt.compound == cn) & (pt.kind == kind)].iloc[0]
            ax.errorbar(r.mean_diff, yy, xerr=[[r.mean_diff - r.lo], [r.hi - r.mean_diff]], fmt=mk, color=col, capsize=3, ms=6,
                        label={"position (as in 0036 v5)": "0036 v5 の位置", "position, colour regressed out": "色味の効果を除いた残り"}[kind] if cn == "CCl4" else None)
            yl.append(yy); yt.append(f"{cn}（記録どおりの向き {r.share_slides_expected_direction:.0%}、{r.n_slides} 枚）")
            yy += 1
        yy += 0.5
    ax.set_yticks(yl, yt)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set(xlabel="slide ごとの差（純プロトタイプ − 同じ slide のその他）。負 = 中心静脈側、正 = 門脈域側", title="色味の効果を除いても、脂肪変性の部位の差は残るか\n点 = slide の平均、線 = slide を単位にした 95% ブートストラップ区間")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.savefig(OUT / "F8_color_check.png", dpi=180)
    print("done", flush=True)


if __name__ == "__main__":
    main()
