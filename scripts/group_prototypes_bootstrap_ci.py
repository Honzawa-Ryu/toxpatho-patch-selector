"""0037 の「試験を 1 つ伏せて当てる AUC」に区間を付ける（2026-11-11 ユーザー: 差やばらつきが誤差の範囲かを確かめる）。
比べるもの: 0037（まとまりで作り直したプロトタイプ）と 0035 の全体のプロトタイプ（scripts/group_prototypes_global_baseline.py の slide_counts.npz）。

ブートストラップ: まとまりごとに、slide を「試験 × 対照群か投与群か」の層の中で復元抽出し、プロトタイプの選び方から AUC までをやり直す
（0037 experiment.py の loso_auc をそのまま使う）。両方の方法・全ての k に同じ抽出を使うので、差は対になる。
区間が表すのは slide の抽出による揺れだけ。試験そのものの抽出（どの試験がまとまりにあるか）は入っていない（試験が 2〜8 しかないため）。
出力: outputs/0037_20261111_group_prototypes_with_treated/default/bootstrap/
  auc_ci.csv（方法 × 組 × 表現 × k の、伏せた試験の平均 AUC と 95% 区間）、diff_ci.csv（0037 − 0035）、k_diff_ci.csv（隣り合う k の差）、
  F3_loso_auc_ci.png、F4_diff_0037_vs_0035.png
usage: python scripts/group_prototypes_bootstrap_ci.py   （コンテナ内、Slurm経由。CPU 12 コア、30 分程度）
"""
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
R37 = ROOT / "outputs/0037_20261111_group_prototypes_with_treated/default"
OUT = R37 / "bootstrap"
KS = [100, 200, 500, 1000]
REPS = ["B128_sphering", "A128_sphering_lda"]
METHODS = ["0037", "0035"]
B, SEED = 500, 0
_W = {}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def one_replicate(b):
    """b = -1 は抽出なし（元の値）。返り値: (method, group, finding, rep, k) → 伏せた試験の平均 AUC（対 記録のない投与群）"""
    W = _W
    out = {}
    for g, d in W["groups"].items():
        if b < 0:
            idx = np.arange(len(d["exp"]))
        else:
            rng = np.random.default_rng([SEED, b, g])
            idx = np.concatenate([rng.choice(s_, len(s_)) for s_ in d["strata"]])
        for (meth, rep, k), cs in d["cs"].items():
            rows = W["ex37"].loso_auc(cs[:, idx], d["pos"][idx], d["exp"][idx], d["ctrl"][idx], d["comp"][idx], W["findings"], W["lo"], {})
            r = pd.DataFrame(rows)
            if len(r):
                for f, v in r.groupby("finding").auc_own_neg.mean().items():
                    out[(meth, g, f, rep, k)] = v
    return out


def main():
    import h5py
    OUT.mkdir(exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "experiments/0037_20261111_group_prototypes_with_treated/config.yml"))
    load_module(ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin/experiment.py", "exp0035")
    ex37 = load_module(ROOT / "experiments/0037_20261111_group_prototypes_with_treated/experiment.py", "exp0037")
    ex35 = sys.modules["exp0035"]
    S = pd.read_csv(R37 / "slides.csv", dtype={"slide_id": str})
    findings = cfg["findings"]

    def n_patches(sid):   # 0030 の _load_sub と同じ間引き: 2,048 を超える slide は 2,048
        for d_ in cfg["features_dirs"]:
            p = ROOT / d_ / f"{sid}.h5"
            if p.exists():
                with h5py.File(p, "r") as h:
                    return min(len(h["coords"]), cfg["patches_per_slide"])
        raise FileNotFoundError(sid)
    npt = np.array([n_patches(s_) for s_ in S.slide_id])
    base = np.load(R37 / "baseline_0035/slide_counts.npz")
    groups = {}
    for g in cfg["groups"]:
        sg = np.where(S.grp.to_numpy() == g)[0]
        so = np.repeat(np.arange(len(sg)), npt[sg])
        cs = {}
        for rep in REPS:
            for k in KS:
                cl = np.load(R37 / f"assign_g{g}_{rep}_k{k}.npy").astype(np.int64)
                assert len(cl) == len(so), "patch counts per slide do not match the 0037 assignment"
                cs[("0037", rep, k)] = ex35.slide_counts(cl, k, so, len(sg))
                cs[("0035", rep, k)] = base[f"g{g}_{rep}_k{k}"]
        e, c = S.exp.to_numpy()[sg], S.is_ctrl.to_numpy()[sg]
        strata = [np.where((e == x) & (c == y))[0] for x in np.unique(e) for y in (True, False) if ((e == x) & (c == y)).any()]
        groups[g] = {"cs": cs, "pos": S[[f"grade__{f}" for f in findings]].to_numpy()[sg] > 0, "exp": e, "ctrl": c,
                     "comp": S.COMPOUND_NAME.to_numpy()[sg], "strata": strata}
    _W.update(groups=groups, ex37=ex37, findings=findings, lo=cfg["loso"])
    point = one_replicate(-1)
    # 確認: 抽出なしの値が 0037 の loso.csv / baseline の loso.csv の平均と一致する
    chk = pd.read_csv(R37 / "loso.csv").groupby(["group", "finding", "rep", "k"]).auc_own_neg.mean()
    dev = max(abs(point[("0037", g, f, r, k)] - v) for (g, f, r, k), v in chk.items() if ("0037", g, f, r, k) in point and np.isfinite(v))
    print(f"max |point - 0037 loso.csv| = {dev:.2e}", flush=True)
    with ProcessPoolExecutor(max_workers=12) as ex:
        boots = list(ex.map(one_replicate, range(B), chunksize=8))
    keys = sorted(point)
    M = np.array([[bt.get(k_, np.nan) for k_ in keys] for bt in boots])          # B × keys
    pv = np.array([point[k_] for k_ in keys])
    lo_, hi_ = np.nanpercentile(M, 2.5, 0), np.nanpercentile(M, 97.5, 0)
    auc = pd.DataFrame([dict(zip(["method", "group", "finding", "rep", "k"], k_)) for k_ in keys]).assign(auc=pv, lo=lo_, hi=hi_, n_valid=np.isfinite(M).sum(0))
    auc.to_csv(OUT / "auc_ci.csv", index=False)
    col = {k_: i for i, k_ in enumerate(keys)}

    def diff_ci(pairs):
        """pairs: [(key_a, key_b)] → 平均（a − b）の点推定と区間"""
        ia, ib = [col[a] for a, _ in pairs], [col[b_] for _, b_ in pairs]
        d_b = np.nanmean(M[:, ia] - M[:, ib], 1)
        return float(np.nanmean(pv[ia] - pv[ib])), float(np.nanpercentile(d_b, 2.5)), float(np.nanpercentile(d_b, 97.5))
    combos = sorted({k_[1:4] for k_ in keys if k_[0] == "0037"})
    rows = []
    for g, f, r in combos:
        for k in KS + ["all_k"]:
            ks = KS if k == "all_k" else [k]
            pr = [(("0037", g, f, r, kk), ("0035", g, f, r, kk)) for kk in ks if ("0037", g, f, r, kk) in col and ("0035", g, f, r, kk) in col]
            if pr:
                rows.append({"group": g, "finding": f, "rep": r, "k": k, **dict(zip(["diff", "lo", "hi"], diff_ci(pr)))})
    allp = [(("0037",) + k_[1:], k_) for k_ in keys if k_[0] == "0035" and ("0037",) + k_[1:] in col]
    rows.append({"group": "all", "finding": "all", "rep": "all", "k": "all_k", **dict(zip(["diff", "lo", "hi"], diff_ci(allp)))})
    for r in REPS:
        rows.append({"group": "all", "finding": "all", "rep": r, "k": "all_k", **dict(zip(["diff", "lo", "hi"], diff_ci([p for p in allp if p[0][3] == r])))})
    dif = pd.DataFrame(rows)
    dif.to_csv(OUT / "diff_ci.csv", index=False)
    kr = []
    for meth in METHODS:
        for g, f, r in combos:
            for k1, k2 in zip(KS[:-1], KS[1:]):
                a, b_ = (meth, g, f, r, k2), (meth, g, f, r, k1)
                if a in col and b_ in col:
                    kr.append({"method": meth, "group": g, "finding": f, "rep": r, "k_from": k1, "k_to": k2, **dict(zip(["diff", "lo", "hi"], diff_ci([(a, b_)])))})
    kd = pd.DataFrame(kr)
    kd["excludes_0"] = (kd.lo > 0) | (kd.hi < 0)
    kd.to_csv(OUT / "k_diff_ci.csv", index=False)
    pd.set_option("display.width", 220)
    print(dif[dif.k == "all_k"].round(3).to_string(index=False), flush=True)
    print("adjacent-k differences whose 95% interval excludes 0:", kd.groupby("method").excludes_0.agg(["sum", "size"]).to_string(), flush=True)
    print("median interval width (AUC):", auc.groupby("method").apply(lambda d: float(np.nanmedian(d.hi - d.lo))).round(3).to_dict(), flush=True)

    # 図
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    fig_mod = load_module(ROOT / "scripts/group_prototypes_with_treated_figure.py", "fig37")
    font_manager.fontManager.addfont(fig_mod.FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=fig_mod.FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    STY = {r: (col_, mk) for r, _, col_, mk in fig_mod.REPS}
    LAB = {r: lab for r, lab, _, _ in fig_mod.REPS}
    pairs = auc[auc.method == "0037"].groupby(["group", "finding"]).size().reset_index()[["group", "finding"]]
    nexp = pd.read_csv(R37 / "loso.csv").groupby(["group", "finding"]).exp.nunique()
    pairs = pairs.assign(n=[nexp[(g, f)] for g, f in pairs.itertuples(index=False)]).sort_values(["group", "n"], ascending=[True, False])
    ncol = 5
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow + 0.9), sharex=True, sharey=True, layout="constrained")
    for ax, p in zip(axes.ravel(), pairs.itertuples()):
        for j, (meth, r) in enumerate([(m_, r_) for r_ in REPS for m_ in METHODS]):
            d = auc[(auc.method == meth) & (auc.group == p.group) & (auc.finding == p.finding) & (auc.rep == r)].set_index("k").reindex(KS)
            c_, mk = STY[r]
            x = np.log2(KS) + (j - 1.5) * 0.09
            ax.errorbar(x, d.auc, yerr=[d.auc - d.lo, d.hi - d.auc], color=c_, marker=mk, ms=4, lw=1.2 if meth == "0037" else 0.9,
                        ls="-" if meth == "0037" else "--", mfc=c_ if meth == "0037" else "white", mec=c_, capsize=1.5, elinewidth=0.8)
        ax.axhline(0.5, color="black", lw=0.6, ls=":")
        ax.set_title(f"まとまり {p.group}：{fig_mod.JP.get(p.finding, p.finding)}（{p.n} 試験）", fontsize=9)
        ax.set_xticks(np.log2(KS), [str(k) for k in KS])
        ax.set_ylim(0.1, 1.05)
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC（対 記録のない投与群）")
    for ax in axes[-1]:
        ax.set_xlabel("k")
    h = [Line2D([], [], color=STY[r][0], marker=STY[r][1], ls="-" if m_ == "0037" else "--", mfc=STY[r][0] if m_ == "0037" else "white",
                label=f"{LAB[r].split('（')[0]}：{'0037（まとまりで作り直す）' if m_ == '0037' else '0035 の全体のプロトタイプ'}") for r in REPS for m_ in METHODS]
    fig.legend(handles=h, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(f"試験を 1 つ伏せた AUC（伏せた試験の平均）と 95% 区間（slide を試験 × 対照群／投与群の中で復元抽出、{B} 回。選び方からやり直す）\n"
                 "区間は slide の抽出による揺れだけ。どの試験がまとまりにあるかの揺れは入っていない", fontsize=10)
    fig.savefig(OUT / "F3_loso_auc_ci.png", dpi=170)
    plt.close(fig)

    d = dif[dif.k == "all_k"].copy()
    d["label"] = [("全体（13 組 × 表現 × k）" if r == "all" else f"全体：{r.split('_')[0]}") if g == "all" else f"まとまり {g}：{fig_mod.JP.get(f, f)}：{r.split('_')[0]}"
                  for g, f, r in zip(d.group, d.finding, d.rep)]
    d = pd.concat([d[d.group != "all"].sort_values(["group", "finding", "rep"]), d[d.group == "all"]])
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(d) + 1.6), layout="constrained")
    y = np.arange(len(d))
    for yy, r in zip(y, d.itertuples()):
        c_ = "#333333" if r.rep == "all" else STY[r.rep][0]
        ax.errorbar(r.diff, yy, xerr=[[r.diff - r.lo], [r.hi - r.diff]], fmt="D" if r.group == "all" else STY.get(r.rep, ("", "o"))[1], color=c_, ms=5, capsize=2)
    ax.axvline(0, color="black", lw=0.7)
    ax.axhline(len(d) - 3.5, color="#BBBBBB", lw=0.8)
    ax.set_yticks(y, d.label, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AUC の差（0037 − 0035、k = 100〜1000 の平均）。右 = まとまりで作り直したほうが高い")
    ax.set_title("まとまりで作り直したプロトタイプ（0037）と 0035 の全体のプロトタイプの差と 95% 区間", fontsize=10)
    fig.savefig(OUT / "F4_diff_0037_vs_0035.png", dpi=170)
    print("done", flush=True)


if __name__ == "__main__":
    main()
