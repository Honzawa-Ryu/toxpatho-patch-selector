"""0015の結果から、試験（exp_id）同士の近さを集計し、比較できる塊の下見をする。

試験間距離 = 同じ時点のslide対のJS距離の平均（試験a×試験b）。対角は同じ試験・同じ時点の別slide対の平均。
全run（k・seed）の平均で行列を作り、平均連結の階層クラスタリングで並べる。
各runで同じ高さで切ったときに同じ塊に入った割合（共塊率）も出し、塊の安定性を見る。
usage: .venv/bin/python scripts/preview_exp_blocks.py [variant]   既定: main
"""
import importlib.util
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/0015_20261012_ctrl_batch_structure"
SCAN_COLORS = {"SS1601": "#2a78d6", "SS1598": "#eb6834", "SS1224": "#1baf7a"}
CUT = 0.85   # 塊を切る高さ（JS距離）。下見用の目安で、確定基準ではない

spec = importlib.util.spec_from_file_location("exp0015", EXP / "experiment.py")
exp0015 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp0015)


def exp_distance(occ, exp, period, exps):
    d = exp0015.js_distance_matrix(occ)
    same_t = period[:, None] == period[None, :]
    np.fill_diagonal(same_t, False)
    out = np.zeros((len(exps), len(exps)))
    idx = [np.flatnonzero(exp == e) for e in exps]
    for i, a in enumerate(idx):
        for j, b in enumerate(idx[i:], start=i):
            m = same_t[np.ix_(a, b)]
            out[i, j] = out[j, i] = d[np.ix_(a, b)][m].mean() if m.any() else np.nan
    return out


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "main"
    run = ROOT / "outputs/0015_20261012_ctrl_batch_structure" / variant
    out = EXP / f"exp_blocks_preview_{variant}_{date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    slides = pd.read_csv(run / "slides.csv", dtype={"slide_id": str})
    exp, period = slides.exp_id.to_numpy(), slides.sacrifice_period.to_numpy()
    exps = np.array(sorted(slides.exp_id.unique()))
    occs = np.load(run / "occupancy.npz")
    mats = {key: exp_distance(occs[key], exp, period, exps) for key in occs.files}

    co = np.zeros((len(exps), len(exps)))
    for m in mats.values():
        off = m.copy()
        np.fill_diagonal(off, 0)
        lab = fcluster(linkage(squareform(off, checks=False), "average"), CUT, criterion="distance")
        co += lab[:, None] == lab[None, :]
    co /= len(mats)

    mean = np.mean(list(mats.values()), axis=0)
    within = np.diag(mean).copy()
    off = mean.copy()
    np.fill_diagonal(off, 0)
    Z = linkage(squareform(off, checks=False), "average")
    blocks = fcluster(Z, CUT, criterion="distance")

    comp = slides.groupby("exp_id").compound_name.first()
    scan = slides.groupby("exp_id").scanner.agg(lambda s: s.value_counts().to_dict())
    labels = [f"e{e} {comp[e][:22]}" for e in exps]
    table = pd.DataFrame({"exp_id": exps, "compound": comp[exps].values, "block": blocks,
                          "within_exp_dist": within, "scanner": [scan[e] for e in exps],
                          "n_slides": slides.exp_id.value_counts()[exps].values})
    nearest = []
    for i in range(len(exps)):
        o = off[i].copy()
        o[i] = np.inf
        j = o.argmin()
        nearest.append((comp[exps[j]], round(o[j], 3), round(co[i, j], 2)))
    table["nearest_exp"], table["nearest_dist"], table["nearest_coblock_rate"] = zip(*nearest)
    table.sort_values(["block", "nearest_dist"]).to_csv(out / "exp_blocks.csv", index=False)
    pd.DataFrame(mean, index=exps, columns=exps).to_csv(out / "exp_distance_mean.csv")

    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(2, 3, width_ratios=[3.2, 10, 10], height_ratios=[1, 10], wspace=0.02, hspace=0.02)
    ax_den = fig.add_subplot(gs[1, 0])
    dn = dendrogram(Z, orientation="left", labels=labels, ax=ax_den, color_threshold=CUT,
                    above_threshold_color="#9aa3ad", leaf_font_size=7)
    order = dn["leaves"][::-1]
    ax_den.axvline(CUT, color="#eb6834", lw=1, ls="--")
    ax_den.set_xlabel("JS distance (average linkage)")
    for s in ["top", "right", "left"]:
        ax_den.spines[s].set_visible(False)
    ax_den.yaxis.tick_right()
    ax_den.set_yticklabels([])
    for k, (name, mat, cmap, vmin, vmax) in enumerate([
            ("mean JS distance between experiments\n(diagonal: within experiment)", mean, "Blues_r", .3, 1),
            (f"co-block rate across {len(mats)} runs (cut at {CUT})", co, "Blues", 0, 1)]):
        ax = fig.add_subplot(gs[1, k + 1])
        im = ax.imshow(mat[np.ix_(order, order)], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks(range(len(order)), [labels[i] for i in order] if k == 0 else [], fontsize=7)
        ax.set_title(name, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=.04, pad=.01)
        ax_s = fig.add_subplot(gs[0, k + 1], sharex=ax)
        frac = np.array([scan[exps[i]].get("SS1598", 0) / sum(scan[exps[i]].values()) for i in order])
        ax_s.imshow(np.array([[matplotlib.colors.to_rgb(SCAN_COLORS["SS1598"]) if f > .5 else
                               matplotlib.colors.to_rgb(SCAN_COLORS["SS1601"]) for f in frac]]), aspect="auto")
        mixed = np.flatnonzero((frac > 0) & (frac < 1))
        ax_s.scatter(mixed, np.zeros_like(mixed), marker="*", color="white", s=30)
        ax_s.set_yticks([0], ["scanner"], fontsize=7)
        ax_s.set_xticks([])
    fig.suptitle(f"Preview of experiment blocks from no-finding Ctrl ({variant}: "
                 f"{'no stain normalization' if variant != 's219_macenko' else 'Macenko'}, {len(slides)} slides, {len(exps)} experiments)\n"
                 "scanner strip: blue = mostly SS1601, orange = mostly SS1598, star = both scanners", fontsize=13)
    fig.savefig(out / "exp_blocks_preview.png", bbox_inches="tight", dpi=150)
    print(out)
    print(table.sort_values(["block", "nearest_dist"]).to_string(index=False))


if __name__ == "__main__":
    main()
