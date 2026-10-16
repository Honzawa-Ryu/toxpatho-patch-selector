"""0015の結果から、試験（exp_id）同士の近さを集計し、比較できる塊の下見をする。

試験間距離 = 同じ時点のslide対のJS距離の平均（試験a×試験b）。対角は同じ試験・同じ時点の別slide対の平均。
全run（k・seed）の平均で行列を作り、平均連結の階層クラスタリングで並べる。
各runで同じ高さで切ったときに同じ塊に入った割合（共塊率）も出し、塊の安定性を見る。
usage: .venv/bin/python scripts/preview_exp_blocks.py [variant] [cut]   既定: main 0.85
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


def slide_heatmap(slides, occ, exps, order, blocks, out, variant, cut):
    """0015の図2と同じslide × クラスタ占有率。行は樹形図の順（塊→試験→時点）、列は最も多く占める塊の順。"""
    days = ["4 day", "8 day", "15 day", "29 day"]
    day_colors = ["#a8cbf2", "#5b9be6", "#2a78d6", "#1f4f8f"]
    block_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    rank = {exps[i]: r for r, i in enumerate(order)}
    block_of = dict(zip(exps, blocks))
    s = slides.assign(_e=slides.exp_id.map(rank), _d=slides.sacrifice_period.map({d: i for i, d in enumerate(days)}))
    rows = s.sort_values(["_e", "_d", "individual_id"]).index.to_numpy()
    row_block = slides.exp_id.map(block_of).to_numpy()[rows]
    ub = list(dict.fromkeys(row_block))                       # 行に出てくる順の塊
    per_block = np.stack([occ[rows][row_block == b].mean(0) for b in ub])
    top = per_block.argmax(0)
    cols = np.lexsort((-per_block.max(0), top))
    m = np.sqrt(occ[np.ix_(rows, cols)])
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(1, 4, width_ratios=[.3, .25, .25, 12], wspace=.02)
    axb, axd, axs, ax = [fig.add_subplot(gs[i]) for i in range(4)]
    im = ax.imshow(m, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=np.quantile(m, .995))
    cmap_b = {b: block_colors[i % len(block_colors)] for i, b in enumerate(ub)}
    axb.imshow(np.array([matplotlib.colors.to_rgb(cmap_b[b]) for b in row_block])[:, None], aspect="auto")
    day = s.loc[rows, "_d"].to_numpy()
    axd.imshow(np.array([matplotlib.colors.to_rgb(day_colors[int(d)]) for d in day])[:, None], aspect="auto")
    axs.imshow(np.array([matplotlib.colors.to_rgb(SCAN_COLORS.get(x, "#999999")) for x in slides.scanner.to_numpy()[rows]])[:, None], aspect="auto")
    bnd = np.flatnonzero(np.r_[True, row_block[1:] != row_block[:-1], True])
    for y in bnd[1:-1]:
        ax.axhline(y - .5, color="black", lw=.8)
    cb = np.flatnonzero(np.r_[True, top[cols][1:] != top[cols][:-1], True])
    for x in cb[1:-1]:
        ax.axvline(x - .5, color="black", lw=.8)
    axb.set_yticks((bnd[:-1] + bnd[1:]) / 2 - .5, [f"block {b}\n({(np.array(list(block_of.values())) == b).sum()} exp)" for b in row_block[bnd[:-1]]], fontsize=8)
    for a, t in [(axb, "block"), (axd, "time"), (axs, "scan")]:
        a.set_xticks([0], [t], fontsize=7)
        for sp in a.spines.values():
            sp.set_visible(False)
    axd.set_yticks([])
    axs.set_yticks([])
    ax.set_yticks([])
    ax.set_xlabel("100 clusters (k=100, run k100_s0_i0), grouped by the block with the highest mean occupancy")
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=.03, pad=.01).set_label("sqrt(occupancy)")
    fig.suptitle(f"Slide x cluster occupancy, {len(rows)} no-finding Ctrl slides / {len(exps)} experiments ({variant}, blocks cut at {cut:g})\n"
                 "rows: blocks -> experiments (dendrogram order) -> time point; black lines: block boundaries", fontsize=12, y=.93)
    fig.savefig(out / "slide_occupancy_by_block.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "main"
    cut = float(sys.argv[2]) if len(sys.argv) > 2 else CUT
    run = ROOT / "outputs/0015_20261012_ctrl_batch_structure" / variant
    out = EXP / (f"exp_blocks_preview_{variant}_{date.today():%Y%m%d}" + ("" if cut == CUT else f"_cut{cut:g}"))
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
        lab = fcluster(linkage(squareform(off, checks=False), "average"), cut, criterion="distance")
        co += lab[:, None] == lab[None, :]
    co /= len(mats)

    mean = np.mean(list(mats.values()), axis=0)
    within = np.diag(mean).copy()
    off = mean.copy()
    np.fill_diagonal(off, 0)
    Z = linkage(squareform(off, checks=False), "average")
    blocks = fcluster(Z, cut, criterion="distance")

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
    dn = dendrogram(Z, orientation="left", labels=labels, ax=ax_den, color_threshold=cut,
                    above_threshold_color="#9aa3ad", leaf_font_size=7)
    order = dn["leaves"][::-1]
    ax_den.axvline(cut, color="#eb6834", lw=1, ls="--")
    ax_den.set_xlabel("JS distance (average linkage)")
    for s in ["top", "right", "left"]:
        ax_den.spines[s].set_visible(False)
    ax_den.yaxis.tick_right()
    ax_den.set_yticklabels([])
    for k, (name, mat, cmap, vmin, vmax) in enumerate([
            ("mean JS distance between experiments\n(diagonal: within experiment)", mean, "Blues_r", .3, 1),
            (f"co-block rate across {len(mats)} runs (cut at {cut})", co, "Blues", 0, 1)]):
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
                 f"{'Macenko stain normalization' if 'macenko' in variant else 'no stain normalization'}, {len(slides)} slides, {len(exps)} experiments)\n"
                 "scanner strip: blue = mostly SS1601, orange = mostly SS1598, star = both scanners", fontsize=13)
    fig.savefig(out / "exp_blocks_preview.png", bbox_inches="tight", dpi=150)
    slide_heatmap(slides, occs["k100_s0_i0"], exps, order, blocks, out, variant, cut)
    print(out)
    print(table.sort_values(["block", "nearest_dist"]).to_string(index=False))


if __name__ == "__main__":
    main()
