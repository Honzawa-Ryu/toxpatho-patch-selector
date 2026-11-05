"""scripts/control_similarity_groups.py の図（F2 ヒートマップ、F3 平面配置）を、保存した表から描き直す（2026-11-05: 帯の色が塊とスキャナで重なっていた、
単回/反復は 143 試験すべて反復で情報がない、撮影月を足す）。計算はしない。
入力: outputs/0035_.../control_similarity/（pairwise_auc_B128.csv、experiments.csv、summary.json）
usage: python scripts/control_similarity_figures.py   （コンテナ内、Slurm経由。数十秒）
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS

ROOT = Path(__file__).resolve().parents[1]
D_ = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin/control_similarity"
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
BLOCK_COL = {1: "#E69F00", 2: "#56B4E9", 3: "#009E73", 4: "#F0E442", 5: "#CC79A7", 6: "#D55E00"}   # Okabe-Ito（塊）
SCAN_COL = {"SS1601": "#222222", "SS1598": "#BBBBBB", "SS1224": "#FFFFFF"}   # スキャナは灰色の濃淡（塊の色と重ねない）


def main():
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    M = pd.read_csv(D_ / "pairwise_auc_B128.csv", index_col=0)
    M.columns = M.columns.astype(int)
    E = pd.read_csv(D_ / "experiments.csv", index_col=0).reindex(M.index)
    summ = json.loads((D_ / "summary.json").read_text())
    thr = summ["null95_auc_B128"]
    D = np.clip((M.to_numpy() - 0.5) / 0.5, 0, 1)
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="average")
    d_thr = max((thr - 0.5) / 0.5, 0)
    order = dendrogram(Z, no_plot=True)["leaves"]
    months = sorted(E.scan_month.dropna().unique())
    mcol = {m: matplotlib.colormaps["Purples"](0.25 + 0.75 * i / max(len(months) - 1, 1)) for i, m in enumerate(months)}

    fig = plt.figure(figsize=(13, 13.5))
    axd = fig.add_axes([0.18, 0.86, 0.7, 0.1])
    dendrogram(Z, ax=axd, no_labels=True, color_threshold=d_thr, above_threshold_color="#999999")
    axd.axhline(d_thr, color="#D55E00", lw=1, ls="--")
    axd.set_ylabel("見分けやすさ d", fontsize=8)
    axd.set_xticks([])
    axd.set_title(f"試験の組ごとの、対照群の見分けやすさ（B128、{len(M)} 試験、すべて反復投与）\n赤破線 = 偶然の 95% 点（AUC {thr:.2f}）で切る高さ。この下でつながる試験は対照群が見分けられない", fontsize=10)
    axh = fig.add_axes([0.18, 0.15, 0.7, 0.69])
    im = axh.imshow(M.to_numpy()[np.ix_(order, order)], vmin=0.5, vmax=1.0, cmap="Blues", interpolation="nearest")
    axh.set_xticks([])
    axh.set_yticks(range(len(M)))
    axh.set_yticklabels([f"{str(E.compound.iloc[i])[:16]}（{M.index[i]}）" for i in order], fontsize=4)
    cax = fig.add_axes([0.9, 0.5, 0.015, 0.3])
    fig.colorbar(im, cax=cax, label="AUC（0.5 = 見分けられない、1 = 完全に見分けられる）")
    strips = [("0015 の塊", [BLOCK_COL[int(b)] for b in E.block.iloc[order]]),
              ("スキャナ", [SCAN_COL.get(s, "#FFFFFF") for s in E.scanner.iloc[order]]),
              ("撮影月", [mcol.get(m, (1, 1, 1, 1)) for m in E.scan_month.iloc[order]])]
    for k, (nm, cols) in enumerate(strips):
        ax_ = fig.add_axes([0.18, 0.13 - 0.022 * (k + 1), 0.7, 0.017])
        ax_.imshow(np.array([[matplotlib.colors.to_rgb(c) for c in cols]]), aspect="auto", interpolation="nearest")
        ax_.set_xticks([]); ax_.set_yticks([])
        ax_.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=8)
    handles = ([Patch(color=c, label=f"塊 {b}") for b, c in BLOCK_COL.items()] + [Patch(facecolor=SCAN_COL["SS1601"], label="SS1601"), Patch(facecolor=SCAN_COL["SS1598"], label="SS1598")] +
               [Patch(color=mcol[m], label=m) for m in months])
    fig.legend(handles=handles, loc="lower center", ncol=13, frameon=False, fontsize=7)
    fig.savefig(D_ / "F2_pairwise_heatmap_B128.png", dpi=170)
    plt.close(fig)

    Y = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto").fit_transform(D)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), layout="constrained")
    for ax, (title, key, cmap_) in zip(axes, [("色 = 0015 の塊", "block", BLOCK_COL), ("色 = 撮影月", "scan_month", mcol)]):
        for v in (sorted(E[key].dropna().unique())):
            for sc_, mk in (("SS1601", "o"), ("SS1598", "^"), ("SS1224", "x")):
                m_ = (E[key] == v).to_numpy() & (E.scanner == sc_).to_numpy()
                if m_.any():
                    ax.scatter(Y[m_, 0], Y[m_, 1], color=cmap_[int(v)] if key == "block" else cmap_[v], marker=mk, s=42, alpha=0.9, edgecolors="#333333", linewidths=0.3)
        hh = [Patch(color=c, label=f"塊 {b}") for b, c in BLOCK_COL.items()] if key == "block" else [Patch(color=mcol[m], label=m) for m in months]
        ax.legend(handles=hh + [Line2D([], [], marker="o", ls="", color="#555555", label="SS1601"), Line2D([], [], marker="^", ls="", color="#555555", label="SS1598")], frameon=False, fontsize=8, loc="best")
        ax.set(xticks=[], yticks=[], title=f"試験の配置（対照群の見分けやすさを距離にした多次元尺度法、B128）\n近い = 対照群が見分けにくい。{title}、印 = スキャナ")
    fig.savefig(D_ / "F3_mds_B128.png", dpi=180)
    print("done")


if __name__ == "__main__":
    main()
