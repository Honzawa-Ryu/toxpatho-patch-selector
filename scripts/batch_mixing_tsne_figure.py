"""TG-GATEs内の試験・スキャナの混ざり方を、0029の「外部データとの混ざり」の図と同じ形（表現ごとに1枚の平面プロット）で描く
（2026-11-02ユーザー依頼）。入力は scripts/tggates_unseen_batch_mixing.py が保存した reps_controls.npz（対照群 857枚・各1,024 patch の raw / D / D13 / C）。

図1 F3_tsne_scanner.png     : 表現ごとに、全43試験から各300 patch を t-SNE。色＝スキャナ（SS1601 / SS1598）
図2 F4_tsne_experiments.png : 表現ごとに、スキャナごとに4試験ずつ選んだ8試験から各500 patch を t-SNE。色＝試験、印＝スキャナ（○ SS1601、× SS1598）
各パネルの題に、summary_controls.csv の近傍の混ざり（別試験・別スキャナの割合）を添える。
usage: python scripts/batch_mixing_tsne_figure.py [版]  （コンテナ内、Slurm経由。CPUで数分）
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
_RUN = sys.argv[1] if len(sys.argv) > 1 else "v4"
OUT = ROOT / "outputs/0030_20261028_tggates_unseen_findings" / _RUN / "batch_mixing"
REPS = ["raw_pca64", "D_unsup", "D13_unsup13", "C_lda"]
TITLES = {"raw_pca64": "raw (no rank, PCA64)", "D_unsup": "D: rank, PCA64 (no labels)", "D13_unsup13": "D13: rank, PCA13 (no labels)", "C_lda": "C: rank, LDA13 (Bayer labels)"}
SCAN_COL = {0: "#0072B2", 1: "#D55E00"}                  # SS1601 青、SS1598 橙（Okabe-Ito）
EXP_COL = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
SEED, PER_EXP_ALL, PER_EXP_SEL, N_SEL_PER_SCANNER = 0, 300, 500, 4


def tsne(X):
    return TSNE(perplexity=30, init="pca", random_state=SEED).fit_transform(X.astype(np.float32))


def style(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    z = np.load(OUT / "reps_controls.npz", allow_pickle=True)
    exp, scan, exp_id = z["exp"], z["scanner"], z["exp_id"]
    comp = z["compound"]
    names = [str(s) for s in z["scanner_names"]]
    summ = pd.read_csv(OUT / "summary_controls.csv").set_index("rep")
    rng = np.random.default_rng(SEED)
    exps = np.unique(exp)

    # 図1: 全試験から各300 patch、色＝スキャナ
    pick = np.concatenate([rng.choice(np.where(exp == e)[0], min(PER_EXP_ALL, int((exp == e).sum())), replace=False) for e in exps])
    pick = pick[scan[pick] >= 0]
    fig, axes = plt.subplots(1, len(REPS), figsize=(5.2 * len(REPS), 5.4))
    for ax, r in zip(axes, REPS):
        Y = tsne(z[f"X_{r}"][pick])
        for s in (0, 1):
            m = scan[pick] == s
            ax.scatter(Y[m, 0], Y[m, 1], s=4, alpha=0.5, color=SCAN_COL[s], linewidths=0, label=f"{names[s]} ({int(m.sum()):,} patches)")
        style(ax, f"{TITLES[r]}\nkNN: other scanner {summ.loc[r, 'knn_other_scanner']:.2f} (fully mixed {summ.loc[r, 'knn_other_scanner_expected']:.2f})")
        ax.legend(loc="lower left", fontsize=8, frameon=False, markerscale=4)
    fig.suptitle(f"TG-GATEs control patches, {len(exps)} experiments, {PER_EXP_ALL} patches per experiment; colour = scanner", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "F3_tsne_scanner.png", dpi=150)
    plt.close(fig)
    print("F3 done", flush=True)

    # 図2: スキャナごとに4試験ずつ（slide 数が多い順）、各500 patch。色＝試験、印＝スキャナ
    sel = []
    for s in (0, 1):
        cand = [e for e in exps if (scan[exp == e] == s).mean() > 0.9]
        cand = sorted(cand, key=lambda e: -int((exp == e).sum()))[:N_SEL_PER_SCANNER]
        sel += cand
    pick2 = np.concatenate([rng.choice(np.where((exp == e) & (scan >= 0))[0], min(PER_EXP_SEL, int(((exp == e) & (scan >= 0)).sum())), replace=False) for e in sel])
    fig, axes = plt.subplots(1, len(REPS), figsize=(5.2 * len(REPS), 5.8))
    for ax, r in zip(axes, REPS):
        Y = tsne(z[f"X_{r}"][pick2])
        for i, e in enumerate(sel):
            m = exp[pick2] == e
            s = int(np.bincount(scan[pick2][m]).argmax())
            cp = comp[np.where(exp == e)[0][0]]
            ax.scatter(Y[m, 0], Y[m, 1], s=9 if s == 0 else 14, alpha=0.6, color=EXP_COL[i], marker="o" if s == 0 else "x", linewidths=0 if s == 0 else 0.8,
                       label=f"exp {exp_id[np.where(exp == e)[0][0]]} {cp[:14]} ({names[s]})")
        style(ax, f"{TITLES[r]}\nkNN: other experiment {summ.loc[r, 'knn_other_experiment']:.2f} (fully mixed {summ.loc[r, 'knn_other_experiment_expected']:.2f})")
    axes[0].legend(loc="lower left", fontsize=7, frameon=False, markerscale=2)
    fig.suptitle(f"TG-GATEs control patches, {len(sel)} experiments ({N_SEL_PER_SCANNER} per scanner), {PER_EXP_SEL} patches each; colour = experiment, marker = scanner (o {names[0]}, x {names[1]})", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "F4_tsne_experiments.png", dpi=150)
    plt.close(fig)
    print("F4 done", flush=True)


if __name__ == "__main__":
    main()
