"""TG-GATEs内の試験・スキャナの混ざり方を、0029の「外部データとの混ざり」の図と同じ形（表現ごとに1枚の平面プロット）で描く
（2026-11-02ユーザー依頼）。入力は scripts/tggates_unseen_batch_mixing.py が保存した reps_controls.npz（対照群 857枚・各1,024 patch の raw / D / D13 / C）。

F3_tsne_scanner.png      : 表現ごとに、全43試験から各300 patch を t-SNE。色＝スキャナ（SS1601 / SS1598）
F4_tsne_experiments.png  : スキャナごとに3試験ずつ選んだ6試験から各500 patch を t-SNE。上段 色＝試験、下段 同じ点を色＝スキャナ（印はすべて丸。2026-11-02ユーザー: 印と色を減らす）
F5_mixing_by_experiment.png : 試験ごとに、近傍30個のうち同じ試験（同じslideは除く）の割合（観測）と完全に混ざったときの期待値。t-SNEの見た目を数値で裏付ける
各パネルの題に、summary_controls.csv の近傍の混ざりを添える。
usage: python scripts/batch_mixing_tsne_figure.py [版]  （コンテナ内、Slurm経由。CPUで10分程度）
"""
import sys
import time
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
EXP_COL = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]   # Okabe-Ito から黄と黒を除いた6色
SEED, PER_EXP_ALL, PER_EXP_SEL, N_SEL_PER_SCANNER = 0, 300, 500, 3
KNN, QUERIES_PER_EXP = 30, 400


def tsne(X):
    return TSNE(perplexity=30, init="pca", random_state=SEED).fit_transform(X.astype(np.float32))


def style(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def knn_same_experiment(X, slide, exp, q):
    """問い合わせ q の近傍 KNN 個（同じ slide は除く）のうち、同じ試験の割合。CPU で塊ごとに距離を計算。"""
    X = X.astype(np.float32)
    nrm = (X ** 2).sum(1)
    out = np.empty(len(q))
    for s in range(0, len(q), 200):
        qq = q[s:s + 200]
        D = nrm[None, :] - 2 * X[qq] @ X.T + nrm[qq][:, None]
        D[slide[qq][:, None] == slide[None, :]] = np.inf
        nb = np.argpartition(D, KNN, axis=1)[:, :KNN]
        out[s:s + 200] = (exp[nb] == exp[qq][:, None]).mean(1)
    return out


def main():
    t0 = time.time()
    z = np.load(OUT / "reps_controls.npz", allow_pickle=True)
    exp, scan, slide, exp_id, comp = z["exp"], z["scanner"], z["slide"], z["exp_id"], z["compound"]
    names = [str(s) for s in z["scanner_names"]]
    summ = pd.read_csv(OUT / "summary_controls.csv").set_index("rep")
    rng = np.random.default_rng(SEED)
    exps = np.unique(exp)
    X = {r: z[f"X_{r}"] for r in REPS}

    # F3: 全試験から各300 patch、色＝スキャナ
    pick = np.concatenate([rng.choice(np.where(exp == e)[0], min(PER_EXP_ALL, int((exp == e).sum())), replace=False) for e in exps])
    pick = pick[scan[pick] >= 0]
    fig, axes = plt.subplots(1, len(REPS), figsize=(5.2 * len(REPS), 5.4))
    for ax, r in zip(axes, REPS):
        Y = tsne(X[r][pick])
        for s in (0, 1):
            m = scan[pick] == s
            ax.scatter(Y[m, 0], Y[m, 1], s=5, alpha=0.45, color=SCAN_COL[s], linewidths=0, label=f"{names[s]} ({int(m.sum()):,} patches)")
        style(ax, f"{TITLES[r]}\nkNN: other scanner {summ.loc[r, 'knn_other_scanner']:.2f} (fully mixed {summ.loc[r, 'knn_other_scanner_expected']:.2f})")
        ax.legend(loc="lower left", fontsize=8, frameon=False, markerscale=4)
    fig.suptitle(f"TG-GATEs control patches, {len(exps)} experiments, {PER_EXP_ALL} patches per experiment; colour = scanner", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "F3_tsne_scanner.png", dpi=150)
    plt.close(fig)
    print(f"F3 done; {time.time() - t0:.0f}s", flush=True)

    # F4: スキャナごとに3試験（slide 数が多い順）、各500 patch。上段 色＝試験、下段 色＝スキャナ（同じ点）
    sel = []
    for s in (0, 1):
        cand = [e for e in exps if (scan[exp == e] == s).mean() > 0.9]
        sel += sorted(cand, key=lambda e: -int((exp == e).sum()))[:N_SEL_PER_SCANNER]
    pick2 = np.concatenate([rng.choice(np.where((exp == e) & (scan >= 0))[0], min(PER_EXP_SEL, int(((exp == e) & (scan >= 0)).sum())), replace=False) for e in sel])
    order = rng.permutation(len(pick2))                 # 描く順を混ぜて、後に描いた色が上に乗らないように
    pick2 = pick2[order]
    fig, axes = plt.subplots(2, len(REPS), figsize=(5.2 * len(REPS), 10.4))
    for j, r in enumerate(REPS):
        Y = tsne(X[r][pick2])
        ax = axes[0, j]
        cols = np.empty(len(pick2), dtype=object)
        for i, e in enumerate(sel):
            cols[exp[pick2] == e] = EXP_COL[i]
        ax.scatter(Y[:, 0], Y[:, 1], s=7, alpha=0.6, c=list(cols), linewidths=0)
        for i, e in enumerate(sel):
            sl0 = slide[np.where(exp == e)[0][0]]
            s = int(np.bincount(scan[exp == e]).argmax())
            ax.scatter([], [], s=30, color=EXP_COL[i], label=f"{comp[sl0][:18]} (exp {exp_id[sl0]}, {names[s]})")
        style(ax, f"{TITLES[r]}\nkNN: other experiment {summ.loc[r, 'knn_other_experiment']:.2f} (fully mixed {summ.loc[r, 'knn_other_experiment_expected']:.2f})")
        if j == 0:
            ax.legend(loc="lower left", fontsize=7, frameon=False)
        ax = axes[1, j]
        for s in (0, 1):
            m = scan[pick2] == s
            ax.scatter(Y[m, 0], Y[m, 1], s=7, alpha=0.5, color=SCAN_COL[s], linewidths=0, label=names[s])
        style(ax, "same points, colour = scanner")
        if j == 0:
            ax.legend(loc="lower left", fontsize=8, frameon=False, markerscale=3)
    fig.suptitle(f"TG-GATEs control patches, {len(sel)} experiments ({N_SEL_PER_SCANNER} per scanner), {PER_EXP_SEL} patches each", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "F4_tsne_experiments.png", dpi=150)
    plt.close(fig)
    print(f"F4 done; {time.time() - t0:.0f}s", flush=True)

    # F5: 試験ごとの、近傍のうち同じ試験の割合（観測 vs 期待）。期待 = 自分の slide を除いた参照のうち同じ試験の割合
    n_e = np.bincount(exp)
    n_s = np.bincount(slide)
    q = np.concatenate([rng.choice(np.where(exp == e)[0], min(QUERIES_PER_EXP, int((exp == e).sum())), replace=False) for e in exps])
    rows = []
    for r in REPS:
        same = knn_same_experiment(X[r], slide, exp, q)
        expd = (n_e[exp[q]] - n_s[slide[q]]) / (len(exp) - n_s[slide[q]])
        for e in exps:
            m = exp[q] == e
            sl0 = slide[np.where(exp == e)[0][0]]
            rows.append({"rep": r, "exp_id": exp_id[sl0], "compound": comp[sl0], "scanner": names[int(np.bincount(scan[exp == e]).argmax())] if (scan[exp == e] >= 0).any() else "",
                         "same_experiment_observed": float(same[m].mean()), "same_experiment_expected": float(expd[m].mean())})
        print(f"F5 kNN {r}; {time.time() - t0:.0f}s", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "mixing_by_experiment.csv", index=False)
    fig, axes = plt.subplots(1, len(REPS), figsize=(5.2 * len(REPS), 4.6), sharey=True)
    for ax, r in zip(axes, REPS):
        t_ = tab[tab.rep == r].sort_values("same_experiment_observed", ascending=False).reset_index(drop=True)
        xs = np.arange(len(t_))
        ax.bar(xs, t_.same_experiment_observed, color=[SCAN_COL[names.index(s)] if s in names else "#999999" for s in t_.scanner], width=0.8, linewidth=0)
        ax.plot(xs, t_.same_experiment_expected, color="black", lw=1, label="expected if fully mixed")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{c[:12]}" for c in t_.compound], rotation=90, fontsize=5)
        ax.set_title(f"{TITLES[r]}\nmean {t_.same_experiment_observed.mean():.3f} vs expected {t_.same_experiment_expected.mean():.3f}", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("share of 30 nearest neighbours from the same experiment\n(same slide excluded)")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Line2D([], [], color="black", lw=1, label="expected if fully mixed")] + [Patch(color=c_, label=names[s_]) for s_, c_ in SCAN_COL.items()], fontsize=8, frameon=False)
    fig.suptitle("Which experiments still stay apart: per-experiment self-neighbour share (control patches)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "F5_mixing_by_experiment.png", dpi=150)
    plt.close(fig)
    print(f"F5 done; {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
