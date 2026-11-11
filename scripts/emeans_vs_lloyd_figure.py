"""0038（e-means と今の k-means の比較）の図。default（単体・今の合議）と v2（e-means の合議）を 1 枚に。
F1: 行 = 方法、左 = 学習時間（seed 5 つの中央値、対数）、右 = seed 間の割り当ての ARI（全 patch。大きい印 = seed の 10 組の平均、薄い点 = 各組）。
    色 = 1 回分の作り方（Lloyd / e-means）、形 = 単体（丸）か合議（四角）。上段 k = 100、下段 k = 1000。
出力: outputs/0038_20261111_emeans_vs_lloyd/F1_time_and_seed_ari.png
usage: python scripts/emeans_vs_lloyd_figure.py   （コンテナ内、Slurm経由。CPU、1 分以内）
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/0038_20261111_emeans_vs_lloyd"
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
LLOYD, EMEANS = "#0072B2", "#D55E00"
# 方法: (表示名（{r} = config の合議の回数）, 色, 形)。emeans_fp32 は emeans と同じ結果なので描かない
METHODS = {
    "lloyd_sub": ("Lloyd 60 万点", LLOYD, "o"),
    "lloyd_full": ("Lloyd 全 patch", LLOYD, "o"),
    "consensus": ("Lloyd の合議（今の方法、60 万点 × {r} 回）", LLOYD, "s"),
    "emeans": ("e-means 全 patch", EMEANS, "o"),
    "emeans_consensus_sub": ("e-means の合議（60 万点 × {r} 回）", EMEANS, "s"),
    "emeans_consensus_full": ("e-means の合議（全 patch × {r} 回）", EMEANS, "s"),
    "emeans_consensus_full50": ("e-means の合議（全 patch × 50 回）", EMEANS, "s"),
}
RUNS = {100: 10, 1000: 5}   # config.yml の consensus_runs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    summ = pd.concat([pd.read_csv(OUT / v / "summary.csv") for v in ("default", "v2")])
    ari = pd.concat([pd.read_csv(OUT / v / "seed_ari.csv") for v in ("default", "v2")])
    ks = sorted(summ.k.unique())
    fig, axes = plt.subplots(len(ks), 2, figsize=(10, 2.5 * len(ks) + 0.6), sharex="col", layout="constrained")
    for (ax_t, ax_a), k in zip(axes, ks):
        for y, (m, (name, col, mk)) in enumerate(METHODS.items()):
            s = summ[(summ.k == k) & (summ.method == m)].iloc[0]
            a = ari[(ari.k == k) & (ari.method == m)].ari
            ax_t.scatter(s.fit_sec, y, s=60, marker=mk, color=col, edgecolors="#333333", linewidths=0.5)
            ax_t.annotate(f"{s.fit_sec:.2g} 秒", (s.fit_sec, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8)
            ax_a.scatter(a, [y] * len(a), s=12, color=col, alpha=0.35, linewidths=0)
            ax_a.scatter(s.ari_mean, y, s=60, marker=mk, color=col, edgecolors="#333333", linewidths=0.5, zorder=3)
            ax_a.annotate(f"{s.ari_mean:.2f}", (a.max(), y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8)
        ax_t.set_yticks(range(len(METHODS)), [n.format(r=RUNS[k]) for n, _, _ in METHODS.values()])
        ax_a.set_yticks(range(len(METHODS)), [])
        for ax in (ax_t, ax_a):
            ax.invert_yaxis()
            ax.grid(axis="y", color="#DDDDDD", lw=0.6)
            ax.set_axisbelow(True)
        ax_t.set_xscale("log")
        ax_t.set_xlim(0.03, 600)
        ax_a.set_xlim(0, 0.7)
        ax_t.set_title(f"k = {k}：学習時間", loc="left", fontsize=9)
        ax_a.set_title(f"k = {k}：seed 間の ARI", loc="left", fontsize=9)
    axes[-1, 0].set_xlabel("学習時間（秒、seed 5 つの中央値、対数）")
    axes[-1, 1].set_xlabel("seed 間の割り当ての ARI（全 patch）")
    hs = [plt.Line2D([], [], ls="", marker="o", color=LLOYD, label="Lloyd"), plt.Line2D([], [], ls="", marker="o", color=EMEANS, label="e-means"),
          plt.Line2D([], [], ls="", marker="o", color="#777777", label="単体"), plt.Line2D([], [], ls="", marker="s", color="#777777", label="合議")]
    fig.legend(handles=hs, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("0038：まとまり 3 の B128（215 万 patch）、seed 5 つ。ARI は大きい印 = seed の 10 組の平均、薄い点 = 各組", fontsize=10)
    fig.savefig(OUT / "F1_time_and_seed_ari.png", dpi=200)


if __name__ == "__main__":
    main()
