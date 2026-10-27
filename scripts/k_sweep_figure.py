"""0025 search_v5（kの掃引）の図: kごとの純度・NMI・seedに対する安定性（くり返しの間のARI）。

元データ: outputs/0025_20261026_context_mixing_search/search_v5/search_runs.csv（各設定5回のくり返し）
出力: outputs/0025_20261026_context_mixing_search/search_v5/figures/K1_k_sweep.png
点はくり返しの平均、ひげは標準偏差（5回、k-meansの学習点とseedを変えた）。指標はBayerのラベル付きpatchだけで計算（ARIは全patchから無作為の30万）。

usage: python scripts/k_sweep_figure.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path("/workspace/andre01/honzawa/.claude/skills/scientific-visualization")
sys.path[:0] = [str(SKILL / "scripts"), str(SKILL / "assets")]
from color_palettes import OKABE_ITO  # noqa: E402
from style_presets import style_context  # noqa: E402

RUN = ROOT / "outputs/0025_20261026_context_mixing_search/search_v5"
r = pd.read_csv(RUN / "search_runs.csv")
r["k"] = r["k"].astype(int)
g = r.groupby(["kind", "k"])
m, s = g.mean(numeric_only=True).reset_index(), g.std(numeric_only=True).reset_index()
panels = [("hom_test", "homogeneity (test experiments)\nhigher = clusters hold one label"),
          ("small_purity_test", "purity of small structures (test)\ncentral vein, portal tract, mitosis, dark hep."),
          ("nmi_test", "NMI (test experiments)"),
          ("ari_between_repeats", "agreement between 5 runs (ARI)\nhigher = less seed-dependent"),
          ("neighbor_same", "adjacent patches in the same prototype")]
style = {"self": ("no neighbourhood", OKABE_ITO["blue"], "o"), "bilateral": ("edge-aware propagation", OKABE_ITO["vermillion"], "s")}
with style_context("default", palette_name="okabe_ito_on_white"):
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 3.8), layout="constrained")
    for ax, (col, lab) in zip(axes, panels):
        for kind, (name, colr, mk) in style.items():
            a, b = m[m.kind == kind], s[s.kind == kind]
            ax.errorbar(a.k, a[col], yerr=b[col] if col != "ari_between_repeats" else None, color=colr, marker=mk, ms=4, lw=1.2, capsize=2, label=name)
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 30, 50, 100, 200], ["10", "20", "30", "50", "100", "200"])
        ax.axvline(30, color="0.6", lw=0.7, ls=":")
        ax.set_xlabel("k (number of prototypes, log scale)")
        ax.set_title(lab, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("K1. Choosing k: 230 annotated slides, out-of-fold LDA, 5 runs per point (dotted line: k = 30 used so far)", fontsize=10)
    (RUN / "figures").mkdir(exist_ok=True)
    fig.savefig(RUN / "figures/K1_k_sweep.png", dpi=180, facecolor="white")
print("ok")
