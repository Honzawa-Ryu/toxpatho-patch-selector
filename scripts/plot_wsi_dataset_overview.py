"""評価用manifest（3,357枚）のWSI構成・所見の付き方を図にまとめる。

usage: .venv/bin/python scripts/plot_wsi_dataset_overview.py [out_dir]
"""
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/0010_20260925_pathology_benchmark_subset/label_audit/manifest_known_unique.parquet"
WSI_DIR = ROOT / "data/TGGATEs/WSI/Liver"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/dataset_overview_20261012"

DOSES = ["Control", "Low", "Middle", "High"]
DAYS = ["4 day", "8 day", "15 day", "29 day"]
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#c3c2b7"
DOSE_COLORS = ["#c3c2b7", "#9ec5f0", "#5b9be6", "#1f5fae"]  # Control=灰、用量は単色の明→暗
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def load():
    m = pd.read_parquet(MANIFEST)
    # 元データでFINDING_TYPEが空の行（3枚）は種類未記載として残す
    m["types"] = m.finding_types.apply(lambda a: sorted({x if x is not None else "(type not recorded)" for x in a}))
    return m


def fig_overview(m, n_local):
    """(a) 枚数の絞り込み、(b) 用量×時点の枚数と所見あり枚数。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    steps = [("local .svs files", n_local), ("tier1 corpus\n(embedded)", 3441),
             ("minus 80 without\nindividual metadata", 3361), ("minus 4 duplicate animals\n(evaluation set)", len(m))]
    y = np.arange(len(steps))[::-1]
    ax.barh(y, [s[1] for s in steps], color=[GRAY] * 3 + [BLUE], height=0.6)
    for yi, (lab, n) in zip(y, steps):
        ax.text(n + 40, yi, f"{n:,}", va="center", color=INK)
    ax.set_yticks(y, [s[0] for s in steps])
    ax.set_xlim(0, 4000)
    ax.set_title("(a) Number of WSIs (liver, repeated dose)", loc="left", color=INK)

    ax = axes[1]
    ct = m.groupby(["sacrifice_period", "dose_level"]).agg(n=("slide_id", "size"), pos=("has_finding", "sum"))
    w = 0.2
    for j, d in enumerate(DOSES):
        xs = np.arange(4) + (j - 1.5) * w
        n = [ct.loc[(t, d), "n"] for t in DAYS]
        p = [ct.loc[(t, d), "pos"] for t in DAYS]
        ax.bar(xs, n, w * 0.92, color="none", edgecolor=DOSE_COLORS[j], linewidth=1.2)
        ax.bar(xs, p, w * 0.92, color=DOSE_COLORS[j], label=d)
        for x, pi, ni in zip(xs, p, n):
            ax.text(x, ni + 3, f"{pi}/{ni}", ha="center", fontsize=6, color=INK2, rotation=90)
    ax.set_xticks(range(4), DAYS)
    ax.set_ylim(0, 280)
    ax.set_ylabel("WSIs")
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=8)
    ax.set_title("(b) WSIs per time point x dose (outline = all, fill = with finding)", loc="left", color=INK)
    fig.text(0.0, -0.04, f"Evaluation set {len(m):,} WSIs = {m.compound_name.nunique()} compounds x 4 time points x 4 dose groups"
             f" ({m.groupby(['compound_name','sacrifice_period']).ngroups} compound x time studies), all male SD rats."
             " With any finding: {:,} ({:.0%})".format(m.has_finding.sum(), m.has_finding.mean()),
             color=INK2)
    return fig


def fig_compound_heatmap(m):
    """化合物×(時点・用量) の所見あり割合。セル内は枚数。"""
    g = m.groupby(["compound_name", "sacrifice_period", "dose_level"]).agg(
        n=("slide_id", "size"), pos=("has_finding", "mean"))
    cols = [(t, d) for t in DAYS for d in DOSES]
    order = m.groupby("compound_name").has_finding.mean().sort_values(ascending=False).index
    frac = np.full((len(order), len(cols)), np.nan)
    num = np.zeros_like(frac)
    for i, c in enumerate(order):
        for j, (t, d) in enumerate(cols):
            if (c, t, d) in g.index:
                frac[i, j], num[i, j] = g.loc[(c, t, d), "pos"], g.loc[(c, t, d), "n"]
    fig, ax = plt.subplots(figsize=(8.5, 11))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(order)):
        for j in range(len(cols)):
            if num[i, j]:
                ax.text(j, i, int(num[i, j]), ha="center", va="center", fontsize=5.5,
                        color="white" if frac[i, j] > 0.6 else INK2)
            else:
                ax.text(j, i, "–", ha="center", va="center", fontsize=6, color=ORANGE)
    for b in range(1, 4):
        ax.axvline(b * 4 - 0.5, color="white", lw=3)
    ax.set_xticks(range(len(cols)), [d[0] if d != "Control" else "C" for _, d in cols], fontsize=7)
    for k, t in enumerate(DAYS):
        ax.text(k * 4 + 1.5, -1.3, t, ha="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("fraction of WSIs with any finding")
    ax.set_title("Compound x time x dose (C/L/M/H): fraction with any finding (number = WSIs, - = none)\n"
                 "sorted by compound-level fraction with finding", loc="left", color=INK, pad=22)
    return fig


def fig_findings_per_slide(m):
    """1枚あたりの所見の種類数。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    k = m.types.str.len()
    ax = axes[0]
    vc = k.value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=[GRAY] + [BLUE] * (len(vc) - 1), width=0.8)
    for x, v in vc.items():
        ax.text(x, v + 20, f"{v:,}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(vc.index)
    ax.set_xlabel("distinct finding types per WSI (FINDING_TYPE)")
    ax.set_ylabel("WSIs")
    ax.set_title(f"(a) {m.has_finding.sum():,} WSIs with finding; {(k >= 2).sum():,} have 2+ types",
                 loc="left", color=INK)
    ax = axes[1]
    bottom = np.zeros(4)
    bins = [(0, "0"), (1, "1"), (2, "2"), (3, "3+")]
    colors = [GRAY, "#9ec5f0", "#5b9be6", "#1f5fae"]
    for (lo, lab), col in zip(bins, colors):
        sel = (k >= 3) if lab == "3+" else (k == lo)
        v = np.array([(sel & (m.dose_level == d)).sum() for d in DOSES])
        ax.barh(range(4), v, left=bottom, color=col, label=lab, height=0.6, edgecolor="white", linewidth=1)
        bottom += v
    ax.set_yticks(range(4), DOSES)
    ax.invert_yaxis()
    ax.legend(title="types", ncol=4, frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0))
    ax.set_xlabel("WSIs")
    ax.set_title("(b) by dose group", loc="left", color=INK)
    return fig


def fig_finding_types(m, top=25):
    """所見ごとの枚数・化合物数、および重複（共起）。"""
    ex = m[["slide_id", "compound_name", "dose_level", "types"]].explode("types").dropna(subset=["types"])
    s = ex.groupby("types").agg(n=("slide_id", "nunique"), n_cpd=("compound_name", "nunique"),
                                n_ctrl=("dose_level", lambda x: (x == "Control").sum()))
    s = s.sort_values("n", ascending=False)
    t = s.head(top)
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]})
    y = np.arange(len(t))
    ax = axes[0]
    ax.barh(y, t.n - t.n_ctrl, color=BLUE, height=0.65, label="treated")
    ax.barh(y, t.n_ctrl, left=t.n - t.n_ctrl, color=GRAY, height=0.65, label="Control")
    for yi, v in zip(y, t.n):
        ax.text(v + 5, yi, f"{v}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, t.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("WSIs")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(f"(a) WSIs per finding type (top {top} of {len(s)})", loc="left", color=INK)
    ax = axes[1]
    ax.barh(y, t.n_cpd, color=ORANGE, height=0.65)
    for yi, v in zip(y, t.n_cpd):
        ax.text(v + 0.4, yi, f"{v}", va="center", fontsize=7, color=INK2)
    ax.set_xlim(0, 44)
    ax.set_xlabel("compounds (of 44)")
    ax.set_title("(b) compounds showing the finding", loc="left", color=INK)
    fig.text(0, -0.02, f"Remaining {len(s) - top} types: {s.n.iloc[top:].sum()} WSI-findings in total"
             f" ({(s.n == 1).sum()} types occur on a single WSI)", color=INK2)
    return fig, s


def fig_cooccurrence(m, s, top=15, n_combo=20):
    """上位所見の共起行列と、多い組合せ。"""
    names = list(s.index[:top])
    pos = m[m.has_finding]
    sets = pos.types.apply(set)
    co = np.array([[sum(a in x and b in x for x in sets) for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(8, 7))
    off = co.astype(float).copy()
    np.fill_diagonal(off, np.nan)
    im = ax.imshow(off, cmap="Blues", vmin=0)
    for i in range(top):
        for j in range(top):
            v = co[i, j]
            ax.text(j, i, v, ha="center", va="center", fontsize=6,
                    color=INK if i == j else ("white" if v > np.nanmax(off) * 0.6 else INK2),
                    fontweight="bold" if i == j else None)
    ax.set_xticks(range(top), names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(top), names, fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("WSIs with both")
    ax.set_title(f"Co-occurrence of top {top} types (diagonal = WSIs with the type)", loc="left", color=INK)

    fig2, ax = plt.subplots(figsize=(8, 6))
    combos = Counter(" + ".join(sorted(x)) for x in sets).most_common(n_combo)
    labs = [c for c, _ in combos]
    vals = [v for _, v in combos]
    singles = [(" + " not in c) for c in labs]
    ax.barh(range(len(labs)), vals, color=[GRAY if sg else BLUE for sg in singles], height=0.65)
    for yi, v in enumerate(vals):
        ax.text(v + 2, yi, v, va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(labs)), labs, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("WSIs")
    n_uniq = sets.apply(frozenset).nunique()
    ax.set_title(f"Top {n_combo} finding combinations (gray = single, blue = multiple)\n"
                 f"{n_uniq} distinct combinations across {len(pos):,} WSIs with finding", loc="left", color=INK)
    return fig, fig2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = load()
    n_local = sum(1 for _ in WSI_DIR.glob("*_day/*.svs"))
    figs = {"01_overview": fig_overview(m, n_local), "02_compound_heatmap": fig_compound_heatmap(m),
            "03_findings_per_slide": fig_findings_per_slide(m)}
    figs["04_finding_types"], s = fig_finding_types(m)
    figs["05_cooccurrence"], figs["06_combinations"] = fig_cooccurrence(m, s)
    for k, f in figs.items():
        f.savefig(OUT / f"{k}.png")
    s.to_csv(OUT / "finding_type_counts.csv")
    print(OUT)


if __name__ == "__main__":
    main()
