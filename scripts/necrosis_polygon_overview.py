"""Bayerの壊死ポリゴン（Necrosis (area)）の面積の分布と、化合物×時点の枚数を図にする（focal necrosisの定義を決めるための下見）。

focalの閾値は面積の分布を見てユーザーと決める。ここでは閾値を置かず、参考の線（patch 1枚 = 128 µm 四方）だけ引く。
出力: outputs/necrosis_polygon_overview_20261026/（図、表、provenance.json）

usage: python scripts/necrosis_polygon_overview.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKILL = Path("/workspace/andre01/honzawa/.claude/skills/scientific-visualization")
sys.path[:0] = [str(SKILL / "scripts"), str(SKILL / "assets")]
from color_palettes import OKABE_ITO  # noqa: E402
from style_presets import style_context  # noqa: E402

O = ROOT / "outputs"
OUT = O / "necrosis_polygon_overview_20261026"
OUT.mkdir(parents=True, exist_ok=True)
SRC = O / "0019_20261024_bayer_annotation_audit/default/annotations.parquet"
FEAT = O / "0016_20261013_tier1_macenko_embeddings/20x_256px_0px_overlap/features_uni_v2_macenko"
PATCH_UM2 = 128.0 ** 2          # 256 px × 約0.5 µm/px
TIMES = ["3 hr", "6 hr", "9 hr", "24 hr", "4 day", "8 day", "15 day", "29 day"]
MM = 1 / 25.4
C = OKABE_ITO

a = pd.read_parquet(SRC)
a = a[a.label == "Necrosis (area)"].copy()
have = {p.stem for p in FEAT.glob("*.h5")}
a["has_features"] = a.slide_id.astype(str).isin(have)
a["log_area"] = np.log10(a.area_um2)
a["time"] = pd.Categorical(a.SACRIFICE_PERIOD, TIMES, ordered=True)
a.to_csv(OUT / "necrosis_polygons.csv", index=False)

g = a.groupby(["COMPOUND_NAME", "time"], observed=True).agg(
    slides=("slide_id", "nunique"), polygons=("idx", "size"), median_area_um2=("area_um2", "median"),
    slides_with_features=("slide_id", lambda s: s[a.loc[s.index, "has_features"]].nunique())).reset_index()
g.to_csv(OUT / "compound_time_counts.csv", index=False)
order = a.groupby("COMPOUND_NAME").size().sort_values(ascending=False).index.tolist()
n_slides, n_feat = a.slide_id.nunique(), a[a.has_features].slide_id.nunique()

with style_context("default", palette_name="okabe_ito_on_white"):
    # N1. 面積の分布（全体、特徴量の有無で分ける）
    fig, ax = plt.subplots(figsize=(150 * MM, 70 * MM), layout="constrained")
    bins = np.linspace(a.log_area.min() - 0.1, a.log_area.max() + 0.1, 50)
    ax.hist(a.log_area[a.has_features], bins, color=C["blue"], label=f"slides with features ({n_feat} slides)")
    ax.hist(a.log_area[~a.has_features], bins, color=C["orange"], histtype="step", lw=1.4,
            label=f"no features yet ({n_slides - n_feat} slides)")
    for k, ls in ((1, "--"), (10, ":")):
        ax.axvline(np.log10(k * PATCH_UM2), color="0.3", ls=ls, lw=0.9)
        ax.text(np.log10(k * PATCH_UM2), ax.get_ylim()[1] * 0.95, f" {k} patch", fontsize=7, va="top")
    ax.set(xlabel="polygon area, log10 µm²", ylabel="polygons",
           title=f"N1. Bayer 'Necrosis (area)' polygons: {len(a)} polygons, {n_slides} slides")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(OUT / "N1_area_hist.png", dpi=200, facecolor="white")
    plt.close(fig)

    # N2. 化合物ごとの面積（点＝ポリゴン1つ、色・形＝時点）
    cmap = plt.get_cmap("viridis")
    marks = "os^Dv<>P"
    fig, ax = plt.subplots(figsize=(170 * MM, (30 + 6 * len(order)) * MM), layout="constrained")
    rng = np.random.default_rng(20261026)
    for i, cpd in enumerate(order):
        s = a[a.COMPOUND_NAME == cpd]
        for j, t in enumerate(TIMES):
            v = s[s.time == t]
            if len(v):
                ax.scatter(v.log_area, i + rng.uniform(-0.25, 0.25, len(v)), s=7, color=cmap(j / (len(TIMES) - 1)),
                           marker=marks[j], alpha=0.7, edgecolor="none", label=t if i == 0 or t not in ax.get_legend_handles_labels()[1] else None)
        nf = s[s.has_features].slide_id.nunique()
        ax.text(1.002, i, f"{s.slide_id.nunique()} sl ({nf} feat), {len(s)} pol", transform=ax.get_yaxis_transform(), fontsize=6, va="center")
    for k, ls in ((1, "--"), (10, ":")):
        ax.axvline(np.log10(k * PATCH_UM2), color="0.3", ls=ls, lw=0.9)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.invert_yaxis()
    ax.set(xlabel="polygon area, log10 µm²  (dashed: 1 patch, dotted: 10 patches)", title="N2. Necrosis polygon area by compound, colour/marker = sacrifice time")
    h, l = ax.get_legend_handles_labels()
    idx = sorted(range(len(l)), key=lambda k: TIMES.index(l[k]))
    ax.legend([h[k] for k in idx], [l[k] for k in idx], fontsize=6, ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.06 - 0.5 / len(order)), frameon=False)
    fig.savefig(OUT / "N2_area_by_compound.png", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # N3. 化合物×時点: slide数（特徴量あり）とポリゴン数。色＝面積の中央値
    piv = g.pivot(index="COMPOUND_NAME", columns="time", values="median_area_um2").reindex(order)
    fig, ax = plt.subplots(figsize=(150 * MM, (25 + 6 * len(order)) * MM), layout="constrained")
    im = ax.imshow(np.log10(piv.to_numpy(dtype=float)), cmap="viridis", aspect="auto")
    for _, r in g.iterrows():
        y, x = order.index(r.COMPOUND_NAME), TIMES.index(r.time)
        val = np.log10(r.median_area_um2)
        ax.text(x, y, f"{r.slides_with_features}/{r.slides} sl\n{r.polygons} pol", ha="center", va="center", fontsize=5.5,
                color="white" if val < np.nanmean(np.log10(piv.to_numpy(dtype=float))) else "black")
    ax.set_xticks(range(len(TIMES)), TIMES, fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    fig.colorbar(im, ax=ax, label="median polygon area, log10 µm²", shrink=0.6)
    ax.set_title("N3. Slides (with features / all), polygons\nper compound × time; blank = no polygon", fontsize=8)
    fig.savefig(OUT / "N3_compound_time.png", dpi=200, facecolor="white")
    plt.close(fig)

(OUT / "provenance.json").write_text(json.dumps({
    "source": str(SRC.relative_to(ROOT)), "label": "Necrosis (area)", "features_dir": str(FEAT.relative_to(ROOT)),
    "patch_um2": PATCH_UM2, "n_polygons": len(a), "n_slides": int(n_slides), "n_slides_with_features": int(n_feat)}, indent=1))
print(g.to_string(index=False))
