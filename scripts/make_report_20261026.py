"""2026-10-22〜26に任されて進めた解析（0017〜0023）のレポート用の図を、各実験の結果表から描き直す。

図の方針は scientific-visualization スキルに従う: Okabe-Itoの配色、色だけに頼らず形・直接ラベルで区別、元の表（CSV）から描く、
軸の範囲を揃える、生の点を見せる。出力は outputs/report_20261026/（gitの管理外）と、図ごとの元データの一覧 provenance.json。

usage: python scripts/make_report_20261026.py
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
OUT = O / "report_20261026"
OUT.mkdir(parents=True, exist_ok=True)
C = OKABE_ITO
MM = 1 / 25.4
prov = {}


def save(fig, name, sources, note=""):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.05)   # 長いラベルが切れないよう、紙面の大きさは変えてよい（画面表示用）
    plt.close(fig)
    prov[name] = {"sources": [str(s.relative_to(ROOT)) for s in sources], "note": note}


rng = np.random.default_rng(20261026)


def jitter(n, w=0.18):
    return rng.uniform(-w, w, n)


with style_context("default", palette_name="okabe_ito_on_white"):
    # F1. 1段目（Controlと同じクラスタに入りにくいpatch）の通過率と、slideの最大重症度（0018）
    src = O / "0018_20261022_ctrl_filter_then_cores/pass_vs_pathology/slide_pass_rates.csv"
    t = pd.read_csv(src)
    cats = [("Control", t.group == "Control"), ("treated,\nno finding", t.group == "treated, no finding")]
    cats += [(f"finding,\n{g}", (t.group == "treated, finding") & (t.max_grade == i + 1)) for i, g in enumerate(["minimal", "slight", "moderate", "severe"])]
    runs = [("block6_clean", "threshold 0.25 (max a-b)", C["orange"], "o"), ("block6_clean_centered", "slide-mean removed, 0.25", C["sky_blue"], "s"),
            ("block6_clean_thr0.07", "threshold 0.07 (valley)", C["blue"], "D")]
    fig, ax = plt.subplots(figsize=(170 * MM, 75 * MM), layout="constrained")
    for j, (col, lab, colr, mk) in enumerate(runs):
        off = (j - 1) * 0.26
        med = []
        for i, (_, m) in enumerate(cats):
            v = t.loc[m, col].dropna().to_numpy()
            ax.scatter(np.full(len(v), i + off) + jitter(len(v), .07), v, s=4, color=colr, alpha=.25, linewidths=0, marker=mk)
            med.append(np.median(v))
        ax.plot(np.arange(len(cats)) + off, med, color=colr, marker=mk, ms=6, lw=1.5, label=lab, markeredgecolor="black", markeredgewidth=.5)
    for i, (_, m) in enumerate(cats):
        ax.text(i, 1.04, f"n={int(m.sum())}", ha="center", fontsize=7)
    ax.set_xticks(range(len(cats)), [c for c, _ in cats], fontsize=8)
    ax.set_ylim(-0.02, 1.1)
    ax.set_ylabel("fraction of the slide's eval patches\npassing stage 1")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3)
    ax.set_title("Stage-1 pass rate rises with the slide's most severe finding (line = median, dots = slides)", fontsize=9)
    save(fig, "F1_stage1_pass_by_grade", [src], "2,271 A_clean slides of the largest Ctrl block (0017 block6_clean); eval patches ~50/slide; grade = most severe non-spontaneous finding")

    # F2. バッチ効果に頑健な表現の比較（0020、Bayerのアノテーション132枚）
    src = O / "0020_20261024_batch_robust_methods/compare_132/metrics.csv"
    srcs = O / "0020_20261024_batch_robust_methods/supervised_132/supervised_metrics.csv"
    m_ = pd.read_csv(src)
    s_ = pd.read_csv(srcs)
    fam = {"raw": ("baseline", C["black"], "o"), "pca64": ("baseline", C["black"], "o"),
           "center_slide": ("within-slide", C["blue"], "s"), "zscore_slide": ("within-slide", C["blue"], "s"), "rank_slide": ("within-slide", C["blue"], "s"),
           "center_stripe": ("within-slide", C["blue"], "s"), "coral_slide": ("integration", C["vermillion"], "^"), "harmony_slide": ("integration", C["vermillion"], "^"),
           "harmony_exp": ("integration", C["vermillion"], "^"), "center_harmony": ("integration", C["vermillion"], "^"),
           "mnn_contrast": ("contrastive", C["bluish_green"], "D"), "mnn_contrast_rank": ("contrastive", C["bluish_green"], "D")}
    offs = {"center_slide": (-4, -10), "rank_slide": (4, 4), "mnn_contrast": (-10, -11), "harmony_slide": (-8, -11), "pca64": (4, -9)}   # 重なるラベルだけずらす
    fig, ax = plt.subplots(figsize=(150 * MM, 95 * MM), layout="constrained")
    for r in m_.dropna(subset=["other_exp_nn"]).itertuples():
        f, col, mk = fam[r.method]
        ax.scatter(r.other_exp_nn, r.bal_acc_exp_out, color=col, marker=mk, s=40, edgecolors="black", linewidths=.4, zorder=3)
        ax.annotate(r.method, (r.other_exp_nn, r.bal_acc_exp_out), fontsize=6.5, xytext=offs.get(r.method, (3, 3)), textcoords="offset points")
    for r in s_[s_.space == "lda"].itertuples():
        ax.scatter(r.other_exp_nn, r.bal_acc_exp_out, color=C["reddish_purple"], marker="*", s=110, edgecolors="black", linewidths=.4, zorder=3)
        ax.annotate(f"{r.input} + LDA (CV by exp.)", (r.other_exp_nn, r.bal_acc_exp_out), fontsize=6.5, xytext=(3, -9), textcoords="offset points")
    for lab_, col, mk in [("baseline", C["black"], "o"), ("within-slide normalisation", C["blue"], "s"), ("integration (Harmony, CORAL)", C["vermillion"], "^"),
                          ("contrastive (MNN pairs)", C["bluish_green"], "D"), ("weakly supervised (LDA on Bayer labels)", C["reddish_purple"], "*")]:
        ax.scatter([], [], color=col, marker=mk, edgecolors="black", linewidths=.4, label=lab_)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_xlabel("batch mixing: share of 30-NN from another experiment")
    ax.set_ylabel("label conservation: balanced accuracy,\nkNN using only other experiments")
    ax.set_title("Mixing vs conservation (132 annotated slides, 21,559 labelled patches)", fontsize=9)
    save(fig, "F2_methods_mixing_vs_conservation", [src, srcs], "LDA rows are 5-fold cross-validated by experiment (evaluated on held-out experiments only)")

    # F3. 隣接情報の効かせ方: まとまりとラベルとの一致の入れ替わり（0021）
    s1 = O / "0021_20261025_patch_prototypes/context_v1/context_metrics.csv"
    s2 = O / "0021_20261025_patch_prototypes/explore_v2/context_explore_metrics.csv"
    s3 = O / "0021_20261025_patch_prototypes/explore_v3_lobular/context_explore_metrics.csv"
    a = pd.read_csv(s1)
    b = pd.read_csv(s2)
    c = pd.read_csv(s3)
    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 68 * MM), layout="constrained", gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axes[0]
    ax.plot(a.neighbor_same_prototype, a.nmi_label, color=C["blue"], marker="o", label="smooth, then assign to fixed prototypes")
    for r in a.itertuples():
        ax.annotate(f"{r.hops} hops", (r.neighbor_same_prototype, r.nmi_label), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    bb = b[b.variant != "self_only"]
    ax.scatter(bb.neighbor_same_prototype, bb.nmi_label, color=C["orange"], marker="s", s=30, edgecolors="black", linewidths=.4, label="rebuild prototypes (concat / two-level)", zorder=3)
    for r in bb.itertuples():
        ax.annotate(r.variant, (r.neighbor_same_prototype, r.nmi_label), fontsize=6, xytext=(3, -8), textcoords="offset points")
    ax.set_xlabel("adjacent patches in the same prototype")
    ax.set_ylabel("NMI with Bayer labels")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    ax.set_title("Spatial context: coherence vs label agreement", fontsize=9)
    x = np.arange(len(c))
    for ax, col, colr, title in [(axes[1], "nmi_label", C["blue"], "NMI with Bayer labels"), (axes[2], "nmi_lobular_pos", C["vermillion"], "NMI with lobular position (3 bins)")]:
        ax.bar(x, c[col], width=.6, color=colr, edgecolor="black", linewidth=.4)
        for i, v in enumerate(c[col]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x, ["self only", "lobular w0.5", "lobular w1", "lobular w2"], fontsize=6.5, rotation=25)
        ax.set_ylim(0, c[col].max() * 1.2)
        ax.set_title(title, fontsize=8)
    save(fig, "F3_spatial_context_tradeoff", [s1, s2, s3], "132 annotated slides, all patches (1.98 M); k = 30")

    # F4. 小葉の位置の層: 病変patchの予測位置と、所見表の分布（0023、試験ごとに分けた予測）
    src = O / "0023_20261025_layer_reality_check/lobular_v1/lesion_positions.csv"
    t = pd.read_csv(src)
    t = t[t.main_topography.isin(["Centrilobular", "Periportal", "Glisson's sheath", "Bile duct cell"])].copy()
    t["side"] = np.where(t.main_topography == "Centrilobular", "recorded centrilobular", "recorded periportal / portal")
    t = t.sort_values("rel_pos_median")
    fig, ax = plt.subplots(figsize=(150 * MM, 65 * MM), layout="constrained")
    for i, r in enumerate(t.itertuples()):
        col, mk = (C["blue"], "o") if r.side == "recorded centrilobular" else (C["vermillion"], "s")
        ax.plot([r.rel_pos_q25, r.rel_pos_q75], [i, i], color=col, lw=2)
        ax.scatter(r.rel_pos_median, i, color=col, marker=mk, s=45, edgecolors="black", linewidths=.4, zorder=3)
    ax.set_yticks(range(len(t)), [f"{r.label.replace('_', ' ')} - {r.compound} ({r.n_patches} p)" for r in t.itertuples()], fontsize=7)
    ax.axvline(.5, color="#999999", lw=.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("predicted lobular position (rank within slide; 0 = central-vein side, 1 = portal side)")
    ax.scatter([], [], color=C["blue"], marker="o", label="recorded centrilobular")
    ax.scatter([], [], color=C["vermillion"], marker="s", label="recorded periportal / Glisson's sheath / bile duct")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(.4, -.22), ncol=2)
    ax.set_title("Lesion patches: predicted lobular position vs recorded topography\n(model trained without the compound's experiment; median and IQR)", fontsize=8.5)
    save(fig, "F4_lobular_layer_vs_topography", [src], "Bayer lesion patches; position predicted out-of-fold by experiment (5 folds); OOF R2 0.25")

    # F5. 外部（エーザイ・シオノギ）: TG-GATEsだけで学習した層をそのまま当てる（0023）
    src = O / "0023_20261025_layer_reality_check/external_v2/external_layers.csv"
    t = pd.read_csv(src)
    t["ctrl"] = t.control_status.isin(["explicit"])
    t["cl_nec"] = t.centrilobular_necrosis.fillna(0)
    t["grp"] = np.where(t.ctrl, "control", np.where(t.cl_nec > 0, "centrilobular necrosis recorded", "treated, no CL necrosis recorded"))
    order = ["control", "treated, no CL necrosis recorded", "centrilobular necrosis recorded"]
    cols = {"control": (C["black"], "o"), "treated, no CL necrosis recorded": (C["orange"], "^"), "centrilobular necrosis recorded": (C["vermillion"], "s")}
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 70 * MM), layout="constrained")
    for ax, col, ylab, ylim in [(axes[0], "share_necrosis", "share of patches in TG-GATEs\nnecrosis prototypes", None),
                                (axes[1], "relpos_necrosis", "lobular position of those patches\n(0 = central-vein side)", (0, 1))]:
        for i, g in enumerate(order):
            v = t.loc[t.grp == g, col].dropna()
            colr, mk = cols[g]
            ax.scatter(np.full(len(v), i) + jitter(len(v)), v, color=colr, marker=mk, s=18, edgecolors="black", linewidths=.3, alpha=.85)
            ax.plot([i - .25, i + .25], [np.median(v)] * 2, color="black", lw=1.5)
            ax.text(i, (ylim[1] if ylim else t[col].max()) * 1.03, f"n={len(v)}", ha="center", fontsize=7)
        ax.set_xticks(range(3), ["control", "treated,\nno CL necrosis", "CL necrosis\nrecorded"], fontsize=8)
        ax.set_ylabel(ylab)
        if ylim:
            ax.set_ylim(ylim[0], ylim[1] * 1.08)
            ax.axhline(.5, color="#999999", lw=.8, ls="--")
    fig.suptitle("External (Eisai 18, Shionogi 60 slides): TG-GATEs-trained layers, no external training", fontsize=9)
    save(fig, "F5_external_necrosis_layers", [src], "CL = centrilobular, from company findings; grades on company scales not combined")

    # F6. 3層（形態・領域の広がり・小葉の位置）と記録（0023 three_layers_v1）
    s1 = O / "0023_20261025_layer_reality_check/three_layers_v1/tgg_slide_layers.csv"
    s2 = O / "0023_20261025_layer_reality_check/three_layers_v1/external_slide_layers.csv"
    tg = pd.read_csv(s1)
    ex = pd.read_csv(s2)
    from scipy.stats import spearmanr
    fig, axes = plt.subplots(2, 3, figsize=(180 * MM, 110 * MM), layout="constrained")
    for rowi, (t, name) in enumerate([(tg, "TG-GATEs, 132 annotated slides"), (ex, "External, 78 slides (no external training)")]):
        grades = sorted(t.necrosis_grade.unique())
        ax = axes[rowi, 0]
        for g in grades:
            v = t.loc[t.necrosis_grade == g, "share_necrosis"]
            ax.scatter(np.full(len(v), g) + jitter(len(v), .12), v, s=9, color=C["blue"], alpha=.7, linewidths=0)
            ax.plot([g - .25, g + .25], [v.median()] * 2, color="black", lw=1.4)
        rho = spearmanr(t.necrosis_grade, t.share_necrosis)[0]
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel("recorded necrosis grade (0 = none)")
        ax.set_ylabel("share in necrosis prototypes\n(symlog, linear below 0.01)")
        ax.set_title(f"{name}\nmorphology: Spearman {rho:.2f}", fontsize=8)
        ax = axes[rowi, 1]
        if rowi == 0:
            for g in grades:
                v = t.loc[t.necrosis_grade == g, "extent_necrosis"].dropna()
                ax.scatter(np.full(len(v), g) + jitter(len(v), .12), v, s=9, color=C["bluish_green"], alpha=.7, linewidths=0, marker="s")
            ok = t.extent_necrosis.notna() & (t.necrosis_grade > 0)
            ax.set_title(f"extent around necrosis patches\nSpearman (necrosis slides) {spearmanr(t.necrosis_grade[ok], t.extent_necrosis[ok])[0]:.2f}", fontsize=8)
            ax.set_xlabel("recorded necrosis grade")
        else:
            cats = [("none", t.necrosis_grade == 0), ("focal", t.necrosis_focal & ~t.necrosis_massive),
                    ("centri-\nlobular", t.necrosis_centrilobular & ~t.necrosis_massive), ("massive", t.necrosis_massive)]
            for i, (nm, mk) in enumerate(cats):
                v = t.loc[mk, "extent_necrosis"].dropna()
                ax.scatter(np.full(len(v), i) + jitter(len(v), .12), v, s=9, color=C["bluish_green"], alpha=.7, linewidths=0, marker="s")
                ax.text(i, .95, f"n={len(v)}", ha="center", fontsize=6.5)
            ax.set_xticks(range(4), [c for c, _ in cats], fontsize=7)
            ax.set_title("extent by recorded pattern", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("neighbourhood density of\nnecrosis prototypes (median)")
        ax = axes[rowi, 2]
        if rowi == 0:
            cats = [("Centrilobular", "centri-\nlobular"), ("Hepatocyte", "hepato-\ncyte"), ("Midlobular", "mid-\nlobular"), ("Periportal", "peri-\nportal")]
            for i, (tp, lab_) in enumerate(cats):
                v = t.loc[t.necrosis_topo == tp, "relpos_necrosis"].dropna()
                ax.scatter(np.full(len(v), i) + jitter(len(v), .12), v, s=9, color=C["vermillion"], alpha=.8, linewidths=0, marker="D")
                ax.text(i, .97, f"n={len(v)}", ha="center", fontsize=6.5)
            ax.set_xticks(range(4), [l for _, l in cats], fontsize=7)
            ax.set_title("lobular position of necrosis patches\n(out-of-fold by experiment)", fontsize=8)
        else:
            cats = [("none\nrecorded", t.necrosis_grade == 0), ("necrosis,\nnot CL", (t.necrosis_grade > 0) & ~t.necrosis_centrilobular), ("centri-\nlobular", t.necrosis_centrilobular)]
            for i, (nm, mk) in enumerate(cats):
                v = t.loc[mk, "relpos_necrosis"].dropna()
                ax.scatter(np.full(len(v), i) + jitter(len(v), .12), v, s=9, color=C["vermillion"], alpha=.8, linewidths=0, marker="D")
                ax.text(i, .97, f"n={len(v)}", ha="center", fontsize=6.5)
            ax.set_xticks(range(3), [c for c, _ in cats], fontsize=7)
            ax.set_title("lobular position of necrosis patches", fontsize=8)
        ax.axhline(.5, color="#999999", lw=.8, ls="--")
        ax.set_ylim(0, 1.03)
        ax.set_ylabel("rank in slide (0 = central-vein side)")
    save(fig, "F6_three_layers_vs_records", [s1, s2], "necrosis prototypes = 0021 combo clusters 7 and 25; extent = 8-hop smoothing of the prototype indicator")

(OUT / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False))
print(OUT)
