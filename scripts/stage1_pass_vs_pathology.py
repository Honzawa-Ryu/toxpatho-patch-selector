"""0018の1段目を通ったpatchの割合（slideごと）を、病理所見の重症度・分布（TOPOGRAPHY_TYPE）と比べる。

slideごとの割合は評価用patch（1枚あたり最大50）で数える。病理はTG-GATEsの所見表（data/corrected/open_tggates_pathology.csv）から、
個体（exp_id, group_id, individual_id）で結びつける。自然発生（SP_FLG=true）の所見は別扱いにし、主な集計からは除く。
同じ所見が複数の分布で記録されていれば、それぞれの行として数え、重症度は最も重いものを採る。

usage: python scripts/stage1_pass_vs_pathology.py run1 [run2 ...]   （0018の出力名。例: block6_clean block6_clean_centered block6_clean_thr0.07）
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/0018_20261022_ctrl_filter_then_cores"
MANIFEST = ROOT / "experiments/0017_20261016_block_consensus_patterns/manifest_block6_cut072.parquet"
GRADES = ["minimal", "slight", "moderate", "severe"]
KEY = ["exp_id", "group_id", "individual_id"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 140})


def pathology(slide_ids):
    m = pd.read_parquet(MANIFEST)
    m["slide_id"] = m.slide_id.astype(str)
    m = m[m.slide_id.isin(slide_ids)][["slide_id", "compound_name", "dose_level", "sacrifice_period"] + KEY]
    p = pd.read_csv(ROOT / "data/corrected/open_tggates_pathology.csv")
    p = p[p.ORGAN == "Liver"].rename(columns={"EXP_ID": "exp_id", "GROUP_ID": "group_id", "INDIVIDUAL_ID": "individual_id"})
    for c in KEY:
        m[c] = pd.to_numeric(m[c]).astype(int)
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p = p.dropna(subset=KEY)
    p[KEY] = p[KEY].astype(int)
    j = m.merge(p, on=KEY, how="inner")
    j["grade_ord"] = j.GRADE_TYPE.map({g: i + 1 for i, g in enumerate(GRADES)})
    j["spontaneous"] = j.SP_FLG.astype(str).str.lower() == "true"
    j["topography"] = j.TOPOGRAPHY_TYPE.fillna("(not given)")
    return m, j


def main():
    runs = sys.argv[1:]
    out = RUNS / "pass_vs_pathology"
    out.mkdir(exist_ok=True)
    rates = {}
    for r in runs:
        s = pd.read_parquet(RUNS / r / "stage1_scores.parquet")
        g = s.groupby("slide_id").agg(kept=("kept", "mean"), n=("kept", "size"), group=("group", "first"))
        rates[r] = g
    base = rates[runs[0]][["group", "n"]].copy()
    for r in runs:
        base[r] = rates[r].kept
    base = base.reset_index()
    m, j = pathology(set(base.slide_id))
    base = base.merge(m[["slide_id", "compound_name", "dose_level", "sacrifice_period"]], on="slide_id", how="left")
    tr = j[~j.spontaneous]
    sl_grade = tr.groupby("slide_id").grade_ord.max()
    base["max_grade"] = base.slide_id.map(sl_grade)
    base["only_spontaneous"] = base.slide_id.isin(j[j.spontaneous].slide_id) & ~base.slide_id.isin(tr.slide_id)
    base.to_csv(out / "slide_pass_rates.csv", index=False)

    # 1. slideの群・最大重症度ごと
    cats = [("Control", base.group == "Control"), ("treated,\nno finding", base.group == "treated, no finding")]
    cats += [(f"finding,\nmax {g}", (base.group == "treated, finding") & (base.max_grade == i + 1)) for i, g in enumerate(GRADES)]
    cats += [("finding,\nspontaneous only", (base.group == "treated, finding") & base.only_spontaneous)]
    fig, axes = plt.subplots(1, len(runs), figsize=(6.5 * len(runs), 5.5), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, r in zip(axes[0], runs):
        data = [base.loc[mk, r].dropna().to_numpy() for _, mk in cats]
        ax.boxplot(data, widths=.55, showfliers=False)
        for i, d in enumerate(data):
            ax.scatter(i + 1 + rng.uniform(-.2, .2, len(d)), d, s=4, alpha=.35, color="#2a78d6", linewidths=0)
            ax.text(i + 1, 1.03, f"n={len(d)}", ha="center", fontsize=7)
        ax.set_xticks(range(1, len(cats) + 1), [c for c, _ in cats], fontsize=7.5)
        ax.set_title(r, fontsize=10)
        ax.set_ylim(-.02, 1.08)
    axes[0, 0].set_ylabel("fraction of the slide's eval patches passing stage 1")
    fig.suptitle("P1. Stage-1 pass rate per slide, by slide group and the slide's most severe treatment-related finding", fontsize=12)
    fig.savefig(out / "P1_pass_rate_by_grade.png")
    plt.close(fig)

    # 2. 所見ごと: 分布（TOPOGRAPHY_TYPE）別・重症度別の、所見ありslideの通過率（中央値）
    rows = []
    for (f, topo), d in tr.groupby(["FINDING_TYPE", "topography"]):
        ids = d.slide_id.unique()
        rec = {"finding": f, "topography": topo, "n_slides": len(ids), "n_exp": d.exp_id.nunique(),
               "median_grade": float(d.groupby("slide_id").grade_ord.max().median())}
        b = base.set_index("slide_id").loc[ids]
        for r in runs:
            rec[r] = float(b[r].median())
        rows.append(rec)
    ft = pd.DataFrame(rows).sort_values(["finding", "n_slides"], ascending=[True, False])
    ft.to_csv(out / "pass_rate_by_finding_topography.csv", index=False)
    show = ft[ft.n_slides >= 9].sort_values(runs[0])
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(show) + 2))
    y = np.arange(len(show))
    for r, c in zip(runs, ["#9aa3ad", "#e34948", "#2a78d6", "#1baf7a"]):
        ax.scatter(show[r], y, color=c, s=28, label=r, zorder=3)
    for i in range(len(show)):
        ax.plot([show.iloc[i][runs].min(), show.iloc[i][runs].max()], [i, i], color="#cccccc", zorder=1)
    ax.set_yticks(y, [f"{a[:30]} @ {b[:22]} ({n} sl, {e} exp, grade {g:.0f})" for a, b, n, e, g in
                      zip(show.finding, show.topography, show.n_slides, show.n_exp, show.median_grade)], fontsize=7.5)
    ax.set_xlabel("median stage-1 pass rate of slides recording the finding")
    ax.set_xlim(-.02, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("P2. Pass rate per finding and topography (findings with >= 9 slides; grade = median of slide max grade, 1 minimal .. 4 severe)", fontsize=10)
    fig.savefig(out / "P2_pass_rate_by_finding_topography.png")
    plt.close(fig)

    # 3. 主な所見で、重症度ごとの通過率
    main_f = tr.groupby("FINDING_TYPE").slide_id.nunique().loc[lambda v: v >= 30].sort_values(ascending=False).index
    fig, axes = plt.subplots(len(runs), len(main_f), figsize=(3.2 * len(main_f), 3.2 * len(runs)), sharey=True, squeeze=False)
    corr = []
    for ci, f in enumerate(main_f):
        g = tr[tr.FINDING_TYPE == f].groupby("slide_id").grade_ord.max()
        b = base.set_index("slide_id").loc[g.index]
        for ri, r in enumerate(runs):
            ax = axes[ri, ci]
            data = [b[r][g == i + 1].to_numpy() for i in range(4)]
            ax.boxplot(data, widths=.5, showfliers=False)
            for i, d in enumerate(data):
                ax.scatter(i + 1 + rng.uniform(-.18, .18, len(d)), d, s=5, alpha=.4, color="#e34948", linewidths=0)
                ax.text(i + 1, 1.03, len(d), ha="center", fontsize=7)
            ax.set_xticks(range(1, 5), ["min", "sli", "mod", "sev"], fontsize=7)
            ax.set_ylim(-.02, 1.1)
            rho = pd.Series(b[r].to_numpy()).corr(pd.Series(g.to_numpy()), method="spearman")
            corr.append({"finding": f, "run": r, "n_slides": len(g), "spearman_pass_vs_grade": rho})
            if ri == 0:
                ax.set_title(f"{f[:28]}\n({len(g)} slides)", fontsize=8.5)
            if ci == 0:
                ax.set_ylabel(r, fontsize=8)
            ax.text(.02, .02, f"rho {rho:.2f}", transform=ax.transAxes, fontsize=7.5)
    fig.suptitle("P3. Pass rate by grade of each finding (slides recording the finding; rows = runs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "P3_pass_rate_by_grade_per_finding.png")
    plt.close(fig)
    pd.DataFrame(corr).to_csv(out / "pass_rate_vs_grade_spearman.csv", index=False)
    print(out)
    print(pd.DataFrame(corr).pivot(index="finding", columns="run", values="spearman_pass_vs_grade").round(2).to_string())
    print(show[["finding", "topography", "n_slides", "n_exp", "median_grade"] + runs].round(2).to_string())


if __name__ == "__main__":
    main()
