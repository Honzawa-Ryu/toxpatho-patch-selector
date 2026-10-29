"""usage: python scripts/external_severity.py  [版]  （コンテナ内、Slurm経由。版の既定は上記）"""
# 0029 v5: 色の面積の割合と、外部の記録の重症度（試験内の順序尺度）の対応。A〜D、特にCとD
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
P = Path("/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector")
R = P / "outputs/0029_20261028_external_finding_maps" / (sys.argv[1] if len(sys.argv) > 1 else "v8"); O = R / "severity"; O.mkdir(exist_ok=True)
sh = pd.read_csv(R / "slide_group_shares.csv", dtype={"slide_id": str})
sl = pd.read_csv(P / "outputs/0022_20261025_external_pharma_embeddings/default/slides.csv", dtype=str)
fr = pd.read_csv("/workspace/andre01/honzawa/02-playground/tggate-bench/outputs/external_data_audit/findings_raw.csv", encoding="utf-8-sig", dtype=str)
fr["g"] = pd.to_numeric(fr.grade_ordinal_within_source, errors="coerce").fillna(0)
def grade(mask):
    gm = fr[mask].groupby(["study", "label_unit"]).g.max()
    return np.array([gm.get((s, u), 0.0) for s, u in zip(sl.study, sl.label_unit)])
sl["necrosis_grade"] = grade(fr.finding_raw.str.contains("ecrosis") & ~fr.finding_raw.str.contains("bileduct|Capsule"))
sl["inflammation_grade"] = grade(fr.finding_raw.str.contains("infiltration"))
d = sh.merge(sl[["slide_id", "provider", "necrosis_grade", "inflammation_grade"]], on="slide_id")
_RUN = sys.argv[1] if len(sys.argv) > 1 else "v8"
import yaml
_C = yaml.safe_load(open(P / "experiments/0029_20261028_external_finding_maps/config.yml"))
MAPS = list({**_C, **_C.get("variants", {}).get(_RUN, {})}["maps"])
rng = np.random.default_rng(0)
rows = []
for g in ["necrosis", "inflammation"]:
    gr = d[f"{g}_grade"].to_numpy()
    for m in MAPS:
        v = d[f"{m}__{g}"].to_numpy()
        # 試験ごとに対照群の平均を引いた差
        dv = v - d.groupby("study")[f"{m}__{g}"].transform(lambda x: x[d.loc[x.index, "is_ctrl"]].mean()).to_numpy()
        within = [spearmanr(x[f"{g}_grade"], x[f"{m}__{g}"])[0] for _, x in d.groupby("study") if x[f"{g}_grade"].nunique() > 1]
        boot = []
        for _ in range(2000):
            i = rng.integers(0, len(d), len(d))
            boot.append(spearmanr(gr[i], v[i])[0])
        rows.append({"finding": g, "map": m, "rho_pooled": spearmanr(gr, v)[0], "rho_lo": np.nanpercentile(boot, 2.5), "rho_hi": np.nanpercentile(boot, 97.5),
                     "rho_delta_vs_ctrl": spearmanr(gr, dv)[0], "rho_within_study_mean": float(np.mean(within)), "n_studies_within": len(within),
                     "median_share_by_grade": {int(k): round(float(np.median(v[gr == k])) * 100, 2) for k in np.unique(gr)}})
    # CとDの差（同じslideの組で bootstrap）
    vc, vd = d[f"C_lda__{g}"].to_numpy(), d[f"D_unsup__{g}"].to_numpy()
    diff = [spearmanr(gr[i], vc[i])[0] - spearmanr(gr[i], vd[i])[0] for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))]
    rows.append({"finding": g, "map": "C minus D", "rho_pooled": spearmanr(gr, vc)[0] - spearmanr(gr, vd)[0], "rho_lo": np.nanpercentile(diff, 2.5), "rho_hi": np.nanpercentile(diff, 97.5)})
o = pd.DataFrame(rows); o.to_csv(O / "severity_spearman.csv", index=False)
pd.set_option("display.width", 220); print(o.round(3).to_string(index=False))
print(d.groupby(["necrosis_grade"]).size().to_dict(), d.groupby(["inflammation_grade"]).size().to_dict())
# 図
fig, axes = plt.subplots(2, len(MAPS), figsize=(3.3 * len(MAPS), 6.4), sharey="row", squeeze=False)
col = {"shionogi": "#0072B2", "eisai": "#D55E00"}
title = {"D_unsup": "D: unsup (no labels, no norm.)", "C_lda": "C: LDA only", "B_sphering": "B: Sphering", "A_sphering_lda": "A: Sphering + LDA"}
for r_, g in enumerate(["necrosis", "inflammation"]):
    for c_, m in enumerate(MAPS):
        ax = axes[r_, c_]; gr = d[f"{g}_grade"]
        for p_, cc in col.items():
            mm = d.provider == p_
            ax.scatter(gr[mm] + rng.uniform(-.12, .12, mm.sum()) + (-.06 if p_ == "shionogi" else .06), d.loc[mm, f"{m}__{g}"] * 100, s=14, color=cc, alpha=.8, label=p_.capitalize() if (r_, c_) == (0, 0) else None)
        rr = o[(o.finding == g) & (o["map"] == m)].iloc[0]
        ax.set_title(f"{title.get(m, m)}\nrho {rr.rho_pooled:.2f} [{rr.rho_lo:.2f}, {rr.rho_hi:.2f}]", fontsize=8.5)
        ax.set_xticks(sorted(gr.unique()))
        if r_ == 1: ax.set_xlabel("recorded grade (within-company ordinal; 0 = not recorded)", fontsize=8)
        if c_ == 0: ax.set_ylabel(f"% patches colored as {g}", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle("External slides (78): colored area vs recorded severity. Spearman rho, 95% bootstrap CI over slides", fontsize=10)
fig.tight_layout(); fig.savefig(O / "S1_severity.png", dpi=150)
