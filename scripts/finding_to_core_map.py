"""所見の側から、0018の2段目のコアが明確に対応するかを見る（2026-10-24）。

slideごとに「評価用patchのうち各コアに入った割合（占有率）」を出し、所見ごと・コアごとに
陽性slide（その所見が記録された投与群slide）と陰性slideを占有率でどれだけ分けられるか（AUC）を出す。
- within: 陽性slideを含む試験の中だけで比べる（陰性 = 同じ試験の、その所見がない投与群slideとControl）。化合物の違いが入らない
- all: 全slideで比べる
あわせて、陽性slideの中で占有率と重症度の順位相関を出す。所見は自然発生（SP_FLG=true）を除き、slide単位の記録。
比較の基準として、1段目を通った割合（全コアの合計に近い）のAUCも出す。

usage: python scripts/finding_to_core_map.py [embedding] [method] [k] run1 [run2 ...]
       例: python scripts/finding_to_core_map.py pca0017 consensus 20 block6_clean_thr0.07_s2block6_clean_centered
"""
import importlib.util
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/0018_20261022_ctrl_filter_then_cores"
MIN_POS = 9
plt.rcParams.update({"font.size": 9, "savefig.bbox": "tight", "savefig.dpi": 140})
warnings.filterwarnings("ignore", message="An input array is constant")   # 占有率が全slideで0のコアなど

spec = importlib.util.spec_from_file_location("pvp", ROOT / "scripts/stage1_pass_vs_pathology.py")
pvp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pvp)


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def analyse(run, emb, col, out):
    sc = pd.read_parquet(RUNS / run / "stage1_scores.parquet")
    ass = pd.read_parquet(RUNS / run / "stage2" / emb / "assignments.parquet")
    n_eval = sc.groupby("slide_id").size()
    occ = pd.crosstab(ass.slide_id, ass[col]).drop(columns=-1, errors="ignore").reindex(n_eval.index, fill_value=0).div(n_eval, axis=0)
    occ["stage1_pass"] = sc.groupby("slide_id").kept.mean()
    m, j = pvp.pathology(set(n_eval.index))
    m = m.set_index("slide_id").reindex(occ.index)
    tr = j[~j.spontaneous]
    is_ctrl = (m.dose_level == "Control").to_numpy()
    counts = tr.groupby("FINDING_TYPE").slide_id.nunique()
    rows = []
    for f in counts[counts >= MIN_POS].index:
        g = tr[tr.FINDING_TYPE == f].groupby("slide_id").grade_ord.max()
        pos = occ.index.isin(g.index) & ~is_ctrl
        exps = m.exp_id[pos].unique()
        in_exp = m.exp_id.isin(exps).to_numpy()
        for c in occ.columns:
            v = occ[c].to_numpy()
            gp = g.reindex(occ.index[pos])
            rows.append({"finding": f, "core": c, "n_pos": int(pos.sum()), "n_exp": len(exps),
                         "auc_within": auc(v[pos], v[in_exp & ~pos]), "auc_all": auc(v[pos], v[~pos]),
                         "mean_occ_pos": float(v[pos].mean()), "mean_occ_neg_within": float(v[in_exp & ~pos].mean()),
                         "spearman_grade": float(pd.Series(v[pos]).corr(pd.Series(gp.to_numpy()), method="spearman"))})
    t = pd.DataFrame(rows)
    t.to_csv(out / f"finding_core_auc_{run}.csv", index=False)
    return t, occ


def main():
    emb, method, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    runs = sys.argv[4:]
    col = f"{method}_k{k}"
    out = RUNS / "finding_to_core"
    out.mkdir(exist_ok=True)
    best_all = []
    for run in runs:
        t, occ = analyse(run, emb, col, out)
        cores = [c for c in occ.columns if c != "stage1_pass"]
        piv = t.pivot(index="finding", columns="core", values="auc_within")[cores + ["stage1_pass"]]
        npos = t.groupby("finding").n_pos.first()
        nexp = t.groupby("finding").n_exp.first()
        piv = piv.loc[npos.sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(0.42 * piv.shape[1] + 5, 0.38 * len(piv) + 2))
        ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="auto")
        for (i, jj), v in np.ndenumerate(piv.to_numpy()):
            if np.isfinite(v) and (v >= .7 or v <= .3):
                ax.text(jj, i, f"{v:.2f}"[1:], ha="center", va="center", fontsize=6.5, color="white" if abs(v - .5) > .25 else "black")
        ax.set_xticks(range(piv.shape[1]), [str(c) for c in cores] + ["stage-1\npass"], fontsize=7.5)
        ax.set_yticks(range(len(piv)), [f"{f[:30]} ({npos[f]} sl, {nexp[f]} exp)" for f in piv.index], fontsize=8)
        ax.set_xlabel("core")
        ax.set_title(f"{run}: AUC of core occupancy, slides recording the finding vs other slides of the same experiments\n"
                     f"(red = more occupied in finding slides; numbers shown if >= .70 or <= .30)", fontsize=10)
        fig.savefig(out / f"F1_auc_within_{run}.png")
        plt.close(fig)
        b = t[t.core != "stage1_pass"].sort_values("auc_within", ascending=False).groupby("finding").head(1)
        s1 = t[t.core == "stage1_pass"].set_index("finding").auc_within
        b = b.assign(run=run, stage1_pass_auc=b.finding.map(s1))
        best_all.append(b)
    best = pd.concat(best_all)
    best.to_csv(out / "best_core_per_finding.csv", index=False)
    show = best.pivot(index="finding", columns="run", values="auc_within")
    show_core = best.pivot(index="finding", columns="run", values="core")
    show_s1 = best.pivot(index="finding", columns="run", values="stage1_pass_auc")
    order = best.groupby("finding").n_pos.first().sort_values(ascending=False).index
    print("best core per finding (AUC within experiments; core id in brackets; stage-1 pass AUC after /)")
    for f in order:
        print(f"{f[:34]:34s} n={best[best.finding == f].n_pos.iloc[0]:3d} | " +
              " | ".join(f"{show.loc[f, r]:.2f} [{show_core.loc[f, r]}] / {show_s1.loc[f, r]:.2f}" for r in runs))
    print(out)


if __name__ == "__main__":
    main()
