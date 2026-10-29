"""0030（学習に使っていないTG-GATEsのWSI）で、名前付きプロトタイプの面積と記録の程度（minimal=1 … severe=4、記録なし=0）の対応。

外部データの重症度解析（scripts/external_severity.py）と同じ形。TG-GATEsの記録は正解としてだけ使う（何も学習しない）。
スコア = Bayerのラベルの名前が付いたプロトタイプの割合 − 同じ試験の対照群の平均。A〜D。
区間は化合物を復元抽出した2,000回のbootstrap。
出力: outputs/0030_20261028_tggates_unseen_findings/v1/severity/
usage: python scripts/tggates_unseen_severity.py  [版]  （コンテナ内、Slurm経由。版の既定は上記）
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
_RUN = sys.argv[1] if len(sys.argv) > 1 else "v4"
_CFG = yaml.safe_load(open(ROOT / "experiments/0030_20261028_tggates_unseen_findings/config.yml"))
_CFG = {**_CFG, **_CFG.get("variants", {}).get(_RUN, {})}
NAMES = _CFG["names_from"]
MAPS = list(_CFG["maps"])
RUN = ROOT / "outputs/0030_20261028_tggates_unseen_findings" / _RUN
OUT = RUN / "severity"
TITLE = {"D_unsup": "D: unsup", "C_lda": "C: LDA only", "B_sphering": "B: Sphering", "A_sphering_lda": "A: Sphering + LDA"}
N_BOOT = 2000


def main():
    OUT.mkdir(exist_ok=True)
    cfg = _CFG
    findings = {f: b for f, b in cfg["findings"].items() if b}
    sl = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str})
    ctrl = sl.DOSE_LEVEL.eq("Control").to_numpy()
    comp = sl.COMPOUND_NAME.to_numpy()
    score = {}
    for m in MAPS:
        sh = np.load(RUN / f"shares_{m}.npy")
        base = pd.DataFrame(sh[ctrl]).groupby(sl.EXP_ID.to_numpy()[ctrl]).mean().reindex(sl.EXP_ID).to_numpy()
        delta = sh - np.where(np.isnan(base), sh[ctrl].mean(0), base)
        names = pd.read_csv(ROOT / "outputs/0029_20261028_external_finding_maps" / NAMES / f"prototype_names_{m}.csv").set_index("prototype").name.fillna("").reindex(range(sh.shape[1])).to_numpy()
        for f, b in findings.items():
            if (names == b).any():
                score[(f, m)] = delta[:, names == b].sum(1) * 100   # % of patches
    comps = np.unique(comp)
    idx = {c: np.where(comp == c)[0] for c in comps}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[c] for c in rng.choice(comps, len(comps))]) for _ in range(N_BOOT)]

    def rho(g, s, ii=None):
        ii = np.arange(len(g)) if ii is None else ii
        return spearmanr(g[ii], s[ii])[0] if len(np.unique(g[ii])) > 1 and s[ii].std() > 0 else np.nan

    rows, diffs = [], []
    for f in findings:
        g = sl[f"grade__{f}"].to_numpy()
        for m in MAPS:
            if (f, m) not in score:
                rows.append({"finding": f, "map": m, "rho": np.nan, "lo": np.nan, "hi": np.nan, "n_named_protos": 0})
                continue
            s = score[(f, m)]
            b = np.array([rho(g, s, ii) for ii in boots])
            rows.append({"finding": f, "map": m, "rho": rho(g, s), "lo": np.nanpercentile(b, 2.5), "hi": np.nanpercentile(b, 97.5),
                         "rho_recorded_only": rho(g[g > 0], s[g > 0]), "n_recorded": int((g > 0).sum()), "n_grades": int(len(np.unique(g[g > 0]))),
                         "median_by_grade": {int(k): round(float(np.median(s[g == k])), 3) for k in np.unique(g)}})
        for a, bb in [("C_lda", "D_unsup"), ("C_lda", "D13_unsup13"), ("A_sphering_lda", "B_sphering")]:
            if a not in MAPS or bb not in MAPS or (f, a) not in score:
                continue
            sb = score.get((f, bb), np.zeros(len(sl)))   # 名前のないプロトタイプ: 相関0扱い
            d = np.array([rho(g, score[(f, a)], ii) - (rho(g, sb, ii) if sb.std() > 0 else 0.0) for ii in boots])
            diffs.append({"finding": f, "comparison": f"{a} - {bb}", "diff": rho(g, score[(f, a)]) - (rho(g, sb) if sb.std() > 0 else 0.0),
                          "lo": np.nanpercentile(d, 2.5), "hi": np.nanpercentile(d, 97.5), "p_le0": float(np.nanmean(d <= 0))})
    res, diffs = pd.DataFrame(rows), pd.DataFrame(diffs)
    res.to_csv(OUT / "severity_spearman.csv", index=False)
    diffs.to_csv(OUT / "severity_differences.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.drop(columns="median_by_grade").round(3).to_string(index=False))
    print(res[["finding", "map", "median_by_grade"]].to_string(index=False))
    print(diffs.round(3).to_string(index=False))

    # 図: 所見 × 表現。横軸は記録の程度、縦軸は名前付きプロトタイプの割合の増分（%）。灰は対照群
    F = list(findings)
    fig, axes = plt.subplots(len(F), len(MAPS), figsize=(3.3 * len(MAPS), 2.7 * len(F)), sharey="row", squeeze=False)
    for r_, f in enumerate(F):
        g = sl[f"grade__{f}"].to_numpy()
        for c_, m in enumerate(MAPS):
            ax = axes[r_, c_]
            if (f, m) in score:
                s = score[(f, m)]
                jit = rng.uniform(-.18, .18, len(g))
                ax.scatter(g[ctrl] + jit[ctrl], s[ctrl], s=5, color="#bbbbbb", alpha=.5, label="control" if (r_, c_) == (0, 0) else None)
                tr = ~ctrl
                ax.scatter(g[tr] + jit[tr], s[tr], s=7, color="#0072B2", alpha=.6, label="treated" if (r_, c_) == (0, 0) else None)
                rr = res[(res.finding == f) & (res["map"] == m)].iloc[0]
                ax.set_title(f"{TITLE.get(m, m)}\nrho {rr.rho:.2f} [{rr.lo:.2f}, {rr.hi:.2f}]", fontsize=8.5)
            else:
                ax.text(.5, .5, "no prototype\nnamed " + findings[f], ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#888888")
                ax.set_title(TITLE.get(m, m), fontsize=8.5)
            ax.set_xticks(sorted(np.unique(g)))
            ax.axhline(0, color="#dddddd", lw=.8)
            ax.spines[["top", "right"]].set_visible(False)
            if c_ == 0:
                ax.set_ylabel(f"{f}\n(% patches, minus study-control mean)", fontsize=8)
            if r_ == len(F) - 1:
                ax.set_xlabel("recorded grade (0 = not recorded; 1 minimal … 4 severe)", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("TG-GATEs slides unseen in training (3,319): named-prototype area vs recorded grade. No TG-GATEs training; Spearman rho, 95% CI over compounds", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "S2_severity_tggates.png", dpi=150)


if __name__ == "__main__":
    main()
