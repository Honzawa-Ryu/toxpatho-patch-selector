"""0024の続き: 所見ごとに「異常の強さ全般」を差し引いても残るプロトタイプがあるかを見る（2026-10-27ユーザー了承）。

0024では、どの所見でも同じプロトタイプ（comboの4・25など）が上位に来た。所見に特有か、異常全般で増えるだけかを分ける。
- 異常の強さ全般（burden）: 所見表で、そのslideの動物に記録された全所見の程度の合計（自然発生を除く）から、注目する所見の程度を引いたもの。
  プロトタイプとは独立に作るので、循環しない。
- 部分順位相関: 注目する所見の程度と、プロトタイプの割合の「同じ試験のControlとの差」の順位相関を、burdenの順位で線形に差し引いた残差どうしで計算。
  対象のslideは0024と同じ（その所見が出た試験のslide、Controlが2枚以上の試験のみ）。
- あわせて、汎用のプロトタイプ（各版で多くの所見の1位になったもの）の画像を、所見ごとに並べる（見た目が所見で違うかを目で確かめる）。

入力: outputs/0024_20261026_prototypes_vs_unlabeled_findings/v1/（slides.csv, slide_findings.csv, slide_shares_<版>.csv, assignments.npz）
出力: 同じフォルダの specificity/

usage: python scripts/finding_specificity.py [v1|v2]
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/0024_20261026_prototypes_vs_unlabeled_findings" / (sys.argv[1] if len(sys.argv) > 1 else "v1")   # 0024の版（v2は脂肪変性の試験を追加）
OUT = RUN / "specificity"
REPS = ["unsup", "weak", "combo"]
GRADE = {"minimal": 1, "slight": 2, "moderate": 3, "severe": 4, "P": 1}
SHEET_FINDINGS = ["Degeneration, fatty", "Hypertrophy", "Vacuolization, cytoplasmic", "Proliferation, bile duct", "Degeneration, granular, eosinophilic",
                  "Change, eosinophilic", "Necrosis", "Increased mitosis", "Cellular infiltration"]
TILES = 8


def partial_spearman(x, y, z):
    """zの順位を線形に差し引いた、xとyの順位の残差どうしの相関。"""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    d = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / d) if d > 0 else np.nan


def main():
    OUT.mkdir(exist_ok=True)
    slides = pd.read_csv(RUN / "slides.csv", dtype=str)
    slides["exp"] = slides.EXP_ID.astype(int)
    is_ctrl = (slides.dose == "Control").to_numpy()
    G = pd.read_csv(RUN / "slide_findings.csv", index_col=0)
    G.index = G.index.astype(str)
    G = G.reindex(slides.slide_id)
    # 全所見の程度の合計（自然発生を除く、0024の対象所見に限らない）
    p = pd.read_csv(ROOT / "data/corrected/open_tggates_pathology.csv", dtype=str)
    p = p[(p.ORGAN == "Liver") & (p.SP_FLG.str.lower() != "true")]
    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
    sk = slides[["slide_id"] + key].copy()
    for c in key:
        sk[c] = pd.to_numeric(sk[c], errors="coerce")
        p[c] = pd.to_numeric(p[c], errors="coerce")
    t = sk.merge(p, on=key)
    t["g"] = t.GRADE_TYPE.map(GRADE).fillna(1)
    total = t.groupby(["slide_id", "FINDING_TYPE"]).g.max().groupby("slide_id").sum().reindex(slides.slide_id).fillna(0).to_numpy()
    exp = slides.exp.to_numpy()
    rows, generic = [], {}
    for rep in REPS:
        sh = pd.read_csv(RUN / f"slide_shares_{rep}.csv", index_col=0)
        sh.index = sh.index.astype(str)
        sh = sh.reindex(slides.slide_id).to_numpy()
        d = np.full_like(sh, np.nan)
        for e in np.unique(exp):
            idx = np.flatnonzero(exp == e)
            c = idx[is_ctrl[idx]]
            if len(c) >= 2:
                d[idx] = sh[idx] - sh[c].mean(0)
        a = pd.read_csv(RUN / f"assoc_{rep}.csv")
        top1 = a.sort_values("rho", ascending=False).drop_duplicates("finding").prototype.value_counts()
        generic[rep] = [int(x) for x in top1.index[:2]]
        for f in G.columns:
            g = G[f].to_numpy()
            exps = np.unique(exp[(g > 0) & ~np.isnan(d[:, 0])])
            m = np.isin(exp, exps) & ~np.isnan(d[:, 0])
            burden = total - g
            for k in range(sh.shape[1]):
                if m.sum() < 8 or len(np.unique(g[m])) < 2:
                    continue
                pr = partial_spearman(g[m], d[m, k], burden[m])
                raw = partial_spearman(g[m], d[m, k], np.zeros(m.sum()))
                ok = [float(np.mean(d[(exp == e) & (g > 0), k]) > 0) for e in exps]
                rows.append({"rep": rep, "finding": f, "prototype": k, "rho_raw": raw, "rho_partial": pr, "n_slides": int(m.sum()),
                             "n_exps": len(exps), "exps_up": int(sum(ok)), "generic": k in generic[rep]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "specificity.csv", index=False)
    best = df.sort_values("rho_partial", ascending=False).drop_duplicates(["rep", "finding"])
    best_raw = df.sort_values("rho_raw", ascending=False).drop_duplicates(["rep", "finding"]).set_index(["rep", "finding"])
    best = best.join(best_raw[["prototype", "rho_raw"]].rename(columns={"prototype": "raw_top_prototype", "rho_raw": "raw_top_rho"}), on=["rep", "finding"])
    best.to_csv(OUT / "best_partial.csv", index=False)
    print(best[best.rep == "combo"][["finding", "prototype", "rho_partial", "rho_raw", "exps_up", "n_exps", "raw_top_prototype", "raw_top_rho"]].round(3).to_string(index=False))
    print("generic prototypes:", generic)

    # 図: 所見ごとに、生の相関の1位（多くは汎用）と、burdenを差し引いた後の1位
    fs = list(G.columns)
    fig, axes = plt.subplots(1, len(REPS), figsize=(5.2 * len(REPS), 0.38 * len(fs) + 1.8), sharey=True)
    for ax, rep in zip(axes, REPS):
        b = best[best.rep == rep].set_index("finding").reindex(fs)
        y = np.arange(len(fs))
        ax.scatter(b.raw_top_rho, y, marker="o", color="#999999", label="top prototype, raw")
        ax.scatter(b.rho_partial, y, marker="D", color="#D55E00", label="top prototype after removing burden")
        for yi, (_, r) in zip(y, b.iterrows()):
            if np.isfinite(r.rho_partial):
                ax.text(r.rho_partial + 0.01, yi, f"p{int(r.prototype)}", fontsize=6.5, va="center")
        ax.axvline(0, color="0.7", lw=0.7)
        ax.set_title(f"{rep} (generic: {', '.join('p' + str(x) for x in generic[rep])})", fontsize=9)
        ax.set_xlabel("Spearman with grade (within-experiment share difference)")
    axes[0].set_yticks(np.arange(len(fs)), fs, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("SP1. Does a finding-specific prototype remain after removing overall lesion burden (sum of all recorded grades except this finding)?", fontsize=10)
    fig.savefig(OUT / "SP1_partial_vs_raw.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 画像: comboの汎用プロトタイプを、所見ごとに（その所見の程度が高いslideから、1枚ずつ）
    spec = importlib.util.spec_from_file_location("exp0017", ROOT / "experiments/0017_20261016_block_consensus_patterns/experiment.py")
    ex17 = importlib.util.module_from_spec(spec)
    sys.modules["exp0017"] = ex17
    spec.loader.exec_module(ex17)
    z = np.load(RUN / "assignments.npz")
    row, xy, size0 = z["row"], z["xy"], dict(zip(slides.slide_id, z["size0"]))
    rng = np.random.default_rng(0)
    for rep in ["combo", "unsup"]:
        A = z[rep]
        for gp in generic[rep]:
            sheet = []
            for f in SHEET_FINDINGS:
                g = G[f].to_numpy()
                order = np.flatnonzero(g > 0)
                order = order[np.argsort(-g[order], kind="stable")]
                picks = []
                for si in order:
                    cand = np.flatnonzero((row == si) & (A == gp))
                    if len(cand):
                        picks.append(int(rng.choice(cand)))
                    if len(picks) == TILES:
                        break
                if picks:
                    tiles = ex17.read_tiles(picks, row, xy, slides, size0,
                                            lambda s_, q_: (f"e{s_.exp} {str(s_.compound)[:13]} {str(s_.dose)[0]}", f"{s_.period} grade {int(G[f].iloc[row[q_]])}"))
                    sheet.append((f"{f}: {rep} prototype {gp} in slides with this finding (highest grade first)", tiles))
            ex17.draw_sheet(f"SP2. {rep} generic prototype {gp}, by finding", sheet, OUT / f"SP2_{rep}_p{gp}.jpg")
    print("done")


if __name__ == "__main__":
    main()
