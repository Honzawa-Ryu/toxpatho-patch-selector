"""scripts/unlabeled_finding_prototypes.py の結果を所見をまたいでまとめる（2026-11-05ユーザー: 肥大以外の所見でも同様のことができるか）。

  overview.csv              : 所見 × 表現ごとに、試験数、選ばれたプロトタイプ数、その割合（所見の slide / 対照群の slide）、化合物を伏せた AUC
  selected_overlap_<rep>.csv: 所見どうしで、選ばれたプロトタイプの重なり（Jaccard）。大きければ「投与による変化全般」、小さければ所見ごとに別
  F4_overview.png           : 左 化合物を伏せた AUC（対照群と、記録のない投与群と）、右 選ばれたプロトタイプの重なり
出力: outputs/0035_20261102_finer_prototypes_split_rejoin/unlabeled_findings/
usage: python scripts/unlabeled_findings_overview.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin/unlabeled_findings"
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
LABELED = {"Necrosis", "Single cell necrosis", "Cellular infiltration", "Increased mitosis", "Hematopoiesis, extramedullary"}
JP = {"Hypertrophy": "肥大", "Change, eosinophilic": "好酸性変化", "Vacuolization, cytoplasmic": "細胞質空胞化", "Proliferation, bile duct": "胆管増生",
      "Degeneration, granular, eosinophilic": "顆粒状好酸性変性", "Necrosis": "壊死", "Cellular infiltration": "細胞浸潤", "Increased mitosis": "核分裂増加",
      "Single cell necrosis": "単細胞壊死", "Hematopoiesis, extramedullary": "髄外造血"}


def main():
    rows, sel = [], {}
    for sub in sorted(p for p in D.iterdir() if p.is_dir()):
        fnd = next((k for k in JP if k.replace(",", "").replace(" ", "_") == sub.name), sub.name)
        if not (sub / "summary.csv").exists():
            continue
        sm = pd.read_csv(sub / "summary.csv")
        for _, r in sm.iterrows():
            rows.append({"finding": fnd, "finding_jp": JP.get(fnd, fnd), "bayer_label": fnd in LABELED, **r.to_dict()})
            pt = pd.read_csv(sub / f"prototypes_{r.rep}_{r.method}.csv")
            sel[(fnd, r.rep, r.method)] = set(pt.prototype[pt.selected_all_exps])
    ov = pd.DataFrame(rows)
    ov.to_csv(D / "overview.csv", index=False)
    keep = ["finding_jp", "bayer_label", "representation", "n_exps", "n_selected_all", "share_finding_slides_in_selected", "share_control_slides_in_selected",
            "auc_all_vs_control", "auc_all_vs_treated_without_record"]
    pd.set_option("display.width", 250)
    print(ov[keep].round(3).to_string(index=False))
    main_rep = ("A128_sphering_lda", "split_record")
    fnds = [f for f in JP if (f, *main_rep) in sel]
    J = pd.DataFrame(index=[JP[f] for f in fnds], columns=[JP[f] for f in fnds], dtype=float)
    for a in fnds:
        for b in fnds:
            A, B = sel[(a, *main_rep)], sel[(b, *main_rep)]
            J.loc[JP[a], JP[b]] = len(A & B) / max(len(A | B), 1)
    J.to_csv(D / f"selected_overlap_{main_rep[0]}_{main_rep[1]}.csv")
    print(J.round(2).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), layout="constrained", gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    o = ov[(ov.rep == main_rep[0]) & (ov.method == main_rep[1])].set_index("finding").reindex(fnds)
    for k, (f, r) in enumerate(o.iterrows()):
        for col, key, mk, dy in (("#D55E00", "auc_all_vs_control", "o", -0.15), ("#0072B2", "auc_all_vs_treated_without_record", "s", 0.15)):
            if np.isfinite(r.get(key, np.nan)):
                ax.errorbar(r[key], k + dy, xerr=[[r[key] - r[key + "_lo"]], [r[key + "_hi"] - r[key]]], fmt=mk, color=col, capsize=3, ms=6,
                            label={"auc_all_vs_control": "対照群と", "auc_all_vs_treated_without_record": "記録のない投与群と"}[key] if k == 0 else None)
    ax.set_yticks(range(len(o)), [f"{JP[f]}{'（Bayer にラベルあり）' if f in LABELED else ''}　{int(r.n_exps)} 化合物" for f, r in o.iterrows()])
    ax.invert_yaxis()
    ax.axvline(0.5, color="black", lw=0.6, ls=":")
    ax.set(xlim=(0.2, 1.05), xlabel="AUC（化合物を伏せた評価）", title="所見のプロトタイプは、伏せた化合物の slide で記録の有無を当てるか\n（A128、split_record。線 = 化合物を単位にした 95% ブートストラップ区間）")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax = axes[1]
    im = ax.imshow(J.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(J)), J.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(J)), J.index)
    for i in range(len(J)):
        for j in range(len(J)):
            v = J.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Jaccard 係数")
    ax.set_title("所見ごとに選ばれたプロトタイプの重なり\n（大きい = 同じプロトタイプ ＝「投与による変化全般」を拾っている可能性）")
    fig.savefig(D / "F4_overview.png", dpi=180)
    print("done")


if __name__ == "__main__":
    main()
