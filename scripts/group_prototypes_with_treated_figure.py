"""0037（まとまり 3・26 の対照群＋投与群で作り直したプロトタイプ）の図。
F1: 試験を 1 つ伏せて所見を当てる AUC（相手 = 伏せた試験の記録のない投与群）を k ごとに。所見 × まとまりの小さな図を並べる。
    線 = 伏せた試験の平均、薄い点 = 伏せた試験ごと。色 = 表現（B128 / A128）。
F2: 記録の純度: share >= 0.8 のプロトタイプに入る、記録ありの slide の patch の割合（coverage）。所見の slide が 20 枚以上の組だけ。
出力: outputs/0037_20261111_group_prototypes_with_treated/default/
usage: python scripts/group_prototypes_with_treated_figure.py   （コンテナ内、Slurm経由。CPU、1 分以内）
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/0037_20261111_group_prototypes_with_treated/default"
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
REPS = [("B128_sphering", "B128（白色化 128、ラベルなし）", "#56B4E9", "o"), ("A128_sphering_lda", "A128（B128 ‖ Bayer の LDA 13）", "#0072B2", "s")]
JP = {"Hypertrophy": "肥大", "Increased mitosis": "核分裂増加", "Ground glass appearance": "すりガラス様", "Necrosis": "壊死", "Cellular infiltration": "細胞浸潤",
      "Change, eosinophilic": "好酸性変化", "Inclusion body, intracytoplasmic": "細胞質内封入体", "Single cell necrosis": "単細胞壊死", "Fibrosis": "線維化",
      "Swelling": "腫脹", "Proliferation, bile duct": "胆管増生"}
KS = [100, 200, 500, 1000]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    lo = pd.read_csv(RUN / "loso.csv")
    base_p = RUN / "baseline_0035/loso.csv"
    bl = pd.read_csv(base_p) if base_p.exists() else None   # 0035 の全体のプロトタイプ（scripts/group_prototypes_global_baseline.py）
    S = pd.read_csv(RUN / "slides.csv")

    # F1
    pairs = lo.groupby(["group", "finding"]).exp.nunique().reset_index().sort_values(["group", "exp"], ascending=[True, False])
    ncol = 5
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.6 * nrow + 0.8), sharex=True, sharey=True, layout="constrained")
    for ax, p in zip(axes.ravel(), pairs.itertuples()):
        d = lo[(lo.group == p.group) & (lo.finding == p.finding)]
        npos = int((S[S.grp == p.group][f"grade__{p.finding}"] > 0).sum())
        for j, (rep, _, col, mk) in enumerate(REPS):
            r = d[d.rep == rep]
            x = np.log2(r.k.to_numpy()) + (j - 0.5) * 0.12
            ax.scatter(x, r.auc_own_neg, s=10, color=col, alpha=0.35, linewidths=0)
            m = r.groupby("k").auc_own_neg.mean().reindex(KS)
            ax.plot(np.log2(KS) + (j - 0.5) * 0.12, m.to_numpy(), marker=mk, color=col, lw=1.6, ms=5, mec="#333333", mew=0.4)
            if bl is not None:
                mb = bl[(bl.group == p.group) & (bl.finding == p.finding) & (bl.rep == rep)].groupby("k").auc_own_neg.mean().reindex(KS)
                ax.plot(np.log2(KS) + (j - 0.5) * 0.12, mb.to_numpy(), marker=mk, color=col, lw=1.2, ls="--", ms=5, mfc="white", mec=col, mew=1.0)
        ax.axhline(0.5, color="black", lw=0.6, ls=":")
        ax.set_title(f"まとまり {p.group}：{JP.get(p.finding, p.finding)}\n{p.exp} 試験・所見の slide {npos} 枚", fontsize=9)
        ax.set_xticks(np.log2(KS), [str(k) for k in KS])
        ax.set_ylim(0.2, 1.02)
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUC（対 記録のない投与群）")
    for ax in axes[-1]:
        ax.set_xlabel("k")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=col, marker=mk, mec="#333333", mew=0.4, label=lab) for _, lab, col, mk in REPS]
    if bl is not None:
        h += [Line2D([], [], color=col, marker=mk, ls="--", mfc="white", mec=col, label=lab.split("（")[0] + "：0035 の全体のプロトタイプ（まとまりで作り直さない）") for _, lab, col, mk in REPS]
    h.append(Line2D([], [], color="#777777", marker="o", ls="", alpha=0.4, ms=4, label="伏せた試験ごと（0037）"))
    fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("まとまりの他の試験だけで選んだプロトタイプで、伏せた試験の所見の slide を、同じ試験の記録のない投与群と判別できるか（0037）\n"
                 "実線 = 0037（まとまりで作り直す）、点線 = 0035 の全体のプロトタイプ（あれば）。伏せた試験の平均。所見の slide と記録のない投与群が 2 枚以上ある試験が 2 つ以上ある組だけ", fontsize=10)
    fig.savefig(RUN / "F1_loso_auc_by_k.png", dpi=170)
    plt.close(fig)
    if bl is not None:   # 0037 と 0035 の全体のプロトタイプの比較表（組 × 表現 × k の、伏せた試験の平均 AUC）
        key = ["group", "finding", "rep", "k"]
        cmp_ = lo.groupby(key).auc_own_neg.mean().rename("auc_0037").to_frame().join(bl.groupby(key).auc_own_neg.mean().rename("auc_0035")).reset_index()
        cmp_["diff"] = cmp_.auc_0037 - cmp_.auc_0035
        cmp_.to_csv(RUN / "compare_0035_global.csv", index=False)
        pd.set_option("display.width", 200)
        print(cmp_.groupby(["group", "finding"])[["auc_0037", "auc_0035", "diff"]].mean().round(3).to_string())
        print(cmp_.groupby(["rep", "k"])[["auc_0037", "auc_0035", "diff"]].mean().round(3).to_string())
        print("pairs (group x finding x rep x k) with 0037 > 0035:", int((cmp_["diff"] > 0).sum()), "/", int(cmp_["diff"].notna().sum()))

    # F2
    rec = pd.read_csv(RUN / "record_purity.csv")
    rec = rec[rec.n_pos_slides >= 20]
    rows = rec[["group", "finding"]].drop_duplicates().sort_values(["group", "finding"]).to_records(index=False).tolist()
    cols = [(r, k) for r, *_ in REPS for k in KS]
    M = np.array([[rec[(rec.group == g) & (rec.finding == f) & (rec.rep == r) & (rec.k == k)].coverage.iloc[0] for r, k in cols] for g, f in rows])
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 2.2), layout="constrained")
    im = ax.imshow(M, vmin=0, vmax=0.5, cmap="Blues", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.3 else "black")
    ax.set_xticks(range(len(cols)), [f"{r.split('_')[0]}\nk={k}" for r, k in cols], fontsize=8)
    ax.set_yticks(range(len(rows)), [f"まとまり {g}：{JP.get(f, f)}（{int(rec[(rec.group == g) & (rec.finding == f)].n_pos_slides.iloc[0])} 枚）" for g, f in rows])
    ax.axvline(len(KS) - 0.5, color="white", lw=3)
    fig.colorbar(im, ax=ax, label="coverage", shrink=0.8)
    ax.set_title("記録の純度: 記録ありの slide 由来が 8 割以上のプロトタイプに入る、記録ありの slide の patch の割合\n"
                 "（純度の高いプロトタイプが 1 つの試験だけでできていても数えている）", fontsize=9)
    fig.savefig(RUN / "F2_record_coverage.png", dpi=170)
    print("done", flush=True)


if __name__ == "__main__":
    main()
