"""セミナー用: 学習に使っていない TG-GATEs の WSI（0030 v4、3,319 枚）で、Bayer のラベルで名前を付けたプロトタイプが所見を当てるか（2026-11-10 ユーザー）。
表現の呼び名は「LDA なし（slide 内の順位変換 → PCA64）」「LDA あり（＋ LDA 13 次元）」。C・D とは書かない（ユーザー指定）。

U1: 学習なしの AUC（Bayer のラベルの名前が付いたプロトタイプの割合が、同じ試験の対照群の平均よりどれだけ増えたか）。ラベルのある 5 所見。
    塗りの印 = 全 3,319 枚（線 = 化合物を復元抽出した 95% 区間、0030 v4 bootstrap/auc_ci.csv）
    白抜きの印 = Bayer に使っていない試験の slide だけ（2,522 枚、0030 v4 strict/named_auc_by_stratum.csv の new_exp。区間なし）
U2: 例。fenofibrate 試験（Bayer にない試験）の核分裂増加の記録がある slide と、同じ試験の対照群。全 patch のうち
    「核分裂像（mitosis）」の名前が付いたプロトタイプの patch だけに色。下に、投与群の slide でその patch の例を並べる
出力: outputs/0030_20261028_tggates_unseen_findings/v4/seminar/
usage: PYTHONPATH=.:scripts python scripts/seminar_unseen_findings_figure.py   （コンテナ内、Slurm経由。CPU、10 分程度）
"""
from pathlib import Path

import numpy as np
import pandas as pd

from seminar_prototype_mixing_figure import FONT_JP, FULL_DIR, ROOT, WSI_DIRS, load_0030_models, nearest

RUN = ROOT / "outputs/0030_20261028_tggates_unseen_findings/v4"
OUT = RUN / "seminar"
NAMES = ROOT / "outputs/0029_20261028_external_finding_maps/v9"
REPS = [("D_unsup", "LDA なし（slide 内の順位変換 → PCA64）", "#999999", "o"), ("C_lda", "LDA あり（＋ Bayer のラベルで LDA、13 次元）", "#D55E00", "s")]
FIND = [("Necrosis", "壊死"), ("Single cell necrosis", "単細胞壊死"), ("Cellular infiltration", "細胞浸潤"), ("Increased mitosis", "核分裂増加"), ("Hematopoiesis, extramedullary", "髄外造血")]
EXAMPLE = {"treated": "35017", "control": "34910", "label": "mitosis", "label_jp": "核分裂像"}
N_TILES, SEED = 6, 0


def main():
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import openslide
    from matplotlib import font_manager
    OUT.mkdir(exist_ok=True)
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

    # U1
    ci = pd.read_csv(RUN / "bootstrap/auc_ci.csv")
    ci = ci[ci.kind == "named"].set_index(["finding", "map"])
    st = pd.read_csv(RUN / "strict/named_auc_by_stratum.csv").set_index(["finding", "stratum"])
    fc = pd.read_csv(RUN / "finding_classification.csv").set_index(["finding", "map"])
    fig, ax = plt.subplots(figsize=(9, 5.6), layout="constrained")
    for i, (f, jp) in enumerate(FIND):
        for j, (r, lab, col, mk) in enumerate(REPS):
            y = i + (j - 0.5) * 0.3
            if (f, r) in ci.index and np.isfinite(ci.loc[(f, r), "AUC"]):
                v = ci.loc[(f, r)]
                ax.errorbar(v.AUC, y, xerr=[[v.AUC - v.lo], [v.hi - v.AUC]], fmt=mk, color=col, ms=8, capsize=3, mec="#333333", mew=0.5)
                ne = st.loc[(f, "new_exp"), r] if (f, "new_exp") in st.index else np.nan
                if np.isfinite(ne):
                    ax.plot(ne, y, mk, ms=8, mfc="white", mec=col, mew=1.6)
                ax.text(1.005, y, f"{v.AUC:.2f}", va="center", fontsize=9, color=col if r == "C_lda" else "#555555")
            else:
                ax.text(0.52, y, "「" + jp + "」の名前が付いたプロトタイプなし", va="center", fontsize=8, color="#777777")
        n = fc.loc[(f, "C_lda")]
        ax.annotate(f"{int(n.n_pos)} 枚・{int(n.n_pos_compounds)} 化合物", (0, i), xycoords=("axes fraction", "data"), xytext=(-8, -13), textcoords="offset points", ha="right", fontsize=8, color="#555555")
    ax.axvline(0.5, color="black", lw=0.6, ls=":")
    ax.set_yticks(range(len(FIND)), [jp for _, jp in FIND], fontsize=11)
    ax.invert_yaxis()
    ax.set(xlim=(0.2, 1.06), xlabel="AUC（その所見の記録がある slide か、ない slide か）")
    ax.set_title("学習に使っていない TG-GATEs の slide 3,319 枚（43 試験）で、所見を当てる。TG-GATEs の記録での学習はなし\n"
                 "名前付きプロトタイプの割合が、同じ試験の対照群よりどれだけ増えたか", fontsize=10)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=mk, ls="", color=col, mec="#333333", ms=8, label=lab) for _, lab, col, mk in REPS]
    h += [Line2D([], [], marker="o", ls="", mfc="#333333", mec="#333333", ms=7, label="全 slide（線 = 化合物を復元抽出した 95% 区間）"),
          Line2D([], [], marker="o", ls="", mfc="white", mec="#333333", mew=1.6, ms=7, label="Bayer に使っていない試験の slide だけ（2,522 枚）")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9, frameon=False)
    fig.savefig(OUT / "U1_named_auc.png", dpi=200)
    plt.close(fig)
    print("U1 done", flush=True)

    # U2
    pre, tf, L, ex23, cen = load_0030_models()
    sl = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str, "EXP_ID": str}).set_index("slide_id")
    named = {r: set(pd.read_csv(NAMES / f"prototype_names_{r}.csv").query("name == @EXAMPLE['label']").prototype) for r, *_ in REPS}
    data = {}
    for k_ in ("treated", "control"):
        sid = EXAMPLE[k_]
        with h5py.File(ROOT / FULL_DIR / f"{sid}.h5", "r") as h:
            f_, c_ = h["features"][:].astype(np.float32), h["coords"][:]
            ps = int(h["coords"].attrs.get("patch_size_level0", 256))
        Kf = pre.transform(ex23.slide_rank(f_)).astype(np.float32)
        lab = {"D_unsup": nearest(Kf[:, :64], cen["D_unsup"]), "C_lda": nearest(tf(Kf)[:, :L], cen["C_lda"])}
        svs = [p_ for d in WSI_DIRS for p_ in d.glob(f"*/{sid}.svs")][0]
        o = openslide.OpenSlide(str(svs))
        th = o.get_thumbnail((1400, 1400)).convert("L")
        data[k_] = dict(c=c_, ps=ps, lab=lab, svs=svs, thumb=np.asarray(th), sc=th.size[0] / o.dimensions[0])
    fig = plt.figure(figsize=(12, 11.5), layout="constrained")
    sf = fig.subfigures(2, 1, height_ratios=[2.3, 1])
    axes = sf[0].subplots(len(REPS), 2)
    rng = np.random.default_rng(SEED)
    for i, (r, lab, col, _) in enumerate(REPS):
        for j, k_ in enumerate(("treated", "control")):
            d, ax = data[k_], axes[i, j]
            m = np.isin(d["lab"][r], list(named[r]))
            ax.imshow(d["thumb"], cmap="gray", vmin=0, vmax=255, alpha=0.3)
            g = (d["c"] // d["ps"]).astype(int)
            H = np.zeros((g[:, 1].max() + 1, g[:, 0].max() + 1, 4))
            H[g[:, 1], g[:, 0]] = matplotlib.colors.to_rgba("#E0E0E0")
            H[g[m, 1], g[m, 0]] = matplotlib.colors.to_rgba(col if r == "C_lda" else "#0072B2")
            ax.imshow(H, extent=(0, H.shape[1] * d["ps"] * d["sc"], H.shape[0] * d["ps"] * d["sc"], 0), interpolation="nearest")
            ax.set(xticks=[], yticks=[], xlim=(0, d["thumb"].shape[1]), ylim=(d["thumb"].shape[0], 0))
            for s in ax.spines.values():
                s.set_visible(False)
            row = sl.loc[EXAMPLE[k_]]
            ax.set_title(f"{'投与群' if k_ == 'treated' else '同じ試験の対照群'}: {row.COMPOUND_NAME}（{row.DOSE_LEVEL}、{row.SACRIFICE_PERIOD}）、slide {EXAMPLE[k_]}\n"
                         f"「{EXAMPLE['label_jp']}」のプロトタイプの patch: {m.mean():.1%}（{len(named[r])} 個のプロトタイプ）", fontsize=9)
            if j == 0:
                ax.set_ylabel(lab.replace("（", "\n（"), fontsize=10)
    sf[0].suptitle(f"Bayer に使っていない fenofibrate 試験。投与群の slide には核分裂増加の記録あり（程度 1）。色 = 「{EXAMPLE['label_jp']}」の名前が付いたプロトタイプの patch（全 patch）", fontsize=10)
    tax = sf[1].subplots(len(REPS), N_TILES)
    d = data["treated"]
    o = openslide.OpenSlide(str(d["svs"]))
    for i, (r, lab, col, _) in enumerate(REPS):
        idx = np.where(np.isin(d["lab"][r], list(named[r])))[0]
        pick = rng.choice(idx, min(N_TILES, len(idx)), replace=False) if len(idx) else []
        for j in range(N_TILES):
            ax = tax[i, j]
            ax.set(xticks=[], yticks=[])
            for s in ax.spines.values():
                s.set_color(col if r == "C_lda" else "#0072B2")
                s.set_linewidth(2)
                s.set_visible(True)
            if j < len(pick):
                x, y = d["c"][pick[j]]
                ax.imshow(np.asarray(o.read_region((int(x), int(y)), 0, (d["ps"], d["ps"])).convert("RGB").resize((256, 256))))
            if j == 0:
                ax.set_ylabel(lab.split("（")[0], fontsize=10)
    sf[1].suptitle(f"投与群の slide で「{EXAMPLE['label_jp']}」のプロトタイプに入った patch の例（無作為に {N_TILES} 個、各 256 px・20 倍）", fontsize=10)
    fig.savefig(OUT / "U2_example_mitosis_fenofibrate.png", dpi=170)
    print("U2 done", {r: len(named[r]) for r in named}, flush=True)


if __name__ == "__main__":
    main()
