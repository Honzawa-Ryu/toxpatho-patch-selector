"""対照群の slide だけで「どの試験（またはスキャナ）の対照群か」を、所見の解析と同じ手順で当てられるかを調べる（2026-11-05ユーザー:
対照群だけを複数の試験・化合物で用意して、同様の条件で当てられるなら、プロトタイプに試験間差が残っている）。

0035 の割当（3,737 枚）。対照群が MIN_CTRL 枚以上ある試験の対照群 slide を、slide 単位で 5 分割する。分割ごとに学習側の slide で:
  試験 e ごとに、e の対照群の構成比の平均 / 他の試験の対照群の平均 が 2 倍以上のプロトタイプを選ぶ（scripts/unlabeled_finding_prototypes.py と同じ基準）
伏せた slide では、各試験 e のプロトタイプの割合をスコアにし、
  (a) 試験ごとの AUC（e の対照群か、他の試験の対照群か）の分布、(b) 最もスコアの高い試験を予測にしたときの正解率（偶然 = 1/試験数）
スキャナ（SS1601 / SS1598）でも同じ（2 クラス）。表現は所見の解析と同じ 3 つ。
出力: outputs/0035_20261102_finer_prototypes_split_rejoin/control_study_identification/
usage: python scripts/control_study_identification.py   （コンテナ内、Slurm経由。CPU、10 分程度）
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
R35 = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin"
OUT = R35 / "control_study_identification"
REPS = [("v1", "A128_sphering_lda", "split_record", "A128（白色化128 ‖ LDA）"), ("v2", "B128_sphering", "split_record", "B128（白色化128、ラベルなし）"),
        ("v1", "A128_sphering_lda", "flat_k1000", "A128、k=1000")]
MIN_CTRL, LR_UP, FOLDS, SEED = 6, 1.0, 5, 0
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"


def main():
    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin/config.yml"))
    S = pd.read_csv(R35 / "v1/slides.csv", dtype={"slide_id": str})
    row_b = np.load(ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"] / "cache.npz")["row"]
    nfs = int((S.set == "fatty").sum())
    row_u = np.load(ROOT / cfg["unseen_features_from"], allow_pickle=True)["row"]
    off_t, off_u = int((S.set == "bayer").sum()), int((S.set == "bayer").sum()) + nfs
    slide_of = np.r_[row_b, np.repeat(np.arange(nfs), cfg["patches_per_slide"]) + off_t, row_u + off_u]
    ns = len(S)
    w_slide = 1.0 / np.maximum(np.bincount(slide_of, minlength=ns), 1)
    q = pd.read_csv(cfg["qc_catalog"], dtype=str).set_index("slide_id")
    S["scanner"] = q.scanscope_id.reindex(S.slide_id).to_numpy()
    nct = S[S.is_ctrl].groupby("exp").size()
    exps = sorted(e for e, n in nct.items() if n >= MIN_CTRL)
    C = S[S.is_ctrl & S.exp.isin(exps)].copy()
    rng = np.random.default_rng(SEED)
    C["fold"] = rng.permutation(np.arange(len(C)) % FOLDS)
    print(f"controls: {len(C)} slides in {len(exps)} experiments ({C.COMPOUND_NAME.nunique()} compounds); scanners {C.scanner.value_counts().to_dict()}", flush=True)
    rows, per_exp = [], []
    for run, rep, method, title in REPS:
        cl = np.load(R35 / run / f"assign_{rep}_{method}.npy").astype(np.int64)
        K = int(cl.max()) + 1
        P = np.bincount(cl * ns + slide_of, weights=w_slide[slide_of], minlength=K * ns).reshape(K, ns).T
        Pc = P[C.index.to_numpy()]
        sc_ = C.scanner.fillna("?").to_numpy()
        sc_ = np.where(np.isin(sc_, ["SS1601", "SS1598"]), sc_, "?")   # SS1224（2 枚）は除く
        for label_name, lab in (("experiment", C.exp.to_numpy()), ("scanner", sc_)):
            classes = [c for c in np.unique(lab) if c != "?"]
            score = np.full((len(C), len(classes)), np.nan)
            n_sel = []
            for f in range(FOLDS):
                tr, te = (C.fold != f).to_numpy(), (C.fold == f).to_numpy()
                for j, c in enumerate(classes):
                    a_, b_ = tr & (lab == c), tr & (lab != c)
                    if a_.sum() == 0 or b_.sum() == 0:
                        continue
                    lr = np.log2((Pc[a_].mean(0) + 1e-4) / (Pc[b_].mean(0) + 1e-4))
                    sel = lr >= LR_UP
                    n_sel.append(int(sel.sum()))
                    score[te, j] = Pc[te][:, sel].sum(1)
            aucs = []
            for j, c in enumerate(classes):
                y = (lab == c).astype(int)
                ok = np.isfinite(score[:, j])
                if len(set(y[ok])) == 2:
                    a = roc_auc_score(y[ok], score[ok, j])
                    aucs.append(a)
                    per_exp.append({"representation": title, "label": label_name, "class": c, "n_slides": int(y.sum()), "auc": a})
            known = lab != "?"
            pred = np.array(classes)[np.nanargmax(np.where(np.isfinite(score), score, -np.inf), 1)]
            acc = float((pred[known] == lab[known]).mean())
            rows.append({"representation": title, "label": label_name, "n_classes": len(classes), "chance_accuracy": float(pd.Series(lab[lab != "?"]).value_counts(normalize=True).max()) if label_name == "scanner" else 1 / len(classes),
                         "top1_accuracy": acc, "auc_median": float(np.median(aucs)), "auc_q25": float(np.percentile(aucs, 25)), "auc_q75": float(np.percentile(aucs, 75)),
                         "share_auc_ge_0.8": float(np.mean(np.array(aucs) >= 0.8)), "median_n_selected": float(np.median(n_sel))})
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "summary.csv", index=False)
    pe = pd.DataFrame(per_exp)
    pe.to_csv(OUT / "auc_per_class.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    cols = {"A128（白色化128 ‖ LDA）": "#D55E00", "B128（白色化128、ラベルなし）": "#0072B2", "A128、k=1000": "#999999"}
    t_ = pe[pe.label == "experiment"]
    for k, (title, g) in enumerate(t_.groupby("representation", sort=False)):
        ax.scatter(g.auc, k + np.random.default_rng(k).uniform(-0.15, 0.15, len(g)), s=14, color=cols[title], alpha=0.7)
        ax.plot([g.auc.median()] * 2, [k - 0.3, k + 0.3], color="black", lw=1.5)
        r = res[(res.representation == title) & (res.label == "experiment")].iloc[0]
        ax.text(1.01, k, f"正解率 {r.top1_accuracy:.0%}（偶然 {r.chance_accuracy:.0%}）", va="center", fontsize=8)
    ax.set_yticks(range(t_.representation.nunique()), list(dict.fromkeys(t_.representation)))
    ax.invert_yaxis()
    ax.axvline(0.5, color="black", lw=0.6, ls=":")
    ax.set(xlim=(0.3, 1.0), xlabel="AUC（伏せた対照群の slide が、その試験の対照群か他の試験の対照群か）。点 = 試験、黒線 = 中央値",
           title=f"対照群だけで、どの試験の slide かを当てられるか（{len(exps)} 試験・{len(C)} 枚、slide 単位の 5 分割）\n当たるほど、プロトタイプに試験間差が残っている")
    fig.savefig(OUT / "F1_control_study_identification.png", dpi=180)
    print("done", flush=True)


if __name__ == "__main__":
    main()
