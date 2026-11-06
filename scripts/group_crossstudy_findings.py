"""対照群で見分けられない試験のまとまり（scripts/control_similarity_groups.py、偶然の 95% 点で切った B128 のまとまり）の中で、
試験をまたいで所見のプロトタイプを選び、評価する（2026-11-06 ユーザー: まとまりを「試験をまたいで比べてよい単位」として使う。A = 今ある特徴量で肥大から）。

対象: 肥大が 2 枚以上記録された試験が 2 つ以上あり、投与群の特徴量（0035 の割当）があるまとまり 25・21・9。
1 試験を伏せる（leave-one-experiment-out）:
  選び方（残りの肥大の試験だけで）: 試験ごとに log2 比 所見/まとまりの対照群（伏せた試験の対照群は除く）、所見/同じ試験の記録のない投与群、記録のない投与群/まとまりの対照群
    → 前の 2 つが >= 1 の試験が min_exps 以上、2 つ目の平均 >= 1、3 つ目（投与全般）が >= 1 の試験が min_exps 未満
  評価（伏せた試験の肥大の slide を、選んだプロトタイプの割合で、次の相手と判別した AUC）:
    自分の対照群 / まとまりの他の試験の対照群 / 他のまとまりの対照群（同じスキャナ・違うスキャナ）/ 自分の記録のない投与群 / まとまりの肥大のない試験の投与群
  まとまりが「入れ替えてよい単位」なら、自分の対照群と、まとまりの他の試験の対照群とで AUC がそろう。
対照群は時点をそろえない（他の試験の対照群と同じ条件にするため）。
出力: outputs/0035_20261102_finer_prototypes_split_rejoin/group_crossstudy/
usage: python scripts/group_crossstudy_findings.py   （コンテナ内、Slurm経由。CPU、数分）
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
R35 = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin"
OUT = R35 / "group_crossstudy"
REPS = [("v2", "B128_sphering", "B128（ラベルなし）"), ("v1", "A128_sphering_lda", "A128（白色化128 ‖ LDA）")]
GROUPS, FINDING, MIN_N, LR_UP = [25, 21, 9], "Hypertrophy", 2, 1.0
REFS = [("own_ctrl", "自分の対照群"), ("group_ctrl", "まとまりの他の試験の対照群"), ("other_ctrl_same_scanner", "他のまとまりの対照群（同じスキャナ）"),
        ("other_ctrl_diff_scanner", "他のまとまりの対照群（違うスキャナ）"), ("own_neg", "自分の記録のない投与群"), ("group_nonhyp_treated", "まとまりの肥大のない試験の投与群")]
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"


def slide_prototype_shares(cfg, S, cl):
    """slide × プロトタイプの割合（slide ごとに patch 数で正規化）。scripts/within_study_finding_prototypes.py と同じ並び。"""
    row_b = np.load(ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"] / "cache.npz")["row"]
    nfs = int((S.set == "fatty").sum())
    row_u = np.load(ROOT / cfg["unseen_features_from"], allow_pickle=True)["row"]
    off_t, off_u = int((S.set == "bayer").sum()), int((S.set == "bayer").sum()) + nfs
    slide_of = np.r_[row_b, np.repeat(np.arange(nfs), cfg["patches_per_slide"]) + off_t, row_u + off_u]
    ns, K = len(S), int(cl.max()) + 1
    w = 1.0 / np.maximum(np.bincount(slide_of, minlength=ns), 1)
    return np.bincount(cl * ns + slide_of, weights=w[slide_of], minlength=K * ns).reshape(K, ns).T


def main():
    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin/config.yml"))
    S = pd.read_csv(R35 / "v1/slides.csv", dtype={"slide_id": str})
    E = pd.read_csv(R35 / "control_similarity/experiments.csv").set_index("exp_id")
    S["grp"] = S.exp.map(E.group_B128)
    S["scanner"] = S.exp.map(E.scanner)
    exp_, grp, ctrl, scan = S.exp.to_numpy(), S.grp.to_numpy(), S.is_ctrl.to_numpy(), S.scanner.to_numpy()
    pos = S[f"grade__{FINDING}"].fillna(0).to_numpy() > 0
    neg = ~ctrl & ~pos
    lr = lambda a, b, P: np.log2((P[a].mean(0) + 1e-4) / (P[b].mean(0) + 1e-4))
    rows = []
    for run, rep, title in REPS:
        P = slide_prototype_shares(cfg, S, np.load(R35 / run / f"assign_{rep}_split_record.npy").astype(np.int64))
        for g in GROUPS:
            ing = grp == g
            hyp_exps = [e for e in np.unique(exp_[ing & pos]) if (exp_ == e)[pos].sum() >= MIN_N and ((exp_ == e) & neg).sum() >= MIN_N]
            nonhyp_treated = ing & ~ctrl & ~np.isin(exp_, hyp_exps) & ~pos
            for e in hyp_exps:
                tr = [x for x in hyp_exps if x != e]
                gc = ing & ctrl & (exp_ != e)                       # まとまりの対照群（伏せた試験の対照群は除く）
                LRc = np.array([lr((exp_ == x) & pos, gc, P) for x in tr])
                LRn = np.array([lr((exp_ == x) & pos, (exp_ == x) & neg, P) for x in tr])
                LRg = np.array([lr((exp_ == x) & neg, gc, P) for x in tr])
                me = min(2, len(tr))
                sel = (((LRc >= LR_UP) & (LRn >= LR_UP)).sum(0) >= me) & (LRn.mean(0) >= LR_UP) & ((LRg >= LR_UP).sum(0) < me)
                score = P[:, sel].sum(1)
                tgt = (exp_ == e) & pos
                ref = {"own_ctrl": (exp_ == e) & ctrl, "group_ctrl": gc, "other_ctrl_same_scanner": ~ing & ctrl & (scan == scan[tgt][0]) & ~np.isnan(grp),
                       "other_ctrl_diff_scanner": ~ing & ctrl & (scan != scan[tgt][0]) & ~np.isnan(grp), "own_neg": (exp_ == e) & neg, "group_nonhyp_treated": nonhyp_treated}
                r = {"representation": title, "rep": rep, "group": g, "exp": e, "compound": S.COMPOUND_NAME[exp_ == e].iloc[0], "n_hyp": int(tgt.sum()), "n_train_exps": len(tr),
                     "n_selected": int(sel.sum()), "share_hyp_in_selected": float(score[tgt].mean())}
                for k, _ in REFS:
                    m_ = ref[k]
                    r[f"auc_{k}"] = float(roc_auc_score(np.r_[np.ones(tgt.sum()), np.zeros(m_.sum())], np.r_[score[tgt], score[m_]])) if m_.sum() >= MIN_N and sel.any() else np.nan
                    r[f"n_{k}"] = int(m_.sum())
                rows.append(r)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("n_") or k in ("n_hyp", "n_selected")}, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig, axes = plt.subplots(1, len(REPS), figsize=(14, 5.8), layout="constrained", sharey=True)
    cols = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#009E73", "#CC79A7"]
    mk = ["o", "o", "s", "s", "D", "^"]
    for ax, (_, rep, title) in zip(axes, REPS):
        d = res[res.rep == rep].reset_index(drop=True)
        for j, ((k, lab), c, m) in enumerate(zip(REFS, cols, mk)):
            ax.scatter(d[f"auc_{k}"], np.arange(len(d)) + (j - 2.5) * 0.11, color=c, marker=m, s=34, label=lab, edgecolors="#333333", linewidths=0.3)
        ax.set_yticks(range(len(d)), [f"まとまり {r.group}：{r.compound}（肥大 {r.n_hyp} 枚）" for r in d.itertuples()])
        for y in np.where(np.diff(d.group.to_numpy()) != 0)[0]:
            ax.axhline(y + 0.5, color="#BBBBBB", lw=0.6)
        ax.axvline(0.5, color="black", lw=0.6, ls=":")
        ax.invert_yaxis()
        ax.set(xlim=(0.3, 1.02), xlabel="AUC（伏せた試験の肥大の slide と、各相手との判別）", title=title)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("まとまりの他の試験だけで選んだ肥大のプロトタイプは、伏せた試験の肥大を当てるか\n自分の対照群と「まとまりの他の試験の対照群」の AUC がそろえば、まとまりの中の試験は入れ替えてよい", fontsize=10)
    fig.savefig(OUT / "F1_group_crossstudy_hypertrophy.png", dpi=180)
    print("done", flush=True)


if __name__ == "__main__":
    main()
