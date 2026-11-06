"""プロトタイプを選ぶときの比べる相手（対照群）を、対照群で見分けられない試験のまとまりの中から取るか外から取るかで、
伏せた試験での所見の当たり方が変わるか（2026-11-06 ユーザー: 試験間差に埋もれやすい弱い所見で、まとまりで分けることが効くかを見る）。

対象: 0035 の割当で、所見が 2 枚以上記録された投与群の試験が、同じまとまりに 2 つ以上ある組（所見 × まとまり）。
まとまりの中で 1 試験を伏せる。残りの試験の所見の slide を、比べる相手 R と比べてプロトタイプを選ぶ:
  log2(所見/R) >= 1 の試験が me 以上（me = min(2, 学習の試験数)）、平均 >= 1、log2(記録のない投与群/R) >= 1 の試験が me 未満（投与全般で増えるものは除く）。
  同じ試験の記録のない投与群とは比べない（R の違いだけが効くようにするため）。
R: まとまりの中の対照群（伏せた試験を除く）/ まとまりの外の対照群（同じスキャナ、違うスキャナ）。外は、中と同じ数の試験を 20 回ランダムに選ぶ。
評価: 伏せた試験の所見の slide を、自分の記録のない投与群・自分の対照群と判別した AUC（選んだプロトタイプの割合の和）。
出力: outputs/0035_20261102_finer_prototypes_split_rejoin/group_reference/
--no-generic: 投与全般で増えるものを除かない（所見の slide と R を比べるだけ。除く処理は R と学習の試験の試験間差も一緒に消すため、その効果を切り分ける）。出力は group_reference/no_generic/
usage: python scripts/group_reference_controls.py [--no-generic]   （コンテナ内、Slurm経由。CPU、数分）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from group_crossstudy_findings import FONT_JP, R35, ROOT, slide_prototype_shares

NO_GENERIC = "--no-generic" in sys.argv
OUT = R35 / "group_reference" / ("no_generic" if NO_GENERIC else "")
REPS = [("v2", "B128_sphering", "B128（ラベルなし）"), ("v1", "A128_sphering_lda", "A128（白色化128 ‖ LDA）")]
JP = {"Hypertrophy": "肥大", "Change, eosinophilic": "好酸性変化", "Vacuolization, cytoplasmic": "細胞質空胞化", "Proliferation, bile duct": "胆管増生",
      "Necrosis": "壊死", "Cellular infiltration": "細胞浸潤", "Increased mitosis": "核分裂増加", "Single cell necrosis": "単細胞壊死", "Hematopoiesis, extramedullary": "髄外造血"}
MIN_N, LR_UP, N_DRAW, SEED = 2, 1.0, 20, 0
REFS = [("in", "まとまりの中の対照群"), ("out_same", "まとまりの外の対照群（同じスキャナ）"), ("out_diff", "まとまりの外の対照群（違うスキャナ）")]


def main():
    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin/config.yml"))
    S = pd.read_csv(R35 / "v1/slides.csv", dtype={"slide_id": str})
    E = pd.read_csv(R35 / "control_similarity/experiments.csv").set_index("exp_id")
    S["grp"] = S.exp.map(E.group_B128)
    S["scanner"] = S.exp.map(E.scanner)
    exp_, grp, ctrl, scan = S.exp.to_numpy(), S.grp.to_numpy(), S.is_ctrl.to_numpy(), S.scanner.to_numpy()
    ctrl_exps = pd.Series(ctrl).groupby(exp_).sum()
    ctrl_exps = ctrl_exps[ctrl_exps >= MIN_N].index.to_numpy()          # 対照群が 2 枚以上ある試験
    auc = lambda a, b, s: float(roc_auc_score(np.r_[np.ones(a.sum()), np.zeros(b.sum())], np.r_[s[a], s[b]]))
    rows = []
    for run, rep, title in REPS:
        P = slide_prototype_shares(cfg, S, np.load(R35 / run / f"assign_{rep}_split_record.npy").astype(np.int64))
        rng = np.random.default_rng(SEED)
        for F in JP:
            pos = S[f"grade__{F}"].fillna(0).to_numpy() > 0
            neg = ~ctrl & ~pos
            for g in np.unique(grp[~np.isnan(grp)]):
                ing = grp == g
                exps = [e for e in np.unique(exp_[ing & pos]) if ((exp_ == e) & pos).sum() >= MIN_N and ((exp_ == e) & neg).sum() >= MIN_N and ((exp_ == e) & ctrl).sum() >= MIN_N]
                if len(exps) < 2:
                    continue
                for e in exps:
                    tr = [x for x in exps if x != e]
                    me = min(2, len(tr))
                    sc = scan[(exp_ == e)][0]
                    in_exps = [x for x in np.unique(exp_[ing & ctrl]) if x != e and x in ctrl_exps]
                    out_pool = {k: [x for x in ctrl_exps if x not in np.unique(exp_[ing]) and (E.scanner.get(x) == sc) == (k == "out_same")] for k in ("out_same", "out_diff")}
                    draws = {"in": [in_exps]} | {k: [rng.choice(v, len(in_exps), replace=False) for _ in range(N_DRAW)] for k, v in out_pool.items()}
                    for k, lab in REFS:
                        a_n, a_c, n_sel = [], [], []
                        for ref_exps in draws[k]:
                            R = np.isin(exp_, ref_exps) & ctrl
                            LRc = np.array([np.log2((P[(exp_ == x) & pos].mean(0) + 1e-4) / (P[R].mean(0) + 1e-4)) for x in tr])
                            LRg = np.array([np.log2((P[(exp_ == x) & neg].mean(0) + 1e-4) / (P[R].mean(0) + 1e-4)) for x in tr])
                            sel = ((LRc >= LR_UP).sum(0) >= me) & (LRc.mean(0) >= LR_UP) & (NO_GENERIC | ((LRg >= LR_UP).sum(0) < me))
                            s = P[:, sel].sum(1)
                            n_sel.append(int(sel.sum()))
                            if sel.any():
                                a_n.append(auc((exp_ == e) & pos, (exp_ == e) & neg, s))
                                a_c.append(auc((exp_ == e) & pos, (exp_ == e) & ctrl, s))
                        rows.append({"representation": title, "rep": rep, "finding": F, "finding_jp": JP[F], "group": int(g), "exp": e, "compound": S.COMPOUND_NAME[exp_ == e].iloc[0],
                                     "n_pos": int(((exp_ == e) & pos).sum()), "n_train_exps": len(tr), "n_ref_exps": len(in_exps), "ref": k, "ref_jp": lab,
                                     "n_selected": float(np.mean(n_sel)), "auc_vs_own_neg": float(np.mean(a_n)) if a_n else np.nan, "auc_vs_own_ctrl": float(np.mean(a_c)) if a_c else np.nan})
                    print({k_: rows[-1][k_] for k_ in ("rep", "finding", "group", "compound")}, [(r["ref"], round(r["auc_vs_own_neg"], 3), round(r["n_selected"])) for r in rows[-3:]], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "summary.csv", index=False)
    W = res.pivot_table(index=["rep", "finding_jp", "group", "compound"], columns="ref", values="auc_vs_own_neg")
    for rep in W.index.get_level_values(0).unique():
        w = W.loc[rep]
        for k in ("out_same", "out_diff"):
            d = (w["in"] - w[k]).dropna()
            print(rep, k, f"in - out: median {d.median():+.3f}, in better in {int((d > 0).sum())}/{len(d)}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig, axes = plt.subplots(1, len(REPS), figsize=(14, 0.32 * len(W) / len(REPS) + 2.2), layout="constrained", sharey=True)
    cols, mk = {"in": "#0072B2", "out_same": "#E69F00", "out_diff": "#D55E00"}, {"in": "o", "out_same": "s", "out_diff": "s"}
    for ax, (_, rep, title) in zip(axes, REPS):
        w = W.loc[rep].sort_index(level=[0, 1])
        y = np.arange(len(w))
        for k, lab in REFS:
            ax.scatter(w[k], y, color=cols[k], marker=mk[k], s=30, label=lab, edgecolors="#333333", linewidths=0.3, zorder=3)
        ax.hlines(y, w[["in", "out_same", "out_diff"]].min(axis=1), w[["in", "out_same", "out_diff"]].max(axis=1), color="#CCCCCC", lw=0.8, zorder=1)
        ax.set_yticks(y, [f"{f}・まとまり {g}：{c}" for f, g, c in w.index])
        ax.axvline(0.5, color="black", lw=0.6, ls=":")
        ax.set(xlim=(0.2, 1.02), xlabel="AUC（伏せた試験の所見の slide と、自分の記録のない投与群）", title=title)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("プロトタイプを選ぶときの比べる相手を、まとまりの中の対照群にするか外の対照群にするか\n外は、中と同じ数の試験を 20 回ランダムに選んだ平均" + ("。投与全般で増えるものを除かない版" if NO_GENERIC else ""), fontsize=10)
    fig.savefig(OUT / "F1_reference_in_vs_out.png", dpi=180)
    print("done", flush=True)


if __name__ == "__main__":
    main()
