"""試験の中だけで完結する比べ方で、ラベルとして与えていない所見のプロトタイプを選び、評価する（2026-11-05ユーザー: 試験間差がプロトタイプに残る以上、
試験の中の比較に限るアプローチでどこまで言えるか）。scripts/unlabeled_finding_prototypes.py の発展版。

0035 の割当（3,737 枚）。所見ごとに:
  比べる相手は同じ試験・同じ時点（解剖日）の対照群と、記録のない投与群（同じ時点に 2 枚以上なければ、その試験の全時点）
  試験ごとに log2 比: 所見/対照群（LRc）、所見/記録のない投与群（LRn）、記録のない投与群/対照群（LRg、投与全般で増えるか）
  選び方（学習側の試験だけで）: LRc >= 1 かつ LRn >= 1 の試験が min_exps 以上、LRn の平均 >= 1、かつ LRg >= 1 の試験が min_exps 未満（投与全般で増えるものは除く）
  評価（化合物単位の 5 分割、伏せた化合物の試験で）:
    within-AUC: 試験の中だけで、所見の slide か記録のない投与群かを、選んだプロトタイプの割合で判別した AUC（試験ごと）の平均
    pooled AUC: 試験をまたいでまとめた AUC（同じ試験・同じ時点の対照群の平均を引いたスコア）
    程度: 所見の slide の中で、スコアと記録の程度の順位相関（試験ごと）の平均
  所見どうしで、選ばれたプロトタイプの重なり（Jaccard）
出力: outputs/0035_20261102_finer_prototypes_split_rejoin/within_study_findings/
usage: python scripts/within_study_finding_prototypes.py   （コンテナ内、Slurm経由。CPU、30 分程度）
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
R35 = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin"
OUT = R35 / "within_study_findings"
REPS = [("v1", "A128_sphering_lda", "split_record", "A128（白色化128 ‖ LDA）"), ("v2", "B128_sphering", "split_record", "B128（ラベルなし）")]
FINDINGS = ["Hypertrophy", "Change, eosinophilic", "Vacuolization, cytoplasmic", "Proliferation, bile duct", "Degeneration, granular, eosinophilic",
            "Necrosis", "Single cell necrosis", "Cellular infiltration", "Increased mitosis", "Hematopoiesis, extramedullary"]
JP = {"Hypertrophy": "肥大", "Change, eosinophilic": "好酸性変化", "Vacuolization, cytoplasmic": "細胞質空胞化", "Proliferation, bile duct": "胆管増生",
      "Degeneration, granular, eosinophilic": "顆粒状好酸性変性", "Necrosis": "壊死", "Cellular infiltration": "細胞浸潤", "Increased mitosis": "核分裂増加",
      "Single cell necrosis": "単細胞壊死", "Hematopoiesis, extramedullary": "髄外造血"}
LABELED = {"Necrosis", "Single cell necrosis", "Cellular infiltration", "Increased mitosis", "Hematopoiesis, extramedullary"}
MIN_N, LR_UP, MIN_EXPS_FRAC, FOLDS, N_BOOT, SEED = 2, 1.0, 0.25, 5, 2000, 0
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"


def main():
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin/config.yml"))
    S = pd.read_csv(R35 / "v1/slides.csv", dtype={"slide_id": str})
    q = pd.read_csv(cfg["qc_catalog"], dtype=str).set_index("slide_id")
    S["period"] = q.SACRIFICE_PERIOD.reindex(S.slide_id).fillna("?").to_numpy()
    row_b = np.load(ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"] / "cache.npz")["row"]
    nfs = int((S.set == "fatty").sum())
    row_u = np.load(ROOT / cfg["unseen_features_from"], allow_pickle=True)["row"]
    off_t, off_u = int((S.set == "bayer").sum()), int((S.set == "bayer").sum()) + nfs
    slide_of = np.r_[row_b, np.repeat(np.arange(nfs), cfg["patches_per_slide"]) + off_t, row_u + off_u]
    ns = len(S)
    w_slide = 1.0 / np.maximum(np.bincount(slide_of, minlength=ns), 1)
    rng = np.random.default_rng(SEED)
    rows, sels = [], {}
    for run, rep, method, title in REPS:
        cl = np.load(R35 / run / f"assign_{rep}_{method}.npy").astype(np.int64)
        K = int(cl.max()) + 1
        P = np.bincount(cl * ns + slide_of, weights=w_slide[slide_of], minlength=K * ns).reshape(K, ns).T
        for F in FINDINGS:
            g = S[f"grade__{F}"].fillna(0).to_numpy()
            pos = g > 0
            ctrl = S.is_ctrl.to_numpy()
            neg = ~ctrl & ~pos
            exp_ = S.exp.to_numpy()
            per_ = S.period.to_numpy()
            # 試験ごとの比べる相手（同じ時点を優先）
            groups = {}
            for e in np.unique(exp_[pos]):
                me = exp_ == e
                pp = np.unique(per_[me & pos])
                c_ = me & ctrl & np.isin(per_, pp)
                c_ = c_ if c_.sum() >= MIN_N else me & ctrl
                n_ = me & neg & np.isin(per_, pp)
                n_ = n_ if n_.sum() >= MIN_N else me & neg
                if c_.sum() >= MIN_N and n_.sum() >= MIN_N:
                    groups[e] = (me & pos, c_, n_)
            exps = sorted(groups)
            if len(exps) < 2:
                print(f"{F}: too few experiments ({len(exps)})", flush=True)
                continue
            comp_of = {e: S.COMPOUND_NAME[exp_ == e].iloc[0] for e in exps}
            LRc = np.array([np.log2((P[groups[e][0]].mean(0) + 1e-4) / (P[groups[e][1]].mean(0) + 1e-4)) for e in exps])
            LRn = np.array([np.log2((P[groups[e][0]].mean(0) + 1e-4) / (P[groups[e][2]].mean(0) + 1e-4)) for e in exps])
            LRg = np.array([np.log2((P[groups[e][2]].mean(0) + 1e-4) / (P[groups[e][1]].mean(0) + 1e-4)) for e in exps])

            def select(ix):
                me = max(2, int(np.ceil(MIN_EXPS_FRAC * len(ix))))
                up = ((LRc[ix] >= LR_UP) & (LRn[ix] >= LR_UP)).sum(0) >= me
                generic = (LRg[ix] >= LR_UP).sum(0) >= me
                return up & (LRn[ix].mean(0) >= LR_UP) & ~generic
            sel_all = select(np.arange(len(exps)))
            sels[(F, rep)] = set(np.where(sel_all)[0])
            comps = sorted(set(comp_of.values()))
            fold_of = {c: k % FOLDS for k, c in enumerate(rng.permutation(comps))}
            score = np.full(ns, np.nan)
            n_sel = []
            for f in range(min(FOLDS, len(comps))):
                tr = np.array([i for i, e in enumerate(exps) if fold_of[comp_of[e]] != f])
                te = [e for e in exps if fold_of[comp_of[e]] == f]
                if len(tr) < 2 or not te:
                    continue
                sel = select(tr)
                n_sel.append(int(sel.sum()))
                for e in te:
                    p_, c_, n_ = groups[e]
                    m_ = p_ | c_ | n_
                    score[m_] = P[m_][:, sel].sum(1) - P[c_][:, sel].sum(1).mean()
            within, rho = [], []
            for e in exps:
                p_, c_, n_ = groups[e]
                m_ = (p_ | n_) & np.isfinite(score)
                if pos[m_].any() and (~pos[m_]).any():
                    within.append(roc_auc_score(pos[m_].astype(int), score[m_]))
                pm = p_ & np.isfinite(score)
                if pm.sum() >= 4 and len(np.unique(g[pm])) > 1:
                    rho.append(spearmanr(g[pm], score[pm])[0])
            within, rho = np.array(within), np.array(rho)
            bs = [within[rng.integers(0, len(within), len(within))].mean() for _ in range(N_BOOT)] if len(within) > 1 else [np.nan]
            allm = np.zeros(ns, bool)
            for e in exps:
                allm |= groups[e][0] | groups[e][2]
            allm &= np.isfinite(score)
            pooled = roc_auc_score(pos[allm].astype(int), score[allm]) if len(set(pos[allm])) == 2 else np.nan
            r = {"finding": F, "finding_jp": JP[F], "bayer_label": F in LABELED, "representation": title, "rep": rep, "n_exps": len(exps), "n_compounds": len(comps),
                 "n_selected_all": int(sel_all.sum()), "median_n_selected_fold": float(np.median(n_sel)) if n_sel else np.nan,
                 "within_auc_mean": float(within.mean()) if len(within) else np.nan, "within_auc_lo": float(np.nanpercentile(bs, 2.5)), "within_auc_hi": float(np.nanpercentile(bs, 97.5)),
                 "n_exps_within": int(len(within)), "share_exps_within_auc_gt_0.5": float((within > 0.5).mean()) if len(within) else np.nan,
                 "pooled_auc_vs_treated": pooled, "grade_rho_mean": float(rho.mean()) if len(rho) else np.nan, "n_exps_grade": int(len(rho)),
                 "share_finding_slides_in_selected": float(np.mean([P[groups[e][0]][:, sel_all].sum(1).mean() for e in exps])),
                 "share_neg_treated_in_selected": float(np.mean([P[groups[e][2]][:, sel_all].sum(1).mean() for e in exps])),
                 "share_control_in_selected": float(np.mean([P[groups[e][1]][:, sel_all].sum(1).mean() for e in exps]))}
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("finding", "rep")}, flush=True)
            pd.DataFrame({"prototype": np.arange(K), "selected": sel_all, "n_exps_up_both": ((LRc >= LR_UP) & (LRn >= LR_UP)).sum(0), "mean_LRn": LRn.mean(0),
                          "n_exps_generic": (LRg >= LR_UP).sum(0)}).to_csv(OUT / f"prototypes_{F.replace(',', '').replace(' ', '_')}_{rep}.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "summary.csv", index=False)
    for rep in dict.fromkeys(r[1] for r in REPS):
        fs = [F for F in FINDINGS if (F, rep) in sels]
        J = pd.DataFrame([[len(sels[(a, rep)] & sels[(b, rep)]) / max(len(sels[(a, rep)] | sels[(b, rep)]), 1) for b in fs] for a in fs], index=[JP[f] for f in fs], columns=[JP[f] for f in fs])
        J.to_csv(OUT / f"selected_overlap_{rep}.csv")
        print(rep, "\n", J.round(2).to_string(), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), layout="constrained", gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    cols = {"A128_sphering_lda": "#D55E00", "B128_sphering": "#0072B2"}
    fs = [F for F in FINDINGS if (res.finding == F).any()]
    for k, F in enumerate(fs):
        for d, (rep, col) in zip((-0.15, 0.15), cols.items()):
            r = res[(res.finding == F) & (res.rep == rep)]
            if len(r) and np.isfinite(r.within_auc_mean.iloc[0]):
                r = r.iloc[0]
                ax.errorbar(r.within_auc_mean, k + d, xerr=[[r.within_auc_mean - r.within_auc_lo], [r.within_auc_hi - r.within_auc_mean]], fmt="o" if rep.startswith("A") else "s", color=col, capsize=3, ms=6,
                            label=r.representation if k == 0 else None)
    r0 = res[res.rep == REPS[0][1]].set_index("finding")
    ax.set_yticks(range(len(fs)), [f"{JP[F]}{'（Bayer にラベルあり）' if F in LABELED else ''}　{int(r0.n_compounds.get(F, 0))} 化合物、選ばれた数 {int(r0.n_selected_all.get(F, 0))}" for F in fs])
    ax.invert_yaxis()
    ax.axvline(0.5, color="black", lw=0.6, ls=":")
    ax.set(xlim=(0.2, 1.05), xlabel="試験の中の AUC（所見の slide か、同じ試験・同じ時点の記録のない投与群か）の平均",
           title="試験の中だけで、所見に特有のプロトタイプは記録の有無を当てるか（化合物を伏せた評価）\n点 = 伏せた化合物の試験の平均、線 = 試験を単位にした 95% ブートストラップ区間")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax = axes[1]
    J = pd.read_csv(OUT / f"selected_overlap_{REPS[0][1]}.csv", index_col=0)
    im = ax.imshow(J.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(J)), J.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(J)), J.index)
    for i in range(len(J)):
        for j in range(len(J)):
            v = J.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Jaccard 係数")
    ax.set_title("所見に特有として選ばれたプロトタイプの重なり（A128）\n（投与全般で増えるものは除いた後）")
    fig.savefig(OUT / "F1_within_study.png", dpi=180)
    print("done", flush=True)


if __name__ == "__main__":
    main()
