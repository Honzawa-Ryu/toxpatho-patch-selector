"""ラベルとして与えていない所見（既定: 肥大）が、正常とは別のプロトタイプに入るか、化合物をまたいで拾えるかを調べる（2026-11-05ユーザー: ②は
「ラベルとして与えていない所見を拾える」で示す。大滴・小滴は試験間差と切り分けられない。まず肥大と正常が別のプロトタイプに入るか）。

0035 の割当（3,737 枚、表現は作り直さない）を使う。slide ごとのプロトタイプ構成比（slide を同じ重みにそろえる）から:
  1. 試験ごとに、所見の記録がある slide と同じ試験の対照群の構成比の比（log2）。プロトタイプごとに、2 倍以上増える試験の数
  2. 化合物単位の分割で、学習側の試験だけで「所見のプロトタイプ」（2 倍以上増える試験が min_exps 以上、平均の log2 比 >= 1）を選び、
     伏せた化合物の slide で、その割合（同じ試験の対照群との差）から記録の有無を判別する（AUC: 対照群との比較、記録のない投与群との比較）
  3. 所見の slide の patch のうち選んだプロトタイプに入る割合と、対照群の slide での割合
  4. 図: 試験ごとの比、化合物を伏せた AUC、タイル（異なる化合物の所見の slide から）
出力: outputs/0035_20261102_finer_prototypes_split_rejoin/unlabeled_findings/<所見>/
usage: python scripts/unlabeled_finding_prototypes.py [所見名]   例: "Hypertrophy"（コンテナ内、Slurm経由。CPU、30 分程度）
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FINDING = sys.argv[1] if len(sys.argv) > 1 else "Hypertrophy"
R35 = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin"
OUT = R35 / "unlabeled_findings" / FINDING.replace(",", "").replace(" ", "_")
REPS = [("v1", "A128_sphering_lda", "split_record", "A128（白色化128 ‖ LDA）"), ("v2", "B128_sphering", "split_record", "B128（白色化128、ラベルなし）"),
        ("v1", "A128_sphering_lda", "flat_k1000", "A128、k=1000")]
MIN_CTRL, LR_UP, MIN_EXPS_FRAC, FOLDS, N_BOOT, SEED = 2, 1.0, 0.25, 5, 2000, 0
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "gray": "#999999"}


def main():
    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin/config.yml"))
    S = pd.read_csv(R35 / "v1/slides.csv", dtype={"slide_id": str})
    row_b = np.load(ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"] / "cache.npz")["row"]
    xy_b = np.load(ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"] / "cache.npz")["xy"]
    nfs = int((S.set == "fatty").sum())
    z = np.load(ROOT / cfg["unseen_features_from"], allow_pickle=True)
    row_u, xy_u = z["row"], z["xy"]
    del z
    off_t, off_u = int((S.set == "bayer").sum()), int((S.set == "bayer").sum()) + nfs
    slide_of = np.r_[row_b, np.repeat(np.arange(nfs), cfg["patches_per_slide"]) + off_t, row_u + off_u]
    nb, nt = len(row_b), nfs * cfg["patches_per_slide"]
    ns = len(S)
    w_slide = 1.0 / np.maximum(np.bincount(slide_of, minlength=ns), 1)
    gcol = f"grade__{FINDING}"
    others = [c for c in S.columns if c.startswith("grade__") and c != gcol]
    S["pos"] = S[gcol].fillna(0) > 0
    S["pure"] = S.pos & (S[others].fillna(0).sum(axis=1) == 0)
    S["neg_treated"] = ~S.is_ctrl & ~S.pos
    ctrl_n = S[S.is_ctrl].groupby("exp").size()
    exps = sorted(e for e in S[S.pos].exp.unique() if ctrl_n.get(e, 0) >= MIN_CTRL)
    comp_of_exp = S.drop_duplicates("exp").set_index("exp").COMPOUND_NAME.to_dict()
    print(f"{FINDING}: {int(S.pos.sum())} slides with records; {len(exps)} experiments with >= {MIN_CTRL} controls ({len(set(comp_of_exp[e] for e in exps))} compounds); pure {int((S.pure & S.exp.isin(exps)).sum())}", flush=True)
    rng = np.random.default_rng(SEED)
    summ, keep_for_tiles = [], {}
    for run, rep, method, title in REPS:
        cl = np.load(R35 / run / f"assign_{rep}_{method}.npy").astype(np.int64)
        K = int(cl.max()) + 1
        P = np.bincount(cl * ns + slide_of, weights=w_slide[slide_of], minlength=K * ns).reshape(K, ns).T   # slide × K、各行の和 1
        # 1. 試験ごとの log2 比（所見の slide の平均 / 対照群の平均）
        LR = np.zeros((len(exps), K))
        for i, e in enumerate(exps):
            pe = P[(S.exp == e).to_numpy() & S.pos.to_numpy()].mean(0)
            ce = P[(S.exp == e).to_numpy() & S.is_ctrl.to_numpy()].mean(0)
            LR[i] = np.log2((pe + 1e-4) / (ce + 1e-4))
        n_up = (LR >= LR_UP).sum(0)
        names = pd.read_csv(R35 / run / f"prototype_names_{rep}_{method}.csv").set_index("prototype") if (R35 / run / f"prototype_names_{rep}_{method}.csv").exists() else None
        tab = pd.DataFrame({"prototype": np.arange(K), "n_exps_up2x": n_up, "mean_log2_ratio": LR.mean(0), "median_log2_ratio": np.median(LR, 0),
                            "share_in_finding_slides": P[S.pos.to_numpy() & S.exp.isin(exps).to_numpy()].mean(0), "share_in_control_slides": P[S.is_ctrl.to_numpy() & S.exp.isin(exps).to_numpy()].mean(0)})
        if names is not None:
            tab["record_name"] = names.record_name.reindex(range(K)).fillna("").to_numpy()
            tab["bayer_name"] = names.bayer_name.reindex(range(K)).fillna("").to_numpy()
        min_exps = max(3, int(np.ceil(MIN_EXPS_FRAC * len(exps))))
        sel_all = (n_up >= min_exps) & (LR.mean(0) >= LR_UP)
        tab["selected_all_exps"] = sel_all
        tab.sort_values(["n_exps_up2x", "mean_log2_ratio"], ascending=False).to_csv(OUT / f"prototypes_{rep}_{method}.csv", index=False)
        pd.DataFrame(LR, index=[f"exp{e} {comp_of_exp[e]}" for e in exps]).to_csv(OUT / f"log2_ratio_by_exp_{rep}_{method}.csv")
        # 2. 化合物単位の分割: 学習側で選び、伏せた化合物で判別
        comps = sorted(set(comp_of_exp[e] for e in exps))
        fold_of = {c: k % FOLDS for k, c in enumerate(rng.permutation(comps))}
        scores = np.full(ns, np.nan)
        n_sel = []
        for f in range(FOLDS):
            tr = [i for i, e in enumerate(exps) if fold_of[comp_of_exp[e]] != f]
            te = [e for e in exps if fold_of[comp_of_exp[e]] == f]
            me = max(3, int(np.ceil(MIN_EXPS_FRAC * len(tr))))
            sel = ((LR[tr] >= LR_UP).sum(0) >= me) & (LR[tr].mean(0) >= LR_UP)
            n_sel.append(int(sel.sum()))
            for e in te:
                m_ = (S.exp == e).to_numpy()
                base = P[m_ & S.is_ctrl.to_numpy()][:, sel].sum(1).mean()
                scores[m_] = P[m_][:, sel].sum(1) - base
        ev = S.exp.isin(exps).to_numpy() & np.isfinite(scores)
        res = {"representation": title, "rep": rep, "method": method, "K": K, "n_exps": len(exps), "n_selected_all": int(sel_all.sum()), "n_selected_per_fold": n_sel}
        cmp_of_slide = S.COMPOUND_NAME.to_numpy()
        for nm_, negm in (("vs_control", S.is_ctrl.to_numpy()), ("vs_treated_without_record", S.neg_treated.to_numpy()), ("vs_both", S.is_ctrl.to_numpy() | S.neg_treated.to_numpy())):
            for pnm, posm in (("all", S.pos.to_numpy()), ("pure", S.pure.to_numpy())):
                m_ = ev & (posm | negm)
                y = posm[m_].astype(int)
                if len(set(y)) < 2:
                    continue
                auc = roc_auc_score(y, scores[m_])
                cs_ = np.unique(cmp_of_slide[m_])
                idx_c = {c: np.where(cmp_of_slide[m_] == c)[0] for c in cs_}
                bs = []
                for _ in range(N_BOOT):
                    ii = np.concatenate([idx_c[c] for c in rng.choice(cs_, len(cs_))])
                    if len(set(y[ii])) == 2:
                        bs.append(roc_auc_score(y[ii], scores[m_][ii]))
                res[f"auc_{pnm}_{nm_}"] = auc
                res[f"auc_{pnm}_{nm_}_lo"], res[f"auc_{pnm}_{nm_}_hi"] = np.percentile(bs, [2.5, 97.5])
        # 3. 選んだプロトタイプ（全試験で選んだもの）に入る patch の割合
        res["share_finding_slides_in_selected"] = float(P[S.pos.to_numpy() & S.exp.isin(exps).to_numpy()][:, sel_all].sum(1).mean())
        res["share_control_slides_in_selected"] = float(P[S.is_ctrl.to_numpy() & S.exp.isin(exps).to_numpy()][:, sel_all].sum(1).mean())
        summ.append(res)
        keep_for_tiles[(rep, method)] = (cl, tab[tab.selected_all_exps].sort_values(["n_exps_up2x", "mean_log2_ratio"], ascending=False), LR, title)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    sm = pd.DataFrame(summ)
    sm.to_csv(OUT / "summary.csv", index=False)

    # 図
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    # F1: 化合物を伏せた AUC（表現ごと、比較相手ごと）
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    comps_ = [("all", "vs_control", "対照群と", "o"), ("all", "vs_treated_without_record", "記録のない投与群と", "s"), ("pure", "vs_control", "（肥大のみの slide）対照群と", "D")]
    cols = [OI["vermillion"], OI["blue"], OI["gray"]]
    for j, (_, r) in enumerate(sm.iterrows()):
        for k, (pnm, nm_, lab_, mk) in enumerate(comps_):
            key = f"auc_{pnm}_{nm_}"
            if key not in r or not np.isfinite(r[key]):
                continue
            y = j + (k - 1) * 0.22
            ax.errorbar(r[key], y, xerr=[[r[key] - r[key + "_lo"]], [r[key + "_hi"] - r[key]]], fmt=mk, color=cols[j], capsize=3, ms=6)
            ax.text(r[key + "_hi"] + 0.01, y, lab_, va="center", fontsize=7, color="#333333")
    ax.set_yticks(range(len(sm)), [f"{r.representation}（選ばれた数 {r.n_selected_all}）" for _, r in sm.iterrows()])
    ax.invert_yaxis()
    ax.axvline(0.5, color="black", lw=0.6, ls=":")
    ax.set(xlim=(0.3, 1.05), xlabel="AUC（伏せた化合物の slide で、選んだプロトタイプの割合から記録の有無を判別）",
           title=f"{FINDING}: 学習側の化合物で選んだ「所見のプロトタイプ」は、\n伏せた化合物でも記録の有無を当てるか（{len(exps)} 試験。線 = 化合物を単位にした 95% ブートストラップ区間）")
    fig.savefig(OUT / "F1_heldout_auc.png", dpi=180)
    plt.close(fig)
    # F2: 試験ごとの log2 比（主表現、選んだプロトタイプ上位 12 個）
    cl, top, LR, title = keep_for_tiles[(REPS[0][1], REPS[0][2])]
    top = top.head(12)
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(top) + 1.8), layout="constrained")
    for k, (_, r) in enumerate(top.iterrows()):
        v = LR[:, int(r.prototype)]
        ax.scatter(np.clip(v, -4, 6), k + np.random.default_rng(k).uniform(-0.15, 0.15, len(v)), s=12, color=OI["blue"], alpha=0.7)
        ax.plot([np.median(v)] * 2, [k - 0.3, k + 0.3], color="black", lw=1.5)
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.axvline(LR_UP, color=OI["vermillion"], lw=0.8, ls="--")
    ax.set_yticks(range(len(top)), [f"p{int(r.prototype):04d}（2 倍以上の試験 {int(r.n_exps_up2x)}/{len(exps)}、記録名 {r.get('record_name', '') or 'なし'}）" for _, r in top.iterrows()])
    ax.invert_yaxis()
    ax.set(xlabel="log2（所見の slide での割合 / 同じ試験の対照群での割合）。点 = 試験（化合物）、黒線 = 中央値、赤破線 = 2 倍", title=f"{FINDING} で増えるプロトタイプ（{title}）は化合物をまたぐか")
    fig.savefig(OUT / "F2_log2_ratio_by_exp.png", dpi=180)
    plt.close(fig)
    print("figures F1 F2", flush=True)

    # F3: タイル（主表現、上位 6 個。異なる化合物の所見の slide から 8 枚、同じプロトタイプの対照群 slide から 4 枚）
    import openslide
    from PIL import Image, ImageDraw, ImageFont
    q = pd.read_csv(cfg["qc_catalog"], dtype=str).set_index("slide_id")
    bayer = pd.read_csv(ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"] / "slides.csv", dtype=str).set_index("slide_id")

    def svs_of(sid):
        p_ = bayer.svs_path.get(sid)
        if isinstance(p_, str) and Path(p_).exists():
            return p_
        per = q.SACRIFICE_PERIOD.get(sid, "")
        c_ = ROOT / "data/TGGATEs/WSI/Liver" / str(per).replace(" ", "_") / f"{sid}.svs"
        if c_.exists():
            return str(c_)
        g = glob.glob(str(ROOT / "data/TGGATEs/WSI" / "**" / f"{sid}.svs"), recursive=True)
        return g[0] if g else ""
    T = 160
    fj, fjs, fjt = ImageFont.truetype(FONT_JP, 15), ImageFont.truetype(FONT_JP, 11), ImageFont.truetype(FONT_JP, 22)
    top6 = top.head(6)
    is_t = np.zeros(len(cl), bool)
    is_t[nb:nb + nt] = True   # 脂肪変性の 3 試験は座標を持たない（間引きの座標を読み直していない）ので除く
    pos_p, ctrl_p = S.pos.to_numpy()[slide_of] & S.exp.isin(exps).to_numpy()[slide_of], S.is_ctrl.to_numpy()[slide_of] & S.exp.isin(exps).to_numpy()[slide_of]
    xy_all = np.zeros((len(cl), 2), dtype=np.int64)
    xy_all[:nb], xy_all[nb + nt:] = xy_b, xy_u
    svs_cache = {}
    blocks = []
    for _, r in top6.iterrows():
        j = int(r.prototype)
        picks = {}
        for kind, mk, n_ in (("finding", pos_p, 8), ("control", ctrl_p, 4)):
            idx = np.where((cl == j) & mk & ~is_t)[0]
            out, seen_c = [], set()
            for p_ in rng.permutation(idx)[:20000]:
                sid = S.slide_id[slide_of[p_]]
                c_ = S.COMPOUND_NAME[slide_of[p_]]
                if (kind == "finding" and c_ in seen_c) or sid in [o[1] for o in out]:
                    continue
                if sid not in svs_cache:
                    svs_cache[sid] = svs_of(sid)
                if not svs_cache[sid]:
                    continue
                out.append((int(p_), sid, c_))
                seen_c.add(c_)
                if len(out) >= n_:
                    break
            picks[kind] = out
        blocks.append((r, picks))
    H = 110 + len(blocks) * (2 * (T + 34) + 50)
    sheet = Image.new("RGB", (8 * T + 20, H), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), f"{FINDING} で増えるプロトタイプ（{REPS[0][3]}、split_record）: 上位 {len(blocks)} 個", fill="black", font=fjt)
    d.text((8, 40), "各プロトタイプについて、上段 = 所見の記録がある slide から（なるべく別の化合物）、下段 = 同じプロトタイプに入った対照群の slide から。20 倍、1 枚 約 128 µm 四方。", fill="#333333", font=fj)
    d.text((8, 62), "見出しの数字: 2 倍以上に増えた試験の数、所見の slide / 対照群の slide での平均の割合。", fill="#333333", font=fj)
    y = 100
    for r, picks in blocks:
        d.rectangle([0, y, 8 * T + 20, y + 40], fill="#F2F2F2")
        d.text((8, y + 2), f"p{int(r.prototype):04d}　2 倍以上の試験 {int(r.n_exps_up2x)}/{len(exps)}　平均 log2 比 {r.mean_log2_ratio:.2f}　記録名 {r.get('record_name', '') or 'なし'}　Bayer {r.get('bayer_name', '') or 'なし'}", fill="#B34700", font=fj)
        d.text((8, y + 21), f"割合: 所見の slide {r.share_in_finding_slides:.2%}、対照群の slide {r.share_in_control_slides:.2%}", fill="#333333", font=fj)
        y += 46
        for row_i, kind in enumerate(("finding", "control")):
            for k_, (p_, sid, c_) in enumerate(picks[kind]):
                s0 = 256
                w = openslide.OpenSlide(svs_cache[sid])
                img = w.read_region((int(xy_all[p_, 0]), int(xy_all[p_, 1])), 0, (s0, s0)).convert("RGB").resize((T, T))
                w.close()
                x0 = 8 + k_ * T
                d.text((x0 + 2, y), f"{'所見' if kind == 'finding' else '対照'} {str(c_)[:14]}", fill="black" if kind == "finding" else "#0072B2", font=fjs)
                sheet.paste(img, (x0, y + 14))
            y += T + 20
        y += 14
    sheet.save(OUT / "F3_tiles_top_prototypes.jpg", quality=88)
    print("done", flush=True)


if __name__ == "__main__":
    main()
