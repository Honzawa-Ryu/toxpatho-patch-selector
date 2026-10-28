"""0030（学習に使っていないTG-GATEsのWSI）の所見分類に、化合物を復元抽出した95%区間を付ける。A〜Dの差（特にC−D）も。

読み出し（ロジスティック回帰、化合物単位5分割）は0030と同じ設定で作り直し、学習なし（名前付きプロトタイプの割合）も同じ定義。
出力: outputs/0030_20261028_tggates_unseen_findings/v1/bootstrap/
usage: python scripts/tggates_unseen_bootstrap.py  （コンテナ内、Slurm経由）
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/0030_20261028_tggates_unseen_findings/v1"
OUT = RUN / "bootstrap"
MAPS = ["D_unsup", "C_lda", "B_sphering", "A_sphering_lda"]
N_BOOT = 2000


def main():
    OUT.mkdir(exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "experiments/0030_20261028_tggates_unseen_findings/config.yml"))
    sl = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str})
    ctrl = sl.DOSE_LEVEL.eq("Control").to_numpy()
    comp = sl.COMPOUND_NAME.to_numpy()
    deltas, names = {}, {}
    for m in MAPS:
        sh = np.load(RUN / f"shares_{m}.npy")
        cm = pd.DataFrame(sh[ctrl]).groupby(sl.EXP_ID.to_numpy()[ctrl]).mean()
        base = cm.reindex(sl.EXP_ID).to_numpy()
        deltas[m] = sh - np.where(np.isnan(base), sh[ctrl].mean(0), base)
        names[m] = pd.read_csv(ROOT / "outputs/0029_20261028_external_finding_maps/v5" / f"prototype_names_{m}.csv").set_index("prototype").name.fillna("").reindex(range(sh.shape[1])).to_numpy()
    scores = {}   # (finding, kind, map) -> 各slideのスコア（読み出しは化合物単位の分割の外側予測）
    for f, blab in cfg["findings"].items():
        y = (sl[f"grade__{f}"].to_numpy() > 0).astype(int)
        npc = len(set(comp[y == 1]))
        for m in MAPS:
            oof = np.full(len(sl), np.nan)
            for tr, te in GroupKFold(n_splits=min(cfg["cv_splits"], npc)).split(deltas[m], y, comp):
                if y[tr].sum():
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
                    oof[te] = clf.fit(deltas[m][tr], y[tr]).predict_proba(deltas[m][te])[:, 1]
            scores[(f, "readout", m)] = oof
            if blab and (names[m] == blab).any():
                scores[(f, "named", m)] = deltas[m][:, names[m] == blab].sum(1)
    comps = np.unique(comp)
    idx = {c: np.where(comp == c)[0] for c in comps}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([idx[c] for c in rng.choice(comps, len(comps))]) for _ in range(N_BOOT)]

    def auc(y, s, ii=None):
        ii = np.arange(len(y)) if ii is None else ii
        ok = ii[~np.isnan(s[ii])]
        return roc_auc_score(y[ok], s[ok]) if len(set(y[ok])) > 1 else np.nan

    rows, diffs = [], []
    for (f, kind, m), s in scores.items():
        y = (sl[f"grade__{f}"].to_numpy() > 0).astype(int)
        b = np.array([auc(y, s, ii) for ii in boots])
        rows.append({"finding": f, "bayer_label": cfg["findings"][f] or "(none)", "kind": kind, "map": m, "AUC": auc(y, s), "lo": np.nanpercentile(b, 2.5), "hi": np.nanpercentile(b, 97.5)})
    for f, blab in cfg["findings"].items():
        y = (sl[f"grade__{f}"].to_numpy() > 0).astype(int)
        for kind in ["readout", "named"]:
            for a, bb in [("C_lda", "D_unsup"), ("A_sphering_lda", "B_sphering"), ("A_sphering_lda", "D_unsup")]:
                if (f, kind, a) not in scores:
                    continue
                if (f, kind, bb) not in scores:   # ラベルなしの側に名前の付いたプロトタイプがない: そのプロトタイプでは当てられない（AUC 0.5扱い）
                    sb = np.zeros(len(sl))
                else:
                    sb = scores[(f, kind, bb)]
                d = np.array([auc(y, scores[(f, kind, a)], ii) - auc(y, sb, ii) for ii in boots])
                diffs.append({"finding": f, "bayer_label": blab or "(none)", "kind": kind, "comparison": f"{a} - {bb}",
                              "diff": auc(y, scores[(f, kind, a)]) - auc(y, sb), "lo": np.nanpercentile(d, 2.5), "hi": np.nanpercentile(d, 97.5), "p_le0": float(np.nanmean(d <= 0))})
    res, diffs = pd.DataFrame(rows), pd.DataFrame(diffs)
    # Bayerのラベルがある5所見の平均の差（同じ復元抽出で）
    fl = [f for f, b in cfg["findings"].items() if b]
    for kind in ["readout", "named"]:
        for a, bb in [("C_lda", "D_unsup"), ("A_sphering_lda", "D_unsup")]:
            fs = [f for f in fl if (f, kind, a) in scores]
            def mean_diff(ii):
                vals = []
                for f in fs:
                    y = (sl[f"grade__{f}"].to_numpy() > 0).astype(int)
                    sb = scores.get((f, kind, bb), np.zeros(len(sl)))
                    vals.append(auc(y, scores[(f, kind, a)], ii) - auc(y, sb, ii))
                return np.nanmean(vals)
            d = np.array([mean_diff(ii) for ii in boots])
            diffs = pd.concat([diffs, pd.DataFrame([{"finding": f"mean of {len(fs)} Bayer-labeled findings", "bayer_label": "", "kind": kind, "comparison": f"{a} - {bb}",
                                                     "diff": mean_diff(np.arange(len(sl))), "lo": np.nanpercentile(d, 2.5), "hi": np.nanpercentile(d, 97.5), "p_le0": float(np.nanmean(d <= 0))}])])
    res.to_csv(OUT / "auc_ci.csv", index=False)
    diffs.to_csv(OUT / "auc_differences.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print(diffs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
