"""0030（学習に使っていないTG-GATEsのWSI）の「学習なし」AUCを、学習との重なりの層ごとに出す（2026-10-29ユーザー: 核分裂像の精度が高すぎる、試験のリークはないか）。

層: all / bayer_exp（Bayerの試験の別slide）/ new_exp（試験は新しいが化合物はBayerにありうる）/ new_compound（化合物ごとBayerにない）。
スコア = Bayerのラベルの名前が付いたプロトタイプの割合 − 同じ試験の対照群の平均（TG-GATEsでは何も学習しない）。
出力: outputs/0030_20261028_tggates_unseen_findings/v1/strict/
usage: python scripts/tggates_unseen_strict.py  [版]  （コンテナ内、Slurm経由。版の既定は上記）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/0030_20261028_tggates_unseen_findings" / _RUN
OUT = RUN / "strict"
_RUN = sys.argv[1] if len(sys.argv) > 1 else "v4"
_CFG = yaml.safe_load(open(ROOT / "experiments/0030_20261028_tggates_unseen_findings/config.yml"))
_CFG = {**_CFG, **_CFG.get("variants", {}).get(_RUN, {})}
NAMES = _CFG["names_from"]
MAPS = list(_CFG["maps"])
_MAPS_OLD = ["D_unsup", "C_lda", "B_sphering", "A_sphering_lda"]


def main():
    OUT.mkdir(exist_ok=True)
    cfg = _CFG
    findings = {f: b for f, b in cfg["findings"].items() if b}
    sl = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str})
    bayer = pd.read_csv(ROOT / "outputs/0025_20261026_context_mixing_search/cache_v1/slides.csv", dtype=str)
    bcomp = set(bayer.COMPOUND_NAME.dropna())
    sl["bayer_compound"] = sl.COMPOUND_NAME.isin(bcomp)
    ctrl = sl.DOSE_LEVEL.eq("Control").to_numpy()
    strata = {"all": np.ones(len(sl), bool), "bayer_exp": sl.bayer_exp.to_numpy(), "new_exp": ~sl.bayer_exp.to_numpy(), "new_compound": ~sl.bayer_compound.to_numpy()}
    score = {}
    for m in MAPS:
        sh = np.load(RUN / f"shares_{m}.npy")
        base = pd.DataFrame(sh[ctrl]).groupby(sl.EXP_ID.to_numpy()[ctrl]).mean().reindex(sl.EXP_ID).to_numpy()
        delta = sh - np.where(np.isnan(base), sh[ctrl].mean(0), base)
        names = pd.read_csv(ROOT / "outputs/0029_20261028_external_finding_maps" / NAMES / f"prototype_names_{m}.csv").set_index("prototype").name.fillna("").reindex(range(sh.shape[1])).to_numpy()
        for f, b in findings.items():
            score[(f, m)] = delta[:, names == b].sum(1) if (names == b).any() else None
    rows = []
    for f in findings:
        y = (sl[f"grade__{f}"].to_numpy() > 0).astype(int)
        for sname, smask in strata.items():
            n_pos, n_comp = int(y[smask].sum()), len(set(sl.COMPOUND_NAME[smask & (y == 1)]))
            r = {"finding": f, "stratum": sname, "n_slides": int(smask.sum()), "n_pos": n_pos, "n_pos_compounds": n_comp,
                 "pos_compounds": "; ".join(sorted(set(sl.COMPOUND_NAME[smask & (y == 1)])))}
            for m in MAPS:
                s = score[(f, m)]
                r[m] = roc_auc_score(y[smask], s[smask]) if s is not None and n_pos and (y[smask] == 0).any() else np.nan
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "named_auc_by_stratum.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.drop(columns="pos_compounds").round(3).to_string(index=False))
    print(res[["finding", "stratum", "pos_compounds"]].to_string(index=False))
    # Bayerの核分裂像ラベルの由来（化合物）と、核分裂像の記録がある化合物の重なり
    c = np.load(ROOT / "outputs/0025_20261026_context_mixing_search/cache_v1/cache.npz")
    prio = yaml.safe_load(open(ROOT / "experiments/0020_20261024_batch_robust_methods/config.yml"))["label_priority"]
    lab, row = c["lab"].astype(int), c["row"]
    for b in ["mitosis", "necrosis", "inflammation", "single_cell_necrosis", "emh"]:
        bc = pd.Series(bayer.COMPOUND_NAME.to_numpy()[row[lab == prio.index(b)]]).value_counts()
        f = [k for k, v in findings.items() if v == b][0]
        pos_c = set(sl.COMPOUND_NAME[sl[f"grade__{f}"] > 0])
        print(f"\n{b}: Bayer-labeled patches {bc.sum():,} from {len(bc)} compounds; share from compounds that have '{f}' recorded in the 0030 set: {bc[bc.index.isin(pos_c)].sum() / bc.sum():.2f}")


if __name__ == "__main__":
    main()
