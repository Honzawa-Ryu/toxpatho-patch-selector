"""プロトタイプとBayerのラベルの対応を、NMIを「分かれ」と「混ざり」に分けて表にする（2026-10-26ユーザー依頼）。

全体: NMI（算術平均＝V-measure）= homogeneity（各クラスタの中身が1ラベルか）と completeness（各ラベルが1クラスタにまとまるか）の調和平均。
ラベルごと:
  k80    : そのラベルのpatchの80%を含むのに要るクラスタの数（広がり。少ないほどまとまる）
  top1   : 最も多く入った1クラスタの割合（これまでの「集中度」）
  purity : そのラベルのpatchが入ったクラスタでの、同じラベルの割合を、patchの数で重みづけした平均（混ざりの逆。1なら他と混ざらない）
いずれもBayerのラベル付きpatchだけで計算する。

対象:
  0021 v3_maps（unsup / weak / combo、約23万patch）: label_concentration_<版>.csv（ラベルごとの行和1）× ラベルのpatch数（0020 compare_132と同じ集合）
  0021 explore_v2（self_only / w05_h4 / ctx_only_h4、132枚の全patch）: assign_<版>.npy と context_build_v2 のラベル
出力: outputs/prototype_label_mixing_20261026/{overall.csv, per_label.csv}

usage: python scripts/prototype_label_mixing.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import completeness_score, homogeneity_score, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/0021_20261025_patch_prototypes"
OUT = ROOT / "outputs/prototype_label_mixing_20261026"


def from_crosstab(C):
    """ラベル×クラスタの件数の表から、ラベルとクラスタの組を展開する（sklearnの指標は件数で重みづけと同じになる）。"""
    li, ci = np.nonzero(C)
    n = C[li, ci].astype(int)
    return np.repeat(li, n), np.repeat(ci, n)


def per_label(C, names):
    pur = C / np.maximum(C.sum(0, keepdims=True), 1)          # クラスタ内のラベルの割合
    rows = []
    for i, nm in enumerate(names):
        r = C[i]
        if r.sum() == 0:
            continue
        s = np.sort(r)[::-1].cumsum() / r.sum()
        rows.append({"label": nm, "n": int(r.sum()), "k80": int(np.searchsorted(s, 0.8) + 1), "top1": float(s[0]),
                     "purity": float((r * pur[i]).sum() / r.sum())})
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg20 = yaml.safe_load(open(ROOT / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    prio = cfg20["label_priority"]
    counts = json.load(open(ROOT / "outputs/0020_20261024_batch_robust_methods/compare_132/results.json"))["labels"]
    overall, perl = [], []

    def add(group, rep, C, names):
        y, c = from_crosstab(C)
        overall.append({"group": group, "rep": rep, "n_labeled": len(y), "nmi": normalized_mutual_info_score(y, c),
                        "homogeneity": homogeneity_score(y, c), "completeness": completeness_score(y, c)})
        perl.extend({"group": group, "rep": rep, **r} for r in per_label(C, names))

    for rep in ["unsup", "weak", "combo"]:
        f = pd.read_csv(SRC / f"v3_maps/label_concentration_{rep}.csv", index_col=0)
        C = np.round(f.to_numpy() * np.array([counts[l] for l in f.index])[:, None])
        add("v3_maps (all 2,361 slides, ~230k patches)", rep, C, list(f.index))

    c = np.load(SRC / "context_build_v2/context_cache.npz")
    lab = c["lab"]
    keep = [k for k in np.unique(lab) if k >= 0 and (lab == k).sum() >= cfg20["min_label_patches"]]
    m = np.isin(lab, keep)
    for rep in ["self_only", "w05_h4", "ctx_only_h4"]:
        cl = np.load(SRC / f"explore_v2/assign_{rep}.npy")
        C = np.zeros((len(keep), cl.max() + 1))
        np.add.at(C, (np.searchsorted(keep, lab[m]), cl[m]), 1)
        add("explore_v2 (132 annotated slides, all patches)", rep, C, [prio[k] for k in keep])

    o, p = pd.DataFrame(overall), pd.DataFrame(perl)
    o.to_csv(OUT / "overall.csv", index=False)
    p.to_csv(OUT / "per_label.csv", index=False)
    print(o.round(3).to_string(index=False))
    print(p.pivot_table(index="label", columns="rep", values=["k80", "purity"]).round(2).to_string())


if __name__ == "__main__":
    main()
