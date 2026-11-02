"""0035 のプロトタイプに名前を付ける（2026-11-02ユーザー: 「これって所見名ついてる？」）。2 種類の名前を並べる。

  bayer_name  : Bayer のポリゴンのラベル（0029 と同じ規則: 割合 >= 0.2 か倍率 >= 3、ラベル付き patch 20 以上）。脂肪変性にはラベルがないので付かない
  record_name : TG-GATEs の病理記録（11 所見）。プロトタイプの patch のうち「その所見の記録がある slide 由来」の割合 share が基準率の min_lift 倍以上、かつ share >= min_share、patch 数 >= min_n のとき、倍率が最大の所見。
                記録は slide 単位なので、名前は「その所見がある slide に偏って出るプロトタイプ」の意味（patch がその所見とは限らない）
出力: outputs/0035_.../<版>/prototype_names_<rep>_<method>.csv（全プロトタイプ）。脂肪変性の純プロトタイプ（fatty_pure_prototypes.csv）には名前を併記して表示する。
usage: python scripts/prototype_record_names.py [版] [rep:method ...]  （コンテナ内、Slurm経由。CPU、数分）
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_RUN = sys.argv[1] if len(sys.argv) > 1 else "v1"
PAIRS = [a.split(":") for a in sys.argv[2:]] or [["A128_sphering_lda", "split_record"], ["A128_sphering_lda", "flat_k1000"]]
EXP = ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin"
RUN = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin" / _RUN
RECORD_RULE = {"min_share": 0.5, "min_lift": 3.0, "min_n": 200}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    ex29 = load_module(ROOT / "experiments" / cfg["exp0029"] / "experiment.py", "exp0029")
    ex35 = load_module(EXP / "experiment.py", "exp0035")
    cfg20 = yaml.safe_load(open(ROOT / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    prio = cfg20["label_priority"]
    findings = list(cfg["findings"])
    # Bayer のラベル（0035 と同じ: 30 patch 未満のラベルと 21 枚のラベルを除く）
    src = ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    laba, row_b = c["lab"].astype(np.int64), c["row"]
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    keep = [k_ for k_ in np.unique(laba) if k_ >= 0 and (laba == k_).sum() >= cfg20["min_label_patches"]]
    ex_ids = pd.read_csv(ROOT / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    laba = np.where(np.isin(laba, keep) & ~np.isin(bayer.slide_id.to_numpy()[row_b], ex_ids), laba, -1)
    nb = len(laba)
    # 全 slide の記録と patch → slide（0035 の slides.csv と、各集合の patch 数）
    S = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str})
    pos = np.stack([S[f"grade__{f}"].to_numpy() > 0 for f in findings], 1)
    n_set = S.set.value_counts()
    nf = int(n_set["fatty"]) * cfg["patches_per_slide"]
    nu = int(n_set["unseen"]) * cfg["patches_per_slide"]
    z = np.load(ROOT / cfg["unseen_features_from"], allow_pickle=True)
    row_u = z["row"]
    assert len(row_u) == nu
    # 脂肪変性の slide の patch は各 2,048（0030 の _load_sub は patch 数が 2,048 以上の slide を 2,048 に間引く）
    row_t = np.repeat(np.arange(int(n_set["fatty"])), cfg["patches_per_slide"])
    off_t = int((S.set == "bayer").sum())
    off_u = off_t + int(n_set["fatty"])
    pure_p = RUN / "fatty_pure_prototypes.csv"
    pure = pd.read_csv(pure_p) if pure_p.exists() else pd.DataFrame(columns=["rep", "method", "compound", "prototype", "n", "purity"])   # 表示用。なければ省く
    for rep, method in PAIRS:
        cl = np.load(RUN / f"assign_{rep}_{method}.npy").astype(np.int64)
        assert len(cl) == nb + nf + nu, (len(cl), nb, nf, nu)
        slide_of = np.r_[row_b, row_t + off_t, row_u + off_u]
        K = int(cl.max()) + 1
        nm = ex29.name_prototypes(cl[:nb], laba, prio, K, cfg["naming"]).set_index("prototype")
        cs = ex35.slide_counts(cl, K, slide_of, len(S))
        share, base, lift, njf, n = ex35.record_shares(cs, pos)
        rows = []
        for j in range(K):
            ok = (n[j] >= RECORD_RULE["min_n"]) & (share[j] >= RECORD_RULE["min_share"]) & (lift[j] >= RECORD_RULE["min_lift"])
            f_ = int(np.argmax(np.where(ok, lift[j], -1))) if ok.any() else -1
            rows.append({"prototype": j, "n_patches": int(n[j]), "bayer_name": nm.name.get(j, "") if isinstance(nm.name.get(j, ""), str) else "", "bayer_top_label": nm.top_label.get(j, ""), "n_labeled_bayer": int(nm.n_labeled.get(j, 0)),
                         "record_name": findings[f_] if f_ >= 0 else "", "record_share": float(share[j, f_]) if f_ >= 0 else np.nan, "record_lift": float(lift[j, f_]) if f_ >= 0 else np.nan,
                         **{f"share__{f}": float(share[j, i]) for i, f in enumerate(findings)}})
        tab = pd.DataFrame(rows)
        tab.to_csv(RUN / f"prototype_names_{rep}_{method}.csv", index=False)
        named = tab[tab.record_name != ""]
        print(f"\n== {rep} {method}: {K} prototypes; Bayer-named {int((tab.bayer_name != '').sum())}, record-named {len(named)}")
        print(named.record_name.value_counts().to_string())
        pp = pure[(pure.rep == rep) & (pure.method == method)].merge(tab[["prototype", "bayer_name", "bayer_top_label", "record_name", "record_share", "record_lift"]], on="prototype")
        pd.set_option("display.width", 220)
        print(pp[["compound", "prototype", "n", "purity", "bayer_name", "bayer_top_label", "record_name", "record_share", "record_lift"]].round(2).to_string(index=False))
    print("done", flush=True)


if __name__ == "__main__":
    main()
