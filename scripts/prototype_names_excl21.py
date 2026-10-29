"""プロトタイプの名前付けを、脂肪滴系の所見がある21枚のラベルを除いてやり直す（0029 v5の名前はこの21枚のラベルを含めていた）。

規則は0029と同じ（name_prototypes）。出力: outputs/0029_20261028_external_finding_maps/v5_names_excl21/prototype_names_{A..D}.csv と、旧名との差分。
usage: python scripts/prototype_names_excl21.py  （コンテナ内、Slurm経由）
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
MAPS = {"A_sphering_lda": ("tree_v3", "sphering_lda"), "B_sphering": ("tree_v3s", "sphering"), "C_lda": ("tree_v3u", "lda"), "D_unsup": ("tree_v3u", "unsup")}


def main():
    spec = importlib.util.spec_from_file_location("exp0029", ROOT / "experiments/0029_20261028_external_finding_maps/experiment.py")
    ex29 = importlib.util.module_from_spec(spec); sys.modules["exp0029"] = ex29; spec.loader.exec_module(ex29)
    cfg29 = yaml.safe_load(open(ROOT / "experiments/0029_20261028_external_finding_maps/config.yml"))
    cfg20 = yaml.safe_load(open(ROOT / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    prio = cfg20["label_priority"]
    c = np.load(ROOT / "outputs/0025_20261026_context_mixing_search/cache_v1/cache.npz")
    laba, row = c["lab"].astype(np.int64), c["row"]
    keep = [k for k in np.unique(laba) if k >= 0 and (laba == k).sum() >= cfg20["min_label_patches"]]
    laba = np.where(np.isin(laba, keep), laba, -1)
    slides = pd.read_csv(ROOT / "outputs/0025_20261026_context_mixing_search/cache_v1/slides.csv", dtype=str)
    ex_ids = pd.read_csv(ROOT / cfg29["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab = np.where(np.isin(slides.slide_id.to_numpy()[row], ex_ids), -1, laba)
    out = ROOT / "outputs/0029_20261028_external_finding_maps/v5_names_excl21"
    out.mkdir(exist_ok=True)
    na = len(laba)
    for m, (run, rep) in MAPS.items():
        cl = np.load(ROOT / "outputs/0026_20261027_fatty_degeneration_prototypes" / run / "tree" / f"{rep}_k100" / "assign_all.npy").astype(np.int64)[:na]
        new = ex29.name_prototypes(cl, lab, prio, 100, cfg29["naming"])
        old = pd.read_csv(ROOT / "outputs/0029_20261028_external_finding_maps/v5" / f"prototype_names_{m}.csv").name.fillna("")
        new.to_csv(out / f"prototype_names_{m}.csv", index=False)
        ch = (new.name.fillna("") != old.to_numpy())
        print(f"{m}: {int(ch.sum())} of 100 names changed; ", [(int(j), old[j] or "-", new.name.fillna("")[j] or "-") for j in np.where(ch)[0]])
        print("   counts:", new.name.fillna("-").value_counts().to_dict())


if __name__ == "__main__":
    main()
