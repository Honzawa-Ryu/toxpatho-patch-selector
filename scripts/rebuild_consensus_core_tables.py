"""0017のコア表（cores_<method>_tau<tau>.csv）を、保存済みのコア割り当てから作り直す（k-meansは回し直さない）。

2026-10-17: 所見の倍率の選び方を修正したため（experiment.pyのcore_table）。安定性の列は元の表から引き継ぐ。
usage: .venv/bin/python scripts/rebuild_consensus_core_tables.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments/0017_20261016_block_consensus_patterns"
RUN = ROOT / "outputs/0017_20261016_block_consensus_patterns/pilot"
spec = importlib.util.spec_from_file_location("exp0017", EXP / "experiment.py")
exp0017 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp0017)


def main():
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    ev = pd.read_parquet(RUN / "eval_patches.parquet")
    m = pd.read_parquet(ROOT / cfg["manifest_parquet"])
    m["slide_id"] = m.slide_id.astype(str)
    scan = pd.read_csv(ROOT / "experiments/0010_20260925_pathology_benchmark_subset/label_audit/slide_scanner_meta.csv",
                       dtype={"slide_id": str})[["slide_id", "aperio.ScanScope ID"]].rename(columns={"aperio.ScanScope ID": "scanner"})
    slides = m[m.exp_id.isin(cfg["exp_ids"])].merge(scan, on="slide_id", how="left").reset_index(drop=True)
    treated = slides[slides.dose_level != "Control"]
    base_rate = treated.finding_types.explode().dropna().value_counts() / len(treated)
    row = ev.slide_id.map({s: i for i, s in enumerate(slides.slide_id)}).to_numpy()
    e = pd.read_parquet(RUN / "edges.parquet")
    a, b, rate = e.a.to_numpy(), e.b.to_numpy(), e.rate.to_numpy()
    for path in sorted(RUN.glob("cores_*_tau*.csv")):
        col = path.stem.removeprefix("cores_")
        core = ev[col].to_numpy()
        same = (core[a] == core[b]) & (core[a] >= 0)
        n = len(ev)
        cs = np.bincount(a[same], weights=rate[same], minlength=n) + np.bincount(b[same], weights=rate[same], minlength=n)
        cn = np.bincount(a[same], minlength=n) + np.bincount(b[same], minlength=n)
        cent = np.where(cn > 0, cs / np.maximum(cn, 1), 0.0)
        old = pd.read_csv(path)
        new = exp0017.core_table(core, row, slides, cent, base_rate)
        new = new.merge(old[["core", "split_half_jaccard_median"]], on="core", how="left").sort_values("n_patches", ascending=False)
        old.to_csv(path.with_suffix(".before_20261017_fix.csv"), index=False)
        new.to_csv(path, index=False)
        k = new.top_lift_findings.fillna("").apply(lambda s: len([x for x in s.split(";") if x.strip()]))
        print(path.name, "findings per core:", k.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
