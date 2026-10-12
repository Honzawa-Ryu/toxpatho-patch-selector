"""評価用manifestの各svsヘッダーからスキャナー情報を抜き出す。

usage: .venv/bin/python scripts/extract_slide_scanner_meta.py [out_csv]
既定の出力: experiments/0010_20260925_pathology_benchmark_subset/label_audit/slide_scanner_meta.csv
"""
import sys
from pathlib import Path

import openslide
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "experiments/0010_20260925_pathology_benchmark_subset/label_audit"
KEYS = ["openslide.vendor", "aperio.ScanScope ID", "aperio.Date", "aperio.AppMag", "aperio.MPP",
        "aperio.Filename", "tiff.ImageDescription"]


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else AUDIT / "slide_scanner_meta.csv"
    m = pd.read_parquet(AUDIT / "manifest_known_unique.parquet")
    rows = []
    for r in m.itertuples():
        p = openslide.OpenSlide(r.svs_path).properties
        rows.append({"slide_id": r.slide_id, "exp_id": r.exp_id, "sacrifice_period": r.sacrifice_period,
                     "dose_level": r.dose_level, **{k: p.get(k) for k in KEYS}})
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
