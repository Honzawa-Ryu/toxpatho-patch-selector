"""系列Cの塊1つ分（全用量）のslide manifestを作る。所見は0010と同じ規則（individual非対応・用量不明はunknown、同一個体の重複は印）。

0017で塊全体（手元にない試験の投与群を転送した後）を扱うため（Issue #11）。scannerは0010の表＋svsヘッダー。
usage: .venv/bin/python scripts/build_block_manifest.py <exp_blocks.csv> <block> <out_parquet>
"""
import sys
from pathlib import Path

import openslide
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.tggates_metadata import attach_pathology_findings, build_liver_slide_manifest  # noqa: E402

DATA = ROOT / "data"
SCANNER_CSV = ROOT / "experiments/0010_20260925_pathology_benchmark_subset/label_audit/slide_scanner_meta.csv"
PERIODS = ["4 day", "8 day", "15 day", "29 day"]
KEYS = ["exp_id", "group_id", "individual_id"]


def main():
    blocks, block, out = pd.read_csv(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    exps = set(blocks[blocks.block == block].exp_id)
    m = build_liver_slide_manifest(DATA / "TGGATEs/WSI/Liver", DATA / "corrected/open_tggates_pathological_image.csv",
                                   DATA / "corrected/open_tggates_individual.csv")
    m = m[m.sacrifice_period.isin(PERIODS) & m.exp_id.isin(exps)]
    m = attach_pathology_findings(m, DATA / "corrected/open_tggates_pathology.csv")
    m["slide_id"] = m.slide_id.astype(str)
    m = m.sort_values("slide_id").copy()
    m["duplicate_animal_image"] = m.duplicated(KEYS)
    scan = pd.read_csv(SCANNER_CSV, dtype={"slide_id": str})[["slide_id", "aperio.ScanScope ID"]]
    m = m.merge(scan, on="slide_id", how="left").rename(columns={"aperio.ScanScope ID": "scanner"})
    for i in m.index[m.scanner.isna()]:
        m.at[i, "scanner"] = openslide.OpenSlide(m.at[i, "svs_path"]).properties.get("aperio.ScanScope ID")
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)
    print(out)
    print(f"{len(m)} slides / {m.exp_id.nunique()} exp (block {block}: {len(exps)} exp)")
    print(m.groupby("dose_level").size().to_string())
    print(m.pathology_label_status.value_counts().to_string())
    print("duplicate animal images:", int(m.duplicate_animal_image.sum()))
    missing = sorted(exps - set(m.exp_id))
    print("experiments without local slides:", missing)


if __name__ == "__main__":
    main()
