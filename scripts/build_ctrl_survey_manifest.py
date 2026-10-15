"""系列C（全反復投与Controlの調査）用のmanifestを作る。

手元の全Liver WSI（tier1の44化合物＋追加Controlの99化合物）から反復投与のControlを取り出し、
病理所見（0010と同じ規則: individual非対応・用量不明はunknown、同一個体の重複画像は除く）と
scanner（svsヘッダー。0010で取得済みの分は再利用）を付ける。

usage: .venv/bin/python scripts/build_ctrl_survey_manifest.py [out_parquet]
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
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments/0015_20261012_ctrl_batch_structure/ctrl_survey_manifest.parquet"
    m = build_liver_slide_manifest(DATA / "TGGATEs/WSI/Liver", DATA / "corrected/open_tggates_pathological_image.csv",
                                   DATA / "corrected/open_tggates_individual.csv")
    m = m[m.sacrifice_period.isin(PERIODS)]
    m = attach_pathology_findings(m, DATA / "corrected/open_tggates_pathology.csv")
    m["slide_id"] = m.slide_id.astype(str)
    ctrl = m[(m.dose_level == "Control") | (m.dose.astype(float) == 0)].sort_values("slide_id").copy()
    ctrl["duplicate_animal_image"] = ctrl.duplicated(KEYS)

    scan = pd.read_csv(SCANNER_CSV, dtype={"slide_id": str})[["slide_id", "aperio.ScanScope ID", "aperio.Date"]]
    ctrl = ctrl.merge(scan, on="slide_id", how="left")
    todo = ctrl["aperio.ScanScope ID"].isna()
    for i in ctrl.index[todo]:   # 0010の表にないslide（追加分など）はヘッダーから読む
        p = openslide.OpenSlide(ctrl.at[i, "svs_path"]).properties
        ctrl.at[i, "aperio.ScanScope ID"] = p.get("aperio.ScanScope ID")
        ctrl.at[i, "aperio.Date"] = p.get("aperio.Date")
    ctrl = ctrl.rename(columns={"aperio.ScanScope ID": "scanner", "aperio.Date": "scan_date"})

    out.parent.mkdir(parents=True, exist_ok=True)
    ctrl.to_parquet(out, index=False)
    print(out)
    print(f"Control slides: {len(ctrl)} / exp {ctrl.exp_id.nunique()} / compounds {ctrl.compound_name.nunique()}")
    print(ctrl.pathology_label_status.value_counts().to_string())
    print("duplicate animal images:", int(ctrl.duplicate_animal_image.sum()))
    print("scanner read from header:", int(todo.sum()))
    print(ctrl.scanner.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
