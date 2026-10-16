"""反復投与・LiverのControl slideのうち、手元にないものの一覧を作る（scripts/download_tggates_liver_wsi.py --slide-list 用）。

背景地図用にControlの試験バッチを増やす（2026-10-13 ユーザー決定: 反復投与のControlを全化合物に広げる）。
DOSE_LEVELはindividual表から取る（download_tggates_liver_wsi.build_targets と同じ結合）。

--exps を渡すと、その試験の全用量（Control＋投与群）のうち手元にないものを一覧にする
（2026-10-16: 系列Cの最大の塊の投与群を転送するため）。

usage: .venv/bin/python scripts/build_ctrl_slide_list.py [--exps 1,2,3] [--name ctrl_repeat]
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIVER_DIR = ROOT / "data/TGGATEs/WSI/Liver"
FTP_PREFIX = "ftp://ftp.biosciencedbc.jp/archive/"
HTTPS_PREFIX = "https://dbarchive.biosciencedbc.jp/data/"
PERIODS = ["4 day", "8 day", "15 day", "29 day"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", default="", help="対象のexp_id（カンマ区切り）。指定時は全用量")
    ap.add_argument("--name", default="ctrl_repeat")
    args = ap.parse_args()
    out = ROOT / "outputs/target_slide_list" / f"{args.name}_missing.csv"
    img = pd.read_csv(ROOT / "data/corrected/open_tggates_pathological_image.csv")
    img = img[(img.ORGAN == "Liver") & (img.SINGLE_REPEAT_TYPE == "Repeat") & img.SACRIFICE_PERIOD.isin(PERIODS)].copy()
    ind = pd.read_csv(ROOT / "data/corrected/open_tggates_individual.csv")
    img = img.merge(ind[["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID", "DOSE_LEVEL"]],
                    on=["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"], how="left")
    img["slide_id"] = img.FILE_LOCATION.apply(lambda u: Path(str(u)).stem)
    local = {p.stem for p in LIVER_DIR.rglob("*.svs")}
    local_compounds = set(img.loc[img.slide_id.isin(local), "COMPOUND_NAME"])

    # individual表に対応がない画像はDOSEで補う（0010監査: individual非対応の画像がある）
    if args.exps:
        ctrl = img[img.EXP_ID.isin([int(e) for e in args.exps.split(",")])]
    else:
        ctrl = img[(img.DOSE_LEVEL == "Control") | (img.DOSE_LEVEL.isna() & (img.DOSE == 0))]
    tgt = ctrl[~ctrl.slide_id.isin(local)].drop_duplicates("slide_id").copy()
    tgt["url"] = tgt.FILE_LOCATION.str.replace(FTP_PREFIX, HTTPS_PREFIX, regex=False)
    if not tgt.url.str.startswith(HTTPS_PREFIX).all():
        raise SystemExit("想定外のFILE_LOCATION形式があります")
    tgt["compound_in_corpus"] = tgt.COMPOUND_NAME.isin(local_compounds)
    tgt["dose_level_from_individual"] = tgt.DOSE_LEVEL.notna()
    cols = ["slide_id", "COMPOUND_NAME", "EXP_ID", "GROUP_ID", "INDIVIDUAL_ID", "SACRIFICE_PERIOD", "DOSE", "DOSE_LEVEL",
            "compound_in_corpus", "dose_level_from_individual", "url"]
    tgt = tgt[cols].rename(columns={"SACRIFICE_PERIOD": "sacrifice_period"}).sort_values(["COMPOUND_NAME", "sacrifice_period", "slide_id"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tgt.to_csv(out, index=False)
    # 別マシンから `rsync --files-from` で手動転送する用（build_target_slide_list.pyと同じ形式）。
    # Liver/ を起点とした相対パス、1ファイル1行（転送1セッション・パスワード入力1回）。
    rel = sorted(f"{p.replace(' ', '_')}/{sid}.svs" for p, sid in zip(tgt.sacrifice_period, tgt.slide_id))
    (out.parent / f"rsync_files_{args.name}.txt").write_text("\n".join(rel) + "\n")
    print(out)
    print(out.parent / f"rsync_files_{args.name}.txt", len(rel), "行")
    print(tgt.DOSE_LEVEL.value_counts(dropna=False).to_string())
    print(f"missing: {len(tgt)} slides / {tgt.COMPOUND_NAME.nunique()} compounds / {tgt.EXP_ID.nunique()} exp")
    print(tgt.groupby("compound_in_corpus").agg(slides=("slide_id", "size"), compounds=("COMPOUND_NAME", "nunique")).to_string())
    print(tgt.groupby("sacrifice_period").size().to_string())
    print("individual非対応（DOSE=0で補完）:", int((~tgt.dose_level_from_individual).sum()))


if __name__ == "__main__":
    main()
