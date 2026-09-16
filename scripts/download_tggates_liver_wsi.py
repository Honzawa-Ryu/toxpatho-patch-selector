"""Download missing Open TG-GATEs Liver WSIs into data/TGGATEs/WSI/Liver/<N>_day/.

対象は「ローカルに既にあるスライドと同じ化合物」の Liver WSI のうち、指定した
反復投与時点・用量レベルでまだ手元に無いもの。ローカルの .svs を数えて化合物集合と
取得済みIDを決めるので、0001 の出力（slide_manifest.parquet）には依存しない。

ホストについて:
    メタデータCSVの FILE_LOCATION は ftp://ftp.biosciencedbc.jp/archive/... を指すが、
    このホストは既に名前解決できない（2026-09時点）。同じアーカイブが
    https://dbarchive.biosciencedbc.jp/data/... で配信されており、こちらは Range
    リクエストに対応しているためレジューム可能。URLのパス部分（化合物名は
    "1%25_cholesterol_%2B_0.25%25_sodium_cholate" のように既にURLエンコード済み）は
    そのまま流用し、スキームとホストと先頭ディレクトリだけ差し替える。

Usage:
    python scripts/download_tggates_liver_wsi.py [--dose-levels High Low Middle]
                                                 [--periods "4 day" ...]
                                                 [--workers 4] [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVER_DIR = PROJECT_ROOT / "data" / "TGGATEs" / "WSI" / "Liver"
IMAGE_CSV = PROJECT_ROOT / "data" / "corrected" / "open_tggates_pathological_image.csv"
INDIVIDUAL_CSV = PROJECT_ROOT / "data" / "corrected" / "open_tggates_individual.csv"

FTP_PREFIX = "ftp://ftp.biosciencedbc.jp/archive/"
HTTPS_PREFIX = "https://dbarchive.biosciencedbc.jp/data/"

# TIFFのマジックナンバー。SVSはTIFF系コンテナなので、落としたファイルが
# HTMLのエラーページ等でないことの最低限の確認に使う。
TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")

_print_lock = Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def targets_from_slide_list(slide_list: Path) -> pd.DataFrame:
    """Read an explicit slide list (from scripts/build_target_slide_list.py).

    「手元にある化合物」から逆算する build_targets() と違い、こちらは取りに行く
    スライドを外から与える。取得を別マシンで行う場合や、まだ1枚も持っていない
    化合物を新たに入れる場合はこちらを使う。
    """
    table = pd.read_parquet(slide_list) if slide_list.suffix == ".parquet" else pd.read_csv(slide_list)
    required = {"slide_id", "sacrifice_period", "url"}
    missing_columns = required - set(table.columns)
    if missing_columns:
        raise SystemExit(f"{slide_list} に必要な列がありません: {sorted(missing_columns)}")

    table = table.copy()
    table["slide_id"] = table["slide_id"].astype(str)
    local_ids = {p.stem for p in LIVER_DIR.rglob("*.svs")}
    table = table[~table["slide_id"].isin(local_ids)]
    table["dest"] = [
        LIVER_DIR / str(period).replace(" ", "_") / f"{sid}.svs"
        for period, sid in zip(table["sacrifice_period"], table["slide_id"])
    ]
    return table.drop_duplicates("slide_id").reset_index(drop=True)


def build_targets(dose_levels: list[str], periods: list[str]) -> pd.DataFrame:
    """まだ手元に無いスライドの一覧（slide_id, url, dest）を作る。"""
    local_paths = sorted(LIVER_DIR.rglob("*.svs"))
    local_ids = {p.stem for p in local_paths}
    if not local_ids:
        raise SystemExit(f"ローカルに .svs が1枚も見つかりません: {LIVER_DIR}")

    img = pd.read_csv(IMAGE_CSV)
    img = img[img["ORGAN"] == "Liver"].copy()
    img["slide_id"] = img["FILE_LOCATION"].apply(lambda u: Path(str(u)).stem)

    compounds = sorted(set(img.loc[img["slide_id"].isin(local_ids), "COMPOUND_NAME"]))
    log(f"ローカル {len(local_ids)} 枚 / 化合物 {len(compounds)} 種: {compounds}")

    ind = pd.read_csv(INDIVIDUAL_CSV)
    img = img.merge(
        ind[["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID", "DOSE_LEVEL"]],
        on=["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"],
        how="left",
    )

    tgt = img[
        img["COMPOUND_NAME"].isin(compounds)
        & img["SACRIFICE_PERIOD"].isin(periods)
        & img["DOSE_LEVEL"].isin(dose_levels)
        & ~img["slide_id"].isin(local_ids)
    ].copy()

    tgt = tgt.drop_duplicates("slide_id")
    tgt["url"] = tgt["FILE_LOCATION"].str.replace(FTP_PREFIX, HTTPS_PREFIX, regex=False)

    unexpected = tgt.loc[~tgt["url"].str.startswith(HTTPS_PREFIX), "FILE_LOCATION"]
    if not unexpected.empty:
        raise SystemExit(
            f"想定外のFILE_LOCATION形式が {len(unexpected)} 件あります: "
            f"{unexpected.head().tolist()}"
        )

    tgt["dest"] = [
        LIVER_DIR / period.replace(" ", "_") / f"{sid}.svs"
        for period, sid in zip(tgt["SACRIFICE_PERIOD"], tgt["slide_id"])
    ]
    return tgt.sort_values(["SACRIFICE_PERIOD", "DOSE_LEVEL", "slide_id"]).reset_index(drop=True)


def remote_size(url: str, timeout: int = 60) -> int | None:
    """Content-Length を取りに行く。取れなければ None（サイズ検証をスキップする）。"""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            length = r.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, ValueError):
        return None


def download_one(url: str, dest: Path, retries: int = 5, timeout: int = 120) -> tuple[str, str]:
    """1枚落とす。途中まで落ちている .part があれば Range で続きから取る。

    Returns:
        (slide_id, status) の組。status は "done" / "skipped" / "failed: <理由>"。
    """
    slide_id = dest.stem
    if dest.exists():
        return slide_id, "skipped"

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    expected = remote_size(url)

    for attempt in range(1, retries + 1):
        pos = part.stat().st_size if part.exists() else 0
        if expected is not None and pos > expected:
            # 前回の中断でおかしなサイズになっている場合は取り直す
            part.unlink()
            pos = 0
        if expected is not None and pos == expected:
            break

        headers = {"Range": f"bytes={pos}-"} if pos else {}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(
                part, "ab" if pos else "wb"
            ) as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            break
        except urllib.error.HTTPError as e:
            if e.code == 416 and expected is not None and part.exists() and part.stat().st_size == expected:
                break  # 既に全部落ちている
            if attempt == retries:
                return slide_id, f"failed: HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                return slide_id, f"failed: {type(e).__name__} {e}"
        time.sleep(min(60, 5 * 2 ** (attempt - 1)))

    if not part.exists():
        return slide_id, "failed: no data"

    size = part.stat().st_size
    if expected is not None and size != expected:
        return slide_id, f"failed: size {size} != {expected}"
    with open(part, "rb") as f:
        if not f.read(4).startswith(TIFF_MAGIC):
            return slide_id, "failed: not a TIFF/SVS"

    part.rename(dest)
    return slide_id, "done"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dose-levels", nargs="+", default=["High", "Low", "Middle"])
    parser.add_argument(
        "--periods", nargs="+", default=["4 day", "8 day", "15 day", "29 day"]
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="先頭N枚だけ落とす（0で全件）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--slide-list", type=Path, default=None,
        help="取得するスライドを明示したparquet/csv（scripts/build_target_slide_list.py の出力）。"
             "指定すると --dose-levels / --periods は無視される。",
    )
    args = parser.parse_args()

    if args.slide_list is not None:
        tgt = targets_from_slide_list(args.slide_list)
        log(f"スライドリスト: {args.slide_list}")
    else:
        tgt = build_targets(args.dose_levels, args.periods)
    if args.limit:
        tgt = tgt.head(args.limit)
    log(f"\n未取得のスライド: {len(tgt)} 枚")
    if len(tgt):
        period_col = "SACRIFICE_PERIOD" if "SACRIFICE_PERIOD" in tgt else "sacrifice_period"
        dose_col = "DOSE_LEVEL" if "DOSE_LEVEL" in tgt else "dose_level"
        log(pd.crosstab(tgt[period_col], tgt[dose_col]).to_string())
    else:
        log("(なし)")

    if args.dry_run:
        log("\n--dry-run のため、先頭5件のURLだけ表示して終了します")
        for _, row in tgt.head().iterrows():
            log(f"  {row['url']}\n    -> {row['dest']}")
        probe = remote_size(tgt["url"].iloc[0]) if len(tgt) else None
        log(f"\n疎通確認 (HEAD Content-Length): {probe}")
        return 0

    if tgt.empty:
        return 0

    counts = {"done": 0, "skipped": 0, "failed": 0}
    failures: list[tuple[str, str]] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, row["url"], row["dest"]): row["slide_id"]
            for _, row in tgt.iterrows()
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            slide_id, status = fut.result()
            key = "failed" if status.startswith("failed") else status
            counts[key] += 1
            if key == "failed":
                failures.append((slide_id, status))
            elapsed = time.time() - started
            log(
                f"[{i}/{len(tgt)}] {slide_id}: {status} "
                f"(done={counts['done']} skipped={counts['skipped']} "
                f"failed={counts['failed']} elapsed={elapsed / 60:.1f}min)"
            )

    log(f"\n完了: {counts}")
    if failures:
        log("失敗した slide_id:")
        for slide_id, status in failures:
            log(f"  {slide_id}: {status}")
        log("再実行すれば取得済みはスキップされ、失敗分だけ再試行されます。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
