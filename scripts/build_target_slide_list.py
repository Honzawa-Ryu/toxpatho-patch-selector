"""Pick the TG-GATEs Liver slides worth downloading, and say why.

これまでの実験で律速だったのは**枚数ではなく化合物数**だった。所見特異的な視覚トークンを
主張するには「同じ所見を出す別の化合物でも同じクラスタが立つ」ことを示す必要があり、
それは1化合物あたりのスライドを増やしても達成できない。

そこでこのスクリプトは、増やす対象を所見（正確には所見×部位）から逆算する:

1. 反復投与・Liver・投与由来（SP_FLG != true）の所見を、`FINDING_TYPE @ TOPOGRAPHY_TYPE`
   の粒度で数える。部位まで見るのは、0004で `Hypertrophy` の下に胆管上皮・肝細胞・
   星細胞・Kupffer細胞の肥大が同居していることが分かったため。
2. **複数化合物が出している所見×部位**だけを対象にする。1化合物しか出さない所見
   （`Hemorrhage` = monocrotalineのみ等）は、何枚集めても化合物特異性と分離できないので
   増やす価値が無い。
3. 対象の所見×部位ごとに、その所見を安定して出している化合物を選ぶ。
4. 選ばれた化合物について、**その実験の全スライド**（Control/Low/Middle/High × 全時点）を
   リストに入れる。controlは比較の基準として必須で、中間用量は用量反応という
   バッチに直交する検証軸を作るために要る。

出力はダウンロード可能なURL付きのテーブル。既に手元にあるものには印を付ける。

Usage:
    python scripts/build_target_slide_list.py --min-compounds-per-site 3 --compounds-per-site 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORRECTED = PROJECT_ROOT / "data" / "corrected"
LIVER_DIR = PROJECT_ROOT / "data" / "TGGATEs" / "WSI" / "Liver"

FTP_PREFIX = "ftp://ftp.biosciencedbc.jp/archive/"
HTTPS_PREFIX = "https://dbarchive.biosciencedbc.jp/data/"
REPEAT_PERIODS = ["4 day", "8 day", "15 day", "29 day"]

# 化合物数の条件では落ちるが、固有の問いのために必要なもの。
#  - 手元の8化合物: 既に埋め込みまで済んでおり、これまでの結果との接続に要る
#  - carbon tetrachloride: `Degeneration, fatty` を出すもう1つの化合物（全2化合物）。
#    しかも部位がCentrilobularで、cholesterol+cholate(Peripheral)と小葉内分布が反対。
#    同一所見名が別クラスタに割れるかを問える唯一の組み合わせ。
ALWAYS_INCLUDE = [
    "1% cholesterol + 0.25% sodium cholate", "WY-14643", "carbon tetrachloride",
    "ethambutol", "monocrotaline", "phenobarbital", "promethazine",
    "thioacetamide", "vitamin A",
]


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pathology findings and the slide inventory, both filtered to Liver."""
    pathology = pd.read_csv(CORRECTED / "open_tggates_pathology.csv")
    pathology = pathology[pathology["ORGAN"] == "Liver"].copy()
    # SP_FLG=true は自然発生とみなされた背景病変。投与の効果を追うので除く。
    pathology = pathology[pathology["SP_FLG"].astype(str).str.lower() != "true"]
    # "DEAD" は死亡の記録であって形態所見ではない。視覚トークンの対象にならない。
    pathology = pathology[pathology["FINDING_TYPE"] != "DEAD"]
    pathology["finding_site"] = (
        pathology["FINDING_TYPE"].astype(str)
        + " @ "
        + pathology["TOPOGRAPHY_TYPE"].fillna("unspecified").astype(str)
    )

    images = pd.read_csv(CORRECTED / "open_tggates_pathological_image.csv")
    images = images[images["ORGAN"] == "Liver"].copy()
    images["slide_id"] = images["FILE_LOCATION"].apply(lambda u: Path(str(u)).stem)
    individual = pd.read_csv(CORRECTED / "open_tggates_individual.csv")
    images = images.merge(
        individual[["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID", "DOSE_LEVEL"]],
        on=["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"],
        how="left",
    )
    return pathology, images


def choose_compounds(
    pathology: pd.DataFrame,
    *,
    periods: list[str],
    min_animals: int,
    min_compounds_per_site: int,
    compounds_per_site: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick compounds so that every eligible finding@site has several of them.

    Returns:
        (sites, selection) — the finding@site table that drove the choice, and
        one row per (finding_site, compound) actually selected.
    """
    repeat = pathology[pathology["SACRIFICE_PERIOD"].isin(periods)]

    # 所見×部位 × 化合物 ごとの個体数。少数例しか出ない組み合わせは「その化合物が
    # その所見を安定して出す」とは言えないので min_animals で足切りする。
    pairs = (
        repeat.groupby(["finding_site", "COMPOUND_NAME"])
        .size()
        .rename("n_animals")
        .reset_index()
    )
    pairs = pairs[pairs["n_animals"] >= min_animals]

    sites = (
        pairs.groupby("finding_site")
        .agg(n_compounds=("COMPOUND_NAME", "nunique"), n_animals=("n_animals", "sum"))
        .reset_index()
        .sort_values("n_compounds", ascending=False)
    )
    # 1化合物しか出さない所見は、何枚増やしても所見特異性と化合物特異性を分離できない。
    eligible = sites[sites["n_compounds"] >= min_compounds_per_site]

    selection = []
    for site in eligible["finding_site"]:
        top = (
            pairs[pairs["finding_site"] == site]
            .sort_values("n_animals", ascending=False)
            .head(compounds_per_site)
        )
        for row in top.itertuples():
            selection.append(
                {
                    "finding_site": site,
                    "compound_name": row.COMPOUND_NAME,
                    "n_animals": int(row.n_animals),
                }
            )
    return sites, pd.DataFrame(selection)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", nargs="+", default=REPEAT_PERIODS)
    parser.add_argument("--min-animals", type=int, default=3,
                        help="その化合物がその所見を出していると認める最小個体数")
    parser.add_argument("--min-compounds-per-site", type=int, default=3,
                        help="対象にする所見×部位が満たすべき化合物数")
    parser.add_argument("--compounds-per-site", type=int, default=6,
                        help="1つの所見×部位あたり何化合物まで取るか")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "target_slide_list")
    parser.add_argument(
        "--always-include", nargs="*", default=ALWAYS_INCLUDE,
        help="化合物数の条件に関わらず必ず含める化合物。既に手元にあるものと、"
             "少数化合物でも固有の問いが立つもの（脂肪変性の2化合物など）。",
    )
    args = parser.parse_args()

    pathology, images = load_tables()
    sites, selection = choose_compounds(
        pathology,
        periods=args.periods,
        min_animals=args.min_animals,
        min_compounds_per_site=args.min_compounds_per_site,
        compounds_per_site=args.compounds_per_site,
    )
    # tier: 各所見×部位の中での順位。1-3位を tier1（その所見が化合物をまたげる最小構成）、
    # 4位以降を tier2 とする。1つの化合物が複数の所見に寄与する場合は最も良い順位を採る。
    selection["rank_in_site"] = (
        selection.groupby("finding_site")["n_animals"].rank(ascending=False, method="first").astype(int)
    )
    selection["tier"] = np.where(selection["rank_in_site"] <= 3, 1, 2)
    tier_by_compound = selection.groupby("compound_name")["tier"].min().to_dict()

    for extra in args.always_include:
        tier_by_compound.setdefault(extra, 1)
    compounds = sorted(tier_by_compound)

    print(f"=== 対象にした所見×部位: {len(sites[sites['n_compounds'] >= args.min_compounds_per_site])} / 全{len(sites)} ===")
    print(sites.head(20).to_string(index=False))
    print(f"\n=== 選ばれた化合物: {len(compounds)} ===")
    print(", ".join(compounds))

    # 選ばれた化合物の、対象時点の全スライド（Control含む全用量）
    local_ids = {p.stem for p in LIVER_DIR.rglob("*.svs")} if LIVER_DIR.exists() else set()
    slides = images[
        images["COMPOUND_NAME"].isin(compounds)
        & images["SACRIFICE_PERIOD"].isin(args.periods)
    ].copy()
    slides["url"] = slides["FILE_LOCATION"].str.replace(FTP_PREFIX, HTTPS_PREFIX, regex=False)
    slides["already_local"] = slides["slide_id"].isin(local_ids)
    slides["tier"] = slides["COMPOUND_NAME"].map(tier_by_compound).astype(int)
    slides = slides.rename(
        columns={
            "COMPOUND_NAME": "compound_name",
            "SACRIFICE_PERIOD": "sacrifice_period",
            "DOSE_LEVEL": "dose_level",
            "EXP_ID": "exp_id",
            "GROUP_ID": "group_id",
            "INDIVIDUAL_ID": "individual_id",
        }
    )[
        [
            "slide_id", "compound_name", "dose_level", "sacrifice_period",
            "exp_id", "group_id", "individual_id", "tier", "url", "already_local",
        ]
    ].drop_duplicates("slide_id").sort_values(["compound_name", "sacrifice_period", "dose_level"])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # rsync --files-from 用のリスト。パスは Liver/ を起点とした相対パスで、
    # ローカルの data/TGGATEs/WSI/Liver/ と同じ階層を前提にしている。
    # 1ファイル1行にしておくと、転送が1セッションで済む（パスワード入力も1回）。
    todo_all = slides[~slides["already_local"]]
    for name, subset in [
        ("all", todo_all),
        ("tier1", todo_all[todo_all["tier"] == 1]),
        ("tier2", todo_all[todo_all["tier"] == 2]),
    ]:
        rel = [
            f"{str(period).replace(' ', '_')}/{sid}.svs"
            for period, sid in zip(subset["sacrifice_period"], subset["slide_id"])
        ]
        (args.out_dir / f"rsync_files_{name}.txt").write_text("\n".join(sorted(rel)) + "\n")
        print(f"rsync_files_{name}.txt: {len(rel)} 行")
    slides.to_parquet(args.out_dir / "target_slides.parquet", index=False)
    slides.to_csv(args.out_dir / "target_slides.csv", index=False)
    selection.to_csv(args.out_dir / "finding_site_compounds.csv", index=False)
    sites.to_csv(args.out_dir / "finding_site_counts.csv", index=False)

    todo = slides[~slides["already_local"]]
    print(f"\n=== スライド数 ===")
    print(f"対象合計: {len(slides)} 枚 / うち取得済み {int(slides['already_local'].sum())} / "
          f"**未取得 {len(todo)} 枚**")
    print(pd.crosstab(slides["sacrifice_period"], slides["dose_level"]).to_string())
    print("\n=== tier別 ===")
    print(slides.groupby("tier").agg(
        枚数=("slide_id", "size"), 取得済み=("already_local", "sum"),
        化合物数=("compound_name", "nunique")).to_string())
    print(f"\n未取得の概算コスト: 約 {len(todo) * 0.8 / 1024:.1f} TB / "
          f"埋め込み抽出 約 {len(todo) * 78 / 3600:.0f} 時間（GPU1枚）")
    print(f"\n-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
