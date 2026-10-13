"""系列A/Bの判断材料: 所見がどの試験（=化合物）に何枚ずつ出ているかを集計する。

usage: .venv/bin/python scripts/summarize_finding_coverage.py [out_dir]
出力:
  finding_by_type.csv      所見ごと: 全slide / 所見1種類slide の枚数・試験数、SP_FLG内訳
  finding_x_compound.csv   所見×化合物の枚数（投与群、全slide基準）
  single_x_compound.csv    所見×化合物の枚数（投与群、所見1種類slideのみ）
  compound_composition.csv 化合物ごと: 無所見Ctrl / 投与群の所見0・1・2以上 / 主な所見
  site_single.csv          所見1種類slideの部位まで分けた内訳
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/0010_20260925_pathology_benchmark_subset/label_audit/manifest_known_unique.parquet"
PATHOLOGY = ROOT / "data/corrected/open_tggates_pathology.csv"
KEYS = ["exp_id", "group_id", "individual_id", "sacrifice_period"]
MIN_PER_EXP = 3   # 「その試験で使える」とみなす投与群slideの最小枚数


def spread(s: pd.Series) -> str:
    return "; ".join(f"{k} {v}" for k, v in s.value_counts().items())


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs/finding_coverage_20261013"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(MANIFEST)
    m["types"] = m.finding_types.apply(lambda a: sorted({x for x in a if x is not None}))
    m["n_types"] = m.types.str.len()
    m["treated"] = m.dose_level != "Control"

    # SP_FLG。join keyは両側intに正規化（EXPERIMENT_NOTES 0002節）
    p = pd.read_csv(PATHOLOGY, dtype=str)
    p = p[p.ORGAN == "Liver"].rename(columns={"EXP_ID": "exp_id", "GROUP_ID": "group_id", "INDIVIDUAL_ID": "individual_id",
                                              "SACRIFICE_PERIOD": "sacrifice_period", "FINDING_TYPE": "type"})
    for c in ["exp_id", "group_id", "individual_id"]:
        p[c] = p[c].astype(int)
    sp = p.groupby(KEYS + ["type"]).SP_FLG.agg(
        lambda s: "true" if (s == "true").all() else ("false" if (s == "false").all() else "mixed")).reset_index()

    ex = m.explode("types").dropna(subset=["types"]).rename(columns={"types": "type"})
    ex = ex.merge(sp, on=KEYS + ["type"], how="left", validate="many_to_one")
    ex["SP_FLG"] = ex.SP_FLG.fillna("not_found")
    ex["single"] = ex.n_types == 1
    tr = ex[ex.treated]

    def summarize(d: pd.DataFrame, prefix: str) -> pd.DataFrame:
        t = d[d.treated]
        per_exp = t.groupby(["type", "compound_name"]).size()
        return pd.DataFrame({
            f"{prefix}_slides": d.groupby("type").size(),
            f"{prefix}_control": d[~d.treated].groupby("type").size(),
            f"{prefix}_treated": t.groupby("type").size(),
            f"{prefix}_n_exp": t.groupby("type").compound_name.nunique(),
            f"{prefix}_n_exp_ge{MIN_PER_EXP}": per_exp.ge(MIN_PER_EXP).groupby(level=0).sum(),
            f"{prefix}_spontaneous": d[d.SP_FLG == "true"].groupby("type").size(),
            f"{prefix}_treatment_related": d[d.SP_FLG == "false"].groupby("type").size(),
        })

    by_type = summarize(ex, "all").join(summarize(ex[ex.single], "single")).fillna(0).astype(int)
    by_type["all_compounds"] = tr.groupby("type").compound_name.apply(spread)
    by_type["single_compounds"] = tr[tr.single].groupby("type").compound_name.apply(spread)
    by_type["dose_levels"] = tr.groupby("type").dose_level.apply(spread)
    by_type["time_points"] = tr.groupby("type").sacrifice_period.apply(spread)
    by_type = by_type.sort_values(["all_n_exp_ge3", "all_slides"], ascending=False)
    by_type.to_csv(out / "finding_by_type.csv")

    tr.pivot_table(index="type", columns="compound_name", values="slide_id", aggfunc="size", fill_value=0) \
        .loc[by_type.index.intersection(tr.type.unique())].to_csv(out / "finding_x_compound.csv")
    tr[tr.single].pivot_table(index="type", columns="compound_name", values="slide_id", aggfunc="size", fill_value=0) \
        .to_csv(out / "single_x_compound.csv")

    g = m.groupby("compound_name")
    comp = pd.DataFrame({
        "exp_id": g.exp_id.first(),
        "ctrl_no_finding": g.apply(lambda d: int((~d.treated & (d.n_types == 0)).sum())),
        "treated_0": g.apply(lambda d: int((d.treated & (d.n_types == 0)).sum())),
        "treated_1": g.apply(lambda d: int((d.treated & (d.n_types == 1)).sum())),
        "treated_2plus": g.apply(lambda d: int((d.treated & (d.n_types >= 2)).sum())),
    })
    comp["top_findings_treated"] = tr.groupby("compound_name").type.apply(
        lambda s: "; ".join(f"{k} {v}" for k, v in s.value_counts().head(5).items())).reindex(comp.index).fillna("")
    comp.sort_values("treated_1", ascending=False).to_csv(out / "compound_composition.csv")

    sites = m[m.n_types == 1].assign(site=lambda d: d.finding_sites.apply(lambda a: " | ".join(sorted({x for x in a if x}))))
    sites.groupby(["site"]).agg(n_slides=("slide_id", "size"), n_exp=("compound_name", "nunique"),
                                compounds=("compound_name", spread)) \
        .sort_values("n_slides", ascending=False).to_csv(out / "site_single.csv")
    print(out)


if __name__ == "__main__":
    main()
