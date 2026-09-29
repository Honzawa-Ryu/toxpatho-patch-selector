"""Freeze a metadata-based, matched slide subset; no clustering is performed.

Run: .venv/bin/python scripts/build_pathology_benchmark_subset.py
Existing snapshots are never overwritten; use --out for a new version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "outputs/0009_20260919_tier1_corpus_clustering/manifest_tier1_corpus.parquet"
PATHOLOGY = ROOT / "data/corrected/open_tggates_pathology.csv"
DEFAULT_OUT = ROOT / "experiments/0010_20260925_pathology_benchmark_subset/snapshot_v1"
KEYS = ["exp_id", "group_id", "individual_id"]
CANDIDATES = ["Necrosis", "Degeneration, fatty", "Degeneration, granular, eosinophilic"]
TARGET = CANDIDATES[2]
GRADES = {"minimal": 1, "slight": 2, "moderate": 3, "severe": 4}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spread_order(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Deterministically alternate strata, then use numeric slide ID as tie break."""
    ordered = frame.assign(_id=pd.to_numeric(frame.slide_id)).sort_values(columns + ["_id"])
    ordered["_round"] = ordered.groupby(columns, dropna=False).cumcount()
    return ordered.sort_values(["_round"] + columns + ["_id"]).drop(columns=["_round", "_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Snapshot already exists: {args.out}; choose a new --out directory")

    m = pd.read_parquet(MANIFEST).sort_values("slide_id").reset_index(drop=True)
    m["slide_id"] = m.slide_id.astype(str)
    assert m.slide_id.is_unique
    p = pd.read_csv(PATHOLOGY, dtype=str).rename(columns={k.upper(): k for k in KEYS})
    p = p[p.ORGAN == "Liver"].copy()
    for key in KEYS:
        m[key] = pd.to_numeric(m[key], errors="raise").astype(int)
        p[key] = pd.to_numeric(p[key], errors="raise").astype(int)
    m["duplicate_animal_slide"] = m.duplicated(KEYS, keep="first")
    a = m[["slide_id", "compound_name", "dose_level", "sacrifice_period", *KEYS]].merge(
        p, on=KEYS, how="inner", validate="many_to_many"
    )
    assert (a.compound_name == a.COMPOUND_NAME).all()
    known_dose = a.dose_level.notna()
    assert (a.loc[known_dose, "dose_level"] == a.loc[known_dose, "DOSE_LEVEL"]).all()
    assert (a.sacrifice_period == a.SACRIFICE_PERIOD).all()
    for finding in CANDIDATES:
        expected = set(m.loc[m.finding_types.map(lambda fs: finding in fs), "slide_id"])
        assert expected == set(a.loc[a.FINDING_TYPE == finding, "slide_id"])

    # TOPOGRAPHY_TYPE is anatomical location, never a collection facility.
    m["facility_id"] = "unknown"
    m["facility_status"] = "not_available_in_local_metadata"
    for column in ["finding_types", "finding_sites"]:
        m[column] = m[column].map(lambda x: json.dumps(list(x), ensure_ascii=False))
    target = a[a.FINDING_TYPE == TARGET].copy()
    target["grade_ord"] = target.GRADE_TYPE.map(GRADES)
    agg = target.groupby("slide_id").agg(
        target_grades=("GRADE_TYPE", lambda s: "|".join(sorted(set(s.dropna())))),
        target_topographies=("TOPOGRAPHY_TYPE", lambda s: "|".join(sorted(set(s.fillna("unspecified"))))),
        target_grade_ord=("grade_ord", "max"),
        target_spontaneous_flags=("SP_FLG", lambda s: "|".join(sorted(set(s.dropna())))),
    )
    m = m.merge(agg, on="slide_id", how="left", validate="one_to_one")
    m["target_label_present"] = m.slide_id.isin(target.slide_id)
    m["target_strict_positive"] = m.target_label_present & (m.target_topographies == "Hepatocyte")
    m["target_finding"] = TARGET
    m["label_status"] = "source_annotation_unreviewed"

    summary = []
    concerns = [
        "Different target cells/locations; minimal grades frequent; requires anatomical restriction",
        "Only two compounds; compound and topography are perfectly associated",
        "Restrict to Hepatocyte; cofindings, dose and facility remain to audit",
    ]
    for finding, concern in zip(CANDIDATES, concerns):
        rows = a[a.FINDING_TYPE == finding]
        slides = rows.drop_duplicates("slide_id")
        comp = slides.compound_name.value_counts()
        exp = slides.exp_id.value_counts()
        summary.append(dict(
            finding=finding, n_compounds=len(comp), n_experiments=len(exp),
            n_positive_slides=len(slides), largest_compound=comp.index[0],
            largest_compound_fraction=comp.iloc[0] / len(slides),
            largest_experiment_fraction=exp.iloc[0] / len(slides),
            n_known_facilities=0, facility_unknown_fraction=1.0,
            concern=concern, decision="provisional_selection" if finding == TARGET else "defer",
        ))

    # Match positive, treated target-negative, and target-negative Control within
    # experiment/time. Negatives may have other findings; retain those labels.
    triplets: dict[str, list[list[pd.Series]]] = {}
    strata_rows = []
    for (compound, exp, time), group in m.groupby(["compound_name", "exp_id", "sacrifice_period"]):
        group = group[~group.duplicate_animal_slide & group.dose_level.isin(["Control", "Low", "Middle", "High"])]
        pos = group[group.target_strict_positive & (group.dose_level != "Control")]
        neg = group[~group.target_label_present & (group.dose_level != "Control")]
        ctrl = group[~group.target_label_present & (group.dose_level == "Control")]
        count = min(len(pos), len(neg), len(ctrl))
        strata_rows.append(dict(compound_name=compound, exp_id=exp, sacrifice_period=time,
                                positive=len(pos), treated_negative=len(neg), control=len(ctrl),
                                triplet_capacity=count))
        pos = spread_order(pos, ["target_grade_ord", "dose_level"])
        neg = spread_order(neg, ["dose_level"])
        ctrl = ctrl.sort_values("slide_id")
        for i in range(count):
            triplets.setdefault(compound, []).append([pos.iloc[i], neg.iloc[i], ctrl.iloc[i]])

    # Equal support per compound and all three roles; require >=10 triplets.
    eligible = sorted(c for c, ts in triplets.items() if len(ts) >= 10)
    assert len(eligible) >= 3, "Insufficient matched compounds"
    selected_rows = []
    for compound in eligible:
        options = triplets[compound]
        positive_options = pd.DataFrame([t[0] for t in options]).reset_index(drop=True)
        chosen = spread_order(positive_options, ["sacrifice_period", "target_grade_ord"]).head(10)
        for i, idx in enumerate(chosen.index):
            for role, row in zip(["positive", "treated_negative", "control"], options[idx]):
                record = row.to_dict()
                record.update(role=role, match_id=f"{compound}:{i + 1:02d}",
                              holdout_group=compound,
                              selection_reason="equal compound/role counts; matched experiment and timepoint")
                selected_rows.append(record)
    selected = pd.DataFrame(selected_rows).sort_values(["compound_name", "match_id", "role"])
    assert selected.slide_id.is_unique
    assert not selected.duplicated(KEYS).any()
    assert selected.groupby(["compound_name", "role"]).size().eq(10).all()
    assert selected.groupby("match_id")["exp_id"].nunique().eq(1).all()
    assert selected.groupby("match_id")["sacrifice_period"].nunique().eq(1).all()
    assert selected.loc[selected.role == "positive", "target_strict_positive"].all()
    assert not selected.loc[selected.role != "positive", "target_label_present"].any()
    assert selected.loc[selected.role == "positive", "target_grade_ord"].nunique() >= 2
    feature_dir = MANIFEST.parent / "features_tier1_corpus"
    assert all(Path(s).is_file() for s in selected.svs_path), "Missing WSI"
    assert all((feature_dir / f"{s}.h5").is_file() for s in selected.slide_id), "Missing features"
    selected["feature_path"] = selected.slide_id.map(lambda s: str((feature_dir / f"{s}.h5").relative_to(ROOT)))
    selected["svs_path"] = selected.svs_path.map(lambda s: str(Path(s).relative_to(ROOT)))

    args.out.mkdir(parents=True)
    pd.DataFrame(summary).to_csv(args.out / "candidate_summary.csv", index=False)
    a[a.FINDING_TYPE.isin(CANDIDATES)].sort_values(["FINDING_TYPE", "compound_name", "slide_id"]).to_csv(
        args.out / "candidate_finding_rows.csv", index=False
    )
    a[a.FINDING_TYPE.isin(CANDIDATES)].groupby(
        ["FINDING_TYPE", "compound_name", "exp_id", "dose_level", "sacrifice_period", "TOPOGRAPHY_TYPE", "GRADE_TYPE"],
        dropna=False,
    ).slide_id.nunique().rename("n_slides").reset_index().to_csv(args.out / "candidate_distribution.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(args.out / "matching_strata.csv", index=False)
    selected.to_csv(args.out / "slides.csv", index=False)
    for role in ["positive", "treated_negative", "control"]:
        (args.out / f"{role}_slide_ids.txt").write_text("\n".join(selected.loc[selected.role == role, "slide_id"]) + "\n")
    audit = m.drop(columns="svs_path").copy()
    roles = selected.set_index("slide_id").role
    audit["selected_role"] = audit.slide_id.map(roles).fillna("not_selected")
    audit["selection_status"] = audit.apply(lambda r:
        "selected" if r.selected_role != "not_selected" else
        "duplicate_animal_slide" if r.duplicate_animal_slide else
        "missing_or_unknown_dose_level" if r.dose_level not in ["Control", "Low", "Middle", "High"] else
        "target_positive_other_or_unspecified_topography" if r.target_label_present and not r.target_strict_positive else
        "compound_has_fewer_than_10_matched_triplets" if r.compound_name not in eligible else
        "outside_matched_strata_or_balanced_cap", axis=1)
    audit.to_csv(args.out / "eligibility_audit.csv", index=False)
    for columns, name in [(["compound_name", "exp_id", "role"], "subset_counts"),
                          (["compound_name", "role", "dose_level", "sacrifice_period"], "subset_dose_time"),
                          (["compound_name", "role", "target_grades"], "subset_severity")]:
        selected.groupby(columns, dropna=False).size().rename("n_slides").reset_index().to_csv(args.out / f"{name}.csv", index=False)
    cofindings = selected[["slide_id", "role", "compound_name"]].merge(a[["slide_id", "FINDING_TYPE"]].drop_duplicates(), on="slide_id")
    cofindings.groupby(["role", "compound_name", "FINDING_TYPE"]).slide_id.nunique().rename("n_slides").reset_index().to_csv(args.out / "cofindings.csv", index=False)
    source_paths = [MANIFEST, PATHOLOGY, Path(__file__).resolve()]
    provenance = dict(
        version="v1", target_finding=TARGET, target_topography="Hepatocyte",
        selection_uses_clusters=False, min_triplets_per_compound=10, selected_compounds=eligible,
        selection_order="deterministic stratum/grade/dose round robin; numeric slide ID tie break",
        facility_verified=False, pathology_visually_verified=False,
        sources={str(p.relative_to(ROOT)): digest(p) for p in source_paths},
        artifacts={p.name: digest(p) for p in sorted(args.out.iterdir()) if p.is_file()},
        pandas_version=pd.__version__,
    )
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(pd.DataFrame(summary).to_string(index=False))
    print(selected.groupby(["compound_name", "role"]).size().to_string())
    print(f"Wrote {len(selected)} slides to {args.out}")


if __name__ == "__main__":
    main()
