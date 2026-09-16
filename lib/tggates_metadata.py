"""Join local TG-GATEs WSI files against the open-tggates metadata tables.

Shared by preprocessing experiments that need compound/dose/timepoint per
slide (0001_extract_wsi_embeddings today, later phases too).
"""

from pathlib import Path

import pandas as pd

# open_tggates_pathology.csv の GRADE_TYPE に出る値の重症度順。
# "P" (present) は等級が付いていないことを表すので、この順序には載せない。
GRADE_ORDER = {"minimal": 1, "slight": 2, "moderate": 3, "severe": 4}


def build_liver_slide_manifest(
    wsi_dir: Path,
    pathological_image_csv: Path,
    individual_csv: Path,
) -> pd.DataFrame:
    """Build a one-row-per-slide manifest for local liver WSIs.

    Args:
        wsi_dir: Directory containing the local .svs files (searched
            recursively — they live under per-timepoint subfolders, e.g.
            wsi_dir/4_day/29495.svs, wsi_dir/15_day/25540.svs, ...).
        pathological_image_csv: Path to open_tggates_pathological_image.csv.
            Columns include EXP_ID, GROUP_ID, INDIVIDUAL_ID, COMPOUND_NAME,
            ORGAN, FILE_LOCATION (an ftp:// URL whose basename is the local
            .svs filename, e.g. ".../Liver/26761.svs" -> "26761.svs"),
            SACRIFICE_PERIOD, DOSE, DOSE_UNIT.
        individual_csv: Path to open_tggates_individual.csv. Columns include
            EXP_ID, GROUP_ID, INDIVIDUAL_ID, SEX_TYPE, STRAIN_TYPE,
            DOSE_LEVEL (Control/Low/Middle/High) — not present in the
            pathological_image table, so this second join is what adds it.

    Join plan:
        1. List every *.svs under wsi_dir (recursive). slide_id = filename
           stem (e.g. "25540"); TRIDENT identifies slides the same way, so
           this is the key everything downstream keys on.
        2. Filter pathological_image_csv to ORGAN == "Liver" (or whatever
           organ wsi_dir represents), and match rows to local files by
           comparing slide_id against the basename-without-extension of
           FILE_LOCATION. All 219 local files are expected to match exactly
           one row each here — treat a slide with zero or >1 matches as a
           data problem worth raising/logging loudly, not silently dropping.
        3. Left-join that onto individual_csv on (EXP_ID, GROUP_ID,
           INDIVIDUAL_ID) to pick up DOSE_LEVEL (+ SEX_TYPE/STRAIN_TYPE if
           useful later).
        4. Sort by slide_id and reset the index. This ordering must be
           stable across reruns — anything that later indexes into this
           manifest by row position (this experiment doesn't, but a future
           one might) depends on it not shuffling between runs.

    Returns:
        DataFrame with (at least) columns: slide_id, svs_path (str, resolved
        path under wsi_dir), sacrifice_period, compound_name, dose,
        dose_unit, dose_level, exp_id, group_id, individual_id.

    Raises:
        Whatever you decide is appropriate for: a local .svs with no
        matching metadata row, or a metadata row matching >1 local file.
        Don't let those pass silently — they'd corrupt every downstream
        phase that trusts this manifest for labels.
    """
    organ = wsi_dir.name

    svs_paths = sorted(wsi_dir.rglob("*.svs"))
    local_slides = pd.DataFrame(
        {
            "slide_id": [p.stem for p in svs_paths],
            "svs_path": [str(p) for p in svs_paths],
        }
    )

    path_df = pd.read_csv(pathological_image_csv)
    path_df = path_df[path_df["ORGAN"] == organ].copy()
    path_df["slide_id"] = path_df["FILE_LOCATION"].apply(lambda url: Path(url).stem)

    merged = local_slides.merge(path_df, on="slide_id", how="left", indicator=True)

    unmatched = merged.loc[merged["_merge"] == "left_only", "slide_id"]
    if not unmatched.empty:
        raise ValueError(
            f"{len(unmatched)} local .svs file(s) have no matching row in "
            f"{pathological_image_csv}: {unmatched.tolist()}"
        )

    dup_mask = merged["slide_id"].duplicated(keep=False) & (merged["_merge"] == "both")
    if dup_mask.any():
        dup_ids = sorted(merged.loc[dup_mask, "slide_id"].unique())
        raise ValueError(
            f"{len(dup_ids)} slide_id(s) match >1 row in "
            f"{pathological_image_csv}: {dup_ids}"
        )

    merged = merged.drop(columns="_merge")

    individual_df = pd.read_csv(individual_csv)
    manifest = merged.merge(
        individual_df[
            ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID", "DOSE_LEVEL", "SEX_TYPE", "STRAIN_TYPE"]
        ],
        on=["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"],
        how="left",
    )

    manifest = manifest.rename(
        columns={
            "SACRIFICE_PERIOD": "sacrifice_period",
            "COMPOUND_NAME": "compound_name",
            "DOSE": "dose",
            "DOSE_UNIT": "dose_unit",
            "DOSE_LEVEL": "dose_level",
            "EXP_ID": "exp_id",
            "GROUP_ID": "group_id",
            "INDIVIDUAL_ID": "individual_id",
            "SEX_TYPE": "sex_type",
            "STRAIN_TYPE": "strain_type",
        }
    )

    columns = [
        "slide_id",
        "svs_path",
        "sacrifice_period",
        "compound_name",
        "dose",
        "dose_unit",
        "dose_level",
        "exp_id",
        "group_id",
        "individual_id",
        "sex_type",
        "strain_type",
    ]
    manifest = manifest[columns].sort_values("slide_id").reset_index(drop=True)

    return manifest


def _as_int_key(series: pd.Series) -> pd.Series:
    """Normalize a TG-GATEs ID column to int so joins across tables line up.

    open_tggates_*.csv holds zero-padded IDs (EXP_ID="0117", GROUP_ID="01").
    pandas strips the padding when it infers an int dtype, but any caller that
    reads with dtype=str keeps it — and merging a padded "0117" against an
    unpadded 117 matches zero rows *silently*, which looks exactly like
    "this slide has no findings". Force both sides through int here.
    """
    return pd.to_numeric(series, errors="raise").astype(int)


def attach_pathology_findings(
    manifest: pd.DataFrame,
    pathology_csv: Path,
    organ: str = "Liver",
) -> pd.DataFrame:
    """Add per-slide pathology findings to a slide manifest.

    Args:
        manifest: Output of build_liver_slide_manifest() — one row per slide,
            with exp_id / group_id / individual_id identifying the animal.
        pathology_csv: Path to open_tggates_pathology.csv. One row per
            (animal, finding): EXP_ID, GROUP_ID, INDIVIDUAL_ID, ORGAN,
            FINDING_TYPE, TOPOGRAPHY_TYPE, GRADE_TYPE (minimal/slight/
            moderate/severe/P), SP_FLG ("true" when the finding is regarded
            as spontaneous/background rather than treatment-related).
        organ: Organ to filter the pathology table to.

    Returns:
        A copy of manifest with these columns added (one row per slide,
        row count and order unchanged):
            n_findings:      number of finding rows for that animal
            has_finding:     n_findings > 0
            finding_types:   sorted unique FINDING_TYPE values (list[str])
            finding_sites:   sorted unique "FINDING_TYPE @ TOPOGRAPHY_TYPE" strings.
                             Keep these: one finding name covers several
                             different lesions and the topography is what
                             separates them — "Hypertrophy" spans hypertrophy
                             of bile duct epithelium, hepatocytes, Ito cells
                             and Kupffer cells, which are not the same thing.
            max_grade:       the most severe GRADE_TYPE present, or None when
                             the slide has no graded finding (findings that
                             only carry "P" leave this None while has_finding
                             stays True)
            max_grade_ord:   max_grade mapped through GRADE_ORDER (NA if None)
            n_findings_treatment_related: findings with SP_FLG != true

    Note:
        An animal with no row in the pathology table is a genuine negative
        (no finding recorded), not a join failure — unlike the manifest join
        in build_liver_slide_manifest(), a missing match here is expected and
        is filled with 0 / False / [] rather than raising.
    """
    key = ["exp_id", "group_id", "individual_id"]

    findings = pd.read_csv(pathology_csv)
    findings = findings[findings["ORGAN"] == organ].copy()
    findings = findings.rename(
        columns={"EXP_ID": "exp_id", "GROUP_ID": "group_id", "INDIVIDUAL_ID": "individual_id"}
    )
    for col in key:
        findings[col] = _as_int_key(findings[col])

    out = manifest.copy()
    for col in key:
        out[col] = _as_int_key(out[col])

    findings["grade_ord"] = findings["GRADE_TYPE"].map(GRADE_ORDER)
    findings["is_treatment_related"] = (
        findings["SP_FLG"].astype(str).str.lower() != "true"
    )
    findings["finding_site"] = (
        findings["FINDING_TYPE"].astype(str)
        + " @ "
        + findings["TOPOGRAPHY_TYPE"].fillna("unspecified").astype(str)
    )

    per_animal = findings.groupby(key).agg(
        n_findings=("FINDING_TYPE", "size"),
        finding_types=("FINDING_TYPE", lambda s: sorted(set(s))),
        finding_sites=("finding_site", lambda s: sorted(set(s))),
        max_grade_ord=("grade_ord", "max"),
        n_findings_treatment_related=("is_treatment_related", "sum"),
    )

    out = out.merge(per_animal, on=key, how="left")

    out["n_findings"] = out["n_findings"].fillna(0).astype(int)
    out["n_findings_treatment_related"] = (
        out["n_findings_treatment_related"].fillna(0).astype(int)
    )
    out["has_finding"] = out["n_findings"] > 0
    for col in ["finding_types", "finding_sites"]:
        out[col] = out[col].apply(lambda v: v if isinstance(v, list) else [])

    ord_to_grade = {v: k for k, v in GRADE_ORDER.items()}
    out["max_grade"] = out["max_grade_ord"].map(ord_to_grade)

    return out
