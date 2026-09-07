"""Join local TG-GATEs WSI files against the open-tggates metadata tables.

Shared by preprocessing experiments that need compound/dose/timepoint per
slide (0001_extract_wsi_embeddings today, later phases too).
"""

from pathlib import Path

import pandas as pd


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
