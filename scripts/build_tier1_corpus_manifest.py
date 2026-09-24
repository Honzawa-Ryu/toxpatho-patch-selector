"""Merge 0001 (base 590 slides) + 0007's 4 CCl4 chunks (100 slides) + 0008's 4
tier1-remainder chunks (2,751 slides) into one manifest + one flat features
directory, so 0006's clustering code (single manifest_parquet + single flat
features_dir) can be reused unmodified on the full tier1 corpus.

0008's exclude_features_dirs already keeps its output disjoint from 0001/0007,
so this is a straight concat (all three sources share the same 12-column raw
manifest schema) rather than the column-intersection exp0003 needed against
0002's already-labeled manifest.

Usage: python scripts/build_tier1_corpus_manifest.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs" / "0009_20260919_tier1_corpus_clustering"

RAW_MANIFEST_PATHS = [
    PROJECT_ROOT / "outputs/0001_20260907_extract_wsi_embeddings/trident_all_slides/slide_manifest.parquet",
    *[
        PROJECT_ROOT / f"outputs/0007_20260917_ccl4_fatty_degeneration_embeddings/chunk_{i}/slide_manifest.parquet"
        for i in range(4)
    ],
    *[
        PROJECT_ROOT / f"outputs/0008_20260918_tier1_corpus_embeddings/chunk_{i}/slide_manifest.parquet"
        for i in range(4)
    ],
]
FEATURE_DIRS = [
    PROJECT_ROOT / "outputs/0001_20260907_extract_wsi_embeddings/trident_job/20x_256px_0px_overlap/features_uni_v2",
    *[
        PROJECT_ROOT
        / f"outputs/0007_20260917_ccl4_fatty_degeneration_embeddings/trident_job_chunk{i}"
        / "20x_256px_0px_overlap/features_uni_v2"
        for i in range(4)
    ],
    *[
        PROJECT_ROOT
        / f"outputs/0008_20260918_tier1_corpus_embeddings/trident_job_chunk{i}"
        / "20x_256px_0px_overlap/features_uni_v2"
        for i in range(4)
    ],
]

MERGED_MANIFEST_PATH = OUT_DIR / "manifest_tier1_corpus.parquet"
MERGED_FEATURES_DIR = OUT_DIR / "features_tier1_corpus"


def main() -> None:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from lib.tggates_metadata import attach_pathology_findings

    raw = pd.concat([pd.read_parquet(p) for p in RAW_MANIFEST_PATHS], ignore_index=True)
    if raw["slide_id"].duplicated().any():
        dupes = raw.loc[raw["slide_id"].duplicated(), "slide_id"].tolist()
        raise SystemExit(f"duplicate slide_id across the raw manifests: {dupes}")

    feature_paths = {p.stem: p for d in FEATURE_DIRS for p in d.glob("*.h5")}
    missing = sorted(set(raw["slide_id"]) - set(feature_paths))
    if missing:
        print(f"dropping {len(missing)} slide(s) with no feature file: {missing}")
        raw = raw[~raw["slide_id"].isin(missing)].copy()

    manifest = attach_pathology_findings(
        raw, DATA_DIR / "corrected/open_tggates_pathology.csv", organ="Liver"
    ).sort_values("slide_id")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(MERGED_MANIFEST_PATH, index=False)
    print(f"wrote {MERGED_MANIFEST_PATH}: {len(manifest)} slides")

    MERGED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    n_linked = 0
    for slide_id in manifest["slide_id"]:
        dest = MERGED_FEATURES_DIR / f"{slide_id}.h5"
        if dest.exists() or dest.is_symlink():
            continue
        dest.symlink_to(feature_paths[slide_id])
        n_linked += 1
    print(f"symlinked {n_linked} feature files into {MERGED_FEATURES_DIR}")


if __name__ == "__main__":
    main()
