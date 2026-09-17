"""Merge the 0002/0003 219-slide manifest with the newly extracted CCl4 100
slides into one manifest + one flat features directory, so 0003's clustering
code (which reads a single manifest_parquet + a single flat features_dir) can
be rerun unmodified on the combined 319-slide corpus.

See experiments/0003_20260912_control_absent_cluster_mining/plan.md (decisive
fatty-degeneration experiment) for why: does carbon tetrachloride's
`Degeneration, fatty` (centrilobular) land in the same cluster as cholesterol
+ sodium cholate's (peripheral, cluster 87 in the original 219-slide k=100
run), or a different one.

Usage: python scripts/build_ccl4_comparison_manifest.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs" / "0003_20260912_control_absent_cluster_mining"

ORIGINAL_MANIFEST = (
    PROJECT_ROOT
    / "outputs/0002_20260912_control_contrast_patch_scoring/control_contrast"
    / "slide_manifest_with_findings.parquet"
)
CCL4_FEATURE_DIRS = [
    PROJECT_ROOT
    / f"outputs/0007_20260917_ccl4_fatty_degeneration_embeddings/trident_job_chunk{i}"
    / "20x_256px_0px_overlap/features_uni_v2"
    for i in range(4)
]
ORIGINAL_FEATURES_DIR = (
    PROJECT_ROOT
    / "outputs/0001_20260907_extract_wsi_embeddings/trident_job"
    / "20x_256px_0px_overlap/features_uni_v2"
)

MERGED_MANIFEST_PATH = OUT_DIR / "manifest_219_plus_ccl4.parquet"
MERGED_FEATURES_DIR = OUT_DIR / "features_219_plus_ccl4"


def build_ccl4_manifest() -> pd.DataFrame:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from lib.tggates_metadata import attach_pathology_findings, build_liver_slide_manifest

    wsi_dir = DATA_DIR / "TGGATEs" / "WSI" / "Liver"
    manifest = build_liver_slide_manifest(
        wsi_dir,
        DATA_DIR / "corrected/open_tggates_pathological_image.csv",
        DATA_DIR / "corrected/open_tggates_individual.csv",
    )
    manifest = manifest[manifest["compound_name"] == "carbon tetrachloride"].copy()
    manifest = attach_pathology_findings(
        manifest, DATA_DIR / "corrected/open_tggates_pathology.csv", organ="Liver"
    )
    return manifest


def main() -> None:
    original = pd.read_parquet(ORIGINAL_MANIFEST)
    ccl4 = build_ccl4_manifest()

    shared_cols = [c for c in original.columns if c in ccl4.columns]
    missing = set(original.columns) - set(ccl4.columns)
    if missing:
        raise SystemExit(f"CCl4 manifest is missing columns present in the 219 one: {missing}")

    merged = pd.concat(
        [original[shared_cols], ccl4[shared_cols]], ignore_index=True
    ).sort_values("slide_id")
    if merged["slide_id"].duplicated().any():
        dupes = merged.loc[merged["slide_id"].duplicated(), "slide_id"].tolist()
        raise SystemExit(f"duplicate slide_id across the two manifests: {dupes}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(MERGED_MANIFEST_PATH, index=False)
    print(f"wrote {MERGED_MANIFEST_PATH}: {len(merged)} slides "
          f"({len(original)} original + {len(ccl4)} CCl4)")

    MERGED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    n_linked = 0
    for slide_id in merged["slide_id"]:
        dest = MERGED_FEATURES_DIR / f"{slide_id}.h5"
        if dest.exists() or dest.is_symlink():
            continue
        src = ORIGINAL_FEATURES_DIR / f"{slide_id}.h5"
        if not src.exists():
            for d in CCL4_FEATURE_DIRS:
                candidate = d / f"{slide_id}.h5"
                if candidate.exists():
                    src = candidate
                    break
        if not src.exists():
            raise SystemExit(f"no feature file found for slide_id={slide_id}")
        dest.symlink_to(src)
        n_linked += 1
    print(f"symlinked {n_linked} feature files into {MERGED_FEATURES_DIR}")


if __name__ == "__main__":
    main()
