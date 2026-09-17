import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument(
        "--chunk_id", type=int, required=True,
        help="Which slice of the compound's slides this array task processes "
        "(0-indexed, < n_chunks in config.yml).",
    )
    return parser.parse_args()


def run_trident_pipeline(
    wsi_dir: Path,
    job_dir: Path,
    custom_list_of_wsis: Path,
    *,
    trident_script: Path,
    task: str,
    segmenter: str,
    reader_type: str,
    mag: float,
    patch_size: int,
    overlap: int,
    patch_encoder: str,
    gpus: list[int],
) -> None:
    """Run TRIDENT's full pipeline (segmentation -> coords -> features) over
    just the slides listed in custom_list_of_wsis (see IO.py's
    `list_valid_wsis`: a CSV with a 'wsi' column of paths relative to
    wsi_dir). wsi_dir stays the full shared Liver directory — untouched,
    read-only — so nothing needs symlinking; the CSV alone scopes each
    array task to its own chunk of slides, and job_dir is per-chunk so
    concurrent array tasks never write into the same TRIDENT state files.

    Raise (don't swallow) on a non-zero return code, same rationale as 0001.
    """
    argv = [
        sys.executable, str(trident_script),
        "--task", task,
        "--wsi_dir", str(wsi_dir),
        "--search_nested",
        "--custom_list_of_wsis", str(custom_list_of_wsis),
        "--job_dir", str(job_dir),
        "--reader_type", reader_type,
        "--segmenter", segmenter,
        "--mag", str(mag),
        "--patch_size", str(patch_size),
        "--overlap", str(overlap),
        "--patch_encoder", patch_encoder,
        "--gpus", *[str(g) for g in gpus],
    ]
    subprocess.run(argv, check=True)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.tggates_metadata import build_liver_slide_manifest

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)
    compound_name: str = config["compound_name"]
    n_chunks: int = config["n_chunks"]

    variant_key = f"chunk_{args.chunk_id}"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:      {run_dir}")
    logger.info(f"dataset_dir:  {dataset_dir}")
    logger.info(f"compound:     {compound_name}")
    logger.info(f"chunk:        {args.chunk_id} / {n_chunks}")

    organ = config.get("organ", "Liver")
    wsi_dir = dataset_dir / "TGGATEs" / "WSI" / organ
    pathological_image_csv = dataset_dir / config.get(
        "pathological_image_csv", "corrected/open_tggates_pathological_image.csv"
    )
    individual_csv = dataset_dir / config.get(
        "individual_csv", "corrected/open_tggates_individual.csv"
    )

    manifest = build_liver_slide_manifest(wsi_dir, pathological_image_csv, individual_csv)
    compound_manifest = manifest[manifest["compound_name"] == compound_name].sort_values(
        "slide_id"
    ).reset_index(drop=True)
    if compound_manifest.empty:
        raise ValueError(f"No local slides found for compound_name={compound_name!r}")

    chunk_indices = np.array_split(np.arange(len(compound_manifest)), n_chunks)[args.chunk_id]
    chunk_manifest = compound_manifest.iloc[chunk_indices].reset_index(drop=True)
    manifest_path = run_dir / "slide_manifest.parquet"
    chunk_manifest.to_parquet(manifest_path, index=False)
    logger.info(
        f"Wrote manifest: {manifest_path} "
        f"({len(chunk_manifest)}/{len(compound_manifest)} {compound_name} slides)"
    )

    # TRIDENT scopes to this chunk via a CSV of paths relative to wsi_dir
    # (see trident.IO.list_valid_wsis) — sacrifice_period/slide_id.svs,
    # matching how the slides are actually laid out under wsi_dir.
    rel_paths = [
        f"{str(period).replace(' ', '_')}/{sid}.svs"
        for period, sid in zip(chunk_manifest["sacrifice_period"], chunk_manifest["slide_id"])
    ]
    custom_list_path = run_dir / "custom_list_of_wsis.csv"
    pd.DataFrame({"wsi": rel_paths}).to_csv(custom_list_path, index=False)

    job_dir = project_root / "outputs" / exp_name / f"trident_job_chunk{args.chunk_id}"

    run_trident_pipeline(
        wsi_dir,
        job_dir,
        custom_list_path,
        trident_script=project_root / "libraries" / "trident_run_batch_of_slides.py",
        task=config.get("trident_task", "all"),
        segmenter=config.get("segmenter", "hest"),
        reader_type=config.get("reader_type", "openslide"),
        mag=config.get("mag", 20.0),
        patch_size=config.get("patch_size", 256),
        overlap=config.get("overlap", 0),
        patch_encoder=config.get("patch_encoder", "uni_v2"),
        gpus=config.get("gpus", [0]),
    )
    logger.info(f"TRIDENT job_dir: {job_dir}")

    results: dict = {
        "compound_name": compound_name,
        "chunk_id": args.chunk_id,
        "n_chunks": n_chunks,
        "n_slides": len(chunk_manifest),
        "manifest_path": str(manifest_path),
        "job_dir": str(job_dir),
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
