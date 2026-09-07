import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

# --- Deep learning (uncomment if needed) ---
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset

# --- Visualization (uncomment if needed) ---
# import matplotlib.pyplot as plt
# import seaborn as sns


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
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
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    """Define CLI args for all variable dimensions (used in GRID_ARGS / RUN_COMMAND)."""
    parser = argparse.ArgumentParser()
    # RUN_MODE is "single" (see run_slurm.sh) so there is no swept dimension.
    # --config is declared (but unused — config.yml is loaded from this
    # script's own directory via load_config()) only because RUN_COMMAND in
    # run_slurm.sh always passes it; without this, argparse would reject it
    # as an unrecognized argument.
    parser.add_argument("--config", type=str, default="config.yml")
    return parser.parse_args()


def run_trident_pipeline(
    wsi_dir: Path,
    job_dir: Path,
    *,
    task: str,
    segmenter: str,
    reader_type: str,
    mag: float,
    patch_size: int,
    overlap: int,
    patch_encoder: str,
    gpus: list[int],
) -> None:
    """Run TRIDENT's full pipeline (segmentation -> coords -> features) once
    over every .svs under wsi_dir, writing into job_dir.

    Implement this as a subprocess call to the `run_batch_of_slides` console
    script that `pip install`ing the `trident` package puts on PATH (see
    pyproject.toml's [tool.uv.sources] entry — this is mahmoodlab/TRIDENT,
    NOT the unrelated PyPI package also named "trident").

    Required flags for this experiment:
        --task <task>               ("all" = segmenter -> coords -> feat in one call)
        --wsi_dir <wsi_dir>
        --search_nested             (slides live under wsi_dir/{4,8,15,29}_day/)
        --job_dir <job_dir>
        --reader_type <reader_type>
        --segmenter <segmenter>
        --mag <mag>
        --patch_size <patch_size>
        --overlap <overlap>
        --patch_encoder <patch_encoder>  (uni_v2 = UNI2-h; gated on HuggingFace,
                                           needs HF_TOKEN in the environment —
                                           already exported in this shell, and
                                           #SBATCH --export=ALL carries it into
                                           the job)
        --gpus <space-separated ints>

    TRIDENT tracks per-slide progress under job_dir/wsi_states/ and skips
    slides already marked done on rerun ("smart resume") — that's exactly
    what makes this safe to resubmit if the 48h job times out partway.
    Re-running this experiment.py a second time after a full success should
    be a no-op regardless, since get_run_dir()'s completed-guard in main()
    already short-circuits before this function would even be called.

    Raise (don't swallow) on a non-zero return code — a partial/failed
    TRIDENT run must not be recorded as complete_run() in main().
    """
    argv = [
        "run_batch_of_slides",
        "--task", task,
        "--wsi_dir", str(wsi_dir),
        "--search_nested",
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

    # RUN_MODE is "single" — one atomic run over the whole Liver WSI corpus,
    # not a sweep — so there's only one variant.
    variant_key = "trident_all_slides"

    # Initialize output directory (exits immediately if already completed).
    # Written on OUTPUT_ROOT (scratch) when set; scripts/slurm_entry.sh
    # rsyncs it back to project_root/outputs/ at job end.
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    organ = config.get("organ", "Liver")
    wsi_dir = dataset_dir / "TGGATEs" / "WSI" / organ
    job_dir = project_root / "outputs" / exp_name / "trident_job"
    pathological_image_csv = dataset_dir / config.get(
        "pathological_image_csv", "corrected/open_tggates_pathological_image.csv"
    )
    individual_csv = dataset_dir / config.get(
        "individual_csv", "corrected/open_tggates_individual.csv"
    )

    manifest = build_liver_slide_manifest(wsi_dir, pathological_image_csv, individual_csv)
    manifest_path = run_dir / "slide_manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    logger.info(f"Wrote manifest: {manifest_path} ({len(manifest)} slides)")

    run_trident_pipeline(
        wsi_dir,
        job_dir,
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
        "n_slides": len(manifest),
        "manifest_path": str(manifest_path),
        "job_dir": str(job_dir),
    }

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()