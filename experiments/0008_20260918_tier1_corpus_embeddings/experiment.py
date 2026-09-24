import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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
        help="Which slice of the not-yet-embedded slides this array task processes "
        "(0-indexed, < n_chunks in config.yml).",
    )
    return parser.parse_args()


def stage_slides_locally(
    rel_paths: list[str], nfs_wsi_dir: Path, staging_dir: Path, logger: logging.Logger
) -> Path:
    """Copy this chunk's .svs files from the NFS-backed wsi_dir to a shared,
    persistent local-SSD staging directory, skipping files already staged.

    Segmentation reads a WSI with many small/seek-heavy accesses (tissue
    detection scans across pyramid levels), which is what made the direct
    NFS run I/O-bound (most worker processes sitting in D-state). Copying
    each file once, as one large sequential read, is NFS-friendly; every
    read after that hits local NVMe instead.

    staging_dir is NOT the per-job SCRATCH_DIR (that gets wiped when the job
    ends) — it's a fixed shared path so a resubmitted array task, or another
    chunk that happens to need an overlapping slide, doesn't re-copy work
    that's already there.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for rel_path in rel_paths:
        dest = staging_dir / rel_path
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = nfs_wsi_dir / rel_path
        tmp_dest = dest.with_suffix(dest.suffix + ".part")
        shutil.copy2(src, tmp_dest)
        tmp_dest.rename(dest)  # atomic-ish: a concurrent chunk never sees a partial file
        n_copied += 1
        if n_copied % 50 == 0:
            logger.info(f"staged {n_copied}/{len(rel_paths)} new slides so far")
    logger.info(f"staging done: {n_copied} newly copied, {len(rel_paths) - n_copied} already present")
    return staging_dir


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
    """Same TRIDENT invocation as 0007 (see that experiment.py's docstring for
    why --custom_list_of_wsis is used instead of symlinking a subset
    directory): wsi_dir stays the full shared Liver directory, and the CSV
    alone scopes this array task to its chunk of not-yet-embedded slides.
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
    n_chunks: int = config["n_chunks"]

    variant_key = f"chunk_{args.chunk_id}"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:      {run_dir}")
    logger.info(f"dataset_dir:  {dataset_dir}")
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

    already_embedded: set[str] = set()
    for rel_dir in config["exclude_features_dirs"]:
        d = project_root / rel_dir
        already_embedded |= {p.stem for p in d.glob("*.h5")}
    logger.info(f"already embedded elsewhere: {len(already_embedded)} slides")

    remaining = manifest[~manifest["slide_id"].isin(already_embedded)].sort_values(
        "slide_id"
    ).reset_index(drop=True)
    if remaining.empty:
        raise ValueError("No slides left to embed — exclude_features_dirs already covers everything local")

    chunk_indices = np.array_split(np.arange(len(remaining)), n_chunks)[args.chunk_id]
    chunk_manifest = remaining.iloc[chunk_indices].reset_index(drop=True)
    manifest_path = run_dir / "slide_manifest.parquet"
    chunk_manifest.to_parquet(manifest_path, index=False)
    logger.info(
        f"Wrote manifest: {manifest_path} "
        f"({len(chunk_manifest)}/{len(remaining)} not-yet-embedded slides)"
    )

    def rel_paths_for(df: pd.DataFrame) -> list[str]:
        return [
            f"{str(period).replace(' ', '_')}/{sid}.svs"
            for period, sid in zip(df["sacrifice_period"], df["slide_id"])
        ]

    job_dir = project_root / "outputs" / exp_name / f"trident_job_chunk{args.chunk_id}"
    staging_dir_cfg = config.get("staging_dir")
    trident_kwargs = dict(
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

    if not staging_dir_cfg:
        # ステージング無し（従来通りNFSを直接読む）
        custom_list_path = run_dir / "custom_list_of_wsis.csv"
        pd.DataFrame({"wsi": rel_paths_for(chunk_manifest)}).to_csv(custom_list_path, index=False)
        run_trident_pipeline(wsi_dir, job_dir, custom_list_path, **trident_kwargs)
    else:
        # ステージングと処理をパイプライン化する。サブバッチiを処理している間に、
        # バックグラウンドスレッドでサブバッチi+1のステージングを進めておく
        # （NFS読み込み=I/O待ちとTRIDENT実行=GPU/CPU計算は別リソースなので重ねられる）。
        # 2026-09-18: 「全689枚ステージングし終えてから処理開始」だと速度差の
        # 実測に12時間近く待つ必要があったため、サブバッチ化した。
        staging_dir = Path(staging_dir_cfg)
        batch_size = config.get("staging_batch_size", 30)
        batches = [
            chunk_manifest.iloc[i : i + batch_size].reset_index(drop=True)
            for i in range(0, len(chunk_manifest), batch_size)
        ]
        logger.info(f"{len(batches)} sub-batches of up to {batch_size} slides each")

        executor = ThreadPoolExecutor(max_workers=1)
        stage_slides_locally(rel_paths_for(batches[0]), wsi_dir, staging_dir, logger)

        for i, batch in enumerate(batches):
            next_stage = None
            if i + 1 < len(batches):
                next_stage = executor.submit(
                    stage_slides_locally, rel_paths_for(batches[i + 1]), wsi_dir, staging_dir, logger
                )

            batch_list_path = run_dir / f"custom_list_of_wsis_batch{i}.csv"
            pd.DataFrame({"wsi": rel_paths_for(batch)}).to_csv(batch_list_path, index=False)
            logger.info(f"--- batch {i + 1}/{len(batches)}: processing {len(batch)} slides ---")
            run_trident_pipeline(staging_dir, job_dir, batch_list_path, **trident_kwargs)

            if next_stage is not None:
                next_stage.result()  # 次バッチの処理を始める前に、ステージング完了を待つ

        executor.shutdown()
    logger.info(f"TRIDENT job_dir: {job_dir}")

    results: dict = {
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
