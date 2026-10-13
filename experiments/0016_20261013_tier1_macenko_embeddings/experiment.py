import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd
import torch

import h5py


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--chunk", type=int, required=True)   # 0..n_chunks-1（array task）
    return parser.parse_args()


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.stain import fit_macenko, macenko_matrix
    from trident.patch_encoder_models.load import encoder_factory

    # 0005の抽出部品（座標読み込み・patch抽出・正規化Dataset・推論）をそのまま使う。
    # TODO: 共通化するならlib/へ移す（未着手）。
    spec = importlib.util.spec_from_file_location(
        "exp0005", project_root / "experiments/0005_20260913_stain_normalized_embeddings/experiment.py")
    exp0005 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp0005)

    args = parse_args()
    exp_name = os.environ["EXP_NAME"]
    config = load_config(Path(__file__).parent)
    variant_key = f"chunk{args.chunk}"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, n_chunks=config["n_chunks"])
    logger.info(f"Starting: {exp_name} / {variant_key}")

    manifest = pd.read_parquet(project_root / config["manifest_parquet"])
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)
    chunk = manifest.iloc[args.chunk::config["n_chunks"]]
    coords_dir = project_root / config["coords_dir"]
    features_dir = project_root / "outputs" / exp_name / config["features_subdir"]
    features_dir.mkdir(parents=True, exist_ok=True)

    # 参照染色: 各実験のControlから一様に抽出（seed固定で全chunk共通）
    rng_ref = np.random.default_rng(config["seed"])
    ctrl = manifest[manifest.dose_level == "Control"]
    ref_ids = [sid for _, g in ctrl.groupby("exp_id")
               for sid in rng_ref.choice(g.slide_id.to_numpy(), size=min(config["reference_slides_per_exp"], len(g)), replace=False)]
    pixels = []
    for sid in ref_ids:
        coords, attrs = exp0005.read_coords(coords_dir / f"{sid}.h5")
        size = int(attrs.get("patch_size_level0", attrs.get("patch_size", 256)))
        svs = manifest.loc[manifest.slide_id == sid, "svs_path"].iloc[0]
        pixels.append(exp0005.sample_patches(svs, coords, size, config["reference_patches_per_slide"], rng_ref).reshape(-1, 3))
    he_ref, maxc_ref = fit_macenko(np.concatenate(pixels))
    (run_dir / "reference.json").write_text(json.dumps(
        {"reference_slides": ref_ids, "reference_he": he_ref.tolist(), "reference_max_c": maxc_ref.tolist()}, indent=2))
    logger.info(f"参照染色（Control {len(ref_ids)}枚・{ctrl.exp_id.nunique()}実験）: max_c={np.round(maxc_ref, 3).tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder_factory(config["patch_encoder"])
    encoder.to(device).eval()
    logger.info(f"device: {device} / {len(chunk)} slides in this chunk / 出力先: {features_dir}")

    fit_rows, failed = [], []
    done = skipped = 0
    for i, row in enumerate(chunk.itertuples(index=False), start=1):
        sid = row.slide_id
        out_path = features_dir / f"{sid}.h5"
        if out_path.exists():
            skipped += 1
            continue
        coords, attrs = exp0005.read_coords(coords_dir / f"{sid}.h5")
        patch_size = int(attrs.get("patch_size_level0", attrs.get("patch_size", 256)))
        rng = np.random.default_rng([config["seed"], int(sid)])   # chunk分割に依存しないslide固有の乱数
        try:
            sample = exp0005.sample_patches(row.svs_path, coords, patch_size, config["fit_patches_per_slide"], rng)
            he, max_c = fit_macenko(sample.reshape(-1, 3))
            matrix = macenko_matrix(he, max_c, he_ref=he_ref, maxc_ref=maxc_ref)
        except ValueError as e:
            # 推定に失敗したslideを正規化なしで混ぜると条件が揃わないので、飛ばして記録する（0005と同じ）
            logger.warning(f"{sid}: 染色推定に失敗したためスキップ: {e}")
            failed.append({"slide_id": sid, "reason": str(e)})
            continue
        fit_rows.append({"slide_id": sid, "compound_name": row.compound_name, "dose_level": row.dose_level,
                         "exp_id": int(row.exp_id), "max_c_h": float(max_c[0]), "max_c_e": float(max_c[1]),
                         "he_cosine": float(abs(he[:, 0] @ he[:, 1]))})
        dataset = exp0005.NormalizedPatchDataset(row.svs_path, coords, patch_size, matrix, encoder.eval_transforms)
        features = exp0005.extract_slide(encoder, device, dataset, config["batch_size"], config["num_workers"])
        tmp_path = out_path.with_suffix(".h5.tmp")
        with h5py.File(tmp_path, "w") as h:
            fset = h.create_dataset("features", data=features.astype(np.float32))
            fset.attrs["encoder"] = config["patch_encoder"]
            fset.attrs["name"] = sid
            fset.attrs["stain_normalization"] = "macenko"
            cset = h.create_dataset("coords", data=coords)
            for k, v in attrs.items():
                cset.attrs[k] = v
        tmp_path.rename(out_path)  # 途中で落ちた.h5を完成品と誤認しないための原子的置換
        done += 1
        logger.info(f"[{i}/{len(chunk)}] {sid}: {features.shape}")
        if done % 50 == 0:  # 長時間ジョブなので途中経過も残す
            pd.DataFrame(fit_rows).to_parquet(run_dir / "stain_fits.parquet", index=False)

    if fit_rows:
        pd.DataFrame(fit_rows).to_parquet(run_dir / "stain_fits.parquet", index=False)
    (run_dir / "results.json").write_text(json.dumps(
        {"chunk": args.chunk, "n_slides": len(chunk), "n_extracted": done,
         "n_skipped_existing": skipped, "n_failed": len(failed), "failed": failed,
         "features_dir": str(features_dir), "reference_slides": ref_ids}, indent=2, ensure_ascii=False))
    if failed:
        logger.warning(f"{len(failed)} slides failed (see results.json)")
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
