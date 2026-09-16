import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

import h5py
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


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
    parser.add_argument("--config", type=str, default="config.yml")
    return parser.parse_args()


def read_coords(h5_path: Path) -> tuple[np.ndarray, dict]:
    """Reuse the patch coordinates 0001 already computed, with their attributes."""
    with h5py.File(h5_path, "r") as h:
        coords = np.asarray(h["coords"][:], dtype=np.int64)
        attrs = {k: v for k, v in h["coords"].attrs.items()}
    return coords, attrs


def sample_patches(
    svs_path: str, coords: np.ndarray, patch_size: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Read n random patches from one slide as a (n, size, size, 3) uint8 array."""
    idx = rng.choice(len(coords), size=min(n, len(coords)), replace=False)
    slide = openslide.OpenSlide(svs_path)
    try:
        tiles = [
            np.asarray(
                slide.read_region((int(coords[i][0]), int(coords[i][1])), 0, (patch_size, patch_size)).convert("RGB")
            )
            for i in idx
        ]
    finally:
        slide.close()
    return np.stack(tiles)


class NormalizedPatchDataset(Dataset):
    """Patches of one slide, stain-normalized, ready for the patch encoder.

    The OpenSlide handle is opened lazily so that each DataLoader worker gets
    its own; an already-open handle cannot be pickled across processes.
    """

    def __init__(self, svs_path, coords, patch_size, matrix, transform):
        self.svs_path = str(svs_path)
        self.coords = coords
        self.patch_size = int(patch_size)
        self.matrix = matrix
        self.transform = transform
        self._slide = None

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, i):
        from lib.stain import apply_macenko

        if self._slide is None:
            self._slide = openslide.OpenSlide(self.svs_path)
        x, y = int(self.coords[i][0]), int(self.coords[i][1])
        rgb = np.asarray(
            self._slide.read_region((x, y), 0, (self.patch_size, self.patch_size)).convert("RGB")
        )
        if self.matrix is not None:
            rgb = apply_macenko(rgb, self.matrix)
        return self.transform(Image.fromarray(rgb))


def extract_slide(
    encoder,
    device: torch.device,
    dataset: NormalizedPatchDataset,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    """Run the patch encoder over one slide.

    autocast with the encoder's own precision, exactly as TRIDENT does in
    WSI.extract_patch_features — the point of this experiment is that the
    only difference from 0001 is the pixels, not the inference path.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    precision = getattr(encoder, "precision", torch.float32)
    out = []
    with torch.inference_mode():
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=precision, enabled=(precision != torch.float32)
            ):
                feats = encoder(imgs)
            out.append(feats.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.stain import fit_macenko, macenko_matrix
    from trident.patch_encoder_models.load import encoder_factory

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    variant_key = "macenko"

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)
    rng = np.random.default_rng(seed)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    manifest = pd.read_parquet(project_root / config["manifest_parquet"])
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)
    source_features = project_root / config["source_features_dir"]

    # 出力は0001と同じレイアウト（features_<encoder>/<slide_id>.h5）に揃えて、
    # 0003/0004が features_dir を差し替えるだけで読めるようにする。
    features_dir = project_root / "outputs" / exp_name / config["features_subdir"]
    features_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device} / 出力先: {features_dir}")

    patch_sizes, coords_cache = {}, {}
    for slide_id in manifest["slide_id"]:
        coords, attrs = read_coords(source_features / f"{slide_id}.h5")
        coords_cache[slide_id] = (coords, attrs)
        patch_sizes[slide_id] = int(attrs.get("patch_size_level0", attrs.get("patch_size", 256)))

    # ── 正規化の参照をデータ内から取る ─────────────────────────────────────────
    # 文献既定の参照濃度はこの切片群より淡く、そこに合わせると全体が褪色して
    # 好酸性の強弱そのものが潰れる。全実験のcontrolスライドからpatchを集めて
    # 「この施設の標準的な染色」を参照にする。
    control_ids = manifest.loc[manifest["dose_level"] == "Control", "slide_id"].tolist()
    ref_ids = list(rng.choice(control_ids, size=min(config["reference_slides"], len(control_ids)), replace=False))
    ref_pixels = np.concatenate(
        [
            sample_patches(
                manifest.loc[manifest["slide_id"] == sid, "svs_path"].iloc[0],
                coords_cache[sid][0],
                patch_sizes[sid],
                config["reference_patches_per_slide"],
                rng,
            ).reshape(-1, 3)
            for sid in ref_ids
        ]
    )
    he_ref, maxc_ref = fit_macenko(ref_pixels)
    logger.info(
        f"参照染色（control {len(ref_ids)}枚から推定）: max_c={np.round(maxc_ref, 3).tolist()}\n"
        f"he_ref=\n{np.round(he_ref, 3)}"
    )

    # ── スライドごとに正規化して埋め込み直す ──────────────────────────────────
    encoder = encoder_factory(config["patch_encoder"])
    encoder.to(device).eval()

    fit_rows = []
    done, skipped, failed = 0, 0, []
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        slide_id = row.slide_id
        out_path = features_dir / f"{slide_id}.h5"
        if out_path.exists():
            skipped += 1
            continue

        coords, attrs = coords_cache[slide_id]
        patch_size = patch_sizes[slide_id]
        try:
            sample = sample_patches(
                row.svs_path, coords, patch_size, config["fit_patches_per_slide"], rng
            )
            he, max_c = fit_macenko(sample.reshape(-1, 3))
            matrix = macenko_matrix(he, max_c, he_ref=he_ref, maxc_ref=maxc_ref)
        except ValueError as e:
            # 染色ベクトルの推定に失敗したスライドは、正規化せずに通すと
            # 他と条件が違う埋め込みが混ざるので、飛ばして記録する。
            logger.warning(f"{slide_id}: 染色推定に失敗したためスキップ: {e}")
            failed.append({"slide_id": slide_id, "reason": str(e)})
            continue

        fit_rows.append(
            {
                "slide_id": slide_id,
                "compound_name": row.compound_name,
                "dose_level": row.dose_level,
                "exp_id": int(row.exp_id),
                "max_c_h": float(max_c[0]),
                "max_c_e": float(max_c[1]),
                "he_cosine": float(abs(he[:, 0] @ he[:, 1])),
            }
        )

        dataset = NormalizedPatchDataset(
            row.svs_path, coords, patch_size, matrix, encoder.eval_transforms
        )
        features = extract_slide(
            encoder, device, dataset, config["batch_size"], config["num_workers"]
        )

        tmp_path = out_path.with_suffix(".h5.tmp")
        with h5py.File(tmp_path, "w") as h:
            fset = h.create_dataset("features", data=features.astype(np.float32))
            fset.attrs["encoder"] = config["patch_encoder"]
            fset.attrs["name"] = slide_id
            fset.attrs["stain_normalization"] = "macenko"
            cset = h.create_dataset("coords", data=coords)
            for k, v in attrs.items():
                cset.attrs[k] = v
        tmp_path.rename(out_path)  # 途中で落ちた.h5を完成品と誤認しないための原子的置換

        done += 1
        logger.info(f"[{i}/{len(manifest)}] {slide_id}: {features.shape} -> {out_path.name}")

    fits = pd.DataFrame(fit_rows)
    if not fits.empty:
        fits.to_parquet(run_dir / "stain_fits.parquet", index=False)
        logger.info(
            "スライドごとの推定濃度（実験ごとの染色差の大きさがここに出る）:\n"
            + fits.groupby("compound_name")[["max_c_h", "max_c_e"]].agg(["mean", "std"]).round(3).to_string()
        )

    results = {
        "n_slides": len(manifest),
        "n_extracted": done,
        "n_skipped_existing": skipped,
        "n_failed": len(failed),
        "failed": failed,
        "features_dir": str(features_dir),
        "reference_slides": ref_ids,
        "reference_max_c": maxc_ref.tolist(),
        "reference_he": he_ref.tolist(),
        "patch_encoder": config["patch_encoder"],
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
