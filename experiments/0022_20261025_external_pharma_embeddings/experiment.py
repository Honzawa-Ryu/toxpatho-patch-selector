import argparse
import importlib.util
import json
import logging
import os
import sys
import time
import zlib
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

import cv2
import h5py
import openslide
import torch
from PIL import Image, ImageDraw


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
    parser.add_argument("--config", default="config.yml")   # run_slurm.shが渡す。読むのは常に実験ディレクトリのconfig.yml
    return parser.parse_args()


def liver_mask(entry, bench, cfg, thumb):
    """tggate-bench 0013 と同じ肝臓の範囲（組織片の番号、または手描きの多角形と組織の積）。縮小画像の画素単位。"""
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
    tissue = (gray < cfg["white_gray"]) & (hsv[..., 1] / 255.0 > cfg["min_sat"])
    if entry["method"] == "pieces":
        lab = np.asarray(Image.open(bench / cfg["pieces_exp"] / entry["segmentation"] / "labels" / f"{entry['slide_id']}.png"))
        if lab.shape != tissue.shape:
            lab = np.asarray(Image.fromarray(lab).resize((tissue.shape[1], tissue.shape[0]), Image.NEAREST))
        return np.isin(lab, entry["pieces"]) & tissue
    w, h = entry["thumb_size"]
    img = Image.new("L", (w, h), 0)
    for poly in entry["polygons"]:
        ImageDraw.Draw(img).polygon([tuple(p) for p in poly], fill=1)
    return np.asarray(img.resize((thumb.shape[1], thumb.shape[0]), Image.NEAREST)).astype(bool) & tissue


def grid_coords(mask, ds, size0, W, H, min_frac):
    """肝臓の範囲の中の、重ならない格子（level 0 の左上の座標）。縮小画像で組織割合 min_frac 以上のものだけ。"""
    xs, ys = np.arange(0, W - size0 + 1, size0), np.arange(0, H - size0 + 1, size0)
    out = []
    ii = np.cumsum(np.cumsum(np.pad(mask, ((1, 0), (1, 0))).astype(np.int64), 0), 1)   # 積分画像
    for y in ys:
        y0, y1 = int(y / ds), max(int((y + size0) / ds), int(y / ds) + 1)
        for x in xs:
            x0, x1 = int(x / ds), max(int((x + size0) / ds), int(x / ds) + 1)
            y1c, x1c = min(y1, mask.shape[0]), min(x1, mask.shape[1])
            if y0 >= mask.shape[0] or x0 >= mask.shape[1]:
                continue
            area = (y1c - y0) * (x1c - x0)
            s_ = ii[y1c, x1c] - ii[y0, x1c] - ii[y1c, x0] + ii[y0, x0]
            if area and s_ / area >= min_frac:
                out.append((x, y))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.stain import fit_macenko, macenko_matrix

    spec = importlib.util.spec_from_file_location("exp0005", project_root / "experiments/0005_20260913_stain_normalized_embeddings/experiment.py")
    exp0005 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp0005)
    parse_args()
    exp_name = os.environ["EXP_NAME"]
    cfg = load_config(Path(__file__).parent)
    run_dir = get_run_dir(project_root, __file__, "default", output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key="default", config=cfg)
    bench = Path(cfg["bench_root"])
    slides = pd.read_csv(bench / cfg["slides_csv"], dtype={"slide_id": str})
    roi = {e["slide_id"]: e for e in json.loads((bench / cfg["roi_json"]).read_text())["slides"]}
    ref = json.loads((project_root / cfg["reference_json"]).read_text())
    he_ref, maxc_ref = np.asarray(ref["reference_he"]), np.asarray(ref["reference_max_c"])
    from trident.patch_encoder_models.load import encoder_factory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder_factory(cfg["patch_encoder"]).to(device).eval()
    fdir = project_root / "outputs" / exp_name / "features_uni_v2_macenko"
    fdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(slides.itertuples(), start=1):
        key = r.slide_id.replace(" ", "_")
        out = fdir / f"{key}.h5"
        if out.exists() or r.slide_id not in roi:
            rows.append({"slide_id": r.slide_id, "key": key, "status": "skip" if out.exists() else "no_roi"})
            continue
        t0 = time.time()
        svs = str(bench / r.path)
        w = openslide.OpenSlide(svs)
        W, H = w.dimensions
        mpp = float(w.properties.get(openslide.PROPERTY_NAME_MPP_X, r.mpp_x))
        thumb = np.asarray(w.get_thumbnail((cfg["thumb_size"], cfg["thumb_size"])).convert("RGB"))
        w.close()
        ds = W / thumb.shape[1]
        size0 = int(round(cfg["patch_px"] * cfg["target_mpp"] / mpp))
        mask = liver_mask(roi[r.slide_id], bench, cfg, thumb)
        coords = grid_coords(mask, ds, size0, W, H, cfg["min_tissue_frac"])
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(key.encode())])
        try:
            sample = exp0005.sample_patches(svs, coords, size0, cfg["fit_patches_per_slide"], rng)
            he, max_c = fit_macenko(sample.reshape(-1, 3))
            matrix = macenko_matrix(he, max_c, he_ref=he_ref, maxc_ref=maxc_ref)
        except ValueError as e:
            logger.warning(f"{key}: stain fit failed, skipped: {e}")
            rows.append({"slide_id": r.slide_id, "key": key, "status": f"stain_failed: {e}"})
            continue
        ds_ = exp0005.NormalizedPatchDataset(svs, coords, size0, matrix, encoder.eval_transforms)
        feats = exp0005.extract_slide(encoder, device, ds_, cfg["batch_size"], cfg["num_workers"])
        tmp = out.with_suffix(".h5.tmp")
        with h5py.File(tmp, "w") as h:
            fset = h.create_dataset("features", data=feats.astype(np.float32))
            fset.attrs["encoder"] = cfg["patch_encoder"]
            fset.attrs["stain_normalization"] = "macenko"
            cset = h.create_dataset("coords", data=coords)
            cset.attrs["patch_size_level0"] = size0
            cset.attrs["mpp"] = mpp
            cset.attrs["target_mpp"] = cfg["target_mpp"]
        tmp.rename(out)
        rows.append({"slide_id": r.slide_id, "key": key, "status": "ok", "provider": r.provider, "study": r.study, "n_patches": len(coords),
                     "patch_size_level0": size0, "mpp": mpp, "max_c_h": float(max_c[0]), "max_c_e": float(max_c[1]), "seconds": round(time.time() - t0, 1)})
        logger.info(f"[{i}/{len(slides)}] {key} ({r.provider}): {len(coords)} patches, size0 {size0}, {time.time() - t0:.0f}s")
    pd.DataFrame(rows).to_csv(run_dir / "extraction.csv", index=False)
    slides.to_csv(run_dir / "slides.csv", index=False)
    (run_dir / "results.json").write_text(json.dumps({"n_ok": sum(x["status"] == "ok" for x in rows), "n_slides": len(slides)}, indent=2))
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
