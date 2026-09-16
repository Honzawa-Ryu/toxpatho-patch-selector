"""Read TRIDENT patch feature files and sample from them.

0001 が書いた `<job_dir>/<mag>_<size>px_<overlap>px_overlap/features_<encoder>/<slide_id>.h5`
を読む側の共通処理。0002（距離ベース）と0003（クラスタリング）の両方が同じ読み方を
するので、experiment.py ごとにコピーを持たない。
"""

from pathlib import Path

import h5py
import numpy as np


def read_slide_features(
    h5_path: Path, l2_normalize: bool = True
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read one slide's patch embeddings, patch coordinates and coord metadata.

    Args:
        h5_path: Path to a TRIDENT feature file. It holds `features`
            (n_patch, dim) and `coords` (n_patch, 2); the coords dataset
            carries the attributes needed to crop the patch back out of the
            WSI (`patch_size_level0`, `level0_magnification`, ...).
        l2_normalize: Scale every embedding to unit length. Distances then
            stop depending on overall embedding magnitude, which tracks
            staining intensity and would otherwise dominate any comparison
            across slides.

    Returns:
        (features (n_patch, dim) float32, coords (n_patch, 2) int64 in level-0
        pixels, coord attributes as a plain dict)
    """
    with h5py.File(h5_path, "r") as h:
        feats = np.asarray(h["features"][:], dtype=np.float32)
        coords = np.asarray(h["coords"][:], dtype=np.int64)
        attrs = {k: v for k, v in h["coords"].attrs.items()}
    if l2_normalize:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats /= np.clip(norms, 1e-8, None)
    return feats, coords, attrs


def sample_patch_matrix(
    feature_paths: dict[str, Path],
    slide_ids: list[str],
    patches_per_slide: int,
    rng: np.random.Generator,
    l2_normalize: bool = True,
) -> np.ndarray:
    """Stack a random per-slide subsample of patches into one matrix.

    Sampling a fixed number *per slide* rather than uniformly over all patches
    keeps slides with many patches from dominating whatever is fitted on the
    result (a PCA basis, a k-means model).
    """
    samples = []
    for slide_id in slide_ids:
        feats, _, _ = read_slide_features(feature_paths[slide_id], l2_normalize)
        take = min(patches_per_slide, len(feats))
        idx = rng.choice(len(feats), size=take, replace=False)
        samples.append(feats[idx])
    return np.concatenate(samples, axis=0)
