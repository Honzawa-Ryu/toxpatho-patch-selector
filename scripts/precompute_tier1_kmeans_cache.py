"""Precompute the PCA-projected patch embeddings for exp0009's tier1 corpus
once, and cache them to disk, so the kmeans_k sweep (100..2000, 20 values)
doesn't redo the same ~55M-patch read-and-project pass for every k.

experiment.py's original loop opens each slide's .h5 one at a time (single
process) — for exp0006's 219-slide corpus that was a few minutes, but tier1's
3,441 slides made it the dominant cost (~1.5h) with the GPU sitting idle the
whole time (PCA transform and h5 reads are both CPU-bound). This script does
the same read-and-project work with a process pool across slides, then writes
straight into a shared memmap so workers never have to ship ~14GB of float32
back through IPC.

Usage: python scripts/precompute_tier1_kmeans_cache.py
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "experiments/0009_20260919_tier1_corpus_clustering"
CACHE_DIR = PROJECT_ROOT / "outputs/0009_20260919_tier1_corpus_clustering/embedding_cache"
N_WORKERS = 24


def _load_shared_config() -> dict:
    with open(EXP_DIR / "config_kmeans_k_sweep.yml") as f:
        return yaml.safe_load(f)


def _slide_n_patches(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as h:
        return h["features"].shape[0]


def _sample_one_slide(args: tuple) -> np.ndarray:
    h5_path, take, l2_normalize, seed = args
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as h:
        feats = np.asarray(h["features"][:], dtype=np.float32)
    if l2_normalize:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats /= np.clip(norms, 1e-8, None)
    take = min(take, len(feats))
    idx = rng.choice(len(feats), size=take, replace=False)
    return feats[idx]


def _project_one_slide(args: tuple) -> tuple[str, int]:
    """Read one slide, PCA-project it, write straight into the shared memmaps."""
    slide_id, h5_path, offset, l2_normalize, components, mean = args
    with h5py.File(h5_path, "r") as h:
        feats = np.asarray(h["features"][:], dtype=np.float32)
        coords = np.asarray(h["coords"][:], dtype=np.float32)
    if l2_normalize:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats /= np.clip(norms, 1e-8, None)
    projected = (feats - mean) @ components.T

    n = len(projected)
    emb_mm = np.load(CACHE_DIR / "embeddings.npy", mmap_mode="r+")
    xy_mm = np.load(CACHE_DIR / "xy.npy", mmap_mode="r+")
    emb_mm[offset : offset + n] = projected.astype(np.float32)
    xy_mm[offset : offset + n] = coords
    emb_mm.flush()
    xy_mm.flush()
    del emb_mm, xy_mm
    return slide_id, n


def main() -> None:
    config = _load_shared_config()
    seed = config.get("seed", 42)
    l2_normalize = config.get("l2_normalize", True)
    pca_components_n = config["pca_components"]
    per_group = config["pca_patches_per_group"]

    manifest = pd.read_parquet(PROJECT_ROOT / config["manifest_parquet"])
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)
    features_dir = PROJECT_ROOT / config["features_dir"]
    feature_paths = {p.stem: p for p in sorted(features_dir.glob("*.h5"))}
    slide_ids = manifest["slide_id"].tolist()
    missing = sorted(set(slide_ids) - set(feature_paths))
    if missing:
        raise SystemExit(f"{len(missing)} slide(s) have no feature file: {missing[:10]}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── パス1: 各スライドのpatch数だけ調べる（データは読まない、shapeだけ）──────
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        counts = list(ex.map(_slide_n_patches, [feature_paths[s] for s in slide_ids]))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_total = int(offsets[-1])
    print(f"[{time.time() - t0:.0f}s] {len(slide_ids)} slides, {n_total} patches total")

    # ── PCA fit（0006/experiment.pyと同じ、control/treated別サンプリング）───────
    rng = np.random.default_rng(seed)
    control_ids = manifest.loc[manifest["dose_level"] == "Control", "slide_id"].tolist()
    treated_ids = manifest.loc[manifest["dose_level"] != "Control", "slide_id"].tolist()
    fit_jobs = [
        (feature_paths[s], per_group // len(control_ids), l2_normalize, seed + i)
        for i, s in enumerate(control_ids)
    ] + [
        (feature_paths[s], per_group // len(treated_ids), l2_normalize, seed + len(control_ids) + i)
        for i, s in enumerate(treated_ids)
    ]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        fit_matrix = np.concatenate(list(ex.map(_sample_one_slide, fit_jobs)))
    pca = PCA(n_components=pca_components_n, svd_solver="randomized", random_state=seed)
    pca.fit(fit_matrix)
    explained_variance = float(pca.explained_variance_ratio_.sum())
    print(f"[{time.time() - t0:.0f}s] PCA: {fit_matrix.shape} -> {pca_components_n}次元, 寄与率 {explained_variance:.3f}")
    np.save(CACHE_DIR / "pca_explained_variance.npy", np.array(explained_variance))
    del fit_matrix

    # ── パス2: 全patchを読んでPCA投影し、共有memmapに直接書き込む ───────────────
    np.lib.format.open_memmap(
        CACHE_DIR / "embeddings.npy", mode="w+", dtype=np.float32, shape=(n_total, pca_components_n)
    )
    np.lib.format.open_memmap(CACHE_DIR / "xy.npy", mode="w+", dtype=np.float32, shape=(n_total, 2))

    components = pca.components_.astype(np.float32)
    mean = pca.mean_.astype(np.float32)
    project_jobs = [
        (s, feature_paths[s], int(offsets[i]), l2_normalize, components, mean)
        for i, s in enumerate(slide_ids)
    ]
    slide_index = np.empty(n_total, dtype=np.int32)
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(_project_one_slide, job) for job in project_jobs]
        for i, fut in enumerate(as_completed(futures)):
            fut.result()
            done += 1
            if done % 200 == 0 or done == len(project_jobs):
                print(f"[{time.time() - t0:.0f}s] projected {done}/{len(project_jobs)} slides")
    for i, n in enumerate(counts):
        slide_index[offsets[i] : offsets[i] + n] = i
    np.save(CACHE_DIR / "slide_index.npy", slide_index)
    np.save(CACHE_DIR / "slide_ids.npy", np.array(slide_ids))

    # ── 共通サブサンプル（本体run・0006と同じ200,000点）──────────────────────
    n_sub = min(config["subsample"], n_total)
    sub_idx = rng.choice(n_total, size=n_sub, replace=False)
    np.save(CACHE_DIR / "sub_idx.npy", sub_idx)

    print(f"[{time.time() - t0:.0f}s] done. cache written to {CACHE_DIR}")


if __name__ == "__main__":
    main()
