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
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


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
    # RUN_MODE is "single" — nothing is swept. --config is declared only
    # because RUN_COMMAND in run_slurm.sh always passes it (config.yml is
    # actually loaded from this script's own directory via load_config()).
    parser.add_argument("--config", type=str, default="config.yml")
    return parser.parse_args()


def read_slide_features(h5_path: Path, l2_normalize: bool) -> tuple[np.ndarray, np.ndarray]:
    """Read one slide's TRIDENT feature file.

    Returns:
        (features (n_patch, 1536) float32, coords (n_patch, 2) int64)

    L2-normalizing each patch embedding makes the downstream distances
    scale-invariant, which matters here because overall staining intensity
    varies slide to slide and we do not want that to dominate "distance from
    control".
    """
    with h5py.File(h5_path, "r") as h:
        feats = np.asarray(h["features"][:], dtype=np.float32)
        coords = np.asarray(h["coords"][:], dtype=np.int64)
    if l2_normalize:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats /= np.clip(norms, 1e-8, None)
    return feats, coords


def fit_pca(
    feature_paths: dict[str, Path],
    control_ids: list[str],
    *,
    n_components: int,
    patches_per_slide: int,
    l2_normalize: bool,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> PCA:
    """Fit PCA on a per-slide subsample of control patches.

    Control patches only: the projection must be defined by what "normal
    liver" looks like, not by the treated slides we are about to score.
    """
    samples = []
    for slide_id in control_ids:
        feats, _ = read_slide_features(feature_paths[slide_id], l2_normalize)
        take = min(patches_per_slide, len(feats))
        idx = rng.choice(len(feats), size=take, replace=False)
        samples.append(feats[idx])
    matrix = np.concatenate(samples, axis=0)
    logger.info(f"PCA fit matrix: {matrix.shape}")

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(matrix)
    logger.info(
        f"PCA: {n_components} components explain "
        f"{pca.explained_variance_ratio_.sum():.3f} of the variance"
    )
    return pca


def knn_mean_distance(
    queries: torch.Tensor, bank: torch.Tensor, k: int, chunk: int
) -> np.ndarray:
    """Mean Euclidean distance from each query patch to its k nearest bank patches.

    Chunked over queries because the full (n_query, n_bank) distance matrix
    does not fit in GPU memory at these sizes.
    """
    out = torch.empty(len(queries), dtype=torch.float32, device=queries.device)
    k = min(k, len(bank))
    for start in range(0, len(queries), chunk):
        stop = min(start + chunk, len(queries))
        d = torch.cdist(queries[start:stop], bank)
        out[start:stop] = d.topk(k, largest=False).values.mean(dim=1)
    return out.cpu().numpy()


def mahalanobis_distance(
    queries: torch.Tensor, bank: torch.Tensor, ridge: float
) -> np.ndarray:
    """Mahalanobis distance of each query patch from the bank's distribution.

    The ridge term keeps the covariance invertible when the bank is small
    (the matched bank is only ~5 slides). It is scaled by the mean variance
    of the bank rather than being absolute: the embeddings are L2-normalized
    and then PCA-projected, so per-component variances are small and an
    absolute ridge would silently dominate the real covariance.
    """
    mu = bank.mean(dim=0, keepdim=True)
    centered = bank - mu
    cov = centered.T @ centered / max(len(bank) - 1, 1)
    scale = torch.diagonal(cov).mean()
    cov += ridge * scale * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    precision = torch.linalg.inv(cov)

    delta = queries - mu
    d2 = ((delta @ precision) * delta).sum(dim=1)
    return torch.sqrt(torch.clamp(d2, min=0)).cpu().numpy()


def build_bank(
    embeddings: dict[str, np.ndarray],
    bank_slide_ids: list[str],
    exclude_slide_id: str,
    bank_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stack control patches into a reference bank, leaving out one slide.

    exclude_slide_id implements leave-one-out: when the slide being scored is
    itself a control, its own patches must not sit in the bank it is scored
    against, or it would trivially look normal.
    """
    parts = [embeddings[sid] for sid in bank_slide_ids if sid != exclude_slide_id]
    if not parts:
        return np.empty((0, next(iter(embeddings.values())).shape[1]), dtype=np.float32)
    bank = np.concatenate(parts, axis=0)
    if bank_size and len(bank) > bank_size:
        idx = rng.choice(len(bank), size=bank_size, replace=False)
        bank = bank[idx]
    return bank


def aggregate_slide_score(scores: np.ndarray, top_k_pct: float) -> float:
    """Slide-level score = mean of the top k% most anomalous patches.

    A slide is "abnormal" because a minority of its patches are, so the mean
    over all patches washes the signal out; the top-k% mean is the standard
    way to keep it while staying more stable than a single max.
    """
    n_top = max(1, int(round(len(scores) * top_k_pct / 100.0)))
    return float(np.sort(scores)[-n_top:].mean())


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.tggates_metadata import attach_pathology_findings

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()

    # RUN_MODE is "single" — one pass over every slide, nothing swept.
    variant_key = "control_contrast"

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
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    source_exp = config["source_experiment"]
    manifest_path = project_root / "outputs" / source_exp / config["manifest_relpath"]
    features_dir = project_root / "outputs" / source_exp / config["features_relpath"]
    pathology_csv = dataset_dir / config["pathology_csv"]

    manifest = pd.read_parquet(manifest_path)
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = attach_pathology_findings(manifest, pathology_csv, organ=config.get("organ", "Liver"))
    logger.info(
        f"manifest: {len(manifest)} slides, "
        f"{int(manifest['has_finding'].sum())} with findings, "
        f"dose_level={manifest['dose_level'].value_counts().to_dict()}"
    )

    feature_paths = {p.stem: p for p in sorted(features_dir.glob("*.h5"))}
    missing = sorted(set(manifest["slide_id"]) - set(feature_paths))
    if missing:
        raise SystemExit(
            f"{len(missing)} slide(s) in the manifest have no feature file under "
            f"{features_dir}: {missing[:10]}"
        )
    # 0001 の出力に manifest より多くのスライドがある場合（追加DL後に0001だけ
    # 再実行した等）は manifest 側に合わせる。ラベルの無いスライドは採点できない。
    extra = sorted(set(feature_paths) - set(manifest["slide_id"]))
    if extra:
        logger.warning(f"{len(extra)} feature file(s) not in the manifest; ignoring them")

    control_ids = manifest.loc[manifest["dose_level"] == "Control", "slide_id"].tolist()
    logger.info(f"control slides: {len(control_ids)}")

    l2_normalize = config.get("l2_normalize", True)
    pca = fit_pca(
        feature_paths,
        control_ids,
        n_components=config["pca_components"],
        patches_per_slide=config["pca_patches_per_slide"],
        l2_normalize=l2_normalize,
        rng=rng,
        logger=logger,
    )

    embeddings: dict[str, np.ndarray] = {}
    coords: dict[str, np.ndarray] = {}
    for i, slide_id in enumerate(manifest["slide_id"], start=1):
        feats, xy = read_slide_features(feature_paths[slide_id], l2_normalize)
        embeddings[slide_id] = pca.transform(feats).astype(np.float32)
        coords[slide_id] = xy
        if i % 25 == 0 or i == len(manifest):
            total = sum(len(v) for v in embeddings.values())
            logger.info(f"projected {i}/{len(manifest)} slides ({total} patches so far)")

    global_mean = np.concatenate(list(embeddings.values()), axis=0).mean(axis=0)

    # ── Patch scoring ─────────────────────────────────────────────────────────
    knn_k = config["knn_k"]
    bank_size = config["bank_size"]
    chunk = config["query_chunk"]
    ridge = config["mahalanobis_ridge"]

    group_cols = ["exp_id", "sacrifice_period"]
    control_by_group = (
        manifest[manifest["dose_level"] == "Control"]
        .groupby(group_cols)["slide_id"]
        .apply(list)
        .to_dict()
    )

    score_frames = []
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        slide_id = row.slide_id
        queries_np = embeddings[slide_id]
        queries = torch.from_numpy(queries_np).to(device)

        record = {
            "slide_id": slide_id,
            "x": coords[slide_id][:, 0],
            "y": coords[slide_id][:, 1],
            # ベースライン1: control情報を一切使わない乱数。
            "random": rng.random(len(queries_np)).astype(np.float32),
            # ベースライン2: 全スライドのpatch平均からの距離。「control比較」が
            # 単なる外れ値検出以上のことをしているかを見るための対照。
            "global_mean_dist": np.linalg.norm(queries_np - global_mean, axis=1),
        }

        for bank_unit in ["matched", "global"]:
            if bank_unit == "matched":
                bank_ids = control_by_group.get((row.exp_id, row.sacrifice_period), [])
            else:
                bank_ids = control_ids
            bank_np = build_bank(embeddings, bank_ids, slide_id, bank_size, rng)
            if len(bank_np) == 0:
                raise SystemExit(f"empty {bank_unit} bank for slide {slide_id}")
            bank = torch.from_numpy(bank_np).to(device)

            record[f"knn_{bank_unit}"] = knn_mean_distance(queries, bank, knn_k, chunk)
            record[f"maha_{bank_unit}"] = mahalanobis_distance(queries, bank, ridge)
            del bank

        score_frames.append(pd.DataFrame(record))
        del queries
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if i % 25 == 0 or i == len(manifest):
            logger.info(f"scored {i}/{len(manifest)} slides")

    patch_scores = pd.concat(score_frames, ignore_index=True)
    # 300万行に文字列の slide_id をそのまま持たせるとメモリもparquetも膨らむ
    patch_scores["slide_id"] = patch_scores["slide_id"].astype("category")
    method_cols = [c for c in patch_scores.columns if c not in ("slide_id", "x", "y")]
    logger.info(f"patch_scores: {len(patch_scores)} rows, methods={method_cols}")

    # ── Slide-level aggregation ───────────────────────────────────────────────
    top_k_pcts = config["top_k_pcts"]
    slide_rows = []
    for slide_id, grp in patch_scores.groupby("slide_id", sort=False, observed=True):
        for method in method_cols:
            values = grp[method].to_numpy()
            for top_k_pct in top_k_pcts:
                slide_rows.append(
                    {
                        "slide_id": slide_id,
                        "method": method,
                        "top_k_pct": top_k_pct,
                        "slide_score": aggregate_slide_score(values, top_k_pct),
                    }
                )
    slide_scores = pd.DataFrame(slide_rows).merge(manifest, on="slide_id", how="left")

    # ── Evaluation ────────────────────────────────────────────────────────────
    evaluation = []
    for (method, top_k_pct), grp in slide_scores.groupby(["method", "top_k_pct"]):
        entry = {"method": method, "top_k_pct": top_k_pct}

        y_finding = grp["has_finding"].astype(int).to_numpy()
        if len(np.unique(y_finding)) == 2:
            entry["auroc_has_finding"] = float(
                roc_auc_score(y_finding, grp["slide_score"].to_numpy())
            )

        y_dose = (grp["dose_level"] != "Control").astype(int).to_numpy()
        if len(np.unique(y_dose)) == 2:
            entry["auroc_treated"] = float(
                roc_auc_score(y_dose, grp["slide_score"].to_numpy())
            )

        graded = grp.dropna(subset=["max_grade_ord"])
        if graded["max_grade_ord"].nunique() > 1:
            rho, p = spearmanr(graded["max_grade_ord"], graded["slide_score"])
            entry["spearman_grade_rho"] = float(rho)
            entry["spearman_grade_p"] = float(p)
            entry["spearman_n"] = int(len(graded))

        evaluation.append(entry)

    evaluation = sorted(
        evaluation, key=lambda e: e.get("auroc_has_finding", 0.0), reverse=True
    )
    for entry in evaluation:
        logger.info(f"eval: {entry}")

    # ── Top patches (0003の可視化・偽陽性確認用) ───────────────────────────────
    n_top = config["n_top_patches"]
    top_rows = []
    for slide_id, grp in patch_scores.groupby("slide_id", sort=False, observed=True):
        for method in method_cols:
            top = grp.nlargest(n_top, method)
            top_rows.append(
                pd.DataFrame(
                    {
                        "slide_id": slide_id,
                        "method": method,
                        "rank": np.arange(1, len(top) + 1),
                        "x": top["x"].to_numpy(),
                        "y": top["y"].to_numpy(),
                        "score": top[method].to_numpy(),
                    }
                )
            )
    top_patches = pd.concat(top_rows, ignore_index=True)

    # ── Save results ──────────────────────────────────────────────────────────
    manifest.to_parquet(run_dir / "slide_manifest_with_findings.parquet", index=False)
    patch_scores.to_parquet(run_dir / "patch_scores.parquet", index=False)
    slide_scores.to_parquet(run_dir / "slide_scores.parquet", index=False)
    top_patches.to_parquet(run_dir / "top_patches.parquet", index=False)

    results = {
        "n_slides": len(manifest),
        "n_patches": int(len(patch_scores)),
        "n_control_slides": len(control_ids),
        "n_slides_with_finding": int(manifest["has_finding"].sum()),
        "pca_components": config["pca_components"],
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "methods": method_cols,
        "evaluation": evaluation,
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
