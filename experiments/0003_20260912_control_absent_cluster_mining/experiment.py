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

from sklearn.cluster import MiniBatchKMeans
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


def load_config(exp_dir: Path, filename: str = "config.yml") -> dict:
    """Load a config file from the experiment directory.

    run_slurm.sh は RUN_COMMAND で常に --config を渡しているので、それを実際に
    使えるようにしておく。同じコードを別の入力（0005の染色正規化版の特徴量など）で
    回すときに、元の config.yml を書き換えずに別ファイルを指せる。
    """
    config_path = Path(filename)
    if not config_path.is_absolute():
        config_path = exp_dir / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    """Define CLI args for all variable dimensions (used in GRID_ARGS / RUN_COMMAND)."""
    parser = argparse.ArgumentParser()
    # RUN_MODE is "single" — k is swept inside this script (the clustering is
    # cheap relative to reading 19GB of features, so one job does every k).
    parser.add_argument("--config", type=str, default="config.yml")
    return parser.parse_args()


def save_exemplars(
    run_dir: Path,
    manifest: pd.DataFrame,
    coords: dict[str, np.ndarray],
    patch_sizes: dict[str, int],
    slide_index: np.ndarray,
    assignments: np.ndarray,
    centroids: np.ndarray,
    embeddings_all: np.ndarray,
    clusters: np.ndarray,
    *,
    n_per_cluster: int,
    grid_cols: int,
    logger: logging.Logger,
    subdir: str = "",
) -> None:
    """Crop the patches closest to each cluster centroid into one contact sheet.

    Reading each WSI is the expensive part, so patches are grouped by slide and
    every slide is opened once.

    subdir: written under run_dir/exemplars/<subdir>/ — used to separate
    candidate ("included") clusters from the rest ("excluded") when the
    caller wants exemplars for every cluster, not just the control-absent
    candidates.
    """
    import openslide
    from PIL import Image

    slide_ids = manifest["slide_id"].tolist()
    svs_paths = dict(zip(manifest["slide_id"], manifest["svs_path"]))
    out_dir = run_dir / "exemplars" / subdir if subdir else run_dir / "exemplars"
    out_dir.mkdir(parents=True, exist_ok=True)
    # 連結済み行列での各スライドの開始位置。global index からスライド内 index を引くのに使う。
    offsets = np.cumsum([0] + [len(coords[s]) for s in slide_ids])

    for cluster_id in clusters:
        member = np.flatnonzero(assignments == cluster_id)
        if len(member) == 0:
            continue
        d = np.linalg.norm(embeddings_all[member] - centroids[cluster_id], axis=1)
        chosen = member[np.argsort(d)[:n_per_cluster]]

        tiles = []
        by_slide: dict[int, list[int]] = {}
        for global_idx in chosen:
            by_slide.setdefault(int(slide_index[global_idx]), []).append(int(global_idx))

        for s_idx, idxs in by_slide.items():
            slide_id = slide_ids[s_idx]
            try:
                slide = openslide.OpenSlide(svs_paths[slide_id])
            except Exception as e:  # 1スライド開けなくても全体は止めない
                logger.warning(f"could not open {slide_id}: {e}")
                continue
            size = patch_sizes[slide_id]
            for global_idx in idxs:
                local = global_idx - offsets[s_idx]
                x, y = coords[slide_id][local]
                tile = slide.read_region((int(x), int(y)), 0, (size, size)).convert("RGB")
                tiles.append(tile.resize((256, 256)))
            slide.close()

        if not tiles:
            continue
        rows = (len(tiles) + grid_cols - 1) // grid_cols
        sheet = Image.new("RGB", (grid_cols * 256, rows * 256), "white")
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % grid_cols) * 256, (i // grid_cols) * 256))
        sheet.save(out_dir / f"cluster_{cluster_id:05d}.png")

    logger.info(f"exemplars written for {len(clusters)} clusters -> {out_dir}")


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.cluster_analysis import (
        cluster_statistics,
        finding_associations,
        leave_one_compound_out_auroc,
        occupancy_matrix,
        select_candidates,
    )
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.patch_features import read_slide_features, sample_patch_matrix

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()

    config = load_config(Path(__file__).parent, args.config)
    seed: int = config.get("seed", 42)

    # 同じコードを別の特徴量（0005の染色正規化版など）で回せるよう、variant_keyは
    # configから取る。完了ガードは variant_key 単位なので、ここを変えないと
    # 2回目以降が即時終了してしまう。
    variant_key = config.get("variant_key", "cluster_mining")

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    manifest_path = project_root / config["manifest_parquet"]
    features_dir = project_root / config["features_dir"]

    manifest = pd.read_parquet(manifest_path)
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)

    feature_paths = {p.stem: p for p in sorted(features_dir.glob("*.h5"))}
    missing = sorted(set(manifest["slide_id"]) - set(feature_paths))
    if missing:
        raise SystemExit(
            f"{len(missing)} slide(s) in {manifest_path} have no feature file: {missing[:10]}"
        )
    logger.info(
        f"manifest: {len(manifest)} slides, "
        f"{int(manifest['has_finding'].sum())} with findings, "
        f"dose_level={manifest['dose_level'].value_counts().to_dict()}"
    )

    l2_normalize = config.get("l2_normalize", True)
    control_ids = manifest.loc[manifest["dose_level"] == "Control", "slide_id"].tolist()
    treated_ids = manifest.loc[manifest["dose_level"] != "Control", "slide_id"].tolist()

    # PCAはcontrolと投与群から同数ずつ取って学習する。0002のようにcontrolだけで
    # 学習すると、所見特有の（＝controlでの分散が小さい）方向が落ちうる。
    per_group = config["pca_patches_per_group"]
    matrix = np.concatenate(
        [
            sample_patch_matrix(
                feature_paths, control_ids, per_group // len(control_ids), rng, l2_normalize
            ),
            sample_patch_matrix(
                feature_paths, treated_ids, per_group // len(treated_ids), rng, l2_normalize
            ),
        ],
        axis=0,
    )
    logger.info(f"PCA fit matrix: {matrix.shape} (control+treated balanced)")
    pca = PCA(n_components=config["pca_components"], svd_solver="randomized", random_state=seed)
    pca.fit(matrix)
    logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    del matrix

    coords: dict[str, np.ndarray] = {}
    patch_sizes: dict[str, int] = {}
    projected: list[np.ndarray] = []
    slide_index_parts: list[np.ndarray] = []
    for i, slide_id in enumerate(manifest["slide_id"]):
        feats, xy, attrs = read_slide_features(feature_paths[slide_id], l2_normalize)
        projected.append(pca.transform(feats).astype(np.float32))
        coords[slide_id] = xy
        patch_sizes[slide_id] = int(attrs.get("patch_size_level0", attrs.get("patch_size", 256)))
        slide_index_parts.append(np.full(len(xy), i, dtype=np.int32))
        if (i + 1) % 25 == 0 or i + 1 == len(manifest):
            logger.info(f"projected {i + 1}/{len(manifest)} slides")

    embeddings = np.concatenate(projected, axis=0)
    slide_index = np.concatenate(slide_index_parts)
    del projected, slide_index_parts
    logger.info(f"patch matrix: {embeddings.shape}")

    fit_sample = rng.choice(
        len(embeddings), size=min(config["kmeans_fit_sample"], len(embeddings)), replace=False
    )

    presence_eps = config["presence_eps"]
    q_threshold = config["q_threshold"]
    # plan.md 2026-09-16の決定: k=100はmin_compounds>=2を外す（同一表現型の別変種を
    # 落としていたため）。k>=500は外すと候補が爆発しLOCOも落ちるので維持。
    # plan.md 2026-09-17の決定: min_treated_presence も同じ理由・同じ形でkごとに
    # 変える。複数化合物が同じ投与群プールを共有する319枚コーパスでは、
    # 「投与群全体の10%」という絶対枚数の壁が化合物単位の本物の信号を
    # 機械的に落としていた（cluster 42/53/54等）。k=100/500は外す方がLOCOも
    # 上がる（0.560→0.836 / 0.507→0.563）。k=2000は逆に必要（外すと0.443→0.412）。
    # どちらも下位互換のためintのままでよく、その場合は全kに同じ値を使う。
    min_compounds_cfg = config["min_compounds"]
    min_treated_presence_cfg = config["min_treated_presence"]

    def min_compounds_for_k(k: int) -> int:
        if isinstance(min_compounds_cfg, dict):
            return min_compounds_cfg[str(k)]
        return min_compounds_cfg

    def min_treated_presence_for_k(k: int) -> float:
        if isinstance(min_treated_presence_cfg, dict):
            return min_treated_presence_cfg[str(k)]
        return min_treated_presence_cfg

    all_stats = []
    all_assoc = []
    all_loco = []
    assignments_out = {"slide_id": manifest["slide_id"].to_numpy()[slide_index]}
    assignments_out["x"] = np.concatenate([coords[s][:, 0] for s in manifest["slide_id"]])
    assignments_out["y"] = np.concatenate([coords[s][:, 1] for s in manifest["slide_id"]])
    summary = []

    for k in config["k_values"]:
        logger.info(f"--- k={k} ---")
        min_compounds = min_compounds_for_k(k)
        min_treated_presence = min_treated_presence_for_k(k)
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=seed,
            batch_size=config["kmeans_batch_size"],
            n_init=config["kmeans_n_init"],
            max_iter=config["kmeans_max_iter"],
        )
        kmeans.fit(embeddings[fit_sample])
        assignments = kmeans.predict(embeddings).astype(np.int32)
        assignments_out[f"cluster_k{k}"] = assignments
        logger.info(f"k={k}: fitted, inertia={kmeans.inertia_:.1f}")

        occ = occupancy_matrix(slide_index, assignments, len(manifest), k)
        stats = cluster_statistics(
            occ,
            manifest,
            presence_eps=presence_eps,
            min_treated_presence=min_treated_presence,
            logger=logger,
        )
        stats.insert(0, "k", k)
        candidates = select_candidates(
            stats, q_threshold=q_threshold, min_compounds=min_compounds
        )
        logger.info(f"k={k}: {len(candidates)} candidate clusters after artifact filtering")

        assoc = finding_associations(
            occ,
            manifest,
            candidates,
            presence_eps=presence_eps,
            max_finding_types=config["max_finding_types"],
        )
        if not assoc.empty:
            assoc.insert(0, "k", k)
            top = assoc.nsmallest(5, "q_value")
            for _, r in top.iterrows():
                logger.info(
                    f"k={k} cluster {int(r['cluster_id'])} ~ {r['finding_type']}: "
                    f"OR={r['odds_ratio']:.1f} q={r['q_value']:.3g} "
                    f"(both={int(r['n_slides_both'])})"
                )

        pooled_auroc, per_compound = leave_one_compound_out_auroc(
            occ,
            manifest,
            presence_eps=presence_eps,
            min_treated_presence=min_treated_presence,
            q_threshold=q_threshold,
            min_compounds=min_compounds,
            logger=logger,
        )
        per_compound.insert(0, "k", k)
        logger.info(f"k={k}: LOCO pooled AUROC = {pooled_auroc}")
        if "auroc" in per_compound.columns:
            by_compound = per_compound.set_index("compound_name")["auroc"].round(3).to_dict()
            logger.info(f"k={k}: per-compound AUROC = {by_compound}")

        in_sample = (
            occ[:, candidates].sum(axis=1) if len(candidates) else np.zeros(len(manifest))
        )
        y = manifest["has_finding"].astype(int).to_numpy()
        auroc_in_sample = (
            float(roc_auc_score(y, in_sample)) if len(np.unique(y)) == 2 else None
        )

        all_stats.append(stats)
        if not assoc.empty:
            all_assoc.append(assoc)
        all_loco.append(per_compound)
        summary.append(
            {
                "k": k,
                "n_control_absent": int(stats["control_absent"].sum()),
                "n_candidates": int(len(candidates)),
                # 選択にラベルを使っているので循環している。LOCOと必ず並べて読むこと。
                "auroc_in_sample_circular": auroc_in_sample,
                "auroc_loco": pooled_auroc,
            }
        )

        if k == config["exemplar_k"]:
            # 「排除された」＝候補選定(select_candidates)に落ちたクラスタ。正常組織や
            # アーチファクトの見え方を目で確認できるよう、候補外も全クラスタ分書き出す。
            included = (
                stats[stats["cluster_id"].isin(candidates)]
                .nsmallest(config["n_exemplar_clusters"], "q_value")["cluster_id"]
                .to_numpy()
                if len(candidates)
                else np.array([], dtype=int)
            )
            excluded = np.setdiff1d(np.arange(k), candidates)
            for subdir, ranked in (("included", included), ("excluded", excluded)):
                if len(ranked) == 0:
                    continue
                save_exemplars(
                    run_dir,
                    manifest,
                    coords,
                    patch_sizes,
                    slide_index,
                    assignments,
                    kmeans.cluster_centers_,
                    embeddings,
                    ranked,
                    n_per_cluster=config["n_exemplars_per_cluster"],
                    grid_cols=config["exemplar_grid_cols"],
                    logger=logger,
                    subdir=subdir,
                )

    # ── Save results ──────────────────────────────────────────────────────────
    assignments_df = pd.DataFrame(assignments_out)
    assignments_df["slide_id"] = assignments_df["slide_id"].astype("category")
    assignments_df.to_parquet(run_dir / "cluster_assignments.parquet", index=False)
    pd.concat(all_stats, ignore_index=True).to_parquet(run_dir / "cluster_stats.parquet", index=False)
    if all_assoc:
        pd.concat(all_assoc, ignore_index=True).to_parquet(
            run_dir / "cluster_finding_association.parquet", index=False
        )
    pd.concat(all_loco, ignore_index=True).to_parquet(
        run_dir / "loco_per_compound.parquet", index=False
    )
    manifest.to_parquet(run_dir / "slide_manifest.parquet", index=False)

    results = {
        "n_slides": len(manifest),
        "n_patches": int(len(embeddings)),
        "pca_components": config["pca_components"],
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "summary": summary,
    }
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
