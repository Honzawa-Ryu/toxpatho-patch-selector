import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

import torch
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
    """Load a config file from the experiment directory."""
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
    # RUN_MODE is "single" — 手法とパラメータはこのスクリプト内で回す。
    # 特徴量の読み込み（19GB）が一番重いので、1ジョブで全手法を見るほうが速い。
    parser.add_argument("--config", type=str, default="config.yml")
    return parser.parse_args()


def build_runs(config: dict) -> list[dict]:
    """Expand the config into one entry per (method, parameter) to run."""
    runs = []
    for k in config.get("kmeans_k", []):
        runs.append({"method": "kmeans", "param_name": "k", "param": float(k), "label": f"kmeans_k{k}"})
    for r in config.get("leiden_resolutions", []):
        runs.append(
            {"method": "leiden", "param_name": "resolution", "param": float(r), "label": f"leiden_r{r}"}
        )
    for r in config.get("leiden_spatial_resolutions", []):
        runs.append(
            {
                "method": "leiden_spatial",
                "param_name": "resolution",
                "param": float(r),
                "label": f"leiden_spatial_r{r}",
            }
        )
    for q in config.get("dbscan_eps_quantiles", []):
        runs.append(
            {"method": "dbscan", "param_name": "eps_quantile", "param": float(q), "label": f"dbscan_q{q}"}
        )
    return runs


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
    from lib.clustering import (
        dbscan_clusters,
        knn_graph,
        leiden_clusters,
        propagate_labels,
        spatial_knn_within_slide,
    )
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.patch_features import read_slide_features, sample_patch_matrix

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()

    config = load_config(Path(__file__).parent, args.config)
    seed: int = config.get("seed", 42)
    variant_key = config.get("variant_key", "method_comparison")

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")

    # ── 入力（0003と完全に同じ手順で埋め込みを用意する）─────────────────────────
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    manifest = pd.read_parquet(project_root / config["manifest_parquet"])
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)
    features_dir = project_root / config["features_dir"]

    cache_dir = config.get("embedding_cache_dir")
    if cache_dir:
        # scripts/precompute_tier1_kmeans_cache.py が事前に計算したPCA投影済み
        # embeddingを読む。3,441スライドの読み込み+PCA投影(exp0009本体runで
        # 約1.5h・GPU未使用)をkごとに繰り返さないための経路。並列kスイープの
        # 各パートが同じキャッシュを共有する。
        cache_dir = project_root / cache_dir
        cached_slide_ids = np.load(cache_dir / "slide_ids.npy", allow_pickle=False).astype(str)
        if list(cached_slide_ids) != manifest["slide_id"].tolist():
            raise SystemExit("embedding_cache_dir slide order does not match manifest_parquet")
        embeddings = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
        xy_all = np.load(cache_dir / "xy.npy", mmap_mode="r")
        slide_index = np.load(cache_dir / "slide_index.npy")
        pca_explained_variance = float(np.load(cache_dir / "pca_explained_variance.npy"))
        # サブサンプル数はconfigの subsample がそのまま効く（キャッシュされた
        # sub_idx.npyの固定サイズには縛られない）。埋め込み自体はキャッシュ済み
        # なので、サイズを変えてもh5の再読み込みは発生しない。
        n_sub = min(config["subsample"], len(embeddings))
        sub_idx = rng.choice(len(embeddings), size=n_sub, replace=False)
        sub = np.ascontiguousarray(embeddings[sub_idx])
        logger.info(f"manifest: {len(manifest)} slides / cache: {cache_dir.name}")
        logger.info(f"patch matrix (cached): {embeddings.shape}")
        logger.info(f"共通の部分集合 (cached): {sub.shape}")
    else:
        feature_paths = {p.stem: p for p in sorted(features_dir.glob("*.h5"))}
        missing = sorted(set(manifest["slide_id"]) - set(feature_paths))
        if missing:
            raise SystemExit(f"{len(missing)} slide(s) have no feature file: {missing[:10]}")
        logger.info(f"manifest: {len(manifest)} slides / features: {features_dir.name}")

        l2_normalize = config.get("l2_normalize", True)
        control_ids = manifest.loc[manifest["dose_level"] == "Control", "slide_id"].tolist()
        treated_ids = manifest.loc[manifest["dose_level"] != "Control", "slide_id"].tolist()
        per_group = config["pca_patches_per_group"]
        matrix = np.concatenate(
            [
                sample_patch_matrix(feature_paths, control_ids, per_group // len(control_ids), rng, l2_normalize),
                sample_patch_matrix(feature_paths, treated_ids, per_group // len(treated_ids), rng, l2_normalize),
            ]
        )
        pca = PCA(n_components=config["pca_components"], svd_solver="randomized", random_state=seed)
        pca.fit(matrix)
        pca_explained_variance = float(pca.explained_variance_ratio_.sum())
        logger.info(f"PCA: {matrix.shape} -> {config['pca_components']}次元, 寄与率 {pca_explained_variance:.3f}")
        del matrix

        projected, slide_index_parts, coords = [], [], {}
        for i, slide_id in enumerate(manifest["slide_id"]):
            feats, xy, _ = read_slide_features(feature_paths[slide_id], l2_normalize)
            projected.append(pca.transform(feats).astype(np.float32))
            coords[slide_id] = xy
            slide_index_parts.append(np.full(len(xy), i, dtype=np.int32))
            if (i + 1) % 50 == 0 or i + 1 == len(manifest):
                logger.info(f"projected {i + 1}/{len(manifest)} slides")
        embeddings = np.concatenate(projected)
        slide_index = np.concatenate(slide_index_parts)
        xy_all = np.concatenate([coords[s] for s in manifest["slide_id"]])
        del projected, slide_index_parts
        logger.info(f"patch matrix: {embeddings.shape}")

        # ── 全手法が同じ部分集合を使う（手法以外の条件を揃えるため）──────────────
        n_sub = min(config["subsample"], len(embeddings))
        sub_idx = rng.choice(len(embeddings), size=n_sub, replace=False)
        sub = np.ascontiguousarray(embeddings[sub_idx])
        logger.info(f"共通の部分集合: {sub.shape}")

    knn_k = config["knn_k"]
    neighbors, kdist = None, None
    if config.get("leiden_resolutions") or config.get("leiden_spatial_resolutions") or config.get("dbscan_eps_quantiles"):
        # kmeansだけのrun（今回のサブサンプル数スイープ等）では誰も使わない。
        # n_subが大きいとchunk×n_subのcdist一発がGPUメモリを圧迫するため、
        # 使わないなら計算自体をスキップする。
        neighbors, distances = knn_graph(sub, knn_k, device, chunk=config["knn_chunk"])
        kdist = distances[:, -1]  # k番目の近傍までの距離。DBSCANのeps候補はここから取る。
        logger.info(
            f"kNNグラフ (k={knn_k}): k距離の分位点 "
            f"{ {q: round(float(np.quantile(kdist, q)), 4) for q in [0.1, 0.3, 0.5, 0.7, 0.9]} }"
        )
    else:
        logger.info("kNNグラフ: kmeansのみのrunのためスキップ")

    # leiden_spatial 用: 特徴量グラフと同じ部分集合上で、同一スライド内の
    # 座標近傍グラフも作っておく（隣接patchが同じクラスタに寄りやすくなる）。
    spatial_neighbors = None
    if config.get("leiden_spatial_resolutions"):
        spatial_k = config.get("spatial_knn_k", knn_k)
        spatial_neighbors = spatial_knn_within_slide(
            xy_all[sub_idx], slide_index[sub_idx], spatial_k, device
        )
        logger.info(f"空間隣接グラフ (同一スライド内, k={spatial_k}) 構築完了")

    # ── 手法ごとにクラスタリング → 同一の解析 ──────────────────────────────────
    presence_eps = config["presence_eps"]
    min_treated_presence = config["min_treated_presence"]
    q_threshold = config["q_threshold"]
    min_compounds = config["min_compounds"]

    assignments_out = {
        "slide_id": manifest["slide_id"].to_numpy()[slide_index],
        "x": np.asarray(xy_all[:, 0]),
        "y": np.asarray(xy_all[:, 1]),
    }
    summary, all_stats, all_assoc, all_loco = [], [], [], []

    for run in build_runs(config):
        label, method = run["label"], run["method"]
        logger.info(f"--- {label} ---")
        started = time.time()

        if method == "kmeans":
            model = MiniBatchKMeans(
                n_clusters=int(run["param"]),
                random_state=seed,
                batch_size=config["kmeans_batch_size"],
                n_init=config["kmeans_n_init"],
                max_iter=config["kmeans_max_iter"],
            ).fit(sub)
            sub_labels = model.labels_
        elif method == "leiden":
            sub_labels = leiden_clusters(neighbors, run["param"], seed)
        elif method == "leiden_spatial":
            sub_labels = leiden_clusters(
                neighbors,
                run["param"],
                seed,
                spatial_neighbors=spatial_neighbors,
                spatial_weight=config.get("spatial_edge_weight", 1.0),
            )
        elif method == "dbscan":
            eps = float(np.quantile(kdist, run["param"]))
            run["eps"] = eps
            sub_labels = dbscan_clusters(sub, eps, config["dbscan_min_samples"])
            logger.info(f"{label}: eps={eps:.4f} (k距離の{run['param']}分位点)")
        else:
            raise SystemExit(f"unknown method: {method}")

        n_noise = int((sub_labels < 0).sum())
        n_clusters = int(len(set(sub_labels.tolist()) - {-1}))
        cluster_time = time.time() - started
        logger.info(
            f"{label}: {n_clusters} クラスタ / ノイズ {n_noise}点 ({n_noise / len(sub_labels):.1%}) "
            f"/ {cluster_time:.0f}s"
        )
        if n_clusters < 2:
            logger.warning(f"{label}: クラスタが{n_clusters}個しかないため解析をスキップ")
            summary.append(
                {"label": label, "method": method, "param": run["param"], "n_clusters": n_clusters,
                 "noise_frac": n_noise / len(sub_labels), "skipped": True}
            )
            continue

        assignments = propagate_labels(
            embeddings, sub, sub_labels, config["propagate_k"], device, chunk=config["propagate_chunk"]
        )
        # ノイズ(-1)は0番のクラスタに寄せて連番にする。解析側は0以上の整数を前提にしている。
        noise_cluster = None
        if assignments.min() < 0:
            noise_cluster = 0
            assignments = assignments + 1
        assignments_out[f"cluster_{label}"] = assignments.astype(np.int32)

        n_total = int(assignments.max()) + 1
        occ = occupancy_matrix(slide_index, assignments, len(manifest), n_total)
        stats = cluster_statistics(
            occ, manifest, presence_eps=presence_eps,
            min_treated_presence=min_treated_presence, logger=logger,
        )
        stats.insert(0, "label", label)
        stats["is_noise_cluster"] = stats["cluster_id"] == noise_cluster

        candidates = select_candidates(stats, q_threshold=q_threshold, min_compounds=min_compounds)
        if noise_cluster is not None:
            candidates = candidates[candidates != noise_cluster]
        logger.info(f"{label}: 候補 {len(candidates)} クラスタ（全{n_total}）")

        assoc = finding_associations(
            occ, manifest, candidates, presence_eps=presence_eps,
            max_finding_types=config["max_finding_types"],
        )
        if not assoc.empty:
            assoc.insert(0, "label", label)
            for _, r in assoc.nsmallest(3, "q_value").iterrows():
                logger.info(
                    f"  cluster {int(r['cluster_id'])} ~ {r['finding_type']}: "
                    f"OR={r['odds_ratio']:.0f} q={r['q_value']:.1e} ({int(r['n_slides_both'])}枚)"
                )

        pooled, per_compound = leave_one_compound_out_auroc(
            occ, manifest, presence_eps=presence_eps,
            min_treated_presence=min_treated_presence, q_threshold=q_threshold,
            min_compounds=min_compounds, logger=logger,
        )
        per_compound.insert(0, "label", label)
        logger.info(f"{label}: LOCO AUROC = {pooled}")

        y = manifest["has_finding"].astype(int).to_numpy()
        in_sample = occ[:, candidates].sum(axis=1) if len(candidates) else np.zeros(len(manifest))
        auroc_in_sample = float(roc_auc_score(y, in_sample)) if len(np.unique(y)) == 2 else None

        all_stats.append(stats)
        if not assoc.empty:
            all_assoc.append(assoc)
        all_loco.append(per_compound)
        summary.append(
            {
                "label": label,
                "method": method,
                "param_name": run["param_name"],
                "param": run["param"],
                "eps": run.get("eps"),
                "n_clusters": n_total,
                "noise_frac_subsample": n_noise / len(sub_labels),
                "n_candidates": int(len(candidates)),
                # 候補の選択にラベルを使っているので循環している。LOCOと必ず並べて読むこと。
                "auroc_in_sample_circular": auroc_in_sample,
                "auroc_loco": pooled,
                "cluster_seconds": round(cluster_time, 1),
            }
        )
        logger.info(f"{label}: 完了 ({time.time() - started:.0f}s)")
        # Leidenは1条件で数十分かかる。途中でジョブが時間切れになっても、
        # そこまでの比較結果は残るように毎回書き出しておく。
        (run_dir / "results_partial.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        )

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(assignments_out)
    df["slide_id"] = df["slide_id"].astype("category")
    df.to_parquet(run_dir / "cluster_assignments.parquet", index=False)
    if all_stats:
        pd.concat(all_stats, ignore_index=True).to_parquet(run_dir / "cluster_stats.parquet", index=False)
    if all_assoc:
        pd.concat(all_assoc, ignore_index=True).to_parquet(
            run_dir / "cluster_finding_association.parquet", index=False
        )
    if all_loco:
        pd.concat(all_loco, ignore_index=True).to_parquet(run_dir / "loco_per_compound.parquet", index=False)
    manifest.to_parquet(run_dir / "slide_manifest.parquet", index=False)

    table = pd.DataFrame(summary)
    logger.info("\n=== 手法の比較 ===\n" + table.to_string(index=False))

    results = {
        "n_slides": len(manifest),
        "n_patches": int(len(embeddings)),
        "features_dir": str(features_dir),
        "subsample": int(n_sub),
        "knn_k": knn_k,
        "pca_components": config["pca_components"],
        "pca_explained_variance": pca_explained_variance,
        "summary": summary,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
