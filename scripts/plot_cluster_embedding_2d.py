"""t-SNE scatter of exp0006's `method_comparison` run (kmeans_k100 vs
leiden_r1.0 vs dbscan_q0.5), so cluster separation in feature space can be
checked visually instead of only via the LOCO AUROC numbers.

Cluster labels are the real ones already computed by experiment.py
(cluster_assignments.parquet); only the 2D layout is freshly derived here
from a random subsample of patches, since the original run only persisted
spatial (WSI pixel) coordinates, not an embedding-space projection.

Usage: python scripts/plot_cluster_embedding_2d.py
"""

import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "outputs/0006_20260915_clustering_method_comparison/method_comparison"
FEATURES_DIR = (
    PROJECT_ROOT / "outputs/0005_20260913_stain_normalized_embeddings"
    / "20x_256px_0px_overlap/features_uni_v2_macenko"
)
OUT_PATH = RUN_DIR / "embedding_scatter_2d.png"

SEED = 42
N_SLIDES = 40
PATCHES_PER_SLIDE = 600
PCA_COMPONENTS = 64
METHOD_COLUMNS = {
    "kmeans_k100": "cluster_kmeans_k100",
    "leiden_r1.0": "cluster_leiden_r1.0",
    "dbscan_q0.5": "cluster_dbscan_q0.5",
}


def _distinct_palette(n: int) -> list[str]:
    palette = []
    for i in range(n):
        hue = (i * 0.6180339887) % 1.0
        sat = 0.55 + 0.35 * ((i % 3) / 2)
        val = 0.55 + 0.35 * (((i + 1) % 3) / 2)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((r, g, b))
    return palette


def main() -> None:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from lib.patch_features import read_slide_features

    assignments = pd.read_parquet(RUN_DIR / "cluster_assignments.parquet")
    assignments["slide_id"] = assignments["slide_id"].astype(str)

    rng = np.random.default_rng(SEED)
    all_slides = assignments["slide_id"].unique()
    sampled_slides = rng.choice(all_slides, size=min(N_SLIDES, len(all_slides)), replace=False)

    feats_parts, labels_parts = [], []
    for slide_id in sampled_slides:
        rows = assignments[assignments["slide_id"] == slide_id]
        take = min(PATCHES_PER_SLIDE, len(rows))
        rows = rows.iloc[rng.choice(len(rows), size=take, replace=False)]

        feats, coords, _ = read_slide_features(FEATURES_DIR / f"{slide_id}.h5", l2_normalize=True)
        coord_to_idx = {(int(x), int(y)): i for i, (x, y) in enumerate(coords)}
        idx = np.array([coord_to_idx[(int(x), int(y))] for x, y in zip(rows["x"], rows["y"])])

        feats_parts.append(feats[idx])
        labels_parts.append(rows[list(METHOD_COLUMNS.values())].reset_index(drop=True))

    pooled_feats = np.concatenate(feats_parts, axis=0)
    pooled_labels = pd.concat(labels_parts, ignore_index=True)
    print(f"pooled {len(pooled_feats)} patches from {len(sampled_slides)} slides")

    pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED, svd_solver="randomized")
    projected = pca.fit_transform(pooled_feats)
    print(f"PCA: {pooled_feats.shape} -> {PCA_COMPONENTS}次元, 寄与率 {pca.explained_variance_ratio_.sum():.3f}")

    layout = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=30).fit_transform(projected)
    print("t-SNE done")

    fig, axes = plt.subplots(1, len(METHOD_COLUMNS), figsize=(6 * len(METHOD_COLUMNS), 5.5))
    for ax, (title, col) in zip(axes, METHOD_COLUMNS.items()):
        cluster_ids = pooled_labels[col].to_numpy()
        uniq = sorted(set(cluster_ids.tolist()))
        palette = _distinct_palette(len(uniq))
        is_noise_col = "dbscan" in col
        for i, cid in enumerate(uniq):
            mask = cluster_ids == cid
            if is_noise_col and cid == 0:
                ax.scatter(layout[mask, 0], layout[mask, 1], s=4, color="lightgray", alpha=0.4, label="noise")
                continue
            ax.scatter(layout[mask, 0], layout[mask, 1], s=4, color=palette[i], alpha=0.7)
        ax.set_title(f"{title} ({len(uniq)} clusters)")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("exp0006 method_comparison — shared t-SNE layout, colored by each method's cluster id")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
