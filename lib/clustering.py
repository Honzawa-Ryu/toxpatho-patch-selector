"""Clustering algorithms for patch embeddings, and label propagation.

0003はMiniBatchKMeansだけを使っている。ここはそれ以外（kNNグラフ + Leiden、DBSCAN）を
同じ土俵に乗せるための実装。

共通の設計:

- **部分集合でクラスタリングし、全patchへ伝播する**。326万patchを直接扱うのは、
  Leidenならグラフが大きすぎ、DBSCANなら64次元での近傍探索が現実的でない。
  MiniBatchKMeansが50万patchで学習して全件に`predict`するのと同じ構図に揃えてある
  （伝播だけ、k-meansの「最近傍セントロイド」からkNN多数決に変えている。
  Leiden/DBSCANのクラスタは凸とは限らず、重心で代表させると手法の利点を潰すため）。
- **近傍探索はGPUで厳密に行う**。faiss/pynndescentのような近似近傍ライブラリを
  足さずに済み、近似誤差が手法比較に混入することもない。
"""

from __future__ import annotations

import numpy as np
import torch


def knn_graph(
    features: np.ndarray, k: int, device: torch.device, chunk: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Exact k nearest neighbours within one set of points.

    Returns:
        (indices (n, k) int64, distances (n, k) float32), self excluded.
    """
    points = torch.from_numpy(features).to(device)
    n = len(points)
    idx_out = np.empty((n, k), dtype=np.int64)
    dist_out = np.empty((n, k), dtype=np.float32)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = torch.cdist(points[start:stop], points)
        # 自分自身が必ず最近傍になるので、k+1件取って先頭を捨てる
        vals, idx = d.topk(k + 1, largest=False)
        idx_out[start:stop] = idx[:, 1:].cpu().numpy()
        dist_out[start:stop] = vals[:, 1:].cpu().numpy().astype(np.float32)

    del points
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return idx_out, dist_out


def spatial_knn_within_slide(
    xy: np.ndarray, slide_index: np.ndarray, k: int, device: torch.device
) -> np.ndarray:
    """k nearest neighbours in (x, y) tissue-coordinate space, restricted to
    points on the same slide (per-slide block, so patches from different
    slides can never look "adjacent" just because their pixel coordinates
    happen to overlap).

    Args:
        xy: (n, 2) patch coordinates (same units as knn_graph's caller — here
            level-0 pixel coords, unnormalized; only relative distance within
            a slide matters).
        slide_index: (n,) integer slide id per point, same order as xy.
        k: neighbours per point. Slides with <=k points get min(k, n_slide-1).

    Returns:
        (n, k) neighbour indices into xy/slide_index (global, not per-slide),
        self excluded. Points on a singleton slide get themselves repeated
        (harmless: leiden_clusters dedupes self-pairs implicitly since
        min==max collapses to a single-node "edge" that igraph ignores when
        building from the deduped pair list — but see caller, which filters
        those out explicitly to avoid depending on that).
    """
    out = np.zeros((len(xy), k), dtype=np.int64)
    for slide in np.unique(slide_index):
        member = np.flatnonzero(slide_index == slide)
        n_slide = len(member)
        if n_slide <= 1:
            out[member] = member[:, None]
            continue
        k_slide = min(k, n_slide - 1)
        idx, _ = knn_graph(np.ascontiguousarray(xy[member]).astype(np.float32), k_slide, device)
        local = member[idx]  # (n_slide, k_slide) -> global indices
        if k_slide < k:
            local = np.pad(local, ((0, 0), (0, k - k_slide)), mode="edge")
        out[member] = local
    return out


def _undirected_pairs(neighbors: np.ndarray) -> np.ndarray:
    n, k = neighbors.shape
    sources = np.repeat(np.arange(n, dtype=np.int64), k)
    targets = neighbors.reshape(-1)
    keep = sources != targets
    sources, targets = sources[keep], targets[keep]
    lo = np.minimum(sources, targets)
    hi = np.maximum(sources, targets)
    return np.unique(np.stack([lo, hi], axis=1), axis=0)


def leiden_clusters(
    neighbors: np.ndarray,
    resolution: float,
    seed: int,
    n_iterations: int = -1,
    *,
    spatial_neighbors: np.ndarray | None = None,
    spatial_weight: float = 1.0,
) -> np.ndarray:
    """Leiden community detection on an undirected kNN graph.

    Args:
        neighbors: (n, k) neighbour indices from knn_graph() (feature space).
        resolution: RBConfiguration resolution. Higher gives more, smaller communities.
        n_iterations: -1 runs until the partition stops improving.
        spatial_neighbors: optional (n, k_spatial) indices from
            spatial_knn_within_slide(). When given, tissue-adjacency edges are
            unioned into the graph alongside the feature-similarity ones, each
            edge weighted by how many of the two neighbour sets support it
            (1.0 for feature-only or spatial-only, 1.0 + spatial_weight for an
            edge both sets agree on) so patches that are both morphologically
            similar AND physically touching are pulled together hardest.
        spatial_weight: weight given to a spatial-only edge relative to a
            feature-only edge (1.0 = equal footing).

    The graph is made undirected by keeping each pair once (i < j): a kNN graph is
    asymmetric (j may be among i's neighbours without the reverse), and leaving it
    directed would weight those pairs twice.
    """
    import igraph as ig
    import leidenalg

    n = neighbors.shape[0]
    feat_pairs = _undirected_pairs(neighbors)

    if spatial_neighbors is None:
        graph = ig.Graph(n=n, edges=[tuple(e) for e in feat_pairs], directed=False)
        weights = None
    else:
        spatial_pairs = _undirected_pairs(spatial_neighbors)
        feat_keys = feat_pairs[:, 0].astype(np.int64) * n + feat_pairs[:, 1]
        spatial_keys = spatial_pairs[:, 0].astype(np.int64) * n + spatial_pairs[:, 1]
        all_pairs = np.unique(np.concatenate([feat_pairs, spatial_pairs]), axis=0)
        all_keys = all_pairs[:, 0].astype(np.int64) * n + all_pairs[:, 1]
        in_feat = np.isin(all_keys, feat_keys)
        in_spatial = np.isin(all_keys, spatial_keys)
        weights = np.where(in_feat, 1.0, 0.0) + np.where(in_spatial, spatial_weight, 0.0)
        graph = ig.Graph(n=n, edges=[tuple(e) for e in all_pairs], directed=False)

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=n_iterations,
        weights=weights,
    )
    return np.asarray(partition.membership, dtype=np.int64)


def dbscan_clusters(features: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN. Noise points come back as -1."""
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(features)


def propagate_labels(
    features: np.ndarray,
    reference: np.ndarray,
    reference_labels: np.ndarray,
    k: int,
    device: torch.device,
    chunk: int = 4096,
) -> np.ndarray:
    """Give every point the majority label of its k nearest reference points.

    Majority vote rather than nearest-centroid on purpose: Leiden and DBSCAN
    clusters can be elongated or non-convex, and collapsing them to a centroid
    would hand back exactly the shape assumption those methods avoid.

    reference_labels may contain -1 (DBSCAN noise); it is treated as a label of
    its own, so a point surrounded by noise is also called noise.
    """
    offset = 1 if reference_labels.min() < 0 else 0
    shifted = torch.from_numpy(reference_labels + offset).to(device)
    n_labels = int(shifted.max().item()) + 1

    ref = torch.from_numpy(reference).to(device)
    query = torch.from_numpy(features)
    out = np.empty(len(features), dtype=np.int64)

    for start in range(0, len(features), chunk):
        stop = min(start + chunk, len(features))
        block = query[start:stop].to(device)
        d = torch.cdist(block, ref)
        _, idx = d.topk(min(k, len(ref)), largest=False)
        votes = shifted[idx]
        counts = torch.zeros(len(block), n_labels, device=device)
        counts.scatter_add_(1, votes, torch.ones_like(votes, dtype=counts.dtype))
        out[start:stop] = counts.argmax(dim=1).cpu().numpy()

    del ref
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out - offset
