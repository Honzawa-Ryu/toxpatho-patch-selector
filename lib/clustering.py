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


def leiden_clusters(
    neighbors: np.ndarray, resolution: float, seed: int, n_iterations: int = -1
) -> np.ndarray:
    """Leiden community detection on an undirected kNN graph.

    Args:
        neighbors: (n, k) neighbour indices from knn_graph().
        resolution: RBConfiguration resolution. Higher gives more, smaller communities.
        n_iterations: -1 runs until the partition stops improving.

    The graph is made undirected by keeping each pair once (i < j): a kNN graph is
    asymmetric (j may be among i's neighbours without the reverse), and leaving it
    directed would weight those pairs twice.
    """
    import igraph as ig
    import leidenalg

    n, k = neighbors.shape
    sources = np.repeat(np.arange(n, dtype=np.int64), k)
    targets = neighbors.reshape(-1)
    lo = np.minimum(sources, targets)
    hi = np.maximum(sources, targets)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)

    graph = ig.Graph(n=n, edges=[tuple(e) for e in pairs], directed=False)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
        n_iterations=n_iterations,
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
