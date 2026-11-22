"""Weighted graph statistics and random null models for positive-edge networks."""

from __future__ import annotations

import warnings

import bct
import numpy as np
from scipy.sparse.csgraph import shortest_path

DEGREE_DENSITY = 0.10


def positive_graph(r: np.ndarray, nodes=None) -> np.ndarray:
    """Subgraph keeping only positive correlations (negative set to 0, diagonal 0)."""
    r = np.asarray(r, dtype=np.float64)
    if nodes is not None:
        r = r[np.ix_(nodes, nodes)]
    w = np.where(r > 0, r, 0.0)
    np.fill_diagonal(w, 0.0)
    return (w + w.T) / 2.0


def absolute_graph(r: np.ndarray, nodes=None) -> np.ndarray:
    """Subgraph using |r| as edge weights (diagonal 0)."""
    r = np.asarray(r, dtype=np.float64)
    if nodes is not None:
        r = r[np.ix_(nodes, nodes)]
    w = np.abs(r)
    np.fill_diagonal(w, 0.0)
    return (w + w.T) / 2.0


def edge_lengths(w: np.ndarray) -> np.ndarray:
    """Convert edge weights to lengths 1/w (0 for absent edges)."""
    out = np.zeros_like(w, dtype=np.float64)
    np.divide(1.0, w, out=out, where=w > 0)
    return out


def shortest_distances(w: np.ndarray) -> np.ndarray:
    """All-pairs shortest weighted distances via Dijkstra's algorithm."""
    return shortest_path(edge_lengths(w), method="D", directed=False)


def betweenness(w: np.ndarray) -> np.ndarray:
    """Normalized weighted betweenness centrality on edge lengths 1/w."""
    n = len(w)
    return np.asarray(bct.betweenness_wei(edge_lengths(w)), dtype=float) / ((n - 1) * (n - 2))


def top_density_adjacency(w: np.ndarray, density: float = DEGREE_DENSITY, node_ids=None) -> np.ndarray:
    """Keep floor(density * n*(n-1)/2) strongest positive edges."""
    n = len(w)
    ids = np.arange(n) if node_ids is None else np.asarray(node_ids)
    row, col = np.triu_indices(n, k=1)
    count = int(np.floor(float(density) * len(row)))
    low, high = np.minimum(ids[row], ids[col]), np.maximum(ids[row], ids[col])
    order = np.lexsort((high, low, -w[row, col]))[:count]
    out = np.zeros_like(w, dtype=np.float64)
    out[row[order], col[order]] = w[row[order], col[order]]
    return out + out.T


def normalized_degree(w: np.ndarray, density: float = DEGREE_DENSITY, node_ids=None) -> np.ndarray:
    """Normalized degree k / (n - 1) in the top-density binary graph."""
    top = top_density_adjacency(w, density=density, node_ids=node_ids)
    return (top > 0).sum(axis=1) / (len(w) - 1)


def onnela_clustering(w: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Onnela weighted clustering coefficient.

    Pass a shared ``scale`` (e.g. max weight of the full network) when comparing
    clustering across different graphs; if None, uses each graph's own max weight.
    """
    w = np.asarray(w, dtype=np.float64)
    if scale is None:
        scale = float(w.max())
    if scale <= 0:
        return np.zeros(len(w), dtype=np.float64)
    degree = np.count_nonzero(w, axis=1)
    root = np.cbrt(w / scale)
    triangles = np.einsum("ij,ij->i", root @ root, root)
    denom = degree * (degree - 1)
    out = np.zeros(len(w), dtype=np.float64)
    np.divide(triangles, denom, out=out, where=denom > 0)
    return out


def binary_clustering(adj: np.ndarray) -> float:
    """Mean unweighted clustering coefficient."""
    a = (np.asarray(adj) > 0).astype(np.float64)
    k = a.sum(axis=1)
    tri = np.einsum("ij,ij->i", a @ a, a)
    denom = k * (k - 1)
    c = np.zeros(len(a), dtype=np.float64)
    np.divide(tri, denom, out=c, where=denom > 0)
    return float(c.mean())


def compute_graph_metrics(
    w: np.ndarray,
    *,
    clustering_scale: float,
    node_ids=None,
    with_nodes: bool = False,
):
    """Compute graph-level (and optionally node-level) weighted network statistics."""
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    strength = w.sum(axis=1) / (n - 1)
    clust = onnela_clustering(w, scale=clustering_scale)
    clust_own = onnela_clustering(w, scale=None)
    dist = shortest_distances(w)
    off = ~np.eye(n, dtype=bool)
    dd = dist[off]
    path = float(np.mean(dd)) if np.isfinite(dd).all() else float("inf")
    eff = float(np.mean(np.where(np.isfinite(dd) & (dd > 0), 1.0 / np.where(dd > 0, dd, 1.0), 0.0)))
    bc = betweenness(w)
    top = top_density_adjacency(w, density=DEGREE_DENSITY, node_ids=node_ids)
    deg = (top > 0).sum(axis=1) / (n - 1)

    metrics = {
        "n_nodes": n,
        "edge_density": float(np.count_nonzero(w[off]) / off.sum()),
        "mean_strength": float(strength.mean()),
        "clustering": float(clust.mean()),
        "clustering_own_max": float(clust_own.mean()),
        "clustering_top10_binary": binary_clustering(top),
        "path_length": path,
        "efficiency": eff,
        "betweenness_mean": float(bc.mean()),
        "betweenness_cv": float(bc.std() / bc.mean()) if bc.mean() > 0 else float("nan"),
        "betweenness_max": float(bc.max()),
        "degree_sd": float(deg.std()),
    }
    if not with_nodes:
        return metrics
    nodes = {
        "strength": strength,
        "clustering": clust,
        "betweenness": bc,
        "normalized_degree": deg,
        "closeness": (n - 1) / dist.sum(axis=1),
    }
    return metrics, nodes


def weight_permutation(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle weights across all unordered node pairs (zeros included)."""
    n = len(w)
    row, col = np.triu_indices(n, 1)
    out = np.zeros_like(w, dtype=np.float64)
    out[row, col] = rng.permutation(w[row, col])
    return out + out.T


def strength_preserving(w: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    """Rubinov-Sporns null model preserving degree and approximate node strength."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out, r = bct.null_model_und_sign(w, seed=int(seed))
    out = np.where(out > 0, out, 0.0)
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 0.0)
    return out, float(r[0])
