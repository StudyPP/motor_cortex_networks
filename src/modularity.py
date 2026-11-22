"""Symmetric signed modularity (Gomez et al. 2009), permutation tests, Louvain, and ARI."""

from __future__ import annotations

import bct
import numpy as np


def modularity_matrix(w: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Symmetric signed modularity matrix B (Gomez et al. 2009, Eq. 18)."""
    w = np.asarray(w, dtype=np.float64)
    scale = float(np.max(np.abs(w)))
    if scale == 0:
        raise ValueError("Modularity is undefined for an edgeless graph.")
    scaled = w / scale
    pos = np.maximum(scaled, 0.0)
    neg = np.maximum(-scaled, 0.0)
    v_pos = float(pos.sum())
    v_neg = float(neg.sum())
    b = pos - neg
    if v_pos > 0:
        s_pos = pos.sum(axis=1)
        b = b - float(gamma) * np.outer(s_pos, s_pos) / v_pos
    if v_neg > 0:
        s_neg = neg.sum(axis=1)
        b = b + float(gamma) * np.outer(s_neg, s_neg) / v_neg
    return b / (v_pos + v_neg)


def score_partition(b: np.ndarray, labels: np.ndarray) -> float:
    """Compute signed modularity Q from modularity matrix B and node labels."""
    _, encoded = np.unique(labels, return_inverse=True)
    return float(b[encoded[:, None] == encoded[None, :]].sum())


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Hubert-Arabie adjusted Rand index (1 = identical, 0 = chance)."""
    _, a = np.unique(labels_a, return_inverse=True)
    _, b = np.unique(labels_b, return_inverse=True)
    n = len(a)
    if n < 2:
        return 1.0
    n_a, n_b = int(a.max()) + 1, int(b.max()) + 1
    table = np.bincount(a * n_b + b, minlength=n_a * n_b).reshape(n_a, n_b)
    choose2 = lambda c: int(np.sum(c * (c - 1) // 2))
    pairs = n * (n - 1) // 2
    joint = choose2(table)
    rows = choose2(table.sum(axis=1))
    cols = choose2(table.sum(axis=0))
    num = 2 * (joint * pairs - rows * cols)
    den = (rows + cols) * pairs - 2 * rows * cols
    return 1.0 if den == 0 else float(num / den)


def label_permutation_test(
    w: np.ndarray,
    labels: np.ndarray,
    n_perm: int = 999,
    seed: int = 5660,
) -> tuple[dict, np.ndarray]:
    """Test a fixed partition against size-preserving random label permutations."""
    b = modularity_matrix(w)
    _, encoded = np.unique(labels, return_inverse=True)
    q_obs = score_partition(b, encoded)
    rng = np.random.default_rng(seed)
    null = np.array([score_partition(b, rng.permutation(encoded)) for _ in range(n_perm)])
    count_ge = int(np.count_nonzero((null >= q_obs) | np.isclose(null, q_obs, atol=1e-12, rtol=0)))
    summary = {
        "q_observed": q_obs,
        "null_mean": float(null.mean()),
        "null_median": float(np.median(null)),
        "null_sd": float(null.std(ddof=1)) if n_perm > 1 else float("nan"),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "delta_q": q_obs - float(np.median(null)),
        "count_ge": count_ge,
        "p_upper": float((count_ge + 1) / (n_perm + 1)),
    }
    return summary, null


def run_louvain(
    w: np.ndarray,
    area_labels: np.ndarray,
    n_runs: int = 20,
    seed: int = 5660,
) -> dict:
    """Run signed Louvain multiple times and select the medoid (highest mean ARI) partition."""
    b = modularity_matrix(w)
    _, area_enc = np.unique(area_labels, return_inverse=True)
    run_seeds = np.random.SeedSequence(seed).generate_state(n_runs)

    runs, partitions = [], []
    for idx, run_seed in enumerate(run_seeds):
        if float(w.sum()) > 1e-12 * float(np.abs(w).sum()):
            raw, _ = bct.community_louvain(w, gamma=1, B="negative_sym", seed=int(run_seed))
        else:
            carrier = np.zeros_like(w)
            carrier[0, 0] = 1.0
            raw, _ = bct.community_louvain(carrier, B=b.tolist(), seed=int(run_seed))
        _, enc = np.unique(raw, return_inverse=True)
        q = score_partition(b, enc)
        partitions.append(enc)
        runs.append({
            "run": idx,
            "seed": int(run_seed),
            "q": q,
            "n_communities": int(len(np.unique(enc))),
            "ari_area": adjusted_rand_index(enc, area_enc),
        })

    pairwise = np.eye(n_runs)
    for i in range(n_runs):
        for j in range(i):
            pairwise[i, j] = pairwise[j, i] = adjusted_rand_index(partitions[i], partitions[j])

    mean_ari = pairwise.mean(axis=1)
    best_candidates = np.flatnonzero(np.isclose(mean_ari, mean_ari.max(), atol=1e-12, rtol=0))
    rep_idx = max(best_candidates.tolist(), key=lambda i: (runs[i]["q"], -i))
    coassignment = sum((p[:, None] == p[None, :]).astype(float) for p in partitions) / n_runs

    return {
        "runs": runs,
        "labels": np.asarray(partitions),
        "pairwise_ari": pairwise,
        "coassignment": coassignment,
        "representative_run": rep_idx,
        "representative_labels": partitions[rep_idx],
        "q_area": score_partition(b, area_enc),
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down familywise multiple-testing correction."""
    vals = np.asarray(p_values, dtype=float)
    order = np.argsort(vals, kind="stable")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(vals[order] * np.arange(len(vals), 0, -1)))
    adj = np.empty_like(vals)
    adj[order] = adj_sorted
    return adj
