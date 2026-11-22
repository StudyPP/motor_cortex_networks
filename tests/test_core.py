"""Core unit tests for graph metrics, signed modularity, moments, and phase splitting."""

from __future__ import annotations

import unittest

import networkx as nx
import numpy as np
import pandas as pd

from src.data import correlation, correlation_from_moments, trial_moments
from src.graph_metrics import (
    betweenness,
    compute_graph_metrics,
    onnela_clustering,
    strength_preserving,
    weight_permutation,
)
from src.modularity import adjusted_rand_index, holm_adjust, modularity_matrix, score_partition
from src.phases import phase_segments


class TestCore(unittest.TestCase):
    def _random_graph(self, n: int = 16, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        a = rng.uniform(0.05, 0.8, (n, n))
        a = (a + a.T) / 2.0
        a[rng.random((n, n)) < 0.35] = 0.0
        a = np.triu(a, 1)
        return a + a.T

    def test_graph_metrics_match_networkx(self):
        w = self._random_graph()
        g = nx.Graph()
        for i in range(len(w)):
            for j in range(i + 1, len(w)):
                if w[i, j] > 0:
                    g.add_edge(i, j, weight=w[i, j], length=1.0 / w[i, j])

        # Clustering with own max agrees with nx.clustering
        c_ours = onnela_clustering(w, scale=None)
        c_nx = np.array([nx.clustering(g, i, weight="weight") for i in range(len(w))])
        np.testing.assert_allclose(c_ours, c_nx, atol=1e-12)

        # Path length and betweenness agree with NetworkX
        m, nodes = compute_graph_metrics(w, clustering_scale=float(w.max()), with_nodes=True)
        nx_path = nx.average_shortest_path_length(g, weight="length")
        self.assertAlmostEqual(m["path_length"], nx_path, places=11)

        nx_bc = nx.betweenness_centrality(g, weight="length", normalized=True)
        bc_nx = np.array([nx_bc[i] for i in range(len(w))])
        np.testing.assert_allclose(betweenness(w), bc_nx, atol=1e-12)

    def test_null_models_preserve_properties(self):
        w = self._random_graph(n=20, seed=1)
        perm = weight_permutation(w, np.random.default_rng(2))
        np.testing.assert_allclose(np.sort(w.ravel()), np.sort(perm.ravel()))

        sp, r = strength_preserving(w, seed=3)
        self.assertGreater(r, 0.70)
        np.testing.assert_allclose(np.count_nonzero(w, axis=1), np.count_nonzero(sp, axis=1))

    def test_signed_modularity_and_ari(self):
        w = np.array([
            [0.0, 0.6, -0.3, -0.2],
            [0.6, 0.0, -0.2, -0.4],
            [-0.3, -0.2, 0.0, 0.7],
            [-0.2, -0.4, 0.7, 0.0],
        ])
        b = modularity_matrix(w)
        q_good = score_partition(b, np.array([0, 0, 1, 1]))
        q_bad = score_partition(b, np.array([0, 1, 0, 1]))
        self.assertGreater(q_good, q_bad)
        self.assertAlmostEqual(adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]), 1.0)
        np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_correlation_from_moments_matches_direct(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 10))
        offsets = np.array([0, 30, 75, 120])
        n, s, c = trial_moments(x, offsets)
        np.testing.assert_allclose(correlation(x), correlation_from_moments(n, s, c), atol=1e-12)

    def test_phase_segments_share_event_bins(self):
        trials = pd.DataFrame({
            "first_session_bin": [100],
            "pooled_row_start": [0],
            "pooled_row_stop": [50],
            "n_bins": [50],
            "target_on_s": [2.20],   # bin 110 -> row 10
            "go_cue_s": [2.50],      # bin 125 -> row 25
            "move_onset_s": [2.70],  # bin 135 -> row 35
        })
        seg = phase_segments(trials)
        self.assertEqual(tuple(seg["baseline"][0]), (0, 11))
        self.assertEqual(tuple(seg["delay"][0]), (10, 26))
        self.assertEqual(tuple(seg["reaction"][0]), (25, 36))
        self.assertEqual(tuple(seg["movement"][0]), (35, 50))


if __name__ == "__main__":
    unittest.main()
