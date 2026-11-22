"""Task-phase segmentation and sliding-window indexing for MC_Maze trials."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BIN_MS, BIN_NS

PHASE_ORDER = ("baseline", "delay", "reaction", "movement")


def event_bins(times_s) -> np.ndarray:
    """Convert event timestamps in seconds to 20 ms session-grid bin indices."""
    return np.rint(np.asarray(times_s, dtype=np.float64) * 1e9).astype(np.int64) // BIN_NS


def ordered_event_mask(trials: pd.DataFrame) -> np.ndarray:
    """Return mask of trials whose target and movement events fall inside the observed trial."""
    tb = event_bins(trials["target_on_s"].to_numpy())
    mb = event_bins(trials["move_onset_s"].to_numpy())
    first = trials["first_session_bin"].to_numpy()
    last = first + trials["n_bins"].to_numpy() - 1
    return (first <= tb) & (mb <= last)


def phase_segments(trials: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return per-trial [start, stop) row indices in the pooled matrix for each task phase.

    Event bins are included in both adjacent phases. In zero-delay trials where the go cue
    is logged 2-3 ms before target onset, the go bin is clamped to the target bin.
    """
    first = trials["first_session_bin"].to_numpy()
    row0 = trials["pooled_row_start"].to_numpy()
    row1 = trials["pooled_row_stop"].to_numpy()
    tb, gb, mb = (
        row0 + event_bins(trials[col].to_numpy()) - first
        for col in ("target_on_s", "go_cue_s", "move_onset_s")
    )
    gb = np.maximum(gb, tb)
    mb = np.maximum(mb, gb)
    return {
        "baseline": np.column_stack([row0, tb + 1]),
        "delay": np.column_stack([tb, gb + 1]),
        "reaction": np.column_stack([gb, mb + 1]),
        "movement": np.column_stack([mb, row1]),
    }


def segment_rows(segments: np.ndarray) -> np.ndarray:
    """Flatten (n_trials, 2) [start, stop) intervals into a 1D array of pooled row indices."""
    return np.concatenate([np.arange(a, b) for a, b in segments])


def segment_moments(
    smoothed: np.ndarray, segments: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-trial sufficient statistics (count, sum, cross-product) for one phase."""
    n_units = smoothed.shape[1]
    n = (segments[:, 1] - segments[:, 0]).astype(np.float64)
    sums = np.empty((len(segments), n_units), dtype=np.float64)
    cross = np.empty((len(segments), n_units, n_units), dtype=np.float64)
    for k, (a, b) in enumerate(segments):
        x = smoothed[a:b]
        sums[k] = x.sum(axis=0)
        cross[k] = x.T @ x
    return n, sums, cross


def sliding_offsets(start_ms: int, stop_ms: int, window_ms: int, step_ms: int) -> np.ndarray:
    """Return window start offsets (in bins) covering [start_ms, stop_ms]."""
    for val, name in ((start_ms, "start_ms"), (stop_ms, "stop_ms"),
                      (window_ms, "window_ms"), (step_ms, "step_ms")):
        if val % BIN_MS:
            raise ValueError(f"{name}={val} ms must be a multiple of {BIN_MS} ms")
    return np.arange(start_ms // BIN_MS, (stop_ms - window_ms) // BIN_MS + 1, step_ms // BIN_MS)
