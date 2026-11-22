"""Load MC_Maze NWB data, bin spikes, smooth within trials, and compute correlations."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

BIN_MS = 20
BIN_NS = BIN_MS * 1_000_000
SIGMA_BINS = 3.0  # 60 ms at 20 ms bins
AREA_NAMES = ("PMd", "M1")  # area code 0 = PMd (unit ID prefix 1), 1 = M1 (prefix 2)


def seconds_to_ns(values) -> np.ndarray:
    """Convert timestamps in seconds to exact integer nanoseconds."""
    arr = np.asarray(values, dtype=np.float64)
    return np.rint(arr * 1_000_000_000).astype(np.int64)


def full_bin_ranges(intervals: np.ndarray, bin_ns: int = BIN_NS) -> tuple[np.ndarray, np.ndarray]:
    """Return [first, stop) session-grid bin indices fully inside each trial interval."""
    ns = seconds_to_ns(intervals)
    first = (ns[:, 0] + bin_ns - 1) // bin_ns
    stop = ns[:, 1] // bin_ns
    return first, stop


def bin_spikes(spike_times: np.ndarray, selected_bins: np.ndarray, bin_ns: int = BIN_NS) -> np.ndarray:
    """Count spikes falling into each selected session-grid bin."""
    bin_ids = seconds_to_ns(spike_times) // bin_ns
    hist = np.bincount(bin_ids, minlength=int(selected_bins[-1]) + 1)
    return hist[selected_bins]


def build_pooled_counts(nwb_path: Path | str, bin_ms: int = BIN_MS) -> dict:
    """Extract binned spike counts, unit metadata, and trial table from an MC_Maze NWB file."""
    bin_ns = int(bin_ms * 1_000_000)
    with h5py.File(nwb_path, "r") as f:
        u = f["units"]
        t = f["intervals/trials"]

        starts, stops = t["start_time"][:], t["stop_time"][:]
        order = np.argsort(starts, kind="stable")
        intervals = np.column_stack([starts[order], stops[order]])
        first, stop = full_bin_ranges(intervals, bin_ns)
        selected_bins = np.concatenate([np.arange(a, b) for a, b in zip(first, stop)])
        offsets = np.r_[0, np.cumsum(stop - first)]

        unit_order = np.argsort(u["id"][:], kind="stable")
        unit_ids = u["id"][:][unit_order]
        prefixes = np.array([str(int(v))[0] for v in unit_ids])
        area_labels = np.where(prefixes == "1", 0, 1).astype(np.int8)

        spike_end = u["spike_times_index"][:]
        spike_start = np.r_[0, spike_end[:-1]]
        counts = np.empty((len(selected_bins), len(unit_ids)), dtype=np.uint16)
        for col, src_row in enumerate(unit_order):
            spikes = u["spike_times"][spike_start[src_row] : spike_end[src_row]]
            counts[:, col] = bin_spikes(spikes, selected_bins, bin_ns)

        splits = [v.decode() if isinstance(v, bytes) else str(v) for v in t["split"][:][order]]
        trials = pd.DataFrame({
            "trial_order": np.arange(len(order)),
            "trial_id": t["id"][:][order],
            "split": splits,
            "start_s": intervals[:, 0],
            "stop_s": intervals[:, 1],
            "target_on_s": t["target_on_time"][:][order],
            "go_cue_s": t["go_cue_time"][:][order],
            "move_onset_s": t["move_onset_time"][:][order],
            "first_session_bin": first,
            "stop_session_bin_exclusive": stop,
            "pooled_row_start": offsets[:-1],
            "pooled_row_stop": offsets[1:],
            "n_bins": stop - first,
        })
        for col in ("trial_type", "trial_version", "maze_id", "success"):
            if col in t:
                trials[col] = t[col][:][order]

    return {
        "counts": counts,
        "offsets": offsets,
        "unit_ids": unit_ids,
        "area_labels": area_labels,
        "trials": trials,
    }


def smooth_trials(counts: np.ndarray, offsets: np.ndarray, sigma_bins: float = SIGMA_BINS) -> np.ndarray:
    """Apply Gaussian smoothing independently inside each trial (never across boundaries)."""
    out = np.empty(counts.shape, dtype=np.float64)
    for a, b in zip(offsets[:-1], offsets[1:]):
        out[a:b] = gaussian_filter1d(
            counts[a:b].astype(np.float64),
            sigma=sigma_bins,
            axis=0,
            mode="nearest",
            truncate=4.0,
        )
    return out


def correlation(samples: np.ndarray, variance_tol: float = 1e-12) -> np.ndarray:
    """Pearson correlation between columns, zero diagonal; constant columns get zero weight."""
    x = np.asarray(samples, dtype=np.float64)
    x = x - x.mean(axis=0)
    ss = np.einsum("ij,ij->j", x, x)
    ok = ss > variance_tol * max(len(x), 1)
    r = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    if ok.any():
        xo = x[:, ok]
        sub = (xo.T @ xo) / np.sqrt(np.outer(ss[ok], ss[ok]))
        r[np.ix_(ok, ok)] = np.clip((sub + sub.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(r, 0.0)
    return r


def trial_moments(smoothed: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-trial sufficient statistics (sample count, column sums, cross-products) for fast resampling."""
    n_trials = len(offsets) - 1
    n_units = smoothed.shape[1]
    n = np.diff(offsets).astype(np.float64)
    sums = np.empty((n_trials, n_units), dtype=np.float64)
    cross = np.empty((n_trials, n_units, n_units), dtype=np.float64)
    for k, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        x = smoothed[a:b]
        sums[k] = x.sum(axis=0)
        cross[k] = x.T @ x
    return n, sums, cross


def correlation_from_moments(
    n: np.ndarray,
    sums: np.ndarray,
    cross: np.ndarray,
    weights: np.ndarray | None = None,
    variance_tol: float = 1e-12,
) -> np.ndarray:
    """Pooled Pearson correlation from per-trial moments with optional trial weights."""
    w = np.ones(len(n), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    total = float(w @ n)
    s = w @ sums
    c = np.tensordot(w, cross, axes=1)
    cov = c - np.outer(s, s) / total
    d = np.diag(cov).copy()
    ok = d > variance_tol * total
    r = np.zeros_like(cov)
    if ok.any():
        sub = cov[np.ix_(ok, ok)] / np.sqrt(np.outer(d[ok], d[ok]))
        r[np.ix_(ok, ok)] = np.clip((sub + sub.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(r, 0.0)
    return r


def load_pooled_data(output_dir: Path | str) -> dict:
    """Load preprocessed counts, trial metadata, and full correlation matrix from outputs/."""
    out = Path(output_dir)
    with h5py.File(out / "pooled_counts.h5", "r") as f:
        counts = f["counts"][:]
        offsets = f["trial_offsets"][:]
        unit_ids = f["unit_ids"][:]
        area_labels = f["area_labels"][:].astype(int)
    trials = pd.read_csv(out / "trials.csv")
    net = np.load(out / "networks.npz")
    corr = np.array(net["full_correlation"], dtype=np.float64)
    np.fill_diagonal(corr, 0.0)
    return {
        "counts": counts,
        "offsets": offsets,
        "unit_ids": unit_ids,
        "area_labels": area_labels,
        "trials": trials,
        "full_correlation": corr,
    }
