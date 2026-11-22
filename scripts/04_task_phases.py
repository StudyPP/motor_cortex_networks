#!/usr/bin/env python
"""Step 4 (Part 3): Task-phase networks, swap tests, bootstrap, Q(t), and per-phase Louvain."""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import multiprocessing as mp
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import (  # noqa: E402
    AREA_NAMES,
    BIN_MS,
    correlation,
    correlation_from_moments,
    load_pooled_data,
    smooth_trials,
)
from src.graph_metrics import compute_graph_metrics, positive_graph  # noqa: E402
from src.modularity import (  # noqa: E402
    adjusted_rand_index,
    holm_adjust,
    label_permutation_test,
    modularity_matrix,
    run_louvain,
    score_partition,
)
from src.phases import (  # noqa: E402
    PHASE_ORDER,
    event_bins,
    ordered_event_mask,
    phase_segments,
    segment_moments,
    segment_rows,
    sliding_offsets,
)

_STATE: dict = {}


def _q_area(r: np.ndarray) -> float:
    return score_partition(modularity_matrix(r), _STATE["areas"])


def _signed_summary(r: np.ndarray, areas: np.ndarray) -> dict:
    out = {}
    for name, a, b in (("pmd", 0, 0), ("m1", 1, 1), ("between", 0, 1)):
        block = r[np.ix_(areas == a, areas == b)]
        vals = block[np.triu_indices(len(block), 1)] if a == b else block.ravel()
        out[f"mean_r_{name}"] = float(vals.mean())
        out[f"negative_fraction_{name}"] = float(np.mean(vals < 0))
    return out


def _area_metrics(r: np.ndarray, areas: np.ndarray, scale: float) -> dict:
    out = {}
    for code, area in enumerate(AREA_NAMES):
        m = compute_graph_metrics(positive_graph(r, areas == code), clustering_scale=scale)
        for key in ("edge_density", "mean_strength", "clustering", "path_length", "efficiency",
                    "betweenness_cv", "degree_sd"):
            out[f"{area}_{key}"] = m[key]
    return out


def _swap_worker(args):
    a, b, index, seed = args
    na, sa, ca = _STATE["moments"][a]
    nb, sb, cb = _STATE["moments"][b]
    flip = np.random.default_rng(seed).random(len(na)) < 0.5
    keep, swap = (~flip).astype(float), flip.astype(float)
    ra = correlation_from_moments(
        np.array([keep @ na + swap @ nb]),
        (keep @ sa + swap @ sb)[None],
        (np.tensordot(keep, ca, axes=1) + np.tensordot(swap, cb, axes=1))[None],
    )
    rb = correlation_from_moments(
        np.array([keep @ nb + swap @ na]),
        (keep @ sb + swap @ sa)[None],
        (np.tensordot(keep, cb, axes=1) + np.tensordot(swap, ca, axes=1))[None],
    )
    return {"phase_a": a, "phase_b": b, "index": index, "q_a": _q_area(ra), "q_b": _q_area(rb)}


def _boot_worker(args):
    index, seed = args
    n_trials = len(next(iter(_STATE["moments"].values()))[0])
    w = np.bincount(np.random.default_rng(seed).integers(0, n_trials, n_trials), minlength=n_trials)
    row = {"index": index}
    for phase, (n, s, cr) in _STATE["moments"].items():
        r = correlation_from_moments(n, s, cr, w)
        row[f"q_{phase}"] = _q_area(r)
    return row


def _matched_worker(args):
    phase, index, seed, target_bins = args
    n, s, cr = _STATE["moments"][phase]
    order = np.random.default_rng(seed).permutation(len(n))
    take = order[: int(np.searchsorted(np.cumsum(n[order]), target_bins)) + 1]
    w = np.zeros(len(n))
    w[take] = 1.0
    return {"phase": phase, "index": index, "q_area": _q_area(correlation_from_moments(n, s, cr, w))}


def _window_worker(args):
    align, offset, seed = args
    trials = _STATE["trials"]
    smoothed, counts = _STATE["smoothed"], _STATE["counts"]
    areas, scale, window = _STATE["areas"], _STATE["scale"], _STATE["window_bins"]

    event = "target_on_s" if align == "target" else "move_onset_s"
    e = event_bins(trials[event].to_numpy())
    first = trials["first_session_bin"].to_numpy()
    row0, row1 = trials["pooled_row_start"].to_numpy(), trials["pooled_row_stop"].to_numpy()
    start = row0 + e - first + offset
    inside = (start >= row0) & (start + window <= row1)
    if align == "target":
        go_row = row0 + np.maximum(event_bins(trials["go_cue_s"].to_numpy()), e) - first
        inside &= start + window <= go_row
    rows = (start[inside][:, None] + np.arange(window)[None, :]).ravel()
    r = correlation(smoothed[rows])
    summary, _ = label_permutation_test(r, areas, n_perm=999, seed=int(seed))
    rates = counts[rows].mean(axis=0) / (BIN_MS / 1000.0)
    return {
        "align": align,
        "offset_bins": int(offset),
        "window_center_ms": (offset + window / 2.0) * BIN_MS,
        "n_trials": int(inside.sum()),
        "n_bins": int(len(rows)),
        "q_area": summary["q_observed"],
        "null_median": summary["null_median"],
        "null_q025": summary["null_q025"],
        "null_q975": summary["null_q975"],
        "delta_q": summary["delta_q"],
        "p_upper": summary["p_upper"],
        "rate_pmd_hz": float(rates[areas == 0].mean()),
        "rate_m1_hz": float(rates[areas == 1].mean()),
        **_signed_summary(r, areas),
        **_area_metrics(r, areas, scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "outputs")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/part3")
    parser.add_argument("--n-perm", type=int, default=999)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-matched", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--step-ms", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=45660)
    parser.add_argument("--comm-seed-base", type=int, default=55660)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = load_pooled_data(args.data_dir)
    ids, areas = data["unit_ids"], data["area_labels"]
    counts, offsets = data["counts"], data["offsets"]
    trials = data["trials"]
    trials = trials[ordered_event_mask(trials)].reset_index(drop=True)
    smoothed = smooth_trials(counts, offsets)
    scale = float(data["full_correlation"].max())
    segments = phase_segments(trials)

    _STATE.update(
        areas=areas,
        trials=trials,
        smoothed=smoothed,
        counts=counts,
        scale=scale,
        window_bins=args.window_ms // BIN_MS,
    )

    root = np.random.SeedSequence(args.seed)
    label_seq, swap_seq, boot_seq, window_seq, matched_seq = root.spawn(5)
    label_seeds = label_seq.generate_state(len(PHASE_ORDER), dtype=np.uint32)

    # 1. Phase networks and label permutation tests
    moments, records, correlations, nulls = {}, [], {}, {}
    for k, phase in enumerate(PHASE_ORDER):
        seg = segments[phase]
        moments[phase] = segment_moments(smoothed, seg)
        r = correlation_from_moments(*moments[phase])
        correlations[phase] = r
        summary, null = label_permutation_test(r, areas, n_perm=args.n_perm, seed=int(label_seeds[k]))
        nulls[phase] = null
        rows = segment_rows(seg)
        rates = counts[rows].mean(axis=0) / (BIN_MS / 1000.0)
        lengths = (seg[:, 1] - seg[:, 0]) * BIN_MS
        records.append({
            "phase": phase,
            "n_trials": len(seg),
            "n_bins": int(len(rows)),
            "median_trial_ms": float(np.median(lengths)),
            "q_area": summary["q_observed"],
            "null_median": summary["null_median"],
            "null_q025": summary["null_q025"],
            "null_q975": summary["null_q975"],
            "delta_q": summary["delta_q"],
            "p_upper": summary["p_upper"],
            "rate_pmd_hz": float(rates[areas == 0].mean()),
            "rate_m1_hz": float(rates[areas == 1].mean()),
            **_signed_summary(r, areas),
            **_area_metrics(r, areas, scale),
        })

    phase_df = pd.DataFrame(records)
    np.savez_compressed(
        out / "phase_networks.npz",
        phases=np.array(PHASE_ORDER),
        unit_ids=ids,
        area_labels=areas,
        correlations=np.stack([correlations[p] for p in PHASE_ORDER]),
        label_nulls=np.stack([nulls[p] for p in PHASE_ORDER]),
    )
    print(f"[{time.time() - t0:.0f}s] Phase networks done.", flush=True)

    _STATE["moments"] = moments
    ctx = mp.get_context("fork")

    # 2. Within-trial phase swap tests
    pairs = list(combinations(PHASE_ORDER, 2))
    swap_jobs = [
        (a, b, i, int(s))
        for (a, b), seq in zip(pairs, swap_seq.spawn(len(pairs)))
        for i, s in enumerate(seq.generate_state(args.n_perm, dtype=np.uint32))
    ]
    with ctx.Pool(args.workers) as pool:
        swaps = pd.DataFrame(pool.map(_swap_worker, swap_jobs, chunksize=16))

    q_map = phase_df.set_index("phase")["q_area"]
    pair_rows = []
    for a, b in pairs:
        null = (swaps.query("phase_a == @a and phase_b == @b").eval("q_a - q_b")).to_numpy()
        obs = float(q_map[a] - q_map[b])
        count = int(np.sum(np.abs(null) >= abs(obs) - 1e-15))
        pair_rows.append({
            "phase_a": a,
            "phase_b": b,
            "delta_q_a_minus_b": obs,
            "p_two_sided": (1 + count) / (len(null) + 1),
        })
    pair_df = pd.DataFrame(pair_rows)
    pair_df["p_holm"] = holm_adjust(pair_df["p_two_sided"].to_numpy())
    print(f"[{time.time() - t0:.0f}s] Phase swap tests done.", flush=True)

    # 3. Paired trial bootstrap
    boot_jobs = [(i, int(s)) for i, s in enumerate(boot_seq.generate_state(args.n_boot, dtype=np.uint32))]
    with ctx.Pool(args.workers) as pool:
        boot = pd.DataFrame(pool.map(_boot_worker, boot_jobs, chunksize=8))
    for phase in PHASE_ORDER:
        vals = boot[f"q_{phase}"].to_numpy()
        idx = phase_df.index[phase_df.phase == phase][0]
        phase_df.loc[idx, "boot_ci025"] = np.quantile(vals, 0.025)
        phase_df.loc[idx, "boot_ci975"] = np.quantile(vals, 0.975)
        phase_df.loc[idx, "basic_ci025"] = 2 * q_map[phase] - np.quantile(vals, 0.975)
        phase_df.loc[idx, "basic_ci975"] = 2 * q_map[phase] - np.quantile(vals, 0.025)
    print(f"[{time.time() - t0:.0f}s] Trial bootstrap done.", flush=True)

    # 4. Size-matched bin-count control
    target_bins = int(phase_df["n_bins"].min())
    matched_jobs = [
        (phase, i, int(sd), target_bins)
        for phase, seq in zip(PHASE_ORDER, matched_seq.spawn(len(PHASE_ORDER)))
        for i, sd in enumerate(seq.generate_state(args.n_matched, dtype=np.uint32))
    ]
    with ctx.Pool(args.workers) as pool:
        matched = pd.DataFrame(pool.map(_matched_worker, matched_jobs, chunksize=4))
    mq = matched.groupby("phase")["q_area"].agg(["mean", "std"])
    for phase in PHASE_ORDER:
        idx = phase_df.index[phase_df.phase == phase][0]
        phase_df.loc[idx, "size_matched_q_mean"] = mq.loc[phase, "mean"]
        phase_df.loc[idx, "size_matched_q_sd"] = mq.loc[phase, "std"]

    phase_df.to_csv(out / "phase_summary.csv", index=False)
    pair_df.to_csv(out / "phase_pair_tests.csv", index=False)

    # 5. Time-resolved sliding windows Q(t)
    del _STATE["moments"]
    win_jobs = [
        (align, int(off))
        for align, lo, hi in (("target", -300, 600), ("movement", -500, 600))
        for off in sliding_offsets(lo, hi, args.window_ms, args.step_ms)
    ]
    win_seeds = window_seq.generate_state(len(win_jobs), dtype=np.uint32)
    win_jobs = [(a, o, int(s)) for (a, o), s in zip(win_jobs, win_seeds)]
    with ctx.Pool(args.workers) as pool:
        windows = pd.DataFrame(pool.map(_window_worker, win_jobs))
    windows["p_holm_all_windows"] = holm_adjust(windows["p_upper"].to_numpy())
    windows.to_csv(out / "sliding_windows.csv", index=False)
    print(f"[{time.time() - t0:.0f}s] Sliding windows done.", flush=True)

    # 6. Per-phase Louvain communities
    run_rows, phase_comm_rows, module_rows, rep_labels = [], [], [], {}
    for k, phase in enumerate(PHASE_ORDER):
        comm = run_louvain(correlations[phase], areas, n_runs=args.n_runs, seed=args.comm_seed_base + k)
        rep = comm["representative_labels"]
        rep_labels[phase] = rep
        between = comm["pairwise_ari"][np.triu_indices(args.n_runs, 1)]
        runs_df = pd.DataFrame(comm["runs"])
        for rrow in comm["runs"]:
            run_rows.append({"phase": phase, **rrow})
        phase_comm_rows.append({
            "phase": phase,
            "q_median": float(runs_df["q"].median()),
            "ari_area_median": float(runs_df["ari_area"].median()),
            "ari_between_runs_mean": float(between.mean()),
            "representative_run": comm["representative_run"],
        })
        sizes = {m: int((rep == m).sum()) for m in np.unique(rep)}
        for rank, m in enumerate(sorted(sizes, key=lambda x: -sizes[x])):
            n_pmd = int(((rep == m) & (areas == 0)).sum())
            module_rows.append({
                "phase": phase,
                "module_rank": rank,
                "module_label": int(m),
                "n_units": sizes[m],
                "n_pmd": n_pmd,
                "n_m1": sizes[m] - n_pmd,
                "pmd_fraction": float(n_pmd / sizes[m]),
            })

    ari_df = pd.DataFrame(np.eye(len(PHASE_ORDER)), index=PHASE_ORDER, columns=PHASE_ORDER)
    for a, b in combinations(PHASE_ORDER, 2):
        ari_df.loc[a, b] = ari_df.loc[b, a] = adjusted_rand_index(rep_labels[a], rep_labels[b])

    pd.DataFrame(run_rows).to_csv(out / "community_runs.csv", index=False)
    pd.DataFrame(phase_comm_rows).to_csv(out / "community_summary.csv", index=False)
    pd.DataFrame(module_rows).to_csv(out / "community_modules.csv", index=False)
    ari_df.to_csv(out / "community_phase_ari.csv")
    np.savez_compressed(
        out / "community_partitions.npz",
        phases=np.array(PHASE_ORDER),
        unit_ids=ids,
        area_labels=areas,
        representative=np.stack([rep_labels[p] for p in PHASE_ORDER]),
    )
    print(f"[{time.time() - t0:.0f}s] Saved Part 3 results to {out}", flush=True)


if __name__ == "__main__":
    main()
