#!/usr/bin/env python
"""Step 2 (Part 1): Single-area graph statistics, random null models, and trial bootstrap."""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import (  # noqa: E402
    AREA_NAMES,
    correlation_from_moments,
    load_pooled_data,
    smooth_trials,
    trial_moments,
)
from src.graph_metrics import (  # noqa: E402
    absolute_graph,
    compute_graph_metrics,
    normalized_degree,
    positive_graph,
    strength_preserving,
    weight_permutation,
)

DENSITIES = (0.05, 0.10, 0.20)
METRICS = (
    "mean_strength",
    "clustering",
    "clustering_own_max",
    "clustering_top10_binary",
    "path_length",
    "efficiency",
    "betweenness_cv",
    "betweenness_max",
    "degree_sd_05",
    "degree_sd_10",
    "degree_sd_20",
)

_STATE: dict = {}


def _metrics_row(w: np.ndarray, **extra) -> dict:
    row = compute_graph_metrics(w, clustering_scale=_STATE["scale"])
    for d in DENSITIES:
        row[f"degree_sd_{int(d * 100):02d}"] = float(normalized_degree(w, density=d).std())
    row.update(extra)
    return row


def _null_worker(args):
    area, kind, index, seed, weight = args
    w = _STATE[(weight, area)]
    if kind == "weight_permutation":
        null = weight_permutation(w, np.random.default_rng(seed))
        strength_r = float("nan")
    else:
        null, strength_r = strength_preserving(w, seed)
    return _metrics_row(
        null,
        area=area,
        reference=kind,
        index=index,
        seed=seed,
        weights=weight,
        strength_corr=strength_r,
    )


def _boot_worker(args):
    index, seed = args
    n, sums, cross, areas = _STATE["moments"]
    rng = np.random.default_rng(seed)
    mult = np.bincount(rng.integers(0, len(n), len(n)), minlength=len(n))
    r = correlation_from_moments(n, sums, cross, mult)
    m = int(min((areas == 0).sum(), (areas == 1).sum()))
    pmd_matched = np.sort(rng.choice(np.flatnonzero(areas == 0), m, replace=False))
    return [
        _metrics_row(positive_graph(r, nodes), area=area, index=index, seed=seed)
        for area, nodes in (("PMd", areas == 0), ("M1", areas == 1), ("PMd_matched", pmd_matched))
    ]


def summarize_nulls(empirical: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (weights, area, ref), group in nulls.groupby(["weights", "area", "reference"]):
        emp = empirical[(empirical.weights == weights) & (empirical.area == area)].iloc[0]
        for metric in METRICS:
            vals = group[metric].to_numpy(float)
            mean, sd = vals.mean(), vals.std(ddof=1)
            obs = float(emp[metric])
            fixed = metric == "mean_strength" or sd <= 1e-12 * max(1.0, abs(mean))
            far = np.abs(vals - mean) >= abs(obs - mean) - 1e-12 * max(1.0, abs(mean))
            rows.append({
                "weights": weights,
                "area": area,
                "reference": ref,
                "metric": metric,
                "observed": obs,
                "null_mean": mean,
                "null_sd": sd,
                "null_q025": np.quantile(vals, 0.025),
                "null_q975": np.quantile(vals, 0.975),
                "ratio_to_null": obs / mean if mean != 0 else np.nan,
                "z": np.nan if fixed else (obs - mean) / sd,
                "p_two_sided": np.nan if fixed else (1 + int(far.sum())) / (len(vals) + 1),
            })
    return pd.DataFrame(rows)


def summarize_differences(
    empirical: pd.DataFrame, boot: pd.DataFrame, matched_emp: pd.DataFrame
) -> pd.DataFrame:
    emp = empirical[empirical.weights == "positive"].set_index("area")
    rows = []
    wide = boot.pivot(index="index", columns="area", values=list(METRICS))
    for label, first in (("all_nodes", "PMd"), ("size_matched", "PMd_matched")):
        for metric in METRICS:
            diff = (wide[(metric, first)] - wide[(metric, "M1")]).to_numpy(float)
            pmd_val = float(emp.loc["PMd", metric]) if label == "all_nodes" else float(matched_emp[metric].mean())
            m1_val = float(emp.loc["M1", metric])
            obs = pmd_val - m1_val
            q025, q975 = np.quantile(diff, [0.025, 0.975])
            rows.append({
                "comparison": label,
                "metric": metric,
                "pmd_observed": pmd_val,
                "m1_observed": m1_val,
                "difference": obs,
                "relative_to_m1": obs / m1_val,
                "boot_mean": diff.mean(),
                "boot_bias": diff.mean() - obs,
                "percentile_ci025": q025,
                "percentile_ci975": q975,
                "basic_ci025": 2 * obs - q975,
                "basic_ci975": 2 * obs - q025,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "outputs")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/part1")
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-matched", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=35660)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    data = load_pooled_data(args.data_dir)
    ids, areas, r = data["unit_ids"], data["area_labels"], data["full_correlation"]
    scale = float(r.max())
    _STATE["scale"] = scale

    # 1. Empirical metrics and node-level tables
    empirical_rows, node_rows = [], []
    for code, area in enumerate(AREA_NAMES):
        mask = areas == code
        signed = r[np.ix_(mask, mask)]
        iu = np.triu_indices(mask.sum(), 1)
        for weight_name, builder in (("positive", positive_graph), ("absolute", absolute_graph)):
            w = builder(r, mask)
            _STATE[(weight_name, area)] = w
            row, nodes = compute_graph_metrics(w, clustering_scale=scale, node_ids=ids[mask], with_nodes=True)
            for d in DENSITIES:
                row[f"degree_sd_{int(d * 100):02d}"] = float(normalized_degree(w, density=d).std())
            row.update(
                area=area,
                weights=weight_name,
                max_weight=float(w.max()),
                negative_edge_fraction=float(np.mean(signed[iu] < 0)),
            )
            empirical_rows.append(row)
            if weight_name == "positive":
                for k, uid in enumerate(ids[mask]):
                    node_rows.append({
                        "unit_id": int(uid),
                        "area": area,
                        **{key: float(val[k]) for key, val in nodes.items()},
                    })

    empirical = pd.DataFrame(empirical_rows)
    empirical.to_csv(out / "empirical_metrics.csv", index=False)
    pd.DataFrame(node_rows).to_csv(out / "node_metrics.csv", index=False)

    root = np.random.SeedSequence(args.seed)
    null_seq, boot_seq, matched_seq = root.spawn(3)

    # Size-matched empirical estimate (PMd subsampled to 90 units)
    m = int((areas == 1).sum())
    rng = np.random.default_rng(matched_seq)
    matched_emp = pd.DataFrame([
        _metrics_row(positive_graph(r, np.sort(rng.choice(np.flatnonzero(areas == 0), m, replace=False))), index=i)
        for i in range(args.n_matched)
    ])
    print(f"[{time.time() - t0:.0f}s] Empirical metrics done.", flush=True)

    # 2. Random null ensembles
    jobs = []
    seeds = iter(null_seq.generate_state(args.n_null * 6, dtype=np.uint32))
    for area in AREA_NAMES:
        for i in range(args.n_null):
            jobs.append((area, "weight_permutation", i, int(next(seeds)), "positive"))
            jobs.append((area, "strength_preserving", i, int(next(seeds)), "positive"))
            jobs.append((area, "weight_permutation", i, int(next(seeds)), "absolute"))

    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        nulls = pd.DataFrame(pool.map(_null_worker, jobs, chunksize=8))
    nulls.to_csv(out / "null_metrics.csv", index=False)
    print(f"[{time.time() - t0:.0f}s] Null models done.", flush=True)

    # 3. Trial bootstrap
    smoothed = smooth_trials(data["counts"], data["offsets"])
    n, sums, cross = trial_moments(smoothed, data["offsets"])
    del smoothed
    _STATE["moments"] = (n, sums, cross, areas)

    boot_jobs = [(i, int(s)) for i, s in enumerate(boot_seq.generate_state(args.n_boot, dtype=np.uint32))]
    with ctx.Pool(args.workers) as pool:
        boot = pd.DataFrame([row for rows in pool.map(_boot_worker, boot_jobs, chunksize=4) for row in rows])
    boot.to_csv(out / "bootstrap_metrics.csv", index=False)
    print(f"[{time.time() - t0:.0f}s] Trial bootstrap done.", flush=True)

    summarize_nulls(empirical, nulls).to_csv(out / "null_comparison.csv", index=False)
    summarize_differences(empirical, boot, matched_emp).to_csv(out / "area_differences.csv", index=False)
    print(f"[{time.time() - t0:.0f}s] Saved Part 1 results to {out}", flush=True)


if __name__ == "__main__":
    main()
