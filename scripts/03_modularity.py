#!/usr/bin/env python
"""Step 3 (Part 2): Signed area modularity, 108 condition networks, and Louvain communities."""

from __future__ import annotations

import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import correlation, load_pooled_data, smooth_trials  # noqa: E402
from src.modularity import holm_adjust, label_permutation_test, run_louvain  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "outputs")
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs/part2")
    parser.add_argument("--n-perm", type=int, default=999)
    parser.add_argument("--n-runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5660)
    parser.add_argument("--cond-seed-base", type=int, default=15660)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    data = load_pooled_data(args.data_dir)
    r_full = data["full_correlation"]
    unit_ids, areas = data["unit_ids"], data["area_labels"]
    trials = data["trials"]

    # 1. Full-network area modularity
    full_summary, full_null = label_permutation_test(r_full, areas, n_perm=args.n_perm, seed=args.seed)
    print(f"Full pooled Q_area = {full_summary['q_observed']:.4f} (p = {full_summary['p_upper']:.3f})")
    (out / "area_modularity.json").write_text(json.dumps(full_summary, indent=2) + "\n")
    np.savez_compressed(out / "permutation_nulls.npz", full=full_null)

    # 2. Full-network Louvain communities
    comm = run_louvain(r_full, areas, n_runs=args.n_runs, seed=args.seed)
    runs_df = pd.DataFrame(comm["runs"])
    runs_df.to_csv(out / "community_runs.csv", index=False)
    np.savez_compressed(
        out / "community_partitions.npz",
        labels=comm["labels"],
        coassignment=comm["coassignment"],
        pairwise_ari=comm["pairwise_ari"],
        representative_labels=comm["representative_labels"],
        unit_ids=unit_ids,
    )
    print(
        f"Louvain: median Q = {runs_df['q'].median():.4f}, "
        f"median ARI with areas = {runs_df['ari_area'].median():.4f}"
    )

    # 3. Condition-specific networks (36 mazes x 3 versions = 108 conditions)
    smoothed = smooth_trials(data["counts"], data["offsets"])
    keys = sorted(set(zip(trials["trial_type"].astype(int), trials["trial_version"].astype(int))))
    cond_rows = []
    for idx, (ttype, tver) in enumerate(keys):
        sub = trials[(trials["trial_type"] == ttype) & (trials["trial_version"] == tver)]
        seg = np.concatenate([smoothed[a:b] for a, b in zip(sub["pooled_row_start"], sub["pooled_row_stop"])], axis=0)
        r_cond = correlation(seg)
        s_cond, _ = label_permutation_test(r_cond, areas, n_perm=args.n_perm, seed=args.cond_seed_base + idx)
        cond_rows.append({
            "condition_index": idx,
            "trial_type": ttype,
            "trial_version": tver,
            "n_trials": len(sub),
            "n_bins": len(seg),
            **s_cond,
        })

    cond_df = pd.DataFrame(cond_rows)
    cond_df["p_holm"] = holm_adjust(cond_df["p_upper"].to_numpy())
    cond_df.to_csv(out / "condition_summary.csv", index=False)
    print(
        f"Conditions ({len(cond_df)}): median Q_area = {cond_df['q_observed'].median():.4f} "
        f"(range {cond_df['q_observed'].min():.4f} - {cond_df['q_observed'].max():.4f})"
    )
    print(f"Saved Part 2 results to {out}")


if __name__ == "__main__":
    main()
