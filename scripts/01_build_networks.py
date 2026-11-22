#!/usr/bin/env python
"""Step 1: Bin spikes (20 ms), smooth within trials (sigma = 60 ms), and build the pooled network."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import BIN_MS, SIGMA_BINS, build_pooled_counts, correlation, smooth_trials  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nwb",
        type=Path,
        default=REPO / "data/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb",
        help="Path to MC_Maze NWB file.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs")
    parser.add_argument("--bin-ms", type=int, default=BIN_MS)
    parser.add_argument("--sigma-bins", type=float, default=SIGMA_BINS)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.nwb} ...")
    data = build_pooled_counts(args.nwb, bin_ms=args.bin_ms)
    counts, offsets = data["counts"], data["offsets"]
    unit_ids, area_labels = data["unit_ids"], data["area_labels"]
    trials = data["trials"]
    print(f"Binned {counts.shape[0]:,} bins x {counts.shape[1]} units across {len(trials):,} trials.")

    smoothed = smooth_trials(counts, offsets, sigma_bins=args.sigma_bins)
    full_corr = correlation(smoothed)

    trials.to_csv(out / "trials.csv", index=False)
    with h5py.File(out / "pooled_counts.h5", "w") as f:
        f.create_dataset("counts", data=counts, compression="gzip", compression_opts=4, shuffle=True)
        f.create_dataset("trial_offsets", data=offsets)
        f.create_dataset("unit_ids", data=unit_ids)
        f.create_dataset("area_labels", data=area_labels)
        f.attrs["bin_ms"] = args.bin_ms
        f.attrs["sigma_bins"] = args.sigma_bins

    np.savez_compressed(
        out / "networks.npz",
        full_correlation=full_corr,
        unit_ids=unit_ids,
        area_labels=area_labels,
    )
    print(f"Saved pooled counts, trial table, and correlation matrix to {out}")


if __name__ == "__main__":
    main()
