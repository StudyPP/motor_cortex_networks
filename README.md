# Motor Cortex Correlation Networks (PMd & M1)

Code to reproduce the analyses and figures for the study of single-unit correlation networks in macaque dorsal premotor cortex (PMd) and primary motor cortex (M1) during a delayed maze-reaching task (Neural Latents Benchmark `MC_Maze`).

## Repository structure

```text
motor_cortex_networks/
├── README.md
├── environment.yml          # Conda environment specification
├── requirements.txt         # Pip dependencies
├── src/
│   ├── data.py              # NWB loading, 20 ms binning, within-trial smoothing, correlations
│   ├── graph_metrics.py     # Positive-edge graph statistics and random null models
│   ├── modularity.py        # Signed modularity (Gomez et al. 2009), permutation tests, Louvain, ARI
│   └── phases.py            # Task-phase segmentation and sliding-window helpers
├── scripts/
│   ├── 01_build_networks.py # Build pooled counts, trial metadata, and full correlation network
│   ├── 02_graph_metrics.py  # Part 1: PMd/M1 graph stats, null models, and trial bootstrap
│   ├── 03_modularity.py     # Part 2: Full-network Q_area, 108 condition networks, Louvain
│   └── 04_task_phases.py    # Part 3: Phase networks, swap tests, bootstrap, Q(t), phase Louvain
├── notebooks/
│   └── figures.ipynb        # Generates all 7 manuscript figures (PDF + PNG)
└── tests/
    └── test_core.py         # Unit tests verifying metrics against NetworkX and basic invariants
```

## Setup

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate motor-networks
```

### Option 2: `venv` + `pip`

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the `MC_Maze` full session (`sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb`) from the [DANDI Archive (dandiset 000128)](https://dandiarchive.org/dandiset/000128) and place it in `data/`:

```text
data/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb
```

## Reproducing the results

Run the analysis scripts in order from the repository root:

```bash
# 1. Bin spikes (20 ms), smooth within trials (sigma = 60 ms), and build pooled network (~10 s)
python scripts/01_build_networks.py --nwb data/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb

# 2. Part 1: Single-area graph statistics, 1,000 nulls, 1,000 trial bootstraps (~10 min)
python scripts/02_graph_metrics.py --workers 6

# 3. Part 2: Area modularity, 108 condition networks, and 20 Louvain runs (~30 s)
python scripts/03_modularity.py

# 4. Part 3: Task-phase networks, swap tests, bootstrap, sliding-window Q(t), and Louvain (~4 min)
python scripts/04_task_phases.py --workers 6
```

All outputs are saved under `outputs/`. You can pass `--n-null` / `--n-boot` / `--n-perm` with smaller values for a quick smoke test.

## Generating the figures

Open and run `notebooks/figures.ipynb` in Jupyter:

```bash
jupyter notebook notebooks/figures.ipynb
```

Each figure (Figures 1–7) is plotted in its own section from the saved files in `outputs/` and exported as both vector PDF and 300-dpi PNG to `figures/`.

## Running tests

```bash
pytest tests/
# or
python -m unittest discover -s tests -v
```
