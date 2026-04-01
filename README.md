# Parflex

This repository implements **Sparse Parallel FlexDTW** (ParFlex): a chunk-based, parallel version of FlexDTW for aligning long sequences (e.g. audio or feature sequences) with controlled memory and runtime.

## Overview

**Parflex.ipynb** defines the core algorithm:

- The cost matrix is split into **overlapping chunks** of length **L** (chunk size). Each chunk is aligned with FlexDTW independently, then boundaries are synchronized and the best path is stitched across chunks.
- **L** is the main tuning parameter: larger L uses more memory per chunk but can reduce overhead; smaller L reduces memory at the cost of more chunks and more stitching.
- The notebook provides:
  - **`parflex(C, steps, weights, beta, L=None)`** — one-shot alignment on a precomputed cost matrix `C` (chunked FlexDTW plus stage-2 scan and stitch).
  - **`tiled_stage1_from_features(F1, F2, ...)`** — stage 1 only: build `C` from feature matrices, run FlexDTW on tiles, return `C` and a tiled structure (for stage 2, plots, or sparsity analysis).
  - **`run_stage2_from_tiled(...)`** — stage 2 only when stage 1 was run separately.
  - Plotting and coordinate helpers (e.g. **`plot_alignment_with_tile_background`**).

The rest of the repo is a **pipeline** (notebooks 01–05) that prepares data, runs alignment with this method, and evaluates results.

---

## Environment

Create and activate the conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate mir
```

The environment name is **`mir`**. All commands below assume this environment is active (or use `conda run -n mir ...`).

---

## Pipeline: Run order

Run the notebooks **in order** (each may depend on outputs of the previous ones):

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | **01_Modify_Original_Data.ipynb** | Modify annotation files (e.g. beat labels with measure information). |
| 2 | **02_Make_Train_Test_Set.ipynb** | Build train/test splits. |
| 3 | **03_DataPrep.ipynb** | Data preparation for alignment. |
| 4 | **04_Align_Benchmarks.ipynb** | Run alignment (uses Parflex / FlexDTW). |
| 5 | **05_Evaluate_Benchmarks.ipynb** | Evaluate alignment results (e.g. for different chunk sizes **L**). |

Execute them in Jupyter/Lab or with `jupyter nbconvert --to notebook --execute <notebook>.ipynb` as needed.
---

## Trying different values of L

Alignment uses the chunk length **L** defined in the pipeline. To sweep or change L:

1. Edit **`parflex_config.py`**.
2. Set **`L_VALUES`** to the chunk lengths you want (e.g. `[100, 200, 500, 1000, 2000, 4000]`).
3. Optionally set **`EXPERIMENTS_TRAIN_ROOT`**, **`SPARSE_PARFLEX_SUBDIR`**, and **`FEATURES_ROOT`** to match your layout (alignment writes under `EXPERIMENTS_TRAIN_ROOT`/`SPARSE_PARFLEX_SUBDIR`/`L_<value>`/).
4. Run your sweep by executing the notebook(s) that import this module (e.g. **04** and **05** use `parflex_config` for `L_VALUES` and paths where applicable).

This module is **configuration only**; it does not run the sweep by itself.

---

## Other details

- **Paths**: Several notebooks and **`parflex_config.py`** use path variables (e.g. annotation dirs, feature roots, `EXPERIMENTS_TRAIN_ROOT`). Update these to match your data layout.
- **Dependencies**: The pipeline uses **FlexDTW**, **DTW**, **NWTW**, and **Parflex** (`import Parflex` from **`Parflex.ipynb`** via `import_ipynb`). Ensure those modules/notebooks are on the path when running 04 and 05.
- **Long runs**: 04 can be slow for large benchmarks; `run_workflow.sh` uses `ExecutePreprocessor.timeout=-1` so the kernel does not time out. Monitor progress via `run.log` when running in the background.

With the environment set up and paths adjusted, running 01 → 02 → 03 → 04 → 05 and then varying L via **`parflex_config.py`** should reproduce and extend the results described in the repo.
