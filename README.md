# Parflex

This repository implements **Sparse Parallel FlexDTW** (ParFlex): a chunk-based, parallel version of FlexDTW for aligning long sequences (e.g. audio or feature sequences) with controlled memory and runtime.

## Overview

**Sparse_Parflex.ipynb** defines the core algorithm:

- The cost matrix is split into **overlapping chunks** of length **L** (chunk size). Each chunk is aligned with FlexDTW independently, then boundaries are synchronized and the best path is stitched across chunks.
- **L** is the main tuning parameter: larger L uses more memory per chunk but can reduce overhead; smaller L reduces memory at the cost of more chunks and more stitching.
- The notebook provides:
  - **`parflex(C, steps, weights, beta, L=None)`** — one-shot: build cost matrix, run chunked FlexDTW, and stitch.
  - **`align_system_sparse_parflex(F1, F2, ...)`** — Stage 1 only: build cost matrix and run chunked FlexDTW (returns `C` and tiled result for Stage 2 or visualization).
  - Helpers for visualization (e.g. FlexDTW vs ParFlex paths) and conversion between chunk outputs and global coordinates.

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
| 4 | **04_Align_Benchmarks.ipynb** | Run alignment (uses Sparse Parflex / FlexDTW). |
| 5 | **05_Evaluate_Benchmarks_L.ipynb** | Evaluate alignment results (e.g. for different L). |

Execute them in Jupyter/Lab or with `jupyter nbconvert --to notebook --execute <notebook>.ipynb` as needed.

---

## Running notebook 04 from the terminal

To run **04_Align_Benchmarks.ipynb** non-interactively (e.g. on a server), use the provided script:

```bash
./run_workflow.sh
```

This runs the notebook in the background with the `mir` environment, no timeout, and logs to `run.log`. Adjust the script if you need a different environment or log path.

---

## Trying different values of L

Alignment uses the chunk length **L** defined in the pipeline. To sweep or change L:

1. Edit **`run_sparse_parflex_L_sweep_config.py`**.
2. Set **`L_VALUES`** to the chunk lengths you want (e.g. `[100, 200, 500, 1000, 2000, 4000]`).
3. Optionally set **`EXPERIMENTS_TRAIN_ROOT`** and **`FEATURES_ROOT`** to your directories (the notebook that uses this config will save outputs under `experiments_train/sparse_parflex/L_<value>/` and load features from `FEATURES_ROOT`).
4. Run your sweep by executing the notebook(s) that import this config (e.g. the alignment/evaluation pipeline that uses `run_sparse_parflex_L_sweep_config`).

This file is **configuration only**; it does not run the sweep by itself.

---

## Other details

- **Paths**: Several notebooks and `run_sparse_parflex_L_sweep_config.py` use path variables (e.g. annotation dirs, feature roots, `experiments_train`). Update these to match your data layout.
- **Dependencies**: The pipeline uses **FlexDTW**, **DTW**, **NWTW**, and **Sparse_Parflex** (via `import_ipynb` from the repo). Ensure those modules/notebooks are on the path when running 04 and 05.
- **Long runs**: 04 can be slow for large benchmarks; `run_workflow.sh` uses `ExecutePreprocessor.timeout=-1` so the kernel does not time out. Monitor progress via `run.log` when running in the background.

With the environment set up and paths adjusted, running 01 → 02 → 03 → 04 → 05 and then varying L via `run_sparse_parflex_L_sweep_config.py` should reproduce and extend the results described in the repo.
