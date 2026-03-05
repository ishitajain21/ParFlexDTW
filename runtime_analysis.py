# %% [markdown]
# ## Runtime Analysis
#
# Benchmark FlexDTW, Parflex (dense), and Sparse Parflex across varying matrix sizes (P)
# and chunk lengths (L).  Results are written to CSV incrementally — one row per
# (system, P, L, trial) — so partial runs are never lost and successive runs accumulate.

# %% [markdown]
# ## Imports

# %%
# Set single-thread limits BEFORE any NumPy / BLAS import so all backends obey them.
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import Parflex
import Sparse_parflex_cpu          # sparse variant with per-chunk profiling
import FlexDTW
import numpy as np
from pathlib import Path
import pandas as pd
import librosa as lb
import time
import gc
import csv

# %%
OUTPUT_DIR = Path("/home/ijain/parflex/symphony_of_tears_features")
OUTPUT_DIR.mkdir(exist_ok=True)

# Directory that will hold per-run profiling sub-directories.
PROFILE_ROOT = OUTPUT_DIR / "profiling"
PROFILE_ROOT.mkdir(exist_ok=True)

# %%
def compute_and_save_chroma(audio_path):
    audio_path = Path(audio_path)
    y, sr = lb.core.load(str(audio_path), sr=None)
    feats = lb.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    feats = lb.util.normalize(feats, norm=2, axis=0)
    output_path = OUTPUT_DIR / f"{audio_path.stem}.npy"
    np.save(output_path, feats)
    return output_path

# %%
rec_1 = Path("/home/ijain/parflex/Symphony_of_Tears_(Manookian,_Jeff)/movement_1/PMLP118441-Armenia_performance.wav")
rec_2 = Path("/home/ijain/parflex/Symphony_of_Tears_(Manookian,_Jeff)/movement_1/PMLP118441-New_Disc_-01-_Track_01.wav")
# Compute features only when the output directory is being created for the first time.
if not OUTPUT_DIR.exists():
    compute_and_save_chroma(rec_1)
    compute_and_save_chroma(rec_2)

# %%
N_TRIALS  = 10
L_VALUES  = [6000, 2000, 4000]
RESULTS_PATH = OUTPUT_DIR / "runtime_trials.csv"
SUMMARY_PATH = OUTPUT_DIR / "runtime_summary.csv"

# CSV header — written only when the file is new/empty so accumulated runs share one header.
_CSV_HEADER = ["system", "P", "L", "trial", "runtime_sec", "distance"]

# %%
steps   = np.array([[1, 1], [1, 2], [2, 1]], dtype=int)
weights = np.array([1.5, 3.0, 3.0], dtype=float)
beta    = 0.1

# %%
P_VALUES = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]

F1 = np.load(OUTPUT_DIR / "PMLP118441-Armenia_performance.npy")
F2 = np.load(OUTPUT_DIR / "PMLP118441-New_Disc_-01-_Track_01.npy")

max_p = min(F1.shape[1], F2.shape[1])

# Open the results CSV in append mode so previous runs are preserved.
# Write the header row only when the file does not yet exist or is empty.
_write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
_results_fh   = open(RESULTS_PATH, "a", newline="")
_csv_writer   = csv.writer(_results_fh)
if _write_header:
    _csv_writer.writerow(_CSV_HEADER)
    _results_fh.flush()

for current_p in P_VALUES:
    if current_p > max_p:
        print(f"Skipping P={current_p}: only {max_p} frames available")
        continue

    # Build the square PxP Euclidean cost matrix for the first current_p frames.
    X  = F1[:, :current_p].T
    Y  = F2[:, :current_p].T
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    C  = np.sqrt(np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0))

    for trial in range(1, N_TRIALS + 1):

        # ---- FlexDTW (no chunking; L=nan) --------------------------------
        gc.collect()
        t_start = time.perf_counter()
        dist    = FlexDTW.flexdtw(C, steps, weights)
        elapsed = time.perf_counter() - t_start

        _csv_writer.writerow(["FlexDTW", current_p, "", trial, elapsed, dist])
        _results_fh.flush()

        # ---- Dense Parflex and Sparse Parflex for each chunk length -------
        for l_val in L_VALUES:

            # Unique sub-directory keeps each run's profiling CSVs separate,
            # preventing overwrite when the same (P, L) is benchmarked again.
            dense_profile_dir  = PROFILE_ROOT / f"dense_P{current_p}_L{l_val}_trial{trial}"
            sparse_profile_dir = PROFILE_ROOT / f"sparse_P{current_p}_L{l_val}_trial{trial}"

            # -- Dense Parflex ----------------------------------------------
            gc.collect()
            t_start  = time.perf_counter()
            dist     = Parflex.parflex(C, steps, weights, beta, l_val,
                                       profile_dir=str(dense_profile_dir))
            elapsed  = time.perf_counter() - t_start

            _csv_writer.writerow(["ParFlexDTW_dense_serial", current_p, l_val, trial, elapsed, dist])
            _results_fh.flush()

            # -- Sparse Parflex --------------------------------------------
            gc.collect()
            t_start  = time.perf_counter()
            dist     = Sparse_parflex_cpu.parflex(C, steps, weights, beta, l_val,
                                                  profile_dir=str(sparse_profile_dir))
            elapsed  = time.perf_counter() - t_start

            _csv_writer.writerow(["ParFlexDTW_sparse_serial", current_p, l_val, trial, elapsed, dist])
            _results_fh.flush()

            gc.collect()

    print(f"Completed P={current_p}")

_results_fh.close()

# %%
# Re-read the full accumulated CSV (may include previous runs) to compute the summary.
runtime_trials_df = pd.read_csv(RESULTS_PATH)

runtime_summary_df = (
    runtime_trials_df
    .groupby(["system", "P", "L"], dropna=False, as_index=False)
    .agg(
        avg_runtime_sec=("runtime_sec", "mean"),
        std_runtime_sec=("runtime_sec", "std"),
        n_trials=("runtime_sec", "count"),
    )
    .sort_values(["P", "system", "L"], na_position="first")
)
runtime_summary_df.to_csv(SUMMARY_PATH, index=False)

print(f"Saved trial runtimes to:      {RESULTS_PATH}")
print(f"Saved average/stdev summary to: {SUMMARY_PATH}")
