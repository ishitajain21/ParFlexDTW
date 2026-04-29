# %% [markdown]
# ## Runtime Analysis
#
# Compare **Parflex** (CPU chunked pipeline, same algorithm as `Parflex.ipynb` / `Parflex.py`)
# with **Parflex_gpu** (GPU Stage-1 FlexDTW). Only **L = 6000** is exercised.
# Results append to CSV: one row per (system, P, L, trial). Dense Parflex, full-matrix
# FlexDTW, classic DTW, and sparse Parflex are omitted.

# %% [markdown]
# ## Imports

# %%
import os

# Set single-thread limits BEFORE any NumPy / BLAS import so all backends obey them.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

import Parflex
import Parflex_gpu

# %%
# Default under the repo; override with PARFLEX_RUNTIME_OUTPUT if needed.
OUTPUT_DIR = Path(
    os.environ.get(
        "PARFLEX_RUNTIME_OUTPUT",
        str(Path(__file__).resolve().parent / "runtime_results"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_ROOT = OUTPUT_DIR / "results_profiling"
PROFILE_ROOT.mkdir(exist_ok=True)

# Optional real audio → chroma; if paths are missing, synthetic features are used.
_REC1 = os.environ.get(
    "PARFLEX_REC1",
    "/home/asharma/ttmp/Flex/Final_Experiments_Parflex/Symphony_No.4,_Op.90_(Mendelssohn,_Felix)/movement_1/PMLP18979-_E3_82_A4_E3_82_BF_E3_83_AA_E3_82_A21.wav",
)
_REC2 = os.environ.get(
    "PARFLEX_REC2",
    "/home/asharma/ttmp/Flex/Final_Experiments_Parflex/Symphony_No.4,_Op.90_(Mendelssohn,_Felix)/movement_1/PMLP18979-HSO_Bosch_Mendelssohn-IV_1-1-Allegro-vivace.wav",
)


def _load_or_synthetic_features(audio_path_1: str, audio_path_2: str, min_frames: int):
    """Return (F1, F2) as (n_chroma, n_frames). Tiling to length P happens in _cost_from_features."""
    p1, p2 = Path(audio_path_1), Path(audio_path_2)
    if p1.is_file() and p2.is_file():
        import librosa as lb

        def _chroma(path: Path):
            y, sr = lb.core.load(str(path), sr=None)
            feats = lb.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
            return lb.util.normalize(feats, norm=2, axis=0)

        return _chroma(p1), _chroma(p2)

    rng = np.random.default_rng(42)
    d = 12
    return (
        rng.standard_normal((d, min_frames)).astype(np.float64),
        rng.standard_normal((d, min_frames)).astype(np.float64),
    )


# %%
N_TRIALS = 10
L_VALUES = [6000]  # only L of interest for now
RESULTS_PATH = OUTPUT_DIR / "runtime_trials.csv"
SUMMARY_PATH = OUTPUT_DIR / "runtime_summary.csv"

_CSV_HEADER = ["system", "P", "L", "trial", "runtime_sec", "distance"]

steps = np.array([[1, 1], [1, 2], [2, 1]], dtype=int)
weights = np.array([1.5, 3.0, 3.0], dtype=float)
beta = 0.1

P_VALUES = [
    50000, 70000
]


def _tile_features_to_length(F, target_len):
    """Librosa chroma layout: (n_chroma, n_frames). Tile then truncate to target_len."""
    n = F.shape[1]
    if n >= target_len:
        return F[:, :target_len]
    reps = (target_len + n - 1) // n
    tiled = np.tile(F, (1, reps))
    return tiled[:, :target_len]


def _cost_from_features(F1, F2, p):
    X = _tile_features_to_length(F1, p).T
    Y = _tile_features_to_length(F2, p).T
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0))


max_p = max(P_VALUES)
F1, F2 = _load_or_synthetic_features(_REC1, _REC2, min_frames=max_p)

_write_header = not RESULTS_PATH.exists() or RESULTS_PATH.stat().st_size == 0
_results_fh = open(RESULTS_PATH, "a", newline="")
_csv_writer = csv.writer(_results_fh)
if _write_header:
    _csv_writer.writerow(_CSV_HEADER)
    _results_fh.flush()


def _already_done(profile_dir: Path, sentinel_file: str) -> bool:
    return (profile_dir / sentinel_file).exists()


for current_p in P_VALUES:
    C = _cost_from_features(F1, F2, current_p)

    for trial in range(1, N_TRIALS + 1):
        l_val = L_VALUES[0]

        cpu_dir = PROFILE_ROOT / f"parflex_cpu_P{current_p}_L{l_val}_trial{trial}"
        gpu_dir = PROFILE_ROOT / f"parflex_gpu_P{current_p}_L{l_val}_trial{trial}"

        # --- Parflex (CPU) — same entrypoint as the notebook -----------------
        if _already_done(cpu_dir, "parflex_phases.csv"):
            print(f"  Skipping {cpu_dir.name}")
        else:
            cpu_dir.mkdir(parents=True, exist_ok=True)
            gc.collect()
            t0 = time.perf_counter()
            dist, _wp = Parflex.parflex(
                C, steps, weights, beta, l_val, profile_dir=str(cpu_dir)
            )
            elapsed = time.perf_counter() - t0
            _csv_writer.writerow(
                ["Parflex_CPU_notebook", current_p, l_val, trial, elapsed, dist]
            )
            _results_fh.flush()
            # Sentinel so reruns skip; phases for CPU are chunk_flexdtw + stage2 files.
            with open(cpu_dir / "parflex_phases.csv", "w", newline="") as _f:
                w = csv.writer(_f)
                w.writerow(["phase", "elapsed_seconds"])
                w.writerow(["total_wall", elapsed])

        # --- Parflex GPU ------------------------------------------------------
        if _already_done(gpu_dir, "parflex_end_to_end_phases.csv"):
            print(f"  Skipping {gpu_dir.name}")
        else:
            gpu_dir.mkdir(parents=True, exist_ok=True)
            gc.collect()
            t0 = time.perf_counter()
            dist, _wp = Parflex_gpu.parflex(
                C,
                steps,
                weights,
                beta,
                l_val,
                profile_dir=str(gpu_dir),
                use_gpu=True,
            )
            elapsed = time.perf_counter() - t0
            _csv_writer.writerow(
                ["Parflex_GPU", current_p, l_val, trial, elapsed, dist]
            )
            _results_fh.flush()

        gc.collect()

    print(f"Completed P={current_p}")

_results_fh.close()

# %%
runtime_trials_df = pd.read_csv(RESULTS_PATH)

runtime_summary_df = (
    runtime_trials_df.groupby(["system", "P", "L"], dropna=False, as_index=False)
    .agg(
        avg_runtime_sec=("runtime_sec", "mean"),
        std_runtime_sec=("runtime_sec", "std"),
        n_trials=("runtime_sec", "count"),
    )
    .sort_values(["P", "system", "L"], na_position="first")
)
runtime_summary_df.to_csv(SUMMARY_PATH, index=False)

print(f"Saved trial runtimes to:         {RESULTS_PATH}")
print(f"Saved average/stdev summary to: {SUMMARY_PATH}")
