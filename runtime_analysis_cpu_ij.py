"""
Runtime analysis for Sparflex_CPU_IJ (two stage-1 implementations) vs FlexDTW.

Mirrors the overall structure of `runtime_analysis.py`:
- loops over P, L, trial
- writes an append-only trials CSV and a grouped summary CSV
- creates per-run profiling directories

What differs:
- focuses on Sparflex_CPU_IJ stage-1 chunk timings (avg time per chunk + sum time)
- compares against FlexDTW runtime on the same cost matrix definition used here

Outputs (by default under OUTPUT_DIR):
- runtime_trials_cpu_ij.csv
- runtime_summary_cpu_ij.csv
"""

# Set single-thread limits BEFORE any NumPy / BLAS import so all backends obey them.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# Avoid Intel OMP shared-memory usage in restricted environments.
os.environ["KMP_DISABLE_SHM"] = "1"
os.environ["KMP_USE_SHM"] = "0"

from pathlib import Path
import csv
import gc
import time

import numpy as np
import pandas as pd

import FlexDTW
import Sparflex_CPU_IJ


OUTPUT_DIR = Path("/home/ijain/parflex/symphony_of_tears_features")
OUTPUT_DIR.mkdir(exist_ok=True)

# Per-run profiling directories (FlexDTW and per-chunk CSVs).
PROFILE_ROOT = OUTPUT_DIR / "results_profiling_cpu_ij"
PROFILE_ROOT.mkdir(exist_ok=True)


def compute_and_save_chroma(audio_path: str | Path) -> Path:
    import librosa as lb

    audio_path = Path(audio_path)
    y, sr = lb.core.load(str(audio_path), sr=None)
    feats = lb.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    feats = lb.util.normalize(feats, norm=2, axis=0)
    output_path = OUTPUT_DIR / f"{audio_path.stem}.npy"
    np.save(output_path, feats)
    return output_path


rec_1 = Path(
    "/home/ijain/parflex/Symphony_of_Tears_(Manookian,_Jeff)/movement_1/PMLP118441-Armenia_performance.wav"
)
rec_2 = Path(
    "/home/ijain/parflex/Symphony_of_Tears_(Manookian,_Jeff)/movement_1/PMLP118441-New_Disc_-01-_Track_01.wav"
)

# Ensure feature files exist (idempotent).
f1_path = OUTPUT_DIR / f"{rec_1.stem}.npy"
f2_path = OUTPUT_DIR / f"{rec_2.stem}.npy"
if not f1_path.exists():
    compute_and_save_chroma(rec_1)
if not f2_path.exists():
    compute_and_save_chroma(rec_2)


N_TRIALS = 3
L_VALUES = [4000]
# Shorter sweep for quick experiments; extend P_VALUES as needed for full runs.
P_VALUES = [5000,8000,9000, 10000, 50000]

RESULTS_PATH = OUTPUT_DIR / "runtime_trials_cpu_ij.csv"
SUMMARY_PATH = OUTPUT_DIR / "runtime_summary_cpu_ij.csv"

_CSV_HEADER = [
    "system",
    "P",
    "L",
    "trial",
    "runtime_sec",
    "distance",
    "n_chunks",
    "avg_chunk_runtime_sec",
    "sum_chunk_runtime_sec",
]


def _already_done(profile_dir: Path, sentinel_file: str) -> bool:
    return (profile_dir / sentinel_file).exists()


def _read_chunk_profile(profile_csv: Path) -> tuple[int, float, float]:
    """
    Return (n_chunks, avg_elapsed, sum_elapsed) from a per-chunk profile CSV.
    """
    df = pd.read_csv(profile_csv)
    if df.empty:
        return 0, float("nan"), float("nan")
    # Standard column used by both implementations' per-chunk logs.
    elapsed = pd.to_numeric(df["elapsed_seconds"], errors="coerce").dropna()
    if elapsed.empty:
        return int(len(df)), float("nan"), float("nan")
    return int(len(elapsed)), float(elapsed.mean()), float(elapsed.sum())


def _write_flexdtw_runtime(profile_dir: Path, t_start: float, elapsed: float) -> None:
    with open(profile_dir / "flexdtw_runtime.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_time", "end_time", "elapsed_seconds"])
        w.writerow([t_start, t_start + elapsed, elapsed])


steps = np.array([[1, 1], [1, 2], [2, 1]], dtype=int)
weights = np.array([1.5, 3.0, 3.0], dtype=float)
beta = 0.1

F1 = np.load(f1_path)
F2 = np.load(f2_path)
max_p = min(F1.shape[1], F2.shape[1])

_write_header = (not RESULTS_PATH.exists()) or RESULTS_PATH.stat().st_size == 0
_results_fh = open(RESULTS_PATH, "a", newline="")
_csv_writer = csv.writer(_results_fh)
if _write_header:
    _csv_writer.writerow(_CSV_HEADER)
    _results_fh.flush()


for current_p in P_VALUES:
    

    # Use the same cost definition as Sparflex_CPU_IJ's local-cost implementation:
    # cosine distance on L2-normalized chroma frames.
    F1p = F1[:, :current_p]
    F2p = F2[:, :current_p]
    F1n = FlexDTW.L2norm(F1p)
    F2n = FlexDTW.L2norm(F2p)
    C = 1.0 - (F1n.T @ F2n)

    for trial in range(1, N_TRIALS + 1):
        for l_val in L_VALUES:
            flexdtw_profile_dir = PROFILE_ROOT / f"flexdtw_P{current_p}_L{l_val}_trial{trial}"
            preslice_profile_dir = PROFILE_ROOT / f"cpuij_preslice_P{current_p}_L{l_val}_trial{trial}"
            local_profile_dir = PROFILE_ROOT / f"cpuij_localcost_P{current_p}_L{l_val}_trial{trial}"

            # -- FlexDTW (full matrix, no chunking) ----------------------------
            if _already_done(flexdtw_profile_dir, "flexdtw_runtime.csv"):
                print(f"  Skipping {flexdtw_profile_dir.name}")
            else:
                flexdtw_profile_dir.mkdir(parents=True, exist_ok=True)
                gc.collect()
                t_start = time.perf_counter()
                dist = FlexDTW.flexdtw(C, steps, weights)
                elapsed = time.perf_counter() - t_start
                _write_flexdtw_runtime(flexdtw_profile_dir, t_start, elapsed)
                _csv_writer.writerow(
                    ["FlexDTW", current_p, l_val, trial, elapsed, dist, "", "", ""]
                )
                _results_fh.flush()

            # -- Sparflex_CPU_IJ impl #1: pre-sliced C_chunk passed to worker --
            if _already_done(preslice_profile_dir, "chunk_flexdtw.csv"):
                print(f"  Skipping {preslice_profile_dir.name}")
            else:
                preslice_profile_dir.mkdir(parents=True, exist_ok=True)
                gc.collect()
                t_start = time.perf_counter()
                chunks_dict, _L_out, n_chunks_1, n_chunks_2 = Sparflex_CPU_IJ.chunk_flexdtw(
                    C,
                    L=l_val,
                    steps=steps,
                    weights=weights,
                    buffer=1,
                    profile_dir=str(preslice_profile_dir),
                )
                elapsed = time.perf_counter() - t_start
                n_chunks = int(n_chunks_1) * int(n_chunks_2)
                _, avg_chunk, sum_chunk = _read_chunk_profile(
                    preslice_profile_dir / "chunk_flexdtw.csv"
                )
                # Distance is not computed at stage-1 here; keep column for parity with other CSVs.
                _csv_writer.writerow(
                    [
                        "Sparflex_CPU_IJ_preslice_stage1",
                        current_p,
                        l_val,
                        trial,
                        elapsed,
                        "",
                        n_chunks,
                        avg_chunk,
                        sum_chunk,
                    ]
                )
                _results_fh.flush()
                _ = chunks_dict  # keep reference alive for timing consistency

            # -- Sparflex_CPU_IJ impl #2: compute local C_chunk inside worker ---
            if _already_done(local_profile_dir, "chunk_flexdtw_local_cost.csv"):
                print(f"  Skipping {local_profile_dir.name}")
            else:
                local_profile_dir.mkdir(parents=True, exist_ok=True)
                gc.collect()
                t_start = time.perf_counter()
                chunks_dict2, _L_out2, n_chunks_12, n_chunks_22 = (
                    Sparflex_CPU_IJ.chunk_flexdtw_local_cost(
                        F1p,
                        F2p,
                        L=l_val,
                        steps=steps,
                        weights=weights,
                        buffer=1,
                        profile_dir=str(local_profile_dir),
                    )
                )
                elapsed = time.perf_counter() - t_start
                n_chunks = int(n_chunks_12) * int(n_chunks_22)
                _, avg_chunk, sum_chunk = _read_chunk_profile(
                    local_profile_dir / "chunk_flexdtw_local_cost.csv"
                )
                _csv_writer.writerow(
                    [
                        "Sparflex_CPU_IJ_localcost_stage1",
                        current_p,
                        l_val,
                        trial,
                        elapsed,
                        "",
                        n_chunks,
                        avg_chunk,
                        sum_chunk,
                    ]
                )
                _results_fh.flush()
                _ = chunks_dict2

            gc.collect()

    print(f"Completed P={current_p}")

_results_fh.close()

# Re-read accumulated trials and compute grouped summary.
runtime_trials_df = pd.read_csv(RESULTS_PATH)

runtime_summary_df = (
    runtime_trials_df.groupby(["system", "P", "L"], dropna=False, as_index=False)
    .agg(
        avg_runtime_sec=("runtime_sec", "mean"),
        std_runtime_sec=("runtime_sec", "std"),
        avg_chunk_runtime_sec=("avg_chunk_runtime_sec", "mean"),
        std_chunk_runtime_sec=("avg_chunk_runtime_sec", "std"),
        avg_sum_chunk_runtime_sec=("sum_chunk_runtime_sec", "mean"),
        std_sum_chunk_runtime_sec=("sum_chunk_runtime_sec", "std"),
        n_trials=("runtime_sec", "count"),
        n_chunks=("n_chunks", "max"),
    )
    .sort_values(["P", "system", "L"], na_position="first")
)

runtime_summary_df.to_csv(SUMMARY_PATH, index=False)

print(f"Saved trial runtimes to:        {RESULTS_PATH}")
print(f"Saved average/stdev summary to: {SUMMARY_PATH}")

