"""Parflex with GPU Stage-1 FlexDTW (CuPy).

Import this module like ``Parflex`` when you want ``run_flexdtw_on_tiles(..., use_gpu=...)``.
The CPU implementation lives in ``Parflex``; this module re-exports it and overrides
Stage-1 tiling to call ``gpu_flexdtw`` when a CUDA device is available.

For ``tiled_stage1_from_features``, the cost matrix is built on the GPU and kept there
for Stage-1 DP. Stage-2 overlap costs are filled via a batched device gather (no full
``C`` host matrix). Pass an explicit NumPy ``C`` to ``run_stage2_from_tiled`` only when
you already have it (e.g. for plotting).

Environment: set ``PARFLEX_DISABLE_GPU`` to any non-empty string to force CPU FlexDTW
(``use_gpu=None`` only).
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import Parflex as _P

from gpu_flexdtw import (
    cost_matrix_from_features_gpu,
    cost_matrix_to_gpu_f32,
    flexdtw_chunk_from_global_C,
    gather_c_at_global_indices,
    is_gpu_available,
)


def _sync_gpu_device():
    """Block until pending GPU work finishes (for meaningful phase timings)."""
    try:
        import cupy as _cp

        _cp.cuda.Device().synchronize()
    except Exception:
        pass


def _cupy_nbytes(arr) -> int:
    try:
        return int(arr.nbytes)
    except Exception:
        return -1


def _write_chunk_flexdtw_summary(profile_dir: str) -> None:
    """Summarize ``chunk_flexdtw.csv`` (per-tile times) for quick Stage-1 diagnosis."""
    if not _P._profiling_enabled(profile_dir):
        return
    path = Path(profile_dir) / "chunk_flexdtw.csv"
    if not path.is_file():
        return
    elapsed_chunks = []
    slowest = (-1.0, -1, -1)
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            ci, cj = row.get("chunk_i"), row.get("chunk_j")
            if ci in (None, "warmup") or cj in (None, "warmup"):
                continue
            try:
                sec = float(row["elapsed_seconds"])
            except (KeyError, ValueError):
                continue
            elapsed_chunks.append(sec)
            i, j = int(ci), int(cj)
            if sec > slowest[0]:
                slowest = (sec, i, j)
    n = len(elapsed_chunks)
    if n == 0:
        return
    arr = np.asarray(elapsed_chunks, dtype=np.float64)
    out = Path(profile_dir) / "chunk_flexdtw_summary.csv"
    with out.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(
            [
                "n_chunks_profiled",
                "sum_elapsed_sec",
                "mean_elapsed_sec",
                "max_elapsed_sec",
                "slowest_chunk_i",
                "slowest_chunk_j",
            ]
        )
        w.writerow(
            [
                n,
                float(arr.sum()),
                float(arr.mean()),
                float(arr.max()),
                slowest[1],
                slowest[2],
            ]
        )


def _append_elapsed(acc, key, elapsed):
    acc[key] = float(acc.get(key, 0.0)) + float(elapsed)


def _write_breakdown_csv(profile_dir: str, filename: str, total_label: str, total_elapsed: float, parts):
    """Write a breakdown CSV with percentages and a residual row."""
    if not _P._profiling_enabled(profile_dir):
        return
    rows = []
    parts_sum = 0.0
    for phase, elapsed in parts:
        elapsed_f = float(elapsed)
        parts_sum += elapsed_f
        pct = (100.0 * elapsed_f / total_elapsed) if total_elapsed > 0 else 0.0
        rows.append([phase, elapsed_f, pct])
    residual = float(total_elapsed) - float(parts_sum)
    residual_pct = (100.0 * residual / total_elapsed) if total_elapsed > 0 else 0.0
    rows.append(["residual_unattributed", residual, residual_pct])
    rows.append([total_label, float(total_elapsed), 100.0])
    _P._write_profile_rows(
        profile_dir,
        filename,
        ["phase", "elapsed_seconds", "percent_of_total"],
        rows,
    )

_m = sys.modules[__name__]
for _name in dir(_P):
    if _name.startswith("_"):
        continue
    setattr(_m, _name, getattr(_P, _name))


def _build_edge_data_gpu(chunks_dict, n_chunks_1, n_chunks_2, L, C_dev):
    """Mirror ``Parflex._build_edge_data`` but fill ``edge_Cf_olap`` from ``C_dev``."""
    shape4 = (n_chunks_1, n_chunks_2, 2, L)
    edge_Df = np.full(shape4, np.inf, dtype=np.float64)
    edge_start_r = np.zeros(shape4, dtype=np.int64)
    edge_start_c = np.zeros(shape4, dtype=np.int64)
    edge_Cf_olap = np.zeros(shape4, dtype=np.float64)
    edge_lens = np.zeros((n_chunks_1, n_chunks_2, 2), dtype=np.int64)
    chunk_rows = np.zeros((n_chunks_1, n_chunks_2), dtype=np.int64)
    chunk_cols = np.zeros((n_chunks_1, n_chunks_2), dtype=np.int64)
    chunk_bounds = np.zeros((n_chunks_1, n_chunks_2, 4), dtype=np.int64)
    chunk_valid = np.zeros((n_chunks_1, n_chunks_2), dtype=np.int64)

    coords_gr = []
    coords_gc = []
    for (i, j), ch in chunks_dict.items():
        r, c = ch["shape"]
        chunk_rows[i, j] = r
        chunk_cols[i, j] = c
        s1, e1, s2, e2 = ch["bounds"]
        chunk_bounds[i, j] = [s1, e1, s2, e2]
        chunk_valid[i, j] = 1
        edge_lens[i, j, 0] = c
        edge_lens[i, j, 1] = r

        S_raw = ch["S"].astype(np.float64)
        S_safe = np.where(np.isnan(S_raw), 0.0, S_raw)

        for pos in range(c):
            lr, lc = r - 1, pos
            s_val = S_safe[lr, lc]
            if s_val > 0:
                sr, sc = 0, int(s_val)
            elif s_val < 0:
                sr, sc = int(-s_val), 0
            else:
                sr, sc = 0, 0
            coords_gr.append(s1 + sr)
            coords_gc.append(s2 + sc)

        for pos in range(r):
            lr, lc = pos, c - 1
            s_val = S_safe[lr, lc]
            if s_val > 0:
                sr, sc = 0, int(s_val)
            elif s_val < 0:
                sr, sc = int(-s_val), 0
            else:
                sr, sc = 0, 0
            coords_gr.append(s1 + sr)
            coords_gc.append(s2 + sc)

    gr = np.asarray(coords_gr, dtype=np.int32)
    gc = np.asarray(coords_gc, dtype=np.int32)
    vals = gather_c_at_global_indices(C_dev, gr, gc)

    vi = 0
    for (i, j), ch in chunks_dict.items():
        r, c = ch["shape"]
        s1, e1, s2, e2 = ch["bounds"]
        D = ch["D"].astype(np.float64)
        S_raw = ch["S"].astype(np.float64)
        S_safe = np.where(np.isnan(S_raw), 0.0, S_raw)

        for pos in range(c):
            lr, lc = r - 1, pos
            edge_Df[i, j, 0, pos] = D[lr, lc]
            s_val = S_safe[lr, lc]
            if s_val > 0:
                sr, sc = 0, int(s_val)
            elif s_val < 0:
                sr, sc = int(-s_val), 0
            else:
                sr, sc = 0, 0
            edge_start_r[i, j, 0, pos] = sr
            edge_start_c[i, j, 0, pos] = sc
            edge_Cf_olap[i, j, 0, pos] = vals[vi]
            vi += 1

        for pos in range(r):
            lr, lc = pos, c - 1
            edge_Df[i, j, 1, pos] = D[lr, lc]
            s_val = S_safe[lr, lc]
            if s_val > 0:
                sr, sc = 0, int(s_val)
            elif s_val < 0:
                sr, sc = int(-s_val), 0
            else:
                sr, sc = 0, 0
            edge_start_r[i, j, 1, pos] = sr
            edge_start_c[i, j, 1, pos] = sc
            edge_Cf_olap[i, j, 1, pos] = vals[vi]
            vi += 1

    return (
        edge_Df,
        edge_start_r,
        edge_start_c,
        edge_Cf_olap,
        edge_lens,
        chunk_rows,
        chunk_cols,
        chunk_bounds,
        chunk_valid,
    )


def propagate_tile_edge_costs(
    chunks_dict,
    L,
    num_chunks_1,
    num_chunks_2,
    buffer_param=0.1,
    profile_dir="Profiling_results",
    C_dev=None,
):
    """Same as ``Parflex.propagate_tile_edge_costs`` with optional GPU overlap-cost gather."""
    if C_dev is not None:
        edge_tuple = _build_edge_data_gpu(
            chunks_dict, num_chunks_1, num_chunks_2, L, C_dev
        )
    else:
        edge_tuple = _P._build_edge_data(
            chunks_dict, num_chunks_1, num_chunks_2, L
        )

    (
        edge_Df,
        edge_start_r,
        edge_start_c,
        edge_Cf_olap,
        edge_lens,
        chunk_rows,
        chunk_cols,
        chunk_bounds,
        chunk_valid,
    ) = edge_tuple

    valid_mask = _P._build_valid_mask(chunks_dict, num_chunks_1, num_chunks_2, L)

    D_arr = np.full((num_chunks_1, num_chunks_2, 2, L), np.inf, dtype=np.float64)
    L_arr = np.full((num_chunks_1, num_chunks_2, 2, L), np.inf, dtype=np.float64)

    t0 = _P._start_profile_timer(_P._profiling_enabled(profile_dir), collect_gc=True)

    _P._nb_initialize_chunks_sparse(
        edge_Df,
        edge_start_r,
        edge_start_c,
        edge_Cf_olap,
        edge_lens,
        chunk_rows,
        chunk_cols,
        chunk_bounds,
        chunk_valid,
        valid_mask,
        num_chunks_1,
        num_chunks_2,
        D_arr,
        L_arr,
    )

    if t0 is not None:
        t0, t1, elapsed = _P._elapsed_profile_seconds(t0)
        _P._write_profile_rows(
            profile_dir,
            "initialize_chunks.csv",
            ["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"],
            [["all", "all", t0, t1, elapsed]],
        )

    t0 = _P._start_profile_timer(_P._profiling_enabled(profile_dir), collect_gc=True)

    _P._nb_dp_fill_chunks_sparse(
        edge_Df,
        edge_start_r,
        edge_start_c,
        edge_Cf_olap,
        edge_lens,
        chunk_rows,
        chunk_cols,
        chunk_bounds,
        chunk_valid,
        valid_mask,
        num_chunks_1,
        num_chunks_2,
        D_arr,
        L_arr,
    )

    if t0 is not None:
        t0, t1, elapsed = _P._elapsed_profile_seconds(t0)
        _P._write_profile_rows(
            profile_dir,
            "dp_fill_chunks.csv",
            ["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"],
            [["all", "all", t0, t1, elapsed]],
        )

    return D_arr, L_arr, edge_lens


def run_flexdtw_on_tiles(
    C,
    L,
    steps=None,
    weights=None,
    buffer=1,
    profile_dir="Profiling_results",
    warn_slow_chunks=False,
    slow_chunk_seconds=1.0,
    use_gpu=None,
    C_dev=None,
):
    """Same as ``Parflex.run_flexdtw_on_tiles`` with optional GPU Stage-1.

    use_gpu: if True, run chunk FlexDTW on GPU when CuPy works. If False, CPU only.
        If None, use GPU unless ``PARFLEX_DISABLE_GPU`` is set.

    C may be ``None`` when ``C_dev`` supplies the on-device cost matrix (features path).
    C_dev: optional CuPy array for the full cost matrix already on device (same shape
        as ``C``). When set with ``use_gpu``, chunk DP reads from this buffer and skips
        a host-to-device copy of ``C``.
    """
    if steps is None:
        steps = [(1, 1), (1, 2), (2, 1)]
    if weights is None:
        weights = [2, 3, 3]

    if C is None:
        if C_dev is None:
            raise ValueError(
                "run_flexdtw_on_tiles: provide NumPy C and/or C_dev (both None)"
            )
        L1, L2 = int(C_dev.shape[0]), int(C_dev.shape[1])
    else:
        L1, L2 = C.shape
        if C_dev is not None and tuple(C.shape) != tuple(C_dev.shape):
            raise ValueError("C_dev shape must match C")

    prof = _P._profiling_enabled(profile_dir)
    stage1_total_start = time.perf_counter()
    stage1_acc = {}
    per_tile_rows = []

    t_setup_start = time.perf_counter()
    hop = L - 1
    n_chunks_1 = math.ceil((L1 - 1) / hop)
    n_chunks_2 = math.ceil((L2 - 1) / hop)
    chunks_dict = {}

    if use_gpu is None:
        use_gpu = os.environ.get("PARFLEX_DISABLE_GPU", "").strip() == ""

    dev_buf = None
    if use_gpu:
        try:
            if not is_gpu_available():
                use_gpu = False
            else:
                dev_buf = C_dev if C_dev is not None else cost_matrix_to_gpu_f32(C)
        except Exception:
            use_gpu = False
            dev_buf = None
    _append_elapsed(stage1_acc, "stage1_setup_and_device_prepare", time.perf_counter() - t_setup_start)

    profile_file, _pw = _P._open_profile_csv(
        profile_dir,
        "chunk_flexdtw.csv",
        ["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"],
    )

    if profile_file is not None:
        _warm_h = min(int(L), int(L1))
        _warm_w = min(int(L), int(L2))
        if _warm_h >= 4 and _warm_w >= 4:
            warm_start = _P._start_profile_timer(True, collect_gc=True)
            try:
                rng = np.random.RandomState(0)
                _wd = C.dtype if C is not None else np.float64
                C_warm = np.ascontiguousarray(
                    rng.rand(_warm_h, _warm_w).astype(_wd, copy=False)
                )
                if use_gpu and dev_buf is not None:
                    try:
                        Cw = cost_matrix_to_gpu_f32(C_warm)
                        H, W = int(C_warm.shape[0]), int(C_warm.shape[1])
                        L2w = int(C_warm.shape[1])
                        for _ in range(2):
                            flexdtw_chunk_from_global_C(
                                Cw,
                                L2w,
                                0,
                                H,
                                0,
                                W,
                                steps,
                                weights,
                                buffer=1,
                            )
                    except Exception:
                        for _ in range(2):
                            _P.FlexDTW.flexdtw(
                                C_warm, steps=steps, weights=weights, buffer=1
                            )
                else:
                    for _ in range(2):
                        _P.FlexDTW.flexdtw(
                            C_warm, steps=steps, weights=weights, buffer=1
                        )
            except ImportError:
                pass
            warm_start, warm_end, warm_elapsed = _P._elapsed_profile_seconds(
                warm_start
            )
            _pw.writerow(["warmup", "warmup", warm_start, warm_end, warm_elapsed])
            profile_file.flush()
            _append_elapsed(stage1_acc, "stage1_warmup", warm_elapsed)

    _time_chunk = profile_file is not None or warn_slow_chunks

    for i in range(n_chunks_1):
        for j in range(n_chunks_2):
            tile_total_start = time.perf_counter()
            if _time_chunk:
                t_start = _P._start_profile_timer(profile_file is not None, collect_gc=True)
                if t_start is None:
                    t_start = tile_total_start

            t_extract_start = time.perf_counter()
            start_1, start_2 = i * hop, j * hop
            end_1 = min(start_1 + L, L1)
            end_2 = min(start_2 + L, L2)
            if C is not None:
                C_chunk = C[int(start_1) : int(end_1), int(start_2) : int(end_2)]
            else:
                C_chunk = None
            shape_chunk = (end_1 - start_1, end_2 - start_2)
            t_extract_elapsed = time.perf_counter() - t_extract_start

            t_flexdtw_start = time.perf_counter()
            try:
                if use_gpu and dev_buf is not None:
                    best_cost, wp, D, P, B, debug = flexdtw_chunk_from_global_C(
                        dev_buf,
                        int(L2),
                        int(start_1),
                        int(end_1),
                        int(start_2),
                        int(end_2),
                        steps,
                        weights,
                        buffer=buffer,
                    )
                else:
                    best_cost, wp, D, P, B, debug = _P.FlexDTW.flexdtw(
                        C_chunk, steps=steps, weights=weights, buffer=buffer
                    )
            except ImportError:
                best_cost = 0
                wp = []
                rh, rw = shape_chunk
                D = np.zeros((rh, rw), dtype=np.float64)
                P = np.zeros((rh, rw), dtype=np.int32)
                B = np.zeros((rh, rw), dtype=np.int8)
                debug = {}
            if use_gpu and dev_buf is not None:
                _sync_gpu_device()
            t_flexdtw_elapsed = time.perf_counter() - t_flexdtw_start

            t_edges_start = time.perf_counter()
            actual_hop_1 = hop if end_1 < L1 else (L1 - start_1)
            actual_hop_2 = hop if end_2 < L2 else (L2 - start_2)
            starts_bot_edge, starts_left_edge = _P._edge_starts_from_S(P)
            t_edges_elapsed = time.perf_counter() - t_edges_start

            t_store_start = time.perf_counter()
            chunks_dict[(i, j)] = {
                "C": C_chunk,
                "D": D,
                "S": P,
                "B": B,
                "debug": debug,
                "best_cost": best_cost,
                "wp": wp,
                "bounds": (start_1, end_1, start_2, end_2),
                "hop": (actual_hop_1, actual_hop_2),
                "shape": shape_chunk,
                "starts_bot_edge": starts_bot_edge,
                "starts_left_edge": starts_left_edge,
            }
            t_store_elapsed = time.perf_counter() - t_store_start
            tile_total_elapsed = time.perf_counter() - tile_total_start

            _append_elapsed(stage1_acc, "tile_extract_chunk_slice", t_extract_elapsed)
            _append_elapsed(stage1_acc, "tile_flexdtw_compute", t_flexdtw_elapsed)
            _append_elapsed(stage1_acc, "tile_edge_start_extraction", t_edges_elapsed)
            _append_elapsed(stage1_acc, "tile_chunk_record_store", t_store_elapsed)
            _append_elapsed(stage1_acc, "tile_total_runtime_sum", tile_total_elapsed)
            if prof:
                per_tile_rows.append(
                    [
                        i,
                        j,
                        float(tile_total_elapsed),
                        float(t_extract_elapsed),
                        float(t_flexdtw_elapsed),
                        float(t_edges_elapsed),
                        float(t_store_elapsed),
                        int(use_gpu and dev_buf is not None),
                    ]
                )

            if _time_chunk:
                if profile_file is not None:
                    t_start, t_end, elapsed = _P._elapsed_profile_seconds(t_start)
                else:
                    t_end = time.perf_counter()
                    elapsed = t_end - t_start
                if warn_slow_chunks and elapsed > slow_chunk_seconds:
                    print(f"  Warning, chunk ({i},{j}) took {elapsed} seconds")
                if profile_file is not None:
                    _pw.writerow([i, j, t_start, t_end, elapsed])
                    profile_file.flush()

    _P._close_profile_csv(profile_file)
    stage1_total_elapsed = time.perf_counter() - stage1_total_start
    if prof:
        if per_tile_rows:
            _P._write_profile_rows(
                profile_dir,
                "stage1_tile_step_profile.csv",
                [
                    "chunk_i",
                    "chunk_j",
                    "tile_total_elapsed_seconds",
                    "extract_chunk_elapsed_seconds",
                    "flexdtw_elapsed_seconds",
                    "edge_starts_elapsed_seconds",
                    "chunk_store_elapsed_seconds",
                    "used_gpu",
                ],
                per_tile_rows,
            )
        tile_substeps_sum = (
            stage1_acc.get("tile_extract_chunk_slice", 0.0)
            + stage1_acc.get("tile_flexdtw_compute", 0.0)
            + stage1_acc.get("tile_edge_start_extraction", 0.0)
            + stage1_acc.get("tile_chunk_record_store", 0.0)
        )
        tile_total_sum = stage1_acc.get("tile_total_runtime_sum", 0.0)
        tile_other = tile_total_sum - tile_substeps_sum
        _write_breakdown_csv(
            profile_dir,
            "stage1_runtime_breakdown.csv",
            "stage1_total_runtime",
            stage1_total_elapsed,
            [
                ("stage1_setup_and_device_prepare", stage1_acc.get("stage1_setup_and_device_prepare", 0.0)),
                ("stage1_warmup", stage1_acc.get("stage1_warmup", 0.0)),
                ("tile_extract_chunk_slice", stage1_acc.get("tile_extract_chunk_slice", 0.0)),
                ("tile_flexdtw_compute", stage1_acc.get("tile_flexdtw_compute", 0.0)),
                ("tile_edge_start_extraction", stage1_acc.get("tile_edge_start_extraction", 0.0)),
                ("tile_chunk_record_store", stage1_acc.get("tile_chunk_record_store", 0.0)),
                ("tile_other_overhead", tile_other),
            ],
        )
    return chunks_dict, L, n_chunks_1, n_chunks_2


def tiled_stage1_from_features(
    F1, F2, steps=None, weights=None, beta=0.1, L=None, use_gpu=None
):
    """Like ``Parflex.tiled_stage1_from_features`` with optional GPU Stage-1."""
    if L is None:
        L = _P.DEFAULT_CHUNK_LENGTH
    if use_gpu is None:
        use_gpu = os.environ.get("PARFLEX_DISABLE_GPU", "").strip() == ""

    C_dev = None
    if use_gpu:
        try:
            if is_gpu_available():
                C_dev = cost_matrix_from_features_gpu(F1, F2)
                C = None
            else:
                use_gpu = False
        except Exception:
            C_dev = None
            use_gpu = False

    if not use_gpu or C_dev is None:
        C = 1 - _P.FlexDTW.L2norm(F1).T @ _P.FlexDTW.L2norm(F2)
        C_dev = None

    steps = steps if steps is not None else np.array([[1, 1], [1, 2], [2, 1]])
    weights = weights if weights is not None else np.array([1.25, 3.0, 3.0])
    steps_arr = np.array(steps).reshape((-1, 2))
    stage1_params = {"steps": steps_arr, "weights": np.array(weights), "buffer": 1.0}
    chunks_dict, L_out, n_chunks_1, n_chunks_2 = run_flexdtw_on_tiles(
        C,
        L=L,
        steps=steps,
        weights=weights,
        buffer=1,
        use_gpu=use_gpu,
        C_dev=C_dev,
    )
    tiled_result = _P.build_tiled_metadata(
        chunks_dict, L_out, n_chunks_1, n_chunks_2, C, stage1_params=stage1_params
    )
    tiled_result["chunks_dict"] = chunks_dict
    tiled_result["C_dev"] = C_dev
    return C, tiled_result


def parflex(
    C,
    steps,
    weights,
    beta,
    L=None,
    profile_dir="Profiling_results",
    backtrace_segments=False,
    warn_slow_chunks=False,
    slow_chunk_seconds=1.0,
    use_gpu=None,
):
    """Like ``Parflex.parflex`` with optional GPU Stage-1."""
    if L is None:
        L = _P.DEFAULT_CHUNK_LENGTH
    L1, L2 = C.shape
    buffer_global = min(L1, L2) * (
        1 - (1 - beta) * min(L1, L2) / max(L1, L2)
    )

    steps_arr = (
        np.array(steps).reshape((-1, 2))
        if hasattr(steps, "__len__")
        else np.array(steps)
    )
    weights_arr = np.array(weights)
    stage1_params = {"steps": steps_arr, "weights": weights_arr, "buffer": 1.0}

    prof = _P._profiling_enabled(profile_dir)
    t_total_start_perf = time.perf_counter()
    total_parts = []
    phase_rows = []

    C_dev = None
    if use_gpu is None:
        use_gpu = os.environ.get("PARFLEX_DISABLE_GPU", "").strip() == ""
    want_gpu = bool(use_gpu)
    if use_gpu:
        t_h2d = _P._start_profile_timer(prof, collect_gc=True)
        try:
            if is_gpu_available():
                C_dev = cost_matrix_to_gpu_f32(C)
                _sync_gpu_device()
        except Exception:
            C_dev = None
        if t_h2d is not None:
            t_h2d, t1, elapsed = _P._elapsed_profile_seconds(t_h2d)
            phase_rows.append(
                ["gpu_host_to_device_C", t_h2d, t1, elapsed, int(C_dev is not None)]
            )
            total_parts.append(("gpu_host_to_device_C", elapsed))

    t_s1 = _P._start_profile_timer(prof, collect_gc=True)
    chunks_dict, L_out, n_chunks_1, n_chunks_2 = run_flexdtw_on_tiles(
        C,
        L=L,
        steps=steps,
        weights=weights,
        buffer=1,
        profile_dir=profile_dir,
        warn_slow_chunks=warn_slow_chunks,
        slow_chunk_seconds=slow_chunk_seconds,
        use_gpu=use_gpu,
        C_dev=C_dev,
    )
    if C_dev is not None:
        _sync_gpu_device()
    if t_s1 is not None:
        t0, t1, elapsed = _P._elapsed_profile_seconds(t_s1)
        phase_rows.append(
            [
                "stage1_chunk_flexdtw_all_tiles",
                t0,
                t1,
                elapsed,
                int(C_dev is not None),
            ]
        )
        total_parts.append(("stage1_chunk_flexdtw_all_tiles", elapsed))
    _write_chunk_flexdtw_summary(profile_dir)

    t_meta = _P._start_profile_timer(prof, collect_gc=False)
    tiled_result = _P.build_tiled_metadata(
        chunks_dict, L_out, n_chunks_1, n_chunks_2, C, stage1_params=stage1_params
    )
    tiled_result["C_dev"] = C_dev
    if t_meta is not None:
        t0, t1, elapsed = _P._elapsed_profile_seconds(t_meta)
        phase_rows.append(["build_tiled_metadata", t0, t1, elapsed, int(C_dev is not None)])
        total_parts.append(("build_tiled_metadata", elapsed))

    t_pe = _P._start_profile_timer(prof, collect_gc=True)
    D_arr, L_arr, edge_lens = propagate_tile_edge_costs(
        chunks_dict,
        L_out,
        n_chunks_1,
        n_chunks_2,
        buffer_param=1,
        profile_dir=profile_dir,
        C_dev=C_dev,
    )
    if t_pe is not None:
        t0, t1, elapsed = _P._elapsed_profile_seconds(t_pe)
        phase_rows.append(
            ["propagate_tile_edge_costs_total", t0, t1, elapsed, int(C_dev is not None)]
        )
        total_parts.append(("propagate_tile_edge_costs_total", elapsed))

    t_sy = _P._start_profile_timer(prof, collect_gc=True)
    D_arr, L_arr = _P.sync_tile_overlap_edges(
        D_arr, L_arr, edge_lens, n_chunks_1, n_chunks_2
    )
    if t_sy is not None:
        t0, t1, elapsed = _P._elapsed_profile_seconds(t_sy)
        phase_rows.append(
            ["sync_tile_overlap_edges", t0, t1, elapsed, int(C_dev is not None)]
        )
        total_parts.append(("sync_tile_overlap_edges", elapsed))

    t_s2 = _P._start_profile_timer(prof, collect_gc=True)
    r = _P.stage2_scan_and_stitch(
        tiled_result,
        chunks_dict,
        D_arr,
        L_arr,
        edge_lens,
        L1,
        L2,
        L_block=L_out,
        buffer_stage2=buffer_global,
        top_k=1,
        profile_dir=profile_dir,
        backtrace_segments=backtrace_segments,
    )
    if t_s2 is not None:
        t0, t1, elapsed = _P._elapsed_profile_seconds(t_s2)
        phase_rows.append(
            ["stage2_scan_and_stitch_total", t0, t1, elapsed, int(C_dev is not None)]
        )
        total_parts.append(("stage2_scan_and_stitch_total", elapsed))

    if prof and phase_rows:
        header = [
            "phase",
            "start_time",
            "end_time",
            "elapsed_seconds",
            "had_device_C",
        ]
        _P._write_profile_rows(
            profile_dir,
            "parflex_end_to_end_phases.csv",
            header,
            phase_rows,
        )
        total_elapsed = time.perf_counter() - t_total_start_perf
        _write_breakdown_csv(
            profile_dir,
            "parflex_total_runtime_breakdown.csv",
            "parflex_total_runtime",
            total_elapsed,
            total_parts,
        )
        c_bytes = _cupy_nbytes(C_dev) if C_dev is not None else 0
        _P._write_profile_rows(
            profile_dir,
            "parflex_gpu_run_meta.csv",
            [
                "L1",
                "L2",
                "L_block",
                "want_gpu",
                "device_C_uploaded",
                "C_dev_bytes",
                "n_chunks_1",
                "n_chunks_2",
                "n_tiles",
            ],
            [
                [
                    L1,
                    L2,
                    L_out,
                    int(want_gpu),
                    int(C_dev is not None),
                    c_bytes,
                    n_chunks_1,
                    n_chunks_2,
                    n_chunks_1 * n_chunks_2,
                ]
            ],
        )

    wp = r["stitched_wp"]
    if wp.size > 0:
        wp = wp.T
    else:
        wp = np.array([[], []], dtype=np.int64)
    return r["best_cost"], wp


def run_stage2_from_tiled(
    tiled_result,
    C=None,
    beta=0.1,
    show_fig=False,
    top_k=1,
    profile_dir=None,
    backtrace_segments=False,
):
    """Like ``Parflex.run_stage2_from_tiled``; reads ``tiled_result['C_dev']`` when ``C`` is omitted."""
    chunks_dict = tiled_result["chunks_dict"]
    if C is None:
        L1, L2 = tiled_result["C_shape"]
    else:
        L1, L2 = C.shape
    L = tiled_result["L_block"]
    n_chunks_1, n_chunks_2 = tiled_result["n_row"], tiled_result["n_col"]
    C_dev = tiled_result.get("C_dev")
    buffer_global = min(L1, L2) * (
        1 - (1 - beta) * min(L1, L2) / max(L1, L2)
    )

    D_arr, L_arr, edge_lens = propagate_tile_edge_costs(
        chunks_dict,
        L,
        n_chunks_1,
        n_chunks_2,
        buffer_param=1,
        profile_dir=profile_dir,
        C_dev=C_dev,
    )
    D_arr, L_arr = _P.sync_tile_overlap_edges(
        D_arr, L_arr, edge_lens, n_chunks_1, n_chunks_2
    )

    r = _P.stage2_scan_and_stitch(
        tiled_result,
        chunks_dict,
        D_arr,
        L_arr,
        edge_lens,
        L1,
        L2,
        L_block=L,
        buffer_stage2=buffer_global,
        top_k=top_k,
        profile_dir=profile_dir,
        backtrace_segments=backtrace_segments,
    )
    if show_fig:
        _P.plot_normalized_tile_edge_costs(
            D_arr, L_arr, edge_lens, n_chunks_1, n_chunks_2
        )
    return r
