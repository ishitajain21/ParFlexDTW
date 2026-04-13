"""
Parflex_gpu.py — GPU-parallelized Stage 1 for Parflex.

Drop-in replacement for run_flexdtw_on_tiles().
Only Stage 1 (the per-chunk FlexDTW calls) is GPU-accelerated.
Stage 2 (propagate_tile_edge_costs, stage2_scan_and_stitch) is unchanged.

Requirements:
    pip install cupy-cuda12x   # or cupy-cuda11x depending on your CUDA version
    # FlexDTW must still be importable (used for fallback + warm-up reference)

Usage:
    # Before (CPU):
    chunks_dict, L, n1, n2 = run_flexdtw_on_tiles(C, L=L, steps=steps, weights=weights)

    # After (GPU):
    chunks_dict, L, n1, n2 = run_flexdtw_on_tiles_gpu(C, L=L, steps=steps, weights=weights)

    Everything downstream (build_tiled_metadata, propagate_tile_edge_costs,
    stage2_scan_and_stitch, Parflex) is unchanged.
"""

import math
import time
import csv
import gc
import os

import numpy as np

# ---------------------------------------------------------------------------
# CUDA kernel source
# ---------------------------------------------------------------------------
# This implements the same DP as FlexDTW but as a CUDA kernel.
# Each thread block handles one (chunk_i, chunk_j) pair.
# Threads within a block cooperate via shared memory along each anti-diagonal.
#
# The DP recurrence (standard FlexDTW steps [(1,1),(1,2),(2,1)]):
#   D[r,c] = C[r,c] + min(
#       w0 * D[r-1, c-1],    # step (1,1)
#       w1 * D[r-1, c-2],    # step (1,2) -- skips one column
#       w2 * D[r-2, c-1],    # step (2,1) -- skips one row
#   )
#
# Anti-diagonal parallelism: cells on the same anti-diagonal k = r+c are
# independent (no cell depends on another cell with the same r+c sum).
# We launch one thread per cell on each diagonal within a block.
#
# Memory layout (all flat, row-major):
#   C_all   : (n_chunks, rows, cols) -- all chunk cost matrices stacked
#   D_all   : (n_chunks, rows, cols) -- accumulated costs (output)
#   P_all   : (n_chunks, rows, cols) -- path start encoding S (output)
#   B_all   : (n_chunks, rows, cols) -- backtrace step indices (output)
#   shapes  : (n_chunks, 2)          -- (rows, cols) per chunk
#   offsets : (n_chunks,)            -- flat offset into C/D/P/B for chunk k
#   weights : (3,)                   -- DTW weights [w0, w1, w2]

_FLEXDTW_KERNEL_SOURCE = r"""

// Encode the path start for cell (r, c):
//   S > 0  => path started at (0, S), i.e. bottom edge, column S
//   S < 0  => path started at (-S, 0), i.e. left edge, row -S
//   S == 0 => path started at (0, 0)
__device__ inline float encode_start(int start_r, int start_c) {
    if (start_r == 0 && start_c == 0) return 0.0f;
    if (start_r == 0) return (float)start_c;
    return -(float)start_r;
}

// Decode start from S value
__device__ inline void decode_start(float s, int *sr, int *sc) {
    if (s > 0.0f)       { *sr = 0;         *sc = (int)s;  }
    else if (s < 0.0f)  { *sr = (int)(-s); *sc = 0;       }
    else                { *sr = 0;         *sc = 0;        }
}

extern "C" __global__
void flexdtw_chunks_kernel(
    const float* __restrict__ C_all,
    float*       __restrict__ D_all,
    float*       __restrict__ P_all,
    short*       __restrict__ B_all,
    const int*   __restrict__ shapes,
    const long*  __restrict__ offsets,
    const float* __restrict__ weights,
    int n_chunks
) {
    int chunk_idx = blockIdx.x;
    if (chunk_idx >= n_chunks) return;

    int nrow = shapes[chunk_idx * 2 + 0];
    int ncol = shapes[chunk_idx * 2 + 1];
    long off = offsets[chunk_idx];

    const float* C = C_all + off;
    float*       D = D_all + off;
    float*       P = P_all + off;
    short*       B = B_all + off;

    float w0 = weights[0], w1 = weights[1], w2 = weights[2];

    int n_diag = nrow + ncol - 1;

    for (int k = 0; k < n_diag; k++) {
        int r_min = max(0, k - (ncol - 1));
        int r_max = min(k, nrow - 1);
        int diag_len = r_max - r_min + 1;

        for (int t = threadIdx.x; t < diag_len; t += blockDim.x) {
            int r = r_min + t;
            int c = k - r;

            float cost = C[r * ncol + c];

            if (r == 0 && c == 0) {
                D[0] = cost;
                P[0] = encode_start(0, 0);
                B[0] = -1;
                continue;
            }

            float best_d = 3.402823466e+38F;
            int   best_step = -1;
            int   best_sr = r, best_sc = c;

            // Step (1,1): from (r-1, c-1)
            if (r >= 1 && c >= 1) {
                float prev = D[(r-1)*ncol + (c-1)];
                if (prev < 1.701411733e+38F) {
                    float cand = w0 * prev + cost;
                    if (cand < best_d) {
                        best_d = cand;
                        best_step = 0;
                        float ps = P[(r-1)*ncol + (c-1)];
                        decode_start(ps, &best_sr, &best_sc);
                    }
                }
            }

            // Step (1,2): from (r-1, c-2)
            if (r >= 1 && c >= 2) {
                float prev = D[(r-1)*ncol + (c-2)];
                if (prev < 1.701411733e+38F) {
                    float cand = w1 * prev + cost;
                    if (cand < best_d) {
                        best_d = cand;
                        best_step = 1;
                        float ps = P[(r-1)*ncol + (c-2)];
                        decode_start(ps, &best_sr, &best_sc);
                    }
                }
            }

            // Step (2,1): from (r-2, c-1)
            if (r >= 2 && c >= 1) {
                float prev = D[(r-2)*ncol + (c-1)];
                if (prev < 1.701411733e+38F) {
                    float cand = w2 * prev + cost;
                    if (cand < best_d) {
                        best_d = cand;
                        best_step = 2;
                        float ps = P[(r-2)*ncol + (c-1)];
                        decode_start(ps, &best_sr, &best_sc);
                    }
                }
            }

            if (best_step == -1 || best_d >= 1.701411733e+38F) {
                best_d    = cost;
                best_step = -1;
                best_sr   = r;
                best_sc   = c;
            }

            D[r * ncol + c] = best_d;
            P[r * ncol + c] = encode_start(best_sr, best_sc);
            B[r * ncol + c] = (short)best_step;
        }

        __syncthreads();
    }
}
"""

_gpu_module = None
_gpu_kernel = None
_gpu_compile_attempted = False
_gpu_compile_error = None

def _ensure_kernel_compiled():
    global _gpu_module, _gpu_kernel, _gpu_compile_attempted, _gpu_compile_error
    if _gpu_kernel is not None:
        return True
    if _gpu_compile_attempted:
        if _gpu_compile_error is not None:
            print(f"[Parflex_gpu] GPU kernel compile failed: {_gpu_compile_error}")
        return False

    _gpu_compile_attempted = True
    try:
        import cupy as cp
        # Prefer NVRTC so CUDA_PATH/nvcc discovery is not required.
        _gpu_module = cp.RawModule(code=_FLEXDTW_KERNEL_SOURCE, backend="nvrtc")
        _gpu_kernel = _gpu_module.get_function("flexdtw_chunks_kernel")
        _gpu_compile_error = None
        return True
    except Exception as nvrtc_error:
        try:
            import cupy as cp
            # Fallback for environments where NVRTC is unavailable.
            _gpu_module = cp.RawModule(code=_FLEXDTW_KERNEL_SOURCE, backend="nvcc")
            _gpu_kernel = _gpu_module.get_function("flexdtw_chunks_kernel")
            _gpu_compile_error = None
            return True
        except Exception as nvcc_error:
            _gpu_compile_error = f"NVRTC: {nvrtc_error}; NVCC: {nvcc_error}"
            print(f"[Parflex_gpu] GPU kernel compile failed: {_gpu_compile_error}")
            return False


def _edge_starts_from_S(S):
    rows, cols = S.shape
    starts_bottom_edge = set()
    starts_left_edge   = set()
    for c in range(cols):
        val = S[rows - 1, c]
        if val > 0:   starts_bottom_edge.add(int(val))
        else:         starts_left_edge.add(abs(int(val)))
    for r in range(rows):
        val = S[r, cols - 1]
        if val > 0:   starts_bottom_edge.add(int(val))
        else:         starts_left_edge.add(abs(int(val)))
    return starts_bottom_edge, starts_left_edge


def _gpu_best_cost_and_wp(D, P, B, steps):
    nrow, ncol = D.shape
    best_cost = np.inf
    best_end  = (nrow - 1, ncol - 1)

    for c in range(ncol):
        if D[nrow - 1, c] < best_cost:
            best_cost = D[nrow - 1, c]
            best_end  = (nrow - 1, c)
    for r in range(nrow - 1):
        if D[r, ncol - 1] < best_cost:
            best_cost = D[r, ncol - 1]
            best_end  = (r, ncol - 1)

    r, c = best_end
    wp   = []
    steps_arr = np.array(steps)
    max_iter  = nrow + ncol
    for _ in range(max_iter):
        wp.append([r, c])
        if r == 0 and c == 0:
            break
        step_idx = int(B[r, c])
        if step_idx < 0 or step_idx >= len(steps_arr):
            break
        dr, dc = steps_arr[step_idx]
        r -= dr; c -= dc
        if r < 0 or c < 0:
            break
    wp.reverse()
    return float(best_cost), wp


def run_flexdtw_on_tiles_gpu(
    C,
    L,
    steps=None,
    weights=None,
    buffer=1,
    batch_size=64,
    threads_per_block=64,
    profile_dir='Profiling_results',
    warn_slow_chunks=False,
    slow_chunk_seconds=1.0,
    fallback_to_cpu=True,
):
    """
    GPU-parallelized drop-in replacement for run_flexdtw_on_tiles().

    Parameters
    ----------
    batch_size        : chunks per GPU batch. Tune to VRAM.
                        Rule of thumb: batch_size * L * L * 4 bytes < 0.5 * VRAM.
                        e.g. L=4000, batch=8 -> 8 * 4000 * 4000 * 4 = 512 MB.
    threads_per_block : CUDA threads cooperating on each chunk's diagonals.
                        64-256 works well. Must be a multiple of 32 (warp size).
    """
    if steps   is None: steps   = [(1, 1), (1, 2), (2, 1)]
    if weights is None: weights = [2, 3, 3]

    gpu_ok = _ensure_kernel_compiled()
    if not gpu_ok:
        if fallback_to_cpu:
            from Parflex import run_flexdtw_on_tiles as _cpu_fn
            return _cpu_fn(C, L, steps=steps, weights=weights, buffer=buffer,
                           profile_dir=profile_dir)
        raise RuntimeError("GPU unavailable and fallback_to_cpu=False")

    import cupy as cp

    L1, L2 = C.shape
    hop = L - 1
    n1  = math.ceil((L1 - 1) / hop)
    n2  = math.ceil((L2 - 1) / hop)

    chunk_specs = []
    for i in range(n1):
        for j in range(n2):
            s1 = i * hop;  e1 = min(s1 + L, L1)
            s2 = j * hop;  e2 = min(s2 + L, L2)
            chunk_specs.append((i, j, s1, e1, s2, e2))

    C_f32 = np.ascontiguousarray(C, dtype=np.float32)
    weights_np = np.array(weights, dtype=np.float32)
    profile_rows = []
    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)

    chunks_dict = {}
    total_t0 = time.perf_counter()

    for batch_start in range(0, len(chunk_specs), batch_size):
        batch = chunk_specs[batch_start : batch_start + batch_size]
        nb    = len(batch)

        shapes_np  = np.zeros((nb, 2), dtype=np.int32)
        offsets_np = np.zeros(nb,      dtype=np.int64)

        total_cells = 0
        for k, (i, j, s1, e1, s2, e2) in enumerate(batch):
            r = e1 - s1; c = e2 - s2
            shapes_np[k]  = [r, c]
            offsets_np[k] = total_cells
            total_cells  += r * c

        # Pinned host memory: avoids OS-level copy during DMA transfer
        C_packed_pinned = cp.cuda.alloc_pinned_memory(total_cells * 4)
        C_packed = np.frombuffer(C_packed_pinned, dtype=np.float32, count=total_cells)

        for k, (i, j, s1, e1, s2, e2) in enumerate(batch):
            r = e1 - s1; c = e2 - s2
            off = int(offsets_np[k])
            C_packed[off : off + r * c] = C_f32[s1:e1, s2:e2].ravel()

        t_h2d_0 = time.perf_counter()
        C_gpu      = cp.asarray(C_packed)
        D_gpu      = cp.full(total_cells, np.inf, dtype=cp.float32)
        P_gpu      = cp.zeros(total_cells, dtype=cp.float32)
        B_gpu      = cp.full(total_cells, -1, dtype=cp.int16)
        shapes_gpu  = cp.asarray(shapes_np.ravel())
        offsets_gpu = cp.asarray(offsets_np)
        weights_gpu = cp.asarray(weights_np)
        t_h2d_1 = time.perf_counter()

        t_kern_0 = time.perf_counter()
        _gpu_kernel(
            grid=(nb,),
            block=(threads_per_block,),
            args=(C_gpu, D_gpu, P_gpu, B_gpu,
                  shapes_gpu, offsets_gpu, weights_gpu,
                  np.int32(nb)),
        )
        cp.cuda.Stream.null.synchronize()
        t_kern_1 = time.perf_counter()

        t_d2h_0 = time.perf_counter()
        D_cpu = cp.asnumpy(D_gpu)
        P_cpu = cp.asnumpy(P_gpu)
        B_cpu = cp.asnumpy(B_gpu)
        t_d2h_1 = time.perf_counter()

        del C_gpu, D_gpu, P_gpu, B_gpu, shapes_gpu, offsets_gpu, weights_gpu
        cp.get_default_memory_pool().free_all_blocks()

        for k, (i, j, s1, e1, s2, e2) in enumerate(batch):
            r = e1 - s1; c = e2 - s2
            off = int(offsets_np[k])
            cells = r * c

            D_chunk = D_cpu[off : off + cells].reshape(r, c).astype(np.float64)
            P_chunk = P_cpu[off : off + cells].reshape(r, c).astype(np.float64)
            B_chunk = B_cpu[off : off + cells].reshape(r, c).astype(np.int32)
            C_chunk = C_f32[s1:e1, s2:e2].astype(np.float64)

            best_cost, wp = _gpu_best_cost_and_wp(D_chunk, P_chunk, B_chunk, steps)
            actual_hop_1  = hop if e1 < L1 else (L1 - s1)
            actual_hop_2  = hop if e2 < L2 else (L2 - s2)
            starts_bot, starts_left = _edge_starts_from_S(P_chunk)

            chunks_dict[(i, j)] = {
                'C': C_chunk, 'D': D_chunk, 'S': P_chunk, 'B': B_chunk,
                'debug': {}, 'best_cost': best_cost, 'wp': wp,
                'bounds': (s1, e1, s2, e2),
                'hop':    (actual_hop_1, actual_hop_2),
                'shape':  (r, c),
                'starts_bot_edge':  starts_bot,
                'starts_left_edge': starts_left,
            }

        if profile_dir is not None:
            profile_rows.append({
                'batch_start': batch_start, 'batch_end': batch_start + nb, 'n_chunks': nb,
                'h2d_elapsed': t_h2d_1 - t_h2d_0,
                'kern_elapsed': t_kern_1 - t_kern_0,
                'd2h_elapsed': t_d2h_1 - t_d2h_0,
            })

    total_t1 = time.perf_counter()

    if profile_dir is not None:
        with open(os.path.join(profile_dir, "chunk_flexdtw_gpu.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["batch_start", "batch_end", "n_chunks",
                        "h2d_elapsed", "kern_elapsed", "d2h_elapsed"])
            for row in profile_rows:
                w.writerow([row['batch_start'], row['batch_end'], row['n_chunks'],
                            f"{row['h2d_elapsed']:.6f}", f"{row['kern_elapsed']:.6f}",
                            f"{row['d2h_elapsed']:.6f}"])

    return chunks_dict, L, n1, n2


def parflex_gpu(C, steps, weights, beta, L=None,
                batch_size=64, threads_per_block=64,
                profile_dir='Profiling_results',
                backtrace_segments=False):
    """Full Parflex pipeline with GPU-accelerated Stage 1. API-identical to Parflex()."""
    from Parflex import (
        build_tiled_metadata, propagate_tile_edge_costs,
        sync_tile_overlap_edges, stage2_scan_and_stitch, DEFAULT_CHUNK_LENGTH,
    )

    if L is None: L = DEFAULT_CHUNK_LENGTH
    L1, L2 = C.shape
    buffer_global = min(L1, L2) * (1 - (1 - beta) * min(L1, L2) / max(L1, L2))
    steps_arr   = np.array(steps).reshape((-1, 2))
    weights_arr = np.array(weights)
    stage1_params = {'steps': steps_arr, 'weights': weights_arr, 'buffer': 1.0}

    chunks_dict, L_out, n1, n2 = run_flexdtw_on_tiles_gpu(
        C, L=L, steps=steps, weights=weights, buffer=1,
        batch_size=batch_size, threads_per_block=threads_per_block,
        profile_dir=profile_dir,
    )

    tiled_result = build_tiled_metadata(chunks_dict, L_out, n1, n2, C,
                                        stage1_params=stage1_params)
    D_arr, L_arr, edge_lens = propagate_tile_edge_costs(
        chunks_dict, L_out, n1, n2, buffer_param=1, profile_dir=profile_dir)
    D_arr, L_arr = sync_tile_overlap_edges(D_arr, L_arr, edge_lens, n1, n2)
    r = stage2_scan_and_stitch(
        tiled_result, chunks_dict, D_arr, L_arr, edge_lens, L1, L2,
        L_block=L, buffer_stage2=buffer_global, top_k=1,
        profile_dir=profile_dir, backtrace_segments=backtrace_segments,
    )
    wp = r["stitched_wp"]
    wp = wp.T if wp.size > 0 else np.array([[], []], dtype=np.int64)
    return r["best_cost"], wp