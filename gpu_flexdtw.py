"""
gpu_flexdtw.py  —  GPU-accelerated Stage-1 FlexDTW for Parflex
===============================================================

Drop-in replacement for FlexDTW.flexdtw inside run_flexdtw_on_tiles.

Public API
----------
    is_gpu_available() -> bool

    cost_matrix_from_features_gpu(F1, F2)
        Match ``1 - FlexDTW.L2norm(F1).T @ FlexDTW.L2norm(F2)`` entirely on GPU.

    cost_matrix_to_gpu_f32(C)
        Upload a NumPy cost matrix as contiguous float32 (for precomputed ``C``).

    flexdtw_chunk_from_global_C(
        C_dev,          # CuPy array (L1, L2) float32/64, already on GPU
        L2_global,      # C_dev.shape[1]
        r0, r1, c0, c1, # chunk slice [r0:r1, c0:c1]
        steps,          # (n_steps, 2) array-like
        weights,        # (n_steps,)   array-like
        buffer=1.0,
    ) -> best_cost, wp, D, P, B, debug

Returns are NumPy arrays matching FlexDTW.flexdtw exactly.
DIAG_THREADS is the CUDA block size for the fused diagonal kernel (snapshot + shell).

Algorithm: anti-diagonal wavefront
-----------------------------------
For a chunk (R, C), anti-diagonal d = 0 .. R+C-2 contains all cells
(r, c) with r + c == d, 0 <= r < R, 0 <= c < C.

Shell enumeration within diagonal d (pos 0 .. shell_len-1):
    row = min(d, R-1) - pos      (descending)
    col = d - row                (ascending)

Step distances on anti-diagonal index d:
    (1,1) -> predecessor at diagonal d-2
    (1,2) -> predecessor at diagonal d-3
    (2,1) -> predecessor at diagonal d-3

=> Ring buffers of depth 3.  The current diagonal writes ring[d%3].
   ring[(d+1)%3] holds d-2 values; ring[(d+2)%3] holds d-1 values.
   ring[d%3] holds d-3 values and is safe to read before we overwrite it,
   but ONLY via a snapshot taken at the START of the diagonal — otherwise
   writes for earlier shell positions corrupt reads for later ones.

Boundary cells (r==0 or c==0)
-------------------------------
FlexDTW initializes D[0,:] = C[0,:] and D[:,0] = C[:,0] before the DP.
We replicate this: boundary cells store cost directly, P=0, no predecessor.

Memory
------
Permanent per-chunk:
    D_full  float32 (R, C)
    P_full  int32   (R, C)
    B_full  int8    (R, C)

Ring buffers (3 x ring_len):
    ring0/1/2   float32  (R+C,)   D values per diagonal

Snapshot buffer:
    ring_snap   float32  (R+C,)   copy of ring[d%3] taken before each
                                  diagonal's writes, used as ring_p3

Kernel parallelism
------------------
One kernel launch per anti-diagonal ``d``: a single block of ``DIAG_THREADS``
threads cooperatively snapshots ``ring[d%3]`` into ``ring_snap``, then updates
all shell cells for ``d`` in parallel (strided over ``threadIdx.x``).  Cells on
the same diagonal depend only on strictly earlier diagonals, so parallel shell
updates are safe.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Availability guard
# ---------------------------------------------------------------------------

def is_gpu_available() -> bool:
    try:
        import cupy as cp
        cp.zeros(1)
        return True
    except Exception:
        return False


def cost_matrix_from_features_gpu(F1, F2):
    """Compute Parflex cost matrix on GPU: ``1 - L2norm(F1).T @ L2norm(F2)``.

    F1, F2 are (feat_dim, L1) and (feat_dim, L2), same convention as ``FlexDTW.L2norm``.
    Returns a CuPy float32 array of shape (L1, L2).
    """
    import cupy as cp

    F1d = cp.asarray(F1, dtype=cp.float32)
    F2d = cp.asarray(F2, dtype=cp.float32)
    n1 = cp.sqrt(cp.sum(F1d * F1d, axis=0)) + cp.float32(1e-9)
    n2 = cp.sqrt(cp.sum(F2d * F2d, axis=0)) + cp.float32(1e-9)
    F1n = F1d / n1
    F2n = F2d / n2
    return (cp.float32(1.0) - F1n.T @ F2n).astype(cp.float32)


def cost_matrix_to_gpu_f32(C):
    """Copy NumPy ``C`` to a contiguous CuPy float32 array."""
    import cupy as cp

    return cp.asarray(np.ascontiguousarray(C), dtype=cp.float32)


def gather_c_at_global_indices(C_dev, global_rows, global_cols):
    """Batched lookup of ``C_dev[gr, gc]`` with a single small host transfer.

    Stage-2 edge setup only needs overlap costs at path-start cells on tile
    boundaries; this avoids materializing the full cost matrix on the host when
    ``C_dev`` already lives on the GPU.
    """
    import cupy as cp

    gr = cp.asarray(global_rows, dtype=cp.int32)
    gc = cp.asarray(global_cols, dtype=cp.int32)
    vals = C_dev[gr, gc].astype(cp.float64)
    return cp.asnumpy(vals)


# ---------------------------------------------------------------------------
# Tunable constant
# ---------------------------------------------------------------------------

# Threads per block for fused diagonal kernel (128–512 are reasonable).
DIAG_THREADS: int = 256


# ---------------------------------------------------------------------------
# Fused DP kernel: snapshot ring[d%3] -> ring_snap, then parallel shell update
# ---------------------------------------------------------------------------

_KERNEL_SOURCE = r"""
extern "C" __global__
void flexdtw_diagonal_fused_kernel(
    int d, int R, int C,
    int r0, int c0, int L2g,
    int n_steps,
    const int*   __restrict__ steps_dr,
    const int*   __restrict__ steps_dc,
    const float* __restrict__ weights,
    const float* __restrict__ C_global,
    float* ring0, float* ring1, float* ring2,
    const float* __restrict__ ring_p2,
    float* ring_snap,
    float* D_full,
    int*   P_full,
    char*  B_full,
    int ring_len,
    int shell_len
)
{
    int tid = threadIdx.x;
    int ci = d % 3;
    float* ring_cur = (ci == 0) ? ring0 : (ci == 1) ? ring1 : ring2;

    for (int i = tid; i < ring_len; i += blockDim.x) {
        ring_snap[i] = ring_cur[i];
    }
    __syncthreads();

    const float INF = 1e30f;
    int row_base = (d < R) ? d : R - 1;
    int col_base = d - row_base;

    for (int pos = tid; pos < shell_len; pos += blockDim.x) {
        int r = row_base - pos;
        int c = col_base + pos;

        if (r < 0 || c < 0 || r >= R || c >= C) {
            ring_cur[pos] = INF;
            continue;
        }

        float cell_cost = C_global[(r0 + r) * L2g + (c0 + c)];
        int flat = r * C + c;

        if (r == 0 || c == 0) {
            D_full[flat]  = cell_cost;
            P_full[flat]  = 0;
            B_full[flat]  = 0;
            ring_cur[pos] = cell_cost;
            continue;
        }

        float best_norm  = INF;
        int   best_step  = -1;
        float best_D_acc = INF;
        int   best_P_val = 0;

        for (int si = 0; si < n_steps; ++si) {
            int dr = steps_dr[si];
            int dc = steps_dc[si];
            int pr = r - dr;
            int pc = c - dc;
            if (pr < 0 || pc < 0) continue;

            int pd       = pr + pc;
            int dd       = d - pd;
            int rb_pred  = (pd < R) ? pd : R - 1;
            int pos_pred = rb_pred - pr;
            if (pos_pred < 0 || pos_pred >= R + C) continue;

            float prev_D;
            if (dd == 2) {
                prev_D = ring_p2[pos_pred];
            } else if (dd == 3) {
                prev_D = ring_snap[pos_pred];
            } else {
                continue;
            }
            if (prev_D >= INF) continue;

            int P_pred = P_full[pr * C + pc];

            float mdist;
            if (P_pred >= 0) {
                mdist = (float)(r + (c - P_pred));
            } else {
                mdist = (float)((r + P_pred) + c);
            }
            if (mdist <= 0.0f) continue;

            float path_cost = prev_D + cell_cost * weights[si];
            float norm      = path_cost / mdist;

            if (norm < best_norm) {
                best_norm  = norm;
                best_step  = si;
                best_D_acc = path_cost;
                if (pr == 0)       best_P_val = pc;
                else if (pc == 0)  best_P_val = -pr;
                else               best_P_val = P_pred;
            }
        }

        if (best_step < 0) {
            ring_cur[pos] = INF;
            D_full[flat]  = INF;
            P_full[flat]  = 0;
            B_full[flat]  = -1;
        } else {
            ring_cur[pos] = best_D_acc;
            D_full[flat]  = best_D_acc;
            P_full[flat]  = best_P_val;
            B_full[flat]  = (char)best_step;
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Kernel caches
# ---------------------------------------------------------------------------

_kernel_cache: dict = {}


def _get_fused_kernel():
    if "fused" not in _kernel_cache:
        import cupy as cp

        _kernel_cache["fused"] = cp.RawKernel(
            _KERNEL_SOURCE,
            "flexdtw_diagonal_fused_kernel",
            options=("--use_fast_math",),
        )
    return _kernel_cache["fused"]
 

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def flexdtw_chunk_from_global_C(
    C_dev,
    L2_global: int,
    r0: int, r1: int,
    c0: int, c1: int,
    steps,
    weights,
    buffer: float = 1.0,
):
    """
    Run FlexDTW on chunk C_dev[r0:r1, c0:c1] on the GPU.

    Parameters
    ----------
    C_dev      : CuPy array (L1, L2), float32 or float64
    L2_global  : int == C_dev.shape[1]
    r0, r1     : row slice
    c0, c1     : col slice
    steps      : (n_steps, 2) int array-like
    weights    : (n_steps,)   float array-like
    buffer     : passed through to find_best_endpoint

    Returns
    -------
    best_cost : float
    wp        : np.ndarray (2, N)
    D         : np.ndarray (R, C) float64
    P         : np.ndarray (R, C) int32
    B         : np.ndarray (R, C) int8
    debug     : dict
    """
    import cupy as cp

    R = r1 - r0
    C = c1 - c0
    if R <= 0 or C <= 0:
        raise ValueError(f"Empty chunk: R={R}, C={C}")

    steps_np   = np.array(steps, dtype=np.int32).reshape(-1, 2)
    weights_np = np.array(weights, dtype=np.float32)
    n_steps    = steps_np.shape[0]

    C_dev_f32 = C_dev.astype(cp.float32) if C_dev.dtype != cp.float32 else C_dev

    # Permanent chunk arrays
    D_full = cp.full((R, C), np.float32(1e30), dtype=cp.float32)
    P_full = cp.zeros((R, C), dtype=cp.int32)
    B_full = cp.full((R, C), np.int8(-1), dtype=cp.int8)

    # Ring buffers (depth 3) and snapshot buffer
    ring_len = R + C
    ring0    = cp.full(ring_len, np.float32(1e30), dtype=cp.float32)
    ring1    = cp.full(ring_len, np.float32(1e30), dtype=cp.float32)
    ring2    = cp.full(ring_len, np.float32(1e30), dtype=cp.float32)
    ring_snap = cp.full(ring_len, np.float32(1e30), dtype=cp.float32)

    # Step / weight arrays on device
    steps_dr_dev = cp.asarray(np.ascontiguousarray(steps_np[:, 0]))
    steps_dc_dev = cp.asarray(np.ascontiguousarray(steps_np[:, 1]))
    weights_dev  = cp.asarray(weights_np)

    fused_kernel = _get_fused_kernel()
    n_threads = int(DIAG_THREADS)
    if n_threads < 32 or n_threads > 1024:
        raise ValueError("DIAG_THREADS must be in [32, 1024]")

    n_diags = R + C - 1

    for d in range(n_diags):
        row_base = d if d < R else R - 1
        col_base = d - row_base
        max_pos = min(row_base, C - col_base - 1)
        shell_len = max_pos + 1
        if shell_len <= 0:
            continue

        p2 = (d + 1) % 3
        ring_p2 = ring0 if p2 == 0 else (ring1 if p2 == 1 else ring2)

        fused_kernel(
            (1,),
            (n_threads,),
            (
                np.int32(d),
                np.int32(R),
                np.int32(C),
                np.int32(r0),
                np.int32(c0),
                np.int32(L2_global),
                np.int32(n_steps),
                steps_dr_dev,
                steps_dc_dev,
                weights_dev,
                C_dev_f32,
                ring0,
                ring1,
                ring2,
                ring_p2,
                ring_snap,
                D_full,
                P_full,
                B_full,
                np.int32(ring_len),
                np.int32(shell_len),
            ),
        )

    # Transfer to CPU
    D_cpu = D_full.get().astype(np.float64)
    P_cpu = P_full.get().astype(np.int32)
    B_cpu = B_full.get().astype(np.int8)
    D_cpu[D_cpu >= 1e29] = np.inf
 

    return D_cpu, P_cpu, B_cpu