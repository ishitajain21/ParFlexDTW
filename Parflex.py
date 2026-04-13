#!/usr/bin/env python
# coding: utf-8

# # Sparse Parallel FlexDTW
# 
# Chunked FlexDTW, sparse edge costs, Stage 2 scan/backtrace (Numba JIT for edge DP, sync, and global-edge scan). Override `DEFAULT_CHUNK_LENGTH` in the next cell or pass `L=` to functions.
# 
# **API:** `tiled_stage1_from_features` → `run_stage2_from_tiled`, or `parflex(C, steps, weights, beta)`. Stage 2 returns `D_arr`, `L_arr`, and `edge_lens` (compact arrays); `plot_alignment_with_tile_background` uses these when present. `plot_alignment_with_tile_background` needs `eval_tools` for beat-pair paths.
# 

# In[ ]:


# Sparse parallel FlexDTW — tiling, edge DP, Stage 2 backtrace (Numba-optimized)
DEFAULT_CHUNK_LENGTH = 6000

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import gc
import math
import time

import FlexDTW
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from numba import njit as _numba_njit


def _parflex_njit(f):
    """Wrap helper kernels with the fast Numba profile.
    
    Use this for small numeric utilities where speed matters and strict IEEE edge
    behavior is not critical. If you hit numerical instability, switch that kernel
    to _parflex_dp_njit instead.
    """
    return _numba_njit(cache=False, fastmath=True)(f)


def _parflex_dp_njit(f):
    """Wrap DP kernels with conservative Numba settings.
    
    This keeps inf/nan semantics predictable, which the sparse DP logic depends on.
    Change this only if you are validating numerical behavior end-to-end.
    """
    return _numba_njit(cache=False, fastmath=False)(f)


def _profiling_enabled(profile_dir):
    """Return whether profiling CSV output should be emitted.
    
    Set profile_dir to None to disable all profiling writes quickly.
    """
    return profile_dir is not None


def _open_profile_csv(profile_dir, filename, header):
    """Open a profiling CSV and write its header row.
    
    This is the canonical place to change profile file layout or add new columns.
    """
    if not _profiling_enabled(profile_dir):
        return None, None
    os.makedirs(profile_dir, exist_ok=True)
    fp = open(os.path.join(profile_dir, filename), "w", newline="")
    writer = csv.writer(fp)
    writer.writerow(header)
    fp.flush()
    return fp, writer


def _close_profile_csv(profile_fp):
    """Close a profiling file handle if one was opened.
    
    Kept as a helper so call sites can stay clean and uniform.
    """
    if profile_fp is not None:
        profile_fp.close()


def _start_profile_timer(enabled, collect_gc=False):
    """Start a wall-clock timer for a profiled section.
    
    collect_gc=True forces a GC pass to reduce run-to-run noise before timing.
    """
    if not enabled:
        return None
    if collect_gc:
        gc.collect()
    return time.perf_counter()


def _elapsed_profile_seconds(start_time):
    """Convert a start timestamp into (start, end, elapsed).
    
    Returns triple None when timing was disabled so callers can branch uniformly.
    """
    if start_time is None:
        return None, None, None
    end_time = time.perf_counter()
    return start_time, end_time, end_time - start_time


def _write_profile_rows(profile_dir, filename, header, rows):
    """Write a complete profiling table in one call.
    
    Use this for batch profile outputs when row-by-row streaming is unnecessary.
    """
    if not _profiling_enabled(profile_dir):
        return
    os.makedirs(profile_dir, exist_ok=True)
    with open(os.path.join(profile_dir, filename), "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


# Sparse parallel FlexDTW — tiling, edge DP, Stage 2 backtrace
DEFAULT_CHUNK_LENGTH = 4000

import csv
import gc
import math
import os
import time

import FlexDTW
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def edge_slot_to_local(edge_type, position, chunk_shape):
    """Map a compact edge slot to local chunk coordinates.
    
    Edge 0 indexes the top boundary (last row in this orientation), and edge 1
    indexes the right boundary. Update this if edge orientation conventions change.
    """
    if edge_type == 0:
        return chunk_shape[0] - 1, position
    return position, chunk_shape[1] - 1


def _edge_length_for_chunk_edge(chunk_shape, edge_type):
    """Return the number of valid positions on one chunk edge.
    
    Keep this consistent with edge_slot_to_local and all edge_lens allocations.
    """
    return chunk_shape[1] if edge_type == 0 else chunk_shape[0]

def plot_alignment_with_tile_background(
    tiled_result,
    cost_matrix,
    flex_wp,
    stage2_result,
    beat_pair_paths=None,
    xy=None,
    chunk_length=None,
    use_valid_edges_only=True,
):
    """Visualize global alignment paths with tile-level context overlays.
    
    Shows: dense FlexDTW path, stitched Parflex path, optional ground-truth markers,
    and faint S-derived start-to-edge segments for debugging. Edit this when you
    need new diagnostics or plotting conventions, not algorithm behavior.
    """
    SR, HOP = 22050, 512

    if beat_pair_paths is not None:
        beat_path_1, beat_path_2 = beat_pair_paths
        import eval_tools
        gt_seconds = eval_tools.getGroundTruthTimestamps(beat_path_1, beat_path_2)
        xy = gt_seconds.copy()
        xy[:, 0] = xy[:, 0] * SR / HOP
        xy[:, 1] = xy[:, 1] * SR / HOP

    blocks = tiled_result['blocks']
    L_block = tiled_result['L_block']
    L_div = chunk_length if chunk_length is not None else L_block
    hop = tiled_result['hop']
    n_row, n_col = stage2_result['n_row'], stage2_result['n_col']
    D_arr = stage2_result.get('D_arr')
    edge_lens_arr = stage2_result.get('edge_lens')
    D_chunks = stage2_result.get('D_chunks')

    L1, L2 = cost_matrix.shape
    base_px = 900
    max_side = max(L1, L2)
    scale = base_px / max_side
    fig_width = int(max(L2 * scale, 400))
    fig_height = int(max(L1 * scale, 400))

    fig = go.Figure()

    x_S, y_S = [], []
    INF = 1e17

    for b in blocks:
        i, j = b['bi'], b['bj']
        if i >= n_row or j >= n_col:
            continue

        S_single = b['S_single']
        rows, cols = b['Ck_shape']
        r_start, r_end = b['rows']
        c_start, c_end = b['cols']

        for edge in (0, 1):  # 0=top edge, 1=right edge
            if edge_lens_arr is not None:
                edge_len = int(edge_lens_arr[i, j, edge])
            else:
                edge_len = min(L_block, cols if edge == 0 else rows)

            for idx in range(edge_len):
                if use_valid_edges_only:
                    if D_arr is not None:
                        D_val = D_arr[i, j, edge, idx]
                    elif D_chunks is not None:
                        D_val = D_chunks[i][j][edge][idx]
                    else:
                        D_val = np.nan
                    if not np.isfinite(D_val) or D_val >= INF:
                        continue

                lr, lc = edge_slot_to_local(edge, idx, (rows, cols))
                if lr < 0 or lc < 0 or lr >= rows or lc >= cols:
                    continue

                s_val = S_single[lr, lc]
                if s_val > 0:
                    start_local_r, start_local_c = 0, int(s_val)
                elif s_val < 0:
                    start_local_r, start_local_c = abs(int(s_val)), 0
                else:
                    start_local_r, start_local_c = 0, 0
                g_start_r = r_start + start_local_r
                g_start_c = c_start + start_local_c
                g_end_r   = r_start + lr
                g_end_c   = c_start + lc
                if not (0 <= g_start_r < L1 and 0 <= g_start_c < L2):
                    continue
                if not (0 <= g_end_r < L1 and 0 <= g_end_c < L2):
                    continue
                x_S.extend([g_start_c, g_end_c, None])
                y_S.extend([g_start_r, g_end_r, None])

    if x_S:
        fig.add_trace(
            go.Scattergl(
                x=x_S,
                y=y_S,
                mode="lines",
                name="Chunk S start→edge segments",
                line=dict(width=1, color="rgba(100,100,100,0.02)"),  # light grey-ish
                showlegend=False)
        )

    flex_wp = np.asarray(flex_wp)
    if flex_wp.shape[1] == 2:
        f1_frames = flex_wp[:, 0]
        f2_frames = flex_wp[:, 1]
    elif flex_wp.shape[0] == 2:
        f1_frames = flex_wp[0, :]
        f2_frames = flex_wp[1, :]
    else:
        raise ValueError(f"Unexpected flex_wp shape: {flex_wp.shape}")

    fig.add_trace(
        go.Scattergl(
            x=f2_frames,
            y=f1_frames,
            mode="lines",
            name="Global FlexDTW",
            line=dict(width=4, color="rgba(0,0,0,1)")
        )
    )

    stitched_wp = stage2_result['stitched_wp']
    if stitched_wp.size > 0:
        fig.add_trace(
            go.Scattergl(
                x=stitched_wp[:, 1],   # cols (F2)
                y=stitched_wp[:, 0],   # rows (F1)
                mode="lines",
                name="Parflex",
                line=dict(width=6, color="rgba(247,14,14,0.5)")
            )
        )

    if xy is not None:
        xy_arr = np.asarray(xy)

        if xy_arr.ndim != 2 or xy_arr.shape[1] != 2:
            raise ValueError(f"xy must have shape (N,2), got {xy_arr.shape}")

        xy_frames = xy

        fig.add_trace(
            go.Scattergl(
                x=xy_frames[:, 1],   # F2 frames
                y=xy_frames[:, 0],   # F1 frames
                mode="markers",
                name="Ground Truth",
                marker=dict(size=5, color="rgba(0,200,0,0.9)")
            )
        )
    x_lo, x_hi = -0.5, L2 - 0.5
    y_lo, y_hi = -0.5, L1 - 0.5

    fig.update_layout(
        title="Global FlexDTW vs Parflex (with chunk-S spiky background)",
        xaxis_title=f"F2 frames (0 … {L2-1})",
        yaxis_title=f"F1 frames (0 … {L1-1})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        width=fig_width,
        height=fig_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(range=[x_lo, x_hi], showgrid=False)
    fig.update_yaxes(range=[y_lo, y_hi], showgrid=False)
    shapes = []
    for x in range(L_div, L2, L_div):
        shapes.append(dict(
            type="line",
            x0=x, x1=x,
            y0=y_lo, y1=y_hi,
            line=dict(width=1)
        ))
    for y in range(L_div, L1, L_div):
        shapes.append(dict(
            type="line",
            x0=x_lo, x1=x_hi,
            y0=y, y1=y,
            line=dict(width=1)
        ))
    fig.update_layout(shapes=shapes)
    fig.show()

def _edge_starts_from_S(S):
    """Extract boundary start indices from a chunk S matrix.
    
    S encodes each cell's local path start; this helper keeps only starts that lie
    on bottom/left boundaries so neighboring chunks know which edge slots they need.
    """
    rows, cols = S.shape
    starts_bottom_edge = set()
    starts_left_edge = set()
    # Top edge: last row S[rows-1, :]
    for c in range(cols):
        val = S[rows - 1, c]
        if val > 0:
            starts_bottom_edge.add(int(val))
        else:
            # if starts on left or ==0, then it's a left edge start
            starts_left_edge.add(abs(int(val)))  # row start; keep for completeness
    # Right edge: last column S[:, cols-1]
    for r in range(rows):
        val = S[r, cols - 1]
        if val > 0:
            starts_bottom_edge.add(int(val))
        else:
            # if starts on left or ==0, then it's a left edge start
            starts_left_edge.add(abs(int(val)))  # row start; keep for completeness
    
    return starts_bottom_edge, starts_left_edge


def run_flexdtw_on_tiles(C, L, steps=None, weights=None, buffer=1, profile_dir='Profiling_results',
                  warn_slow_chunks=False, slow_chunk_seconds=1.0):
    """Run Stage 1: tile C, execute FlexDTW per tile, collect metadata.
    
    This controls chunk geometry, overlap policy, and per-tile artifacts (D/L/B/S).
    Change this when modifying tiling strategy or what stage-2 data each tile carries.
    """
    if steps is None:
        steps = [(1, 1), (1, 2), (2, 1)]
    if weights is None:
        weights = [2, 3, 3]

    L1, L2 = C.shape
    hop = L - 1
    n_chunks_1 = math.ceil((L1 - 1) / hop)
    n_chunks_2 = math.ceil((L2 - 1) / hop)
    chunks_dict = {}

    profile_file, _pw = _open_profile_csv(
        profile_dir,
        "chunk_flexdtw.csv",
        ["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"],
    )

    # Warm-up only when profiling is enabled; include it in profiling output.
    if profile_file is not None:
        _warm_h = min(int(L), int(L1))
        _warm_w = min(int(L), int(L2))
        if _warm_h >= 4 and _warm_w >= 4:
            warm_start = _start_profile_timer(True, collect_gc=True)
            try:
                import FlexDTW
                rng = np.random.RandomState(0)
                C_warm = np.ascontiguousarray(
                    rng.rand(_warm_h, _warm_w).astype(C.dtype, copy=False))
                for _ in range(2):
                    FlexDTW.flexdtw(C_warm, steps=steps, weights=weights, buffer=1)
            except ImportError:
                pass
            warm_start, warm_end, warm_elapsed = _elapsed_profile_seconds(warm_start)
            _pw.writerow(["warmup", "warmup", warm_start, warm_end, warm_elapsed])
            profile_file.flush()

    _time_chunk = profile_file is not None or warn_slow_chunks

    for i in range(n_chunks_1):
        for j in range(n_chunks_2):
            if _time_chunk:
                t_start = _start_profile_timer(profile_file is not None, collect_gc=True)
                if t_start is None:
                    t_start = time.perf_counter()

            start_1, start_2 = i * hop, j * hop
            end_1 = min(start_1 + L, L1)
            end_2 = min(start_2 + L, L2)
            C_chunk = C[int(start_1):int(end_1), int(start_2):int(end_2)]

            try:
                import FlexDTW
                best_cost, wp, D, P, B, debug = FlexDTW.flexdtw(
                    C_chunk, steps=steps, weights=weights, buffer=1)
            except ImportError:
                best_cost = 0; wp = []
                D = np.zeros_like(C_chunk); P = np.zeros_like(C_chunk)
                B = np.zeros_like(C_chunk); debug = {}

            actual_hop_1 = hop if end_1 < L1 else (L1 - start_1)
            actual_hop_2 = hop if end_2 < L2 else (L2 - start_2)
            starts_bot_edge, starts_left_edge = _edge_starts_from_S(P)

            chunks_dict[(i, j)] = {
                'C': C_chunk, 'D': D, 'S': P, 'B': B, 'debug': debug,
                'best_cost': best_cost, 'wp': wp,
                'bounds': (start_1, end_1, start_2, end_2),
                'hop': (actual_hop_1, actual_hop_2),
                'shape': C_chunk.shape,
                'starts_bot_edge':  starts_bot_edge,
                'starts_left_edge': starts_left_edge,
            }

            if _time_chunk:
                if profile_file is not None:
                    t_start, t_end, elapsed = _elapsed_profile_seconds(t_start)
                else:
                    t_end = time.perf_counter()
                    elapsed = t_end - t_start
                if warn_slow_chunks and elapsed > slow_chunk_seconds:
                    print(f"  Warning, chunk ({i},{j}) took {elapsed} seconds")
                if profile_file is not None:
                    _pw.writerow([i, j, t_start, t_end, elapsed])
                    profile_file.flush()

    _close_profile_csv(profile_file)
    return chunks_dict, L, n_chunks_1, n_chunks_2


@_parflex_njit
def _nb_edge_index_to_local_coords(edge_type, position, nrow, ncol):
    """Numba helper mapping edge slot to local (row, col).
    
    Must stay identical to edge_slot_to_local semantics used in Python code.
    """
    if edge_type == 0:
        return nrow - 1, position
    return position, ncol - 1

def local_to_global_cell(chunk_i, chunk_j, local_row, local_col, chunks_dict):
    """Convert local tile coordinates into global matrix coordinates.
    
    Use when stitching paths or resolving cross-tile transitions.
    """
    start_1, _, start_2, _ = chunks_dict[(chunk_i, chunk_j)]['bounds']
    return start_1 + local_row, start_2 + local_col


def global_cell_to_prev_tile_edge(global_row, global_col, prev_i, prev_j, chunks_dict, L):
    """Map a global coordinate onto a predecessor tile edge index.
    
    Raises when the point is not on predecessor boundary; this catches broken
    cross-tile handoff logic early during debugging.
    """
    start_1, _, start_2, _ = chunks_dict[(prev_i, prev_j)]['bounds']
    local_row = global_row - start_1
    local_col = global_col - start_2
    prev_shape = chunks_dict[(prev_i, prev_j)]['D'].shape
    if local_row == prev_shape[0] - 1:
        return 0, local_col
    if local_col == prev_shape[1] - 1:
        return 1, local_row
    raise ValueError(f"({local_row},{local_col}) not on edge of prev chunk")



def _build_edge_data(chunks_dict, n_chunks_1, n_chunks_2, L):
    """Materialize compact edge tensors consumed by sparse stage-2 kernels.
    
    This is where chunk dictionaries become contiguous arrays for Numba.
    Edit here when adding/removing edge features used by DP kernels.
    """
    shape4 = (n_chunks_1, n_chunks_2, 2, L)
    edge_Df      = np.full(shape4, np.inf,  dtype=np.float64)
    edge_start_r = np.zeros(shape4,         dtype=np.int64)
    edge_start_c = np.zeros(shape4,         dtype=np.int64)
    edge_Cf_olap = np.zeros(shape4,         dtype=np.float64)
    edge_lens    = np.zeros((n_chunks_1, n_chunks_2, 2), dtype=np.int64)
    chunk_rows   = np.zeros((n_chunks_1, n_chunks_2),    dtype=np.int64)
    chunk_cols   = np.zeros((n_chunks_1, n_chunks_2),    dtype=np.int64)
    chunk_bounds = np.zeros((n_chunks_1, n_chunks_2, 4), dtype=np.int64)
    chunk_valid  = np.zeros((n_chunks_1, n_chunks_2),    dtype=np.int64)

    for (i, j), ch in chunks_dict.items():
        r, c = ch['shape']
        chunk_rows[i, j]   = r
        chunk_cols[i, j]   = c
        s1, e1, s2, e2     = ch['bounds']
        chunk_bounds[i, j] = [s1, e1, s2, e2]
        chunk_valid[i, j]  = 1
        edge_lens[i, j, 0] = c
        edge_lens[i, j, 1] = r

        D     = ch['D'].astype(np.float64)
        S_raw = ch['S'].astype(np.float64)
        S_safe = np.where(np.isnan(S_raw), 0.0, S_raw)
        Cc    = ch['C'].astype(np.float64)

        for pos in range(c):   # top edge
            lr, lc = r - 1, pos
            edge_Df[i, j, 0, pos] = D[lr, lc]
            s_val = S_safe[lr, lc]
            if s_val > 0:   sr, sc = 0, int(s_val)
            elif s_val < 0: sr, sc = int(-s_val), 0
            else:           sr, sc = 0, 0
            edge_start_r[i, j, 0, pos] = sr
            edge_start_c[i, j, 0, pos] = sc
            edge_Cf_olap[i, j, 0, pos] = Cc[sr, sc]

        for pos in range(r):   # right edge
            lr, lc = pos, c - 1
            edge_Df[i, j, 1, pos] = D[lr, lc]
            s_val = S_safe[lr, lc]
            if s_val > 0:   sr, sc = 0, int(s_val)
            elif s_val < 0: sr, sc = int(-s_val), 0
            else:           sr, sc = 0, 0
            edge_start_r[i, j, 1, pos] = sr
            edge_start_c[i, j, 1, pos] = sc
            edge_Cf_olap[i, j, 1, pos] = Cc[sr, sc]

    return (edge_Df, edge_start_r, edge_start_c, edge_Cf_olap,
            edge_lens, chunk_rows, chunk_cols, chunk_bounds, chunk_valid)


# ---------------------------------------------------------------------------
# Build valid-position mask from sparse start sets
#
# valid_mask[i, j, edge, pos] = True  → compute this position
#                              = False → leave as inf (not needed by any neighbor)
#
# Rules (matching original _valid_positions + special-case overrides):
#   edge 0 (top,  indexed by col):
#     - Always True if i == n1-1 (last row — needed for global top-edge scan)
#     - Otherwise: True for pos in {0, edge_len-1} ∪ chunks_dict[(i+1,j)].starts_bot_edge
#   edge 1 (right, indexed by row):
#     - Always True if j == n2-1 (last col — needed for global right-edge scan)
#     - Otherwise: True for pos in {0, edge_len-1} ∪ chunks_dict[(i,j+1)].starts_left_edge
# ---------------------------------------------------------------------------

def _build_valid_mask(chunks_dict, n_chunks_1, n_chunks_2, L):
    """Build sparse compute mask for required edge slots only.
    
    Mask rules encode dependency flow between neighboring chunks plus mandatory
    global-boundary slots. Update this when changing sparsification policy.
    """
    valid_mask = np.zeros((n_chunks_1, n_chunks_2, 2, L), dtype=np.bool_)

    for (i, j), ch in chunks_dict.items():
        r, c = ch['shape']

        # ── Top edge (edge 0, positions 0..c-1) ──────────────────────────────
        elen0 = c
        if i == n_chunks_1 - 1:
            # Last row: all positions needed for global top-edge scan.
            valid_mask[i, j, 0, :elen0] = True
        else:
            # Non-last row: positions needed by the chunk directly above (i+1, j).
            above = (i + 1, j)
            starts = chunks_dict[above].get('starts_bot_edge', set()) if above in chunks_dict else set()
            for pos in starts:
                if 0 <= pos < elen0:
                    valid_mask[i, j, 0, pos] = True
            # Always include boundary positions for edge continuity.
            valid_mask[i, j, 0, 0]         = True
            valid_mask[i, j, 0, elen0 - 1] = True

        # ── Right edge (edge 1, positions 0..r-1) ────────────────────────────
        elen1 = r
        if j == n_chunks_2 - 1:
            # Last col: all positions needed for global right-edge scan.
            valid_mask[i, j, 1, :elen1] = True
        else:
            # Non-last col: positions needed by the chunk to the right (i, j+1).
            right = (i, j + 1)
            starts = chunks_dict[right].get('starts_left_edge', set()) if right in chunks_dict else set()
            for pos in starts:
                if 0 <= pos < elen1:
                    valid_mask[i, j, 1, pos] = True
            valid_mask[i, j, 1, 0]         = True
            valid_mask[i, j, 1, elen1 - 1] = True

    return valid_mask


# ---------------------------------------------------------------------------
# JIT DP kernels (sparse versions — check valid_mask before computing)
# ---------------------------------------------------------------------------

@_parflex_dp_njit
def _nb_initialize_chunks_sparse(edge_Df, edge_start_r, edge_start_c, edge_Cf_olap,
                                  edge_lens, chunk_rows, chunk_cols, chunk_bounds, chunk_valid,
                                  valid_mask, n_chunks_1, n_chunks_2, D_arr, L_arr):
    """Initialize stage-2 DP values for first row/column frontier chunks.
    
    These chunks have no full predecessor pair, so they are seeded separately before
    interior propagation. Modify when changing boundary initialization rules.
    """
    # ── Chunk (0, 0) ─────────────────────────────────────────────────────────
    nrow00 = chunk_rows[0, 0]
    ncol00 = chunk_cols[0, 0]
    for edge in range(2):
        elen = ncol00 if edge == 0 else nrow00
        for pos in range(elen):
            if not valid_mask[0, 0, edge, pos]:
                continue
            start_r = edge_start_r[0, 0, edge, pos]
            start_c = edge_start_c[0, 0, edge, pos]
            if edge == 0:
                lr = nrow00 - 1; lc = pos
            else:
                lr = pos;        lc = ncol00 - 1
            if start_r == 0 or start_c == 0:
                D_arr[0, 0, edge, pos] = edge_Df[0, 0, edge, pos]
                L_arr[0, 0, edge, pos] = abs(lr - start_r) + abs(lc - start_c)

    # ── First row: j = 1 … n_chunks_2-1 ─────────────────────────────────────
    for j in range(1, n_chunks_2):
        if chunk_valid[0, j] == 0:
            continue
        nrow = chunk_rows[0, j]
        ncol = chunk_cols[0, j]

        prev_top_elen = chunk_cols[0, j - 1]
        D_arr[0, j, 0, 0] = D_arr[0, j - 1, 0, prev_top_elen - 1]
        L_arr[0, j, 0, 0] = L_arr[0, j - 1, 0, prev_top_elen - 1]

        for edge in range(2):
            elen = ncol if edge == 0 else nrow
            for pos in range(1, elen):
                if not valid_mask[0, j, edge, pos]:
                    continue
                start_r = edge_start_r[0, j, edge, pos]
                start_c = edge_start_c[0, j, edge, pos]
                if edge == 0:
                    lr = nrow - 1; lc = pos
                else:
                    lr = pos;      lc = ncol - 1

                if start_r == 0:
                    D_arr[0, j, edge, pos] = edge_Df[0, j, edge, pos]
                    L_arr[0, j, edge, pos] = abs(lr - start_r) + abs(lc - start_c)
                elif start_c == 0:
                    g_start_r = chunk_bounds[0, j, 0] + start_r
                    g_start_c = chunk_bounds[0, j, 2]
                    ps1 = chunk_bounds[0, j - 1, 0]
                    ps2 = chunk_bounds[0, j - 1, 2]
                    p_lr = g_start_r - ps1
                    p_lc = g_start_c - ps2
                    p_nrow = chunk_rows[0, j - 1]
                    p_ncol = chunk_cols[0, j - 1]
                    if p_lr == p_nrow - 1:
                        p_edge = np.int64(0); p_pos = p_lc; p_elen = p_ncol
                    elif p_lc == p_ncol - 1:
                        p_edge = np.int64(1); p_pos = p_lr; p_elen = p_nrow
                    else:
                        continue
                    if p_pos < 0 or p_pos >= p_elen:
                        continue
                    prev_cost = D_arr[0, j - 1, p_edge, p_pos]
                    if not np.isfinite(prev_cost):
                        continue
                    overlap = edge_Cf_olap[0, j, edge, pos]
                    D_arr[0, j, edge, pos] = edge_Df[0, j, edge, pos] + prev_cost - overlap
                    L_arr[0, j, edge, pos] = (L_arr[0, j - 1, p_edge, p_pos]
                                               + abs(lr - start_r) + abs(lc - start_c))

    # ── First column: i = 1 … n_chunks_1-1 ──────────────────────────────────
    for i in range(1, n_chunks_1):
        if chunk_valid[i, 0] == 0:
            continue
        nrow = chunk_rows[i, 0]
        ncol = chunk_cols[i, 0]

        prev_right_elen = chunk_rows[i - 1, 0]
        D_arr[i, 0, 1, 0] = D_arr[i - 1, 0, 1, prev_right_elen - 1]
        L_arr[i, 0, 1, 0] = L_arr[i - 1, 0, 1, prev_right_elen - 1]

        for edge in range(2):
            elen = ncol if edge == 0 else nrow
            for pos in range(1, elen):
                if not valid_mask[i, 0, edge, pos]:
                    continue
                start_r = edge_start_r[i, 0, edge, pos]
                start_c = edge_start_c[i, 0, edge, pos]
                if edge == 0:
                    lr = nrow - 1; lc = pos
                else:
                    lr = pos;      lc = ncol - 1

                if start_c == 0:
                    D_arr[i, 0, edge, pos] = edge_Df[i, 0, edge, pos]
                    L_arr[i, 0, edge, pos] = abs(lr - start_r) + abs(lc - start_c)
                elif start_r == 0:
                    g_start_r = chunk_bounds[i, 0, 0]
                    g_start_c = chunk_bounds[i, 0, 2] + start_c
                    ps1 = chunk_bounds[i - 1, 0, 0]
                    ps2 = chunk_bounds[i - 1, 0, 2]
                    p_lr = g_start_r - ps1
                    p_lc = g_start_c - ps2
                    p_nrow = chunk_rows[i - 1, 0]
                    p_ncol = chunk_cols[i - 1, 0]
                    if p_lr == p_nrow - 1:
                        p_edge = np.int64(0); p_pos = p_lc; p_elen = p_ncol
                    elif p_lc == p_ncol - 1:
                        p_edge = np.int64(1); p_pos = p_lr; p_elen = p_nrow
                    else:
                        continue
                    if p_pos < 0 or p_pos >= p_elen:
                        continue
                    prev_cost = D_arr[i - 1, 0, p_edge, p_pos]
                    if not np.isfinite(prev_cost):
                        continue
                    overlap = edge_Cf_olap[i, 0, edge, pos]
                    D_arr[i, 0, edge, pos] = edge_Df[i, 0, edge, pos] + prev_cost - overlap
                    L_arr[i, 0, edge, pos] = (L_arr[i - 1, 0, p_edge, p_pos]
                                               + abs(lr - start_r) + abs(lc - start_c))

    return D_arr, L_arr


@_parflex_dp_njit
def _nb_dp_fill_chunks_sparse(edge_Df, edge_start_r, edge_start_c, edge_Cf_olap,
                               edge_lens, chunk_rows, chunk_cols, chunk_bounds, chunk_valid,
                               valid_mask, n_chunks_1, n_chunks_2, D_arr, L_arr):
    """Run sparse stage-2 DP propagation for interior chunks.
    
    Consumes edge arrays and valid_mask, writes D_arr/L_arr only for required slots.
    This is the main kernel to edit for recurrence or transition-cost changes.
    """
    for i in range(n_chunks_1):
        for j in range(n_chunks_2):
            if i == 0 or j == 0:
                continue
            if chunk_valid[i, j] == 0:
                continue

            nrow = chunk_rows[i, j]
            ncol = chunk_cols[i, j]

            for edge in range(2):
                elen = ncol if edge == 0 else nrow

                for pos in range(elen):
                    if not valid_mask[i, j, edge, pos]:
                        continue

                    if pos == 0:
                        if edge == 0:
                            p_elen = chunk_cols[i, j - 1]
                            lc_val = D_arr[i, j - 1, 0, p_elen - 1]
                            ll_val = L_arr[i, j - 1, 0, p_elen - 1]
                            if np.isfinite(lc_val):
                                D_arr[i, j, 0, 0] = lc_val
                                L_arr[i, j, 0, 0] = ll_val
                        else:
                            p_elen = chunk_rows[i - 1, j]
                            tc_val = D_arr[i - 1, j, 1, p_elen - 1]
                            tl_val = L_arr[i - 1, j, 1, p_elen - 1]
                            if np.isfinite(tc_val):
                                D_arr[i, j, 1, 0] = tc_val
                                L_arr[i, j, 1, 0] = tl_val
                        continue

                    if edge == 0:
                        lr = nrow - 1; lc = pos
                    else:
                        lr = pos;      lc = ncol - 1

                    start_r = edge_start_r[i, j, edge, pos]
                    start_c = edge_start_c[i, j, edge, pos]

                    if start_r == 0:
                        pi = np.int64(i - 1); pj = np.int64(j)
                    elif start_c == 0:
                        pi = np.int64(i);     pj = np.int64(j - 1)
                    else:
                        # Path started inside this chunk.
                        D_arr[i, j, edge, pos] = edge_Df[i, j, edge, pos]
                        L_arr[i, j, edge, pos] = abs(lr - start_r) + abs(lc - start_c)
                        continue

                    if pi < 0 or pj < 0:
                        continue

                    g_start_r = chunk_bounds[i, j, 0] + start_r
                    g_start_c = chunk_bounds[i, j, 2] + start_c
                    ps1 = chunk_bounds[pi, pj, 0]
                    ps2 = chunk_bounds[pi, pj, 2]
                    p_lr = g_start_r - ps1
                    p_lc = g_start_c - ps2
                    p_nrow = chunk_rows[pi, pj]
                    p_ncol = chunk_cols[pi, pj]

                    if p_lr == p_nrow - 1:
                        p_edge = np.int64(0); p_pos = p_lc; p_elen = p_ncol
                    elif p_lc == p_ncol - 1:
                        p_edge = np.int64(1); p_pos = p_lr; p_elen = p_nrow
                    else:
                        continue

                    if p_pos < 0 or p_pos >= p_elen:
                        continue

                    prev_cost = D_arr[pi, pj, p_edge, p_pos]
                    prev_len  = L_arr[pi, pj, p_edge, p_pos]
                    if not np.isfinite(prev_cost):
                        continue

                    overlap = edge_Cf_olap[i, j, edge, pos]
                    D_arr[i, j, edge, pos] = prev_cost + edge_Df[i, j, edge, pos] - overlap
                    L_arr[i, j, edge, pos] = prev_len + abs(lr - start_r) + abs(lc - start_c)

    return D_arr, L_arr


# ---------------------------------------------------------------------------
# propagate_tile_edge_costs  (sparse wrapper — Numba JIT init + DP fill)
# ---------------------------------------------------------------------------

def propagate_tile_edge_costs(chunks_dict, L, num_chunks_1, num_chunks_2, buffer_param=0.1,
                               profile_dir='Profiling_results'):
    """High-level stage-2 propagation wrapper.
    
    Builds edge tensors + sparse mask, runs initialization and interior DP kernels,
    then returns compact arrays used by scan/backtrace. Start here for stage-2 tuning.
    """
    (edge_Df, edge_start_r, edge_start_c, edge_Cf_olap,
     edge_lens, chunk_rows, chunk_cols, chunk_bounds, chunk_valid) = \
        _build_edge_data(chunks_dict, num_chunks_1, num_chunks_2, L)

    valid_mask = _build_valid_mask(chunks_dict, num_chunks_1, num_chunks_2, L)

    D_arr = np.full((num_chunks_1, num_chunks_2, 2, L), np.inf, dtype=np.float64)
    L_arr = np.full((num_chunks_1, num_chunks_2, 2, L), np.inf, dtype=np.float64)

    # ── Init phase ───────────────────────────────────────────────────────────
    t0 = _start_profile_timer(_profiling_enabled(profile_dir), collect_gc=True)

    _nb_initialize_chunks_sparse(
        edge_Df, edge_start_r, edge_start_c, edge_Cf_olap,
        edge_lens, chunk_rows, chunk_cols, chunk_bounds, chunk_valid,
        valid_mask, num_chunks_1, num_chunks_2, D_arr, L_arr)

    if t0 is not None:
        t0, t1, elapsed = _elapsed_profile_seconds(t0)
        _write_profile_rows(
            profile_dir,
            "initialize_chunks.csv",
            ["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"],
            [["all", "all", t0, t1, elapsed]],
        )

    # ── DP fill phase ─────────────────────────────────────────────────────────
    t0 = _start_profile_timer(_profiling_enabled(profile_dir), collect_gc=True)

    _nb_dp_fill_chunks_sparse(
        edge_Df, edge_start_r, edge_start_c, edge_Cf_olap,
        edge_lens, chunk_rows, chunk_cols, chunk_bounds, chunk_valid,
        valid_mask, num_chunks_1, num_chunks_2, D_arr, L_arr)

    if t0 is not None:
        t0, t1, elapsed = _elapsed_profile_seconds(t0)
        _write_profile_rows(
            profile_dir,
            "dp_fill_chunks.csv",
            ["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"],
            [["all", "all", t0, t1, elapsed]],
        )

    return D_arr, L_arr, edge_lens


# ---------------------------------------------------------------------------
# Sync (identical to dense — JIT'd)
# ---------------------------------------------------------------------------

@_parflex_dp_njit
def sync_tile_overlap_edges(D_arr, L_arr, edge_lens, n_chunks_1, n_chunks_2):
    """Force consistency on 1-cell overlaps between neighboring tiles.
    
    When two representations refer to the same global cell, keep D/L synchronized to
    avoid seam artifacts before global edge scanning.
    """
    for i in range(n_chunks_1):
        for j in range(n_chunks_2 - 1):
            elen = int(edge_lens[i, j, 0])
            if elen == 0: continue
            D_left  = D_arr[i, j,     0, elen - 1]
            D_right = D_arr[i, j + 1, 0, 0]
            if np.isfinite(D_left) and not np.isfinite(D_right):
                D_arr[i, j + 1, 0, 0] = D_left
                L_arr[i, j + 1, 0, 0] = L_arr[i, j, 0, elen - 1]
            elif not np.isfinite(D_left) and np.isfinite(D_right):
                D_arr[i, j, 0, elen - 1] = D_right
                L_arr[i, j, 0, elen - 1] = L_arr[i, j + 1, 0, 0]

    for i in range(n_chunks_1 - 1):
        for j in range(n_chunks_2):
            elen = int(edge_lens[i, j, 1])
            if elen == 0: continue
            D_bot = D_arr[i,     j, 1, elen - 1]
            D_top = D_arr[i + 1, j, 1, 0]
            if np.isfinite(D_bot) and not np.isfinite(D_top):
                D_arr[i + 1, j, 1, 0] = D_bot
                L_arr[i + 1, j, 1, 0] = L_arr[i, j, 1, elen - 1]
            elif not np.isfinite(D_bot) and np.isfinite(D_top):
                D_arr[i, j, 1, elen - 1] = D_top
                L_arr[i, j, 1, elen - 1] = L_arr[i + 1, j, 1, 0]

    for i in range(n_chunks_1):
        for j in range(n_chunks_2):
            et = int(edge_lens[i, j, 0])
            er = int(edge_lens[i, j, 1])
            if et == 0 or er == 0: continue
            D_tc = D_arr[i, j, 0, et - 1]
            D_rc = D_arr[i, j, 1, er - 1]
            if np.isfinite(D_tc) and not np.isfinite(D_rc):
                D_arr[i, j, 1, er - 1] = D_tc
                L_arr[i, j, 1, er - 1] = L_arr[i, j, 0, et - 1]
            elif not np.isfinite(D_tc) and np.isfinite(D_rc):
                D_arr[i, j, 0, et - 1] = D_rc
                L_arr[i, j, 0, et - 1] = L_arr[i, j, 1, er - 1]
            elif np.isfinite(D_tc) and np.isfinite(D_rc) and abs(D_tc - D_rc) > 1e-10:
                D_arr[i, j, 1, er - 1] = D_tc
                L_arr[i, j, 1, er - 1] = L_arr[i, j, 0, et - 1]

    return D_arr, L_arr


# ---------------------------------------------------------------------------
# Backtrace kernel (identical to dense, fixed max_iters)
# ---------------------------------------------------------------------------

@_parflex_njit
def _nb_backtrace_within_chunk(B_single, steps, start_r, start_c, end_r, end_c,
                               global_r_offset, global_c_offset):
    """Backtrace one intra-chunk path segment using B and steps.
    
    Walks from selected edge endpoint back to that segment start and emits globalized
    coordinates. If path rules change, keep this aligned with stage-1 step encoding.
    """
    nrows, ncols = B_single.shape
    nsteps = steps.shape[0]
    max_iters = nrows + ncols   # tight bound: every step decreases r by ≥1
    out = np.empty((max_iters + 4, 2), dtype=np.int64)
    k = 0
    r, c = end_r, end_c
    gro, gco = global_r_offset, global_c_offset

    for iters in range(max_iters):
        out[k, 0] = r + gro; out[k, 1] = c + gco; k += 1
        if r == start_r and c == start_c:
            break
        step_idx = int(B_single[r, c])
        if step_idx < 0 or step_idx >= nsteps:
            if r != start_r or c != start_c:
                out[k, 0] = start_r + gro; out[k, 1] = start_c + gco; k += 1
            break
        dr = int(steps[step_idx, 0]); dc = int(steps[step_idx, 1])
        prev_r = r - dr;              prev_c = c - dc
        if prev_r < 0 or prev_c < 0 or prev_r >= nrows or prev_c >= ncols:
            if r != start_r or c != start_c:
                out[k, 0] = start_r + gro; out[k, 1] = start_c + gco; k += 1
            break
        r, c = prev_r, prev_c
    else:
        if r != start_r or c != start_c:
            out[k, 0] = start_r + gro; out[k, 1] = start_c + gco; k += 1

    return out[:k]


def _stage2_edge_to_local(edge, idx, rows, cols):
    """Python wrapper for stage-2 edge-slot to local-coordinate mapping.
    """
    lr, lc = _nb_edge_index_to_local_coords(int(edge), int(idx), int(rows), int(cols))
    return int(lr), int(lc)


def _stage2_chunk_path_points(B_single, steps, start_r, start_c, end_r, end_c, gro, gco):
    """Return a Python list of global points for one chunk backtrace segment.
    """
    out = _nb_backtrace_within_chunk(
        np.ascontiguousarray(B_single),
        np.ascontiguousarray(steps, dtype=np.int64),
        int(start_r), int(start_c), int(end_r), int(end_c), int(gro), int(gco))
    return [(int(out[k, 0]), int(out[k, 1])) for k in range(out.shape[0])]


def backtrace_and_stitch(tiled_result, chunks_dict, start_i, start_j, start_edge, start_idx):
    """Follow predecessor tiles and stitch chunk segments into one global path.
    
    This enforces cross-tile predecessor routing (up, left, or diagonal corner case).
    Edit this when changing stage-2 predecessor selection or stitch ordering.
    """
    path = []
    cur_i, cur_j, cur_edge, cur_idx = start_i, start_j, start_edge, start_idx
    steps = tiled_result['stage1_params']['steps']
    visited = set()

    for _ in range(101):
        chunk_key = (cur_i, cur_j, cur_edge, cur_idx)
        if chunk_key in visited:
            break
        visited.add(chunk_key)
        if (cur_i, cur_j) not in chunks_dict:
            break

        b = chunks_dict[(cur_i, cur_j)]
        rows, cols = b['shape']
        r_start, r_end, c_start, c_end = b['bounds']

        end_r, end_c = _stage2_edge_to_local(cur_edge, cur_idx, rows, cols)
        S_val = b['S'][end_r, end_c]

        if S_val >= 0:
            start_r, start_c = 0, int(S_val)
        else:
            start_r, start_c = int(-S_val), 0

        for pt in _stage2_chunk_path_points(
                b['B'], steps, start_r, start_c, end_r, end_c, r_start, c_start):
            if not path or path[-1] != pt:
                path.append(pt)

        g_start_row = r_start + start_r
        g_start_col = c_start + start_c
        if g_start_row == 0 or g_start_col == 0:
            break

        corner_landed = (start_r == 0 and start_c == 0)
        if corner_landed:
            prev_i, prev_j, prev_edge = cur_i - 1, cur_j - 1, 0
        elif S_val >= 0:
            prev_i, prev_j, prev_edge = cur_i - 1, cur_j, 0
        else:
            prev_i, prev_j, prev_edge = cur_i, cur_j - 1, 1

        if prev_i < 0 or prev_j < 0:
            break
        if (prev_i, prev_j) not in chunks_dict:
            break

        prev_b = chunks_dict[(prev_i, prev_j)]
        prev_r_start, _, prev_c_start, _ = prev_b['bounds']
        prev_rows, prev_cols = prev_b['shape']

        if corner_landed:
            prev_idx = prev_cols - 1
            max_prev_idx = prev_cols
        else:
            prev_lr = g_start_row - prev_r_start
            prev_lc = g_start_col - prev_c_start
            prev_idx = prev_lc if prev_edge == 0 else prev_lr
            max_prev_idx = prev_cols if prev_edge == 0 else prev_rows

        if prev_idx < 0 or prev_idx >= max_prev_idx:
            break
        cur_i, cur_j, cur_edge, cur_idx = prev_i, prev_j, prev_edge, prev_idx

    return path[::-1]


# ---------------------------------------------------------------------------
# Edge scan (identical to dense)
# ---------------------------------------------------------------------------

def _build_edge_lookups(chunks_dict, L1, L2):
    """Precompute global-boundary lookup tables for O(1) edge access.
    
    Maps each global top/right position to the owning tile edge slot used in stage 2.
    """
    top_ci  = np.full(L2, -1, dtype=np.int64); top_cj  = np.full(L2, -1, dtype=np.int64)
    top_et  = np.full(L2, -1, dtype=np.int64); top_idx = np.full(L2, -1, dtype=np.int64)
    right_ci  = np.full(L1, -1, dtype=np.int64); right_cj  = np.full(L1, -1, dtype=np.int64)
    right_et  = np.full(L1, -1, dtype=np.int64); right_idx = np.full(L1, -1, dtype=np.int64)

    g_row_top, g_col_right = L1 - 1, L2 - 1

    for (bi, bj), ch in chunks_dict.items():
        s1, e1, s2, e2 = ch['bounds']
        rows, cols = ch['shape']

        if s1 <= g_row_top < e1:
            local_r = g_row_top - s1
            for g_col in range(s2, e2):
                if top_ci[g_col] >= 0: continue
                local_c = g_col - s2
                if local_r == rows - 1:
                    top_ci[g_col] = bi; top_cj[g_col] = bj
                    top_et[g_col] = 0;  top_idx[g_col] = local_c
                elif local_c == cols - 1:
                    top_ci[g_col] = bi; top_cj[g_col] = bj
                    top_et[g_col] = 1;  top_idx[g_col] = local_r

        if s2 <= g_col_right < e2:
            local_c = g_col_right - s2
            for g_row in range(s1, e1):
                if right_ci[g_row] >= 0: continue
                local_r = g_row - s1
                if local_c == cols - 1:
                    right_ci[g_row] = bi; right_cj[g_row] = bj
                    right_et[g_row] = 1;  right_idx[g_row] = local_r
                elif local_r == rows - 1:
                    right_ci[g_row] = bi; right_cj[g_row] = bj
                    right_et[g_row] = 0;  right_idx[g_row] = local_c

    return top_ci, top_cj, top_et, top_idx, right_ci, right_cj, right_et, right_idx


@_parflex_dp_njit
def _nb_scan_edges(D_arr, L_arr, edge_lens,
                   top_ci, top_cj, top_et, top_idx,
                   right_ci, right_cj, right_et, right_idx,
                   L1, L2, L_block, buf,
                   top_D_out, top_L_out, right_D_out, right_L_out,
                   seg_best_norm, seg_best_D, seg_best_L,
                   seg_best_ci, seg_best_cj, seg_best_et, seg_best_idx,
                   seg_best_grow, seg_best_gcol,
                   best_out):
    """Scan global top/right boundaries for best normalized terminal candidate.
    
    Also records per-segment winners for optional diagnostics/visualization. Adjust
    this when changing terminal objective (for example cost/length formulation).
    """
    LARGE = 1e9
    best_norm = LARGE
    best_out[6] = 0

    for g_col in range(L2):
        if buf > 0 and g_col < buf: continue
        ci = top_ci[g_col]; cj = top_cj[g_col]
        et = top_et[g_col]; idx = top_idx[g_col]
        if ci < 0: continue
        p_elen = edge_lens[ci, cj, et]
        if idx < 0 or idx >= p_elen: continue
        D_val = D_arr[ci, cj, et, idx]; L_val = L_arr[ci, cj, et, idx]
        if not np.isfinite(D_val) or D_val >= LARGE or L_val <= 0.0: continue
        norm = D_val / L_val
        top_D_out[g_col] = D_val; top_L_out[g_col] = L_val
        if norm < best_norm:
            best_norm = norm
            best_out[0] = L1 - 1; best_out[1] = g_col
            best_out[2] = ci;     best_out[3] = cj
            best_out[4] = et;     best_out[5] = idx
            best_out[6] = 1
        seg_idx = g_col // L_block
        if seg_idx < seg_best_norm.shape[1] and norm < seg_best_norm[0, seg_idx]:
            seg_best_norm[0, seg_idx] = norm
            seg_best_D[0, seg_idx] = D_val;  seg_best_L[0, seg_idx] = L_val
            seg_best_ci[0, seg_idx] = ci;    seg_best_cj[0, seg_idx] = cj
            seg_best_et[0, seg_idx] = et;    seg_best_idx[0, seg_idx] = idx
            seg_best_grow[0, seg_idx] = L1 - 1; seg_best_gcol[0, seg_idx] = g_col

    for g_row in range(L1):
        if buf > 0 and g_row < buf: continue
        ci = right_ci[g_row]; cj = right_cj[g_row]
        et = right_et[g_row]; idx = right_idx[g_row]
        if ci < 0: continue
        p_elen = edge_lens[ci, cj, et]
        if idx < 0 or idx >= p_elen: continue
        D_val = D_arr[ci, cj, et, idx]; L_val = L_arr[ci, cj, et, idx]
        if not np.isfinite(D_val) or L_val <= 0.0: continue
        norm = D_val / L_val
        right_D_out[g_row] = D_val; right_L_out[g_row] = L_val
        if norm < best_norm:
            best_norm = norm
            best_out[0] = g_row;  best_out[1] = L2 - 1
            best_out[2] = ci;     best_out[3] = cj
            best_out[4] = et;     best_out[5] = idx
            best_out[6] = 1
        seg_idx = g_row // L_block
        if seg_idx < seg_best_norm.shape[1] and norm < seg_best_norm[1, seg_idx]:
            seg_best_norm[1, seg_idx] = norm
            seg_best_D[1, seg_idx] = D_val;  seg_best_L[1, seg_idx] = L_val
            seg_best_ci[1, seg_idx] = ci;    seg_best_cj[1, seg_idx] = cj
            seg_best_et[1, seg_idx] = et;    seg_best_idx[1, seg_idx] = idx
            seg_best_grow[1, seg_idx] = g_row; seg_best_gcol[1, seg_idx] = L2 - 1

    return best_norm


# ---------------------------------------------------------------------------
# Stage 2 backtrace
# ---------------------------------------------------------------------------

def stage2_scan_and_stitch(tiled_result, chunks_dict, D_arr, L_arr, edge_lens,
                                  L1, L2, L_block, buffer_stage2=200, top_k=1,
                                  profile_dir='Profiling_results',
                                  backtrace_segments=False):
    """Choose best terminal edge candidate, then reconstruct stitched path.
    
    This is the stage-2 orchestration point after propagation/synchronization.
    """
    n_row = tiled_result['n_row']
    n_col = tiled_result['n_col']

    top_ci, top_cj, top_et, top_idx, right_ci, right_cj, right_et, right_idx = \
        _build_edge_lookups(chunks_dict, L1, L2)

    top_D   = np.full(L2, np.nan, dtype=np.float64)
    top_L   = np.zeros(L2,        dtype=np.float64)
    right_D = np.full(L1, np.nan, dtype=np.float64)
    right_L = np.zeros(L1,        dtype=np.float64)

    buf          = int(buffer_stage2)
    n_segs_top   = (L2 + L_block - 1) // L_block
    n_segs_right = (L1 + L_block - 1) // L_block
    max_segs     = max(n_segs_top, n_segs_right, 1)

    seg_best_norm = np.full((2, max_segs), np.inf,  dtype=np.float64)
    seg_best_D    = np.full((2, max_segs), np.inf,  dtype=np.float64)
    seg_best_L    = np.zeros((2, max_segs),          dtype=np.float64)
    seg_best_ci   = np.full((2, max_segs), -1,       dtype=np.int64)
    seg_best_cj   = np.full((2, max_segs), -1,       dtype=np.int64)
    seg_best_et   = np.full((2, max_segs), -1,       dtype=np.int64)
    seg_best_idx  = np.full((2, max_segs), -1,       dtype=np.int64)
    seg_best_grow = np.full((2, max_segs), -1,       dtype=np.int64)
    seg_best_gcol = np.full((2, max_segs), -1,       dtype=np.int64)
    best_out      = np.full(7, -1, dtype=np.int64)

    _profile_rows = []

    profile_enabled = _profiling_enabled(profile_dir)
    _t0 = _start_profile_timer(profile_enabled, collect_gc=True)

    best_norm = _nb_scan_edges(
        D_arr, L_arr, edge_lens,
        top_ci, top_cj, top_et, top_idx,
        right_ci, right_cj, right_et, right_idx,
        L1, L2, L_block, buf,
        top_D, top_L, right_D, right_L,
        seg_best_norm, seg_best_D, seg_best_L,
        seg_best_ci, seg_best_cj, seg_best_et, seg_best_idx,
        seg_best_grow, seg_best_gcol,
        best_out,
    )

    if _t0 is not None:
        _t0, _t1, elapsed = _elapsed_profile_seconds(_t0)
        _profile_rows.append(["edge_scan", _t0, _t1, elapsed])

    if best_out[6] == 0:
        raise ValueError("Stage 2: No valid endpoint found on global top/right edges.")

    best_overall_end = tuple(int(best_out[k]) for k in range(6))

    top_mask   = (top_L > 0) & np.isfinite(top_D)
    right_mask = (right_L > 0) & np.isfinite(right_D)
    top_norm   = np.full(L2, np.nan); right_norm = np.full(L1, np.nan)
    top_norm[top_mask]     = top_D[top_mask]   / top_L[top_mask]
    right_norm[right_mask] = right_D[right_mask] / right_L[right_mask]

    edge_names = ['top', 'right']
    best_per_segment = {}
    for etype in range(2):
        n_segs = n_segs_top if etype == 0 else n_segs_right
        for seg_idx in range(n_segs):
            ci = int(seg_best_ci[etype, seg_idx])
            if ci < 0: continue
            key = (edge_names[etype], seg_idx)
            best_per_segment[key] = {
                'chunk_i':      ci,
                'chunk_j':      int(seg_best_cj[etype, seg_idx]),
                'edge':         int(seg_best_et[etype, seg_idx]),
                'idx':          int(seg_best_idx[etype, seg_idx]),
                'norm_cost':    float(seg_best_norm[etype, seg_idx]),
                'global_coord': (int(seg_best_grow[etype, seg_idx]),
                                 int(seg_best_gcol[etype, seg_idx])),
                'segment':      key,
            }

    # ── Backtrace (Python — cross-chunk loop; helpers are module-level) ───────
    _t0 = _start_profile_timer(profile_enabled, collect_gc=True)

    paths_per_segment = {}
    if backtrace_segments:
        for seg_key, meta in best_per_segment.items():
            path = backtrace_and_stitch(tiled_result, chunks_dict,
                                         meta['chunk_i'], meta['chunk_j'],
                                         meta['edge'], meta['idx'])
            paths_per_segment[seg_key] = {'endpoint': meta, 'path': path}

    stitched_wp = np.array([], dtype=int).reshape(0, 2)
    g_row, g_col, best_i, best_j, best_edge, best_idx = best_overall_end
    best_path = backtrace_and_stitch(tiled_result, chunks_dict, best_i, best_j, best_edge, best_idx)
    if best_path:
        stitched_wp = np.array(best_path, dtype=int)

    if _t0 is not None:
        _t0, _t1, elapsed = _elapsed_profile_seconds(_t0)
        _profile_rows.append(["backtrace_stitch", _t0, _t1, elapsed])
        _write_profile_rows(
            profile_dir,
            "stage_2_backtrace_compatible.csv",
            ["phase", "start_time", "end_time", "elapsed_seconds"],
            _profile_rows,
        )

    return {
        'D_arr':     D_arr,
        'L_arr':     L_arr,
        'edge_lens': edge_lens,
        'best_cost': float(best_norm),
        'best_end':  best_overall_end,
        'stitched_wp': stitched_wp,
        'n_row': n_row, 'n_col': n_col,
        'edge_summary': {
            'top':   {'D': top_D,   'L': top_L,   'norm': top_norm},
            'right': {'D': right_D, 'L': right_L, 'norm': right_norm},
        },
        'best_per_segment':  best_per_segment,
        'paths_per_segment': paths_per_segment,
    }



def build_tiled_metadata(chunks_dict, L, n_chunks_1, n_chunks_2, C, stage1_params=None):
    """Convert stage-1 chunk outputs into plotting/stage-2 metadata bundles.
    
    Use this adapter if downstream consumers need additional per-block fields.
    """
    L1, L2 = C.shape
    hop = L - 1  # 1-frame overlap

    # Convert chunks dictionary to list of block dicts
    blocks = []

    for (i, j), chunk_data in chunks_dict.items():
        # Extract bounds and shape
        start_1, end_1, start_2, end_2 = chunk_data['bounds']
        rows, cols = chunk_data['shape']

        # Skip chunks without a valid warping path
        wp_local = np.array(chunk_data['wp'])
        if wp_local.size == 0:
            continue

        block_dict = {
            'bi': i,
            'bj': j,
            'rows': (int(start_1), int(end_1)),
            'cols': (int(start_2), int(end_2)),
            'Ck_shape': (rows, cols),
            'S_single': chunk_data['S'],
        }

        blocks.append(block_dict)

    # Default stage1 parameters if not provided
    if stage1_params is None:
        stage1_params = {
            'steps': np.array([[1, 1], [1, 2], [2, 1]], dtype=int),
            'weights': np.array([1.5, 3.0, 3.0], dtype=float),
            'buffer': 1.0,
        }

    # Build the tiled_result dictionary
    tiled_result = {
        'C_shape': (L1, L2),
        'L_block': L,
        'hop': hop,
        'n_row': n_chunks_1,
        'n_col': n_chunks_2,
        'blocks': blocks,
        'C': C,
        'stage1_params': stage1_params,
    }

    return tiled_result


def plot_normalized_tile_edge_costs(D_arr, L_arr, edge_lens, num_chunks_1, num_chunks_2):
    """Plot normalized edge costs along global top and right boundaries.
    
    Useful for validating stage-2 terminal selection and spotting weak regions.
    """
    i_top, j_right = num_chunks_1 - 1, num_chunks_2 - 1
    global_edge_data = []
    for j in range(num_chunks_2):
        elen = int(edge_lens[i_top, j, 0])
        costs = D_arr[i_top, j, 0, 1:elen]
        lens = L_arr[i_top, j, 0, 1:elen]
        valid = np.isfinite(costs) & (lens > 0)
        norm = np.full_like(costs, np.nan)
        norm[valid] = costs[valid] / lens[valid]
        for v in norm:
            global_edge_data.append((v, "Top"))
    for i in range(num_chunks_1 - 1, -1, -1):
        elen = int(edge_lens[i, j_right, 1])
        costs = D_arr[i, j_right, 1, 1:elen]
        lens = L_arr[i, j_right, 1, 1:elen]
        valid = np.isfinite(costs) & (lens > 0)
        norm = np.full_like(costs, np.nan)
        norm[valid] = costs[valid] / lens[valid]
        for v in norm[::-1]:
            global_edge_data.append((v, "Right"))
    costs = [d[0] for d in global_edge_data]
    edge_types = [d[1] for d in global_edge_data]
    top_c = [costs[k] for k, e in enumerate(edge_types) if e == "Top"]
    right_c = [costs[k] for k, e in enumerate(edge_types) if e == "Right"]
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(-len(top_c), 0), top_c, label="Global Top Edge", color="C0", linewidth=2)
    plt.scatter(np.arange(-len(top_c), 0), top_c, color="C0", s=10)
    plt.plot(np.arange(1, len(right_c) + 1), right_c, label="Global Right Edge", color="C3", linewidth=2)
    plt.scatter(np.arange(1, len(right_c) + 1), right_c, color="C3", s=10)
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.7, label="Corner (0)")
    plt.title("Normalized Accumulated Cost (Cost / Length) along Global Edge", fontsize=14)
    plt.xlabel(f"Global Edge Position Index (Total Points: {len(costs)})", fontsize=12)
    plt.ylabel("Normalized Cost (Accumulated Cost / Path Length)", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()


def tiled_stage1_from_features(F1, F2, steps=None, weights=None, beta=0.1, L=None):
    """Convenience API: features -> cost matrix -> stage-1 tiled results.
    
    Adjust this if you want different defaults for chunk length or FlexDTW params.
    """
    if L is None:
        L = DEFAULT_CHUNK_LENGTH
    C = 1 - FlexDTW.L2norm(F1).T @ FlexDTW.L2norm(F2)
    steps = steps if steps is not None else np.array([[1, 1], [1, 2], [2, 1]])
    weights = weights if weights is not None else np.array([1.25, 3.0, 3.0])
    steps_arr = np.array(steps).reshape((-1, 2))
    stage1_params = {'steps': steps_arr, 'weights': np.array(weights), 'buffer': 1.0}
    chunks_dict, L_out, n_chunks_1, n_chunks_2 = run_flexdtw_on_tiles(C, L=L, steps=steps, weights=weights, buffer=1)
    tiled_result = build_tiled_metadata(
        chunks_dict, L_out, n_chunks_1, n_chunks_2, C, stage1_params=stage1_params
    )
    tiled_result['chunks_dict'] = chunks_dict
    return C, tiled_result


def run_stage2_from_tiled(tiled_result, C, beta=0.1, show_fig=False, top_k=1,
                            profile_dir=None, backtrace_segments=False):
    """Convenience API: run stage 2 from an existing tiled stage-1 result.
    
    Handles propagation, overlap sync, edge scan, and backtrace with optional plots.
    """
    chunks_dict = tiled_result['chunks_dict']
    L1, L2 = C.shape
    L = tiled_result['L_block']
    n_chunks_1, n_chunks_2 = tiled_result['n_row'], tiled_result['n_col']
    D_arr, L_arr, edge_lens = propagate_tile_edge_costs(
        chunks_dict, L, n_chunks_1, n_chunks_2, buffer_param=1, profile_dir=profile_dir)
    D_arr, L_arr = sync_tile_overlap_edges(D_arr, L_arr, edge_lens, n_chunks_1, n_chunks_2)
    buffer_global = min(L1, L2) * (1 - (1 - beta) * min(L1, L2) / max(L1, L2))
    r = stage2_scan_and_stitch(
        tiled_result, chunks_dict, D_arr, L_arr, edge_lens, L1, L2,
        L_block=L, buffer_stage2=buffer_global, top_k=top_k, profile_dir=profile_dir,
        backtrace_segments=backtrace_segments,
    )
    if show_fig:
        plot_normalized_tile_edge_costs(D_arr, L_arr, edge_lens, n_chunks_1, n_chunks_2)
    return r


def parflex(C, steps, weights, beta, L=None, profile_dir='Profiling_results',
            backtrace_segments=False, warn_slow_chunks=False, slow_chunk_seconds=1.0):
    """End-to-end public API for Parflex on a precomputed cost matrix.
    
    Use this for normal inference. For experimentation, modify stage behavior in
    run_flexdtw_on_tiles / propagate_tile_edge_costs / stage2_scan_and_stitch.
    """
    if L is None:
        L = DEFAULT_CHUNK_LENGTH
    L1, L2 = C.shape
    buffer_global = min(L1, L2) * (1 - (1 - beta) * min(L1, L2) / max(L1, L2))

    steps_arr = np.array(steps).reshape((-1, 2)) if hasattr(steps, '__len__') else np.array(steps)
    weights_arr = np.array(weights)
    stage1_params = {'steps': steps_arr, 'weights': weights_arr, 'buffer': 1.0}

    chunks_dict, L_out, n_chunks_1, n_chunks_2 = run_flexdtw_on_tiles(
        C, L=L, steps=steps, weights=weights, buffer=1, profile_dir=profile_dir,
        warn_slow_chunks=warn_slow_chunks, slow_chunk_seconds=slow_chunk_seconds)
    tiled_result = build_tiled_metadata(
        chunks_dict, L_out, n_chunks_1, n_chunks_2, C, stage1_params=stage1_params
    )

    D_arr, L_arr, edge_lens = propagate_tile_edge_costs(
        chunks_dict, L_out, n_chunks_1, n_chunks_2, buffer_param=1, profile_dir=profile_dir)
    D_arr, L_arr = sync_tile_overlap_edges(D_arr, L_arr, edge_lens, n_chunks_1, n_chunks_2)

    r = stage2_scan_and_stitch(
        tiled_result, chunks_dict, D_arr, L_arr, edge_lens, L1, L2,
        L_block=L, buffer_stage2=buffer_global, top_k=1, profile_dir=profile_dir,
        backtrace_segments=backtrace_segments,
    )
    wp = r["stitched_wp"]
    if wp.size > 0:
        wp = wp.T  # (N, 2) -> (2, N)
    else:
        wp = np.array([[], []], dtype=np.int64)
    return r["best_cost"], wp


# ## Try it
# 
# Set `FEATURE_1` and `FEATURE_2` to your `.npy` feature paths (or leave both `None` for a small random demo). Then run the cell below for Stage 1 + Stage 2, global FlexDTW, and the alignment plot.
# 

# In[ ]:


# # --- set paths to your .npy files, or None for a synthetic demo ---
# FEATURE_1 = "/home/ijain/ttmp/Chopin_Mazurkas_features/matching/Chopin_Op063No3/Chopin_Op063No3_Zak-1951_pid918713-20.npy"
# FEATURE_2 = "/home/ijain/ttmp/Chopin_Mazurkas_features/matching/Chopin_Op063No3/Chopin_Op063No3_Tomsic-1995_pid9190-09.npy"
# STEPS = np.array([[1, 1], [1, 2], [2, 1]], dtype=int)
# WEIGHTS = np.array([1.25, 3.0, 3.0], dtype=float)
# BETA = 0.1

# if FEATURE_1 and FEATURE_2:
#     F1 = np.load(FEATURE_1, allow_pickle=True)
#     F2 = np.load(FEATURE_2, allow_pickle=True)
# else:
#     rng = np.random.default_rng(0)
#     T1, T2, d = 120, 160, 24
#     F1 = rng.standard_normal((d, T1))
#     F2 = rng.standard_normal((d, T2))

# C, tiled_result = tiled_stage1_from_features(F1, F2, steps=STEPS, weights=WEIGHTS, beta=BETA)
# stage2_result = run_stage2_from_tiled(tiled_result, C, beta=BETA, show_fig=False)

# L1, L2 = C.shape
# buffer_flex = min(L1, L2) * (1 - (1 - BETA) * min(L1, L2) / max(L1, L2))
# _, global_flex_wp, *_ = FlexDTW.flexdtw(C, steps=STEPS, weights=WEIGHTS, buffer=buffer_flex)

# plot_alignment_with_tile_background(tiled_result, C, global_flex_wp, stage2_result)


# In[ ]:




