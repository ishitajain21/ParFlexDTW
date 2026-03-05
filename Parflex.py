#!/usr/bin/env python
# coding: utf-8

# # Parflex
# 

# ## Parameters
# 
# Chunk length and other defaults. Override in function calls (e.g. `parflex(..., L=2000)` or `align_system_parflex(..., L=2000)`).

# In[1]:


# Chunk size for tiling the cost matrix. Drives memory/speed tradeoff.
DEFAULT_CHUNK_LENGTH = 4000


# In[2]:


import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import time
import gc
import csv
import numpy as np
import math
import plotly.graph_objects as go


# ## Visualization

# In[3]:


def plot_parflex_with_chunk_S_background(tiled_result, C_global, flex_wp, parflex_res, xy=None,
                                         chunk_length=None, use_valid_edges_only=True):
    """
    Plot FlexDTW vs ParFlex paths. Background: chunk S start→edge segments (global coords).
    Foreground: global FlexDTW path, ParFlex stitched path, best-per-segment paths.

    chunk_length : used for grid lines; if None, uses tiled_result['L_block'].
    """
    blocks = tiled_result['blocks']
    L_block = tiled_result['L_block']
    L_div = chunk_length if chunk_length is not None else L_block
    hop = tiled_result['hop']
    D_chunks = parflex_res['D_chunks']
    n_row, n_col = parflex_res['n_row'], parflex_res['n_col']

    L1, L2 = C_global.shape
    base_px = 900
    max_side = max(L1, L2)
    scale = base_px / max_side
    fig_width = int(max(L2 * scale, 400))
    fig_height = int(max(L1 * scale, 400))

    fig = go.Figure()

    x_S, y_S = [], []
    INF = 1e17

    def edge_index_to_local_coords(edge, idx, rows, cols):
        """Edge 0 = top → (rows-1, idx); edge 1 = right → (idx, cols-1)."""
        return (rows - 1, idx) if edge == 0 else (idx, cols - 1)

    for b in blocks:
        i, j = b['bi'], b['bj']
        if i >= n_row or j >= n_col:
            continue

        S_single = b['S_single']
        rows, cols = b['Ck_shape']
        r_start, r_end = b['rows']
        c_start, c_end = b['cols']

        for edge in (0, 1):  # 0=top edge, 1=right edge
            edge_len = min(L_block, cols if edge == 0 else rows)

            for idx in range(edge_len):
                if use_valid_edges_only:
                    D_val = D_chunks[i][j][edge][idx]
                    if not np.isfinite(D_val) or D_val >= INF:
                        continue

                lr, lc = edge_index_to_local_coords(edge, idx, rows, cols)
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
    stitched_wp = parflex_res['stitched_wp']
    if stitched_wp.size > 0:
        fig.add_trace(
            go.Scattergl(
                x=stitched_wp[:, 1],   # cols (F2)
                y=stitched_wp[:, 0],   # rows (F1)
                mode="lines",
                name="ParFlex stitched (global best)",
                line=dict(width=6,color="rgba(247,14,14,0.5)")
            )
        )
    paths_per_segment = parflex_res['paths_per_segment']

    for (edge_name, seg_idx), info in paths_per_segment.items():
        path = np.array(info['path'], dtype=int)
        if path.size == 0:
            continue
        fig.add_trace(
            go.Scattergl(
                x=path[:, 1],   # col
                y=path[:, 0],   # row
                mode="lines",
                name=f"{edge_name} seg={seg_idx}",
                line=dict(width=3, color="rgba(0,128,255,0.5)")
            )
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
    if xy is not None:
        xy_arr = np.asarray(xy)

        if xy_arr.ndim != 2 or xy_arr.shape[1] != 2:
            raise ValueError(f"xy must have shape (N,2), got {xy_arr.shape}")

        xy_frames = xy

        fig.add_trace(
            go.Scattergl(
                x=xy_frames[:, 1],   # F2 frames
                y=xy_frames[:, 0],   # F1 frames
                mode="lines",
                name="Ground Truth",
                line=dict(width=4, dash="dash", color="rgba(0,200,0,0.9)")
            )
        )
    x_lo, x_hi = -0.5, L2 - 0.5
    y_lo, y_hi = -0.5, L1 - 0.5

    fig.update_layout(
        title="Global FlexDTW vs ParFlex (with chunk-S spiky background)",
        xaxis_title=f"F2 frames (0 … {L2-1})",
        yaxis_title=f"F1 frames (0 … {L1-1})",
        legend=dict(x=0.01, y=0.99),
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
            line=dict(width=1, dash="dot")
        ))
    for y in range(L_div, L1, L_div):
        shapes.append(dict(
            type="line",
            x0=x_lo, x1=x_hi,
            y0=y, y1=y,
            line=dict(width=1, dash="dot")
        ))
    fig.update_layout(shapes=shapes)
    fig.show()


# In[ ]:





# ## Chunking and coordinate helpers
# 
# Split cost matrix into overlapping chunks; run FlexDTW per chunk. Helpers map edge indices to local/global coords and decode FlexDTW start encoding (S).

# In[4]:


import numpy as np

 
def chunk_flexdtw(C, L, steps=None, weights=None, buffer=1, profile_dir='Profiling_results'):
    """
    Tile cost matrix C into overlapping L×L chunks (1-cell overlap), run FlexDTW on each.
    Returns chunks_dict keyed by (i, j) with 'C', 'D', 'S', 'B', 'bounds', 'hop', 'shape', etc.

    profile_dir : str or None
        Directory for per-chunk timing CSV (<profile_dir>/chunk_flexdtw.csv).
        Columns: chunk_i, chunk_j, start_time, end_time, elapsed_seconds.
        The file is created fresh on each call (write mode). Pass None to disable.
    """
    import math

    if steps is None:
        steps = [(1,1), (1,2), (2,1)]
    if weights is None:
        weights = [2, 3, 3]

    L1, L2 = C.shape
    hop = L - 1

    n_chunks_1 = math.ceil((L1 - 1) / hop)
    n_chunks_2 = math.ceil((L2 - 1) / hop)
    chunks_dict = {}

    # Open profiling CSV once; flush after every row so partial runs are not lost.
    profile_file = None
    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)
        profile_file = open(os.path.join(profile_dir, "chunk_flexdtw.csv"), "w", newline="")
        _pwriter = csv.writer(profile_file)
        _pwriter.writerow(["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"])
        profile_file.flush()

    for i in range(n_chunks_1):
        for j in range(n_chunks_2):
            # GC before timing to avoid including collection cost in the measurement.
            if profile_file is not None:
                gc.collect()
                t_start = time.perf_counter()

            start_1, start_2 = i * hop, j * hop
            end_1, end_2 = start_1 + L, start_2 + L
            if end_1 > L1:
                end_1 = L1
            if end_2 > L2:
                end_2 = L2
            C_chunk = C[int(start_1):int(end_1), int(start_2):int(end_2)]

            try:
                import FlexDTW
                best_cost, wp, D, P, B, debug = FlexDTW.flexdtw(
                    C_chunk,
                    steps=steps,
                    weights=weights,
                    buffer=1
                )
            except ImportError:
                best_cost = 0
                wp = []
                D = np.zeros_like(C_chunk)
                P = np.zeros_like(C_chunk)
                B = np.zeros_like(C_chunk)
                debug = {}

            actual_hop_1 = hop if end_1 < L1 else (L1 - start_1)
            actual_hop_2 = hop if end_2 < L2 else (L2 - start_2)

            chunks_dict[(i, j)] = {
                'C': C_chunk,
                'D': D,
                'S': P,
                'B': B,
                'debug': debug,
                'best_cost': best_cost,
                'wp': wp,
                'bounds': (start_1, end_1, start_2, end_2),
                'hop': (actual_hop_1, actual_hop_2),
                'shape': C_chunk.shape
            }

            if profile_file is not None:
                t_end = time.perf_counter()
                _pwriter.writerow([i, j, t_start, t_end, t_end - t_start])
                profile_file.flush()

    if profile_file is not None:
        profile_file.close()
    return chunks_dict, L, n_chunks_1, n_chunks_2


# In[ ]:


def edge_index_to_local_coords(edge_type, position, chunk_shape):
    """Edge 0 = top → (last_row, position); edge 1 = right → (position, last_col)."""
    if edge_type == 0:
        return chunk_shape[0] - 1, position
    return position, chunk_shape[1] - 1


def local_to_global_coords(chunk_i, chunk_j, local_row, local_col, chunks_dict):
    """Map (local_row, local_col) in chunk (i, j) to global (row, col)."""
    start_1, _, start_2, _ = chunks_dict[(chunk_i, chunk_j)]['bounds']
    return start_1 + local_row, start_2 + local_col


def decode_path_start_from_S(S_single, end_row, end_col):
    """From FlexDTW S encoding at (end_row, end_col): S>0 → start (0, S); S<0 → start (-S, 0); else (0,0)."""
    val = S_single[end_row, end_col]
    if val > 0:
        return 0, int(val)
    if val < 0:
        return abs(int(val)), 0
    return 0, 0


def _on_bottom_edge(start_row, start_col, chunk_shape):
    return start_row == 0


def _on_left_edge(start_row, start_col, chunk_shape):
    return start_col == 0


def global_to_prev_chunk_edge(global_row, global_col, prev_chunk_i, prev_chunk_j, chunks_dict, L):
    """Map global (row, col) to (edge_type, position) on the previous chunk's top or right edge."""
    start_1, _, start_2, _ = chunks_dict[(prev_chunk_i, prev_chunk_j)]['bounds']
    local_row = global_row - start_1
    local_col = global_col - start_2
    prev_chunk_shape = chunks_dict[(prev_chunk_i, prev_chunk_j)]['D'].shape
    if local_row == prev_chunk_shape[0] - 1:
        return 0, local_col
    if local_col == prev_chunk_shape[1] - 1:
        return 1, local_row
    raise ValueError(f"({local_row}, {local_col}) not on edge of previous chunk")


def initialize_chunks(chunks_dict, num_chunks_1, num_chunks_2, L, profile_dir='Profiling_results'):
    """
    Initialize D_chunks and L_chunks for the first row and first column of the chunk grid.

    Ensures edge continuity between adjacent chunks: position 0 of each chunk's top/right
    edge is seeded from the final position of the previous chunk's corresponding edge.

    Parameters
    ----------
    chunks_dict : dict
        Per-chunk data produced by chunk_flexdtw.
    num_chunks_1, num_chunks_2 : int
        Grid dimensions.
    L : int
        Nominal chunk size.
    profile_dir : str or None
        If set, write per-chunk timing to <profile_dir>/initialize_chunks.csv.
        Columns: chunk_i, chunk_j, start_time, end_time, elapsed_seconds.
        Pass None to disable.

    Returns
    -------
    D_chunks, L_chunks : list[list[dict[int, list]]]
        Indexed as [chunk_row][chunk_col][edge_type][position].
    """
    D_chunks = [[{0: [], 1: []} for _ in range(num_chunks_2)] for _ in range(num_chunks_1)]
    L_chunks = [[{0: [], 1: []} for _ in range(num_chunks_2)] for _ in range(num_chunks_1)]

    def _edge_length(chunk_shape, edge_type):
        # top edge (0) spans columns; right edge (1) spans rows
        return chunk_shape[1] if edge_type == 0 else chunk_shape[0]

    profile_file = None
    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)
        profile_file = open(os.path.join(profile_dir, "initialize_chunks.csv"), "w", newline="")
        _pwriter = csv.writer(profile_file)
        _pwriter.writerow(["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"])
        profile_file.flush()

    # ------------------------------------------------------------------
    # Chunk (0, 0): seed directly from FlexDTW results; no predecessor.
    # ------------------------------------------------------------------
    if profile_file is not None:
        gc.collect()
        _t0 = time.perf_counter()

    D_single_00 = chunks_dict[(0, 0)]['D']
    S_single_00 = chunks_dict[(0, 0)]['S']

    for edge_type in range(2):
        edge_len = _edge_length(D_single_00.shape, edge_type)
        D_chunks[0][0][edge_type] = [np.inf] * edge_len
        L_chunks[0][0][edge_type] = [np.inf] * edge_len

        for position in range(edge_len):
            local_row, local_col = edge_index_to_local_coords(edge_type, position, D_single_00.shape)
            if local_row < D_single_00.shape[0] and local_col < D_single_00.shape[1]:
                start_row, start_col = decode_path_start_from_S(S_single_00, local_row, local_col)
                # Accept paths whose start lies on the bottom or left boundary of this chunk.
                if _on_bottom_edge(start_row, start_col, D_single_00.shape) or \
                   _on_left_edge(start_row, start_col, D_single_00.shape):
                    D_chunks[0][0][edge_type][position] = D_single_00[local_row, local_col]
                    path_length = abs(local_row - start_row) + abs(local_col - start_col)
                    L_chunks[0][0][edge_type][position] = path_length

    if profile_file is not None:
        _t1 = time.perf_counter()
        _pwriter.writerow([0, 0, _t0, _t1, _t1 - _t0])
        profile_file.flush()

    # ------------------------------------------------------------------
    # First row: chunks (0, j) for j = 1, 2, ...
    # Position 0 on the top edge is inherited from the previous chunk's
    # rightmost top-edge value to guarantee continuity.
    # ------------------------------------------------------------------
    for j in range(1, num_chunks_2):
        if (0, j) not in chunks_dict:
            continue
        if profile_file is not None:
            gc.collect()
            _t0 = time.perf_counter()

        D_single = chunks_dict[(0, j)]['D']
        S_single = chunks_dict[(0, j)]['S']
        C_chunk = chunks_dict[(0, j)].get('C', None)

        # Seed position 0 of the top edge from the previous chunk's top-edge tail.
        D_chunks[0][j][0] = [D_chunks[0][j-1][0][-1]] + [np.inf] * (_edge_length(D_single.shape, 0) - 1)
        L_chunks[0][j][0] = [L_chunks[0][j-1][0][-1]] + [np.inf] * (_edge_length(D_single.shape, 0) - 1)

        edge_len_right = _edge_length(D_single.shape, 1)
        D_chunks[0][j][1] = [np.inf] * edge_len_right
        L_chunks[0][j][1] = [np.inf] * edge_len_right

        for edge_type in range(2):
            edge_len = _edge_length(D_single.shape, edge_type)
            for position in range(1, edge_len):
                if edge_type == 0 and position == 0:
                    continue  # already seeded above

                local_row, local_col = edge_index_to_local_coords(edge_type, position, D_single.shape)
                if local_row >= D_single.shape[0] or local_col >= D_single.shape[1]:
                    continue

                start_row, start_col = decode_path_start_from_S(S_single, local_row, local_col)

                if _on_bottom_edge(start_row, start_col, D_single.shape):
                    # Path started at the bottom of this chunk; cost is self-contained.
                    D_chunks[0][j][edge_type][position] = D_single[local_row, local_col]
                    path_length = abs(local_row - start_row) + abs(local_col - start_col)
                    L_chunks[0][j][edge_type][position] = path_length

                elif _on_left_edge(start_row, start_col, D_single.shape):
                    # Path entered from the left; look up accumulated cost in chunk (0, j-1).
                    global_start_row, global_start_col = local_to_global_coords(
                        0, j, start_row, start_col, chunks_dict
                    )
                    prev_edge_type, prev_position = global_to_prev_chunk_edge(
                        global_start_row, global_start_col, 0, j - 1, chunks_dict, L
                    )
                    prev_edge_len = len(D_chunks[0][j-1][prev_edge_type])
                    if prev_position >= prev_edge_len:
                        continue
                    prev_cost = D_chunks[0][j-1][prev_edge_type][prev_position]
                    if np.isfinite(prev_cost):
                        # Subtract the overlap cell to avoid counting it twice.
                        overlap_cost = C_chunk[start_row, 0] if C_chunk is not None else 0
                        D_chunks[0][j][edge_type][position] = D_single[local_row, local_col] + prev_cost - overlap_cost
                        prev_length = L_chunks[0][j-1][prev_edge_type][prev_position]
                        curr_length = abs(local_row - start_row) + abs(local_col - start_col)
                        L_chunks[0][j][edge_type][position] = prev_length + curr_length

        if profile_file is not None:
            _t1 = time.perf_counter()
            _pwriter.writerow([0, j, _t0, _t1, _t1 - _t0])
            profile_file.flush()

    # ------------------------------------------------------------------
    # First column: chunks (i, 0) for i = 1, 2, ...
    # Position 0 on the right edge is inherited from the previous chunk's
    # bottommost right-edge value to guarantee continuity.
    # ------------------------------------------------------------------
    for i in range(1, num_chunks_1):
        if (i, 0) not in chunks_dict:
            continue
        if profile_file is not None:
            gc.collect()
            _t0 = time.perf_counter()

        D_single = chunks_dict[(i, 0)]['D']
        S_single = chunks_dict[(i, 0)]['S']
        C_chunk = chunks_dict[(i, 0)].get('C', None)

        # Seed position 0 of the right edge from the previous chunk's right-edge tail.
        D_chunks[i][0][1] = [D_chunks[i-1][0][1][-1]] + [np.inf] * (_edge_length(D_single.shape, 1) - 1)
        L_chunks[i][0][1] = [L_chunks[i-1][0][1][-1]] + [np.inf] * (_edge_length(D_single.shape, 1) - 1)

        edge_len_top = _edge_length(D_single.shape, 0)
        D_chunks[i][0][0] = [np.inf] * edge_len_top
        L_chunks[i][0][0] = [np.inf] * edge_len_top

        for edge_type in range(2):
            edge_len = _edge_length(D_single.shape, edge_type)
            for position in range(1, edge_len):
                if edge_type == 1 and position == 0:
                    continue  # already seeded above

                local_row, local_col = edge_index_to_local_coords(edge_type, position, D_single.shape)
                if local_row >= D_single.shape[0] or local_col >= D_single.shape[1]:
                    continue

                start_row, start_col = decode_path_start_from_S(S_single, local_row, local_col)

                if _on_left_edge(start_row, start_col, D_single.shape):
                    # Path started at the left of this chunk; cost is self-contained.
                    D_chunks[i][0][edge_type][position] = D_single[local_row, local_col]
                    path_length = abs(local_row - start_row) + abs(local_col - start_col)
                    L_chunks[i][0][edge_type][position] = path_length

                elif _on_bottom_edge(start_row, start_col, D_single.shape):
                    # Path entered from below; look up accumulated cost in chunk (i-1, 0).
                    global_start_row, global_start_col = local_to_global_coords(
                        i, 0, start_row, start_col, chunks_dict
                    )
                    prev_edge_type, prev_position = global_to_prev_chunk_edge(
                        global_start_row, global_start_col, i - 1, 0, chunks_dict, L
                    )
                    prev_edge_len = len(D_chunks[i-1][0][prev_edge_type])
                    if prev_position >= prev_edge_len:
                        continue
                    prev_cost = D_chunks[i-1][0][prev_edge_type][prev_position]
                    if np.isfinite(prev_cost):
                        overlap_cost = C_chunk[0, start_col] if C_chunk is not None else 0
                        D_chunks[i][0][edge_type][position] = D_single[local_row, local_col] + prev_cost - overlap_cost
                        prev_length = L_chunks[i-1][0][prev_edge_type][prev_position]
                        curr_length = abs(local_row - start_row) + abs(local_col - start_col)
                        L_chunks[i][0][edge_type][position] = prev_length + curr_length

        if profile_file is not None:
            _t1 = time.perf_counter()
            _pwriter.writerow([i, 0, _t0, _t1, _t1 - _t0])
            profile_file.flush()

    if profile_file is not None:
        profile_file.close()
    return D_chunks, L_chunks


def dp_fill_chunks(chunks_dict, D_chunks, L_chunks, num_chunks_1, num_chunks_2, L,
                   profile_dir='Profiling_results'):
    """
    Fill D_chunks and L_chunks for all interior chunks (i>0, j>0) using DP.

    For position 0 on each edge, the value is inherited from the adjacent already-filled
    chunk to ensure continuity at overlap cells.  For all other positions the path start
    is decoded from S, the predecessor chunk's accumulated cost is looked up, and the
    overlap cell is subtracted to avoid double-counting.

    Parameters
    ----------
    chunks_dict : dict
        Per-chunk data from chunk_flexdtw.
    D_chunks, L_chunks : list[list[dict[int, list]]]
        Partially filled edge-cost/length arrays (first row/col already done).
    num_chunks_1, num_chunks_2 : int
        Grid dimensions.
    L : int
        Nominal chunk size.
    profile_dir : str or None
        If set, write per-chunk timing to <profile_dir>/dp_fill_chunks.csv.
        Columns: chunk_i, chunk_j, start_time, end_time, elapsed_seconds.
        Pass None to disable.
    """
    def _edge_length(chunk_shape, edge_type):
        return chunk_shape[1] if edge_type == 0 else chunk_shape[0]

    profile_file = None
    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)
        profile_file = open(os.path.join(profile_dir, "dp_fill_chunks.csv"), "w", newline="")
        _pwriter = csv.writer(profile_file)
        _pwriter.writerow(["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"])
        profile_file.flush()

    for i in range(num_chunks_1):
        for j in range(num_chunks_2):
            # First row and first column were handled by initialize_chunks.
            if i == 0 or j == 0:
                continue

            if profile_file is not None:
                gc.collect()
                _t0 = time.perf_counter()

            D_single = chunks_dict[(i, j)]['D']
            S_single = chunks_dict[(i, j)]['S']
            C_chunk = chunks_dict[(i, j)].get('C', None)

            for edge_type in range(2):
                edge_len = _edge_length(D_single.shape, edge_type)
                D_chunks[i][j][edge_type] = [np.inf] * edge_len
                L_chunks[i][j][edge_type] = [np.inf] * edge_len

                for position in range(edge_len):

                    # Position 0: inherit from the adjacent chunk for continuity at the
                    # 1-cell overlap boundary rather than re-computing.
                    if position == 0:
                        inherited = False

                        if edge_type == 0 and j > 0:
                            # Top edge pos 0 ← left chunk's rightmost top-edge value.
                            left_len = len(D_chunks[i][j-1][0])
                            if left_len > 0:
                                lc = D_chunks[i][j-1][0][left_len - 1]
                                ll = L_chunks[i][j-1][0][left_len - 1]
                                if np.isfinite(lc):
                                    D_chunks[i][j][edge_type][0] = lc
                                    L_chunks[i][j][edge_type][0] = ll
                                    inherited = True

                        elif edge_type == 1 and i > 0:
                            # Right edge pos 0 ← top chunk's bottommost right-edge value.
                            top_len = len(D_chunks[i-1][j][1])
                            if top_len > 0:
                                tc = D_chunks[i-1][j][1][top_len - 1]
                                tl = L_chunks[i-1][j][1][top_len - 1]
                                if np.isfinite(tc):
                                    D_chunks[i][j][edge_type][0] = tc
                                    L_chunks[i][j][edge_type][0] = tl
                                    inherited = True

                        if inherited:
                            continue

                    local_row, local_col = edge_index_to_local_coords(edge_type, position, D_single.shape)
                    if local_row >= D_single.shape[0] or local_col >= D_single.shape[1]:
                        continue

                    start_row, start_col = decode_path_start_from_S(S_single, local_row, local_col)

                    if _on_bottom_edge(start_row, start_col, D_single.shape):
                        prev_i, prev_j = i - 1, j
                    elif _on_left_edge(start_row, start_col, D_single.shape):
                        prev_i, prev_j = i, j - 1
                    else:
                        # Path started inside this chunk; no cross-chunk cost needed.
                        D_chunks[i][j][edge_type][position] = D_single[local_row, local_col]
                        L_chunks[i][j][edge_type][position] = (
                            abs(local_row - start_row) + abs(local_col - start_col)
                        )
                        continue

                    global_start_row, global_start_col = local_to_global_coords(
                        i, j, start_row, start_col, chunks_dict
                    )
                    prev_edge_type, prev_position = global_to_prev_chunk_edge(
                        global_start_row, global_start_col, prev_i, prev_j, chunks_dict, L
                    )

                    prev_edge_len = len(D_chunks[prev_i][prev_j][prev_edge_type])
                    if prev_position >= prev_edge_len:
                        continue

                    prev_cost = D_chunks[prev_i][prev_j][prev_edge_type][prev_position]
                    prev_length = L_chunks[prev_i][prev_j][prev_edge_type][prev_position]
                    if not np.isfinite(prev_cost):
                        continue

                    # Subtract the shared boundary cell to avoid counting it twice.
                    first_cell_cost = C_chunk[start_row, start_col] if C_chunk is not None else 0
                    D_chunks[i][j][edge_type][position] = (
                        prev_cost + D_single[local_row, local_col] - first_cell_cost
                    )
                    L_chunks[i][j][edge_type][position] = (
                        prev_length + abs(local_row - start_row) + abs(local_col - start_col)
                    )

            if profile_file is not None:
                _t1 = time.perf_counter()
                _pwriter.writerow([i, j, _t0, _t1, _t1 - _t0])
                profile_file.flush()

    if profile_file is not None:
        profile_file.close()
    return D_chunks, L_chunks
def chunked_flexdtw(chunks_dict, L, num_chunks_1, num_chunks_2, buffer_param=0.1,
                    profile_dir='Profiling_results'):
    """Propagate cost/length on chunk edges: init first row/col, then DP fill. Returns (D_chunks, L_chunks)."""
    D_chunks, L_chunks = initialize_chunks(
        chunks_dict, num_chunks_1, num_chunks_2, L, profile_dir=profile_dir
    )
    D_chunks, L_chunks = dp_fill_chunks(
        chunks_dict, D_chunks, L_chunks, num_chunks_1, num_chunks_2, L, profile_dir=profile_dir
    )
    return D_chunks, L_chunks


# ## Tiled result and Stage 2
# 
# Convert chunk dict to tiled format for plotting; then run Stage 2 backtrace (best path from chunk edges).

# In[6]:


import numpy as np

def convert_chunks_to_tiled_result(chunks_dict, L, n_chunks_1, n_chunks_2, C, stage1_params=None):
    """Convert chunk_flexdtw output to tiled format for plot_parflex_with_chunk_S_background and Stage 2."""
    L1, L2 = C.shape
    hop = L - 1  # Your code uses 1-frame overlap
    
    # Convert chunks dictionary to list of block dicts
    blocks = []
    
    for (i, j), chunk_data in chunks_dict.items():
        # Extract bounds
        start_1, end_1, start_2, end_2 = chunk_data['bounds']
        rows, cols = chunk_data['shape']
        
        # Get the warping path (ensure it's in the right format)
        wp_local = np.array(chunk_data['wp'])
        if wp_local.size == 0:
            continue

        # Ensure wp_local is (N, 2)
        if wp_local.ndim == 2 and wp_local.shape[0] == 2:
            wp_local = wp_local.T
        
        # Calculate raw cost and path length
        C_chunk = chunk_data['C']
        raw_cost_blk = float(C_chunk[wp_local[:, 0], wp_local[:, 1]].sum())
        path_len_blk = int(np.abs(np.diff(wp_local, axis=0)).sum(axis=1).sum() + 1)
        
        # Map local path to global coordinates
        wp_global = np.column_stack([
            wp_local[:, 0] + start_1,
            wp_local[:, 1] + start_2
        ])
        
        block_dict = {
            'bi': i,
            'bj': j,
            'rows': (int(start_1), int(end_1)),
            'cols': (int(start_2), int(end_2)),
            'bounds': (start_1, end_1, start_2, end_2),
            'Ck_shape': (rows, cols),
            'shape': (rows, cols),
            'best_cost': float(chunk_data['best_cost']),
            'wp_global': wp_global,
            'wp_local': wp_local.copy(),
            'raw_cost': raw_cost_blk,
            'path_len': path_len_blk,
            'D_single': chunk_data['D'],
            'B_single': chunk_data['B'],
            'B': chunk_data['B'],
            'S_single': chunk_data['S'],
            'S': chunk_data['S'],
        }
        
        blocks.append(block_dict)
    
    # Default stage1 parameters if not provided
    if stage1_params is None:
        stage1_params = {
            'steps': np.array([[1, 1], [1, 2], [2, 1]], dtype=int),
            'weights': np.array([1.5, 3.0, 3.0], dtype=float),
            'buffer': 1.0
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
        'stage1_params': stage1_params
    }
    
    return tiled_result


# In[7]:


import numpy as np

def stage_2_backtrace_compatible(tiled_result, all_blocks, D_chunks, L_chunks, L1, L2,
                                  L_block, buffer_stage2=200, top_k=1,
                                  profile_dir='Profiling_results'):
    """
    Scan top/right global edges for the best normalised cost endpoint, then backtrace
    and stitch a path across chunks.

    profile_dir : str or None
        If set, write three phase rows to <profile_dir>/stage_2_backtrace_compatible.csv.
        Phases: top_scan, right_scan, backtrace_stitch.
        Columns: phase, start_time, end_time, elapsed_seconds.
        Pass None to disable.
    """
    
    INF = 1e9
    n_row = len(D_chunks)
    n_col = len(D_chunks[0]) if n_row > 0 else 0
    
    def edge_to_local(edge, idx, rows, cols):
        """Convert edge representation to local coordinates."""
        if edge == 0:  # top
            return rows - 1, idx
        else:  # right
            return idx, cols - 1
    
    def backtrace_within_chunk(B_single, steps, start_r, start_c, end_r, end_c, 
                                global_r_offset, global_c_offset):
        """
        Backtrace from (end_r, end_c) back to (start_r, start_c) within a chunk.
        Returns path in GLOBAL coordinates, in END → START order.
        """
        path = []
        r, c = end_r, end_c
        max_iters = B_single.shape[0] * B_single.shape[1]
        iters = 0

        while iters < max_iters:
            path.append((r + global_r_offset, c + global_c_offset))

            if r == start_r and c == start_c:
                if len(path) == 0 or path[-1] != (start_r + global_r_offset, start_c + global_c_offset):
                    path.append((start_r + global_r_offset, start_c + global_c_offset))
                break

            step_idx = int(B_single[r, c])
            if step_idx < 0 or step_idx >= len(steps):
                if (r != start_r or c != start_c):
                    path.append((start_r + global_r_offset, start_c + global_c_offset))
                break

            dr, dc = steps[step_idx]
            prev_r = r - dr
            prev_c = c - dc

            if (prev_r < 0 or prev_c < 0 or 
                prev_r >= B_single.shape[0] or prev_c >= B_single.shape[1]):
                if (r != start_r or c != start_c):
                    path.append((start_r + global_r_offset, start_c + global_c_offset))
                break

            r, c = prev_r, prev_c
            iters += 1

        if iters >= max_iters:
            
            if (r != start_r or c != start_c):
                path.append((start_r + global_r_offset, start_c + global_c_offset))

        return path

    def backtrace_and_stitch(start_i, start_j, start_edge, start_idx):
        """
        Backtrack from a specific edge endpoint and stitch across chunks.
        Returns the GLOBAL path in START → END order.
        """
        path = []
        cur_i, cur_j, cur_edge, cur_idx = start_i, start_j, start_edge, start_idx
        steps = tiled_result['stage1_params']['steps']
        visited_chunks = set()
        stop_reason = None
        iteration = 0

        while True:
            iteration += 1
            chunk_key = (cur_i, cur_j, cur_edge, cur_idx)

            if chunk_key in visited_chunks:
                stop_reason = f"loop at chunk {chunk_key}"
                break
            visited_chunks.add(chunk_key)

            if (cur_i, cur_j) not in all_blocks:
                stop_reason = f"missing chunk ({cur_i},{cur_j})"
                break

            b = all_blocks[(cur_i, cur_j)]
            rows, cols = b['shape']
            r_start, r_end, c_start, c_end = b['bounds']

            end_r, end_c = edge_to_local(cur_edge, cur_idx, rows, cols)
            S_val = b['S'][end_r, end_c]

            if S_val >= 0:
                start_r = 0
                start_c = int(S_val)
            else:
                start_r = int(-S_val)
                start_c = 0

            chunk_path = backtrace_within_chunk(
                b['B'], steps, start_r, start_c, end_r, end_c, r_start, c_start
            )

            for pt in chunk_path:
                if not path or path[-1] != pt:
                    path.append(pt)

            g_start_row = r_start + start_r
            g_start_col = c_start + start_c

            if g_start_row == 0:
                stop_reason = f"hit bottom edge at global row 0, col {g_start_col}"
                break
            if g_start_col == 0:
                stop_reason = f"hit left edge at global row {g_start_row}, col 0"
                break

            # If we land exactly on the chunk corner (local (0,0)), we explicitly
            # jump to the bottom-left chunk and continue from its top-edge corner.
            # This avoids ambiguity about whether we should move "up" or "left".
            corner_landed = (start_r == 0 and start_c == 0)

            if corner_landed:
                prev_i, prev_j = cur_i - 1, cur_j - 1
                prev_edge = 0  # top edge
            elif S_val >= 0:
                prev_i, prev_j = cur_i - 1, cur_j
                prev_edge = 0
            else:
                prev_i, prev_j = cur_i, cur_j - 1
                prev_edge = 1
             

            if prev_i < 0 or prev_j < 0:
                stop_reason = f"corner transition out of bounds to ({prev_i},{prev_j})"
                break

            if (prev_i, prev_j) not in all_blocks:
                stop_reason = f"previous chunk ({prev_i},{prev_j}) missing"
                break

            prev_b = all_blocks[(prev_i, prev_j)]
            prev_r_start, prev_r_end, prev_c_start, prev_c_end = prev_b['bounds']
            prev_rows, prev_cols = prev_b['shape']

            if corner_landed:
                # Bottom-left chunk's top-edge corner (its top-right) touches current chunk's corner.
                prev_idx = prev_cols - 1
                max_prev_idx = prev_cols
            else:
                prev_lr = g_start_row - prev_r_start
                prev_lc = g_start_col - prev_c_start

                prev_idx = prev_lc if prev_edge == 0 else prev_lr
                max_prev_idx = prev_cols if prev_edge == 0 else prev_rows

            if prev_idx < 0 or prev_idx >= max_prev_idx:
                stop_reason = f"prev_idx out of bounds({prev_idx}/{max_prev_idx})"
                break

            cur_i, cur_j, cur_edge, cur_idx = prev_i, prev_j, prev_edge, prev_idx

            if iteration > 100:
                stop_reason = f"iteration limit ({iteration})"
                break

        path = path[::-1]
        return path

    def global_to_chunk_edge(g_row, g_col):
        """
        Given a GLOBAL (g_row, g_col) on the top or right edge of C_global,
        find the corresponding chunk, edge, and local edge index.
        """
        for (bi, bj), b in all_blocks.items():
            r_start, r_end, c_start, c_end = b['bounds']

            if r_start <= g_row < r_end and c_start <= g_col < c_end:
                rows, cols = b['shape']
                local_r = g_row - r_start
                local_c = g_col - c_start

                if local_r == rows - 1:
                    return (bi, bj, 0, local_c)
                elif local_c == cols - 1:
                    return (bi, bj, 1, local_r)
        return None

    top_D = np.full(L2, np.nan)
    top_L = np.zeros(L2)
    right_D = np.full(L1, np.nan)
    right_L = np.zeros(L1)

    best_overall_cost = float('inf')
    best_overall_end = None
    best_per_segment = {}
    candidate_endpoints = []

    # Open profiling CSV for the three timed phases: top_scan, right_scan, backtrace_stitch.
    _st2_profile_file = None
    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)
        _st2_profile_file = open(
            os.path.join(profile_dir, "stage_2_backtrace_compatible.csv"), "w", newline=""
        )
        _st2_writer = csv.writer(_st2_profile_file)
        _st2_writer.writerow(["phase", "start_time", "end_time", "elapsed_seconds"])
        _st2_profile_file.flush()

    # ------------------------------------------------------------------
    # Scan TOP edge (last row of the global cost matrix)
    # ------------------------------------------------------------------
    if _st2_profile_file is not None:
        gc.collect()
        _st2_t0 = time.perf_counter()

    valid_endpoints_found = 0
    buf = int(buffer_stage2)

    for g_col in range(L2):
        g_row = L1 - 1
       
        if buf > 0 and g_col < buf: 
            continue

        chunk_info = global_to_chunk_edge(g_row, g_col)
        if chunk_info is None: 
            continue

        chunk_i, chunk_j, edge, idx = chunk_info
        
        # Access D_chunks and L_chunks with flexible structure
        if chunk_i >= len(D_chunks) or chunk_j >= len(D_chunks[chunk_i]): 
            continue
        if edge not in D_chunks[chunk_i][chunk_j]: 
            continue
        if idx >= len(D_chunks[chunk_i][chunk_j][edge]): 
            continue
            
        D_val = D_chunks[chunk_i][chunk_j][edge][idx]
        L_val = L_chunks[chunk_i][chunk_j][edge][idx]
         

        if D_val >= INF or L_val <= 0: 
            continue
        valid_endpoints_found += 1
        norm_cost = D_val / L_val
        

        top_D[g_col] = D_val
        top_L[g_col] = L_val
        
        if norm_cost < best_overall_cost:
            best_overall_cost = norm_cost
            best_overall_end = (g_row, g_col, chunk_i, chunk_j, edge, idx)

        seg_idx = int(g_col // L_block)
        key = ('top', seg_idx)

        prev = best_per_segment.get(key)
        if (prev is None) or (norm_cost < prev['norm_cost']):
            best_per_segment[key] = {
                'chunk_i': chunk_i,
                'chunk_j': chunk_j,
                'edge': edge,
                'idx': idx,
                'norm_cost': norm_cost,
                'global_coord': (g_row, g_col),
                'segment': key,
            }
            
        candidate_endpoints.append({
            'g_row': g_row,
            'g_col': g_col,
            'chunk_i': chunk_i,
            'chunk_j': chunk_j,
            'edge': edge,
            'idx': idx,
            'norm_cost': norm_cost
        })

    if _st2_profile_file is not None:
        _st2_t1 = time.perf_counter()
        _st2_writer.writerow(["top_scan", _st2_t0, _st2_t1, _st2_t1 - _st2_t0])
        _st2_profile_file.flush()

    # ------------------------------------------------------------------
    # Scan RIGHT edge (last column of the global cost matrix)
    # ------------------------------------------------------------------
    if _st2_profile_file is not None:
        gc.collect()
        _st2_t0 = time.perf_counter()

    valid_endpoints_found = 0

    for g_row in range(L1):
        g_col = L2 - 1

        if buf > 0 and g_row < buf:
            continue

        chunk_info = global_to_chunk_edge(g_row, g_col)
        if chunk_info is None:
            continue

        chunk_i, chunk_j, edge, idx = chunk_info
        
        # Access with flexible structure
        if chunk_i >= len(D_chunks) or chunk_j >= len(D_chunks[chunk_i]):
            continue
        if edge not in D_chunks[chunk_i][chunk_j]:
            continue
        if idx >= len(D_chunks[chunk_i][chunk_j][edge]):
            continue
            
        D_val = D_chunks[chunk_i][chunk_j][edge][idx]
        L_val = L_chunks[chunk_i][chunk_j][edge][idx]

        valid_endpoints_found += 1
        norm_cost = D_val / L_val
        right_D[g_row] = D_val
        right_L[g_row] = L_val
        if norm_cost < best_overall_cost:
            best_overall_cost = norm_cost
            best_overall_end = (g_row, g_col, chunk_i, chunk_j, edge, idx)
        seg_idx = int(g_row // L_block)
        key = ('right', seg_idx)
        prev = best_per_segment.get(key)
        if (prev is None) or (norm_cost < prev['norm_cost']):
            best_per_segment[key] = {
                'chunk_i': chunk_i,
                'chunk_j': chunk_j,
                'edge': edge,
                'idx': idx,
                'norm_cost': norm_cost,
                'global_coord': (g_row, g_col),
                'segment': key,
            }
        candidate_endpoints.append({
            'g_row': g_row,
            'g_col': g_col,
            'chunk_i': chunk_i,
            'chunk_j': chunk_j,
            'edge': edge,
            'idx': idx,
            'norm_cost': norm_cost
        })

    if _st2_profile_file is not None:
        _st2_t1 = time.perf_counter()
        _st2_writer.writerow(["right_scan", _st2_t0, _st2_t1, _st2_t1 - _st2_t0])
        _st2_profile_file.flush()

    if not candidate_endpoints:
        raise ValueError("Stage 2: No valid endpoint found on global top/right edges.")

    candidate_endpoints_sorted = sorted(candidate_endpoints, key=lambda c: c['norm_cost'])

    best = candidate_endpoints_sorted[0]
    best_overall_cost = best['norm_cost'] 
    best_overall_end = (
        best['g_row'], best['g_col'], best['chunk_i'], 
        best['chunk_j'], best['edge'], best['idx']
    )

    g_row, g_col, best_i, best_j, best_edge, best_idx = best_overall_end

    # Compute normalized arrays
    top_norm = np.full(L2, np.nan)
    right_norm = np.full(L1, np.nan)

    top_mask = (top_L > 0) & np.isfinite(top_D)
    right_mask = (right_L > 0) & np.isfinite(right_D)

    top_norm[top_mask] = top_D[top_mask] / top_L[top_mask]
    right_norm[right_mask] = right_D[right_mask] / right_L[right_mask]

    # ------------------------------------------------------------------
    # Backtrace and stitch all per-segment best paths + the global best.
    # ------------------------------------------------------------------
    if _st2_profile_file is not None:
        gc.collect()
        _st2_t0 = time.perf_counter()

    paths_per_segment = {}

    for seg_key, meta in best_per_segment.items():
        ci = meta['chunk_i']
        cj = meta['chunk_j']
        ce = meta['edge']
        cidx = meta['idx']
        path = backtrace_and_stitch(ci, cj, ce, cidx)
        paths_per_segment[seg_key] = {
            'endpoint': meta,
            'path': path
        }

    # Backtrace the single global best endpoint to produce the stitched warping path.
    stitched_wp = np.array([], dtype=int).reshape(0, 2)
    if best_overall_end is not None:
        g_row, g_col, best_i, best_j, best_edge, best_idx = best_overall_end
        best_path = backtrace_and_stitch(best_i, best_j, best_edge, best_idx)
        if best_path:
            stitched_wp = np.array(best_path, dtype=int)

    if _st2_profile_file is not None:
        _st2_t1 = time.perf_counter()
        _st2_writer.writerow(["backtrace_stitch", _st2_t0, _st2_t1, _st2_t1 - _st2_t0])
        _st2_profile_file.flush()
        _st2_profile_file.close()

    return {
        'D_chunks': D_chunks,
        'L_chunks': L_chunks,
        'best_cost': best_overall_cost,
        'best_end': best_overall_end,
        'stitched_wp': stitched_wp,
        'n_row': n_row,
        'n_col': n_col,
        'edge_summary': {
            'top':   {'D': top_D,   'L': top_L,   'norm': top_norm},
            'right': {'D': right_D, 'L': right_L, 'norm': right_norm},
        },
        'best_per_segment': best_per_segment,
        'paths_per_segment': paths_per_segment,
    }


# ## Sync and public API
# 
# Sync overlapping edge values; then **align_system_parflex** (Stage 1) and **parflex_2a** (Stage 2). **parflex** runs both in one call.

# In[8]:


import matplotlib.pyplot as plt

def plot_normalized_global_edge_cost(D_chunks, L_chunks, num_chunks_1, num_chunks_2):
    """
    Plot normalized accumulated cost (Cost / Length) along the global top and right edges.
    Traversal order: Top-Left -> Top-Right -> Bottom-Right.
    """
    i_top = num_chunks_1 - 1
    j_right = num_chunks_2 - 1
    global_edge_data = []
    for j in range(num_chunks_2):
        costs = np.array(D_chunks[i_top][j][0][1:])
        lengths = np.array(L_chunks[i_top][j][0][1:])
        valid_indices = np.isfinite(costs) & (lengths > 0)
        normalized_costs = np.full_like(costs, np.nan, dtype=float)
        normalized_costs[valid_indices] = costs[valid_indices] / lengths[valid_indices]
        for cost in normalized_costs:
            global_edge_data.append((cost, 'Top'))
    for i in range(num_chunks_1 - 1, -1, -1):
        costs = np.array(D_chunks[i][j_right][1][1:])
        lengths = np.array(L_chunks[i][j_right][1][1:])
        valid_indices = np.isfinite(costs) & (lengths > 0)
        normalized_costs = np.full_like(costs, np.nan, dtype=float)
        normalized_costs[valid_indices] = costs[valid_indices] / lengths[valid_indices]
        for cost in normalized_costs[::-1]:
            global_edge_data.append((cost, 'Right'))
    costs = [d[0] for d in global_edge_data]
    edge_types = [d[1] for d in global_edge_data]
    top_costs = [costs[i] for i, e in enumerate(edge_types) if e == "Top"]
    right_costs = [costs[i] for i, e in enumerate(edge_types) if e == "Right"]
    top_x = np.arange(-len(top_costs), 0)
    right_x = np.arange(1, len(right_costs) + 1)
    plt.figure(figsize=(15, 6))
    plt.plot(top_x, top_costs, label='Global Top Edge', color='C0', linewidth=2)
    plt.scatter(top_x, top_costs, color='C0', s=10)
    plt.plot(right_x, right_costs, label='Global Right Edge', color='C3', linewidth=2)
    plt.scatter(right_x, right_costs, color='C3', s=10)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7, label='Corner (0)')
    plt.title("Normalized Accumulated Cost (Cost / Length) along Global Edge", fontsize=14)
    plt.xlabel(f"Global Edge Position Index (Total Points: {len(costs)})", fontsize=12)
    plt.ylabel("Normalized Cost (Accumulated Cost / Path Length)", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()


# In[9]:


def sync_overlapping_positions(D_chunks, L_chunks, num_chunks_1, num_chunks_2):
    """Ensure 1-cell overlaps between chunks share the same D/L values."""
    sync_count = 0

    # Horizontal overlaps: chunk[i][j] top[-1] <-> chunk[i][j+1] top[0]
    for i in range(num_chunks_1):
        for j in range(num_chunks_2 - 1):
            if len(D_chunks[i][j][0]) > 0 and len(D_chunks[i][j + 1][0]) > 0:
                last_pos = len(D_chunks[i][j][0]) - 1
                D_left = D_chunks[i][j][0][last_pos]
                D_right = D_chunks[i][j + 1][0][0]

                if np.isfinite(D_left) and not np.isfinite(D_right):
                    D_chunks[i][j + 1][0][0] = D_left
                    L_chunks[i][j + 1][0][0] = L_chunks[i][j][0][last_pos]
                    sync_count += 1
                elif not np.isfinite(D_left) and np.isfinite(D_right):
                    D_chunks[i][j][0][last_pos] = D_right
                    L_chunks[i][j][0][last_pos] = L_chunks[i][j + 1][0][0]
                    sync_count += 1

    # Vertical overlaps: chunk[i][j] right[-1] <-> chunk[i+1][j] right[0]
    for i in range(num_chunks_1 - 1):
        for j in range(num_chunks_2):
            if len(D_chunks[i][j][1]) > 0 and len(D_chunks[i + 1][j][1]) > 0:
                last_pos = len(D_chunks[i][j][1]) - 1
                D_bottom = D_chunks[i][j][1][last_pos]
                D_top = D_chunks[i + 1][j][1][0]

                if np.isfinite(D_bottom) and not np.isfinite(D_top):
                    D_chunks[i + 1][j][1][0] = D_bottom
                    L_chunks[i + 1][j][1][0] = L_chunks[i][j][1][last_pos]
                    sync_count += 1
                elif not np.isfinite(D_bottom) and np.isfinite(D_top):
                    D_chunks[i][j][1][last_pos] = D_top
                    L_chunks[i][j][1][last_pos] = L_chunks[i + 1][j][1][0]
                    sync_count += 1

    # Within-chunk corner: chunk[i][j] top[-1] must equal chunk[i][j] right[-1]
    for i in range(num_chunks_1):
        for j in range(num_chunks_2):
            if len(D_chunks[i][j][0]) > 0 and len(D_chunks[i][j][1]) > 0:
                D_top_corner = D_chunks[i][j][0][-1]
                D_right_corner = D_chunks[i][j][1][-1]

                if np.isfinite(D_top_corner) and not np.isfinite(D_right_corner):
                    D_chunks[i][j][1][-1] = D_top_corner
                    L_chunks[i][j][1][-1] = L_chunks[i][j][0][-1]
                    sync_count += 1
                elif not np.isfinite(D_top_corner) and np.isfinite(D_right_corner):
                    D_chunks[i][j][0][-1] = D_right_corner
                    L_chunks[i][j][0][-1] = L_chunks[i][j][1][-1]
                    sync_count += 1
                elif np.isfinite(D_top_corner) and np.isfinite(D_right_corner):
                    if not np.isclose(D_top_corner, D_right_corner):
                        # Prefer top corner value for determinism
                        D_chunks[i][j][1][-1] = D_top_corner
                        L_chunks[i][j][1][-1] = L_chunks[i][j][0][-1]
                        sync_count += 1

    return D_chunks, L_chunks


def align_system_parflex(F1, F2, steps=None, weights=None, beta=0.1, L=None):
    """Stage 1 only: build C, chunk with FlexDTW, return (C, tiled_result). L defaults to DEFAULT_CHUNK_LENGTH."""
    if L is None:
        L = DEFAULT_CHUNK_LENGTH
    import FlexDTW
    C = 1 - FlexDTW.L2norm(F1).T @ FlexDTW.L2norm(F2)
    steps = steps if steps is not None else np.array([[1, 1], [1, 2], [2, 1]])
    weights = weights if weights is not None else np.array([1.25, 3.0, 3.0])
    steps_arr = np.array(steps).reshape((-1, 2))
    stage1_params = {'steps': steps_arr, 'weights': np.array(weights), 'buffer': 1.0}
    chunks_dict, L_out, n_chunks_1, n_chunks_2 = chunk_flexdtw(C, L=L, steps=steps, weights=weights, buffer=1)
    tiled_result = convert_chunks_to_tiled_result(
        chunks_dict, L_out, n_chunks_1, n_chunks_2, C, stage1_params=stage1_params
    )
    tiled_result['chunks_dict'] = chunks_dict
    return C, tiled_result


def parflex_2a(tiled_result, C, beta=0.1, show_fig=False, top_k=1,
               profile_dir='Profiling_results'):
    """
    Run Parflex Stage 2: propagate costs across chunk edges and backtrace the best path.

    profile_dir : str or None
        Passed to chunked_flexdtw and stage_2_backtrace_compatible for per-chunk/per-phase
        CSV profiling.  Pass None to disable.  Each call overwrites the CSVs in that
        directory, so use a unique directory per run when calling in a loop.
    """
    chunks_dict = tiled_result['chunks_dict']
    L1, L2 = C.shape
    L = tiled_result['L_block']
    n_chunks_1, n_chunks_2 = tiled_result['n_row'], tiled_result['n_col']
    D_chunks, L_chunks = chunked_flexdtw(
        chunks_dict, L, n_chunks_1, n_chunks_2, buffer_param=1, profile_dir=profile_dir
    )
    D_chunks, L_chunks = sync_overlapping_positions(D_chunks, L_chunks, n_chunks_1, n_chunks_2)
    buffer_global = min(L1, L2) * (1 - (1 - beta) * min(L1, L2) / max(L1, L2))
    r = stage_2_backtrace_compatible(
        tiled_result, chunks_dict, D_chunks, L_chunks, L1, L2,
        L_block=L, buffer_stage2=buffer_global, top_k=top_k, profile_dir=profile_dir
    )
    if show_fig:
        plot_normalized_global_edge_cost(D_chunks, L_chunks, n_chunks_1, n_chunks_2)
    return r



# In[10]:


def parflex(C, steps, weights, beta, L=None, profile_dir='Profiling_results'):
    """
    Run full Parflex pipeline on cost matrix C.

    Returns (best_cost, wp) where wp has shape (2, N).
    L defaults to DEFAULT_CHUNK_LENGTH.

    profile_dir : str or None
        Directory for per-chunk/per-phase profiling CSVs (stage 1 and stage 2).
        Each call writes fresh CSVs, so pass a unique directory per run when calling
        in a loop (e.g. profile_dir='runs/P1000_L500_trial3').
        Pass None to disable all profiling.
    """
    if L is None:
        L = DEFAULT_CHUNK_LENGTH
    L1, L2 = C.shape
    buffer_global = min(L1, L2) * (1 - (1 - beta) * min(L1, L2) / max(L1, L2))

    steps_arr = np.array(steps).reshape((-1, 2)) if hasattr(steps, '__len__') else np.array(steps)
    weights_arr = np.array(weights)
    stage1_params = {'steps': steps_arr, 'weights': weights_arr, 'buffer': 1.0}

    # Stage 1: run FlexDTW on every chunk.
    chunks_dict, L_out, n_chunks_1, n_chunks_2 = chunk_flexdtw(
        C, L=L, steps=steps, weights=weights, buffer=1, profile_dir=profile_dir
    )
    tiled_result = convert_chunks_to_tiled_result(
        chunks_dict, L_out, n_chunks_1, n_chunks_2, C, stage1_params=stage1_params
    )

    # Stage 2: propagate edge costs and backtrace.
    D_chunks, L_chunks = chunked_flexdtw(
        chunks_dict, L_out, n_chunks_1, n_chunks_2, buffer_param=1, profile_dir=profile_dir
    )
    D_chunks, L_chunks = sync_overlapping_positions(D_chunks, L_chunks, n_chunks_1, n_chunks_2)

    r = stage_2_backtrace_compatible(
        tiled_result, chunks_dict, D_chunks, L_chunks, L1, L2,
        L_block=L, buffer_stage2=buffer_global, top_k=1, profile_dir=profile_dir
    )
    wp = r["stitched_wp"]
    if wp.size > 0:
        wp = wp.T  # (N, 2) → (2, N)
    else:
        wp = np.array([[], []], dtype=np.int64)
    return r["best_cost"], wp



# In[ ]:




