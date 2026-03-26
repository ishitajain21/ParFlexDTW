# %% [markdown]
# # Sparse Parallel FlexDTW

# %% [markdown]
# ## Parameters
# 
# Chunk length and other defaults. Override in function calls (e.g. `parflex(..., L=2000)` or `align_system_sparse_parflex(..., L=2000)`).
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# %%
# Chunk size for tiling the cost matrix. Drives memory/speed tradeoff.
DEFAULT_CHUNK_LENGTH = 4000

# %%
import numpy as np
import math
import plotly.graph_objects as go
import csv

# %% [markdown]
# ## Visualization

# %%
def plot_parflex_with_chunk_S_background(
    tiled_result,
    C_global,
    flex_wp,
    parflex_res,
    ground_truth_=None,
    xy=None,
    chunk_length=None,
    use_valid_edges_only=True,
    extra_paths=None,
    extra_labels=None,
):
    """
    Plot FlexDTW vs ParFlex paths. Background: chunk S start→edge segments (global coords).
    Foreground: global FlexDTW path, ParFlex stitched path, best-per-segment paths.

    ground_truth_ : tuple of two .beat annotation file paths (beat_path_1, beat_path_2).
                    Beat times (seconds) are converted to frames (sr=22050, hop=512)
                    and plotted as a dashed green line. Overrides xy if both are given.
    xy            : (N,2) array of ground-truth correspondences already in frame space.
    chunk_length  : used for grid lines; if None, uses tiled_result['L_block'].
    """
    SR, HOP = 22050, 512

    if ground_truth_ is not None:
        beat_path_1, beat_path_2 = ground_truth_
        import eval_tools
        gt_seconds = eval_tools.getGroundTruthTimestamps(beat_path_1, beat_path_2)
        xy = gt_seconds.copy()
        xy[:, 0] = xy[:, 0] * SR / HOP
        xy[:, 1] = xy[:, 1] * SR / HOP

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
                line=dict(width=3, color="rgba(0,128,255,0.5)"),
                showlegend=False
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

    # Optional additional warped paths (e.g., DTW baselines)
    if extra_paths is not None:
        labeled_paths = []
        if isinstance(extra_paths, dict):
            for label, path in extra_paths.items():
                labeled_paths.append((str(label), path))
        else:
            paths_list = list(extra_paths)
            if extra_labels is not None:
                for label, path in zip(extra_labels, paths_list):
                    labeled_paths.append((str(label), path))
            else:
                for i, path in enumerate(paths_list):
                    labeled_paths.append((f"Path {i+1}", path))

        for i, (label, path) in enumerate(labeled_paths):
            path_arr = np.asarray(path)
            if path_arr.ndim != 2:
                continue
            if path_arr.shape[1] == 2:
                r = path_arr[:, 0]
                c = path_arr[:, 1]
            elif path_arr.shape[0] == 2:
                r = path_arr[0, :]
                c = path_arr[1, :]
            else:
                continue

            fig.add_trace(
                go.Scattergl(
                    x=c,
                    y=r,
                    mode="lines",
                    name=str(label),
                    line=dict(width=3)
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
        title="Global FlexDTW vs ParFlex (with chunk-S spiky background)",
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


# %%

def _start_points_from_S(S):
    """
    From chunk S matrix (P in FlexDTW: signed encoding of path start per cell), compute
    unique start points for the top and right edges of the chunk.
    Only reads S on the last row and last column (no full-matrix scan).

    Returns
    -------
    starts_bottom_edge : set of int
        Column indices where paths ending on the top edge started on the bottom edge (row 0).
    starts_left_edge : set of int
        Row indices where paths ending on the right edge started on the left edge (col 0).
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

# %% [markdown]
# ## Chunking and coordinate helpers
# 
# Split cost matrix into overlapping chunks; run FlexDTW per chunk. Helpers map edge indices to local/global coords and decode FlexDTW start encoding (S).

# %%


# %%
import time
import os
import gc
import csv
import FlexDTW


def chunk_flexdtw(C, L, steps=None, weights=None, buffer=1, profile_dir='results_profiling'):
    """
    Break cost matrix C into overlapping chunks and run FlexDTW on each.
    Ensures exactly 1-cell overlap between chunks. Last chunks may be smaller
    than L×L to maintain the 1-cell overlap constraint.
    
    Now stores flexible hop information for each chunk to handle boundary truncation.
    
    Parameters:
    -----------
    C : ndarray
        Cost matrix of shape (L1, L2)
    L : int
        Standard chunk size
    steps : list, optional
        Step patterns for DTW
    weights : list, optional
        Weights for each step pattern
    buffer : int
        Buffer parameter for FlexDTW
    profile_dir : str or path-like, optional
        If set, profile each chunk and write to <profile_dir>/chunk_flexdtw.csv.
        Columns: chunk_i, chunk_j, start_time, end_time, elapsed_seconds.

    Returns:
    --------
    chunks_dict : dict
        Dictionary with chunk data, including 'bounds' and 'hop' for each chunk
    L : int
        Chunk size
    n_chunks_1, n_chunks_2 : int
        Number of chunks in each dimension
    """
    import math
    
    if steps is None:
        steps = [(1,1), (1,2), (2,1)]
    if weights is None:
        weights = [2, 3, 3]
    
    L1, L2 = C.shape
    hop = L - 1  # Standard overlap of 1
    
    # Compute number of chunks along each axis
    n_chunks_1 = math.ceil((L1 - 1) / hop)
    n_chunks_2 = math.ceil((L2 - 1) / hop)
    
    chunks_dict = {}

    # Profiling: open CSV and write header once if requested
    profile_file = None
    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)
        profile_file = open(os.path.join(profile_dir, "chunk_flexdtw.csv"), "w", newline="")
        writer = csv.writer(profile_file)
        writer.writerow(["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"])
        profile_file.flush()

    for i in range(n_chunks_1):
        for j in range(n_chunks_2):
            if profile_file is not None:
                gc.collect()
                t_start = time.perf_counter()
            # Standard start positions with hop size
            start_1 = i * hop
            start_2 = j * hop
            
            # Standard end positions
            end_1 = start_1 + L
            end_2 = start_2 + L
            
            # For boundary chunks, don't shift start - just truncate end
            # This ensures we always have exactly 1-frame overlap
            if end_1 > L1:
                end_1 = L1
            
            if end_2 > L2:
                end_2 = L2
            
            # Extract chunk
            C_chunk = C[int(start_1):int(end_1), int(start_2):int(end_2)]

            # Profiling: garbage collect then time this chunk (single-run, good practice)
            
            try:
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
            

            # Calculate actual hop for this chunk
            # For boundary chunks, the hop to the next chunk may be different
            actual_hop_1 = hop if end_1 < L1 else (L1 - start_1)
            actual_hop_2 = hop if end_2 < L2 else (L2 - start_2)
            
            
            starts_bot_edge, starts_left_edge = _start_points_from_S(P)

            chunks_dict[(i, j)] = {
                'C': C_chunk,
                'D': D,
                'S': P,  # Start positions (signed encoding)
                'B': B,  # Backpointer matrix
                'debug': debug,
                'best_cost': best_cost,
                'wp': wp,
                'bounds': (start_1, end_1, start_2, end_2),
                'hop': (actual_hop_1, actual_hop_2),  # Store flexible hop sizes
                'shape': C_chunk.shape,
                'starts_bot_edge': starts_bot_edge,
                'starts_left_edge': starts_left_edge,
            }
            if profile_file is not None:
                t_end = time.perf_counter()
                writer.writerow([i, j, t_start, t_end, t_end - t_start])
                profile_file.flush()

    if profile_file is not None:
        profile_file.close()
    return chunks_dict, L, n_chunks_1, n_chunks_2


# %%
import time
import os
import gc
import csv
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


def _valid_positions(chunks_dict, i, j, edge_type, edge_len, num_chunks_1, num_chunks_2):
    """
    Returns sorted list of valid positions to compute on chunk (i,j)'s edges.
    Valid positions are determined by what neighboring chunks need as inputs:
    
    - edge_type=0 (top/bottom edge, indexed by col): the chunk BELOW (i+1, j) 
      will start paths at these column positions -> use starts_bot_edge of (i+1, j)
    - edge_type=1 (right edge, indexed by row): the chunk to the RIGHT (i, j+1)
      will start paths at these row positions -> use starts_left_edge of (i, j+1)
    
    Always includes 0 and edge_len-1 for continuity.
    """
    starts = set()

    if edge_type == 0:
        # look at the chunk above
        above = (i + 1, j)
        if above in chunks_dict:
            starts = chunks_dict[above].get('starts_bot_edge', set())
    else:
        # Chunk to the right needs these row positions as path starts
        right = (i, j + 1)
        if right in chunks_dict:
            starts = chunks_dict[right].get('starts_left_edge', set())
    

    positions = set(starts) | {0, edge_len - 1}
    positions = {p for p in positions if 0 <= p < edge_len}
    return sorted(positions)


def initialize_chunks(chunks_dict, num_chunks_1, num_chunks_2, L, profile_dir='results_profiling'):
    """
    Initialize D_chunks and L_chunks for the first row and first column.
    Uses flexible data structure to accommodate non-square boundary chunks.
    Ensures edge continuity between adjacent chunks.
    Only iterates over valid start positions (sparse) plus endpoints for continuity.
    
    Parameters:
    -----------
    chunks_dict : dict
        Dictionary containing chunk data (including 'hop' for each chunk)
    num_chunks_1, num_chunks_2 : int, int
        Number of chunks in each dimension
    L : int
        Chunk size
    profile_dir : str or path-like, optional
        If set, profile each (i,j) and write to <profile_dir>/initialize_chunks.csv.

    Returns:
    --------
    D_chunks, L_chunks : list of list of dicts of lists
        Indexed as [chunk_row][chunk_col][edge_type][position]
        Where position is a Python list that can vary in length
    """
    D_chunks = [[{0: [], 1: []} for _ in range(num_chunks_2)] for _ in range(num_chunks_1)]
    L_chunks = [[{0: [], 1: []} for _ in range(num_chunks_2)] for _ in range(num_chunks_1)]

    profile_file = None
    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)
        profile_file = open(os.path.join(profile_dir, "initialize_chunks.csv"), "w", newline="")
        _profile_writer = csv.writer(profile_file)
        _profile_writer.writerow(["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"])
        profile_file.flush()

    def _edge_length(chunk_shape, edge_type):
        return chunk_shape[1] if edge_type == 0 else chunk_shape[0]

    # ==================================================================================
    # Initialize chunk (0, 0)
    # ==================================================================================
    if profile_file is not None:
        gc.collect()
        _t0 = time.perf_counter()
    D_single_00 = chunks_dict[(0, 0)]['D']
    S_single_00 = chunks_dict[(0, 0)]['S']

    for edge_type in range(2):
        edge_len = _edge_length(D_single_00.shape, edge_type)
        D_chunks[0][0][edge_type] = [np.inf] * edge_len
        L_chunks[0][0][edge_type] = [np.inf] * edge_len
        valid_positions = _valid_positions(chunks_dict, 0, 0, edge_type, edge_len, num_chunks_1, num_chunks_2)

        if edge_type == 0 and num_chunks_1 == 1:
            valid_positions = set(range(edge_len))
        if edge_type == 1 and num_chunks_2 == 1:
            valid_positions = set(range(edge_len))
        
        
        for position in valid_positions:
            local_row, local_col = edge_index_to_local_coords(edge_type, position, D_single_00.shape)

            if local_row < D_single_00.shape[0] and local_col < D_single_00.shape[1]:
                start_row, start_col = decode_path_start_from_S(S_single_00, local_row, local_col)

                if _on_bottom_edge(start_row, start_col, D_single_00.shape) or \
                   _on_left_edge(start_row, start_col, D_single_00.shape):
                    D_chunks[0][0][edge_type][position] = D_single_00[local_row, local_col]
                    path_length = abs(local_row - start_row) + abs(local_col - start_col)
                    L_chunks[0][0][edge_type][position] = path_length
    if profile_file is not None:
        _t1 = time.perf_counter()
        _profile_writer.writerow([0, 0, _t0, _t1, _t1 - _t0])
        profile_file.flush()

    # ==================================================================================
    # CASE 1: Initialize first row - chunks (0, j) for j = 1, 2, ...
    # ==================================================================================
    for j in range(1, num_chunks_2):
        if (0, j) not in chunks_dict:
            continue
        if profile_file is not None:
            gc.collect()
            _t0 = time.perf_counter()
        D_single = chunks_dict[(0, j)]['D']
        S_single = chunks_dict[(0, j)]['S']
        C_chunk = chunks_dict[(0, j)].get('C', None)

        # Edge continuity: 0th index on top edge = previous chunk's last index on top edge
        D_chunks[0][j][0] = [D_chunks[0][j-1][0][-1]] + [np.inf] * (_edge_length(D_single.shape, 0) - 1)
        L_chunks[0][j][0] = [L_chunks[0][j-1][0][-1]] + [np.inf] * (_edge_length(D_single.shape, 0) - 1)

        edge_len_right = _edge_length(D_single.shape, 1)
        D_chunks[0][j][1] = [np.inf] * edge_len_right
        L_chunks[0][j][1] = [np.inf] * edge_len_right

        for edge_type in range(2):
            edge_len = _edge_length(D_single.shape, edge_type)
            valid_positions_first_row = _valid_positions(chunks_dict, 0, j, edge_type, edge_len, num_chunks_1, num_chunks_2)

            if edge_type == 1 and j == num_chunks_2 - 1:
                valid_positions_first_row = set(range(edge_len))
            if edge_type == 0 and num_chunks_1 == 1:
                valid_positions_first_row = set(range(edge_len))
            for position in valid_positions_first_row:
                # Skip position 0 for top edge - already set by continuity
                if edge_type == 0 and position == 0:
                    continue

                local_row, local_col = edge_index_to_local_coords(edge_type, position, D_single.shape)

                if local_row >= D_single.shape[0] or local_col >= D_single.shape[1]:
                    continue

                start_row, start_col = decode_path_start_from_S(S_single, local_row, local_col)

                if _on_bottom_edge(start_row, start_col, D_single.shape):
                    D_chunks[0][j][edge_type][position] = D_single[local_row, local_col]
                    path_length = abs(local_row - start_row) + abs(local_col - start_col)
                    L_chunks[0][j][edge_type][position] = path_length

                elif _on_left_edge(start_row, start_col, D_single.shape):
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
                        overlap_cost = C_chunk[start_row, 0] if C_chunk is not None else 0
                        D_chunks[0][j][edge_type][position] = D_single[local_row, local_col] + prev_cost - overlap_cost
                        prev_length = L_chunks[0][j-1][prev_edge_type][prev_position]
                        curr_length = abs(local_row - start_row) + abs(local_col - start_col)
                        L_chunks[0][j][edge_type][position] = prev_length + curr_length
        if profile_file is not None:
            _t1 = time.perf_counter()
            _profile_writer.writerow([0, j, _t0, _t1, _t1 - _t0])
            profile_file.flush()

    # ==================================================================================
    # CASE 2: Initialize first column - chunks (i, 0) for i = 1, 2, ...
    # ==================================================================================
    for i in range(1, num_chunks_1):
        if (i, 0) not in chunks_dict:
            continue
        if profile_file is not None:
            gc.collect()
            _t0 = time.perf_counter()
        D_single = chunks_dict[(i, 0)]['D']
        S_single = chunks_dict[(i, 0)]['S']
        C_chunk = chunks_dict[(i, 0)].get('C', None)

        # Edge continuity: 0th index on right edge = previous chunk's last index on right edge
        D_chunks[i][0][1] = [D_chunks[i-1][0][1][-1]] + [np.inf] * (_edge_length(D_single.shape, 1) - 1)
        L_chunks[i][0][1] = [L_chunks[i-1][0][1][-1]] + [np.inf] * (_edge_length(D_single.shape, 1) - 1)

        edge_len_top = _edge_length(D_single.shape, 0)
        D_chunks[i][0][0] = [np.inf] * edge_len_top
        L_chunks[i][0][0] = [np.inf] * edge_len_top

        for edge_type in range(2):
            edge_len = _edge_length(D_single.shape, edge_type)
            valid_positions_first_column = _valid_positions(chunks_dict, i, 0, edge_type, edge_len, num_chunks_1, num_chunks_2) 
            if edge_type == 0 and i == num_chunks_1 - 1:
                valid_positions_first_column = set(range(edge_len))
            if edge_type == 1 and num_chunks_2 == 1:
                valid_positions_first_column = set(range(edge_len))
            for position in valid_positions_first_column:
                # Skip position 0 for right edge - already set by continuity
                if edge_type == 1 and position == 0:
                    continue

                local_row, local_col = edge_index_to_local_coords(edge_type, position, D_single.shape)

                if local_row >= D_single.shape[0] or local_col >= D_single.shape[1]:
                    continue

                start_row, start_col = decode_path_start_from_S(S_single, local_row, local_col)

                if _on_left_edge(start_row, start_col, D_single.shape):
                    D_chunks[i][0][edge_type][position] = D_single[local_row, local_col]
                    path_length = abs(local_row - start_row) + abs(local_col - start_col)
                    L_chunks[i][0][edge_type][position] = path_length

                elif _on_bottom_edge(start_row, start_col, D_single.shape):
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
            _profile_writer.writerow([i, 0, _t0, _t1, _t1 - _t0])
            profile_file.flush()

    if profile_file is not None:
        profile_file.close()
    return D_chunks, L_chunks


def dp_fill_chunks(chunks_dict, D_chunks, L_chunks, num_chunks_1, num_chunks_2, L, profile_dir='results_profiling'):
    """
    Fill in D_chunks and L_chunks for all interior chunks using dynamic programming.
    Uses flexible hop sizes from chunks_dict.
    Only iterates over valid start positions (sparse) plus endpoints for continuity.

    Parameters:
    -----------
    chunks_dict : dict
        Dictionary containing chunk data (including flexible 'hop' per chunk)
    D_chunks, L_chunks : list of list of dicts of lists
        Chunk-level cost and length tensors (partially filled)
    num_chunks_1, num_chunks_2 : int, int
        Number of chunks in each dimension
    L : int
        Standard chunk size (for reference)
    profile_dir : str or path-like, optional
        If set, profile each (i,j) and write to <profile_dir>/dp_fill_chunks.csv.
    """
    def _edge_length(chunk_shape, edge_type):
        return chunk_shape[1] if edge_type == 0 else chunk_shape[0]
    _dp_profile_file = None
    
 
    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)
        _dp_profile_file = open(os.path.join(profile_dir, "dp_fill_chunks.csv"), "w", newline="")
        _dp_writer = csv.writer(_dp_profile_file)
        _dp_writer.writerow(["chunk_i", "chunk_j", "start_time", "end_time", "elapsed_seconds"])
        _dp_profile_file.flush()
    for i in range(num_chunks_1):
        for j in range(num_chunks_2):
            if i == 0 or j == 0:
                continue
            if _dp_profile_file is not None:
                gc.collect()
                _t0 = time.perf_counter()
            D_single = chunks_dict[(i, j)]['D']
            S_single = chunks_dict[(i, j)]['S']
            C_chunk = chunks_dict[(i, j)].get('C', None)

            for edge_type in range(2):
                edge_len = _edge_length(D_single.shape, edge_type)
                D_chunks[i][j][edge_type] = [np.inf] * edge_len
                L_chunks[i][j][edge_type] = [np.inf] * edge_len
                
                # check if it's in the last column or last row:
                if edge_type == 1 and j == num_chunks_2 - 1:
                    # valid_positions_dp = all the points on the edge
                    valid_positions_dp = set(range(edge_len))
                elif edge_type == 0 and i == num_chunks_1 - 1:
                    # valid_positions_dp = all the points on the edge
                    valid_positions_dp = set(range(edge_len))
                else:    
                    valid_positions_dp = _valid_positions(chunks_dict, i, j, edge_type, edge_len, num_chunks_1, num_chunks_2)
                for position in valid_positions_dp:

                    # Position 0: inherit from adjacent chunk for continuity
                    if position == 0:
                        inherited = False

                        if edge_type == 0 and j > 0:
                            left_edge_len = len(D_chunks[i][j-1][0])
                            if left_edge_len > 0:
                                rightmost_pos = left_edge_len - 1
                                left_cost = D_chunks[i][j-1][0][rightmost_pos]
                                left_length = L_chunks[i][j-1][0][rightmost_pos]
                                if np.isfinite(left_cost):
                                    D_chunks[i][j][edge_type][position] = left_cost
                                    L_chunks[i][j][edge_type][position] = left_length
                                    inherited = True

                        elif edge_type == 1 and i > 0:
                            top_edge_len = len(D_chunks[i-1][j][1])
                            if top_edge_len > 0:
                                bottommost_pos = top_edge_len - 1
                                top_cost = D_chunks[i-1][j][1][bottommost_pos]
                                top_length = L_chunks[i-1][j][1][bottommost_pos]
                                if np.isfinite(top_cost):
                                    D_chunks[i][j][edge_type][position] = top_cost
                                    L_chunks[i][j][edge_type][position] = top_length
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
                        # Path started within this chunk
                        D_chunks[i][j][edge_type][position] = D_single[local_row, local_col]
                        path_length = abs(local_row - start_row) + abs(local_col - start_col)
                        L_chunks[i][j][edge_type][position] = path_length
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

                    first_cell_cost = C_chunk[start_row, start_col] if C_chunk is not None else 0
                    curr_cost_contribution = D_single[local_row, local_col] - first_cell_cost
                    curr_length = abs(local_row - start_row) + abs(local_col - start_col)

                    D_chunks[i][j][edge_type][position] = prev_cost + curr_cost_contribution
                    L_chunks[i][j][edge_type][position] = prev_length + curr_length
            if _dp_profile_file is not None:
                _t1 = time.perf_counter()
                _dp_writer.writerow([i, j, _t0, _t1, _t1 - _t0])
                _dp_profile_file.flush()

    if _dp_profile_file is not None:
        _dp_profile_file.close()
    return D_chunks, L_chunks
def chunked_flexdtw(chunks_dict, L, num_chunks_1, num_chunks_2, buffer_param=0.1):
    """Propagate cost/length on chunk edges: init first row/col, then DP fill. Returns (D_chunks, L_chunks)."""
    D_chunks, L_chunks = initialize_chunks(chunks_dict, num_chunks_1, num_chunks_2, L) 
    
    D_chunks, L_chunks = dp_fill_chunks(chunks_dict, D_chunks, L_chunks, 
                                        num_chunks_1, num_chunks_2, L)
    return D_chunks, L_chunks


# %% [markdown]
# ## Tiled result and Stage 2
# 
# Convert chunk dict to tiled format for plotting; then run Stage 2 backtrace (best path from chunk edges).

# %%
import numpy as np

def convert_chunks_to_tiled_result(chunks_dict, L, n_chunks_1, n_chunks_2, C, stage1_params=None):
    """Convert chunk_flexdtw output to tiled format for plot_parflex_with_chunk_S_background and Stage 2.

    The `blocks` entries are intentionally kept minimal; they only store the fields
    required by downstream functionality (currently visualization).
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


# %%
import numpy as np

def convert_chunks_to_tiled_result(chunks_dict, L, n_chunks_1, n_chunks_2, C, stage1_params=None):
    """Convert chunk_flexdtw output to tiled format for plot_parflex_with_chunk_S_background and Stage 2.

    The `blocks` entries are intentionally kept minimal; they only store the fields
    required by downstream functionality (currently visualization).
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


# %%
import time
import os
import gc
import csv
import numpy as np

def stage_2_backtrace_compatible(tiled_result, all_blocks, D_chunks, L_chunks, L1, L2,
                                  L_block, buffer_stage2=200, top_k=1, profile_dir='results_profiling'):
    """Scan top/right edges for best normalized cost; backtrace and stitch path across chunks. Returns dict with stitched_wp, best_cost, paths_per_segment, etc.
    profile_dir: if set, write to <profile_dir>/stage_2_backtrace_compatible.csv. Rows: top_scan, right_scan, backtrace_stitch (whole backtrace+stitch timed once). Columns: phase, start_time, end_time, elapsed_seconds."""
    
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

    _st2_profile_file = None
    if profile_dir is not None:
        os.makedirs(profile_dir, exist_ok=True)
        _st2_profile_file = open(os.path.join(profile_dir, "stage_2_backtrace_compatible.csv"), "w", newline="")
        _st2_writer = csv.writer(_st2_profile_file)
        _st2_writer.writerow(["phase", "start_time", "end_time", "elapsed_seconds"])
        _st2_profile_file.flush()

    # Scan TOP edge
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

    # Scan RIGHT edge
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

    # Backtrace per segment
    paths_per_segment = {}
 

    for seg_key, meta in best_per_segment.items():
        ci = meta['chunk_i']
        cj = meta['chunk_j']
        ce = meta['edge']
        cidx = meta['idx']
        norm_cost = meta['norm_cost']
        g_row, g_col = meta['global_coord']

        path = backtrace_and_stitch(ci, cj, ce, cidx)

        paths_per_segment[seg_key] = {
            'endpoint': meta,
            'path': path
        }

    # Get best overall path
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


# %% [markdown]
# ## Sync and public API
# 
# Sync overlapping edge values; then **align_system_sparse_parflex** (Stage 1) and **sparse_parflex_2a** (Stage 2). **parflex** runs both in one call.

# %%
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

# %%
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


def align_system_sparse_parflex(F1, F2, steps=None, weights=None, beta=0.1, L=None):
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


def sparse_parflex_2a(tiled_result, C, beta=0.1, show_fig=False, top_k=1):
    """Run Parflex Stage 2: propagate costs and backtrace. Returns dict with stitched_wp, best_cost, etc."""
    chunks_dict = tiled_result['chunks_dict']
    L1, L2 = C.shape
    L = tiled_result['L_block']
    n_chunks_1, n_chunks_2 = tiled_result['n_row'], tiled_result['n_col']
    D_chunks, L_chunks = chunked_flexdtw(chunks_dict, L, n_chunks_1, n_chunks_2, buffer_param=1)
    D_chunks, L_chunks = sync_overlapping_positions(D_chunks, L_chunks, n_chunks_1, n_chunks_2)
    buffer_global = min(L1, L2) * (1 - (1 - beta) * min(L1, L2) / max(L1, L2))
    r = stage_2_backtrace_compatible(
        tiled_result, chunks_dict, D_chunks, L_chunks, L1, L2,
        L_block=L, buffer_stage2=buffer_global, top_k=top_k
    )
    if show_fig:
        plot_normalized_global_edge_cost(D_chunks, L_chunks, n_chunks_1, n_chunks_2)
    return r



# %%

def parflex(C, steps, weights, beta, L=None):
    """Run Parflex on cost matrix C. Returns (best_cost, wp) with wp shape (2, N). L defaults to DEFAULT_CHUNK_LENGTH."""
    if L is None:
        L = DEFAULT_CHUNK_LENGTH
    L1, L2 = C.shape
    buffer_global = min(L1, L2) * (1 - (1 - beta) * min(L1, L2) / max(L1, L2))

    steps_arr = np.array(steps).reshape((-1, 2)) if hasattr(steps, '__len__') else np.array(steps)
    weights_arr = np.array(weights)
    stage1_params = {'steps': steps_arr, 'weights': weights_arr, 'buffer': 1.0}

    chunks_dict, L_out, n_chunks_1, n_chunks_2 = chunk_flexdtw(C, L=L, steps=steps, weights=weights, buffer=1)
    tiled_result = convert_chunks_to_tiled_result(
        chunks_dict, L_out, n_chunks_1, n_chunks_2, C, stage1_params=stage1_params
    )

    D_chunks, L_chunks = chunked_flexdtw(chunks_dict, L_out, n_chunks_1, n_chunks_2, buffer_param=1)
    D_chunks, L_chunks = sync_overlapping_positions(D_chunks, L_chunks, n_chunks_1, n_chunks_2)

    r = stage_2_backtrace_compatible(
        tiled_result, chunks_dict, D_chunks, L_chunks, L1, L2,
        L_block=L, buffer_stage2=buffer_global, top_k=1
    )
    wp = r["stitched_wp"]
    if wp.size > 0:
        wp = wp.T  # (N, 2) -> (2, N)
    else:
        wp = np.array([[], []], dtype=np.int64)
    return r["best_cost"], wp



# %% [markdown]
# ## Testing
# 
# Run the cells above so `align_system_sparse_parflex`, `sparse_parflex_2a`, and `plot_parflex_with_chunk_S_background` are defined. Then load two feature matrices `F1`, `F2` (e.g. from `.npy` files), run Stage 1 + Stage 2, and optionally plot. For batch runs over multiple chunk sizes **L**, see **Optional: L sweep** below.

# %%

import os
beat_base = '/home/ijain/ttmp/Chopin_Mazurkas/annotations_beat'
import Parflex
import builtins
if not hasattr(builtins, 'get_ipython'):
    _magic_stub = type('_MagicStub', (), {
        'run_line_magic': staticmethod(lambda *a, **kw: None),
        'run_cell_magic': staticmethod(lambda *a, **kw: None),
    })()
    builtins.get_ipython = lambda: _magic_stub
import import_ipynb
import DTW
folder = "Chopin_Op063No3"
benchmark_dir_full = "/home/ijain/ttmp/Chopin_Mazurkas_features/matching"


steps = {'dtw1': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'dtw2': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'dtw3': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'subseqdtw1': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'subseqdtw2': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'subseqdtw3': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'nwtw': 0, # transitions are specified in NWTW algorithm
        'flexdtw': np.array([1,1,1,2,2,1]).reshape((-1,2)), 
        'parflex': np.array([1,1,1,2,2,1]).reshape((-1,2)),
        'sparse_parflex': np.array([1,1,1,2,2,1]).reshape((-1,2))
        }
weights = {'dtw1': np.array([2,3,3]),
          'dtw2': np.array([1,1,1]),
          'dtw3': np.array([1,2,2]),
          'subseqdtw1': np.array([1,1,2]),
          'subseqdtw2': np.array([2,3,3]),
          'subseqdtw3': np.array([1,2,2]),
          'nwtw': 0, # weights are specified in NWTW algorithm
          'flexdtw': np.array([1.25,3,3]),
          'parflex': np.array([1.25,3.0,3.0]),
          'sparse_parflex': np.array([1.25,3.0,3.0])
          }
other_params = {
                'flexdtw': {'beta': 0.1}, 
                'parflex': {'beta': 0.1},
                'sparse_parflex': {'beta': 0.1}
               }
file_1 = "/home/ijain/ttmp/Chopin_Mazurkas_features/matching/Chopin_Op063No3/Chopin_Op063No3_Francois-1956_pid9070b-24.npy"
file_2 = "/home/ijain/ttmp/Chopin_Mazurkas_features/matching/Chopin_Op063No3/Chopin_Op063No3_Michalowski-1933_pid9083-16.npy"
beat_1 = beat_base + "/" + folder + "/Chopin_Op063No3_Francois-1956_pid9070b-24.beat"
beat_2 = beat_base + "/" + folder  + "/Chopin_Op063No3_Michalowski-1933_pid9083-16.beat"
 
print(beat_1)
print(beat_2)


try:
    steps
except NameError:
    steps = {"flexdtw": np.array([[1, 1], [1, 2], [2, 1]], dtype=int)}

try:
    weights
except NameError:
    weights = {"flexdtw": np.array([1.25, 3.0, 3.0], dtype=float)}

try:
    other_params
except NameError:
    other_params = {"flexdtw": {"beta": 0.1}}

# Load the files
F1 = np.load(file_1, allow_pickle=True)
F2 = np.load(file_2, allow_pickle=True)

L1 = F1.shape[1]
L2 = F2.shape[1]
 
# ----------------- GLOBAL FLEXDTW PATH -----------------
C_full = 1.0 - FlexDTW.L2norm(F1).T @ FlexDTW.L2norm(F2)
buffer_flex = min(L1, L2) * (1 - (1 - other_params['flexdtw']['beta']) * min(L1, L2) / max(L1, L2))
# print(buffer_flex)
best_cost_full, global_flex_path, D, P, B, debug = FlexDTW.flexdtw(
    C_full,
    steps=steps['flexdtw'],
    weights=weights['flexdtw'],
    buffer=buffer_flex,
)
S = P  # P is the start encoding (used for grey start→endpoint background)
# print(f"Global FlexDTW path shape: {global_flex_path.shape}")

# FlexDTW plot: grey = possible starts per global endpoint, black = selected path, dotted grid like Parflex
# FlexDTW plot overlays:
# - grey = possible starts per global endpoint (from P)
# - black = selected global FlexDTW path
# (chunk-based overlays are drawn after we build chunks via Parflex stage-1)
# sparse_parflex.plot_flex_with_global_S_background(
#     C_global=C_full,
#     flex_wp=global_flex_path,
#     S=S,
#     L_div=4000,
#     title="Global FlexDTW with S-based start→endpoint background",
# )


# ----------------- PARFLEX -----------------

C, tiled_result = Parflex.align_system_parflex(
    F1,
    F2,
    steps=steps['flexdtw'],
    weights=weights['flexdtw'],
    beta=other_params['flexdtw']['beta'],
)

# print("\n" + "=" * 80)
# print("STAGE 1 COMPLETE - TESTING RESULTS")
# print("=" * 80)

stage2_result_parflex = Parflex.parflex_2a(
    tiled_result,
    C,
    beta=other_params['flexdtw']['beta'],
    show_fig=False,
    top_k=1,
)


# Aligning system sparse parflex:
C, tiled_result = align_system_sparse_parflex(
    F1,
    F2,
    steps=steps['flexdtw'],
    weights=weights['flexdtw'],
    beta=other_params['flexdtw']['beta'],
)
# print(f"Number of chunks: {tiled_result['n_row']}×{tiled_result['n_col']}")

# print("\n" + "=" * 80)
# print("STAGE 1 COMPLETE - TESTING RESULTS")
# print("=" * 80)

stage2_result_sparse = sparse_parflex_2a(
    tiled_result,
    C,
    beta=other_params['flexdtw']['beta'],
    show_fig=False,
    top_k=1,
)

# compare numberically stage2_result_sparse and stage2_result_parflex's ['stitched_wp']
# comparisons = np.allclose(stage2_result_sparse['stitched_wp'], stage2_result_parflex['stitched_wp'])
# # print(comparisons)
import DTW
dtw_systems = ['dtw1', 'dtw2', 'dtw3']
dtw_paths = {}
for system in dtw_systems:
    wp = DTW.alignDTW(
        C,
        steps=steps[system],
        weights=weights[system],
        downsample=1,
        outfile=None,
        subseq=False,
    )
    dtw_paths[system] = wp
 
print(f"FlexDTW path: {global_flex_path.shape[1]} pts | Chunks: {tiled_result['n_row']}×{tiled_result['n_col']}")
plot_parflex_with_chunk_S_background(tiled_result, C, global_flex_path, stage2_result_sparse, ground_truth_=(beat_1, beat_2),
    extra_paths=dtw_paths,)


# %%



