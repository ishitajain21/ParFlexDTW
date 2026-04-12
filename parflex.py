"""
ParFlex — sparse parallel FlexDTW (Numba).

Canonical implementation: `sparflex_numba`. This module re-exports it under the
names used by `Parflex.ipynb` so the notebook can stay a thin import layer.
"""

from __future__ import annotations

import numpy as np

from sparflex_numba import (
    DEFAULT_CHUNK_LENGTH,
    _parflex_dp_njit,
    _parflex_njit,
    _build_edge_data,
    _build_edge_lookups,
    _build_valid_mask,
    _nb_backtrace_within_chunk,
    _nb_dp_fill_chunks_sparse,
    _nb_edge_index_to_local_coords,
    _nb_initialize_chunks_sparse,
    _nb_scan_edges,
    _start_points_from_S,
    align_system_sparse_parflex,
    chunk_flexdtw,
    chunked_flexdtw,
    convert_chunks_to_tiled_result,
    edge_index_to_local_coords,
    global_to_prev_chunk_edge,
    local_to_global_coords,
    parflex as _parflex_full,
    plot_normalized_global_edge_cost,
    plot_parflex_with_chunk_S_background,
    sparse_parflex_2a,
    stage_2_backtrace_compatible,
    sync_overlapping_positions,
)

# ---------------------------------------------------------------------------
# Notebook / legacy aliases (same objects as sparflex_numba)
# ---------------------------------------------------------------------------

plot_alignment_with_tile_background = plot_parflex_with_chunk_S_background
_edge_starts_from_S = _start_points_from_S
run_flexdtw_on_tiles = chunk_flexdtw
_edge_slot_to_local_nb = _nb_edge_index_to_local_coords
edge_slot_to_local = edge_index_to_local_coords
local_to_global_cell = local_to_global_coords
global_cell_to_prev_tile_edge = global_to_prev_chunk_edge
_sparse_edge_positions = _build_valid_mask
_init_tile_edge_costs_nb = _nb_initialize_chunks_sparse
_fill_tile_edge_costs_nb = _nb_dp_fill_chunks_sparse
propagate_tile_edge_costs = chunked_flexdtw
sync_tile_overlap_edges = sync_overlapping_positions
_backtrace_within_chunk_nb = _nb_backtrace_within_chunk
_scan_edges_nb = _nb_scan_edges
build_tiled_metadata = convert_chunks_to_tiled_result
stage2_scan_and_stitch = stage_2_backtrace_compatible
plot_normalized_tile_edge_costs = plot_normalized_global_edge_cost

tiled_stage1_from_features = align_system_sparse_parflex
run_stage2_from_tiled = sparse_parflex_2a
parflex = _parflex_full


def _edge_length_for_chunk_edge(chunk_shape, edge_type):
    rows, cols = int(chunk_shape[0]), int(chunk_shape[1])
    return cols if int(edge_type) == 0 else rows


def decode_start_from_S(S_single, local_row, local_col):
    s_val = S_single[int(local_row), int(local_col)]
    if s_val > 0:
        return 0, int(s_val)
    if s_val < 0:
        return int(-s_val), 0
    return 0, 0


def _on_bottom_edge(local_row, chunk_shape):
    return int(local_row) == int(chunk_shape[0]) - 1


def _on_left_edge(local_col):
    return int(local_col) == 0


def global_cell_to_tile_edge(global_row, global_col, tile_i, tile_j, chunks_dict):
    start_1, _, start_2, _ = chunks_dict[(tile_i, tile_j)]["bounds"]
    local_row = int(global_row) - int(start_1)
    local_col = int(global_col) - int(start_2)
    rows, cols = chunks_dict[(tile_i, tile_j)]["shape"]
    if local_row == rows - 1:
        return 0, local_col
    if local_col == cols - 1:
        return 1, local_row
    raise ValueError("Global cell is not on tile edge")


def init_tile_edge_costs(
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
    n_chunks_1,
    n_chunks_2,
    D_arr,
    L_arr,
):
    return _init_tile_edge_costs_nb(
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
        n_chunks_1,
        n_chunks_2,
        D_arr,
        L_arr,
    )


def fill_tile_edge_costs(
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
    n_chunks_1,
    n_chunks_2,
    D_arr,
    L_arr,
):
    return _fill_tile_edge_costs_nb(
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
        n_chunks_1,
        n_chunks_2,
        D_arr,
        L_arr,
    )


def backtrace_within_chunk(
    B_single,
    steps,
    start_r,
    start_c,
    end_r,
    end_c,
    global_r_offset,
    global_c_offset,
):
    out = _backtrace_within_chunk_nb(
        np.ascontiguousarray(B_single),
        np.ascontiguousarray(steps, dtype=np.int64),
        int(start_r),
        int(start_c),
        int(end_r),
        int(end_c),
        int(global_r_offset),
        int(global_c_offset),
    )
    return [(int(out[k, 0]), int(out[k, 1])) for k in range(out.shape[0])]


def backtrace_and_stitch(*args, **kwargs):
    raise RuntimeError(
        "Use stage2_scan_and_stitch(), which performs backtrace and stitching."
    )
