#!/usr/bin/env python3
"""
compare_stage1.py

Compare Stage-1 outputs from two implementations (Implementation 1: .npy, Implementation 2: .pkl).
Compares structure and numeric values for each chunk/block and produces diagnostic plots for ALL blocks.

Assumptions based on your setup:
- Implementation 1: each sample folder contains "stage_1.npy" (a dict saved with numpy.save).
- Implementation 2: each sample folder contains one .pkl (chunks_dict). After your rename,
  keys in chunks_dict should match Implementation 1 naming:
    D_single, B_single, S_single, rows, cols, Ck_shape, best_cost, wp_local, wp_global, C (optional)
- Both impl1 and impl2 root dirs contain identically named subfolders for each sample.

Outputs:
- For each sample, a subfolder in --out will be created containing:
    - images for each block: heatmaps, diff, overlayed paths
    - a JSON summary "comparison_summary.json"
    - a CSV "comparison_table.csv" listing per-block results (pass/fail and metrics)

Usage:
    python compare_stage1.py --impl1 /path/to/impl1_root \
                             --impl2 /path/to/impl2_root \
                             --out  /path/to/output_root
"""

import argparse
import json
import os
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import math
import csv
from typing import Dict, Tuple, Any

# ----- Config -----
RTOL = 1e-6
ATOL = 1e-8
VERBOSE = True
PLOT_DPI = 150

# ----- Utilities -----


def load_impl1_stage1(stage1_npy_path: Path) -> Dict:
    """
    Load Implementation 1: numpy-saved dict (stage_1.npy)
    """
    data = np.load(stage1_npy_path, allow_pickle=True)
    # np.save(np.array(dict)) can return an array-like object; coerce to item() if needed
    try:
        loaded = data.item()
    except Exception:
        # data might already be a dict-like
        loaded = data
    return loaded


def load_impl2_chunks(pkl_path: Path) -> Dict:
    """
    Load Implementation 2: a pickle file containing chunks_dict (mapping (i,j) -> block dict)
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def allclose(a, b, rtol=RTOL, atol=ATOL) -> bool:
    """
    Robust np.allclose wrapper that also handles shape mismatch.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def ensure_array(x):
    if x is None:
        return None
    return np.asarray(x)


def safe_get(d: Dict, key: str, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def compare_arrays(a, b) -> Tuple[bool, float]:
    """
    Compare numeric arrays; return (is_close, max_abs_diff)
    """
    try:
        a = np.asarray(a)
        b = np.asarray(b)
    except Exception:
        return False, float("nan")
    if a.shape != b.shape:
        return False, float("inf")
    diff = np.abs(a - b)
    maxdiff = float(np.nanmax(diff))
    return (np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True), maxdiff)


# ----- Plotting -----


def plot_block_pair(
    outdir: Path,
    sample_name: str,
    bi: int,
    bj: int,
    block1: Dict,
    block2: Dict,
):
    """
    Create and save three figures for a block:
      1) C (impl1) heatmap and overlay path
      2) C (impl2) heatmap and overlay path
      3) diff heatmap + overlay both paths
    Also a combined figure that shows side-by-side C1, C2, diff
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Attempt to fetch C matrices; fall back to D_single if C not present
    C1 = safe_get(block1, "C", None)
    C2 = safe_get(block2, "C", None)

    if C1 is None and "Ck_shape" in block1:
        # Implementation1 didn't store C explicitly in blocks_2d. But top-level 'C' exists in result.
        # We cannot reconstruct here. We'll try to use D_single if present.
        C1 = None
    if C2 is None and "C" in block2:
        C2 = block2["C"]

    # Path local coordinates
    wp1 = safe_get(block1, "wp_local", safe_get(block1, "wp", None))
    wp2 = safe_get(block2, "wp_local", safe_get(block2, "wp", None))

    # Rows/cols bounds
    rows1 = safe_get(block1, "rows", safe_get(block1, "bounds", None))
    cols1 = safe_get(block1, "cols", None)
    rows2 = safe_get(block2, "rows", safe_get(block2, "bounds", None))
    cols2 = safe_get(block2, "cols", None)

    # Convert to arrays, and map both wps to local indices if they are global
    # We will plot local indices on the C chunk's axes
    # Determine chunk shapes if available
    shape1 = safe_get(block1, "Ck_shape", None)
    shape2 = safe_get(block2, "Ck_shape", None)

    # Choose a base matrix for plotting: C1 if available else C2 else zeros of shape from Ck_shape
    base_shape = None
    if C1 is not None:
        base_shape = C1.shape
    elif C2 is not None:
        base_shape = C2.shape
    elif shape1 is not None:
        base_shape = tuple(shape1)
    elif shape2 is not None:
        base_shape = tuple(shape2)

    if base_shape is None:
        # nothing to plot
        return

    # Ensure matrices are 2D arrays for plotting; if a matrix missing, create NaN matrix for plotting
    def prepare_matrix(mat, shape):
        if mat is None:
            return np.full(shape, np.nan)
        else:
            return np.asarray(mat)

    C1_plot = prepare_matrix(C1, base_shape)
    C2_plot = prepare_matrix(C2, base_shape)
    diff_plot = None
    try:
        diff_plot = C1_plot - C2_plot
    except Exception:
        diff_plot = np.full(base_shape, np.nan)

    # Convert wp to local indices if they are global by subtracting start indices
    def local_wp(wp, rows, cols):
        if wp is None:
            return None
        wp = np.asarray(wp)
        # If wp looks like global (values larger than shape), try subtracting start
        if rows is not None and cols is not None:
            if isinstance(rows, tuple) and isinstance(cols, tuple):
                row_start = rows[0]
                col_start = cols[0]
                # If wp values are > shape, map them
                if wp.size > 0 and (wp[:, 0].max() >= base_shape[0] or wp[:, 1].max() >= base_shape[1]):
                    wp_local = wp.copy().astype(int)
                    wp_local[:, 0] = wp_local[:, 0] - int(row_start)
                    wp_local[:, 1] = wp_local[:, 1] - int(col_start)
                    return wp_local
        # fallback: assume it's already local
        return wp.astype(int)

    wp1_local = local_wp(wp1, rows1, cols1)
    wp2_local = local_wp(wp2, rows2, cols2)

    # plotting helpers
    def save_fig(fig, fname):
        path = outdir / fname
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI)
        plt.close(fig)

    # Combined figure: side-by-side C1, C2, diff
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im0 = axes[0].imshow(C1_plot, aspect="auto", origin="lower")
        axes[0].set_title(f"Impl1 C (block {bi},{bj})")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
        if wp1_local is not None:
            axes[0].plot(wp1_local[:, 1], wp1_local[:, 0], linewidth=1.5, label="path1")
            axes[0].legend()

        im1 = axes[1].imshow(C2_plot, aspect="auto", origin="lower")
        axes[1].set_title(f"Impl2 C (block {bi},{bj})")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
        if wp2_local is not None:
            axes[1].plot(wp2_local[:, 1], wp2_local[:, 0], linewidth=1.5, label="path2")
            axes[1].legend()

        im2 = axes[2].imshow(diff_plot, aspect="auto", origin="lower")
        axes[2].set_title("Diff (Impl1 - Impl2)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)

        save_fig(fig, f"block_{bi}_{bj}__C_side_by_side.png")
    except Exception as e:
        print(f"Warning: failed to make combined C figure for block ({bi},{bj}): {e}")

    # Single overlay figure (paths on top of impl1)
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(C1_plot, aspect="auto", origin="lower")
        ax.set_title(f"Impl1 C + paths (block {bi},{bj})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        if wp1_local is not None:
            ax.plot(wp1_local[:, 1], wp1_local[:, 0], linewidth=1.5, label="path1")
        if wp2_local is not None:
            ax.plot(wp2_local[:, 1], wp2_local[:, 0], linewidth=1.0, linestyle="--", label="path2")
        ax.legend()
        save_fig(fig, f"block_{bi}_{bj}__paths_overlay.png")
    except Exception as e:
        print(f"Warning: failed to make overlay figure for block ({bi},{bj}): {e}")

    # Diff heatmap
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(diff_plot, aspect="auto", origin="lower")
        ax.set_title(f"Diff heatmap (block {bi},{bj})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        save_fig(fig, f"block_{bi}_{bj}__diff.png")
    except Exception as e:
        print(f"Warning: failed to make diff figure for block ({bi},{bj}): {e}")


# ----- Main comparison logic -----


def compare_blocks(block1: Dict, block2: Dict) -> Dict:
    """
    Compare two block dictionaries and return a dict describing the comparison results.
    """
    report = {}
    # keys to compare
    keys_arr = [
        ("Ck_shape", "Ck_shape"),
        ("D_single", "D_single"),
        ("B_single", "B_single"),
        ("S_single", "S_single"),
    ]

    for k1, k2 in keys_arr:
        v1 = safe_get(block1, k1, None)
        v2 = safe_get(block2, k2, None)
        ok, maxdiff = compare_arrays(v1, v2)
        report[f"{k1}_match"] = bool(ok)
        report[f"{k1}_maxdiff"] = float(maxdiff)

    # best_cost
    bc1 = safe_get(block1, "best_cost", None)
    bc2 = safe_get(block2, "best_cost", None)
    try:
        bc_close = np.isclose(float(bc1), float(bc2), rtol=RTOL, atol=ATOL)
        bc_diff = float(abs(float(bc1) - float(bc2)))
    except Exception:
        bc_close = False
        bc_diff = float("nan")
    report["best_cost_match"] = bool(bc_close)
    report["best_cost_diff"] = bc_diff

    # wp_local (paths)
    wp1 = safe_get(block1, "wp_local", safe_get(block1, "wp", None))
    wp2 = safe_get(block2, "wp_local", safe_get(block2, "wp", None))
    try:
        wp1_arr = np.asarray(wp1) if wp1 is not None else None
        wp2_arr = np.asarray(wp2) if wp2 is not None else None
        if wp1_arr is None or wp2_arr is None:
            wp_match = False
            wp_maxdiff = float("nan")
        else:
            if wp1_arr.shape != wp2_arr.shape:
                wp_match = False
                wp_maxdiff = float("inf")
            else:
                wp_match = np.array_equal(wp1_arr, wp2_arr)
                wp_maxdiff = float(np.nanmax(np.abs(wp1_arr - wp2_arr)))
    except Exception:
        wp_match = False
        wp_maxdiff = float("nan")
    report["wp_local_match"] = bool(wp_match)
    report["wp_local_maxdiff"] = wp_maxdiff

    # raw_cost vs sum of path values if present
    raw1 = safe_get(block1, "raw_cost", None)
    raw2 = safe_get(block2, "raw_cost", None)
    report["raw_cost_1"] = float(raw1) if raw1 is not None else None
    report["raw_cost_2"] = float(raw2) if raw2 is not None else None
    if raw1 is not None and raw2 is not None:
        report["raw_cost_match"] = bool(np.isclose(raw1, raw2, rtol=RTOL, atol=ATOL))
        report["raw_cost_diff"] = float(abs(raw1 - raw2))
    else:
        report["raw_cost_match"] = None
        report["raw_cost_diff"] = None

    return report


def compare_sample(sample_name: str, impl1_sample_dir: Path, impl2_sample_dir: Path, out_root: Path) -> Dict:
    """
    Compare one sample's outputs. Returns a dictionary summary for the sample.
    """
    sample_out_dir = out_root / sample_name
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    # Load implementation 1 result
    impl1_file = impl1_sample_dir / "stage_1.npy"
    if not impl1_file.exists():
        raise FileNotFoundError(f"Implementation 1 file missing: {impl1_file}")

    impl1_res = load_impl1_stage1(impl1_file)
    # Implementation 1 may have 'blocks_2d' populated (we prepared earlier)
    blocks1 = impl1_res.get("blocks_2d", None)
    if blocks1 is None:
        # fall back to 'blocks' list
        blocks_list = impl1_res.get("blocks", [])
        # convert to dict keyed by (bi,bj)
        blocks1 = {}
        for b in blocks_list:
            blocks1[(b["bi"], b["bj"])] = b

    # Load implementation 2 result: find first .pkl in impl2_sample_dir
    pkl_candidates = list(impl2_sample_dir.glob("*.pkl"))
    if len(pkl_candidates) == 0:
        raise FileNotFoundError(f"No .pkl file found in {impl2_sample_dir}")
    if len(pkl_candidates) > 1:
        # prefer stage_1-like filename if present
        chosen = None
        for p in pkl_candidates:
            if "stage" in p.stem or "stage_1" in p.stem:
                chosen = p
                break
        if chosen is None:
            chosen = pkl_candidates[0]
    else:
        chosen = pkl_candidates[0]

    impl2_chunks = load_impl2_chunks(chosen)
    # impl2_chunks is expected to be a dict keyed by (i,j) tuples (or string keys)
    # Normalize keys to tuple form
    chunks2 = {}
    for k, v in impl2_chunks.items():
        # keys may be strings like "(0, 1)" or tuples already
        if isinstance(k, str):
            try:
                key = eval(k)
                if isinstance(key, tuple):
                    chunks2[key] = v
                else:
                    chunks2[(int(key), 0)] = v
            except Exception:
                # fallback: use integer index if possible
                try:
                    chunks2[(int(k), 0)] = v
                except Exception:
                    # put under a unique numeric id
                    chunks2[(hash(k) % 1000000, 0)] = v
        else:
            chunks2[k] = v

    # Determine union of block keys
    keys1 = set(blocks1.keys())
    keys2 = set(chunks2.keys())
    all_keys = sorted(list(keys1.union(keys2)))

    sample_summary = {
        "sample_name": sample_name,
        "n_blocks_impl1": len(keys1),
        "n_blocks_impl2": len(keys2),
        "blocks": {},
    }

    # CSV rows
    csv_rows = []
    csv_header = [
        "sample", "bi", "bj",
        "exists_impl1", "exists_impl2",
        "Ck_shape_match",
        "D_single_match", "D_single_maxdiff",
        "B_single_match", "B_single_maxdiff",
        "S_single_match", "S_single_maxdiff",
        "best_cost_match", "best_cost_diff",
        "wp_local_match", "wp_local_maxdiff",
        "raw_cost_match", "raw_cost_diff"
    ]

    for key in all_keys:
        bi, bj = key
        blk1 = blocks1.get(key, None)
        blk2 = chunks2.get(key, None)

        block_report = {
            "exists_impl1": blk1 is not None,
            "exists_impl2": blk2 is not None,
        }

        if blk1 is None or blk2 is None:
            # record and continue
            block_report.update({
                "note": "missing_block_in_one_impl"
            })
            sample_summary["blocks"][f"{bi},{bj}"] = block_report
            csv_rows.append([
                sample_name, bi, bj,
                bool(blk1 is not None), bool(blk2 is not None),
                False, False, float("nan"),
                False, float("nan"),
                False, float("nan"),
                False, float("nan"),
                False, float("nan"),
                None, None
            ])
            continue

        # Compare the two blocks
        cmp = compare_blocks(blk1, blk2)
        block_report.update(cmp)
        sample_summary["blocks"][f"{bi},{bj}"] = block_report

        # Make plots for all blocks
        try:
            block_out_dir = sample_out_dir / f"block_{bi}_{bj}"
            plot_block_pair(block_out_dir, sample_name, bi, bj, blk1, blk2)
        except Exception as e:
            print(f"Warning: failed plotting block {bi},{bj} in sample {sample_name}: {e}")

        csv_rows.append([
            sample_name, bi, bj,
            True, True,
            bool(cmp.get("Ck_shape_match", False)),
            bool(cmp.get("D_single_match", False)), float(cmp.get("D_single_maxdiff", math.nan)),
            bool(cmp.get("B_single_match", False)), float(cmp.get("B_single_maxdiff", math.nan)),
            bool(cmp.get("S_single_match", False)), float(cmp.get("S_single_maxdiff", math.nan)),
            bool(cmp.get("best_cost_match", False)), float(cmp.get("best_cost_diff", math.nan)),
            bool(cmp.get("wp_local_match", False)), float(cmp.get("wp_local_maxdiff", math.nan)),
            cmp.get("raw_cost_match", None), cmp.get("raw_cost_diff", None)
        ])

    # Save JSON summary
    summary_path = sample_out_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(sample_summary, f, indent=2)

    # Save CSV
    csv_path = sample_out_dir / "comparison_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)

    return sample_summary


def main(args):
    impl1_root = Path(args.impl1).expanduser()
    impl2_root = Path(args.impl2).expanduser()
    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    # list subfolders in impl1 root
    impl1_subs = sorted([p for p in impl1_root.iterdir() if p.is_dir()])
    impl2_subs = sorted([p for p in impl2_root.iterdir() if p.is_dir()])

    # build mapping from sample_name -> path for both
    impl1_map = {p.name: p for p in impl1_subs}
    impl2_map = {p.name: p for p in impl2_subs}

    common_samples = sorted([n for n in impl1_map.keys() if n in impl2_map.keys()])
    only_impl1 = sorted([n for n in impl1_map.keys() if n not in impl2_map.keys()])
    only_impl2 = sorted([n for n in impl2_map.keys() if n not in impl1_map.keys()])

    print(f"Found {len(common_samples)} common samples, {len(only_impl1)} only in impl1, {len(only_impl2)} only in impl2")

    overall_results = {
        "common_samples": len(common_samples),
        "only_impl1": only_impl1,
        "only_impl2": only_impl2,
        "samples": {}
    }

    for sample_name in common_samples:
        print(f"\nComparing sample: {sample_name}")
        try:
            res = compare_sample(sample_name, impl1_map[sample_name], impl2_map[sample_name], out_root)
            overall_results["samples"][sample_name] = res
            # quick summary print
            n_blocks = len(res["blocks"])
            mismatches = sum(1 for b in res["blocks"].values()
                             if (not b.get("D_single_match", False)) or (not b.get("B_single_match", False)) or (not b.get("S_single_match", False)) or (not b.get("best_cost_match", False)))
            print(f"  blocks compared: {n_blocks}, suspected mismatches: {mismatches}")
        except Exception as e:
            print(f"  ERROR comparing sample {sample_name}: {e}")
            overall_results["samples"][sample_name] = {"error": str(e)}

    # save overall_results
    overall_path = out_root / "comparison_overall_summary.json"
    with open(overall_path, "w") as f:
        json.dump(overall_results, f, indent=2)

    print(f"\nComparison finished. Results saved to: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare stage-1 outputs from two implementations")
    parser.add_argument("--impl1", required=True, help="Root directory for implementation 1 (contains sample subfolders with stage_1.npy)")
    parser.add_argument("--impl2", required=True, help="Root directory for implementation 2 (contains sample subfolders with .pkl files)")
    parser.add_argument("--out", required=True, help="Output root directory to write comparison results")
    args = parser.parse_args()
    main(args)
