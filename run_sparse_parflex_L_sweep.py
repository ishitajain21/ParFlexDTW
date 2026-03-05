"""
Run Sparse Parflex for each L in L_VALUES and save the system under experiments_train.

Usage from Sparse_Parflex.ipynb (after running the Testing cell so F1, F2 and
align_system_sparse_parflex, sparse_parflex_2a are defined):

    from run_sparse_parflex_L_sweep_config import (
        L_VALUES,
        EXPERIMENTS_TRAIN_ROOT,
        SPARSE_PARFLEX_SUBDIR,
    )
    from run_sparse_parflex_L_sweep import run_L_sweep_and_save

    run_L_sweep_and_save(
        align_system_sparse_parflex,
        sparse_parflex_2a,
        F1,
        F2,
        steps=steps["flexdtw"],
        weights=weights["flexdtw"],
        beta=other_params["flexdtw"]["beta"],
        basename="pair",  # optional: for filenames, e.g. "pair" -> L_100/pair.npy
    )

Do not run this file as __main__; it is meant to be called from the notebook.
"""

from pathlib import Path
import numpy as np

from run_sparse_parflex_L_sweep_config import (
    L_VALUES,
    EXPERIMENTS_TRAIN_ROOT,
    SPARSE_PARFLEX_SUBDIR,
)


def run_L_sweep_and_save(
    align_system_sparse_parflex,
    sparse_parflex_2a,
    F1,
    F2,
    steps=None,
    weights=None,
    beta=0.1,
    basename="pair",
):
    """
    For each L in L_VALUES: run Stage 1 + Stage 2 and save the system under
    experiments_train/sparse_parflex/L_<L>/.

    Saves per L:
      - <basename>.npy: warping path (2, N) for compatibility with comparison scripts.
      - <basename>_system.npz: best_cost, stitched_wp, n_row, n_col, L (optional extras).
    """
    out_root = EXPERIMENTS_TRAIN_ROOT / SPARSE_PARFLEX_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    for L in L_VALUES:
        out_dir = out_root / f"L_{L}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1
        C, tiled_result = align_system_sparse_parflex(
            F1, F2, steps=steps, weights=weights, beta=beta, L=L
        )
        # Stage 2
        stage2_result = sparse_parflex_2a(
            tiled_result, C, beta=beta, show_fig=False, top_k=1
        )

        # Warping path (2, N) for compatibility
        stitched_wp = stage2_result.get("stitched_wp")
        if stitched_wp is not None and stitched_wp.size > 0:
            wp = np.asarray(stitched_wp)
            if wp.ndim == 2 and wp.shape[1] == 2:
                wp = wp.T  # (N, 2) -> (2, N)
            np.save(out_dir / f"{basename}.npy", wp)

        # System snapshot (essential keys only to keep size down)
        np.savez(
            out_dir / f"{basename}_system.npz",
            best_cost=stage2_result.get("best_cost"),
            stitched_wp=stage2_result.get("stitched_wp"),
            n_row=stage2_result.get("n_row"),
            n_col=stage2_result.get("n_col"),
            L=np.int64(L),
        )

        print(f"L={L} -> {out_dir}")

    print(f"Done. Results under {out_root}")


if __name__ == "__main__":
    print(
        "Run this from Sparse_Parflex.ipynb after defining F1, F2 and running "
        "align_system_sparse_parflex / sparse_parflex_2a. See docstring."
    )
