"""Parity tests: GPU wavefront FlexDTW vs Numba FlexDTW.flexdtw."""
import os

import numpy as np
import pytest

import FlexDTW


@pytest.fixture(scope="module")
def cupy_modules():
    cupy = pytest.importorskip("cupy")
    from gpu_flexdtw import flexdtw_chunk_from_global_C, is_gpu_available

    if not is_gpu_available():
        pytest.skip("No CUDA device")
    return cupy, flexdtw_chunk_from_global_C


def test_flexdtw_gpu_matches_cpu_small(cupy_modules):
    cupy, flexdtw_chunk_from_global_C = cupy_modules
    rng = np.random.RandomState(1)
    steps = [(1, 1), (1, 2), (2, 1)]
    weights = [2, 3, 3]
    for H, W in [(5, 6), (17, 19), (48, 50)]:
        C = np.abs(rng.randn(H, W).astype(np.float64)) * 0.1
        bc, wp_c, Dc, Pc, Bc, _ = FlexDTW.flexdtw(
            C, steps=steps, weights=weights, buffer=1)
        from gpu_flexdtw import cost_matrix_to_gpu_f32

        C_dev = cost_matrix_to_gpu_f32(C)
        bg, wp_g, Dg, Pg, Bg, _ = flexdtw_chunk_from_global_C(
            C_dev, C.shape[1], 0, H, 0, W, steps, weights, buffer=1)
        assert np.allclose(Dc, Dg), (H, W)
        assert np.array_equal(Pc, Pg)
        assert np.array_equal(Bc, Bg)
        assert np.isclose(bc, bg)
        assert wp_c.shape == wp_g.shape
        assert np.array_equal(wp_c, wp_g)


def test_run_flexdtw_on_tiles_gpu_matches_cpu(cupy_modules):
    pytest.importorskip("Parflex_gpu")
    import Parflex_gpu as P

    assert cupy_modules[0] is not None
    rng = np.random.RandomState(2)
    C = np.abs(rng.randn(40, 44).astype(np.float64)) * 0.08
    steps = [(1, 1), (1, 2), (2, 1)]
    weights = [2, 3, 3]
    L = 14
    d_cpu, Lc, n1, n2 = P.run_flexdtw_on_tiles(
        C, L=L, steps=steps, weights=weights, profile_dir=None, use_gpu=False)
    d_gpu, Lg, _, _ = P.run_flexdtw_on_tiles(
        C, L=L, steps=steps, weights=weights, profile_dir=None, use_gpu=True)
    assert Lc == Lg and n1 and n2
    assert set(d_cpu.keys()) == set(d_gpu.keys())
    for k in d_cpu:
        assert np.allclose(d_cpu[k]["D"], d_gpu[k]["D"]), k
        assert np.array_equal(d_cpu[k]["S"], d_gpu[k]["S"]), k
        assert np.array_equal(d_cpu[k]["B"], d_gpu[k]["B"]), k
        assert np.isclose(d_cpu[k]["best_cost"], d_gpu[k]["best_cost"]), k


def test_build_edge_gather_matches_cpu(cupy_modules):
    """Batched GPU gather for edge_Cf_olap matches dense chunk C reads."""
    cupy = cupy_modules[0]
    import Parflex as P
    import Parflex_gpu as PG

    rng = np.random.RandomState(4)
    C = np.abs(rng.randn(24, 28).astype(np.float64)) * 0.06
    L = 11
    steps = [(1, 1), (1, 2), (2, 1)]
    weights = [2, 3, 3]
    C_dev = cupy.asarray(np.ascontiguousarray(C), dtype=cupy.float32)
    chunks, n1, n2, _ = PG.run_flexdtw_on_tiles(
        C,
        L=L,
        steps=steps,
        weights=weights,
        profile_dir=None,
        use_gpu=True,
        C_dev=C_dev,
    )
    t_cpu = P._build_edge_data(chunks, n1, n2, L)
    t_gpu = PG._build_edge_data_gpu(chunks, n1, n2, L, C_dev)
    for a, b in zip(t_cpu, t_gpu):
        assert np.allclose(a, b), (np.max(np.abs(a - b)),)


def test_parflex_disable_gpu_env(cupy_modules):
    import Parflex_gpu as P

    rng = np.random.RandomState(3)
    C = np.abs(rng.randn(16, 18).astype(np.float64)) * 0.07
    os.environ["PARFLEX_DISABLE_GPU"] = "1"
    try:
        d1, _, _, _ = P.run_flexdtw_on_tiles(
            C, L=10, profile_dir=None, use_gpu=None)
        d2, _, _, _ = P.run_flexdtw_on_tiles(
            C, L=10, profile_dir=None, use_gpu=False)
        for k in d1:
            assert np.allclose(d1[k]["D"], d2[k]["D"])
    finally:
        os.environ.pop("PARFLEX_DISABLE_GPU", None)
