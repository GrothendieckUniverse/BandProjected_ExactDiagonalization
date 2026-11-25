"""Tests for BravaisLattice, Smith-Normal-Form TorusGeometry, and BlochBand."""

import numpy as np
from bandprojected_exactdiagonalization.lattice import (
    BravaisLattice,
    TorusGeometry,
    smith_normal_form_2x2,
)
from bandprojected_exactdiagonalization.bloch import BlochBand, InternalSpace


def _qwz_plane_wave_band(shape: tuple[int, int]) -> BlochBand:
    lat = BravaisLattice.square(1.0)
    geom = TorusGeometry(lat, shape)
    gs = np.indices((5, 5)).reshape(2, -1).T - 2
    n_pw = len(gs)
    b = lat.reciprocal
    rec_idx = np.vstack([gs, gs])
    comps = np.concatenate([np.zeros(n_pw, dtype=int), np.ones(n_pw, dtype=int)])
    frames = np.zeros((geom.size, 2 * n_pw, 1), dtype=np.complex128)
    for s, k in enumerate(geom.momenta_cart):
        pk = -k
        dx, dy = np.sin(pk[0]), np.sin(pk[1])
        dz = -1.0 + np.cos(pk[0]) + np.cos(pk[1])
        h_qwz = np.array([[dz, dx - 1j * dy], [dx + 1j * dy, -dz]], dtype=np.complex128)
        _, vecs = np.linalg.eigh(h_qwz)
        v_low = vecs[:, 0]
        w = np.exp(-0.2 * np.sum((gs @ b + pk[None, :]) ** 2, axis=1))
        w /= np.linalg.norm(w)
        frames[s, :n_pw, 0] = w * v_low[0]
        frames[s, n_pw:, 0] = w * v_low[1]
        frames[s] /= np.linalg.norm(frames[s])
    return BlochBand(geom, frames, np.zeros((geom.size, 1)), rec_idx, comps, InternalSpace.bands(1))


def test_smith_normal_form_and_tilted_torus():
    lat = BravaisLattice.triangular(1.0, 60.0)
    for mat in (np.array([[1, 5], [0, 24]]), np.array([[5, 1], [1, 5]])):
        u, d, v = smith_normal_form_2x2(mat)
        assert np.array_equal(u @ mat @ v, d)
        assert d[0, 0] > 0 and d[1, 1] > 0 and d[1, 1] % d[0, 0] == 0
    for mat in (np.array([[1, 5], [0, 24]]), np.array([[5, 1], [1, 5]]), np.array([[4, 0], [0, 6]])):
        u, d, v = smith_normal_form_2x2(mat)
        assert np.array_equal(u @ mat @ v, d)
        geom = TorusGeometry(lat, mat, twists=(0.2, -0.3))
        assert geom.size == 24
        for q in ([1, 0], [0, 1], [2, -3], [5, 7]):
            target, wind = geom.shift(q)
            q_cart = geom.transfer_cart(q)
            reconstructed = geom.momenta_cart[target] + wind @ lat.reciprocal
            diff = reconstructed - (geom.momenta_cart + q_cart[None, :])
            assert np.max(np.abs(diff)) < 1e-11


def test_bloch_fft_form_factors_and_chern():
    band6 = _qwz_plane_wave_band((6, 6))
    qs = band6.geometry.transfer_disk(3.5)
    ff_batch = band6.form_factors(qs)
    for i, q in enumerate(qs[:15]):
        np.testing.assert_allclose(ff_batch[i], band6.form_factor(q), atol=1e-13)
    ch6 = band6.chern_number()
    assert ch6["integer"] in (-1, 1) and ch6["branch_resolved"]

    band16 = _qwz_plane_wave_band((16, 16))
    ch16 = band16.chern_number()
    assert ch16["integer"] == ch6["integer"]
    qg = band16.quantum_geometry()
    da = (2.0 * np.pi) ** 2 / (band16.geometry.lattice.unit_cell_area * band16.geometry.size)
    assert np.isclose(np.sum(qg["berry_curvature"]) * da / (2.0 * np.pi), ch16["integer"], atol=0.05)
    assert np.all(qg["trace_defect"] >= -1e-10)
