"""Tests for Torus LLL magnetic fibers, Weyl form factors, Haldane V1, and 1D/2D Chern pumps."""

import numpy as np
from bandprojected_exactdiagonalization import (
    BandProjectedED,
    many_body_chern_number_and_pump,
    thouless_charge_pump_1d,
)
from bandprojected_exactdiagonalization.models import TorusLLL, haldane_v1_hamiltonian


def test_lll_haldane_v1_laughlin_triplet():
    # Single-particle Chern, ideal quantum geometry, and fiber vs analytic Weyl agreement
    for tw in [(0.0, 0.0), (0.7, 1.3)]:
        band = TorusLLL(chirality=1).band((4, 6), twists=tw)
        assert band.chern_number()["integer"] == 1
        assert band.quantum_geometry()["trace_defect"].max() < 1.5e-2
        q = np.array([1, 2])
        tgt, windings = band.geometry.shift(q)
        wb = windings @ band.geometry.lattice.reciprocal
        fibers = band.frames[:, :, 0]
        real_ff = np.sum(fibers[tgt].conj() * np.exp(-1j * (wb @ band._quad_points.T)) * fibers, axis=1)
        np.testing.assert_allclose(real_ff, band.form_factor(q)[:, 0, 0], atol=1e-10)
    assert TorusLLL(chirality=1).band((12, 12)).quantum_geometry()["trace_defect"].max() < 5e-4

    # Two-particle m=1 relative angular momentum eigenvalue = +v1 = 1.0 on 6x6 torus
    _, ham66 = haldane_v1_hamiltonian(shape=(6, 6), v1=1.0, qell_cutoff=7.5, include_q0=True)
    res2 = BandProjectedED(ham66, n_particles=2).solve_sector((0, 0), nev=18)
    assert np.isclose(res2.energies[-1], 1.0, atol=1e-6)

    # Exact 3-fold degenerate zero-energy Laughlin ground manifold on 3x4 at N_e=4
    _, ham34 = haldane_v1_hamiltonian(shape=(3, 4), v1=1.0, qell_cutoff=7.5, include_q0=True)
    ed = BandProjectedED(ham34, n_particles=4, num_threads=2)
    scan = ed.scan_sectors(nev=4)
    levels = scan["ordered_levels"]
    assert all(abs(e) < 1e-10 for e, _, _ in levels[:3])
    assert levels[3][0] - levels[2][0] > 0.1

    # Verify 1D Resta-Thouless charge pump along BOTH axes -> Hall conductivity = 1/3
    p_y = thouless_charge_pump_1d(
        lambda tw: haldane_v1_hamiltonian(shape=(2, 3), v1=1.0, qell_cutoff=7.5, twists=tw, include_q0=True),
        n_particles=2,
        n_steps=8,
        flux_axis=1,
        manifold_size=3,
        nev=3,
    )
    p_x = thouless_charge_pump_1d(
        lambda tw: haldane_v1_hamiltonian(shape=(3, 4), v1=1.0, qell_cutoff=7.5, twists=tw, include_q0=True),
        n_particles=4,
        n_steps=8,
        flux_axis=0,
        manifold_size=3,
        nev=3,
    )
    assert p_y["resolved"] and p_y["integer_chern"] == 1
    assert p_x["resolved"] and p_x["integer_chern"] == 1
    assert np.isclose(p_y["averaged_hall_conductivity"], 1.0 / 3.0, atol=1e-10)
    assert np.isclose(p_x["averaged_hall_conductivity"], 1.0 / 3.0, atol=1e-10)

    # Cross-check against 2D FHS many-body Chern number on (2, 3)
    topo = many_body_chern_number_and_pump(
        lambda tw: haldane_v1_hamiltonian(shape=(2, 3), v1=1.0, qell_cutoff=7.5, twists=tw, include_q0=True),
        n_particles=2,
        grid_shape=(4, 4),
        manifold_size=3,
        nev=3,
    )
    assert topo["integer_chern"] == 1
    assert np.isclose(topo["averaged_hall_conductivity"], 1.0 / 3.0, atol=1e-10)
