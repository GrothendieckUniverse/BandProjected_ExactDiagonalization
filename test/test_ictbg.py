"""Tests for zero-Chern intervalley-coupled TBG (icTBG) at nu=1/3."""

import numpy as np
from bandprojected_exactdiagonalization import (
    BandProjectedED,
    occupation_matrix,
    thouless_charge_pump_1d,
)
from bandprojected_exactdiagonalization.models import IntervalleyTBG


def test_ictbg_c0_fci_and_hall_response():
    model = IntervalleyTBG(cutoff=4.0, dielectric=4.0)
    # 12x12 resolves both opposite-Chern valley peaks and yields C = 0
    assert IntervalleyTBG(cutoff=3.5, dielectric=4.0).band((12, 12)).chern_number()["integer"] == 0

    _, ham = model.hamiltonian((4, 6), qell_cutoff=6.0)
    ed = BandProjectedED(ham, n_particles=8)
    scan = ed.scan_sectors(nev=3, mode="matrix", sector_workers=8)
    top3 = sorted(s for _, s, _ in scan["ordered_levels"][:3])
    assert top3 == [(0, 0), (0, 2), (0, 4)]
    assert np.isclose(scan["ordered_levels"][0][0], -84.94376438030795, atol=1e-5)
    assert np.isclose(scan["ordered_levels"][3][0] - scan["ordered_levels"][2][0], 3.030818, atol=1e-4)

    # Check occupation matrix sum on 4x6 and kappa' barrier depletion (< 0.005) on tilted 24-cell torus
    res04 = scan["sectors"][(0, 4)]
    occ46 = occupation_matrix(res04.basis, res04.states[:, 0], nbands=1)[:, 0, 0].real
    assert np.isclose(occ46.sum(), 8.0, atol=1e-10)

    _, ham_tilt = model.hamiltonian(np.array([[1, 5], [0, 24]]), qell_cutoff=6.0, smooth=False)
    r_tilt = BandProjectedED(ham_tilt, n_particles=8, num_threads=8).solve_sector((0, 4), nev=1, mode="matrix")
    occ_tilt = occupation_matrix(r_tilt.basis, r_tilt.states[:, 0], nbands=1)[:, 0, 0].real
    assert occ_tilt.min() < 0.005

    # Verify 1D Resta-Thouless charge pump C = 1 (Hall conductivity C_e = 1/3)
    topo = thouless_charge_pump_1d(
        lambda tw: IntervalleyTBG(cutoff=3.5, dielectric=4.0).hamiltonian((4, 6), qell_cutoff=5.5, twists=tw),
        n_particles=8,
        n_steps=8,
        flux_axis=1,
        manifold_size=3,
        sectors=[(0, 0), (0, 2), (0, 4)],
        nev=2,
        mode="matrix",
        sector_workers=3,
    )
    assert topo["resolved"] and topo["integer_chern"] == 1
    assert topo["minimum_manifold_gap"] > 2.0
