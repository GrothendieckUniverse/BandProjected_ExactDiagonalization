"""Tests for vortexable two-band p4mm strain model at nu=4/3 along magic lines."""

import numpy as np
from bandprojected_exactdiagonalization import BandProjectedED
from bandprojected_exactdiagonalization.models import P4MMStrainModel, solve_p4mm_magic_line


def test_p4mm_magic_line_ph_and_endpoints():
    line = solve_p4mm_magic_line(ratios=(0.0, 0.7183 / 0.6651), cutoff=6.0)
    assert np.isclose(line[0]["alpha"], 0.5279, atol=5e-4)
    assert np.isclose(line[1]["alpha"], 0.6651, atol=5e-4)

    # Verify total Chern = +1 on 6x6
    m0 = P4MMStrainModel(alpha=0.5279, beta=0.0, cutoff=10.0)
    assert m0.band((6, 6)).chern_number()["integer"] == 1

    # Fast exact multi-band PH equivalence on 2x3 (N_e=8 vs N_h=4)
    b23, h23 = m0.hamiltonian((2, 3), qell_cutoff=12.0)
    _, h23_ph, e_full23, _ = m0.hole_partner_hamiltonian((2, 3), qell_cutoff=12.0)
    k_full23 = np.repeat(b23.geometry.indices, 2, axis=0).sum(axis=0) % (2, 3)
    res_d23 = BandProjectedED(h23, n_particles=8).solve_sector((0, 0), nev=2)
    res_p23 = BandProjectedED(h23_ph, n_particles=4).solve_sector((int(k_full23[0]), int(k_full23[1])), nev=2)
    np.testing.assert_allclose(res_d23.energies, e_full23 + res_p23.energies, atol=1e-9)

    # First magic-line endpoint (0.5279, 0.0) on 3x4 (N_h=8): matrix vs zero-loop matrixfree agreement
    _, h0_ph, e_full0, _ = m0.hole_partner_hamiltonian((3, 4), qell_cutoff=16.0)
    ed_ph0 = BandProjectedED(h0_ph, n_particles=8, num_threads=12)
    r_mat = ed_ph0.solve_sector((1, 0), nev=3, mode="matrix")
    r_mf = ed_ph0.solve_sector((1, 0), nev=3, mode="matrixfree")
    np.testing.assert_allclose(r_mat.energies, r_mf.energies, atol=1e-9)
    assert r_mf.elapsed_seconds < 15.0

    scan0 = ed_ph0.scan_sectors(nev=3, mode="matrix", sector_workers=12)
    assert np.isclose(e_full0 + scan0["ordered_levels"][0][0], -24.113451870307962, atol=1e-5)
    assert sorted(s for _, s, _ in scan0["ordered_levels"][:3]) == [(0, 0), (1, 0), (2, 0)]
    assert scan0["ordered_levels"][3][0] - scan0["ordered_levels"][2][0] > 0.05

    # Second magic-line endpoint (alpha=0.6651, beta=0.7183) at nu=4/3
    m1 = P4MMStrainModel(alpha=0.6651, beta=0.7183, cutoff=10.0)
    _, h1_ph, _, _ = m1.hole_partner_hamiltonian((3, 4), qell_cutoff=16.0)
    scan1 = BandProjectedED(h1_ph, n_particles=8).scan_sectors(nev=3, mode="matrix", sector_workers=12)
    assert sorted(s for _, s, _ in scan1["ordered_levels"][:3]) == [(0, 0), (1, 0), (2, 0)]
    assert scan1["ordered_levels"][3][0] - scan1["ordered_levels"][2][0] > 0.05
