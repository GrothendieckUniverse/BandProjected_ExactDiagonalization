"""Tests for twisted MoTe2 C=1 FCI at hole-filling nu_h = -2/3."""

import numpy as np
from bandprojected_exactdiagonalization import BandProjectedED
from bandprojected_exactdiagonalization.models import TMDContinuum


def test_tmote2_ph_equivalence_and_4x6_fci():
    # Exact PH identity on 3x4 (bare hole N_h=8 vs electron partner N_e=4)
    model34 = TMDContinuum(twist_degrees=4.0, dielectric=18.0, cutoff=4)
    hole_band, ham_h = model34.bare_hole_hamiltonian(supercell=(3, 4), qell_cutoff=6.5)
    assert hole_band.chern_number()["integer"] == 1
    _, ham_p, e_full, _ = model34.hole_partner_hamiltonian(supercell=(3, 4), qell_cutoff=6.5)
    ed_h = BandProjectedED(ham_h, n_particles=8)
    ed_p = BandProjectedED(ham_p, n_particles=4)
    k_full = hole_band.geometry.indices.sum(axis=0) % (3, 4)
    e_h0 = ed_h.solve_sector((0, 0), nev=2).energies
    e_p0 = ed_p.solve_sector((int(k_full[0]), int(k_full[1])), nev=2).energies
    np.testing.assert_allclose(e_h0, e_full + e_p0, atol=1e-10)

    # Full 4x6 tMoTe2 at theta=4.0 deg, eps=15.0 cross-checked against Non-FQH_FCI reference
    model46_15 = TMDContinuum(twist_degrees=4.0, dielectric=15.0, cutoff=6)
    _, ham46_15, _, _ = model46_15.hole_partner_hamiltonian(supercell=(4, 6), qell_cutoff=7.5)
    ed46_15 = BandProjectedED(ham46_15, n_particles=8, num_threads=8)
    res00 = ed46_15.solve_sector((0, 0), nev=3, mode="matrix")
    res02 = ed46_15.solve_sector((0, 2), nev=3, mode="matrix")
    assert np.isclose(res00.energies[0], 507.8026164247897, atol=1e-6)
    assert np.isclose(res02.energies[0], 507.7287770028321, atol=1e-6)
    assert res00.energies[1] - res00.energies[0] > 2.5

    # Primary multipolar benchmark target: theta=4.0 deg, eps=18.0
    model46_18 = TMDContinuum(twist_degrees=4.0, dielectric=18.0, cutoff=6)
    _, ham46_18, _, _ = model46_18.hole_partner_hamiltonian(supercell=(4, 6), qell_cutoff=7.5)
    ed46_18 = BandProjectedED(ham46_18, n_particles=8, num_threads=8)
    s18 = ed46_18.scan_sectors(sectors=[(0, 0), (0, 2), (0, 4)], nev=3, mode="matrix")
    top3 = sorted(s for _, s, _ in s18["ordered_levels"][:3])
    assert top3 == [(0, 0), (0, 2), (0, 4)]
    assert s18["ordered_levels"][3][0] - s18["ordered_levels"][2][0] > 0.7
