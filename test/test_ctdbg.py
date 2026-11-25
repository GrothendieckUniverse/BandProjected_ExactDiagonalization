"""Tests for chiral TDBG C=2 FCI at beta=0.1 and beta=1.0 (rectangular & tilted)."""

import numpy as np
from bandprojected_exactdiagonalization import BandProjectedED, thouless_charge_pump_1d
from bandprojected_exactdiagonalization.models import ChiralTDBG


def test_ctdbg_endpoints_and_hall_transition():
    # Single-particle Chern=2 check: beta=0.1 needs finer 12x12 mesh to resolve dual Berry peaks
    assert ChiralTDBG(beta=0.1, cutoff=4.0).band((12, 12)).chern_number()["integer"] == 2
    assert ChiralTDBG(beta=1.0, cutoff=5.0).band((4, 6)).chern_number()["integer"] == 2

    # Rectangular 4x6 endpoints at beta=0.1 (C_e=1/3) and beta=1.0 (C_e=2/3)
    m01 = ChiralTDBG(beta=0.1, cutoff=5.0)
    _, h01 = m01.hamiltonian((4, 6), qell_cutoff=6.0)
    ed01 = BandProjectedED(h01, n_particles=8)
    scan01 = ed01.scan_sectors(nev=3, mode="matrix", sector_workers=8)
    top3_01 = sorted(s for _, s, _ in scan01["ordered_levels"][:3])
    assert top3_01 == [(0, 0), (0, 2), (0, 4)]
    assert np.isclose(scan01["ordered_levels"][0][0], -154.73460570484235, atol=1e-5)
    assert scan01["ordered_levels"][3][0] - scan01["ordered_levels"][2][0] > 3.0

    m10 = ChiralTDBG(beta=1.0, cutoff=5.0)
    _, h10 = m10.hamiltonian((4, 6), qell_cutoff=6.0)
    ed10 = BandProjectedED(h10, n_particles=8)
    scan10 = ed10.scan_sectors(nev=3, mode="matrix", sector_workers=8)
    top3_10 = sorted(s for _, s, _ in scan10["ordered_levels"][:3])
    assert top3_10 == [(0, 0), (0, 2), (0, 4)]
    assert scan10["ordered_levels"][3][0] - scan10["ordered_levels"][2][0] > 1.2

    # Tilted 24-cell torus supercell [[1, 5], [0, 24]] at beta=0.2 gives triplet at (0,4),(0,12),(0,20)
    m02 = ChiralTDBG(beta=0.2, cutoff=4.5)
    _, h02_tilt = m02.hamiltonian(np.array([[1, 5], [0, 24]]), qell_cutoff=6.0, smooth=False)
    ed_tilt = BandProjectedED(h02_tilt, n_particles=8)
    scan_tilt = ed_tilt.scan_sectors(
        sectors=[(0, 4), (0, 12), (0, 20), (0, 0)], nev=2, mode="matrix", sector_workers=4
    )
    assert scan_tilt["sectors"][(0, 4)].energies[0] < scan_tilt["sectors"][(0, 0)].energies[0] - 2.0

    # 1D Resta-Thouless charge pump confirming Hall transition C_e: 1/3 (beta=0.1) -> 2/3 (beta=1.0)
    t01 = thouless_charge_pump_1d(
        lambda tw: ChiralTDBG(beta=0.1, cutoff=3.5).hamiltonian((4, 6), qell_cutoff=5.5, twists=tw),
        n_particles=8,
        n_steps=8,
        flux_axis=1,
        manifold_size=3,
        sectors=[(0, 0), (0, 2), (0, 4)],
        nev=2,
        mode="matrix",
        sector_workers=3,
    )
    t10 = thouless_charge_pump_1d(
        lambda tw: ChiralTDBG(beta=1.0, cutoff=3.5).hamiltonian((4, 6), qell_cutoff=5.5, twists=tw),
        n_particles=8,
        n_steps=8,
        flux_axis=1,
        manifold_size=3,
        sectors=[(0, 0), (0, 2), (0, 4)],
        nev=2,
        mode="matrix",
        sector_workers=3,
    )
    assert t01["resolved"] and t01["integer_chern"] == 1
    assert t10["resolved"] and t10["integer_chern"] == 2
    assert t01["minimum_manifold_gap"] > 1.0 and t10["minimum_manifold_gap"] > 1.0
