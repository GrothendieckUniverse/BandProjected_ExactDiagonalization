"""Tests for observables: occupation matrix, structure factor, PES, flux flow, and SVG plots."""

import numpy as np
from bandprojected_exactdiagonalization import (
    BandProjectedED,
    BandProjectedHamiltonian,
    BlochBand,
    BravaisLattice,
    InternalSpace,
    TorusGeometry,
)
from bandprojected_exactdiagonalization.models import haldane_v1_hamiltonian
from bandprojected_exactdiagonalization.observables import (
    many_body_chern_number_and_pump,
    occupation_matrix,
    particle_entanglement_spectrum,
    plot_ed_spectrum,
    static_structure_factor_sector,
)
from bandprojected_exactdiagonalization.interaction import dual_gate_coulomb


def test_observables_and_svg(tmp_path):
    lat = BravaisLattice.triangular(1.0, 60.0)
    geom = TorusGeometry(lat, (2, 3))
    gs = np.indices((5, 5)).reshape(2, -1).T - 2
    n_pw = len(gs)
    b = lat.reciprocal
    rec_idx = np.vstack([gs, gs])
    comps = np.concatenate([np.zeros(n_pw, dtype=int), np.ones(n_pw, dtype=int)])
    frames = np.zeros((geom.size, 2 * n_pw, 2), dtype=np.complex128)
    rng = np.random.default_rng(11)
    for s, k in enumerate(geom.momenta_cart):
        w = np.exp(-0.25 * np.sum((gs @ b - k[None, :]) ** 2, axis=1))
        m = (rng.normal(size=(2 * n_pw, 2)) + 1j * rng.normal(size=(2 * n_pw, 2))) * np.concatenate([w, w])[:, None]
        frames[s], _ = np.linalg.qr(m, mode="reduced")
    e1 = np.zeros((geom.size, 2))
    e1[:, 1] = 5.0
    band = BlochBand(geom, frames, e1, rec_idx, comps, InternalSpace.bands(2))
    ham = BandProjectedHamiltonian.from_potential(
        band, lambda q, a: dual_gate_coulomb(q, a, dielectric=10.0, gate_distance=5.0), qell_cutoff=3.5
    )
    ed = BandProjectedED(ham, n_particles=3)
    scan = ed.scan_sectors(nev=3)
    top_e, top_s, top_l = scan["ordered_levels"][0]
    res = scan["sectors"][top_s]
    st = res.states[:, top_l]

    occ = occupation_matrix(res.basis, st, nbands=2)
    assert np.allclose(occ, occ.conj().swapaxes(-1, -2), atol=1e-12)
    assert np.isclose(np.trace(occ, axis1=-2, axis2=-1).real.sum(), 3.0, atol=1e-12)

    qs = geom.transfer_disk(2.5)
    ssf = static_structure_factor_sector(band, ed.catalog, res.basis, st, qs)
    assert np.all(ssf >= -1e-12)

    pes = particle_entanglement_spectrum(ed.catalog, [(res.basis, st)], n_a=1)
    # Sum of exp(-xi) == 1.0 (trace of reduced density matrix rho_A)
    tr_rho = sum(np.sum(np.exp(-xi)) for xi in pes.values())
    assert np.isclose(tr_rho, 1.0, atol=1e-12)

    svg_path = plot_ed_spectrum(scan, tmp_path / "test_spec.svg")
    assert svg_path.is_file() and svg_path.read_text().lstrip().startswith("<")

    # Fast exact filled-band (N_e=4 on 2x2 LLL) many-body Chern check including twist wrap seam
    topo = many_body_chern_number_and_pump(
        lambda tw: haldane_v1_hamiltonian(shape=(2, 2), v1=1.0, qell_cutoff=6.0, twists=tw, include_q0=True),
        n_particles=4,
        grid_shape=(4, 4),
        manifold_size=1,
        nev=1,
    )
    assert topo["integer_chern"] == 1
