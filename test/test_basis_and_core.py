"""Core tests: bitmasks, meet-in-the-middle basis, solvers, and exact PH identity."""

from itertools import combinations
import numpy as np
from bandprojected_exactdiagonalization import (
    BandProjectedED,
    BandProjectedHamiltonian,
    BlochBand,
    BravaisLattice,
    InternalSpace,
    MomentumSectorCatalog,
    SectorLookup,
    TorusGeometry,
)


def _synthetic_multiband_model(shape=(3, 4), nbands=2, seed=42) -> tuple[BlochBand, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lat = BravaisLattice.triangular(1.0, 60.0)
    geom = TorusGeometry(lat, shape)
    gs = np.indices((5, 5)).reshape(2, -1).T - 2
    n_pw = len(gs)
    rec_idx = np.vstack([gs, gs])
    comps = np.concatenate([np.zeros(n_pw, dtype=int), np.ones(n_pw, dtype=int)])
    b = lat.reciprocal
    frames = np.zeros((geom.size, 2 * n_pw, nbands), dtype=np.complex128)
    for s, k in enumerate(geom.momenta_cart):
        w = np.exp(-0.25 * np.sum((gs @ b - k[None, :]) ** 2, axis=1))
        mat = rng.normal(size=(2 * n_pw, nbands)) + 1j * rng.normal(size=(2 * n_pw, nbands))
        mat *= np.concatenate([w, w])[:, None]
        frames[s], _ = np.linalg.qr(mat, mode="reduced")
    h1 = np.zeros((geom.size, nbands, nbands), dtype=np.complex128)
    for s in range(geom.size):
        m = rng.normal(size=(nbands, nbands)) + 1j * rng.normal(size=(nbands, nbands))
        h1[s] = 0.5 * (m + m.conj().T)
    band = BlochBand(geom, frames, h1, rec_idx, comps, InternalSpace.bands(nbands))
    qs = geom.transfer_disk(2.8)
    q_cart = geom.transfer_cart(qs)
    weights = np.exp(-0.15 * np.sum(q_cart**2, axis=1))
    return band, qs, weights


def test_meet_in_the_middle_and_lookup():
    lat = BravaisLattice.square(1.0)
    geom = TorusGeometry(lat, (3, 4))
    orb_mom = np.repeat(geom.indices, 2, axis=0)
    cat = MomentumSectorCatalog(24, geom.shape, orb_mom, max_particles=4)

    # Brute-force oracle on 24 orbitals, 3 particles
    all_masks = []
    all_sectors = []
    for c in combinations(range(24), 3):
        m = sum(1 << i for i in c)
        s = orb_mom[list(c)].sum(axis=0) % geom.shape
        all_masks.append(m)
        all_sectors.append((int(s[0]), int(s[1])))
    all_masks = np.array(all_masks, dtype=np.uint64)

    total = 0
    for s in geom.indices:
        st = (int(s[0]), int(s[1]))
        sb = cat.basis(3, st)
        expected = np.sort(all_masks[[i for i, x in enumerate(all_sectors) if x == st]])
        np.testing.assert_array_equal(sb.masks, expected)
        total += len(sb)
        if len(sb) > 0:
            # Verify SectorLookup on all valid + random invalid queries
            idx = sb.lookup.find(sb.masks)
            np.testing.assert_array_equal(idx, np.arange(len(sb)))
            rng = np.random.default_rng(100 + st[0] * 10 + st[1])
            bad = rng.integers(0, 1 << 24, size=500, dtype=np.uint64)
            got = sb.lookup.find(bad)
            ref = np.array([int(np.searchsorted(sb.masks, x)) if x in set(sb.masks.tolist()) else -1 for x in bad])
            np.testing.assert_array_equal(got, ref)
    assert total == 2024


def test_solvers_agree_and_exact_ph_identity(tmp_path):
    # Test on 2x3 two-band (12 orbitals) at N=4 and N_h=8!
    band, qs, weights = _synthetic_multiband_model(shape=(2, 3), nbands=2, seed=7)
    ham = BandProjectedHamiltonian.from_band(band, qs, weights)
    ed = BandProjectedED(ham, n_particles=4, num_threads=2, dense_threshold=10)

    # Compare dense, matrix, and matrixfree modes across all 6 sectors
    for s in band.geometry.indices:
        st = (int(s[0]), int(s[1]))
        res_d = ed.solve_sector(st, nev=4, mode="dense")
        res_m = ed.solve_sector(st, nev=4, mode="matrix")
        res_f = ed.solve_sector(st, nev=4, mode="matrixfree")
        np.testing.assert_allclose(res_m.energies, res_d.energies, atol=1e-11)
        np.testing.assert_allclose(res_f.energies, res_d.energies, atol=1e-11)
        # Verify eigenspace overlaps
        sv_m = np.linalg.svd(res_d.states.conj().T @ res_m.states, compute_uv=False)
        sv_f = np.linalg.svd(res_d.states.conj().T @ res_f.states, compute_uv=False)
        assert np.all(sv_m > 1.0 - 1e-10)
        assert np.all(sv_f > 1.0 - 1e-10)

    # Exact particle-hole energy identity: E_parent(12 - 4 = 8) = E_full + E_partner(4)
    # Wait: test N_e=8 parent vs N_h=4 partner!
    ed_parent = BandProjectedED(ham, n_particles=8, num_threads=2)
    partner_band, _, e_full = band.particle_hole_transform(qs, weights, diagonalize_one_body=False)
    ham_ph = BandProjectedHamiltonian.from_band(partner_band, qs, weights)
    ed_ph = BandProjectedED(ham_ph, n_particles=4, num_threads=2)

    # Full-band momentum sum mod shape
    k_full = np.repeat(band.geometry.indices, 2, axis=0).sum(axis=0) % band.geometry.shape
    for s in band.geometry.indices:
        sp = (int(s[0]), int(s[1]))
        sh = (int((k_full[0] - sp[0]) % band.geometry.shape[0]), int((k_full[1] - sp[1]) % band.geometry.shape[1]))
        ep = ed_parent.solve_sector(sp, nev=4, mode="dense").energies
        eh = ed_ph.solve_sector(sh, nev=4, mode="dense").energies
        np.testing.assert_allclose(ep, e_full + eh, atol=1e-11)

    # Verify checkpoint-resume
    ckpt = tmp_path / "scan.pkl"
    scan1 = ed.scan_sectors(sectors=[(0, 0), (0, 1)], nev=3, mode="matrixfree", checkpoint_path=ckpt)
    scan2 = ed.scan_sectors(sectors="all", nev=3, mode="matrixfree", checkpoint_path=ckpt)
    assert len(scan2["sectors"]) == 6
    np.testing.assert_allclose(scan1["sectors"][(0, 0)].energies, scan2["sectors"][(0, 0)].energies, atol=1e-14)


def test_custom_four_fermion_tensor():
    band, qs, weights = _synthetic_multiband_model(shape=(2, 2), nbands=1, seed=9)
    ham_band = BandProjectedHamiltonian.from_band(band, qs, weights)
    # Reconstruct 4-fermion tensor explicitly from D_q
    n_orb = band.n_orbitals
    v4 = np.zeros((n_orb, n_orb, n_orb, n_orb), dtype=np.complex128)
    for q, w in zip(qs, weights, strict=True):
        d = band.density_matrix(q)
        dh = d.conj().T
        v4 += 0.5 * w * np.einsum("ik,jl->ijkl", dh, d)
    ham_v4 = BandProjectedHamiltonian.from_four_fermion_tensor(
        band.geometry, 1, band.one_body_blocks, v4
    )
    e1 = BandProjectedED(ham_band, n_particles=2).scan_sectors(nev=2)["ground_energy"]
    e2 = BandProjectedED(ham_v4, n_particles=2).scan_sectors(nev=2)["ground_energy"]
    assert np.isclose(e1, e2, atol=1e-12)
