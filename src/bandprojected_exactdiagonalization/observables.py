"""Many-body observables, topological invariants, charge pumping, PES, and SVG spectra."""

from __future__ import annotations

from math import comb
from pathlib import Path
from typing import Any, Callable
import numpy as np

from .basis import MomentumSectorCatalog, SectorBasis, make_between_masks, popcount_u64
from .bloch import BlochBand
from .interaction import BandProjectedHamiltonian
from .solver import BandProjectedED


def _resolve_sector_basis(
    basis_or_catalog: SectorBasis | MomentumSectorCatalog,
    n_particles: int,
    sector: tuple[int, int],
) -> SectorBasis:
    if isinstance(basis_or_catalog, SectorBasis):
        return basis_or_catalog
    return basis_or_catalog.basis(n_particles, sector)


def apply_one_body_sector(
    matrix: np.ndarray,
    source: SectorBasis,
    target: SectorBasis,
    states: np.ndarray,
    between_masks: np.ndarray | None = None,
) -> np.ndarray:
    """Apply dGamma(matrix) = sum_{i,j} matrix[i,j] c_i^dag c_j from ``source`` to ``target``."""
    mat = np.asarray(matrix, dtype=np.complex128)
    s = np.asarray(states, dtype=np.complex128)
    vec = s.ndim == 1
    if vec:
        s = s[:, None]
    if mat.shape != (source.n_orbitals, source.n_orbitals) or s.shape[0] != len(source):
        raise ValueError("Incompatible one-body operator or state shape")
    out = np.zeros((len(target), s.shape[1]), dtype=np.complex128)
    if len(source) == 0 or len(target) == 0:
        return out[:, 0] if vec else out

    btw = make_between_masks(source.n_orbitals) if between_masks is None else between_masks
    # Diagonal part (only when source == target)
    if source.sector == target.sector and len(source) == len(target):
        diag_m = np.diagonal(mat)
        for orb in np.flatnonzero(np.abs(diag_m) > 1e-14):
            occ = (source.masks & np.uint64(1 << int(orb))) != 0
            if np.any(occ):
                out[occ] += diag_m[orb] * s[occ]

    # Off-diagonal part
    for j in range(source.n_orbitals):
        mj = np.uint64(1 << j)
        sel = np.flatnonzero((source.masks & mj) != 0)
        if len(sel) == 0:
            continue
        rem = source.masks[sel] ^ mj
        nz_i = np.flatnonzero(np.abs(mat[:, j]) > 1e-14)
        for i in nz_i:
            i = int(i)
            if i == j:
                continue
            mi = np.uint64(1 << i)
            allowed = (rem & mi) == 0
            if not np.any(allowed):
                continue
            src_idx = sel[allowed]
            tgt_masks = rem[allowed] | mi
            rows = target.lookup.find(tgt_masks)
            valid = rows >= 0
            if not np.any(valid):
                continue
            parity = popcount_u64(source.masks[src_idx[valid]] & btw[i, j]) & 1
            coeff = mat[i, j] * (1.0 - 2.0 * parity)
            out[rows[valid]] += coeff[:, None] * s[src_idx[valid]]
    return out[:, 0] if vec else out


def occupation_matrix(
    basis: SectorBasis,
    states: np.ndarray,
    nbands: int,
) -> np.ndarray:
    """Compute momentum-resolved 1-RDM n_{alpha,beta}(k) = <c_{k,alpha}^dag c_{k,beta}>.

    If ``states`` is 2D ``(D, M)``, averages equally over the ``M`` orthonormal columns.
    Returns Hermitian array of shape ``(N_uc, nbands, nbands)``.
    """
    s = np.asarray(states, dtype=np.complex128)
    if s.ndim == 1:
        s = s[:, None]
    if s.shape[0] != len(basis):
        raise ValueError("State dimension does not match SectorBasis")
    n_k = int(np.prod(basis.geometry_shape))
    if n_k * nbands != basis.n_orbitals:
        raise ValueError("n_k * nbands must equal n_orbitals")

    probs = np.mean(np.abs(s) ** 2, axis=1)
    n_mat = np.zeros((n_k, nbands, nbands), dtype=np.complex128)
    for k in range(n_k):
        for a in range(nbands):
            orb = k * nbands + a
            occ = (basis.masks & np.uint64(1 << orb)) != 0
            n_mat[k, a, a] = float(np.sum(probs[occ]))
    if nbands == 1:
        return n_mat

    btw = make_between_masks(basis.n_orbitals)
    for k in range(n_k):
        base = k * nbands
        for a in range(nbands):
            for b in range(a + 1, nbands):
                oa, ob = base + a, base + b
                ma, mb = np.uint64(1 << oa), np.uint64(1 << ob)
                src = np.flatnonzero(((basis.masks & mb) != 0) & ((basis.masks & ma) == 0))
                if len(src) == 0:
                    continue
                tgt_masks = basis.masks[src] ^ (ma | mb)
                dst = basis.lookup.find_valid(tgt_masks)
                parity = popcount_u64(basis.masks[src] & btw[oa, ob]) & 1
                sgn = 1.0 - 2.0 * parity
                val = np.mean(
                    np.sum(s[dst].conj() * s[src] * sgn[:, None], axis=0)
                )
                n_mat[k, a, b] = val
                n_mat[k, b, a] = np.conj(val)
    return n_mat


def static_structure_factor_sector(
    band: BlochBand,
    catalog: MomentumSectorCatalog,
    basis: SectorBasis,
    state: np.ndarray,
    transfers: np.ndarray,
) -> np.ndarray:
    """Connected static structure factor S(q) = (<rho_q rho_{-q}> - |<rho_q>|^2) / N_e."""
    st = np.asarray(state, dtype=np.complex128)
    if st.ndim != 1 or len(st) != len(basis):
        raise ValueError("1D normalized state matching SectorBasis required")
    qs = np.asarray(transfers, dtype=np.int64)
    btw = make_between_masks(basis.n_orbitals)
    s1, s2 = basis.sector
    n1, n2 = band.geometry.shape
    out = np.empty(len(qs), dtype=np.float64)
    n_e = max(1, basis.n_particles)
    for idx, q in enumerate(qs):
        q_snf = band.geometry.to_snf_coords(q)
        tgt_sector = ((s1 - int(q_snf[0])) % n1, (s2 - int(q_snf[1])) % n2)
        tgt_basis = catalog.basis(basis.n_particles, tgt_sector)
        d_minus = band.density_matrix(-q)
        acted = apply_one_body_sector(d_minus, basis, tgt_basis, st, btw)
        val = float(np.vdot(acted, acted).real)
        if tgt_sector == basis.sector:
            mean_val = np.vdot(st, acted)
            val -= float(abs(mean_val) ** 2)
        out[idx] = val / n_e
    return out


def particle_entanglement_spectrum(
    catalog: MomentumSectorCatalog,
    sector_states: list[tuple[SectorBasis, np.ndarray]],
    n_a: int,
) -> dict[tuple[int, int], np.ndarray]:
    """Momentum-resolved Particle Entanglement Spectrum (PES) xi = -ln(lambda) for partition N_A."""
    if not sector_states:
        raise ValueError("At least one sector state required")
    n_e = sector_states[0][0].n_particles
    n_b = n_e - int(n_a)
    if not (0 < n_a < n_e):
        raise ValueError(f"Partition n_a={n_a} must satisfy 0 < n_a < {n_e}")
    n_orb = catalog.n_orbitals
    shape = catalog.shape
    total_states = sum(st.shape[1] if st.ndim == 2 else 1 for _, st in sector_states)

    # Precompute bit-position inversions for fast parity between disjoint masks m_A, m_B
    # sgn(m_A, m_B) = (-1)^{sum_{a in m_A} popcount(m_B & below_a)}
    below = np.array([(1 << bit) - 1 for bit in range(n_orb)], dtype=np.uint64)

    pes_by_sector: dict[tuple[int, int], np.ndarray] = {}
    for k_a in np.ndindex(shape):
        basis_a = catalog.basis(n_a, k_a)
        d_a = len(basis_a)
        if d_a == 0:
            continue
        configs_a = basis_a.configurations()  # (d_a, n_a)
        columns = []
        for basis_s, vecs in sector_states:
            v = vecs[:, None] if vecs.ndim == 1 else vecs
            k_b = ((basis_s.sector[0] - k_a[0]) % shape[0], (basis_s.sector[1] - k_a[1]) % shape[1])
            basis_b = catalog.basis(n_b, k_b)
            d_b = len(basis_b)
            if d_b == 0:
                continue
            # Disjoint pairs (m_A & m_B == 0)
            disjoint = (basis_a.masks[:, None] & basis_b.masks[None, :]) == 0
            ia, ib = np.nonzero(disjoint)
            if len(ia) == 0:
                continue
            full_masks = basis_a.masks[ia] | basis_b.masks[ib]
            rows = basis_s.lookup.find_valid(full_masks)
            # Compute fermion splitting parity for each (ia, ib)
            par = np.zeros(len(ia), dtype=np.int64)
            mb_sel = basis_b.masks[ib]
            for col in range(n_a):
                par += popcount_u64(mb_sel & below[configs_a[ia, col]])
            sgn = 1.0 - 2.0 * (par & 1)
            for m_col in range(v.shape[1]):
                mat = np.zeros((d_a, d_b), dtype=np.complex128)
                mat[ia, ib] = sgn * v[rows, m_col]
                columns.append(mat)
        if not columns:
            continue
        big = np.hstack(columns) / np.sqrt(comb(n_e, n_a) * total_states)
        sv = np.linalg.svd(big, compute_uv=False)
        evals = sv**2
        evals = evals[evals > 1e-30]
        pes_by_sector[k_a] = -np.log(np.sort(evals)[::-1])
    return pes_by_sector


def _apply_per_k_blocks(
    m_blocks: np.ndarray,
    basis_a: SectorBasis,
    basis_b: SectorBasis,
    states_b: np.ndarray,
    perm_k: np.ndarray | None = None,
) -> np.ndarray:
    """Apply block-diagonal (with optional momentum permutation) single-particle operator to Fock states."""
    n_k, b, b2 = m_blocks.shape
    if b != b2:
        raise ValueError("Square per-momentum operator blocks required")
    v = np.asarray(states_b, dtype=np.complex128)
    vec = v.ndim == 1
    if vec:
        v = v[:, None]
    n_col = v.shape[1]
    if len(basis_a) == 0 or len(basis_b) == 0:
        out = np.zeros((len(basis_a), n_col), dtype=np.complex128)
        return out[:, 0] if vec else out

    if perm_k is None:
        cur_masks = basis_b.masks
        cur_vecs = v
    else:
        perm_orb = (np.repeat(perm_k, b) * b + np.tile(np.arange(b), n_k)).astype(np.int64)
        cur_masks = np.zeros_like(basis_b.masks)
        parity = np.zeros(len(basis_b), dtype=np.int64)
        for o_src in range(basis_b.n_orbitals):
            occ = (basis_b.masks & np.uint64(1 << o_src)) != 0
            if not np.any(occ):
                continue
            p_dst = int(perm_orb[o_src])
            if p_dst + 1 < basis_b.n_orbitals:
                parity[occ] += popcount_u64(cur_masks[occ] >> np.uint64(p_dst + 1))
            cur_masks[occ] |= np.uint64(1 << p_dst)
        cur_vecs = v * (1.0 - 2.0 * (parity & 1))[:, None]

    # Fast path for b == 1: strictly diagonal in Fock masks
    if b == 1:
        scalars = m_blocks[:, 0, 0]
        amp = np.ones(len(cur_masks), dtype=np.complex128)
        for k in range(n_k):
            occ = (cur_masks & np.uint64(1 << k)) != 0
            if np.any(occ):
                amp[occ] *= scalars[k]
        idx = basis_a.lookup.find_valid(cur_masks)
        out = np.zeros((len(basis_a), n_col), dtype=np.complex128)
        out[idx] = amp[:, None] * cur_vecs
        return out[:, 0] if vec else out

    # General multi-band b >= 2: sequential momentum-block action on bitmasks
    masks_cur = cur_masks
    vecs_cur = cur_vecs
    for k in range(n_k):
        mk = m_blocks[k]
        if np.allclose(mk, np.eye(b), atol=1e-13):
            continue
        base = k * b
        local_shift = np.uint64(base)
        local_mask = np.uint64(((1 << b) - 1) << base)
        occ_pat = ((masks_cur & local_mask) >> local_shift).astype(np.int64)
        n_pat = 1 << b
        pat_bits = [tuple(i for i in range(b) if (p >> i) & 1) for p in range(n_pat)]
        pat_pop = np.array([len(x) for x in pat_bits], dtype=np.int64)
        cleared = masks_cur & (~local_mask)
        out_masks = []
        out_vecs = []
        for p_in in range(n_pat):
            sel = np.flatnonzero(occ_pat == p_in)
            if len(sel) == 0:
                continue
            r = pat_pop[p_in]
            if r == 0:
                out_masks.append(masks_cur[sel])
                out_vecs.append(vecs_cur[sel])
                continue
            cols_in = list(pat_bits[p_in])
            for p_out in range(n_pat):
                if pat_pop[p_out] != r:
                    continue
                rows_out = list(pat_bits[p_out])
                amp = complex(np.linalg.det(mk[np.ix_(rows_out, cols_in)]))
                if abs(amp) < 1e-14:
                    continue
                out_masks.append(cleared[sel] | np.uint64(p_out << base))
                out_vecs.append(amp * vecs_cur[sel])
        if not out_masks:
            out = np.zeros((len(basis_a), n_col), dtype=np.complex128)
            return out[:, 0] if vec else out
        all_m = np.concatenate(out_masks)
        all_v = np.vstack(out_vecs)
        s_ord = np.argsort(all_m)
        sm = all_m[s_ord]
        sv = all_v[s_ord]
        uniq, inv = np.unique(sm, return_inverse=True)
        merged = np.zeros((len(uniq), n_col), dtype=np.complex128)
        np.add.at(merged, inv, sv)
        masks_cur = uniq
        vecs_cur = merged

    idx_a = basis_a.lookup.find_valid(masks_cur)
    out = np.zeros((len(basis_a), n_col), dtype=np.complex128)
    out[idx_a] = vecs_cur
    return out[:, 0] if vec else out


def _apply_fock_overlap(
    u_a: np.ndarray,
    u_b: np.ndarray,
    basis_a: SectorBasis,
    basis_b: SectorBasis,
    states_b: np.ndarray,
    perm_k: np.ndarray | None = None,
    seam_mat: np.ndarray | None = None,
) -> np.ndarray:
    """Compute <Psi_A | Gamma(U_A^dag seam U_B) | Psi_B> on arbitrary multi-band frames.

    Handles both interior links (``perm_k is None``, block-diagonal in k) and boundary
    wrap links across ``twist -> 0`` (shifting momentum index ``k -> perm_k[k]`` with
    optional reciprocal seam matrix ``seam_mat`` when wrapping across primitive BZ).
    """
    n_k, _, b = u_a.shape
    if perm_k is None:
        m_blocks = np.einsum("kpa,kpb->kab", u_a.conj(), u_b)
    else:
        m_blocks = np.empty((n_k, b, b), dtype=np.complex128)
        for k_src in range(n_k):
            k_dst = int(perm_k[k_src])
            frame_src = (
                u_b[k_src]
                if seam_mat is None or seam_mat[k_src] is None
                else seam_mat[k_src] @ u_b[k_src]
            )
            m_blocks[k_dst] = u_a[k_dst].conj().T @ frame_src
    return _apply_per_k_blocks(m_blocks, basis_a, basis_b, states_b, perm_k=perm_k)


def _single_particle_seam_map(band_a: BlochBand, band_b: BlochBand, axis: int) -> tuple[np.ndarray, list[np.ndarray | None]]:
    """Map source k at end-of-twist to target k' at twist=0 along ``axis``."""
    geom_a, geom_b = band_a.geometry, band_b.geometry
    if not geom_a.is_diagonal or not geom_b.is_diagonal:
        raise ValueError("Twist-wrap overlap map currently requires diagonal rectangular grid")
    n1, n2 = geom_a.shape
    s0, s1 = band_a.seam_matrix(0), band_a.seam_matrix(1)
    # As phi_axis wraps 2*pi -> 0, k_cart(phi=2*pi) = k_cart(phi=0) + B_SC[axis],
    # which corresponds to k_idx shifted by +1 along `axis`!
    shift = np.eye(2, dtype=int)[axis]
    perm = np.empty(geom_a.size, dtype=np.int64)
    seams: list[np.ndarray | None] = [None] * geom_a.size
    for idx_b, (i, j) in enumerate(geom_b.indices):
        ti, wi = divmod(int(i) + shift[0], n1)
        tj, wj = divmod(int(j) + shift[1], n2)
        perm[idx_b] = wi * n2 + wj
        if ti != 0 or tj != 0:
            mat = np.eye(band_a.frames.shape[1], dtype=np.complex128)
            if ti > 0:
                mat = s0.conj().T @ mat
            if tj > 0:
                mat = s1.conj().T @ mat
            seams[idx_b] = mat
    return perm, seams


def many_body_chern_number_and_pump(
    model_builder: Callable[[tuple[float, float]], tuple[BlochBand, BandProjectedHamiltonian]],
    n_particles: int,
    grid_shape: tuple[int, int] = (6, 6),
    manifold_size: int = 3,
    sectors: Any = "all",
    nev: int = 4,
    mode: str = "auto",
    num_threads: int = 1,
    sector_workers: int = 1,
) -> dict[str, Any]:
    """Compute exact Fukui-Hatsugai-Suzuki many-body Chern number & Wilson charge pumps.

    Tracks the globally lowest ``manifold_size`` eigenstates across the 2D boundary-twist
    torus ``[0, 2*pi)^2``, keeping the exact single-particle Bloch frame overlaps and
    momentum-wrapping seams in every many-body overlap link!
    """
    nx, ny = int(grid_shape[0]), int(grid_shape[1])
    bands: list[list[BlochBand]] = [[None] * ny for _ in range(nx)]  # type: ignore
    multiplet_states: list[list[list[tuple[SectorBasis, np.ndarray]]]] = [[None] * ny for _ in range(nx)]  # type: ignore
    gaps = np.empty((nx, ny), dtype=np.float64)

    shared_catalog: MomentumSectorCatalog | None = None
    prev_row_v0: dict[tuple[int, int], np.ndarray] | None = None
    rng_seed = np.random.default_rng(4242)
    for ix in range(nx):
        prev_v0 = prev_row_v0
        for iy in range(ny):
            tw = (2.0 * np.pi * ix / nx, 2.0 * np.pi * iy / ny)
            band, ham = model_builder(tw)
            if shared_catalog is None:
                shared_catalog = MomentumSectorCatalog(
                    ham.n_orbitals,
                    ham.geometry.shape,
                    ham.orbital_momenta,
                    max_particles=n_particles,
                )
            ed = BandProjectedED(
                ham,
                n_particles=n_particles,
                num_threads=num_threads,
                catalog=shared_catalog,
            )
            scan = ed.scan_sectors(
                sectors=sectors,
                nev=nev,
                mode=mode,
                sector_workers=sector_workers,
                v0_by_sector=prev_v0,
            )
            prev_v0 = {}
            for s, res in scan["sectors"].items():
                if res.dimension > 0 and res.states.shape[1] > 0:
                    seed = res.states.sum(axis=1)
                    noise = 0.05 * (rng_seed.normal(size=res.dimension) + 1j * rng_seed.normal(size=res.dimension))
                    prev_v0[s] = seed + noise / np.sqrt(max(1, res.dimension))
            if iy == 0:
                prev_row_v0 = prev_v0
            levels = scan["ordered_levels"]
            if len(levels) < manifold_size:
                raise ValueError("Not enough eigenvalues to form requested ground manifold")
            gaps[ix, iy] = (
                levels[manifold_size][0] - levels[manifold_size - 1][0]
                if len(levels) > manifold_size
                else np.inf
            )
            # Group selected global levels by momentum sector
            by_sec: dict[tuple[int, int], list[int]] = {}
            for _, s, lvl in levels[:manifold_size]:
                by_sec.setdefault(s, []).append(lvl)
            sel = [
                (scan["sectors"][s].basis, scan["sectors"][s].states[:, lvls])
                for s, lvls in sorted(by_sec.items())
            ]
            bands[ix][iy] = band
            multiplet_states[ix][iy] = sel

    links = np.zeros((nx, ny, 2, manifold_size, manifold_size), dtype=np.complex128)
    for ix in range(nx):
        for iy in range(ny):
            b_cur = bands[ix][iy]
            st_cur = multiplet_states[ix][iy]
            for axis in (0, 1):
                nix = (ix + (axis == 0)) % nx
                niy = (iy + (axis == 1)) % ny
                b_nxt = bands[nix][niy]
                st_nxt = multiplet_states[nix][niy]
                perm = seams = None
                if (axis == 0 and ix == nx - 1) or (axis == 1 and iy == ny - 1):
                    perm, seams = _single_particle_seam_map(b_nxt, b_cur, axis)

                # Assemble manifold_size x manifold_size overlap <Psi(cur) | Psi(nxt)>
                # Match sectors: sector_cur + shift = sector_nxt (if perm)
                mat = np.zeros((manifold_size, manifold_size), dtype=np.complex128)
                # Build column offsets in st_cur and st_nxt
                cur_offsets, nxt_offsets = [], []
                off = 0
                for _, cols in st_cur:
                    cur_offsets.append((off, off + cols.shape[1]))
                    off += cols.shape[1]
                off = 0
                for _, cols in st_nxt:
                    nxt_offsets.append((off, off + cols.shape[1]))
                    off += cols.shape[1]

                shape = b_cur.geometry.shape
                # Compute L_{dst, src} = <Psi_next | O | Psi_cur>, then links = L^dag = <Psi_cur | Psi_next>
                l_mat = np.zeros((manifold_size, manifold_size), dtype=np.complex128)
                for (bas_dst, cols_dst), (rd0, rd1) in zip(st_nxt, nxt_offsets, strict=True):
                    for (bas_src, cols_src), (rs0, rs1) in zip(st_cur, cur_offsets, strict=True):
                        if perm is None:
                            if bas_dst.sector != bas_src.sector:
                                continue
                        else:
                            exp_dst = (
                                (bas_src.sector[0] + (n_particles if axis == 0 else 0)) % shape[0],
                                (bas_src.sector[1] + (n_particles if axis == 1 else 0)) % shape[1],
                            )
                            if bas_dst.sector != exp_dst:
                                continue
                        acted = _apply_fock_overlap(
                            b_nxt.frames, b_cur.frames, bas_dst, bas_src, cols_src,
                            perm_k=perm, seam_mat=seams,
                        )
                        l_mat[rd0:rd1, rs0:rs1] = cols_dst.conj().T @ acted
                links[ix, iy, axis] = l_mat.conj().T  # U(cur)^dag U(next)

    min_sv = float(np.min(np.linalg.svd(links, compute_uv=False)))
    sign, _ = np.linalg.slogdet(links)
    loop = sign[:, :, 0] * np.roll(sign[:, :, 1], -1, axis=0)
    loop /= np.roll(sign[:, :, 0], -1, axis=1) * sign[:, :, 1]
    flux = np.angle(loop)
    total_chern = float(flux.sum() / (2.0 * np.pi))

    # Wilson loop along axis 0 and axis 1
    pumps = {}
    u_links = np.empty_like(links)
    left, _, right = np.linalg.svd(links)
    u_links = left @ right
    for ax in (0, 1):
        trans = 1 - ax
        cnt = ny if ax == 0 else nx
        w_phases = np.empty(cnt, dtype=np.float64)
        for j in range(cnt):
            w = np.eye(manifold_size, dtype=np.complex128)
            for i in range(nx if ax == 0 else ny):
                w = w @ (u_links[i, j, 0] if ax == 0 else u_links[j, i, 1])
            w_phases[j] = np.angle(np.linalg.det(w))
        inc = np.angle(np.exp(1j * (np.roll(w_phases, -1) - w_phases)))
        orient = -1.0 if ax == 0 else 1.0
        pumps[ax] = {
            "total_chern_from_pump": float(orient * inc.sum() / (2.0 * np.pi)),
            "averaged_pumped_charge": float(orient * inc.sum() / (2.0 * np.pi * manifold_size)),
            "phases": w_phases,
            "max_step": float(np.max(np.abs(inc))),
        }

    return {
        "total_chern": total_chern,
        "integer_chern": int(np.rint(total_chern)),
        "averaged_hall_conductivity": float(np.rint(total_chern) / manifold_size),
        "minimum_manifold_gap": float(gaps.min()),
        "minimum_link_singular_value": min_sv,
        "maximum_plaquette_flux": float(np.max(np.abs(flux))),
        "pumps": pumps,
        "gaps": gaps,
    }


def resta_polarization_matrix(
    band: BlochBand,
    multiplet_states: list[tuple[SectorBasis, np.ndarray]],
    pump_axis: int = 0,
) -> np.ndarray:
    """Resta many-body polarization matrix Z_{alpha,beta} = <Psi_alpha | Gamma(rho_{q0}) | Psi_beta>."""
    if pump_axis not in (0, 1):
        raise ValueError("pump_axis must be 0 or 1")
    if not multiplet_states:
        raise ValueError("Non-empty multiplet_states required")
    q0 = np.array([1, 0] if pump_axis == 0 else [0, 1], dtype=np.int64)
    target_k, _ = band.geometry.shift(q0)
    ff = band.form_factor(q0)
    m_blocks = np.empty_like(ff)
    m_blocks[target_k] = ff
    q_snf = band.geometry.to_snf_coords(q0)
    shape = band.geometry.shape
    n_e = multiplet_states[0][0].n_particles

    offsets: list[tuple[int, int]] = []
    cols_list: list[np.ndarray] = []
    off = 0
    for _, cols in multiplet_states:
        c = cols[:, None] if cols.ndim == 1 else np.asarray(cols, dtype=np.complex128)
        offsets.append((off, off + c.shape[1]))
        cols_list.append(c)
        off += c.shape[1]

    z_mat = np.zeros((off, off), dtype=np.complex128)
    for (bas_dst, _), cols_dst, (rd0, rd1) in zip(multiplet_states, cols_list, offsets, strict=True):
        for (bas_src, _), cols_src, (rs0, rs1) in zip(multiplet_states, cols_list, offsets, strict=True):
            expected_dst = (
                (bas_src.sector[0] + n_e * int(q_snf[0])) % shape[0],
                (bas_src.sector[1] + n_e * int(q_snf[1])) % shape[1],
            )
            if bas_dst.sector != expected_dst:
                continue
            acted = _apply_per_k_blocks(m_blocks, bas_dst, bas_src, cols_src, perm_k=target_k)
            z_mat[rd0:rd1, rs0:rs1] = cols_dst.conj().T @ acted
    return z_mat


def thouless_charge_pump_1d(
    model_builder: Callable[[tuple[float, float]], tuple[BlochBand, BandProjectedHamiltonian]],
    n_particles: int,
    n_steps: int = 12,
    flux_axis: int | str = "auto",
    manifold_size: int = 3,
    sectors: Any = "all",
    nev: int = 4,
    mode: str = "auto",
    num_threads: int = 1,
    sector_workers: int = 1,
) -> dict[str, Any]:
    """1D boundary-flux insertion Resta-Thouless many-body charge pump.

    Sweeps phi in [0, 2*pi) along ``flux_axis`` (fixing the transverse twist to 0)
    and evaluates det(Z_{q0}(phi)) for ``q0 = (1, 0)`` if ``flux_axis == 1`` or
    ``q0 = (0, 1)`` if ``flux_axis == 0``. When ``N_e % N_{pump} == 0``, the
    pumped fractional charge equals C_total / manifold_size.
    """
    band0, ham0 = model_builder((0.0, 0.0))
    shape = ham0.geometry.shape
    if flux_axis == "auto":
        if n_particles % shape[0] == 0:
            ax_flux = 1
        elif n_particles % shape[1] == 0:
            ax_flux = 0
        else:
            ax_flux = 1
    else:
        ax_flux = int(flux_axis)
        if ax_flux not in (0, 1):
            raise ValueError("flux_axis must be 0, 1, or 'auto'")
    ax_pump = 1 - ax_flux

    shared_catalog = MomentumSectorCatalog(
        ham0.n_orbitals,
        shape,
        ham0.orbital_momenta,
        max_particles=n_particles,
    )
    steps = int(n_steps)
    thetas = 2.0 * np.pi * np.arange(steps, dtype=np.float64) / steps
    dets = np.empty(steps, dtype=np.complex128)
    gaps = np.empty(steps, dtype=np.float64)
    flow_by_sector: dict[tuple[int, int], list[np.ndarray]] = {}

    prev_v0: dict[tuple[int, int], np.ndarray] | None = None
    rng_seed = np.random.default_rng(4242)
    for j, th in enumerate(thetas):
        tw = (float(th), 0.0) if ax_flux == 0 else (0.0, float(th))
        if j == 0:
            band, ham = band0, ham0
        else:
            band, ham = model_builder(tw)
        ed = BandProjectedED(
            ham,
            n_particles=n_particles,
            num_threads=num_threads,
            catalog=shared_catalog,
        )
        scan = ed.scan_sectors(
            sectors=sectors,
            nev=nev,
            mode=mode,
            sector_workers=sector_workers,
            v0_by_sector=prev_v0,
        )
        prev_v0 = {}
        for s, res in scan["sectors"].items():
            flow_by_sector.setdefault(s, []).append(res.energies)
            if res.dimension > 0 and res.states.shape[1] > 0:
                seed = res.states.sum(axis=1)
                noise = 0.05 * (rng_seed.normal(size=res.dimension) + 1j * rng_seed.normal(size=res.dimension))
                prev_v0[s] = seed + noise / np.sqrt(max(1, res.dimension))

        levels = scan["ordered_levels"]
        if len(levels) < manifold_size:
            raise ValueError("Not enough eigenvalues to form requested ground manifold")
        gaps[j] = (
            levels[manifold_size][0] - levels[manifold_size - 1][0]
            if len(levels) > manifold_size
            else np.inf
        )
        by_sec: dict[tuple[int, int], list[int]] = {}
        for _, s, lvl in levels[:manifold_size]:
            by_sec.setdefault(s, []).append(lvl)
        sel = [
            (scan["sectors"][s].basis, scan["sectors"][s].states[:, lvls])
            for s, lvls in sorted(by_sec.items())
        ]
        z_mat = resta_polarization_matrix(band, sel, pump_axis=ax_pump)
        dets[j] = np.linalg.det(z_mat)

    dets_closed = np.append(dets, dets[0])
    increments = np.angle(np.exp(1j * np.diff(np.angle(dets_closed))))
    orientation = 1.0 if ax_flux == 1 else -1.0
    total_chern = float(orientation * np.sum(increments) / (2.0 * np.pi))
    int_chern = int(np.rint(total_chern))
    pumped_cycle = np.r_[0.0, np.cumsum(orientation * increments)] / (2.0 * np.pi * manifold_size)
    max_step = float(np.max(np.abs(increments)))
    min_det = float(np.min(np.abs(dets)))
    min_gap = float(gaps.min())
    resolved = bool(
        abs(total_chern - int_chern) < 1e-3
        and max_step < np.pi - 0.1
        and min_det > 1e-6
        and min_gap > 1e-8
    )

    return {
        "total_chern": total_chern,
        "integer_chern": int_chern,
        "averaged_hall_conductivity": float(int_chern / manifold_size),
        "flux_axis": ax_flux,
        "pump_axis": ax_pump,
        "thetas": thetas,
        "det_Z": dets,
        "pumped_cycle": pumped_cycle,
        "increments": increments,
        "maximum_phase_step": max_step,
        "minimum_manifold_gap": min_gap,
        "minimum_abs_det_Z": min_det,
        "resolved": resolved,
        "spectral_flow_by_sector": {s: np.asarray(v) for s, v in flow_by_sector.items()},
        "gaps": gaps,
    }


def flux_spectral_flow(
    model_builder: Callable[[tuple[float, float]], tuple[BlochBand, BandProjectedHamiltonian]],
    n_particles: int,
    n_steps: int = 16,
    axis: int = 1,
    sectors: Any = "all",
    nev: int = 4,
    mode: str = "auto",
    num_threads: int = 1,
) -> dict[str, Any]:
    """Compute momentum-resolved spectral flow under 0..2*pi boundary flux insertion."""
    thetas = np.linspace(0.0, 2.0 * np.pi, int(n_steps) + 1)
    by_sector: dict[tuple[int, int], list[np.ndarray]] = {}
    global_levels = []
    for th in thetas:
        tw = (float(th), 0.0) if axis == 0 else (0.0, float(th))
        _, ham = model_builder(tw)
        ed = BandProjectedED(ham, n_particles=n_particles, num_threads=num_threads)
        scan = ed.scan_sectors(sectors=sectors, nev=nev, mode=mode)
        for s, res in scan["sectors"].items():
            by_sector.setdefault(s, []).append(res.energies)
        global_levels.append([e for e, _, _ in scan["ordered_levels"][: max(12, nev * 2)]])
    return {
        "thetas": thetas,
        "by_sector": {s: np.asarray(v) for s, v in by_sector.items()},
        "global_levels": np.asarray(global_levels),
    }


def plot_ed_spectrum(
    scan_result: dict[str, Any],
    save_path: str | Path,
    title: str = "Momentum-Resolved ED Spectrum",
    highlight_ground: int = 3,
    energy_unit: str = "meV",
) -> Path:
    """Save momentum-resolved ED spectrum strictly as an SVG file."""
    path = Path(save_path)
    if path.suffix.lower() != ".svg":
        raise ValueError(f"Only .svg figures are allowed, got {path.suffix!r}")
    path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sectors = sorted(scan_result["sectors"].keys())
    n1, n2 = scan_result["shape"]
    e0 = float(scan_result["ground_energy"])
    top_sectors = {(s, lvl) for _, s, lvl in scan_result["ordered_levels"][:highlight_ground]}

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    xs = [s[0] * n2 + s[1] for s in sectors]
    for s, x in zip(sectors, xs, strict=True):
        es = scan_result["sectors"][s].energies - e0
        for lvl, e in enumerate(es):
            if (s, lvl) in top_sectors:
                ax.scatter([x], [e], color="#d95f02", s=46, zorder=4, edgecolors="#222222", linewidths=0.6)
            else:
                ax.scatter([x], [e], color="#3b6ea5", s=26, alpha=0.85, zorder=3)
    ax.set_xlabel(r"Total momentum sector index $k_1 N_2 + k_2$")
    ax.set_ylabel(rf"$E - E_0$ ({energy_unit})" if energy_unit else r"$E - E_0$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path
