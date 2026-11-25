"""Projected two-body interaction assembly and standard momentum-space potentials."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable
import numpy as np

from .bloch import BlochBand
from .lattice import TorusGeometry

E2_OVER_4PI_EPS0_MEV_ANG = 14399.64547842567  # e^2 / (4 pi eps0) in meV * Angstrom


def bare_coulomb(
    q_cart: np.ndarray,
    area: float,
    dielectric: float = 1.0,
    e2_prefactor: float = E2_OVER_4PI_EPS0_MEV_ANG,
) -> np.ndarray:
    """2D bare Coulomb interaction V(q) = 2*pi * e2 / (eps * A * |q|), zero at q=0."""
    if area <= 0.0 or dielectric <= 0.0:
        raise ValueError("Positive area and dielectric required")
    q = np.linalg.norm(np.asarray(q_cart, dtype=np.float64), axis=-1)
    return np.divide(
        2.0 * np.pi * e2_prefactor / (dielectric * area),
        q,
        out=np.zeros_like(q, dtype=np.float64),
        where=q > 0.0,
    )


def dual_gate_coulomb(
    q_cart: np.ndarray,
    area: float,
    dielectric: float = 15.0,
    gate_distance: float = 300.0,
    e2_prefactor: float = E2_OVER_4PI_EPS0_MEV_ANG,
    background: bool = True,
) -> np.ndarray:
    """Dual-gate screened Coulomb V(q) = 2*pi * e2 * tanh(q*d) / (eps * A * q)."""
    if area <= 0.0 or dielectric <= 0.0 or gate_distance < 0.0:
        raise ValueError("Positive area/dielectric and nonnegative gate distance required")
    q = np.linalg.norm(np.asarray(q_cart, dtype=np.float64), axis=-1)
    screened = np.full_like(q, gate_distance, dtype=np.float64)
    np.divide(np.tanh(q * gate_distance), q, out=screened, where=q > 0.0)
    val = 2.0 * np.pi * e2_prefactor * screened / (dielectric * area)
    return np.where(q == 0.0, 0.0, val) if background else val


def haldane_pseudopotential_v1(
    q_cart: np.ndarray,
    ell: float,
    area: float,
    v1: float = 1.0,
    include_q0: bool = True,
) -> np.ndarray:
    """LLL Haldane V_1 pseudopotential V(q) = (2*pi*ell^2 / A) * v1 * (1 - q^2 ell^2).

    With ``include_q0=True``, the N-particle Laughlin state at nu=1/3 has exact zero energy
    in the limit of large q*ell cutoff. Note that L_1(q^2 ell^2) = 1 - q^2 ell^2 gives
    eigenvalue +v1 in the m=1 relative angular momentum channel and 0 in m != 1.
    """
    if ell <= 0.0 or area <= 0.0:
        raise ValueError("Positive ell and area required")
    q2 = np.sum(np.asarray(q_cart, dtype=np.float64) ** 2, axis=-1)
    val = (2.0 * np.pi * (ell**2) / area) * v1 * (1.0 - q2 * (ell**2))
    if not include_q0:
        val = np.where(q2 == 0.0, 0.0, val)
    return val


@dataclass(frozen=True)
class PairBlock:
    """Antisymmetrized pair scattering block at conserved total momentum P (and QN)."""

    momentum_code: int
    pairs: np.ndarray  # (n_pairs, 2) int64 with pairs[:, 0] < pairs[:, 1]
    matrix: np.ndarray  # (n_pairs, n_pairs) complex128 Hermitian W^{(P)}
    diag: np.ndarray  # (n_pairs,) float64 diagonal elements Re(W_{aa})
    offdiag: np.ndarray  # (n_pairs, n_pairs) complex128 with zero diagonal


class BandProjectedHamiltonian:
    """Momentum-conserving projected two-body Hamiltonian in an arbitrary Bloch manifold.

    The operator is:
        H = sum_{k,a,b} h_{k,ab} c_{k,a}^dag c_{k,b}
          + sum_P sum_{alpha,beta in pairs(P)} W^{(P)}_{alpha,beta}
                c_{i1(alpha)}^dag c_{i2(alpha)}^dag c_{j2(beta)} c_{j1(beta)}
    with strictly ordered fermionic orbitals ``0 <= i1 < i2 < N_orb``.
    """

    def __init__(
        self,
        geometry: TorusGeometry,
        nbands: int,
        one_body_blocks: np.ndarray,
        pair_blocks: dict[int, PairBlock],
        orbital_momenta: np.ndarray | None = None,
        orbital_qn: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.geometry = geometry
        self.nbands = int(nbands)
        self.n_orbitals = int(geometry.size * self.nbands)
        h1 = np.asarray(one_body_blocks, dtype=np.complex128)
        if h1.shape != (geometry.size, self.nbands, self.nbands):
            raise ValueError(f"Expected one_body_blocks ({geometry.size}, {nbands}, {nbands})")
        defect = np.max(np.abs(h1 - h1.conj().swapaxes(-1, -2)), initial=0.0)
        if defect > 1e-9:
            raise ValueError(f"One-body Hamiltonian blocks are not Hermitian ({defect:g})")
        self.one_body_blocks = 0.5 * (h1 + h1.conj().swapaxes(-1, -2))
        self.one_body_diag = np.diagonal(self.one_body_blocks, axis1=-2, axis2=-1).real.ravel()
        self.pair_blocks = dict(pair_blocks)
        self.orbital_momenta = (
            np.repeat(geometry.indices, self.nbands, axis=0)
            if orbital_momenta is None
            else np.asarray(orbital_momenta, dtype=np.int64)
        )
        self.orbital_qn = None if orbital_qn is None else np.asarray(orbital_qn, dtype=np.int64)
        self.metadata = dict(metadata or {})

    @classmethod
    def from_band(
        cls,
        band: BlochBand,
        transfers: np.ndarray,
        weights: np.ndarray,
        vertices: np.ndarray | None = None,
        tol: float = 1e-9,
    ) -> "BandProjectedHamiltonian":
        """Build exact momentum-blocked pair Hamiltonian from a BlochBand and V(q)."""
        qs = np.asarray(transfers, dtype=np.int64)
        ws = np.asarray(weights, dtype=np.float64)
        if qs.ndim != 2 or qs.shape[1] != 2 or ws.shape != (len(qs),):
            raise ValueError("Expected matching transfers (n_q, 2) and weights (n_q,)")
        if not np.all(np.isfinite(ws)):
            raise ValueError("Finite interaction weights required")

        # Verify symmetric (+q, -q) list with matching weights
        idx_map = {(int(q[0]), int(q[1])): idx for idx, q in enumerate(qs)}
        if len(idx_map) != len(qs):
            raise ValueError("Transfers must not contain duplicates")
        for idx, q in enumerate(qs):
            other = idx_map.get((-int(q[0]), -int(q[1])))
            if other is None or not np.isclose(ws[idx], ws[other], atol=1e-13, rtol=1e-12):
                raise ValueError("Symmetric (+q, -q) transfers with equal weights required")

        if vertices is None:
            vertices = band.form_factors(qs)
        else:
            vertices = np.asarray(vertices, dtype=np.complex128)
        n_k, b = band.geometry.size, band.nbands
        n_orb = n_k * b
        if vertices.shape != (len(qs), n_k, b, b):
            raise ValueError("vertices shape mismatch")

        # Enumerate all ordered pairs (i < j) grouped by conserved total SNF momentum P
        all_pairs = np.asarray(list(combinations(range(n_orb), 2)), dtype=np.int64)
        orb_snf = np.repeat(band.geometry.indices, b, axis=0)
        pair_mom = (orb_snf[all_pairs[:, 0]] + orb_snf[all_pairs[:, 1]]) % band.geometry.shape
        n2_dim = band.geometry.shape[1]
        pair_p_codes = pair_mom[:, 0] * n2_dim + pair_mom[:, 1]

        # Precompute mapping from ordered pair (i, j) to (P, local_idx)
        order_p = np.argsort(pair_p_codes, kind="mergesort")
        sorted_p = pair_p_codes[order_p]
        sorted_pairs = all_pairs[order_p]
        p_starts = np.searchsorted(sorted_p, np.arange(n_k + 1))

        pair_to_block = np.full((n_orb, n_orb), -1, dtype=np.int64)
        pair_to_local = np.full((n_orb, n_orb), -1, dtype=np.int64)
        block_matrices: dict[int, np.ndarray] = {}
        block_pairs: dict[int, np.ndarray] = {}
        for p_code in range(n_k):
            lo, hi = int(p_starts[p_code]), int(p_starts[p_code + 1])
            p_arr = sorted_pairs[lo:hi]
            block_pairs[p_code] = p_arr
            block_matrices[p_code] = np.zeros((len(p_arr), len(p_arr)), dtype=np.complex128)
            if len(p_arr):
                pair_to_block[p_arr[:, 0], p_arr[:, 1]] = p_code
                pair_to_local[p_arr[:, 0], p_arr[:, 1]] = np.arange(len(p_arr), dtype=np.int64)

        # Group transfers by SNF residue mod geometry.shape
        q_snf = qs @ band.geometry._snf_u.T
        residues = q_snf % band.geometry.shape
        shape_arr = np.asarray(band.geometry.shape, dtype=np.int64)

        for residue in np.unique(residues, axis=0):
            sel = np.flatnonzero(np.all(residues == residue[None, :], axis=1))
            # Gram matrix G[(k_1, a_k, a_i), (k_2, a_j, a_l)]
            # = sum_{q in sel} w_q * Lambda^*_{a_k, a_i}(k_1, q) * Lambda_{a_j, a_l}(k_2, q)
            vecs = vertices[sel].reshape(len(sel), -1)
            gram = (vecs.conj().T @ (ws[sel, None] * vecs)).reshape(n_k, b, b, n_k, b, b)

            # k_i -> k_k = [k_i + res], k_l -> k_j = [k_l + res]
            # Equally, iterate over annihilated pairs (k < l) grouped by P!
            # Given annihilated (k, l) with k_k, k_l:
            #   direct: i_k = [k_k - res], j_k = [k_l + res]
            #   swap:   j_k = [k_k + res], i_k = [k_l - res]
            res_tuple = (int(residue[0]), int(residue[1]))
            # Shift indices directly in SNF coords
            k_grid = band.geometry.indices
            plus_k = ((k_grid + res_tuple) % shape_arr)
            plus_idx = plus_k[:, 0] * n2_dim + plus_k[:, 1]
            minus_k = ((k_grid - res_tuple) % shape_arr)
            minus_idx = minus_k[:, 0] * n2_dim + minus_k[:, 1]

            for p_code in range(n_k):
                p_arr = block_pairs[p_code]
                if len(p_arr) == 0:
                    continue
                w_mat = block_matrices[p_code]
                col_ids = np.arange(len(p_arr), dtype=np.int64)
                left0, right0 = p_arr[:, 0], p_arr[:, 1]
                for ann_left, ann_right, sign_ann in ((left0, right0, 1.0), (right0, left0, -1.0)):
                    k1, a_k = ann_left // b, ann_left % b
                    k2, a_l = ann_right // b, ann_right % b
                    ki = minus_idx[k1]
                    kj = plus_idx[k2]
                    # Contribution to H_{row, col}: (sign_ann / 2) * gram[ki, a_k, a_i, k2, a_j, a_l] * sgn(i < j)
                    sub_g = gram[ki, :, :, k2, :, a_l]  # shape (n_pairs_P, a_k_all? Wait: index pairs!)
                    # Wait: indexing with ki (n_P,), k2 (n_P,), a_k (n_P,), a_l (n_P,) gives shape (n_P, b_i, b_j)!
                    sub_g = gram[ki, a_k, :, k2, :, a_l]  # shape (n_P, b, b) -> axes (pair, a_i, a_j)
                    for a_i in range(b):
                        for a_j in range(b):
                            i_orb = ki * b + a_i
                            j_orb = kj * b + a_j
                            valid = i_orb != j_orb
                            if not np.any(valid):
                                continue
                            lo_orb = np.minimum(i_orb[valid], j_orb[valid])
                            hi_orb = np.maximum(i_orb[valid], j_orb[valid])
                            sgn_create = np.where(i_orb[valid] < j_orb[valid], 1.0, -1.0)
                            row_ids = pair_to_local[lo_orb, hi_orb]
                            vals = (0.5 * sign_ann) * sgn_create * sub_g[valid, a_i, a_j]
                            np.add.at(w_mat, (row_ids, col_ids[valid]), vals)

        pair_blocks: dict[int, PairBlock] = {}
        for p_code, p_arr in block_pairs.items():
            w_mat = block_matrices[p_code]
            defect = float(np.max(np.abs(w_mat - w_mat.conj().T), initial=0.0))
            if defect > tol:
                raise ArithmeticError(
                    f"Pair block P={p_code} failed Hermiticity check (defect={defect:g})"
                )
            w_mat = 0.5 * (w_mat + w_mat.conj().T)
            diag = np.diagonal(w_mat).real.copy()
            off = w_mat.copy()
            np.fill_diagonal(off, 0.0)
            pair_blocks[p_code] = PairBlock(
                momentum_code=p_code,
                pairs=p_arr,
                matrix=w_mat,
                diag=diag,
                offdiag=off,
            )

        qn = (
            band.internal_space.quantum_numbers
            if band.internal_space.quantum_numbers is not None
            else None
        )
        return cls(
            geometry=band.geometry,
            nbands=band.nbands,
            one_body_blocks=band.one_body_blocks,
            pair_blocks=pair_blocks,
            orbital_momenta=orb_snf,
            orbital_qn=None if qn is None else np.tile(qn, (n_k, 1)),
            metadata={
                **band.metadata,
                "n_transfers": int(len(qs)),
            },
        )

    @classmethod
    def from_potential(
        cls,
        band: BlochBand,
        potential_fn: Callable[[np.ndarray, float], np.ndarray],
        qell_cutoff: float,
        include_q0: bool = False,
        tol: float = 1e-9,
    ) -> "BandProjectedHamiltonian":
        """Convenience constructor using a transfer disk ``|q|*ell <= qell_cutoff``."""
        transfers = band.geometry.transfer_disk(qell_cutoff)
        if include_q0:
            transfers = np.vstack([np.array([[0, 0]], dtype=np.int64), transfers])
        q_cart = band.geometry.transfer_cart(transfers)
        weights = np.asarray(potential_fn(q_cart, band.geometry.area), dtype=np.float64)
        return cls.from_band(band, transfers, weights, tol=tol)

    @classmethod
    def from_four_fermion_tensor(
        cls,
        geometry: TorusGeometry,
        nbands: int,
        one_body_blocks: np.ndarray,
        tensor: np.ndarray,
        tol: float = 1e-9,
    ) -> "BandProjectedHamiltonian":
        """Construct from arbitrary 4-fermion tensor ``H_2 = sum V[i,j,k,l] c_i^dag c_j^dag c_l c_k``."""
        v = np.asarray(tensor, dtype=np.complex128)
        n_orb = geometry.size * nbands
        if v.shape != (n_orb, n_orb, n_orb, n_orb):
            raise ValueError(f"Expected 4-fermion tensor of shape {(n_orb,)*4}")
        all_pairs = np.asarray(list(combinations(range(n_orb), 2)), dtype=np.int64)
        orb_snf = np.repeat(geometry.indices, nbands, axis=0)
        pair_mom = (orb_snf[all_pairs[:, 0]] + orb_snf[all_pairs[:, 1]]) % geometry.shape
        p_codes = pair_mom[:, 0] * geometry.shape[1] + pair_mom[:, 1]

        # Antisymmetrize: W_{(i<j), (k<l)} = V[i,j,k,l] - V[i,j,l,k] - V[j,i,k,l] + V[j,i,l,k]
        i, j = all_pairs[:, 0], all_pairs[:, 1]
        w_full = (
            v[i[:, None], j[:, None], i[None, :], j[None, :]]
            - v[i[:, None], j[:, None], j[None, :], i[None, :]]
            - v[j[:, None], i[:, None], i[None, :], j[None, :]]
            + v[j[:, None], i[:, None], j[None, :], i[None, :]]
        )
        permitted = p_codes[:, None] == p_codes[None, :]
        if np.max(np.abs(w_full[~permitted]), initial=0.0) > tol:
            raise ValueError("Supplied 4-fermion tensor violates momentum conservation")
        defect = float(np.max(np.abs(w_full - w_full.conj().T), initial=0.0))
        if defect > tol:
            raise ValueError(f"Antisymmetrized pair tensor is not Hermitian ({defect:g})")

        pair_blocks: dict[int, PairBlock] = {}
        for p_code in range(geometry.size):
            sel = np.flatnonzero(p_codes == p_code)
            w_p = 0.5 * (w_full[np.ix_(sel, sel)] + w_full[np.ix_(sel, sel)].conj().T)
            diag = np.diagonal(w_p).real.copy()
            off = w_p.copy()
            np.fill_diagonal(off, 0.0)
            pair_blocks[p_code] = PairBlock(
                momentum_code=p_code,
                pairs=all_pairs[sel],
                matrix=w_p,
                diag=diag,
                offdiag=off,
            )
        return cls(geometry=geometry, nbands=nbands, one_body_blocks=one_body_blocks, pair_blocks=pair_blocks)
