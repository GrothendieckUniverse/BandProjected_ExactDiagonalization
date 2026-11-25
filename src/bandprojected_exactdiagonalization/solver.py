"""Exact diagonalization solvers: dense, sparse CSR, and BLAS-3 ZGEMM matrix-free."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import pickle
import time
from typing import Any
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh

from .basis import (
    MomentumSectorCatalog,
    SectorBasis,
    make_below_masks,
    make_between_masks,
    popcount_u64,
)
from .interaction import BandProjectedHamiltonian


def _requested_threads(num_threads: int | None = None) -> int:
    if num_threads is not None:
        return max(1, min(int(num_threads), os.cpu_count() or int(num_threads)))
    raw = os.environ.get("BPED_NUM_THREADS", "1")
    try:
        req = int(raw)
    except ValueError as exc:
        raise ValueError(f"BPED_NUM_THREADS must be positive integer, got {raw!r}") from exc
    return max(1, min(req, os.cpu_count() or req))


def orthonormal_eigenstates(
    matvec_fn, states: np.ndarray, tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    """Rayleigh-Ritz orthonormalization inside the returned low-energy subspace."""
    s = np.asarray(states, dtype=np.complex128)
    if s.ndim != 2 or s.shape[1] == 0:
        raise ValueError("Expected non-empty 2D column array of eigenvectors")
    sv = np.linalg.svd(s, compute_uv=False)
    if sv.min() < tol:
        raise ValueError(f"Near-dependent returned eigenvectors (min sv={sv.min():g})")
    q, _ = np.linalg.qr(s, mode="reduced")
    hq = np.column_stack([matvec_fn(q[:, j]) for j in range(q.shape[1])])
    red = q.conj().T @ hq
    red = 0.5 * (red + red.conj().T)
    evals, rot = np.linalg.eigh(red)
    return evals.real, q @ rot


def hamiltonian_fingerprint(ham: BandProjectedHamiltonian, n_particles: int) -> str:
    """Deterministic SHA-256 fingerprint of one-body and pair-block Hamiltonian data."""
    h = hashlib.sha256()
    h.update(f"norb={ham.n_orbitals};np={n_particles};shape={ham.geometry.shape}".encode())
    h.update(np.ascontiguousarray(ham.one_body_blocks).view(np.uint8))
    for p_code in sorted(ham.pair_blocks):
        blk = ham.pair_blocks[p_code]
        h.update(np.ascontiguousarray(blk.pairs).view(np.uint8))
        h.update(np.ascontiguousarray(blk.matrix).view(np.uint8))
    return h.hexdigest()


@dataclass
class SectorResult:
    sector: tuple[int, int]
    energies: np.ndarray
    states: np.ndarray
    residuals: np.ndarray
    dimension: int
    mode: str
    basis: SectorBasis
    elapsed_seconds: float


class BandProjectedED:
    """High-performance pure-NumPy momentum-space ED solver for projected multi-band models."""

    def __init__(
        self,
        hamiltonian: BandProjectedHamiltonian,
        n_particles: int,
        num_threads: int | None = None,
        dense_threshold: int = 256,
        chunk_size: int = 16384,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.n_particles = int(n_particles)
        if not 0 <= self.n_particles <= hamiltonian.n_orbitals:
            raise ValueError("Particle number outside [0, n_orbitals]")
        self.num_threads = _requested_threads(num_threads)
        self.dense_threshold = int(dense_threshold)
        self.chunk_size = int(chunk_size)

        self.catalog = MomentumSectorCatalog(
            hamiltonian.n_orbitals,
            hamiltonian.geometry.shape,
            hamiltonian.orbital_momenta,
            max_particles=self.n_particles,
        )
        self._between = make_between_masks(hamiltonian.n_orbitals)
        self._below = make_below_masks(hamiltonian.n_orbitals)
        self._one_body_offdiag = self._extract_one_body_hops()
        self.fingerprint = hamiltonian_fingerprint(hamiltonian, self.n_particles)

    def _extract_one_body_hops(self) -> list[tuple[int, int, complex]]:
        b = self.hamiltonian.nbands
        hops: list[tuple[int, int, complex]] = []
        for k_idx, blk in enumerate(self.hamiltonian.one_body_blocks):
            base = k_idx * b
            for a1 in range(b):
                for a2 in range(b):
                    if a1 == a2:
                        continue
                    val = complex(blk[a1, a2])
                    if abs(val) > 1e-13:
                        hops.append((base + a1, base + a2, val))
        return hops

    def sector_diagonal(self, basis: SectorBasis) -> np.ndarray:
        """Vectorized diagonal energy (1-body diag + diagonal pair terms) for every mask."""
        masks = basis.masks
        d = len(masks)
        diag = np.zeros(d, dtype=np.float64)
        if d == 0:
            return diag
        h1_d = self.hamiltonian.one_body_diag
        for orb in np.flatnonzero(np.abs(h1_d) > 0.0):
            occ = (masks & np.uint64(1 << int(orb))) != 0
            diag[occ] += h1_d[orb]
        for blk in self.hamiltonian.pair_blocks.values():
            if len(blk.pairs) == 0:
                continue
            nz = np.flatnonzero(np.abs(blk.diag) > 1e-14)
            for idx in nz:
                i, j = int(blk.pairs[idx, 0]), int(blk.pairs[idx, 1])
                pmask = np.uint64((1 << i) | (1 << j))
                diag[(masks & pmask) == pmask] += blk.diag[idx]
        return diag

    def assemble_sparse_sector(self, basis: SectorBasis, threshold: float = 1e-13) -> sparse.csr_matrix:
        """Assemble exact Hermitian sparse CSR Hamiltonian matrix for a momentum sector."""
        masks = basis.masks
        d = len(masks)
        diag = self.sector_diagonal(basis)
        rows = [np.arange(d, dtype=np.int64)]
        cols = [np.arange(d, dtype=np.int64)]
        vals = [diag.astype(np.complex128)]
        if d == 0 or self.n_particles < 1:
            return sparse.csr_matrix((d, d), dtype=np.complex128)

        # Off-diagonal one-body terms inside each k-block
        for i_to, i_from, t_val in self._one_body_offdiag:
            m_from = np.uint64(1 << i_from)
            m_to = np.uint64(1 << i_to)
            src = np.flatnonzero(((masks & m_from) != 0) & ((masks & m_to) == 0))
            if len(src) == 0:
                continue
            new_masks = masks[src] ^ (m_from | m_to)
            dst = basis.lookup.find_valid(new_masks)
            parity = popcount_u64(masks[src] & self._between[i_from, i_to]) & 1
            rows.append(dst)
            cols.append(src)
            vals.append(t_val * (1.0 - 2.0 * parity))

        # Off-diagonal two-body pair terms grouped by P
        if self.n_particles >= 2:
            def process_block(blk):
                if len(blk.pairs) == 0:
                    return None
                b_rows, b_cols, b_vals = [], [], []
                pair_masks = (np.uint64(1) << blk.pairs[:, 0].astype(np.uint64)) | (
                    np.uint64(1) << blk.pairs[:, 1].astype(np.uint64)
                )
                for col_p, (k, ell) in enumerate(blk.pairs):
                    k, ell = int(k), int(ell)
                    pm = pair_masks[col_p]
                    src = np.flatnonzero((masks & pm) == pm)
                    if len(src) == 0:
                        continue
                    rem = masks[src] ^ pm
                    p_rem = popcount_u64(masks[src] & self._between[k, ell]) & 1
                    nz_rows = np.flatnonzero(np.abs(blk.offdiag[:, col_p]) > threshold)
                    for row_p in nz_rows:
                        i, j = int(blk.pairs[row_p, 0]), int(blk.pairs[row_p, 1])
                        cm = pair_masks[row_p]
                        allowed = (rem & cm) == 0
                        if not np.any(allowed):
                            continue
                        rest = rem[allowed]
                        tgt_masks = rest | cm
                        dst = basis.lookup.find_valid(tgt_masks)
                        parity = p_rem[allowed] ^ (popcount_u64(rest & self._between[i, j]) & 1)
                        b_rows.append(dst)
                        b_cols.append(src[allowed])
                        b_vals.append(blk.offdiag[row_p, col_p] * (1.0 - 2.0 * parity))
                if not b_rows:
                    return None
                return np.concatenate(b_rows), np.concatenate(b_cols), np.concatenate(b_vals)

            blocks = [blk for blk in self.hamiltonian.pair_blocks.values() if len(blk.pairs) > 0]
            if self.num_threads == 1 or len(blocks) <= 1:
                results = map(process_block, blocks)
            else:
                with ThreadPoolExecutor(max_workers=min(self.num_threads, len(blocks))) as pool:
                    results = list(pool.map(process_block, blocks))
            for res in results:
                if res is not None:
                    rows.append(res[0])
                    cols.append(res[1])
                    vals.append(res[2])

        mat = sparse.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(d, d),
            dtype=np.complex128,
        ).tocsr()
        mat.sum_duplicates()
        defect = mat - mat.conj().T
        if np.max(np.abs(defect.data), initial=0.0) > 1e-8:
            raise ArithmeticError("Assembled sector sparse Hamiltonian failed Hermiticity")
        return mat

    def make_matrixfree_operator(self, basis: SectorBasis) -> tuple[LinearOperator, np.ndarray]:
        """BLAS-3 intermediate-state ZGEMM matrix-free LinearOperator (memory-bounded).

        For each pair-momentum block P, iterates over chunks of the intermediate
        (N_e - 2, K - P) basis, gathers X[r, beta] = s(r, beta) * v[r U beta],
        applies ALL off-diagonal scatterings via a single matrix multiply Y = X @ W_off_P.T,
        and scatters s(r, alpha) * Y[r, alpha] back into out.
        """
        d = len(basis)
        diag = self.sector_diagonal(basis)
        n1, n2 = self.hamiltonian.geometry.shape
        s1, s2 = basis.sector

        # Precompute (inter_basis, pair_masks, active_blocks) only once per sector
        active_blocks = []
        if self.n_particles >= 2:
            for blk in self.hamiltonian.pair_blocks.values():
                if len(blk.pairs) == 0 or np.max(np.abs(blk.offdiag), initial=0.0) <= 1e-14:
                    continue
                p1, p2 = divmod(blk.momentum_code, n2)
                r_sector = ((s1 - p1) % n1, (s2 - p2) % n2)
                inter_basis = self.catalog.basis(self.n_particles - 2, r_sector)
                if len(inter_basis) == 0:
                    continue
                pmasks = (np.uint64(1) << blk.pairs[:, 0].astype(np.uint64)) | (
                    np.uint64(1) << blk.pairs[:, 1].astype(np.uint64)
                )
                btw = self._between[blk.pairs[:, 0], blk.pairs[:, 1]]
                active_blocks.append((blk, inter_basis.masks, pmasks, btw))

        # Precompute one-body off-diagonal maps if nbands > 1
        one_body_maps = []
        for i_to, i_from, t_val in self._one_body_offdiag:
            m_from = np.uint64(1 << i_from)
            m_to = np.uint64(1 << i_to)
            src = np.flatnonzero(((basis.masks & m_from) != 0) & ((basis.masks & m_to) == 0))
            if len(src) == 0:
                continue
            dst = basis.lookup.find_valid(basis.masks[src] ^ (m_from | m_to))
            parity = popcount_u64(basis.masks[src] & self._between[i_from, i_to]) & 1
            one_body_maps.append((src, dst, t_val * (1.0 - 2.0 * parity)))

        chunk = self.chunk_size
        n_threads = self.num_threads

        def apply_pair_block(item, vec: np.ndarray) -> np.ndarray:
            blk, r_masks, pmasks, btw = item
            out_local = np.zeros(d, dtype=np.complex128)
            n_p = len(blk.pairs)
            w_off_t = np.ascontiguousarray(blk.offdiag.T)
            n_r = len(r_masks)
            for start in range(0, n_r, chunk):
                stop = min(start + chunk, n_r)
                rc = r_masks[start:stop]
                m = stop - start
                x_mat = np.zeros((m, n_p), dtype=np.complex128)
                cached_targets: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = [None] * n_p
                # Gather over all pairs in block P
                for col in range(n_p):
                    allowed = (rc & pmasks[col]) == 0
                    pos = np.flatnonzero(allowed)
                    if len(pos) == 0:
                        continue
                    r_sub = rc[pos]
                    tgt_masks = r_sub | pmasks[col]
                    idx = basis.lookup.find_valid(tgt_masks)
                    sgn = 1.0 - 2.0 * (popcount_u64(r_sub & btw[col]) & 1)
                    x_mat[pos, col] = sgn * vec[idx]
                    cached_targets[col] = (pos, idx, sgn)
                # Single BLAS-3 ZGEMM over all pairs in block P
                y_mat = x_mat @ w_off_t
                # Scatter back
                for row in range(n_p):
                    item_tgt = cached_targets[row]
                    if item_tgt is None:
                        continue
                    pos, idx, sgn = item_tgt
                    np.add.at(out_local, idx, sgn * y_mat[pos, row])
            return out_local

        def matvec(v: np.ndarray) -> np.ndarray:
            vec = np.asarray(v, dtype=np.complex128)
            out = diag * vec
            for src, dst, amp in one_body_maps:
                np.add.at(out, dst, amp * vec[src])
            if not active_blocks:
                return out
            if n_threads == 1 or len(active_blocks) <= 1:
                for blk_item in active_blocks:
                    out += apply_pair_block(blk_item, vec)
            else:
                with ThreadPoolExecutor(max_workers=min(n_threads, len(active_blocks))) as pool:
                    for part in pool.map(lambda it: apply_pair_block(it, vec), active_blocks):
                        out += part
            return out

        op = LinearOperator((d, d), matvec=matvec, rmatvec=matvec, dtype=np.complex128)
        return op, diag

    def solve_sector(
        self,
        sector: tuple[int, int],
        nev: int = 5,
        mode: str = "auto",
        tol: float = 1e-10,
        v0: np.ndarray | None = None,
        maxiter: int = 5000,
    ) -> SectorResult:
        """Diagonalize a single momentum sector in ``dense``, ``matrix``, or ``matrixfree`` mode."""
        t0 = time.perf_counter()
        basis = self.catalog.basis(self.n_particles, sector)
        d = len(basis)
        s = basis.sector
        if d == 0:
            return SectorResult(
                sector=s,
                energies=np.empty(0, dtype=np.float64),
                states=np.empty((0, 0), dtype=np.complex128),
                residuals=np.empty(0, dtype=np.float64),
                dimension=0,
                mode="empty",
                basis=basis,
                elapsed_seconds=time.perf_counter() - t0,
            )

        k_req = min(int(nev), d)
        chosen = mode
        if chosen == "auto":
            if d <= self.dense_threshold or k_req >= d - 1:
                chosen = "dense"
            elif d <= 50000:
                chosen = "matrix"
            else:
                chosen = "matrixfree"

        if chosen == "dense" or k_req >= d - 1:
            mat = self.assemble_sparse_sector(basis).toarray()
            evals, evecs = np.linalg.eigh(mat)
            evals, evecs = evals[:k_req].real, evecs[:, :k_req]
            mv = lambda x: mat @ x
        elif chosen == "matrix":
            mat = self.assemble_sparse_sector(basis)
            zero = float(mat.diagonal().real.mean())
            centered = mat - zero * sparse.eye(d, format="csr", dtype=np.complex128)
            if v0 is None:
                rng = np.random.default_rng(1337 + s[0] * 1009 + s[1])
                v0 = rng.normal(size=d) + 1j * rng.normal(size=d)
            evals, evecs = eigsh(centered, k=k_req, which="SA", tol=tol, v0=v0, maxiter=maxiter)
            order = np.argsort(evals)
            evals, evecs = evals[order].real + zero, evecs[:, order]
            mv = lambda x: mat @ x
        elif chosen == "matrixfree":
            op, diag = self.make_matrixfree_operator(basis)
            zero = float(diag.mean())
            centered_op = LinearOperator(
                (d, d), matvec=lambda x: op @ x - zero * x, rmatvec=lambda x: op @ x - zero * x, dtype=np.complex128
            )
            if v0 is None:
                rng = np.random.default_rng(1337 + s[0] * 1009 + s[1])
                v0 = rng.normal(size=d) + 1j * rng.normal(size=d)
            evals, evecs = eigsh(centered_op, k=k_req, which="SA", tol=tol, v0=v0, maxiter=maxiter)
            order = np.argsort(evals)
            evals, evecs = evals[order].real + zero, evecs[:, order]
            mv = lambda x: op @ x
        else:
            raise ValueError(f"Unknown solver mode {mode!r}; use 'auto', 'dense', 'matrix', or 'matrixfree'")

        evals, evecs = orthonormal_eigenstates(mv, evecs)
        residuals = np.array(
            [np.linalg.norm(mv(evecs[:, j]) - evals[j] * evecs[:, j]) for j in range(k_req)],
            dtype=np.float64,
        )
        if residuals.max(initial=0.0) > max(1e-7, 1e3 * tol):
            raise ArithmeticError(
                f"Sector {s} residual {residuals.max():g} exceeded tolerance in mode {chosen}"
            )
        return SectorResult(
            sector=s,
            energies=evals,
            states=evecs,
            residuals=residuals,
            dimension=d,
            mode=chosen,
            basis=basis,
            elapsed_seconds=time.perf_counter() - t0,
        )

    def scan_sectors(
        self,
        sectors: Any = "all",
        nev: int = 5,
        mode: str = "auto",
        tol: float = 1e-10,
        checkpoint_path: str | Path | None = None,
        progress=None,
    ) -> dict[str, Any]:
        """Solve all or selected momentum sectors with optional atomic checkpoint-resume."""
        if sectors == "all":
            sector_list = [tuple(map(int, s)) for s in self.hamiltonian.geometry.indices]
        else:
            sector_list = [
                (
                    int(s[0]) % self.hamiltonian.geometry.shape[0],
                    int(s[1]) % self.hamiltonian.geometry.shape[1],
                )
                for s in sectors
            ]

        results: dict[tuple[int, int], SectorResult] = {}
        ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
        if ckpt is not None and ckpt.is_file():
            with ckpt.open("rb") as f:
                saved = pickle.load(f)
            if saved.get("fingerprint") != self.fingerprint:
                raise ValueError("Checkpoint fingerprint does not match current Hamiltonian")
            results = saved["results"]

        for s in sector_list:
            existing = results.get(s)
            if existing is not None and len(existing.energies) >= min(nev, existing.dimension):
                continue
            res = self.solve_sector(s, nev=nev, mode=mode, tol=tol)
            results[s] = res
            if progress is not None:
                progress(res)
            if ckpt is not None:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
                with tmp.open("wb") as f:
                    pickle.dump({"fingerprint": self.fingerprint, "results": results}, f)
                tmp.replace(ckpt)

        ordered_levels = sorted(
            (float(e), s, lvl)
            for s, res in results.items()
            for lvl, e in enumerate(res.energies)
        )
        return {
            "fingerprint": self.fingerprint,
            "n_particles": self.n_particles,
            "shape": self.hamiltonian.geometry.shape,
            "sectors": results,
            "ordered_levels": ordered_levels,
            "ground_energy": ordered_levels[0][0] if ordered_levels else None,
        }
