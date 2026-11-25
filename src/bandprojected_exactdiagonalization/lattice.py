"""Bravais lattices, reciprocal geometry, and Smith-Normal-Form 2D tori."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import numpy as np


def _as_int_pair(value, name: str = "pair") -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (2,) or not np.all(np.isfinite(arr)) or not np.all(arr == np.rint(arr)):
        raise ValueError(f"{name} must be a 2-component integer vector, got {value!r}")
    return arr.astype(np.int64)


def reciprocal_basis(lattice: np.ndarray) -> np.ndarray:
    """Return reciprocal row vectors B obeying A @ B.T = 2*pi * I_2."""
    a = np.asarray(lattice, dtype=np.float64)
    if a.shape != (2, 2):
        raise ValueError(f"Expected 2x2 row lattice, got {a.shape}")
    det = float(np.linalg.det(a))
    if not np.isfinite(det) or det <= 0.0:
        raise ValueError(f"Positively oriented 2D lattice required, got det={det}")
    return 2.0 * np.pi * np.linalg.inv(a).T


def smith_normal_form_2x2(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact integer Smith Normal Form U @ M @ V = diag(d1, d2) with d1 | d2.

    For diagonal matrices with positive entries, keeps the axis order when
    diagonal entries are already equal or when M is diagonal and caller sets
    preserve_diagonal, wait: general SNF preserves exact subgroup quotient.
    """
    m = np.asarray(matrix, dtype=np.int64).copy()
    if m.shape != (2, 2) or not np.array_equal(m, np.asarray(matrix)):
        raise ValueError("Expected 2x2 integer supercell matrix")
    det = int(round(float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])))
    if det <= 0:
        raise ValueError(f"Supercell matrix must have positive determinant, got {det}")

    u = np.eye(2, dtype=np.int64)
    v = np.eye(2, dtype=np.int64)

    # Fast path for standard diagonal supercells (N1, N2): keep exact axis order
    # so sector labels stay (k1, k2) in [0, N1) x [0, N2) with trivial U = V = I.
    if m[0, 1] == 0 and m[1, 0] == 0 and m[0, 0] > 0 and m[1, 1] > 0:
        return u, m.copy(), v

    def swap_rows(a: np.ndarray, uu: np.ndarray, i: int, j: int) -> None:
        a[[i, j]] = a[[j, i]]
        uu[[i, j]] = uu[[j, i]]

    def swap_cols(a: np.ndarray, vv: np.ndarray, i: int, j: int) -> None:
        a[:, [i, j]] = a[:, [j, i]]
        vv[:, [i, j]] = vv[:, [j, i]]

    def add_row(a: np.ndarray, uu: np.ndarray, dst: int, src: int, q: int) -> None:
        a[dst] += q * a[src]
        uu[dst] += q * uu[src]

    def add_col(a: np.ndarray, vv: np.ndarray, dst: int, src: int, q: int) -> None:
        a[:, dst] += q * a[:, src]
        vv[:, dst] += q * vv[:, src]

    def negate_row(a: np.ndarray, uu: np.ndarray, i: int) -> None:
        a[i] = -a[i]
        uu[i] = -uu[i]

    def negate_col(a: np.ndarray, vv: np.ndarray, j: int) -> None:
        a[:, j] = -a[:, j]
        vv[:, j] = -vv[:, j]

    # Step 1: reduce (0,0) to gcd of all entries
    while True:
        # Move smallest nonzero absolute entry to (0,0)
        nz = [(abs(int(m[i, j])), i, j) for i in range(2) for j in range(2) if m[i, j] != 0]
        _, bi, bj = min(nz)
        if bi != 0:
            swap_rows(m, u, 0, bi)
        if bj != 0:
            swap_cols(m, v, 0, bj)
        if m[0, 0] < 0:
            negate_row(m, u, 0)
        if m[1, 0] != 0:
            q = int(m[1, 0] // m[0, 0])
            add_row(m, u, 1, 0, -q)
            continue
        if m[0, 1] != 0:
            q = int(m[0, 1] // m[0, 0])
            add_col(m, v, 1, 0, -q)
            continue
        if m[1, 1] % m[0, 0] != 0:
            add_row(m, u, 0, 1, 1)
            continue
        break

    if m[1, 1] < 0:
        negate_row(m, u, 1)

    assert np.array_equal(u @ np.asarray(matrix, dtype=np.int64) @ v, m)
    assert abs(int(round(np.linalg.det(u)))) == 1
    assert abs(int(round(np.linalg.det(v)))) == 1
    return u, m, v


@dataclass(frozen=True)
class BravaisLattice:
    """Two-dimensional Bravais lattice stored as row vectors [a1; a2]."""

    vectors: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.vectors, dtype=np.float64)
        reciprocal_basis(arr)
        arr = arr.copy()
        arr.flags.writeable = False
        object.__setattr__(self, "vectors", arr)

    @classmethod
    def square(cls, a: float = 1.0) -> "BravaisLattice":
        return cls(np.array([[a, 0.0], [0.0, a]], dtype=np.float64))

    @classmethod
    def triangular(cls, a: float = 1.0, angle_deg: float = 60.0) -> "BravaisLattice":
        rad = np.deg2rad(angle_deg)
        return cls(np.array([[a, 0.0], [a * np.cos(rad), a * np.sin(rad)]], dtype=np.float64))

    @cached_property
    def reciprocal(self) -> np.ndarray:
        b = reciprocal_basis(self.vectors)
        b.flags.writeable = False
        return b

    @property
    def unit_cell_area(self) -> float:
        return float(np.linalg.det(self.vectors))

    @property
    def ell(self) -> float:
        """Reference magnetic length sqrt(A_uc / (2*pi)) for 1 flux quantum per cell."""
        return float(np.sqrt(self.unit_cell_area / (2.0 * np.pi)))


@dataclass(frozen=True)
class TorusGeometry:
    """Finite 2D torus defined by an arbitrary integer supercell matrix N.

    Supercell lattice vectors are rows of ``N @ lattice.vectors``.
    Momentum sectors are canonically indexed via the Smith Normal Form
    ``U @ N @ V = diag(d1, d2)``, preserving exact ``(k1, k2)`` order for
    diagonal supercells ``(N1, N2)`` while supporting arbitrary sheared tori.
    """

    lattice: BravaisLattice
    supercell: tuple[int, int] | np.ndarray
    twists: tuple[float, float] = (0.0, 0.0)
    flux_per_cell: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.lattice, BravaisLattice):
            object.__setattr__(self, "lattice", BravaisLattice(self.lattice))
        sc = np.asarray(self.supercell, dtype=np.int64)
        if sc.shape == (2,):
            if np.any(sc <= 0):
                raise ValueError(f"Positive supercell dimensions required, got {sc}")
            sc_mat = np.diag(sc)
        elif sc.shape == (2, 2):
            if not np.array_equal(sc, np.asarray(self.supercell)):
                raise ValueError("Integer 2x2 supercell matrix required")
            if int(round(np.linalg.det(sc))) <= 0:
                raise ValueError("Supercell matrix must have positive determinant")
            sc_mat = sc.copy()
        else:
            raise ValueError("supercell must be a 2-tuple (N1, N2) or 2x2 integer matrix")
        tw = np.asarray(self.twists, dtype=np.float64)
        if tw.shape != (2,) or not np.all(np.isfinite(tw)):
            raise ValueError("Two finite boundary twists required")
        if int(self.flux_per_cell) != self.flux_per_cell or self.flux_per_cell < 1:
            raise ValueError("Positive integral flux_per_cell required")
        sc_mat.flags.writeable = False
        object.__setattr__(self, "supercell_matrix", sc_mat)
        object.__setattr__(self, "twists", (float(tw[0]), float(tw[1])))
        object.__setattr__(self, "flux_per_cell", int(self.flux_per_cell))

        u, d, v = smith_normal_form_2x2(sc_mat)
        for arr in (u, d, v):
            arr.flags.writeable = False
        object.__setattr__(self, "_snf_u", u)
        object.__setattr__(self, "_snf_d", d)
        object.__setattr__(self, "_snf_v", v)
        object.__setattr__(self, "_snf_u_inv", np.rint(np.linalg.inv(u)).astype(np.int64))
        object.__setattr__(self, "_snf_v_inv", np.rint(np.linalg.inv(v)).astype(np.int64))
        object.__setattr__(self, "shape", (int(d[0, 0]), int(d[1, 1])))

    @property
    def is_diagonal(self) -> bool:
        return bool(self.supercell_matrix[0, 1] == 0 and self.supercell_matrix[1, 0] == 0)

    @property
    def size(self) -> int:
        return int(self.shape[0] * self.shape[1])

    @cached_property
    def sample_lattice(self) -> np.ndarray:
        lat = self.supercell_matrix @ self.lattice.vectors
        lat.flags.writeable = False
        return lat

    @cached_property
    def sample_reciprocal(self) -> np.ndarray:
        b = reciprocal_basis(self.sample_lattice)
        b.flags.writeable = False
        return b

    @property
    def area(self) -> float:
        return float(self.size * self.lattice.unit_cell_area)

    @property
    def ell(self) -> float:
        return float(np.sqrt(self.area / (2.0 * np.pi * self.size * self.flux_per_cell)))

    @cached_property
    def indices(self) -> np.ndarray:
        """Canonical SNF integer momentum labels of shape (N_uc, 2) in [0, d1) x [0, d2)."""
        idx = np.indices(self.shape, dtype=np.int64).reshape(2, -1).T
        idx.flags.writeable = False
        return idx

    @cached_property
    def _raw_fractional(self) -> np.ndarray:
        # m = idx @ U^{-T}, so k / B = (m + twists/(2*pi)) @ N^{-T}
        # = idx @ D^{-1} @ V^T + (twists / (2*pi)) @ N^{-T}
        n_inv_t = np.linalg.inv(self.supercell_matrix.astype(np.float64)).T
        d_inv = np.diag([1.0 / self.shape[0], 1.0 / self.shape[1]])
        base = self.indices.astype(np.float64) @ d_inv @ self._snf_v.T.astype(np.float64)
        twist_shift = (np.asarray(self.twists, dtype=np.float64) / (2.0 * np.pi)) @ n_inv_t
        return base + twist_shift[None, :]

    @cached_property
    def fractional_momenta(self) -> np.ndarray:
        """Folded fractional momenta in [0, 1)^2 in the primitive reciprocal basis."""
        # Exact shift relative to primitive BZ [0, 1)^2; snap tiny negative roundoff
        frac = np.mod(self._raw_fractional, 1.0)
        frac[np.isclose(frac, 1.0, atol=1e-13)] = 0.0
        frac.flags.writeable = False
        return frac

    @cached_property
    def momenta_cart(self) -> np.ndarray:
        """Cartesian crystal momenta (N_uc, 2) inside the primitive BZ [0, 1)^2."""
        k = self.fractional_momenta @ self.lattice.reciprocal
        k.flags.writeable = False
        return k

    def to_snf_coords(self, q_sample: np.ndarray) -> np.ndarray:
        """Convert integer momentum transfer in sample-reciprocal basis to SNF coords."""
        q = _as_int_pair(q_sample, "transfer")
        return q @ self._snf_u.T

    def sector_index(self, sector: tuple[int, int] | np.ndarray) -> int:
        s = _as_int_pair(sector, "sector") % self.shape
        return int(s[0] * self.shape[1] + s[1])

    def shift(self, q_sample: tuple[int, int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Shift all grid momenta by integer sample-reciprocal transfer ``q_sample``.

        Returns ``(target_indices, umklapp_windings)`` such that
        ``k_raw(s) + q_cart = k_cart(target[s]) + umklapp_windings[s] @ B_prim``
        with exact integer Umklapp reciprocal vectors ``umklapp_windings[s]``.
        """
        q = _as_int_pair(q_sample, "transfer")
        q_snf = q @ self._snf_u.T
        target_pairs = (self.indices + q_snf[None, :]) % self.shape
        target = target_pairs[:, 0] * self.shape[1] + target_pairs[:, 1]
        n_inv_t = np.linalg.inv(self.supercell_matrix.astype(np.float64)).T
        q_frac = q.astype(np.float64) @ n_inv_t
        diff = self.fractional_momenta + q_frac[None, :] - self.fractional_momenta[target]
        windings = np.rint(diff).astype(np.int64)
        if np.max(np.abs(diff - windings)) > 1e-9:
            raise ArithmeticError("Non-integer Umklapp winding encountered in TorusGeometry.shift")
        return target, windings

    def transfer_cart(self, q_sample: tuple[int, int] | np.ndarray) -> np.ndarray:
        q = np.asarray(q_sample, dtype=np.float64)
        return q @ self.sample_reciprocal

    def transfer_disk(self, qell_max: float) -> np.ndarray:
        """All nonzero sample-reciprocal transfers ``q`` within ``|q| * ell <= qell_max``."""
        if not np.isfinite(qell_max) or qell_max <= 0.0:
            raise ValueError("Positive q*ell cutoff required")
        b_sc = self.sample_reciprocal
        min_sv = float(np.linalg.svd(b_sc, compute_uv=False)[-1])
        limit = int(np.ceil(qell_max / (self.ell * min_sv))) + 1
        q = np.indices((2 * limit + 1, 2 * limit + 1), dtype=np.int64).reshape(2, -1).T - limit
        lengths = np.linalg.norm(q @ b_sc, axis=1) * self.ell
        keep = (lengths > 0.0) & (lengths <= qell_max * (1.0 + 1e-12))
        q, lengths = q[keep], lengths[keep]
        order = np.lexsort((q[:, 1], q[:, 0], lengths))
        return q[order]
