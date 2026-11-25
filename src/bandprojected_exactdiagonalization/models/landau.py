"""Torus Lowest Landau Level (LLL) model and Haldane V1 pseudopotential."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.special import eval_hermite

from ..bloch import BlochBand, InternalSpace
from ..interaction import BandProjectedHamiltonian, haldane_pseudopotential_v1
from ..lattice import BravaisLattice, TorusGeometry, _as_int_pair


class _MagneticWeyl:
    """Canonical integer-flux magnetic Bloch operators matching exp(-i q.r)."""

    def __init__(self, shape: tuple[int, int], chirality: int = 1, twists: tuple[float, float] = (0.0, 0.0)) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.size = self.shape[0] * self.shape[1]
        self.chirality = int(chirality)
        self.twists = (float(twists[0]), float(twists[1]))
        self.indices = np.indices(self.shape, dtype=np.int64).reshape(2, -1).T

    def density_monomial(self, transfer: tuple[int, int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, b = _as_int_pair(transfer, "transfer")
        n1, n2 = self.shape
        k1, k2 = self.indices.T
        wraps = (k2 + b) // n2
        target = ((k1 + a) % n1) * n2 + ((k2 + b) % n2)
        phase = self.chirality * (
            2.0 * np.pi * a * (k2 + 0.5 * b) / self.size
            - 2.0 * np.pi * (k1 + a) * wraps / n1
        ) + (a * self.twists[0] + b * self.twists[1]) / self.size
        return target, np.exp(1j * phase)


def _torus_lll_sections(
    sample_lattice: np.ndarray,
    flux: int,
    points: np.ndarray,
    chirality: int = 1,
    twists: tuple[float, float] = (0.0, 0.0),
    tol: float = 1e-14,
) -> np.ndarray:
    """Normalized LLL Jacobi-theta orbitals on an arbitrary Bravais torus."""
    if chirality == -1:
        pos = _torus_lll_sections(sample_lattice, flux, points, 1, (-twists[0], -twists[1]), tol)
        return pos[:, (-np.arange(flux)) % flux].conj()
    a = np.asarray(sample_lattice, dtype=np.float64)
    w1, w2 = a[0, 0] + 1j * a[0, 1], a[1, 0] + 1j * a[1, 1]
    r = np.asarray(points, dtype=np.float64)
    z = (r[:, 0] + 1j * r[:, 1]) * np.conj(w1) / abs(w1)
    l1 = abs(w1)
    shear = np.real(w2 * np.conj(w1) / abs(w1))
    height = np.imag(w2 * np.conj(w1) / abs(w1))
    ell = np.sqrt(abs(np.linalg.det(a)) / (2.0 * np.pi * flux))
    k0 = (2.0 * np.pi * np.arange(flux) + twists[0]) / l1
    step = 2.0 * np.pi * flux / l1
    span = np.sqrt(-2.0 * np.log(tol)) * ell
    lo = int(np.floor((-z.imag.max() - span - ell**2 * k0.max()) / height)) - 1
    hi = int(np.ceil((-z.imag.min() + span - ell**2 * k0.min()) / height)) + 1
    wave = np.zeros((len(points), flux), dtype=np.complex128)
    for img in range(lo, hi + 1):
        k = k0 + img * step
        fac = np.exp(1j * img * (k0 * shear - twists[1]) + 0.5j * step * shear * img**2)
        coord = (z.imag[:, None] + ell**2 * k) / ell
        wave += fac * eval_hermite(0, coord) * np.exp(1j * z.real[:, None] * k - 0.5 * coord**2)
    wave *= np.exp(0.5j * z.real * z.imag / ell**2)[:, None] / np.sqrt(l1 * np.sqrt(np.pi) * ell)
    j = np.arange(flux)
    alpha = twists[0] / (2.0 * np.pi)
    diag = np.exp(1j * np.pi * shear / (l1 * flux) * (j + alpha) ** 2 - 1j * j * twists[1] / flux)
    return np.fft.ifft(wave * diag[None], axis=1) * np.sqrt(flux)


class LLLBand(BlochBand):
    """Exact torus LLL BlochBand using analytic GMP form factors and real-space magnetic fibers."""

    def __init__(
        self,
        geometry: TorusGeometry,
        frames: np.ndarray,
        rec_idx: np.ndarray,
        weyl: _MagneticWeyl,
        quad_points: np.ndarray,
        metadata: dict,
    ) -> None:
        super().__init__(
            geometry=geometry,
            frames=frames,
            energies=np.zeros((geometry.size, 1), dtype=np.float64),
            reciprocal_indices=rec_idx,
            components=np.zeros(len(rec_idx), dtype=np.int64),
            internal_space=InternalSpace.bands(1, "LLL"),
            metadata=metadata,
        )
        self._weyl = weyl
        self._quad_points = np.asarray(quad_points, dtype=np.float64)

    def seam_matrix(self, axis: int) -> np.ndarray:
        """Exact magnetic Bloch reciprocal-translation seam in real-space quadrature basis."""
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        return np.diag(np.exp(1j * (self._quad_points @ self.geometry.lattice.reciprocal[axis])))

    def form_factor(self, transfer: tuple[int, int] | np.ndarray) -> np.ndarray:
        q = _as_int_pair(transfer, "transfer")
        _, phase = self._weyl.density_monomial(q)
        q_cart = self.geometry.transfer_cart(q)
        q2 = float(np.dot(q_cart, q_cart))
        env = np.exp(-0.25 * q2 * (self.geometry.ell**2))
        return (env * phase)[:, None, None]

    def form_factors(self, transfers: np.ndarray) -> np.ndarray:
        qs = np.asarray(transfers, dtype=np.int64)
        out = np.empty((len(qs), self.geometry.size, 1, 1), dtype=np.complex128)
        ell2 = self.geometry.ell**2
        q_carts = self.geometry.transfer_cart(qs)
        envs = np.exp(-0.25 * np.sum(q_carts**2, axis=1) * ell2)
        for i, q in enumerate(qs):
            _, phase = self._weyl.density_monomial(q)
            out[i, :, 0, 0] = envs[i] * phase
        return out


@dataclass(frozen=True)
class TorusLLL:
    """Lowest Landau Level on a 2D torus with one flux quantum per unit cell."""

    lattice: BravaisLattice | None = None
    chirality: int = 1
    quadrature: tuple[int, int] = (24, 24)

    def __post_init__(self) -> None:
        if self.lattice is None:
            object.__setattr__(self, "lattice", BravaisLattice.triangular(np.sqrt(4.0 * np.pi / np.sqrt(3.0)), 60.0))
        if self.chirality not in (-1, 1):
            raise ValueError("chirality must be ±1")

    def band(self, shape: tuple[int, int], twists: tuple[float, float] = (0.0, 0.0)) -> LLLBand:
        geom = TorusGeometry(self.lattice, shape, twists)
        phi = (-float(twists[0]), -float(twists[1]))
        weyl_twists = (float(self.chirality * twists[1]), float(-self.chirality * twists[0]))
        weyl = _MagneticWeyl(geom.shape, self.chirality, weyl_twists)

        nq1, nq2 = int(self.quadrature[0]), int(self.quadrature[1])
        uv = (np.indices((nq1, nq2)).reshape(2, -1).T + 0.5) / np.array([nq1, nq2], dtype=np.float64)
        pts = uv @ self.lattice.vectors
        canonical = _torus_lll_sections(geom.sample_lattice, geom.size, pts, self.chirality, phi)
        n1, n2 = geom.shape
        transform = np.fft.ifft if self.chirality == 1 else np.fft.fft
        m_wave = transform(canonical.T.reshape(n1, n2, -1), axis=0, norm="ortho").reshape(geom.size, len(pts))
        fibers = m_wave * np.exp(1j * (geom.momenta_cart @ pts.T)) * np.sqrt(geom.area / len(pts))
        fibers /= np.linalg.norm(fibers, axis=1, keepdims=True)
        rec_idx = np.indices((nq1, nq2), dtype=np.int64).reshape(2, -1).T

        meta = {
            "model": "torus_LLL",
            "chirality": int(self.chirality),
            "ell": float(geom.ell),
        }
        return LLLBand(geom, fibers[:, :, None], rec_idx, weyl, pts, meta)


def haldane_v1_hamiltonian(
    shape: tuple[int, int] = (3, 4),
    v1: float = 1.0,
    qell_cutoff: float = 7.5,
    twists: tuple[float, float] = (0.0, 0.0),
    lattice: BravaisLattice | None = None,
    include_q0: bool = True,
) -> tuple[LLLBand, BandProjectedHamiltonian]:
    """Construct exact LLL Haldane V_1 pseudopotential Hamiltonian on a torus."""
    model = TorusLLL(lattice=lattice, chirality=1)
    band = model.band(shape, twists=twists)
    ell = band.geometry.ell
    ham = BandProjectedHamiltonian.from_potential(
        band,
        lambda q_cart, area: haldane_pseudopotential_v1(q_cart, ell=ell, area=area, v1=v1, include_q0=include_q0),
        qell_cutoff=qell_cutoff,
        include_q0=include_q0,
    )
    return band, ham
