"""Vortexable degenerate two-band p4mm QBCP strain model (arXiv:2505.07790)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.linalg import svd

from ..bloch import BlochBand, InternalSpace
from ..interaction import BandProjectedHamiltonian
from ..lattice import BravaisLattice, TorusGeometry


def solve_p4mm_magic_line(
    ratios: tuple[float, ...] = (0.0, 0.2, 0.39, 0.6, 0.7183 / 0.6651),
    cutoff: float = 6.0,
    alpha_range: tuple[float, float] = (0.45, 0.75),
    count: int = 401,
) -> list[dict[str, float]]:
    """Locate magic parameters (alpha, beta = ratio * alpha) where 4 exact flat bands emerge."""
    probe = np.array([[0.23, 0.17], [0.31, 0.11], [0.41, 0.27]], dtype=np.float64)
    template = P4MMStrainModel(alpha=1.0, beta=1.0, cutoff=cutoff)
    indices = template.indices
    q = probe[:, None, :] + (indices @ template.lattice.reciprocal)[None, :, :]
    kinetic = (q[..., 0] + 1j * q[..., 1]) ** 2

    out = []
    for ratio in ratios:
        r = float(ratio)
        m_ref = P4MMStrainModel(alpha=1.0, beta=r, cutoff=cutoff)
        pot = [m_ref.d_matrix(p, indices) - np.diag(kinetic[i]) for i, p in enumerate(probe)]

        def metric(alpha_val: float) -> float:
            vals = []
            for i in range(len(probe)):
                s = svd(np.diag(kinetic[i]) + alpha_val * pot[i], compute_uv=False)
                vals.append(float(s[-1] + s[-2]))
            return float(np.mean(vals))

        alphas = np.linspace(alpha_range[0], alpha_range[1], int(count))
        best_m, best_a = min((metric(float(a)), float(a)) for a in alphas)
        fine = np.linspace(best_a - 0.004, best_a + 0.004, 201)
        f_m, f_a = min((metric(float(a)), float(a)) for a in fine)
        if f_m < best_m:
            best_m, best_a = f_m, f_a
        out.append({"ratio": r, "alpha": best_a, "beta": r * best_a, "zero_mode_metric": best_m})
    return out


@dataclass(frozen=True)
class P4MMStrainModel:
    """Two-band vortexable A-sublattice manifold of p4mm QBCP strain (arXiv:2505.07790 Eqs. 1, 3)."""

    alpha: float = 0.5279
    beta: float = 0.0
    cutoff: float = 10.0
    gate_distance_over_a: float = 2.0
    coulomb_energy: float = 1.0

    def __post_init__(self) -> None:
        if min(self.cutoff, self.gate_distance_over_a, self.coulomb_energy) <= 0.0:
            raise ValueError("Positive cutoff, gate_distance_over_a, and coulomb_energy required")
        a = 2.0 * np.pi
        lat = BravaisLattice.square(a)
        rec = lat.reciprocal  # I_2
        span = int(np.ceil(self.cutoff)) + 4
        raw = np.indices((2 * span + 1, 2 * span + 1)).reshape(2, -1).T - span
        keep = np.linalg.norm(raw @ rec, axis=1) <= self.cutoff + 1e-9
        indices = raw[keep]
        order = np.lexsort((indices[:, 1], indices[:, 0]))
        indices = indices[order]
        lookup = {(int(g[0]), int(g[1])): i for i, g in enumerate(indices)}
        object.__setattr__(self, "lattice", lat)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "lookup", lookup)

    def d_matrix(self, physical_k: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
        idx = self.indices if indices is None else np.asarray(indices, dtype=np.int64)
        q = np.asarray(physical_k, dtype=np.float64)[None, :] + idx @ self.lattice.reciprocal
        n = len(idx)
        mat = np.zeros((n, n), dtype=np.complex128)
        mat[np.diag_indices(n)] = (q[:, 0] + 1j * q[:, 1]) ** 2
        lookup = self.lookup if indices is None else {(int(g[0]), int(g[1])): i for i, g in enumerate(idx)}
        terms = (
            ((1, 1), -1j * self.alpha),
            ((1, -1), +1j * self.alpha),
            ((1, 0), +self.beta),
            ((0, 1), -self.beta),
        )
        for delta, coeff in terms:
            for shift in (delta, (-delta[0], -delta[1])):
                for i, g in enumerate(idx):
                    j = lookup.get((int(g[0]) + shift[0], int(g[1]) + shift[1]))
                    if j is not None:
                        mat[i, j] += coeff
        return mat

    def band(
        self,
        supercell: tuple[int, int] | np.ndarray = (3, 4),
        twists: tuple[float, float] = (0.0, 0.0),
    ) -> BlochBand:
        """Two exact A-sublattice flat bands (total Chern C=+1, flux_per_cell=2)."""
        geom = TorusGeometry(self.lattice, supercell, twists, flux_per_cell=2)
        n_pw = len(self.indices)
        frames = np.zeros((geom.size, n_pw, 2), dtype=np.complex128)
        res_sv = np.empty((geom.size, 2), dtype=np.float64)
        gaps = np.empty(geom.size, dtype=np.float64)
        for idx, k in enumerate(geom.momenta_cart):
            _, s, vh = svd(self.d_matrix(-k), check_finite=False)
            frames[idx] = vh[-2:].conj().T
            res_sv[idx] = s[-2:]
            gaps[idx] = float(s[-3] - s[-2])

        meta = {
            "model": "p4mm_strain",
            "material_parameters": asdict(self),
            "maximum_zero_mode_singular_value": float(res_sv.max()),
            "minimum_remote_gap": float(gaps.min()),
            "expected_total_chern": 1,
            "energy_unit": "U",
        }
        return BlochBand(
            geom,
            frames,
            np.zeros((geom.size, 2), dtype=np.float64),
            self.indices,
            np.zeros(n_pw, dtype=np.int64),
            InternalSpace.bands(2, "flat_band"),
            meta,
        )

    def interaction_weights(self, geom: TorusGeometry, transfers: np.ndarray) -> np.ndarray:
        """Screened Coulomb weights matching arXiv:2505.07790 Eq. (2) / p4mm_reproduction.py."""
        q = np.linalg.norm(geom.transfer_cart(transfers), axis=1)
        a = float(np.linalg.norm(geom.lattice.vectors[0]))
        d = self.gate_distance_over_a * a
        return (4.0 * np.pi * self.coulomb_energy / (np.sqrt(3.0) * geom.size * a)) * np.divide(
            np.tanh(q * d), q, out=np.zeros_like(q), where=q > 0.0
        )

    def hamiltonian(
        self,
        supercell: tuple[int, int] | np.ndarray = (3, 4),
        qell_cutoff: float = 16.0,
        twists: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[BlochBand, BandProjectedHamiltonian]:
        band = self.band(supercell=supercell, twists=twists)
        qs = band.geometry.transfer_disk(qell_cutoff)
        ws = self.interaction_weights(band.geometry, qs)
        ham = BandProjectedHamiltonian.from_band(band, qs, ws)
        return band, ham

    def hole_partner_hamiltonian(
        self,
        supercell: tuple[int, int] | np.ndarray = (3, 4),
        qell_cutoff: float = 16.0,
        twists: tuple[float, float] = (0.0, 0.0),
        diagonalize_one_body: bool = False,
    ) -> tuple[BlochBand, BandProjectedHamiltonian, float, np.ndarray]:
        """Exact antiunitary PH partner for nu=4/3 (N_e=16 <-> N_h=8 on 3x4)."""
        band = self.band(supercell=supercell, twists=twists)
        qs = band.geometry.transfer_disk(qell_cutoff)
        ws = self.interaction_weights(band.geometry, qs)
        partner, sigma, e_full = band.particle_hole_transform(
            qs, ws, diagonalize_one_body=diagonalize_one_body
        )
        ham = BandProjectedHamiltonian.from_band(partner, qs, ws)
        return partner, ham, e_full, sigma
