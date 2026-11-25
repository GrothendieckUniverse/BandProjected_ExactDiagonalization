"""Chiral twisted double bilayer graphene (cTDBG) ideal C=2 flatband model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.linalg import svd

from ..bloch import BlochBand, InternalSpace
from ..interaction import BandProjectedHamiltonian, bare_coulomb
from ..lattice import BravaisLattice, TorusGeometry


@dataclass(frozen=True)
class ChiralTDBG:
    """Sublattice-polarized ideal C=±2 flat band of chiral AB/AB TDBG (arXiv:2510.15027)."""

    beta: float = 0.1
    cutoff: float = 5.0
    alpha: float = 0.58566355838955
    w1: float = 110.0
    t0: float = 2610.0
    lattice_constant: float = 2.46
    sublattice: str = "A"
    dielectric: float = 4.0

    def __post_init__(self) -> None:
        if min(self.beta, self.cutoff, self.alpha, self.w1, self.t0, self.dielectric) <= 0.0:
            raise ValueError("Positive cTDBG parameters required")
        if self.sublattice not in ("A", "B"):
            raise ValueError("sublattice must be 'A' or 'B'")

        theta = 2.0 * np.arcsin(np.sqrt(3.0) * self.w1 / (4.0 * np.pi * self.alpha * self.t0))
        k_theta = 8.0 * np.pi / (3.0 * self.lattice_constant) * np.sin(theta / 2.0)
        rec = k_theta * np.array([[np.sqrt(3.0) / 2.0, 1.5], [-np.sqrt(3.0) / 2.0, 1.5]], dtype=np.float64)
        lat = BravaisLattice(2.0 * np.pi * np.linalg.inv(rec).T)
        kappas = np.array([[1.0 / 3.0, 1.0 / 3.0]] * 4 + [[2.0 / 3.0, 2.0 / 3.0]] * 4, dtype=np.float64)

        span = int(2 * self.cutoff + 3)
        gs = np.indices((2 * span + 1, 2 * span + 1)).reshape(2, -1).T - span
        radius = self.cutoff * np.linalg.norm(rec[0]) * (1.0 + 1e-12)
        records = [
            (g, c)
            for c in range(8)
            for g in gs
            if np.linalg.norm((g - kappas[c]) @ rec) <= radius
        ]
        indices = np.array([g for g, _ in records], dtype=np.int64)
        components = np.array([c for _, c in records], dtype=np.int64)
        lookup = {(int(g[0]), int(g[1]), int(c)): i for i, (g, c) in enumerate(records)}
        a_rows = np.flatnonzero(components % 2 == 0)
        b_rows = np.flatnonzero(components % 2 == 1)
        velocity = np.sqrt(3.0) * self.lattice_constant * self.t0 / 2.0
        t1 = self.beta * self.w1 / self.alpha

        static = np.zeros((len(records), len(records)), dtype=np.complex128)
        for j, delta in enumerate(((0, 0), (-1, 0), (0, -1))):
            phase = np.exp(-2j * np.pi * j / 3.0)
            tunnel = self.w1 * np.array([[0.0, phase], [phase.conjugate(), 0.0]], dtype=np.complex128)
            for g, c in records:
                if c not in (4, 5):
                    continue
                b_i = lookup[(int(g[0]), int(g[1]), int(c))]
                for ac in (2, 3):
                    a_i = lookup.get((int(g[0]) + delta[0], int(g[1]) + delta[1], ac))
                    if a_i is not None:
                        static[a_i, b_i] += tunnel[ac - 2, c - 4]
                        static[b_i, a_i] += tunnel[ac - 2, c - 4].conjugate()
        for ca, cb in ((0, 3), (4, 7)):
            for g, c in records:
                if c == ca:
                    ia, ib = lookup[(int(g[0]), int(g[1]), ca)], lookup[(int(g[0]), int(g[1]), cb)]
                    static[ia, ib] = static[ib, ia] = t1

        for attr, val in (
            ("theta", theta),
            ("lattice", lat),
            ("kappas", kappas),
            ("indices", indices),
            ("components", components),
            ("lookup", lookup),
            ("a_rows", a_rows),
            ("b_rows", b_rows),
            ("velocity", velocity),
            ("t1", t1),
            ("static", static),
        ):
            object.__setattr__(self, attr, val)

    def hamiltonian_matrix(self, physical_k: np.ndarray) -> np.ndarray:
        h = self.static.copy()
        pk = np.asarray(physical_k, dtype=np.float64)
        rec = self.lattice.reciprocal
        for block in (0, 2, 4, 6):
            angle = -self.theta / 2.0 if block < 4 else self.theta / 2.0
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float64)
            for a_i in np.flatnonzero(self.components == block):
                g = self.indices[a_i]
                b_i = self.lookup[(int(g[0]), int(g[1]), block + 1)]
                q = (pk + (g - self.kappas[block]) @ rec) @ rot.T
                val = self.velocity * (q[0] - 1j * q[1])
                h[a_i, b_i] = val
                h[b_i, a_i] = val.conjugate()
        return h

    def band(
        self,
        supercell: tuple[int, int] | np.ndarray = (4, 6),
        twists: tuple[float, float] = (0.0, 0.0),
        smooth: bool = True,
    ) -> BlochBand:
        """Exact chiral zero-mode BlochBand on rectangular or sheared/tilted supercells."""
        geom = TorusGeometry(self.lattice, supercell, twists)
        n_pw = len(self.indices)
        frames = np.empty((geom.size, n_pw, 1), dtype=np.complex128)
        residuals = np.empty(geom.size, dtype=np.float64)
        gaps = np.empty(geom.size, dtype=np.float64)
        for idx, k in enumerate(geom.momenta_cart):
            h = self.hamiltonian_matrix(-k)
            u, s, vh = svd(h[np.ix_(self.a_rows, self.b_rows)], check_finite=False)
            z = np.zeros(n_pw, dtype=np.complex128)
            if self.sublattice == "A":
                z[self.a_rows] = u[:, -1]
            else:
                z[self.b_rows] = vh[-1].conjugate()
            residuals[idx] = float(np.linalg.norm(h @ z))
            gaps[idx] = float(s[-2])
            frames[idx, :, 0] = z
        if residuals.max() > 0.05:
            raise ArithmeticError(f"Chiral zero mode residual {residuals.max():g} exceeded 0.05 meV")

        meta = {
            "model": "ideal_TDBG_C2",
            "material_parameters": asdict(self),
            "energy_unit": "meV",
            "length_unit": "angstrom",
            "maximum_chiral_zero_mode_residual_meV": float(residuals.max()),
            "minimum_remote_band_gap_meV": float(gaps.min()),
            "expected_band_chern": 2 if self.sublattice == "A" else -2,
        }
        band = BlochBand(
            geom,
            frames,
            np.zeros((geom.size, 1), dtype=np.float64),
            self.indices,
            self.components,
            InternalSpace.bands(1, self.sublattice),
            meta,
        )
        return band.smooth_gauge()[0] if (smooth and geom.is_diagonal) else band

    def hamiltonian(
        self,
        supercell: tuple[int, int] | np.ndarray = (4, 6),
        qell_cutoff: float = 6.0,
        twists: tuple[float, float] = (0.0, 0.0),
        smooth: bool = True,
    ) -> tuple[BlochBand, BandProjectedHamiltonian]:
        band = self.band(supercell=supercell, twists=twists, smooth=smooth)
        ham = BandProjectedHamiltonian.from_potential(
            band,
            lambda q_cart, area: bare_coulomb(q_cart, area, dielectric=self.dielectric),
            qell_cutoff=qell_cutoff,
        )
        return band, ham
