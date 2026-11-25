"""Intervalley-coupled twisted bilayer graphene (icTBG) zero-Chern FCI model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.linalg import eigvalsh, eigh

from ..bloch import BlochBand, InternalSpace
from ..interaction import BandProjectedHamiltonian, bare_coulomb
from ..lattice import BravaisLattice, TorusGeometry


@dataclass(frozen=True)
class IntervalleyTBG:
    """Six-component continuum model of icTBG with isolated C=0 lowest conduction band."""

    twist_degrees: float = 1.13
    cutoff: float = 4.0
    lattice_constant: float = 2.46
    t0: float = 2610.0
    w0: float = 30.0
    w1: float = 110.0
    sublattice_mass: float = 10.0
    intervalley: float = 20.0
    dielectric: float = 4.0

    def __post_init__(self) -> None:
        if min(self.twist_degrees, self.cutoff, self.lattice_constant, self.t0, self.dielectric) <= 0.0:
            raise ValueError("Positive icTBG parameters required")
        theta = np.deg2rad(self.twist_degrees)
        k_theta = 8.0 * np.pi / (3.0 * self.lattice_constant) * np.sin(theta / 2.0)
        rec = k_theta * np.array([[np.sqrt(3.0) / 2.0, 1.5], [-np.sqrt(3.0) / 2.0, 1.5]], dtype=np.float64)
        lat = BravaisLattice(2.0 * np.pi * np.linalg.inv(rec).T)
        # Top K (0,1) around (1/3,1/3); bottom K (2,3) and K' (4,5) around (2/3,2/3)
        kappas = np.array([[1.0 / 3.0, 1.0 / 3.0]] * 2 + [[2.0 / 3.0, 2.0 / 3.0]] * 4, dtype=np.float64)
        span = int(2 * self.cutoff + 3)
        gs = np.indices((2 * span + 1, 2 * span + 1)).reshape(2, -1).T - span
        radius = self.cutoff * np.linalg.norm(rec[0]) * (1.0 + 1e-12)
        records = [
            (g, c)
            for c in range(6)
            for g in gs
            if np.linalg.norm((g - kappas[c]) @ rec) <= radius
        ]
        indices = np.array([g for g, _ in records], dtype=np.int64)
        components = np.array([c for _, c in records], dtype=np.int64)
        lookup = {(int(g[0]), int(g[1]), int(c)): i for i, (g, c) in enumerate(records)}
        velocity = np.sqrt(3.0) * self.lattice_constant * self.t0 / 2.0

        static = np.zeros((len(records), len(records)), dtype=np.complex128)
        for j, delta in enumerate(((0, 0), (-1, 0), (0, -1))):
            tj = np.array(
                [
                    [self.w0, self.w1 * np.exp(-2j * np.pi * j / 3.0)],
                    [self.w1 * np.exp(2j * np.pi * j / 3.0), self.w0],
                ],
                dtype=np.complex128,
            )
            for g, c in records:
                if c not in (2, 3):
                    continue
                b_i = lookup[(int(g[0]), int(g[1]), int(c))]
                for ac in (0, 1):
                    a_i = lookup.get((int(g[0]) + delta[0], int(g[1]) + delta[1], ac))
                    if a_i is not None:
                        static[a_i, b_i] += tj[ac, c - 2]
                        static[b_i, a_i] += tj[ac, c - 2].conjugate()

        for g, c in records:
            i = lookup[(int(g[0]), int(g[1]), int(c))]
            static[i, i] += self.sublattice_mass if c % 2 == 0 else -self.sublattice_mass
            if c == 2:
                j = lookup[(int(g[0]), int(g[1]), 5)]
                static[i, j] = static[j, i] = self.intervalley

        for attr, val in (
            ("theta", theta),
            ("lattice", lat),
            ("kappas", kappas),
            ("indices", indices),
            ("components", components),
            ("lookup", lookup),
            ("velocity", velocity),
            ("static", static),
        ):
            object.__setattr__(self, attr, val)

    def hamiltonian_matrix(self, physical_k: np.ndarray) -> np.ndarray:
        h = self.static.copy()
        pk = np.asarray(physical_k, dtype=np.float64)
        rec = self.lattice.reciprocal
        for block, rot_sign, kp_sign in ((0, -1.0, 1.0), (2, 1.0, 1.0), (4, 1.0, -1.0)):
            ang = rot_sign * self.theta / 2.0
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=np.float64)
            for a_i in np.flatnonzero(self.components == block):
                g = self.indices[a_i]
                b_i = self.lookup[(int(g[0]), int(g[1]), block + 1)]
                q = (pk + (g - self.kappas[block]) @ rec) @ rot.T
                val = self.velocity * (q[0] - 1j * kp_sign * q[1])
                h[a_i, b_i] = val
                h[b_i, a_i] = val.conjugate()
        return h

    def band(
        self,
        supercell: tuple[int, int] | np.ndarray = (4, 6),
        twists: tuple[float, float] = (0.0, 0.0),
        smooth: bool = True,
    ) -> BlochBand:
        """Lowest conduction band (target index N_pw // 2, Chern number C=0)."""
        geom = TorusGeometry(self.lattice, supercell, twists)
        n_pw = len(self.indices)
        target_idx = n_pw // 2
        frames = np.empty((geom.size, n_pw, 1), dtype=np.complex128)
        energies = np.empty((geom.size, 1), dtype=np.float64)
        gaps = np.empty(geom.size, dtype=np.float64)
        for idx, k in enumerate(geom.momenta_cart):
            e, u = eigh(
                self.hamiltonian_matrix(-k),
                subset_by_index=(target_idx - 1, target_idx + 1),
                check_finite=False,
            )
            frames[idx, :, 0] = u[:, 1]
            energies[idx, 0] = e[1]
            gaps[idx] = min(e[1] - e[0], e[2] - e[1])

        meta = {
            "model": "intervalley_TBG",
            "material_parameters": asdict(self),
            "target_eigenvalue_index": int(target_idx),
            "minimum_isolation_gap_meV": float(gaps.min()),
            "expected_band_chern": 0,
            "energy_unit": "meV",
            "length_unit": "angstrom",
        }
        band = BlochBand(
            geom,
            frames,
            energies,
            self.indices,
            self.components,
            InternalSpace.bands(1, "conduction"),
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
