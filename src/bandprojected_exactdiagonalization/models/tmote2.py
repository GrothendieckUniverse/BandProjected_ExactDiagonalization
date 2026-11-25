"""R-stacked twisted MoTe2 continuum model (C=1 valence band and hole partner)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.constants import elementary_charge, hbar, m_e
from scipy.linalg import eigh

from ..bloch import BlochBand, InternalSpace
from ..interaction import BandProjectedHamiltonian, dual_gate_coulomb
from ..lattice import BravaisLattice, TorusGeometry


@dataclass(frozen=True)
class TMDContinuum:
    """Single-valley R-stacked tMoTe2 continuum Hamiltonian (energies in meV, lengths in Angstrom).

    Default parameters match Wang et al., Phys. Rev. Lett. 132, 036501 (2024) [arXiv:2304.11864]
    at twist angle 3.89 deg, dielectric constant 15.0, and dual-gate distance 300.0 A.
    """

    twist_degrees: float = 3.89
    cutoff: int = 6
    a: float = 3.52
    mass: float = 0.6
    potential: float = 20.75
    phase_degrees: float = -107.7
    tunneling: float = -23.82
    valley: int = 1
    dielectric: float = 15.0
    gate_distance: float = 300.0

    def __post_init__(self) -> None:
        if not 0.0 < self.twist_degrees < 180.0 or int(self.cutoff) != self.cutoff or self.cutoff < 1:
            raise ValueError("Positive twist angle and integral plane-wave cutoff required")
        if self.valley not in (-1, 1) or min(self.a, self.mass, self.dielectric) <= 0.0:
            raise ValueError("Valley ±1 and positive length/mass/dielectric required")
        moire_a = self.a / (2.0 * np.sin(np.deg2rad(self.twist_degrees) / 2.0))
        lat = BravaisLattice(moire_a * np.array([[np.sqrt(3.0) / 2.0, -0.5], [0.0, 1.0]]))
        rec = lat.reciprocal
        kappas = np.array([[2.0 / 3.0, -1.0 / 3.0], [1.0 / 3.0, 1.0 / 3.0]], dtype=np.float64)
        limit = 2 * self.cutoff + 2
        gs = np.indices((2 * limit + 1, 2 * limit + 1)).reshape(2, -1).T - limit
        radius = self.cutoff * np.linalg.norm(rec[0]) * (1.0 + 1e-12)
        records = [
            (g, l)
            for l, center in enumerate(kappas)
            for g in gs
            if np.linalg.norm((g - center) @ rec) <= radius
        ]
        indices = np.asarray([g for g, _ in records], dtype=np.int64)
        layers = np.asarray([l for _, l in records], dtype=np.int64)
        lookup = {(int(g[0]), int(g[1]), int(l)): j for j, (g, l) in enumerate(records)}

        psi = np.deg2rad(self.phase_degrees)
        vp = self.potential * np.diag([np.exp(1j * psi), np.exp(-1j * psi)])
        vm = vp.conj().T
        w = self.tunneling
        fourier = {
            (1, 0): vp.copy(),
            (-1, 1): vp.copy(),
            (0, -1): vp.copy(),
            (-1, 0): vm.copy(),
            (1, -1): vm.copy(),
            (0, 1): vm.copy(),
            (0, 0): np.array([[0.0, w], [np.conj(w), 0.0]], dtype=np.complex128),
        }
        for delta in ((0, -1), (1, -1)):
            fourier[delta] = fourier[delta] + np.array([[0.0, w], [0.0, 0.0]], dtype=np.complex128)
            rev = (-delta[0], -delta[1])
            fourier[rev] = fourier[rev] + np.array([[0.0, 0.0], [np.conj(w), 0.0]], dtype=np.complex128)

        static = np.zeros((len(records), len(records)), dtype=np.complex128)
        for i, (g, layer) in enumerate(records):
            for delta, block in fourier.items():
                for other_layer in (0, 1):
                    j = lookup.get((int(g[0]) - delta[0], int(g[1]) - delta[1], other_layer))
                    if j is not None:
                        static[i, j] += block[layer, other_layer]
        if not np.allclose(static, static.conj().T):
            raise ArithmeticError("tMoTe2 static potential matrix is not Hermitian")

        unit = hbar**2 / m_e / elementary_charge * 1e23
        for attr, val in (
            ("lattice", lat),
            ("kappas", kappas),
            ("indices", indices),
            ("layers", layers),
            ("static", static),
            ("unit", unit),
        ):
            object.__setattr__(self, attr, val)

    def hamiltonian_matrix(self, physical_k: np.ndarray) -> np.ndarray:
        mom = (
            self.valley * np.asarray(physical_k, dtype=np.float64)
            + (self.indices - self.kappas[self.layers]) @ self.lattice.reciprocal
        )
        kinetic = -self.unit * np.sum(mom**2, axis=1) / (2.0 * self.mass)
        mat = self.static + np.diag(kinetic)
        return mat if self.valley == 1 else mat.conj()

    def band(
        self,
        supercell: tuple[int, int] | np.ndarray = (4, 6),
        twists: tuple[float, float] = (0.0, 0.0),
        smooth: bool = True,
    ) -> BlochBand:
        """Construct the top valence electron BlochBand (Chern number C=+1 in valley=+1)."""
        geom = TorusGeometry(self.lattice, supercell, twists)
        frames = np.empty((geom.size, len(self.indices), 1), dtype=np.complex128)
        energies = np.empty((geom.size, 1), dtype=np.float64)
        gaps = np.empty(geom.size, dtype=np.float64)
        n_pw = len(self.indices)
        for idx, k in enumerate(geom.momenta_cart):
            e, u = eigh(self.hamiltonian_matrix(-k), subset_by_index=(n_pw - 2, n_pw - 1), check_finite=False)
            frames[idx, :, 0] = u[:, -1]
            energies[idx, 0] = e[-1]
            gaps[idx] = e[-1] - e[-2]

        meta = {
            "model": "tMoTe2",
            "material_parameters": asdict(self),
            "length_unit": "angstrom",
            "energy_unit": "meV",
            "minimum_isolation_gap_meV": float(gaps.min()),
            "plane_wave_count": int(n_pw),
            "electronic_boundary_twists": list(twists),
            "microscopic_state": "top valence electron band",
        }
        band = BlochBand(
            geom,
            frames,
            energies,
            self.valley * self.indices,
            self.layers,
            InternalSpace.bands(1, "valence"),
            meta,
        )
        return band.smooth_gauge()[0] if (smooth and geom.is_diagonal) else band

    def bare_hole_hamiltonian(
        self,
        supercell: tuple[int, int] | np.ndarray = (4, 6),
        qell_cutoff: float = 7.5,
        twists: tuple[float, float] = (0.0, 0.0),
        smooth: bool = True,
    ) -> tuple[BlochBand, BandProjectedHamiltonian]:
        """Hole-flipped valence Hamiltonian (h = -E_valence) before PH mapping to electrons."""
        valence = self.band(supercell=supercell, twists=twists, smooth=smooth)
        hole_band = BlochBand(
            valence.geometry,
            valence.frames,
            -valence.one_body_blocks,
            valence.reciprocal_indices,
            valence.components,
            InternalSpace.bands(1, "valence_hole"),
            {**valence.metadata, "microscopic_state": "bare valence hole-flipped band"},
        )
        qs = hole_band.geometry.transfer_disk(qell_cutoff)
        ws = dual_gate_coulomb(
            hole_band.geometry.transfer_cart(qs),
            hole_band.geometry.area,
            dielectric=self.dielectric,
            gate_distance=self.gate_distance,
        )
        ham = BandProjectedHamiltonian.from_band(hole_band, qs, ws)
        return hole_band, ham

    def hole_partner_hamiltonian(
        self,
        supercell: tuple[int, int] | np.ndarray = (4, 6),
        qell_cutoff: float = 7.5,
        twists: tuple[float, float] = (0.0, 0.0),
        smooth: bool = True,
    ) -> tuple[BlochBand, BandProjectedHamiltonian, float, np.ndarray]:
        """Exact electron-partner Hamiltonian corresponding to valence hole doping nu_h = -(1 - nu_e).

        Here ``partner_h1 = +E_valence - Sigma_full`` and ``e_full = -sum E_valence + 0.5 sum Sigma_full``,
        matching ``Non-FQH_FCI``'s ``electron_partner_of_holes``.
        Returns ``(partner_band, hamiltonian, hole_vacuum_constant_meV, sigma_full)``.
        """
        hole_band, _ = self.bare_hole_hamiltonian(
            supercell=supercell, qell_cutoff=qell_cutoff, twists=twists, smooth=smooth
        )
        qs = hole_band.geometry.transfer_disk(qell_cutoff)
        ws = dual_gate_coulomb(
            hole_band.geometry.transfer_cart(qs),
            hole_band.geometry.area,
            dielectric=self.dielectric,
            gate_distance=self.gate_distance,
        )
        partner, sigma, e_full = hole_band.particle_hole_transform(qs, ws)
        ham = BandProjectedHamiltonian.from_band(partner, qs, ws)
        return partner, ham, e_full, sigma

    def valence_hamiltonian(
        self,
        supercell: tuple[int, int] | np.ndarray = (4, 6),
        qell_cutoff: float = 7.5,
        twists: tuple[float, float] = (0.0, 0.0),
        smooth: bool = True,
    ) -> tuple[BlochBand, BandProjectedHamiltonian]:
        """Direct electron Hamiltonian in the top valence band (for testing exact PH equivalence)."""
        valence = self.band(supercell=supercell, twists=twists, smooth=smooth)
        ham = BandProjectedHamiltonian.from_potential(
            valence,
            lambda q_cart, area: dual_gate_coulomb(
                q_cart, area, dielectric=self.dielectric, gate_distance=self.gate_distance
            ),
            qell_cutoff=qell_cutoff,
        )
        return valence, ham
