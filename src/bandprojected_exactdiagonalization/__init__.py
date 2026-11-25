"""Clean pure-NumPy band-projected momentum-space exact diagonalization."""

from .lattice import BravaisLattice, TorusGeometry, reciprocal_basis, smith_normal_form_2x2
from .bloch import BlochBand, InternalSpace
from .interaction import (
    BandProjectedHamiltonian,
    PairBlock,
    bare_coulomb,
    dual_gate_coulomb,
    haldane_pseudopotential_v1,
)
from .basis import MomentumSectorCatalog, SectorBasis, SectorLookup, popcount_u64
from .solver import BandProjectedED, SectorResult, hamiltonian_fingerprint
from .observables import (
    apply_one_body_sector,
    flux_spectral_flow,
    many_body_chern_number_and_pump,
    occupation_matrix,
    particle_entanglement_spectrum,
    plot_ed_spectrum,
    static_structure_factor_sector,
)

__all__ = [
    "BravaisLattice",
    "TorusGeometry",
    "reciprocal_basis",
    "smith_normal_form_2x2",
    "BlochBand",
    "InternalSpace",
    "BandProjectedHamiltonian",
    "PairBlock",
    "bare_coulomb",
    "dual_gate_coulomb",
    "haldane_pseudopotential_v1",
    "MomentumSectorCatalog",
    "SectorBasis",
    "SectorLookup",
    "popcount_u64",
    "BandProjectedED",
    "SectorResult",
    "hamiltonian_fingerprint",
    "apply_one_body_sector",
    "occupation_matrix",
    "static_structure_factor_sector",
    "particle_entanglement_spectrum",
    "many_body_chern_number_and_pump",
    "flux_spectral_flow",
    "plot_ed_spectrum",
]


def main() -> None:
    print("BandProjected_ExactDiagonalization: pure-NumPy band-projected ED engine.")
