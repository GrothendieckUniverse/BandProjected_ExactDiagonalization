"""Clean pure-NumPy band-projected momentum-space exact diagonalization."""

from .lattice import BravaisLattice, TorusGeometry, reciprocal_basis, smith_normal_form_2x2
from .bloch import BlochBand, InternalSpace

__all__ = [
    "BravaisLattice",
    "TorusGeometry",
    "reciprocal_basis",
    "smith_normal_form_2x2",
    "BlochBand",
    "InternalSpace",
]


def main() -> None:
    print("BandProjected_ExactDiagonalization: pure-NumPy band-projected ED engine.")
