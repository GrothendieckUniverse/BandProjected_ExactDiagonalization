"""Built-in benchmark models for band-projected exact diagonalization."""

from .landau import TorusLLL, haldane_v1_hamiltonian
from .tmote2 import TMDContinuum
from .ctdbg import ChiralTDBG
from .ictbg import IntervalleyTBG
from .p4mm import P4MMStrainModel, solve_p4mm_magic_line

__all__ = [
    "TorusLLL",
    "haldane_v1_hamiltonian",
    "TMDContinuum",
    "ChiralTDBG",
    "IntervalleyTBG",
    "P4MMStrainModel",
    "solve_p4mm_magic_line",
]
