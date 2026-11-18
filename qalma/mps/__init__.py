"""Matrix product states and a minimal DMRG, written with numpy only.

The main entry point is :class:`~qalma.mps.gs_operator.MPSGSOperator`, a
density operator that stores the ground state of a Hamiltonian as an MPS.
"""

from qalma.mps.dmrg import dmrg
from qalma.mps.gs_operator import MPSGSOperator
from qalma.mps.mpo import build_mpo, mpo_to_dense
from qalma.mps.mps import MPS

__all__ = ["MPS", "MPSGSOperator", "build_mpo", "dmrg", "mpo_to_dense"]
