r"""RPA reduction of Gibbs states.

Given :math:`\rho = e^{-K}/Z` and a subsystem :math:`A`, the reduced
generator is approximated by :math:`K_A' = K_{A,\mathrm{MF}} +
K_{A,\mathrm{RPA}}`, where :math:`K_{A,\mathrm{MF}}` is the mean-field
projection of the boundary terms and :math:`K_{A,\mathrm{RPA}}` accounts for
the Gaussian fluctuations of the environment around its mean-field state.

The reductions are computed by :class:`RPAGibbsDensityOperator`:

* static correction of the generator (default), with the static RPA kernel
  :math:`M_{AE}\chi_E M_{EA}` (:func:`rpa_coupling_kernel`);
* dynamical correction (``dynamic=True``), with the imaginary-time kernel of
  the bosonized environment (:func:`rpa_dynamic_kernel`);
* hybrid reduction (``hybrid=True``), with the subsystem exact and the
  environment replaced by a few effective bosons (:func:`rpa_hybrid_state`).

The bosonization of the whole system (:func:`build_rpa_boson_system`) gives
the entropies and negativities of the bosonic RPA state
(:func:`rpa_entanglement_entropy`, :func:`rpa_negativity`), which describe
the quantum fluctuations (the static modes are left out) and are not defined
with zero modes. With a broken continuous symmetry, the entanglement is
computed from the hybrid reduced states with the ``symmetry_projections`` of
the state. The formulation, the treatment of zero modes and the limitations of
the method are described in the user guide (:ref:`rpa-formulation`).
"""

from .bosonized import rpa_bosonized_state
from .dynamic import rpa_dynamic_kernel
from .free_energy import RPAFreeEnergy, broken_symmetries, rpa_free_energy
from .functions import rpa_entanglement_entropy, rpa_negativity
from .hybrid import effective_bath_modes, rpa_hybrid_state
from .rpa import RPAGibbsDensityOperator, rpa_generator_correction
from .rpa_structure import (
    build_rpa_boson_system,
    build_rpa_structure,
    rpa_coupling_kernel,
    rpa_quadratic_form,
)
from .utils import kubo_mori_susceptibility, local_boson_modes

__all__ = [
    "RPAFreeEnergy",
    "RPAGibbsDensityOperator",
    "broken_symmetries",
    "build_rpa_boson_system",
    "rpa_bosonized_state",
    "build_rpa_structure",
    "effective_bath_modes",
    "kubo_mori_susceptibility",
    "local_boson_modes",
    "rpa_coupling_kernel",
    "rpa_entanglement_entropy",
    "rpa_negativity",
    "rpa_quadratic_form",
    "rpa_dynamic_kernel",
    "rpa_free_energy",
    "rpa_generator_correction",
    "rpa_hybrid_state",
]
