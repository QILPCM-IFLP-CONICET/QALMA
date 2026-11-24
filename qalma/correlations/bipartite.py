r"""Correlation estimators for a bipartition :math:`A|B`.

All the functions take the global state and the two subsystems, build the
dense reduced state :math:`\rho_{AB}` (see
:func:`~qalma.correlations.reduction.reduced_state`) and evaluate the
estimator on it. The extra keyword arguments (``max_exact_sites``,
``allow_approximate``, ``max_dimension``) are passed to
:func:`~qalma.correlations.reduction.reduced_state`.

Product states (:class:`~qalma.operators.states.product.ProductDensityOperator` and
:class:`~qalma.operators.states.gibbs.GibbsProductDensityOperator`) are recognized
and give zero without building any matrix.
"""

from typing import Optional

from ._dense import (
    log_negativity_dense,
    mutual_information_dense,
    negativity_dense,
    von_neumann_entropy_dense,
)
from .reduction import (
    Subsystem,
    check_subsystems,
    complement,
    is_product_state,
    reduced_state,
)

__all__ = [
    "entanglement_entropy",
    "log_negativity",
    "mutual_information",
    "negativity",
]


def negativity(
    state,
    subsystem_a: Subsystem,
    subsystem_b: Optional[Subsystem] = None,
    **kwargs,
) -> float:
    r"""Negativity :math:`\mathcal{N}=(\|\rho_{AB}^{T_B}\|_1-1)/2`.

    The convention is the one of
    :meth:`~qalma.gaussian.BosonicGaussianSystem.negativity`, so the result
    can be compared with :func:`~qalma.meanfield.rpa.rpa_negativity`.

    Parameters
    ----------
    state : DensityOperatorProtocol or qutip.Qobj
        The global state.
    subsystem_a : str, int or iterable
        Sites (or factor positions, for a Qobj) of :math:`A`.
    subsystem_b : str, int or iterable, optional
        Sites of :math:`B`. If ``None``, the complement of :math:`A`.
    **kwargs
        Passed to :func:`~qalma.correlations.reduction.reduced_state`.

    Returns
    -------
    float
        The negativity. It vanishes for separable states; a positive value
        certifies entanglement between :math:`A` and :math:`B`.
    """
    if is_product_state(state):
        check_subsystems(state, subsystem_a, subsystem_b)
        return 0.0
    if subsystem_b is None:
        subsystem_b = complement(state, subsystem_a)
    rho = reduced_state(state, subsystem_a, subsystem_b, **kwargs)
    return negativity_dense(rho.matrix, rho.dims, rho.factors_b)


def log_negativity(
    state,
    subsystem_a: Subsystem,
    subsystem_b: Optional[Subsystem] = None,
    **kwargs,
) -> float:
    r"""Logarithmic negativity :math:`E_N=\log_2\|\rho_{AB}^{T_B}\|_1`.

    It is an upper bound to the distillable entanglement, in ebits. See
    :func:`negativity` for the parameters.

    Returns
    -------
    float
        The logarithmic negativity.
    """
    if is_product_state(state):
        check_subsystems(state, subsystem_a, subsystem_b)
        return 0.0
    if subsystem_b is None:
        subsystem_b = complement(state, subsystem_a)
    rho = reduced_state(state, subsystem_a, subsystem_b, **kwargs)
    return log_negativity_dense(rho.matrix, rho.dims, rho.factors_b)


def entanglement_entropy(
    state, subsystem_a: Subsystem, base: float = 2, **kwargs
) -> float:
    r"""Von Neumann entropy :math:`S(\rho_A)` of a subsystem.

    It measures the entanglement between :math:`A` and its complement only
    when the global state is pure; for mixed states it also counts classical
    correlations and the mixedness of the global state.

    Parameters
    ----------
    state : DensityOperatorProtocol or qutip.Qobj
        The global state.
    subsystem_a : str, int or iterable
        Sites (or factor positions, for a Qobj) of :math:`A`.
    base : float
        Base of the logarithm. ``2`` (default) gives bits.
    **kwargs
        Passed to :func:`~qalma.correlations.reduction.reduced_state`.

    Returns
    -------
    float
        The entropy.
    """
    rho = reduced_state(state, subsystem_a, None, **kwargs)
    return von_neumann_entropy_dense(rho.matrix, base)


def mutual_information(
    state,
    subsystem_a: Subsystem,
    subsystem_b: Subsystem,
    base: float = 2,
    **kwargs,
) -> float:
    r"""Quantum mutual information :math:`I(A{:}B)=S_A+S_B-S_{AB}`.

    It quantifies the total (quantum and classical) correlations between
    :math:`A` and :math:`B`, and bounds every connected correlation
    function: :math:`\langle O_A O_B\rangle_c^2\le 2\|O_A\|^2\|O_B\|^2
    I(A{:}B)\ln 2`.

    Parameters
    ----------
    state : DensityOperatorProtocol or qutip.Qobj
        The global state.
    subsystem_a, subsystem_b : str, int or iterable
        Sites (or factor positions, for a Qobj) of the two subsystems.
    base : float
        Base of the logarithm. ``2`` (default) gives bits.
    **kwargs
        Passed to :func:`~qalma.correlations.reduction.reduced_state`.

    Returns
    -------
    float
        The mutual information.
    """
    if is_product_state(state):
        check_subsystems(state, subsystem_a, subsystem_b)
        return 0.0
    rho = reduced_state(state, subsystem_a, subsystem_b, **kwargs)
    return mutual_information_dense(
        rho.matrix, rho.dims, rho.factors_a, rho.factors_b, base
    )
