r"""Reduced density matrices of two subsystems.

The estimators of :mod:`qalma.correlations` work on the dense reduced state
:math:`\rho_{AB}` of the two subsystems involved. This module builds it from
any QALMA state (or from a :class:`qutip.Qobj`), with the factors of
:math:`A` first and those of :math:`B` after, in a fixed order, so that the
estimators know which factors to transpose or trace.

Subsystems are given as a site name or an iterable of site names. A physical
site that the system represents as several qubits (see
:attr:`~qalma.model.SystemDescriptor.site_qubits`) is expanded into its
qubits. For a :class:`qutip.Qobj`, subsystems are given by the positions of
its tensor factors instead.
"""

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from qutip import Qobj, ket2dm

from qalma.model import FermionicSignWarning, SystemDescriptor

__all__ = [
    "ApproximateReductionWarning",
    "DEFAULT_MAX_DENSE_DIMENSION",
    "ReducedState",
    "check_subsystems",
    "complement",
    "is_product_state",
    "reduced_state",
    "resolve_subsystem",
]

#: Largest dimension of :math:`\rho_{AB}` that is built as a dense matrix.
#: Each estimator accepts a ``max_dimension`` keyword to change it per call.
DEFAULT_MAX_DENSE_DIMENSION = 4096

Subsystem = Union[str, int, Iterable]


class ApproximateReductionWarning(UserWarning):
    """Warn that a reduced state (and its correlations) is approximate.

    Raised when a :class:`~qalma.operators.states.gibbs.GibbsDensityOperator` is too
    large to be reduced exactly and ``allow_approximate=True`` was passed: the
    reduction then relies on mean field (or on the RPA for
    :class:`~qalma.meanfield.rpa.RPAGibbsDensityOperator`), which in general
    underestimates the correlations between the subsystems.
    """


@dataclass(frozen=True)
class ReducedState:
    """Dense reduced state of two subsystems, as built by :func:`reduced_state`."""

    #: Normalized density matrix of :math:`A\cup B`.
    matrix: np.ndarray
    #: Local dimensions of the factors, those of :math:`A` first.
    dims: Tuple[int, ...]
    #: Site names (or factor positions) of the factors, in the same order.
    labels: tuple
    #: Number of factors of :math:`A`.
    n_a: int

    @property
    def factors_a(self) -> Tuple[int, ...]:
        """Positions of the factors of :math:`A` in :attr:`matrix`."""
        return tuple(range(self.n_a))

    @property
    def factors_b(self) -> Tuple[int, ...]:
        """Positions of the factors of :math:`B` in :attr:`matrix`."""
        return tuple(range(self.n_a, len(self.dims)))


def _as_tuple(subsystem: Subsystem) -> tuple:
    """Turn a single label or an iterable of labels into an ordered tuple."""
    if isinstance(subsystem, (str, int, np.integer)):
        return (subsystem,)
    if isinstance(subsystem, (set, frozenset)):
        return tuple(sorted(subsystem))
    return tuple(subsystem)


def resolve_subsystem(system: SystemDescriptor, subsystem: Subsystem) -> tuple:
    """Return the names of the sites of ``system`` that form ``subsystem``.

    Parameters
    ----------
    system : SystemDescriptor
        The system.
    subsystem : str or iterable of str
        A site name, or an iterable of site names. Physical sites that the
        system represents as several qubits are expanded into their qubits.
        Sets are sorted; other iterables keep their order.

    Returns
    -------
    tuple of str
        Site names of ``system.sites``.

    Raises
    ------
    KeyError
        If a name is neither a site nor a physical site of the system.
    ValueError
        If the subsystem is empty or has repeated sites.
    """
    result = []
    site_qubits = getattr(system, "site_qubits", {}) or {}
    for label in _as_tuple(subsystem):
        if label in system.sites:
            result.append(label)
        elif label in site_qubits:
            result.extend(site_qubits[label])
        else:
            raise KeyError(f"{label!r} is not a site of the system.")
    if not result:
        raise ValueError("Empty subsystem.")
    if len(set(result)) != len(result):
        raise ValueError(f"Repeated sites in the subsystem {result}.")
    return tuple(result)


def _warn_fermionic(system: SystemDescriptor, sites: tuple) -> None:
    """Emit a FermionicSignWarning if any of ``sites`` is fermionic."""
    physical = getattr(system, "physical_sites", None) or system.sites
    owner = {}
    for node, qubits in (getattr(system, "site_qubits", {}) or {}).items():
        for qubit in qubits:
            owner[qubit] = node
    fermionic = sorted(
        {
            owner.get(site, site)
            for site in sites
            if any(
                qn.get("fermionic", False)
                for qn in physical.get(owner.get(site, site), {}).get("qn", {}).values()
            )
        }
    )
    if fermionic:
        warnings.warn(
            f"The sites {fermionic} carry fermionic quantum numbers. The partial "
            "transpose used here is the one of the qubit (hard-core boson) "
            "representation, not the fermionic partial transpose, so negativities "
            "computed between fermionic modes are not fermionic entanglement "
            "measures.",
            FermionicSignWarning,
            stacklevel=3,
        )


def is_product_state(state) -> bool:
    """Return ``True`` if ``state`` is a product of single-site states.

    The test is structural (based on the type of the state), so a state
    that happens to be a product but is not represented as such returns
    ``False``.
    """
    # pylint: disable=import-outside-toplevel
    from qalma.operators.states import (
        GibbsProductDensityOperator,
        ProductDensityOperator,
    )

    return isinstance(state, (ProductDensityOperator, GibbsProductDensityOperator))


def _check_dimension(dims: Tuple[int, ...], max_dimension: Optional[int]) -> None:
    """Refuse to build dense matrices larger than ``max_dimension``."""
    if max_dimension is None:
        max_dimension = DEFAULT_MAX_DENSE_DIMENSION
    total = int(np.prod(dims))
    if total > max_dimension:
        raise ValueError(
            f"The reduced state of the subsystems has dimension {total}, above "
            f"max_dimension={max_dimension}. Use smaller subsystems or raise "
            "max_dimension."
        )


def _reduced_from_qobj(
    state: Qobj, labels_a: tuple, labels_b: tuple, max_dimension: Optional[int]
) -> ReducedState:
    """Build the reduced state from a Qobj, with subsystems as factor positions."""
    if state.isket:
        state = ket2dm(state)
    if not state.isoper:
        raise TypeError("The Qobj must be a ket or a density operator.")
    all_dims = tuple(int(d) for d in state.dims[0])
    keep = labels_a + labels_b
    for k in keep:
        if not isinstance(k, (int, np.integer)) or not 0 <= k < len(all_dims):
            raise KeyError(
                f"{k!r} is not a factor position of a Qobj with dims {all_dims}."
            )
    if len(set(keep)) != len(keep):
        raise ValueError("The two subsystems overlap.")
    dims = tuple(all_dims[k] for k in keep)
    _check_dimension(dims, max_dimension)
    # pylint: disable=import-outside-toplevel
    from ._dense import partial_trace_dense

    matrix = partial_trace_dense(state.full(), all_dims, keep)
    return ReducedState(matrix, dims, tuple(int(k) for k in keep), len(labels_a))


def _reduce_qalma_state(
    state, keep: frozenset, max_exact_sites: Optional[int], allow_approximate: bool
):
    """Partial trace of a QALMA state, refusing silent approximations."""
    # pylint: disable=import-outside-toplevel
    from qalma.operators.states import GibbsDensityOperator

    if not isinstance(state, GibbsDensityOperator):
        return state.partial_trace(keep)

    from qalma.meanfield.gibbs_partial_trace import gibbs_exact_partial_trace

    result = gibbs_exact_partial_trace(state, keep, max_exact_sites)
    if result is not None:
        return result
    if not allow_approximate:
        raise ValueError(
            "The Gibbs state is too large to be reduced exactly on "
            f"{sorted(keep)}. Raise max_exact_sites to reduce it exactly, or pass "
            "allow_approximate=True to use the mean-field (or RPA) reduction, "
            "which in general underestimates the correlations."
        )
    warnings.warn(
        f"The reduction of the Gibbs state on {sorted(keep)} is approximate; "
        "the correlations computed from it are approximate too.",
        ApproximateReductionWarning,
        stacklevel=3,
    )
    return state.partial_trace(keep)


def reduced_state(
    state,
    subsystem_a: Subsystem,
    subsystem_b: Optional[Subsystem] = None,
    *,
    max_exact_sites: Optional[int] = None,
    allow_approximate: bool = False,
    max_dimension: Optional[int] = None,
) -> ReducedState:
    r"""Return the dense reduced state :math:`\rho_{AB}`.

    Parameters
    ----------
    state : DensityOperatorProtocol or qutip.Qobj
        The global state. A :class:`qutip.Qobj` can be a ket or a density
        operator; its subsystems are given by factor positions.
    subsystem_a : str, int or iterable
        Sites (or factor positions) of :math:`A`.
    subsystem_b : str, int or iterable, optional
        Sites (or factor positions) of :math:`B`. If ``None``, :math:`B` is
        empty and only :math:`\rho_A` is built.
    max_exact_sites : int, optional
        Passed to the exact reduction of
        :class:`~qalma.operators.states.gibbs.GibbsDensityOperator` states (see
        :func:`~qalma.meanfield.gibbs_partial_trace.gibbs_exact_partial_trace`).
    allow_approximate : bool
        If ``False`` (default), a Gibbs state that cannot be reduced exactly
        raises :class:`ValueError`. If ``True``, its approximate reduction is
        used and an :class:`ApproximateReductionWarning` is emitted.
    max_dimension : int, optional
        Largest dimension of :math:`\rho_{AB}` allowed. Defaults to
        :data:`DEFAULT_MAX_DENSE_DIMENSION`.

    Returns
    -------
    ReducedState
        The normalized reduced state, with the factors of :math:`A` first.

    Raises
    ------
    ValueError
        If the subsystems overlap, the reduced state is too large, or a Gibbs
        state cannot be reduced exactly and ``allow_approximate`` is
        ``False``.
    """
    labels_a = _as_tuple(subsystem_a)
    labels_b = () if subsystem_b is None else _as_tuple(subsystem_b)

    if isinstance(state, Qobj):
        return _reduced_from_qobj(state, labels_a, labels_b, max_dimension)

    system = state.system
    sites_a = resolve_subsystem(system, labels_a)
    sites_b = () if subsystem_b is None else resolve_subsystem(system, labels_b)
    if set(sites_a) & set(sites_b):
        raise ValueError("The two subsystems overlap.")
    block = sites_a + sites_b
    dims = tuple(int(system.sites[site]["dimension"]) for site in block)
    _check_dimension(dims, max_dimension)
    _warn_fermionic(system, block)

    reduced = _reduce_qalma_state(
        state, frozenset(block), max_exact_sites, allow_approximate
    )
    if isinstance(reduced, Qobj):
        qobj = reduced
    else:
        qobj = reduced.to_qutip(block)
    if not isinstance(qobj, Qobj) or tuple(qobj.dims[0]) != dims:
        raise RuntimeError(
            f"The reduced state of {type(state).__name__} on {block} has an "
            "unexpected tensor structure."
        )
    # pylint: disable=import-outside-toplevel
    from ._dense import normalize_density_matrix

    matrix = normalize_density_matrix(qobj.full())
    return ReducedState(matrix, dims, block, len(sites_a))


def check_subsystems(
    state, subsystem_a: Subsystem, subsystem_b: Optional[Subsystem]
) -> None:
    """Check the subsystems without building any matrix.

    Used by the estimators on product states, which return zero without
    reducing the state but must still reject invalid subsystems.

    Raises
    ------
    KeyError
        If a label is not a site (or factor position) of the state.
    ValueError
        If a subsystem is empty or the two subsystems overlap.
    """
    labels_a = _as_tuple(subsystem_a)
    labels_b = () if subsystem_b is None else _as_tuple(subsystem_b)
    if isinstance(state, Qobj):
        n_factors = len(state.dims[0])
        sites_a, sites_b = labels_a, labels_b
        for k in sites_a + sites_b:
            if not isinstance(k, (int, np.integer)) or not 0 <= k < n_factors:
                raise KeyError(f"{k!r} is not a factor position of the Qobj.")
    else:
        sites_a = resolve_subsystem(state.system, labels_a)
        sites_b = (
            () if subsystem_b is None else resolve_subsystem(state.system, labels_b)
        )
    if set(sites_a) & set(sites_b):
        raise ValueError("The two subsystems overlap.")


def complement(state, subsystem_a: Subsystem) -> tuple:
    """Return the sites (or factor positions) of ``state`` outside ``subsystem_a``."""
    labels_a = _as_tuple(subsystem_a)
    if isinstance(state, Qobj):
        return tuple(k for k in range(len(state.dims[0])) if k not in labels_a)
    sites_a = set(resolve_subsystem(state.system, labels_a))
    return tuple(site for site in sorted(state.system.sites) if site not in sites_a)
