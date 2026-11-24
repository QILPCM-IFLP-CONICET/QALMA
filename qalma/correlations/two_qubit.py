r"""Entanglement measures of a pair of qubits (spin-1/2 sites).

For two qubits the concurrence has the closed form of Wootters, and the
entanglement of formation is a monotonic function of it. Neither has a
closed form for larger local dimensions; for spin-1 or larger sites use
:func:`~qalma.correlations.bipartite.negativity` or
:func:`~qalma.correlations.bipartite.log_negativity` instead.

The extra keyword arguments (``max_exact_sites``, ``allow_approximate``,
``max_dimension``) are passed to
:func:`~qalma.correlations.reduction.reduced_state`.
"""

from ._dense import concurrence_dense, entanglement_of_formation_from_concurrence
from .reduction import ReducedState, Subsystem, reduced_state

__all__ = ["concurrence", "entanglement_of_formation"]


def _two_qubit_state(state, site_a: Subsystem, site_b: Subsystem, **kwargs):
    """Return the reduced state of the pair, checking that it is two qubits."""
    rho: ReducedState = reduced_state(state, site_a, site_b, **kwargs)
    if rho.dims != (2, 2):
        raise ValueError(
            f"The sites {rho.labels} have local dimensions {rho.dims}; the "
            "concurrence and the entanglement of formation are only available for "
            "a pair of two-level sites. Use the negativity instead."
        )
    return rho


def concurrence(state, site_a: Subsystem, site_b: Subsystem, **kwargs) -> float:
    r"""Wootters concurrence of a pair of two-level sites.

    :math:`C=\max(0,\lambda_1-\lambda_2-\lambda_3-\lambda_4)`, with
    :math:`\lambda_k` the decreasingly ordered square roots of the
    eigenvalues of :math:`\sqrt\rho\,\tilde\rho\sqrt\rho` and
    :math:`\tilde\rho=(\sigma_y\otimes\sigma_y)\rho^*(\sigma_y\otimes\sigma_y)`,
    where :math:`\rho` is the reduced state of the pair.

    Parameters
    ----------
    state : DensityOperatorProtocol or qutip.Qobj
        The global state.
    site_a, site_b : str or int
        The two sites (or factor positions, for a Qobj). Each must be a single
        two-level site.
    **kwargs
        Passed to :func:`~qalma.correlations.reduction.reduced_state`.

    Returns
    -------
    float
        The concurrence, in :math:`[0, 1]`.

    Raises
    ------
    ValueError
        If the sites are not two two-level sites.
    """
    rho = _two_qubit_state(state, site_a, site_b, **kwargs)
    return concurrence_dense(rho.matrix)


def entanglement_of_formation(
    state, site_a: Subsystem, site_b: Subsystem, **kwargs
) -> float:
    r"""Entanglement of formation of a pair of two-level sites, in ebits.

    :math:`E_F=h\left(\frac{1+\sqrt{1-C^2}}{2}\right)`, with :math:`C` the
    :func:`concurrence` and :math:`h` the binary entropy. See
    :func:`concurrence` for the parameters.

    Returns
    -------
    float
        The entanglement of formation, in :math:`[0, 1]`.
    """
    return entanglement_of_formation_from_concurrence(
        concurrence(state, site_a, site_b, **kwargs)
    )
