"""Correlation estimators evaluated on many pairs of sites."""

from collections.abc import Callable, Iterable
from itertools import combinations
from typing import Dict, Optional, Tuple, Union, cast

from .bipartite import log_negativity, mutual_information, negativity
from .two_qubit import concurrence, entanglement_of_formation

__all__ = ["PAIR_MEASURES", "pairwise", "system_pairs"]

#: Estimators accepted by name in :func:`pairwise`.
PAIR_MEASURES: Dict[str, Callable] = {
    "concurrence": concurrence,
    "entanglement_of_formation": entanglement_of_formation,
    "log_negativity": log_negativity,
    "mutual_information": mutual_information,
    "negativity": negativity,
}


def system_pairs(system, which: str = "edges") -> Tuple[Tuple[str, str], ...]:
    """Return pairs of physical sites of ``system``.

    Parameters
    ----------
    system : SystemDescriptor
        The system.
    which : {"edges", "all"}
        ``"edges"`` (default) returns the edges of the lattice graph, of every
        bond type, without repetitions. ``"all"`` returns every unordered pair
        of physical sites.

    Returns
    -------
    tuple of tuple
        The pairs, in a deterministic order.
    """
    if which == "all":
        nodes = getattr(system, "physical_sites", None) or system.sites
        return tuple(combinations(sorted(nodes), 2))
    if which != "edges":
        raise ValueError(f"which must be 'edges' or 'all', got {which!r}.")
    seen = set()
    result = []
    for edges in system.spec["graph"].edges.values():
        for site_a, site_b in edges:
            key = frozenset((site_a, site_b))
            if site_a == site_b or key in seen:
                continue
            seen.add(key)
            result.append((site_a, site_b))
    return tuple(result)


def pairwise(
    state,
    measure: Union[str, Callable] = "concurrence",
    pairs: Optional[Union[str, Iterable]] = None,
    **kwargs,
) -> Dict[Tuple, float]:
    """Evaluate a two-site correlation estimator on several pairs of sites.

    Parameters
    ----------
    state : DensityOperatorProtocol
        The global state.
    measure : str or callable
        A key of :data:`PAIR_MEASURES`, or a callable with the signature
        ``measure(state, site_a, site_b, **kwargs) -> float``.
    pairs : iterable of tuple, "edges" or "all", optional
        The pairs. ``None`` or ``"edges"`` use the edges of the lattice graph;
        ``"all"`` uses every pair of physical sites (see
        :func:`system_pairs`).
    **kwargs
        Passed to the estimator (and from it to
        :func:`~qalma.correlations.reduction.reduced_state`).

    Returns
    -------
    dict
        ``{(site_a, site_b): value}``.

    Notes
    -----
    A :class:`~qalma.operators.states.gibbs.GibbsDensityOperator` small enough to
    be reduced exactly is converted once to its dense form, instead of once
    per pair.
    """
    if isinstance(measure, str):
        try:
            measure = PAIR_MEASURES[measure]
        except KeyError as exc:
            raise ValueError(
                f"Unknown measure {measure!r}; use one of {sorted(PAIR_MEASURES)}."
            ) from exc
    pair_list: Iterable[Tuple]
    if pairs is None:
        pair_list = system_pairs(state.system)
    elif isinstance(pairs, str):
        pair_list = system_pairs(state.system, pairs)
    else:
        pair_list = cast(Iterable[Tuple], pairs)

    state = _densify_small_gibbs(state, kwargs.get("max_exact_sites"))
    return {
        (site_a, site_b): measure(state, site_a, site_b, **kwargs)
        for site_a, site_b in pair_list
    }


def _densify_small_gibbs(state, max_exact_sites: Optional[int]):
    """Convert a small Gibbs state to its dense form, so it is built only once."""
    # pylint: disable=import-outside-toplevel
    from qalma.meanfield.gibbs_partial_trace import resolve_max_exact_sites
    from qalma.operators.states import GibbsDensityOperator

    if isinstance(state, GibbsDensityOperator) and len(
        state.system.sites
    ) <= resolve_max_exact_sites(max_exact_sites):
        return state.to_qutip_operator()
    return state
