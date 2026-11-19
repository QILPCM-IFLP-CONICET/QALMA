"""Gibbs partial trace.

This module implement functions to approximate the partial trace of a
Gibbs state.
"""

import logging
from typing import List, Optional

from qutip import tensor as qutip_tensor

from qalma import settings
from qalma.meanfield import (
    variational_quadratic_mfa,
)
from qalma.operators import (
    Operator,
    ProductOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
)
from qalma.operators.arithmetic import iterable_to_operator
from qalma.operators.states.basic import (
    DensityOperatorProtocol,
)
from qalma.operators.states.gibbs import GibbsDensityOperator
from qalma.operators.states.product import (
    ProductDensityOperator,
)

logger = logging.getLogger(__name__)

__all__ = [
    "gibbs_exact_partial_trace",
    "gibbs_meanfield_generator",
    "gibbs_meanfield_partial_trace",
    "project_boundary_term",
    "resolve_max_exact_sites",
]


def project_boundary_term(term, sigma: ProductDensityOperator, sites: frozenset):
    """
    Project the boundary term.

    Convert terms of the form O_a Q_b in to O_a <Q_b> with <Q_b> the
    expectation value regarding sigma, and Q_b acting on the sub-system
    associated to sigma.
    """
    acts_over = term.acts_over()
    sites = frozenset({site for site in acts_over if site in sites})
    environment = frozenset(site for site in acts_over if site not in sites)
    system = term.system
    if len(sites) == 0:
        return ScalarOperator(sigma.expect(term), system)
    if all(site in sites for site in acts_over):
        return term

    local_states = sigma.site_factors_qutip
    local_states = {site: local_states[site] for site in environment}

    if isinstance(term, SumOperator):
        return iterable_to_operator(
            (project_boundary_term(sub_term, sigma, sites) for sub_term in term.terms),
            system,
            isherm=True,
        )
    if isinstance(term, ProductOperator):
        prefactor = term.prefactor
        site_factors_qutip = term.site_factors_qutip
        for site in environment:
            prefactor = prefactor * (site_factors_qutip[site] * local_states[site]).tr()
        site_factors_qutip = {
            site: op for site, op in site_factors_qutip.items() if site in sites
        }
        return ProductOperator(site_factors_qutip, prefactor, system)
    if isinstance(term, QutipOperator):
        block = tuple(sites) + tuple(environment)
        # ``to_qutip`` already includes the prefactor of the term.
        qutip_op = term.to_qutip(block)
        qutip_op = qutip_op * qutip_tensor(
            [
                local_states.get(site, None) or system.site_identity(site)
                for site in block
            ]
        )
        qutip_op = qutip_op.ptrace(list(range(len(sites))))
        names = {site: pos for pos, site in enumerate(sites)}
        return QutipOperator(qutip_op, system, names)
    # QuadraticFormOperator
    if hasattr(term, "as_sum_of_products"):
        term = term.as_sum_of_products()
        return project_boundary_term(term, sigma, sites)
    logger.warning(
        "boundary term is not Product or Qutip (%s). Converting to QutipOperator",
        type(term),
    )
    return project_boundary_term(term.to_qutip_operator(), sigma, sites)


def resolve_max_exact_sites(max_exact_sites: Optional[int] = None) -> int:
    """Return the number of sites up to which Gibbs states are reduced exactly.

    Parameters
    ----------
    max_exact_sites : int, optional
        Explicit value. If ``None``, the current value of
        ``qalma.settings.MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE`` is used.

    Returns
    -------
    int
        The threshold. ``0`` disables the exact reductions.

    Raises
    ------
    ValueError
        If ``max_exact_sites`` is negative.
    """
    if max_exact_sites is None:
        return int(settings.MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE)
    if max_exact_sites < 0:
        raise ValueError(f"max_exact_sites must be >= 0, got {max_exact_sites}.")
    return int(max_exact_sites)


def gibbs_exact_partial_trace(
    state: GibbsDensityOperator,
    sites: frozenset,
    max_exact_sites: Optional[int] = None,
) -> Optional[DensityOperatorProtocol]:
    """Reduce a Gibbs state exactly, if it is small or the reduction is trivial.

    The reduction is exact in three cases:

    * the whole system has at most ``max_exact_sites`` sites: the dense state
      is built and traced;
    * the generator acts on at most ``max_exact_sites`` sites: the dense
      state of that block is built and traced, and the result is returned as
      a :class:`~qalma.operators.states.gibbs.GibbsDensityOperator`;
    * the generator acts only inside ``sites`` or only outside them
      (regardless of ``max_exact_sites``).

    Parameters
    ----------
    state : GibbsDensityOperator
        The Gibbs state to reduce.
    sites : frozenset
        Sites of the subsystem to keep.
    max_exact_sites : int, optional
        Maximum number of sites of the exact reductions. If ``None``, use
        ``qalma.settings.MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE``. ``0`` leaves only
        the trivial cases.

    Returns
    -------
    DensityOperatorProtocol or None
        The reduced state, or ``None`` if none of the cases applies and the
        reduction must be approximated.
    """
    max_exact_sites = resolve_max_exact_sites(max_exact_sites)
    if len(state.system.sites) <= max_exact_sites:
        return state.to_qutip_operator().partial_trace(sites)

    prefactor = state.prefactor
    generator = state.k
    full_acts_over = generator.acts_over()
    environment = frozenset(site for site in full_acts_over if site not in sites)
    system = state.k.system
    subsystem = system.subsystem(sites)
    # pylint: disable=protected-access
    sigma_mf = state._meanfield

    if len(environment) == 0:
        return GibbsDensityOperator(
            generator, system=subsystem, prefactor=prefactor, meanfield=sigma_mf
        )
    if len(full_acts_over) == len(environment):
        return GibbsDensityOperator(
            ScalarOperator(0.0, system), system=subsystem, prefactor=prefactor
        )
    # The generator acts on a small block: build the state of the block,
    # compute the partial trace, and get back the generator.
    if len(full_acts_over) <= max_exact_sites:
        sites_in_superblock = frozenset(
            site for site in full_acts_over if site in sites
        )
        sigma_superblock = (
            GibbsDensityOperator(
                generator,
                system=system.subsystem(full_acts_over),
                prefactor=1,
                meanfield=sigma_mf,
            )
            .to_qutip_operator()
            .partial_trace(sites_in_superblock)
        )
        k_reduced = -sigma_superblock.logm()
        # k_reduced is associated to the state of the subsystem.
        # We need to reset it to the global system:
        k_reduced._set_system_(system)
        # The subsystem can still be large, so a GibbsDensityOperator is returned.
        return GibbsDensityOperator(k_reduced, system=subsystem, prefactor=prefactor)
    return None


def gibbs_meanfield_generator(state: GibbsDensityOperator, sites: frozenset):
    r"""Build the mean-field generator of the reduced state of a Gibbs state.

    The generator :math:`K` of the state is split into the terms inside the
    subsystem :math:`A`, the terms in the environment and the boundary terms.
    The boundary terms are projected over the algebra of :math:`A` using the
    self-consistent mean-field state, giving :math:`K_{A,MF}`. The RPA
    corrections to it are added by
    :class:`~qalma.meanfield.rpa.RPAGibbsDensityOperator`.

    Parameters
    ----------
    state : GibbsDensityOperator
        The Gibbs state to reduce. If it has no mean-field state yet, it is
        computed and stored in the state.
    sites : frozenset
        Sites of the subsystem ``A``.

    Returns
    -------
    k_reduced : Operator
        The mean-field generator :math:`K_{A,MF}` of the reduced state.
    has_boundary : bool
        Whether there were terms coupling ``A`` with the environment.
    """
    generator = state.k.flat()
    system = state.k.system
    sigma_mf = state._meanfield  # pylint: disable=protected-access
    environment = frozenset(site for site in generator.acts_over() if site not in sites)

    # Decompose generator in terms inside (subsystem), terms outside
    # (environment) and boundary (interaction)
    all_terms = generator.terms if isinstance(generator, SumOperator) else [generator]
    terms_in: List[Operator] = []
    terms_boundary: List[Operator] = []
    for term in all_terms:
        term_acts_over = term.acts_over()
        if term_acts_over.issubset(sites):
            terms_in.append(term)
        elif term_acts_over.issubset(environment):
            continue
        else:
            terms_boundary.append(term)

    if terms_boundary:
        # If there are boundary terms, project them
        if sigma_mf is None:
            sigma_mf = variational_quadratic_mfa(generator)
            # pylint: disable=protected-access
            state._meanfield = sigma_mf

        # Project the terms onto the algebra of the local subsystem
        terms_boundary_gen = (
            project_boundary_term(term, sigma_mf, sites) for term in terms_boundary
        )
        # Remove empty terms
        terms_boundary = [term for term in terms_boundary_gen if term]
        terms_in.extend(terms_boundary)

    k_in = iterable_to_operator(terms_in, system, isherm=True)
    return k_in, bool(terms_boundary)


def gibbs_meanfield_partial_trace(
    state: GibbsDensityOperator,
    sites: frozenset,
    max_exact_sites: Optional[int] = None,
) -> DensityOperatorProtocol:
    """Reduce a Gibbs state, exactly if it is small and in mean field otherwise.

    Parameters
    ----------
    state : GibbsDensityOperator
        The Gibbs state to reduce.
    sites : frozenset
        Sites of the subsystem to keep.
    max_exact_sites : int, optional
        Reductions involving at most this number of sites are computed
        exactly (see :func:`gibbs_exact_partial_trace`). If ``None``, use
        ``qalma.settings.MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE``; ``0`` always
        uses the mean-field approximation.

    Returns
    -------
    DensityOperatorProtocol
        The reduced density operator.
    """
    result = gibbs_exact_partial_trace(state, sites, max_exact_sites)
    if result is not None:
        return result
    k_in, has_boundary = gibbs_meanfield_generator(state, sites)
    subsystem = state.k.system.subsystem(sites)
    result = GibbsDensityOperator(
        k_in, subsystem, prefactor=state.prefactor
    ).to_qutip_operator()
    # If there were non-trivial terms in the boundary, restore symmetries.
    if has_boundary:
        for symm in state.symmetry_projections:
            result = symm(result)
    return result
