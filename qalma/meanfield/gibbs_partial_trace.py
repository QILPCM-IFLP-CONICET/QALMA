"""Gibbs partial trace.

This module implement functions to approximate the partial trace of a
Gibbs state.
"""

import logging
from typing import List, Optional

from qutip import tensor as qutip_tensor

from qalma.meanfield import (
    variational_quadratic_mfa,
)
from qalma.model import SystemDescriptor
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
from qalma.settings import MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE

__all__ = ["gibbs_meanfield_generator", "gibbs_meanfield_partial_trace"]


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
    logging.warning(
        "boundary term is not Product or Qutip (%s). Converting to QutipOperator",
        type(term),
    )
    return project_boundary_term(term.to_qutip_operator(), sigma, sites)


def gibbs_meanfield_generator(
    state: GibbsDensityOperator,
    sites: frozenset,
    rpa: bool = False,
    dynamic: bool = False,
    n_tau: int = 32,
):
    r"""Build the approximate generator of the reduced state of a Gibbs state.

    The generator :math:`K` of the state is split into the terms inside the
    subsystem :math:`A`, the terms in the environment and the boundary terms.
    The boundary terms are projected over the algebra of :math:`A` using the
    self-consistent mean-field state, giving :math:`K_{A,MF}`. Optionally,
    the RPA correction :math:`K_{A,RPA}` of the Gaussian fluctuations of the
    environment is added (see :mod:`qalma.meanfield.rpa`).

    Parameters
    ----------
    state : GibbsDensityOperator
        The Gibbs state to reduce.
    sites : frozenset
        Sites of the subsystem ``A``.
    rpa : bool, optional
        If ``True``, add the RPA correction. The default is False.
    dynamic : bool, optional
        If ``True`` (and ``rpa``), use the dynamical RPA correction instead
        of the static one. The default is False.
    n_tau : int, optional
        Number of imaginary-time quadrature nodes of the dynamical
        correction. The default is 32.

    Returns
    -------
    k_reduced : Operator
        The approximate generator of the reduced state.
    has_boundary : bool
        Whether there were terms coupling ``A`` with the environment.

    Raises
    ------
    ValueError
        If ``rpa`` is requested and the RPA is unstable.
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
    has_boundary = bool(terms_boundary)
    if rpa and has_boundary:
        # pylint: disable=import-outside-toplevel
        from qalma.meanfield.rpa import rpa_generator_correction

        # States that store the RPA data (see RPAGibbsDensityOperator)
        # provide it through the ``rpa_structure`` attribute.
        correction = rpa_generator_correction(
            generator,
            k_in,
            sites,
            sigma_mf,
            dynamic=dynamic,
            n_tau=n_tau,
            structure=getattr(state, "rpa_structure", None),
        )
        if correction is not None:
            k_in = (k_in + correction).simplify()
    return k_in, has_boundary


def gibbs_meanfield_partial_trace(
    state: GibbsDensityOperator,
    sites: frozenset,
    rpa: bool = False,
    dynamic: bool = False,
    n_tau: int = 32,
    exact: bool = True,
    hybrid: bool = False,
    hybrid_options: Optional[dict] = None,
) -> DensityOperatorProtocol:
    """Build a self-consistent Mean Field approximation to the local state.

    Parameters
    ----------
    state : GibbsDensityOperator
        The Gibbs state to reduce.
    sites : frozenset
        Sites of the subsystem to keep.
    rpa : bool, optional
        If ``True``, add to the mean-field generator the RPA correction
        (see :func:`gibbs_meanfield_generator`). It only applies to the
        mean-field branch: reductions that are computed exactly (small
        systems) ignore it. The default is False.
    dynamic : bool, optional
        If ``True`` (and ``rpa``), use the dynamical RPA correction.
    n_tau : int, optional
        Number of imaginary-time quadrature nodes of the dynamical
        correction.
    exact : bool, optional
        If ``True`` (default), reductions of small systems or of operators
        acting on a small block are computed exactly. If ``False``, the
        mean-field branch is always used.
    hybrid : bool, optional
        If ``True`` (and ``rpa``), the subsystem is kept exact and only the
        environment is treated in the RPA, solving a non perturbative hybrid
        problem (see :func:`qalma.meanfield.rpa.rpa_hybrid_state`), instead
        of adding a correction to the generator. The default is False.
    hybrid_options : dict, optional
        Options of :func:`qalma.meanfield.rpa.rpa_hybrid_state`.

    Returns
    -------
    DensityOperatorProtocol
        The reduced density operator.
    """
    prefactor: complex
    generator: Operator
    full_acts_over: frozenset
    environment: frozenset
    system: SystemDescriptor
    subsystem: SystemDescriptor

    # For states in small subsystems, just compute the partial trace
    # *exactly* by exponentiating the state.
    if exact and len(state.system.sites) <= MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE:
        result = state.to_qutip_operator().partial_trace(sites)
        return result

    prefactor = state.prefactor
    generator = state.k
    full_acts_over = generator.acts_over()
    environment = frozenset(site for site in full_acts_over if site not in sites)
    system = state.k.system
    subsystem = system.subsystem(sites)
    # pylint: disable=protected-access
    sigma_mf = state._meanfield

    # Trivial cases:
    if len(environment) == 0:
        return GibbsDensityOperator(
            generator, system=subsystem, prefactor=prefactor, meanfield=sigma_mf
        )
    if len(full_acts_over) == len(environment):
        return GibbsDensityOperator(
            ScalarOperator(0.0, system), system=subsystem, prefactor=prefactor
        )
    # Shortcut for density operators acting on a small subblock:
    # construct the state of the subblock, compute the partial trace,
    # and get back the generator:
    if exact and len(full_acts_over) <= MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE:
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
        # pylint: disable=protected-access
        k_reduced._set_system_(system)
        # Notice that subsystem can still be "large", so we return a GibbsDensityOperator:
        return GibbsDensityOperator(k_reduced, system=subsystem, prefactor=prefactor)

    k_in, has_boundary = gibbs_meanfield_generator(
        state, sites, rpa=rpa and not hybrid, dynamic=dynamic, n_tau=n_tau
    )
    if rpa and hybrid and has_boundary:
        # pylint: disable=import-outside-toplevel
        from qalma.meanfield.rpa import build_rpa_structure, rpa_hybrid_state
        from qalma.operators.states import QutipDensityOperator

        structure = getattr(state, "rpa_structure", None)
        if structure is None:
            structure = build_rpa_structure(
                state.k, state._meanfield  # pylint: disable=protected-access
            )
        rho = rpa_hybrid_state(structure, sites, k_in, **(hybrid_options or {}))
        block = tuple(sorted(sites))
        result = QutipDensityOperator(
            rho,
            names={name: pos for pos, name in enumerate(block)},
            system=subsystem,
            prefactor=prefactor,
        )
    else:
        result = GibbsDensityOperator(
            k_in, subsystem, prefactor=prefactor
        ).to_qutip_operator()

    # If there were non-trivial terms in the boundary,
    # restore symmetries:
    if has_boundary:
        for symm in state.symmetry_projections:
            result = symm(result)
    return result
