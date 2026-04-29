"""Gibbs partial trace.

This module implement functions to approximate the partial trace of a
Gibbs state.
"""

import logging
from typing import List

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


def project_boundary_term(term, sigma: ProductDensityOperator, sites: frozenset):
    """Convert terms of the form O_a Q_b in to O_a <Q_b> with <Q_b> the
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
        qutip_op = term.to_qutip(block)
        qutip_op = qutip_op * qutip_tensor(
            [
                local_states.get(site, None) or system.site_identity(site)
                for site in block
            ]
        )
        qutip_op = qutip_op.ptrace(list(range(len(sites)))) * term.prefactor
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


def gibbs_meanfield_partial_trace(
    state: GibbsDensityOperator, sites: frozenset
) -> DensityOperatorProtocol:
    """Build a self-consistent Mean Field approximation to the local state."""
    terms_in: List[Operator]
    terms_boundary: List[Operator]
    prefactor: complex
    generator: Operator
    full_acts_over: frozenset
    environment: frozenset
    system: SystemDescriptor
    subsystem: SystemDescriptor

    # For states in small subsystems, just compute the partial trace
    # *exactly* by exponentiating the state.
    if len(state.system.sites) <= MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE:
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
    if len(full_acts_over) <= MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE:
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

    # Decompose generator in terms inside (subsystem), terms outside (environment) and
    # boundary (interaction)
    generator = generator.flat()
    all_terms = generator.terms if isinstance(generator, SumOperator) else [generator]
    terms_in, terms_boundary = [], []
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

    result = GibbsDensityOperator(
        k_in, subsystem, prefactor=prefactor
    ).to_qutip_operator()

    # If there were non-trivial terms in the boundary,
    # restore symmetries:
    if terms_boundary:
        for symm in state.symmetry_projections:
            result = symm(result)
    return result
