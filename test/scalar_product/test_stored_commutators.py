"""Operator bases store the commutators of their elements with the generator."""

import numpy as np
import pytest

from qalma import build_system
from qalma.evolution import leakage_gram, update_basis
from qalma.operators.functions import commutator
from qalma.operators.states import GibbsProductDensityOperator
from qalma.scalarprod import (
    HierarchicalOperatorBasis,
    OperatorBasis,
    fetch_covar_scalar_product,
)


@pytest.fixture(scope="module")
def chain():
    """A five-site XXZ chain in a field, a product state and some operators."""
    system = build_system("open chain lattice", "spin", L=5, J=1.0)
    sites = tuple(system.sites)
    sx = [system.site_operator(f"Sx@{s}") for s in sites]
    sy = [system.site_operator(f"Sy@{s}") for s in sites]
    sz = [system.site_operator(f"Sz@{s}") for s in sites]
    ham = (
        sum(
            sx[i] * sx[i + 1] + sy[i] * sy[i + 1] + 0.5 * sz[i] * sz[i + 1]
            for i in range(4)
        )
        + system.global_operator("Sz")
    ).simplify()
    sigma = GibbsProductDensityOperator((0.3 * sum(sx) - 0.2 * sz[1]).simplify())
    return system, sites, ham, sx, sz, sigma.to_product_state()


def check_commutators(basis, sites):
    """Each stored commutator equals [Q_j, G] for the j-th element."""
    comms = basis._commutators
    assert len(comms) == len(basis.operator_basis)
    for op, comm in zip(basis.operator_basis, comms):
        expected = commutator(op, basis.generator)
        diff = (comm - expected).to_qutip(sites).full()
        assert np.abs(diff).max() < 1e-10


@pytest.mark.parametrize("deep", [1, 3])
def test_hierarchical_basis_stores_commutators(chain, deep):
    """The hierarchical basis keeps the unprojected commutators it computes."""
    system, sites, ham, sx, sz, sigma = chain
    sp = fetch_covar_scalar_product(sigma)
    basis = HierarchicalOperatorBasis(0.5 * sx[1], ham, deep, sp)
    check_commutators(basis, sites)


def test_closed_hierarchical_basis_stores_commutators(chain):
    """When the iterated commutators close, there is one commutator per element."""
    system, sites, ham, sx, sz, sigma = chain
    heis = system.global_operator("Hamiltonian")
    sp = fetch_covar_scalar_product(GibbsProductDensityOperator({}, system))
    basis = HierarchicalOperatorBasis(system.global_operator("Sz"), heis, 3, sp)
    check_commutators(basis, sites)


def test_generic_basis_stores_commutators(chain):
    """A generic basis with a generator stores its commutators."""
    system, sites, ham, sx, sz, sigma = chain
    sp = fetch_covar_scalar_product(sigma)
    basis = OperatorBasis((sx[0], sz[2], (sx[1] * sz[2]).simplify()), 1j * ham, sp)
    check_commutators(basis, sites)


def test_appended_basis_reuses_and_stores_commutators(chain):
    """Appending operators keeps one commutator per element, in order."""
    system, sites, ham, sx, sz, sigma = chain
    sp = fetch_covar_scalar_product(sigma)
    hierarchical = HierarchicalOperatorBasis(0.5 * sx[1], ham, 2, sp)
    extended = hierarchical + (sz[3], (sx[0] * sx[1]).simplify())
    check_commutators(extended, sites)
    # The commutators of the hierarchical part are reused, not recomputed.
    n_hier = len(hierarchical.operator_basis)
    for stored, reused in zip(
        hierarchical._commutators, extended._commutators[:n_hier]
    ):
        assert stored is reused


def test_update_basis_and_leakage_gram_use_stored_commutators(chain):
    """The basis of the evolution stores its commutators, and E does not change."""
    system, sites, ham, sx, sz, sigma = chain
    k = (0.5 * sx[1] + 0.2 * sx[1] * sx[2]).simplify()
    basis, _, _ = update_basis(
        k, GibbsProductDensityOperator({}, system), ham, 3, 2, ()
    )
    check_commutators(basis, sites)
    e_mat = leakage_gram(basis)
    assert basis._leakage_commutators is not None
    # Same matrix from commutators computed from scratch.
    fresh = OperatorBasis(basis.operator_basis, basis.generator, basis.sp)
    del fresh._commutators  # force leakage_gram to compute them again
    assert np.allclose(leakage_gram(fresh), e_mat, atol=1e-10)
