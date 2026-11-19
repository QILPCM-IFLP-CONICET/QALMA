"""Restricted Max-Ent evolution and its building blocks on fermionic models.

Fermionic models are represented with Pauli strings (Jordan-Wigner), so these
tests cover the paths where Pauli sums play the role of one-body operators.
"""

import numpy as np
import pytest
from scipy.linalg import eigh, expm

from qalma import build_system
from qalma.evolution import adaptive_maxent_evolution, update_basis
from qalma.operators import PauliStringOperator, PauliSumOperator, ProductOperator
from qalma.operators.quadratic import QuadraticFormOperator
from qalma.operators.states import GibbsProductDensityOperator
from qalma.operators.states.utils import k_by_site_from_operator
from qalma.projections import n_body_projection
from qalma.projections.nbody import DISPATCH_PROJECTION_METHOD, projection_method


@pytest.fixture(scope="module")
def fermion_chain():
    """Chain of spinless fermions, its sites and the occupation operators."""
    system = build_system("open chain lattice", "spinless fermions", L=4)
    names = sorted(system.dimensions)
    occupations = [system.site_operator(f"n@{name}") for name in names]
    return system, tuple(system.sites), occupations


def gibbs(kmat):
    """rho = exp(-K) / Tr exp(-K) for a dense K."""
    evals, evecs = eigh(kmat)
    weights = np.exp(-(evals - evals.min()))
    return (evecs * (weights / weights.sum())) @ evecs.conj().T


def test_occupations_and_hamiltonian_are_pauli_sums(fermion_chain):
    """The model builder represents fermionic operators as Pauli sums."""
    system, _, occupations = fermion_chain
    assert isinstance(occupations[1], PauliSumOperator)
    assert isinstance(system.global_operator("Hamiltonian"), PauliSumOperator)


def test_one_body_pauli_sum_as_linear_term(fermion_chain):
    """A Pauli sum acting on single sites is accepted as a linear term."""
    system, sites, occupations = fermion_chain
    linear = (0.3 * occupations[0] + 0.7 * occupations[2]).simplify()
    form = QuadraticFormOperator(tuple(), tuple(), system, linear, None)
    diff = form.to_qutip(sites) - linear.to_qutip(sites)
    assert np.abs(diff.full()).max() < 1e-12


def test_quadratic_form_rejects_many_body_linear_terms(fermion_chain):
    """A linear term acting on two sites is rejected."""
    system, _, _ = fermion_chain
    hop = system.global_operator("Hamiltonian")
    with pytest.raises(ValueError, match="not a one-body operator"):
        QuadraticFormOperator(tuple(), tuple(), system, hop, None)


def test_gibbs_product_state_from_pauli_generator(fermion_chain):
    """A one-body generator given as a Pauli sum defines a product state."""
    system, sites, occupations = fermion_chain
    k = (0.4 * occupations[1] - 0.2 * occupations[3]).simplify()
    factors = k_by_site_from_operator(k)
    assert set(factors) == set(k.acts_over())
    state = GibbsProductDensityOperator(k)
    expected = gibbs(k.to_qutip(sites).full())
    assert np.abs(state.to_qutip(sites).full() - expected).max() < 1e-12


def test_projection_dispatch_follows_the_class_hierarchy(fermion_chain):
    """Pauli strings use the n-body projection of product operators."""
    system, sites, occupations = fermion_chain
    ham = system.global_operator("Hamiltonian")
    string = next(t for t in ham.terms if isinstance(t, PauliStringOperator))
    assert projection_method(string) is DISPATCH_PROJECTION_METHOD[ProductOperator]
    sigma = GibbsProductDensityOperator((0.3 * occupations[0]).simplify())
    one_body = n_body_projection(ham, n_max=1, sigma=sigma)
    assert one_body.n_body_sector() <= 1


def test_update_basis_on_a_fermionic_model(fermion_chain):
    """The hierarchical basis can be built from a Pauli-sum generator."""
    system, sites, occupations = fermion_chain
    ham = system.global_operator("Hamiltonian")
    k0 = (0.7 * occupations[1]).simplify()
    basis, sigma, _ = update_basis(
        k0, GibbsProductDensityOperator({}, system), ham, 3, 2, ()
    )
    assert len(basis.operator_basis) > 1
    assert np.linalg.cond(basis.gram) < 1e8


def test_fermionic_evolution_against_exact(fermion_chain):
    """Occupations follow the exact evolution, better with the memory closure."""
    system, sites, occupations = fermion_chain
    ham = system.global_operator("Hamiltonian")
    k0 = (0.7 * occupations[1]).simplify()
    t_span = np.linspace(0, 3, 13)
    hmat = ham.to_qutip(sites).full()
    k0mat = k0.to_qutip(sites).full()
    mats = [op.to_qutip(sites).full() for op in occupations]
    exact = np.array(
        [
            [
                np.real(
                    np.trace(
                        gibbs(expm(-1j * t * hmat) @ k0mat @ expm(1j * t * hmat)) @ m
                    )
                )
                for m in mats
            ]
            for t in t_span
        ]
    )
    errors, short_time_errors = {}, {}
    for memory_time in (None, "auto"):
        sim = adaptive_maxent_evolution(
            ham, k0, t_span, 4, 2, memory_time=memory_time, store_states=True
        )
        approx = np.array(
            [
                [np.real(np.trace(gibbs(k.to_qutip(sites).full()) @ m)) for m in mats]
                for k in sim.states
            ]
        )
        deviation = np.abs(approx - exact)
        errors[memory_time] = deviation.max()
        short_time_errors[memory_time] = deviation[t_span <= 1.5].max()
    assert short_time_errors["auto"] < 0.03
    assert errors["auto"] < 0.6 * errors[None]
