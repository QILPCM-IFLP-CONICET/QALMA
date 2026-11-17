"""Symbolic simplification of sums with Pauli strings and tensor products."""

import warnings

import numpy as np
import pytest

from qalma import build_system
from qalma.model import build_spin_chain, build_system_from_dims
from qalma.operators import (
    LocalOperator,
    PauliStringOperator,
    PauliSumOperator,
    ProductOperator,
    QutipOperator,
    ScalarOperator,
    SumOperator,
    TensorProductOperator,
)
from qalma.operators.functions import commutator

# Two-level sites a, c, e; three-level sites b, d.
SYSTEM = build_system_from_dims({"a": 2, "b": 3, "c": 2, "d": 3, "e": 2})
BLOCK = tuple(SYSTEM.sites)
RNG = np.random.default_rng(23)


def dense(op, block=BLOCK, system=SYSTEM):
    if isinstance(op, ScalarOperator):
        dim = int(np.prod([system.dimensions[s] for s in block]))
        return op.prefactor * np.eye(dim)
    return op.to_qutip(block).full()


def matrix(dim):
    return RNG.normal(size=(dim, dim)) + 1j * RNG.normal(size=(dim, dim))


def code(**labels):
    positions = {"a": 0, "c": 2, "e": 4}
    return sum(("IXYZ".index(v)) << (2 * positions[k]) for k, v in labels.items())


def random_terms():
    """A mixture of every kind of term, with repetitions."""
    a_b = TensorProductOperator(code(a="X"), {"b": matrix(3)}, 0.7, SYSTEM)
    return [
        PauliStringOperator({"a": "X", "c": "Z"}, 0.5, SYSTEM),
        PauliStringOperator({"a": "X", "c": "Z"}, -0.25j, SYSTEM),
        PauliSumOperator([code(e="Y"), code(a="Z", e="X")], [1.0, 2.0j], SYSTEM),
        a_b,
        2.0 * a_b,
        TensorProductOperator(code(a="X"), {"b": matrix(3)}, 1.0, SYSTEM),
        TensorProductOperator(
            code(c="Y"), {"b": matrix(3), "d": matrix(3)}, 1.0, SYSTEM
        ),
        TensorProductOperator(code(e="Z"), {"d": np.eye(3)}, 3.0, SYSTEM),
        LocalOperator("b", matrix(3), SYSTEM),
        LocalOperator("b", matrix(3), SYSTEM),
        LocalOperator("e", matrix(2), SYSTEM),
        ProductOperator({"b": matrix(3), "c": matrix(2)}, 1.5, SYSTEM),
        ScalarOperator(0.75, SYSTEM),
        QutipOperator(
            ProductOperator({"b": matrix(3), "d": matrix(3)}, 1.0, SYSTEM).to_qutip(
                ("b", "d")
            ),
            SYSTEM,
            names={"b": 0, "d": 1},
        ),
    ]


def test_mixture_is_simplified_and_equivalent():
    terms = random_terms()
    total = SumOperator(tuple(terms), SYSTEM)
    result = total.simplify()
    assert np.allclose(dense(result), sum(dense(t) for t in terms))
    kinds = [type(term) for term in result.terms]
    # A single sum of Pauli strings collects the strings, the scalar, the
    # local operator on the two-level site and the identity factor.
    assert kinds.count(PauliSumOperator) == 1
    assert ScalarOperator not in kinds and PauliStringOperator not in kinds
    # Terms with the same code and a single local site are merged.
    single = [
        (t.code, next(iter(t.local_factors)))
        for t in result.terms
        if isinstance(t, TensorProductOperator) and len(t.local_factors) == 1
    ]
    assert len(single) == len(set(single))
    locals_ = [t.site for t in result.terms if isinstance(t, LocalOperator)]
    assert locals_ == ["b"]
    # Idempotent.
    assert result.simplify() is result


def test_cancellations_and_identities():
    a_b = TensorProductOperator(code(a="X"), {"b": matrix(3)}, 1.0, SYSTEM)
    string = PauliStringOperator({"a": "X"}, 1.0, SYSTEM)
    assert isinstance(SumOperator((a_b, -a_b), SYSTEM).simplify(), ScalarOperator)
    assert (a_b - a_b).is_zero
    identity_factor = TensorProductOperator(
        code(a="X"), {"b": 2 * np.eye(3)}, 1.0, SYSTEM
    )
    result = SumOperator((identity_factor, -2 * string), SYSTEM).simplify()
    assert isinstance(result, ScalarOperator) and not result
    # The trace of a single-site factor goes to the Pauli part.
    factor = np.diag([1.0, 2.0, 6.0])
    result = SumOperator(
        (TensorProductOperator(code(a="X"), {"b": factor}, 1.0, SYSTEM), -3 * string),
        SYSTEM,
    ).simplify()
    assert isinstance(result, TensorProductOperator)
    assert np.isclose(np.trace(result.local_factors["b"]), 0)


def test_products_differing_on_one_site_are_merged():
    shared = matrix(3)
    first = TensorProductOperator(
        code(c="Y"), {"b": shared, "d": matrix(3)}, 1.0, SYSTEM
    )
    second = TensorProductOperator(
        code(c="Y"), {"b": 2.0 * shared, "d": matrix(3)}, 0.5j, SYSTEM
    )
    result = (first + second).simplify()
    assert isinstance(result, TensorProductOperator)
    assert np.allclose(dense(result), dense(first) + dense(second))
    # Factors differing on both sites are kept apart.
    other = TensorProductOperator(
        code(c="Y"), {"b": matrix(3), "d": matrix(3)}, 1.0, SYSTEM
    )
    result = (first + other).simplify()
    assert isinstance(result, SumOperator) and len(result.terms) == 2


def test_hubbard_sums_become_a_pauli_sum():
    system = build_system("chain lattice", "fermion Hubbard", L=3, t=1, U=2)
    ham = system.global_operator("Hamiltonian")
    mixture = SumOperator(
        (ham, 0.3 * ham.terms[0], SumOperator((ham.terms[1], ham.terms[2]), system)),
        system,
    )
    result = mixture.simplify()
    assert isinstance(result, PauliSumOperator)
    block = tuple(system.sites)
    assert np.allclose(
        dense(result, block, system),
        dense(ham, block, system)
        + 0.3 * dense(ham.terms[0], block, system)
        + dense(ham.terms[1], block, system)
        + dense(ham.terms[2], block, system),
    )


def test_t_j_sums_keep_the_tensor_products():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = build_system("chain lattice", "t-J", L=4, t=1, J=0.5, mu=0.2)
    block = tuple(system.sites)
    ham = system.global_operator("Hamiltonian")
    simplified = ham.simplify()
    doubled = (ham + ham).simplify()
    assert len(doubled.terms) == len(simplified.terms)
    assert np.allclose(dense(doubled, block, system), 2 * dense(ham, block, system))
    c_up = system.site_operator("c_up", "1[0]")
    result = commutator(ham, c_up)
    assert not any(isinstance(t, QutipOperator) for t in result.terms)
    d_h, d_c = dense(ham, block, system), dense(c_up, block, system)
    assert np.allclose(dense(result, block, system), d_h @ d_c - d_c @ d_h)


def test_sums_without_pauli_terms_keep_the_generic_simplification():
    system = build_spin_chain(4)
    ham = system.global_operator("Hamiltonian")
    result = (ham + ham).simplify()
    terms = getattr(result, "terms", (result,))
    assert not any(
        isinstance(t, (PauliSumOperator, TensorProductOperator)) for t in terms
    )


@pytest.mark.parametrize("trial", range(5))
def test_random_sums(trial):
    terms = random_terms()
    picked = [terms[k] for k in RNG.choice(len(terms), size=6)]
    total = SumOperator(tuple(picked), SYSTEM)
    assert np.allclose(dense(total.simplify()), sum(dense(t) for t in picked))


def test_pauli_sum_to_qutip_with_sites_of_other_dimensions():
    """The strings act on two-level sites; the other sites get identities."""
    op = PauliSumOperator(
        [code(a="X", e="Z"), code(c="Y"), 0], [1.0, 0.5j, 2.0], SYSTEM
    )
    expected = sum(dense(term) for term in op.terms)
    assert np.allclose(dense(op), expected)
    shuffled = ("d", "e", "b", "a", "c")
    assert np.allclose(
        dense(op, shuffled), sum(dense(term, shuffled) for term in op.terms)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kondo = build_system(
            "2 band open chain lattice", "Kondo lattice", L=2, local_S=1
        )
    c_up = kondo.site_operator("c_up", "1[1]")
    assert isinstance(c_up, PauliSumOperator)
    full = c_up.to_qutip(tuple(kondo.sites))
    assert full.dims[0] == [kondo.dimensions[s] for s in kondo.sites]
