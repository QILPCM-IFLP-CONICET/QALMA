"""Tests for PauliSumOperator."""

import itertools
import pickle

import numpy as np
import pytest

from qalma.model import build_spin_chain, build_system
from qalma.operators import (
    PauliStringOperator,
    PauliSumOperator,
    ProductOperator,
    ScalarOperator,
    SumOperator,
    pauli_sum as pauli_sum_module,
    to_pauli_sum,
)
from qalma.operators.functions import anticommutator, commutator
from qalma.operators.pauli import pauli_product
from qalma.operators.states import GibbsProductDensityOperator

SYSTEM = build_spin_chain(5)
SITES = tuple(SYSTEM.sites)
BLOCK = tuple(sorted(SITES))
RNG = np.random.default_rng(42)


def dense(op, system=SYSTEM):
    block = tuple(sorted(system.sites))
    if isinstance(op, ScalarOperator):
        return op.prefactor * np.eye(2 ** len(block))
    return op.to_qutip(block).full()


def random_sum(num_terms=12, system=SYSTEM, weight=None):
    size = len(system.sites)
    codes = []
    for _ in range(num_terms):
        local = RNG.integers(0, 4, size=size)
        if weight is not None:
            local[RNG.permutation(size)[weight:]] = 0
        codes.append(sum(int(c) << (2 * k) for k, c in enumerate(local)))
    coeffs = RNG.normal(size=num_terms) + 1j * RNG.normal(size=num_terms)
    return PauliSumOperator(codes, coeffs, system)


def reference_product(op_1, op_2, mode="mul"):
    """Products term by term with PauliStringOperator, as a dict ``code -> coeff``."""
    result = {}

    def add(term, sign):
        code = getattr(term, "code", 0)
        result[code] = result.get(code, 0) + sign * term.prefactor

    for t_1, t_2 in itertools.product(op_1.terms, op_2.terms):
        add(t_1 * t_2, 1)
        if mode == "commutator":
            add(t_2 * t_1, -1)
    return {c: v for c, v in result.items() if abs(v) > 1e-12}


def test_canonical_form():
    op = PauliSumOperator([3, 1, 3, 0, 1], [1.0, 2.0, 0.5, 4.0, -2.0], SYSTEM)
    assert op.to_dict() == {0: 4.0, 3: 1.5}
    assert list(op.codes[:, 0]) == [0, 3]
    assert len(op.terms) == 2
    assert isinstance(op.terms[0], ScalarOperator)
    assert isinstance(op.terms[1], PauliStringOperator)
    assert isinstance(
        PauliSumOperator([5], [1.0], SYSTEM).collapse(), PauliStringOperator
    )
    assert not PauliSumOperator([5, 5], [1.0, -1.0], SYSTEM)


@pytest.mark.parametrize("trial", range(5))
def test_algebra_against_dense(trial):
    op_1, op_2 = random_sum(), random_sum()
    d_1, d_2 = dense(op_1), dense(op_2)
    assert np.allclose(dense(op_1), sum(dense(t) for t in op_1.terms))
    assert np.allclose(dense(op_1 * op_2), d_1 @ d_2)
    assert np.allclose(dense(op_1 + op_2), d_1 + d_2)
    assert np.allclose(dense(op_1 - op_2), d_1 - d_2)
    assert np.allclose(dense(commutator(op_1, op_2)), d_1 @ d_2 - d_2 @ d_1)
    assert np.allclose(dense(anticommutator(op_1, op_2)), d_1 @ d_2 + d_2 @ d_1)
    assert np.allclose(dense(op_1.dag()), d_1.conj().T)
    assert np.allclose(dense(2.5j * op_1), 2.5j * d_1)
    assert np.allclose(dense(op_1 + 0.7), d_1 + 0.7 * np.eye(len(d_1)))
    assert np.allclose(dense(op_1**2), d_1 @ d_1)
    assert np.isclose(op_1.tr(), np.trace(d_1))
    assert np.isclose(op_1.norm("fro"), np.linalg.norm(d_1, "fro"))
    assert op_1.hermitian_part().isherm
    assert np.allclose(dense(op_1.hermitian_part()), 0.5 * (d_1 + d_1.conj().T))


def test_mixed_with_strings_scalars_and_other_operators():
    op = random_sum()
    d_op = dense(op)
    string = PauliStringOperator.from_label("XZIYI", SYSTEM, 0.3)
    scalar = ScalarOperator(1.5, SYSTEM)
    local = SYSTEM.site_operator(f"Sx@{SITES[1]}")
    product = ProductOperator(
        {SITES[0]: local.operator, SITES[3]: local.operator}, 2.0, SYSTEM
    )
    ham = SYSTEM.global_operator("Hamiltonian")
    for other in (string, scalar, local, product, ham, ham.to_qutip_operator()):
        d_other = dense(other)
        assert np.allclose(dense(op * other), d_op @ d_other)
        assert np.allclose(dense(other * op), d_other @ d_op)
        assert np.allclose(dense(op + other), d_op + d_other)
        assert np.allclose(
            dense(commutator(op, other)), d_op @ d_other - d_other @ d_op
        )
    # Results within the Pauli algebra keep the representation.
    assert isinstance(op * string, PauliSumOperator)
    assert isinstance(
        string + 2 * string.dag() + PauliStringOperator.from_label("IIIIZ", SYSTEM),
        PauliSumOperator,
    )
    assert isinstance(commutator(op, to_pauli_sum(ham)), PauliSumOperator)


def test_partial_trace_reduce_and_acts_over():
    op = random_sum(20, weight=2)
    kept = frozenset(SITES[:2])
    reduced = op.partial_trace(kept)
    expected = sum(
        (t.partial_trace(kept) for t in op.terms[1:]), op.terms[0].partial_trace(kept)
    )
    assert np.allclose(dense(reduced, reduced.system), dense(expected, reduced.system))
    assert op.acts_over() == frozenset().union(*(t.acts_over() for t in op.terms))
    normalized = op.reduce(kept)
    for term in getattr(normalized, "terms", (normalized,)):
        assert term.acts_over() <= kept
    # Without a state, reduce keeps the terms supported inside ``kept``.
    expected_terms = {
        getattr(t, "code", 0): t.prefactor for t in op.terms if t.acts_over() <= kept
    }
    got = (
        normalized.to_dict()
        if isinstance(normalized, PauliSumOperator)
        else {getattr(normalized, "code", 0): normalized.prefactor}
    )
    assert got.keys() == expected_terms.keys()


def test_expectation_over_product_states():
    op = random_sum(30)
    k = sum(
        (
            0.3 * SYSTEM.site_operator(f"Sx@{s}")
            + 0.2 * (i + 1) * SYSTEM.site_operator(f"Sz@{s}")
            for i, s in enumerate(SITES)
        ),
        0.0 * SYSTEM.site_operator(f"Sz@{SITES[0]}"),
    )
    state = GibbsProductDensityOperator(k, system=SYSTEM)
    rho = state.to_qutip(BLOCK).full()
    assert np.isclose(state.expect(op), np.trace(rho @ dense(op)))
    assert np.isclose(state.expect(op), sum(state.expect(term) for term in op.terms))


def test_many_words():
    """Systems with more than 32 sites need more than one 64-bit word."""
    system = build_spin_chain(70)
    assert pauli_sum_module._num_words(system) == 3
    op_1 = random_sum(15, system, weight=4)
    op_2 = random_sum(15, system, weight=4)
    for mode, result in (
        ("mul", op_1 * op_2),
        ("commutator", commutator(op_1, op_2)),
    ):
        expected = reference_product(op_1, op_2, mode)
        got = (
            result.to_dict()
            if isinstance(result, PauliSumOperator)
            else {getattr(result, "code", 0): result.prefactor}
        )
        assert expected.keys() == got.keys()
        assert all(np.isclose(expected[c], got[c]) for c in expected)


@pytest.mark.parametrize("dtype", [np.uint64, np.int64])
def test_popcount_fallback(dtype):
    values = RNG.integers(0, 2**62, size=(4, 3)).astype(dtype)
    expected = np.vectorize(lambda v: bin(int(v)).count("1"))(values)
    assert np.array_equal(pauli_sum_module._popcount_bytes(values), expected)
    assert np.array_equal(pauli_sum_module._popcount(values), expected)


def test_fermionic_models_use_pauli_sums():
    system = build_system("chain lattice", "fermion Hubbard", L=3, t=1, U=2, mu=0.3)
    ham = system.global_operator("Hamiltonian")
    assert isinstance(ham, PauliSumOperator)
    assert isinstance(system.site_operator("c_up", "1[1]"), PauliSumOperator)
    result = system.site_operator("c_up", "1[0]")
    for _ in range(3):
        result = commutator(ham, result)
        assert isinstance(result, PauliSumOperator)
    dense_h = dense(ham, system)
    expected = dense(system.site_operator("c_up", "1[0]"), system)
    for _ in range(3):
        expected = dense_h @ expected - expected @ dense_h
    assert np.allclose(dense(result, system), expected)


def test_generic_sum_operations_still_work():
    """Routines written for SumOperator see the terms."""
    op = random_sum(8)
    assert isinstance(op, SumOperator)
    assert len(op.terms) == op.num_terms() == 8
    generic = SumOperator(op.terms, SYSTEM)
    assert np.allclose(dense(generic), dense(op))
    assert np.allclose(
        dense(pauli_product(op.terms[1], op.terms[2])),
        dense(op.terms[1]) @ dense(op.terms[2]),
    )


def test_pickle_does_not_store_cached_terms():
    op = random_sum(200)
    size_before = len(pickle.dumps(op))
    _ = op.terms
    assert len(pickle.dumps(op)) == size_before
    restored = pickle.loads(pickle.dumps(op))
    assert restored.to_dict() == op.to_dict()
    string = op.terms[3]
    _ = string.site_factors
    restored_string = pickle.loads(pickle.dumps(string))
    assert "site_factors" not in restored_string.__dict__
    assert restored_string.code == string.code
