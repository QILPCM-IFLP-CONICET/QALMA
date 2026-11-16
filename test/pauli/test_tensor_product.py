"""Tests for TensorProductOperator."""

import pickle

import numpy as np
import pytest

from qalma.model import build_system_from_dims
from qalma.operators import (
    LocalOperator,
    PauliStringOperator,
    ProductOperator,
    ScalarOperator,
    SumOperator,
    TensorProductOperator,
)
from qalma.operators.functions import commutator

# Two-level sites a, c, e for the Pauli parts; three-level sites b, d.
SYSTEM = build_system_from_dims({"a": 2, "b": 3, "c": 2, "d": 3, "e": 2})
BLOCK = tuple(SYSTEM.sites)
RNG = np.random.default_rng(11)


def dense(op):
    if isinstance(op, ScalarOperator):
        return op.prefactor * np.eye(2 * 3 * 2 * 3 * 2)
    return op.to_qutip(BLOCK).full()


def random_matrix(dim):
    return RNG.normal(size=(dim, dim)) + 1j * RNG.normal(size=(dim, dim))


def random_tensor_product():
    code = sum(int(c) << (2 * k) for k, c in zip((0, 2), RNG.integers(1, 4, size=2)))
    return TensorProductOperator(
        code,
        {"b": random_matrix(3), "d": random_matrix(3)},
        complex(*RNG.normal(size=2)),
        SYSTEM,
    )


def other_operators():
    return [
        PauliStringOperator({"a": "X", "c": "Z"}, 0.5, SYSTEM),
        PauliStringOperator({"e": "Y"}, 1.5j, SYSTEM),
        ProductOperator({"b": random_matrix(3), "e": random_matrix(2)}, 1.3, SYSTEM),
        LocalOperator("d", random_matrix(3), SYSTEM),
        # overlaps the Pauli part of the tensor products: dense fallback
        LocalOperator("a", random_matrix(2), SYSTEM),
        ScalarOperator(2.0, SYSTEM),
    ]


def test_construction_and_parts():
    op = TensorProductOperator(0b01, {"b": np.diag([1.0, 2.0, 3.0])}, 0.5, SYSTEM)
    assert op.paulis == {"a": 1}
    assert op.acts_over() == frozenset({"a", "b"})
    assert set(op.site_factors) == {"a", "b"}
    assert np.allclose(dense(op), dense(0.5 * op.pauli_part * op.local_part))
    with pytest.raises(ValueError):
        TensorProductOperator(0b01, {"a": np.eye(2)}, 1.0, SYSTEM)


@pytest.mark.parametrize("trial", range(10))
def test_products_against_dense(trial):
    x_op, y_op = random_tensor_product(), random_tensor_product()
    assert isinstance(x_op * y_op, TensorProductOperator)
    for left, right in [(x_op, y_op)] + [
        pair for other in other_operators() for pair in ((x_op, other), (other, x_op))
    ]:
        assert np.allclose(dense(left * right), dense(left) @ dense(right))


def test_pauli_times_products_are_tensor_products():
    string = PauliStringOperator({"a": "X", "c": "Y"}, 1.0, SYSTEM)
    product = ProductOperator(
        {"b": random_matrix(3), "d": random_matrix(3)}, 2.0, SYSTEM
    )
    local = LocalOperator("b", random_matrix(3), SYSTEM)
    for result in (string * product, product * string, string * local, local * string):
        assert isinstance(result, TensorProductOperator)
    # The strings cancel symbolically: no identity factors are left.
    result = (string * product) * (string * local)
    assert result.acts_over() == frozenset({"b", "d"})
    assert not isinstance(result, TensorProductOperator)


def test_unary_operations_and_sums():
    op = random_tensor_product()
    other = random_tensor_product()
    assert np.allclose(dense(op.dag()), dense(op).conj().T)
    assert np.allclose(dense(-op), -dense(op))
    assert np.allclose(dense(3j * op), 3j * dense(op))
    assert np.isclose(op.tr(), np.trace(dense(op)))
    identity_string = TensorProductOperator(0, {"b": random_matrix(3)}, 1.0, SYSTEM)
    assert np.isclose(identity_string.tr(), np.trace(dense(identity_string)))
    total = op + other
    assert isinstance(total, SumOperator)
    assert np.allclose(dense(total), dense(op) + dense(other))
    assert np.allclose(
        dense(commutator(op, other)),
        dense(op) @ dense(other) - dense(other) @ dense(op),
    )
    assert np.allclose(dense(op.to_product_operator()), dense(op))


def test_simplify_and_pickle():
    no_string = TensorProductOperator(0, {"b": np.eye(3), "d": np.eye(3)}, 2.0, SYSTEM)
    assert type(no_string.simplify()) is ProductOperator
    no_local = TensorProductOperator._new(0b0101, {}, 2.0, SYSTEM)
    assert isinstance(no_local.simplify(), PauliStringOperator)
    op = random_tensor_product()
    _ = op.site_factors
    restored = pickle.loads(pickle.dumps(op))
    assert "site_factors" not in restored.__dict__
    assert np.allclose(dense(restored), dense(op))
