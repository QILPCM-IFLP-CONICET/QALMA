"""Arithmetic bindings for TensorProductOperator.

Products between Pauli strings, tensor products, products and local
operators go through :func:`qalma.operators.tensor_product.tensor_product`,
which keeps the Pauli part symbolic whenever the supports allow it. Sums and
any other combination use the handlers of ProductOperator.
"""

from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.pauli import PauliStringOperator
from qalma.operators.product import ProductOperator, ScalarOperator
from qalma.operators.tensor_product import (
    TensorProductOperator,
    tensor_product,
    tensor_scale,
)
from qalma.operators.utils import register_dispatch_fallback

from ._common import NUMERIC_TYPES

register_dispatch_fallback(TensorProductOperator, ProductOperator)

_FACTOR_TYPES = (
    TensorProductOperator,
    PauliStringOperator,
    ProductOperator,
    LocalOperator,
    ScalarOperator,
)
_PAIRS = [
    (a, b)
    for a in _FACTOR_TYPES
    for b in _FACTOR_TYPES
    if TensorProductOperator in (a, b)
    or {a, b}
    in ({PauliStringOperator, ProductOperator}, {PauliStringOperator, LocalOperator})
]


@Operator.register_mul_handler(_PAIRS)
def _(x_op: Operator, y_op: Operator) -> Operator:
    return tensor_product(x_op, y_op)


@Operator.register_mul_handler(
    [(TensorProductOperator, t) for t in NUMERIC_TYPES]
    + [(t, TensorProductOperator) for t in NUMERIC_TYPES]
)
def _(x_op, y_op) -> Operator:
    if isinstance(x_op, TensorProductOperator):
        return tensor_scale(x_op, y_op)
    return tensor_scale(y_op, x_op)
