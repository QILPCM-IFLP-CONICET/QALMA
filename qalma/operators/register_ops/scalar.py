"""
Arithmetic bindings for ScalarOperator.

Covers:
  - ScalarOperator + ScalarOperator
  - ScalarOperator * ScalarOperator
  - ScalarOperator +/* Number  (and reversed)
  - Operator + Operator  (base-class catch-all)
  - ProductOperator + OneBodyOperator  (catch-all override)
"""

from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import Operator
from qalma.operators.product import ProductOperator, ScalarOperator

from ._common import NUMERIC_TYPES


@Operator.register_add_handler(
    [
        (Operator, Operator),
        (ProductOperator, OneBodyOperator),
    ]
)
def _standard_sum_operator(op1: Operator, op2: Operator):
    system = op1.system.union(op2.system)
    return SumOperator(tuple((op1, op2)), system)


@Operator.register_add_handler((ScalarOperator, ScalarOperator))
def _(x_op: ScalarOperator, y_op: ScalarOperator):
    return ScalarOperator(x_op.prefactor + y_op.prefactor, x_op.system or y_op.system)


@Operator.register_mul_handler((ScalarOperator, ScalarOperator))
def _(x_op: ScalarOperator, y_op: ScalarOperator):
    return ScalarOperator(x_op.prefactor * y_op.prefactor, x_op.system or y_op.system)


@Operator.register_add_handler([(ScalarOperator, t) for t in NUMERIC_TYPES])
def _(x_op: ScalarOperator, y_value: complex) -> Operator:
    return ScalarOperator(x_op.prefactor + y_value, x_op.system)


@Operator.register_mul_handler([(ScalarOperator, t) for t in NUMERIC_TYPES])
def _(x_op: ScalarOperator, y_value: complex) -> Operator:
    return ScalarOperator(x_op.prefactor * y_value, x_op.system)


@Operator.register_mul_handler([(t, ScalarOperator) for t in NUMERIC_TYPES])
def _(y_value: complex, x_op: ScalarOperator):
    return ScalarOperator(x_op.prefactor * y_value, x_op.system)
