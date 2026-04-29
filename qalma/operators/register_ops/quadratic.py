"""Arithmetic bindings for QuadraticFormOperator.

Covers:
  - QuadraticFormOperator + Number
  - QuadraticFormOperator + OneBodyOperator / ScalarOperator / LocalOperator
  - QuadraticFormOperator + ProductOperator / QutipOperator
  - QuadraticFormOperator * Number  (and reversed)
  - QuadraticFormOperator * ScalarOperator  (and reversed)
  - QuadraticFormOperator * any Operator  (and reversed, via as_sum_of_products)
"""

from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.product import ProductOperator, ScalarOperator
from qalma.operators.quadratic import QuadraticFormOperator
from qalma.operators.qutip import QutipOperator

from ._common import NUMERIC_TYPES


@Operator.register_add_handler(
    [(QuadraticFormOperator, t) for t in NUMERIC_TYPES]
    + [
        (QuadraticFormOperator, op_type)
        for op_type in (OneBodyOperator, ScalarOperator, LocalOperator)
    ]
)
def _(qf_operator: QuadraticFormOperator, op_other: Operator):
    linear_term = qf_operator.linear_term
    if linear_term is None:
        if isinstance(op_other, NUMERIC_TYPES):
            op_other = ScalarOperator(op_other, qf_operator.system)
        linear_term = op_other
    else:
        linear_term = linear_term + op_other
    return QuadraticFormOperator(
        qf_operator.basis,
        qf_operator.weights,
        qf_operator.system,
        linear_term,
        qf_operator.offset,
    )


@Operator.register_add_handler(
    [(QuadraticFormOperator, op_type) for op_type in (ProductOperator, QutipOperator)]
)
def _(qf_operator: QuadraticFormOperator, op_other: Operator):
    offset = qf_operator.offset
    offset = op_other if offset is None else offset + op_other
    return QuadraticFormOperator(
        qf_operator.basis,
        qf_operator.weights,
        qf_operator.system,
        qf_operator.linear_term,
        offset,
    )


@Operator.register_mul_handler([(t, QuadraticFormOperator) for t in NUMERIC_TYPES])
def _(value: complex, qf_operator: QuadraticFormOperator):
    linear_term = qf_operator.linear_term
    offset = qf_operator.offset
    if linear_term is not None:
        linear_term = value * linear_term
    if offset is not None:
        offset = value * offset
    return QuadraticFormOperator(
        qf_operator.basis, qf_operator.weights, qf_operator.system, linear_term, offset
    )


@Operator.register_mul_handler([(QuadraticFormOperator, t) for t in NUMERIC_TYPES])
def _(qf_operator: QuadraticFormOperator, value: complex):
    linear_term = qf_operator.linear_term
    offset = qf_operator.offset
    if linear_term is not None:
        linear_term = value * linear_term
    if offset is not None:
        offset = value * offset
    return QuadraticFormOperator(
        qf_operator.basis,
        [w * value for w in qf_operator.weights],
        qf_operator.system,
        linear_term,
        offset,
    )


@Operator.register_mul_handler((QuadraticFormOperator, ScalarOperator))
def _(qf_operator: QuadraticFormOperator, sc_operator: ScalarOperator):
    system = qf_operator.system.union(sc_operator.system)
    value = sc_operator.prefactor
    linear_term = qf_operator.linear_term
    offset = qf_operator.offset
    if linear_term is not None:
        linear_term = value * linear_term
    if offset is not None:
        offset = value * offset
    return QuadraticFormOperator(
        qf_operator.basis,
        [w * value for w in qf_operator.weights],
        system,
        linear_term,
        offset,
    )


@Operator.register_mul_handler((ScalarOperator, QuadraticFormOperator))
def _(sc_operator: ScalarOperator, qf_operator: QuadraticFormOperator):
    system = qf_operator.system.union(sc_operator.system)
    value = sc_operator.prefactor
    linear_term = qf_operator.linear_term
    offset = qf_operator.offset
    if linear_term is not None:
        linear_term = value * linear_term
    if offset is not None:
        offset = value * offset
    return QuadraticFormOperator(
        qf_operator.basis,
        tuple(w * value for w in qf_operator.weights),
        system,
        linear_term,
        offset,
    )


@Operator.register_mul_handler(
    [
        (op_type, QuadraticFormOperator)
        for op_type in (
            Operator,
            LocalOperator,
            ProductOperator,
            SumOperator,
            OneBodyOperator,
            QutipOperator,
        )
    ]
)
def _(op1: Operator, op2: QuadraticFormOperator):
    return op1 * op2.as_sum_of_products()


@Operator.register_mul_handler(
    [
        (QuadraticFormOperator, op_type)
        for op_type in (
            Operator,
            LocalOperator,
            ProductOperator,
            SumOperator,
            OneBodyOperator,
            QutipOperator,
        )
    ]
)
def _(op_1: QuadraticFormOperator, op_2: Operator):
    return op_1.as_sum_of_products() * op_2
