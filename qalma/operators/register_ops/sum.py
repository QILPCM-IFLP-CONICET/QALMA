"""Arithmetic bindings for SumOperator.

Covers:
  - SumOperator +/* Number  (and reversed)
  - SumOperator +/* ScalarOperator  (and reversed)
  - SumOperator +/* LocalOperator  (and reversed)
  - SumOperator + any Operator
  - SumOperator + SumOperator
  - SumOperator * SumOperator  (and OneBodyOperator cross-products)
  - SumOperator * Operator / ProductOperator / QutipOperator  (and reversed)
"""

from typing import Dict

from qalma.operators.arithmetic import (
    OneBodyOperator,
    SumOperator,
    iterable_to_operator,
)
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.product import ProductOperator, ScalarOperator
from qalma.operators.quadratic import QuadraticFormOperator
from qalma.operators.qutip import QutipOperator

from ._common import NUMERIC_TYPES

SUM_TYPES = (SumOperator, OneBodyOperator)


@Operator.register_add_handler([(SumOperator, t) for t in NUMERIC_TYPES])
def _(x_op: SumOperator, y_value: complex):
    return x_op + ScalarOperator(y_value, x_op.system)


@Operator.register_mul_handler([(SumOperator, t) for t in NUMERIC_TYPES])
def _(x_op: SumOperator, y_value: complex):
    if y_value == 0:
        return ScalarOperator(0, x_op.system)
    terms = tuple(term * y_value for term in x_op.terms)
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
    return SumOperator(
        terms, x_op.system, isherm=isherm or None, simplified=x_op._simplified
    )


@Operator.register_mul_handler([(t, SumOperator) for t in NUMERIC_TYPES])
def _(y_value: complex, x_op: SumOperator):
    if y_value == 0:
        return ScalarOperator(0, x_op.system)
    terms = tuple(term * y_value for term in x_op.terms)
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
    return SumOperator(
        terms, x_op.system, isherm=isherm or None, simplified=x_op._simplified
    )


@Operator.register_mul_handler((SumOperator, ScalarOperator))
def _(x_op: SumOperator, y_op: ScalarOperator):
    system = x_op.system or y_op.system
    y_value = y_op.prefactor
    if y_value == 0:
        return ScalarOperator(0, system)
    terms = tuple(term * y_value for term in x_op.terms)
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
    return iterable_to_operator(terms, system, isherm=isherm or None)


@Operator.register_mul_handler((ScalarOperator, SumOperator))
def _(y_op: ScalarOperator, x_op: SumOperator):
    system = x_op.system or y_op.system
    y_value = y_op.prefactor
    if y_value == 0:
        return ScalarOperator(0, system)
    terms = tuple(term * y_value for term in x_op.terms)
    isherm = x_op._isherm and (not isinstance(y_value, complex) or y_value.imag == 0)
    return iterable_to_operator(
        terms, system, isherm=isherm or None, simplified=x_op._simplified
    )


@Operator.register_mul_handler(
    [(SumOperator, LocalOperator), (OneBodyOperator, LocalOperator)]
)
def _(x_op: SumOperator, y_op: LocalOperator):
    system = x_op.system.union(y_op.system)
    terms = tuple(term * y_op for term in x_op.terms if bool(term))
    isherm = x_op._isherm and y_op.isherm
    return iterable_to_operator(terms, system, isherm=isherm or None)


@Operator.register_mul_handler(
    [(LocalOperator, SumOperator), (LocalOperator, OneBodyOperator)]
)
def _(y_op: LocalOperator, x_op: SumOperator):
    system = x_op.system.union(y_op.system)
    terms = tuple(y_op * term for term in x_op.terms if bool(term))
    return iterable_to_operator(terms, system)


@Operator.register_add_handler(
    [
        (SumOperator, op_type)
        for op_type in (
            Operator,
            ScalarOperator,
            LocalOperator,
            ProductOperator,
            QutipOperator,
            OneBodyOperator,
            QuadraticFormOperator,
        )
    ]
)
def _(x_op: SumOperator, y_op: Operator):
    system = x_op.system.union(y_op.system)
    terms = x_op.terms + (y_op,)
    if len(terms) == 1:
        return terms[0]
    isherm = x_op._isherm and y_op.isherm
    return iterable_to_operator(terms, system, isherm=isherm or None)


@Operator.register_add_handler((SumOperator, SumOperator))
def _(x_op: SumOperator, y_op: SumOperator):
    system = x_op.system.union(y_op.system)
    terms = x_op.terms + y_op.terms
    isherm = (x_op._isherm and y_op._isherm) or None
    return iterable_to_operator(terms, system, isherm=isherm)


@Operator.register_mul_handler(tuple((s1, s2) for s1 in SUM_TYPES for s2 in SUM_TYPES))
def _(x_op: SumOperator, y_op: SumOperator):
    isherm = (x_op is y_op and x_op._isherm) or None
    system = x_op.system.union(y_op.system)
    block_terms: Dict[frozenset, Operator] = {}
    for x_term in x_op.flat().terms:
        if x_term.is_zero:
            continue
        for y_term in y_op.flat().terms:
            if y_term.is_zero:
                continue
            xy_term = x_term * y_term
            xy_acts_over = xy_term.acts_over()
            if xy_acts_over in block_terms:
                xy_term = block_terms[xy_acts_over] + xy_term
            block_terms[xy_acts_over] = xy_term
    terms = tuple(block_terms.values())
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    if all(acts_over and len(acts_over) < 2 for acts_over in block_terms):
        return OneBodyOperator(terms, system, False, isherm=isherm)
    return SumOperator(terms, system, isherm=isherm)


@Operator.register_mul_handler(
    [
        (sum_type, op_type)
        for sum_type in SUM_TYPES
        for op_type in (Operator, ProductOperator, QutipOperator)
    ]
)
def _(x_op: SumOperator, y_op: Operator):
    system = x_op.system.union(y_op.system)
    if y_op.is_zero:
        return ScalarOperator(0.0, system)
    terms = tuple(factor_x * y_op for factor_x in x_op.terms)
    return iterable_to_operator(terms, system)


@Operator.register_mul_handler(
    [
        (op_type, sum_type)
        for sum_type in SUM_TYPES
        for op_type in (Operator, ProductOperator, QutipOperator)
    ]
)
def _(y_op: Operator, x_op: SumOperator):
    system = x_op.system.union(y_op.system)
    if y_op.is_zero:
        return ScalarOperator(0.0, system)
    terms = tuple(y_op * factor_x for factor_x in x_op.terms)
    return iterable_to_operator(terms, system)
