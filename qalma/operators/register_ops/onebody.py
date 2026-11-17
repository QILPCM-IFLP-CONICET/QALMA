"""Arithmetic bindings for OneBodyOperator.

Covers:
  - OneBodyOperator + OneBodyOperator
  - OneBodyOperator + Number
  - OneBodyOperator + ScalarOperator
  - OneBodyOperator + LocalOperator
  - OneBodyOperator * Number  (and reversed)
  - OneBodyOperator * ScalarOperator  (and reversed)
"""

from qalma.operators.arithmetic import OneBodyOperator
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.product import ScalarOperator

from ._common import NUMERIC_TYPES


@Operator.register_add_handler((OneBodyOperator, OneBodyOperator))
def _(x_op: OneBodyOperator, y_op: OneBodyOperator):
    system = x_op.system or y_op.system
    terms = x_op.terms + y_op.terms
    if len(terms) == 0:
        return ScalarOperator(0, system)
    if len(terms) == 1:
        return terms[0]
    isherm = (x_op._isherm and y_op._isherm) or None
    return OneBodyOperator(terms, system, isherm=isherm)


@Operator.register_add_handler([(OneBodyOperator, t) for t in NUMERIC_TYPES])
def _(x_op: OneBodyOperator, y_value: complex):
    system = x_op.system
    terms = x_op.terms + (ScalarOperator(y_value, system),)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_add_handler((OneBodyOperator, ScalarOperator))
def _(x_op: OneBodyOperator, y_op: ScalarOperator):
    system = x_op.system.union(y_op.system)
    terms = x_op.terms + (y_op,)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_add_handler((OneBodyOperator, LocalOperator))
def _(x_op: OneBodyOperator, y_op: LocalOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    terms = x_op.terms + (y_op,)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, system)


@Operator.register_mul_handler([(OneBodyOperator, t) for t in NUMERIC_TYPES])
def _(x_op: OneBodyOperator, y_value: complex):
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, x_op.system)


@Operator.register_mul_handler([(t, OneBodyOperator) for t in NUMERIC_TYPES])
def _(y_value: complex, x_op: OneBodyOperator):
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, x_op.system)


@Operator.register_mul_handler((OneBodyOperator, ScalarOperator))
def _(x_op: OneBodyOperator, y_op: ScalarOperator):
    y_value = y_op.prefactor
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, x_op.system)


@Operator.register_mul_handler((ScalarOperator, OneBodyOperator))
def _(y_op: ScalarOperator, x_op: OneBodyOperator):
    y_value = y_op.prefactor
    terms = tuple(term * y_value for term in x_op.terms)
    if len(terms) == 1:
        return terms[0]
    return OneBodyOperator(terms, x_op.system)
