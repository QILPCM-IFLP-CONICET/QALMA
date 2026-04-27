"""
Arithmetic handlers involving MixtureDensityOperator.

Registered operations:
  - MixtureDensityOperator + any density state  ->  MixtureDensityOperator
  - BasicOperator          * MixtureDensityOperator  ->  SumOperator
  - SumOperator            * MixtureDensityOperator  ->  SumOperator
"""

from qalma.operators.arithmetic import SumOperator
from qalma.operators.basic import Operator
from qalma.operators.product import ScalarOperator
from qalma.operators.states.arithmetic import MixtureDensityOperator
from qalma.operators.states.basic import DensityOperatorMixin
from qalma.operators.states.product import ProductDensityOperator

from ._types import (
    ANY_OPERATOR_TYPES,
    BASIC_OPERATOR_TYPES,
    COMPLEX_NUMERIC_TYPES,
    DENSITY_OPERATOR_BASIC_TYPES,
    REAL_NUMERIC_TYPES,
    SUM_TYPES,
)
from ._wrappers import _wrapper_sum as _mixture_to_sum

# Numbers


@Operator.register_add_handler(
    [(MixtureDensityOperator, type_num) for type_num in REAL_NUMERIC_TYPES]
)
def add_mixture_real_(x_op: MixtureDensityOperator, y_op: float):
    """
    Add a mixture operator with a real number.
    If the number is non-negative, treat as a mixture with a maximally mixed state.
    Otherwise, convert the mixture to a regular ``SumOperator``
    and add it as an scalar.
    """
    if y_op == 0:
        return x_op
    if y_op > 0:
        return x_op + ProductDensityOperator({}, weight=y_op, system=x_op.system)
    return _mixture_to_sum(x_op) + y_op


@Operator.register_add_handler(
    [(type_num, MixtureDensityOperator) for type_num in REAL_NUMERIC_TYPES]
)
def add_real_mixture_(y_op: float, x_op: MixtureDensityOperator):
    """
    Add a mixture operator with a real number.
    If the number is non-negative, treat as a mixture with a maximally mixed state.
    Otherwise, convert the mixture to a regular ``SumOperator``
    and add it as an scalar.
    """
    if y_op == 0:
        return x_op
    if y_op > 0:
        return x_op + ProductDensityOperator({}, weight=y_op, system=x_op.system)
    return _mixture_to_sum(x_op) + y_op


@Operator.register_add_handler(
    [(MixtureDensityOperator, type_num) for type_num in COMPLEX_NUMERIC_TYPES]
)
def add_mixture_complex_(x_op: MixtureDensityOperator, y_op: complex):
    """
    Add a mixture operator with a complex number.
    Convert the mixture to a regular ``SumOperator``
    and add it as an scalar.
    """
    if y_op.imag == 0:
        return x_op + y_op.real
    return _mixture_to_sum(x_op) + y_op


@Operator.register_add_handler(
    [(type_num, MixtureDensityOperator) for type_num in COMPLEX_NUMERIC_TYPES]
)
def add_complex_mixture_(y_op: complex, x_op: MixtureDensityOperator):
    """
    Add a mixture operator with a complex number.
    Convert the mixture to a regular ``SumOperator``
    and add it as an scalar.
    """
    if y_op.imag == 0:
        return x_op + y_op.real
    return _mixture_to_sum(x_op) + y_op


@Operator.register_add_handler(
    (
        MixtureDensityOperator,
        MixtureDensityOperator,
    )
)
def _(x_op: MixtureDensityOperator, y_op: MixtureDensityOperator):
    """
    Add two mixture operators, and produce a new mixture.
    """
    return MixtureDensityOperator(x_op.terms + y_op.terms, x_op.system * y_op.system)


@Operator.register_add_handler(
    [
        (
            MixtureDensityOperator,
            type_op,
        )
        for type_op in BASIC_OPERATOR_TYPES
    ]
)
@Operator.register_add_handler(
    [
        (
            MixtureDensityOperator,
            type_op,
        )
        for type_op in SUM_TYPES
    ]
)
def _(x_op: MixtureDensityOperator, y_op: Operator):
    """
    Convert the mixture into a SumOperator, and then add
    the other operator
    """
    # HACK to convert terms from a MixtureDensityOperator
    # to terms in a sum.
    #
    result = _mixture_to_sum(x_op) + y_op
    return result


@Operator.register_add_handler(
    [
        (
            type_op,
            MixtureDensityOperator,
        )
        for type_op in BASIC_OPERATOR_TYPES
    ]
)
@Operator.register_add_handler(
    [
        (
            type_op,
            MixtureDensityOperator,
        )
        for type_op in SUM_TYPES
    ]
)
def _(x_op: Operator, y_op: MixtureDensityOperator):
    """
    Convert the mixture into a SumOperator, and then add
    the other operator
    """
    # HACK to convert terms from a MixtureDensityOperator
    # to terms in a sum.
    #
    return _mixture_to_sum(y_op) + x_op


@Operator.register_add_handler(
    [
        (MixtureDensityOperator, state_type)
        for state_type in DENSITY_OPERATOR_BASIC_TYPES
    ]
)
def _(x_op: MixtureDensityOperator, y_op: DensityOperatorMixin):
    """
    Add a mixture with another density operator.
    """
    terms = x_op.terms + (y_op,)
    # If there is just one term, return it:
    if len(terms) == 1:
        return terms[0]

    # For empty terms, return 0
    system = x_op.system or y_op.system
    if len(terms) == 0:
        return ScalarOperator(0.0, system)
    # General case
    return MixtureDensityOperator(terms, system)


##### MUL


@Operator.register_mul_handler(
    [(MixtureDensityOperator, type_num) for type_num in REAL_NUMERIC_TYPES]
)
def _(x_op: MixtureDensityOperator, y_op: float):
    """
    Multiply a ``MixtureDensityOperator`` with a real number. If the number is non-negative,
    produce a new mixture with all the weights multiplied by the real factor.
    If negative, first convert the ``MixtureDensityOperator`` to a regular operator,
    and then evaluate the product.
    """
    if y_op == 0:
        return ProductDensityOperator({}, weight=y_op, system=x_op.system)
    if y_op > 0:
        return MixtureDensityOperator(
            tuple(term * y_op for term in x_op.terms), x_op.system
        )
    return SumOperator(
        tuple((-term) * (-y_op) for term in x_op.terms), x_op.system, isherm=True
    )


@Operator.register_mul_handler(
    [(type_num, MixtureDensityOperator) for type_num in REAL_NUMERIC_TYPES]
)
def _(y_op: float, x_op: MixtureDensityOperator):
    """
    Multiply a ``MixtureDensityOperator`` with a real number. If the number is non-negative,
    produce a new mixture with all the weights multiplied by the real factor.
    If negative, first convert the ``MixtureDensityOperator`` to a regular operator,
    and then evaluate the product.
    """
    if y_op == 0:
        return ProductDensityOperator({}, weight=y_op, system=x_op.system)
    if y_op > 0:
        return MixtureDensityOperator(
            tuple(term * y_op for term in x_op.terms), x_op.system
        )
    return SumOperator(
        tuple((-term) * (-y_op) for term in x_op.terms), x_op.system, isherm=True
    )


@Operator.register_mul_handler(
    [(MixtureDensityOperator, type_num) for type_num in COMPLEX_NUMERIC_TYPES]
)
def _(x_op: MixtureDensityOperator, y_op: complex):
    """
    Multiply a ``MixtureDensityOperator`` with a complex number.
    First convert the ``MixtureDensityOperator`` to a regular operator,
    and then evaluate the product.
    """
    if y_op.imag == 0:
        return x_op * y_op.real
    return _mixture_to_sum(x_op) * y_op


@Operator.register_mul_handler(
    [(type_num, MixtureDensityOperator) for type_num in COMPLEX_NUMERIC_TYPES]
)
def _(x_op: complex, y_op: MixtureDensityOperator):
    """
    Multiply a ``MixtureDensityOperator`` with a complex number.
    First convert the ``MixtureDensityOperator`` to a regular operator,
    and then evaluate the product.
    """
    if x_op.imag == 0:
        return y_op * x_op.real
    return _mixture_to_sum(y_op) * x_op


@Operator.register_mul_handler(
    (
        MixtureDensityOperator,
        MixtureDensityOperator,
    )
)
def _(x_op: MixtureDensityOperator, y_op: MixtureDensityOperator):
    """
    Multiply two ``MixtureDensityOperators``. Convert them to
    regular ``SumOperators``, and return their product.
    """
    # multiply by -1 convert a Mixture into a Sum operator
    if x_op is y_op:
        x_op_sum = _mixture_to_sum(x_op)
        return x_op_sum * x_op_sum

    return _mixture_to_sum(x_op) * _mixture_to_sum(y_op)


@Operator.register_mul_handler(
    [
        (type_op, MixtureDensityOperator)
        for type_op in ANY_OPERATOR_TYPES
        if type_op is not MixtureDensityOperator
    ]
)
def any_times_mixture_(y_op: Operator, x_op: MixtureDensityOperator):
    """
    Multiply a ``MixtureDensityOperators`` with any other operator.
    Convert it into a regular ``SumOperators``, and then return
    its product  product with the other operand.
    """
    return y_op * _mixture_to_sum(x_op)


@Operator.register_mul_handler(
    [
        (
            MixtureDensityOperator,
            type_op,
        )
        for type_op in ANY_OPERATOR_TYPES
        if type_op is not MixtureDensityOperator
    ]
)
def mixture_times_any_(x_op: MixtureDensityOperator, y_op: Operator):
    """
    Multiply a ``MixtureDensityOperators`` with any other operator.
    Convert it into a regular ``SumOperators``, and then return
    its product  product with the other operand.
    """
    return _mixture_to_sum(x_op) * y_op
