"""Arithmetic handlers for BlockDensityOperators.

Arithmetic handlers involving non-product density operators:
BlockDensityOperator.

The common strategy for all handlers here is to unwrap the density operator
into a plain operator via `.to_qutip_operator()`, then delegate to the
corresponding operator arithmetic.

Registered operations:
  - BlockDensityOperator   + BlockDensityOperator   -> (mixture)
  - BlockDensityOperator   * real                   -> (BlockDensityOperator)
  - BlockDensityOperator   * complex                -> (BlockOperator)
  - BlockDensityOperator   * BlockDensityOperator   -> (cast, then mul)
  - BlockDensityOperator   * BlockOperator          -> (cast, then mul)
  - BlockOperator          * BlockDensityOperator   -> (cast, then mul)
  - ProductDensity         * BlockDensityOperator   -> (cast, then mul)
"""

from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import Operator
from qalma.operators.states.arithmetic import MixtureDensityOperator
from qalma.operators.states.product import ProductDensityOperator
from qalma.operators.states.qutip import BlockDensityOperator

from ._types import (
    ANY_OPERATOR_TYPES,
    BASIC_OPERATOR_TYPES,
    COMPLEX_NUMERIC_TYPES,
    REAL_NUMERIC_TYPES,
)
from ._wrappers import _wrapper_qutip as _wrapper


@Operator.register_add_handler(
    [
        (
            BlockDensityOperator,
            type_num,
        )
        for type_num in REAL_NUMERIC_TYPES
    ]
)
def _(x_op: BlockDensityOperator, y_op: float):
    """Add a qutip densityoperator with a real number."""
    if y_op == 0:
        return x_op
    if y_op > 0:
        y_op_state = ProductDensityOperator(
            {},
            system=x_op.system,
            weight=y_op,
        )
        return MixtureDensityOperator(
            (
                x_op,
                y_op_state,
            ),
            x_op.system,
        )
    return _wrapper(x_op) + y_op


@Operator.register_add_handler(
    [
        (
            type_num,
            BlockDensityOperator,
        )
        for type_num in REAL_NUMERIC_TYPES
    ]
)
def _(
    y_op: float,
    x_op: BlockDensityOperator,
):
    """Add a qutip densityoperator with a real number."""
    if y_op == 0:
        return x_op
    if y_op > 0:
        y_op_state = ProductDensityOperator(
            {},
            system=x_op.system,
            weight=y_op,
        )
        return MixtureDensityOperator(
            (
                x_op,
                y_op_state,
            ),
            x_op.system,
        )
    return _wrapper(x_op) + y_op


@Operator.register_add_handler(
    [
        (
            BlockDensityOperator,
            type_num,
        )
        for type_num in COMPLEX_NUMERIC_TYPES
    ]
)
def _(x_op: BlockDensityOperator, y_op: complex):
    """Add a qutip densityoperator with a complex number."""
    if y_op.imag == 0:
        return x_op + y_op.real
    return _wrapper(x_op) + y_op


@Operator.register_add_handler(
    [
        (
            type_num,
            BlockDensityOperator,
        )
        for type_num in COMPLEX_NUMERIC_TYPES
    ]
)
def _(
    y_op: complex,
    x_op: BlockDensityOperator,
):
    """Add a qutip densityoperator with a complex number."""
    if y_op.imag == 0:
        return x_op + y_op.real
    return _wrapper(x_op) + y_op


@Operator.register_add_handler(
    (
        BlockDensityOperator,
        BlockDensityOperator,
    )
)
def _(x_op: BlockDensityOperator, y_op: BlockDensityOperator):
    """Add a qutip densityoperator with a mixture density operator."""
    return MixtureDensityOperator(
        (
            x_op,
            y_op,
        ),
        x_op.system * y_op.system,
    )


@Operator.register_add_handler(
    [(BlockDensityOperator, type_op) for type_op in BASIC_OPERATOR_TYPES]
)
@Operator.register_add_handler((BlockDensityOperator, SumOperator))
@Operator.register_add_handler((BlockDensityOperator, OneBodyOperator))
def add_qdo_sum(x_op: BlockDensityOperator, y_op: Operator):
    """Add a qutip densityoperator with a general density operator."""
    return _wrapper(x_op) + y_op


#### Multiply by numbers


@Operator.register_mul_handler(
    [
        (
            num_type,
            BlockDensityOperator,
        )
        for num_type in REAL_NUMERIC_TYPES
    ]
)
def _(x_op: float, y_op: BlockDensityOperator):
    """Multiply a qutip densityoperator with a real number."""
    if x_op == 0:
        return ProductDensityOperator({}, system=y_op.system, weight=0.0)
    if x_op > 0:
        return BlockDensityOperator(
            y_op.operator,
            y_op.system,
            y_op.site_names,
            y_op.prefactor * x_op,
            normalized=y_op._normalized,
        )
    return _wrapper(y_op, x_op)


@Operator.register_mul_handler(
    [
        (
            BlockDensityOperator,
            num_type,
        )
        for num_type in REAL_NUMERIC_TYPES
    ]
)
def _(x_op: BlockDensityOperator, y_op: float):
    """Multiply a qutip densityoperator with a real number."""
    if y_op == 0:
        return ProductDensityOperator({}, system=x_op.system, weight=0.0)
    if y_op > 0:
        return BlockDensityOperator(
            x_op.operator,
            x_op.system,
            x_op.site_names,
            x_op.prefactor * y_op,
            normalized=x_op._normalized,
        )
    return _wrapper(x_op, y_op)


@Operator.register_mul_handler(
    [
        (
            num_type,
            BlockDensityOperator,
        )
        for num_type in COMPLEX_NUMERIC_TYPES
    ]
)
def _(x_op: complex, y_op: BlockDensityOperator):
    """Multiply a qutip densityoperator with a complex number."""
    if x_op.imag == 0:
        return y_op * x_op.real

    return _wrapper(y_op, x_op)


@Operator.register_mul_handler(
    [
        (
            BlockDensityOperator,
            num_type,
        )
        for num_type in COMPLEX_NUMERIC_TYPES
    ]
)
def _(x_op: BlockDensityOperator, y_op: complex):
    """Multiply a qutip densityoperator with a complex number."""
    if y_op.imag == 0:
        return x_op * y_op.real

    return _wrapper(x_op, y_op)


# BlockDensityOperator * BlockOperator


@Operator.register_mul_handler(
    (
        BlockDensityOperator,
        BlockDensityOperator,
    )
)
def _(x_op: BlockDensityOperator, y_op: BlockDensityOperator):
    """Multiply two qutip densityoperators."""
    if x_op is y_op:
        x_op_qutip = _wrapper(x_op)
        return x_op_qutip * x_op_qutip
    return _wrapper(x_op) * _wrapper(y_op)


@Operator.register_mul_handler(
    [
        (
            BlockDensityOperator,
            type_op,
        )
        for type_op in ANY_OPERATOR_TYPES
        if type_op is not BlockDensityOperator
    ]
)
def _(x_op: BlockDensityOperator, y_op: Operator):
    """Multiply a qutip densityoperator with general operators."""
    return _wrapper(x_op) * y_op


@Operator.register_mul_handler(
    [
        (
            type_op,
            BlockDensityOperator,
        )
        for type_op in ANY_OPERATOR_TYPES
        if type_op is not BlockDensityOperator
    ]
)
def _(
    x_op: Operator,
    y_op: BlockDensityOperator,
):
    """Multiply a qutip densityoperator with general operators."""
    return x_op * _wrapper(y_op)
