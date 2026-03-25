"""
Arithmetic handlers involving non-product density operators:
QutipDensityOperator.

The common strategy for all handlers here is to unwrap the density operator
into a plain operator via `.to_qutip_operator()`, then delegate to the
corresponding operator arithmetic.

Registered operations:
  - QutipDensityOperator   + QutipDensityOperator   -> (mixture)
  - QutipDensityOperator   * real                   -> (QutipDensityOperator)
  - QutipDensityOperator   * complex                -> (QutipOperator)
  - QutipDensityOperator   * QutipDensityOperator   -> (cast, then mul)
  - QutipDensityOperator   * QutipOperator          -> (cast, then mul)
  - QutipOperator          * QutipDensityOperator   -> (cast, then mul)
  - ProductDensity         * QutipDensityOperator   -> (cast, then mul)
"""

from qalma.operators.basic import Operator
from qalma.operators.qutip import QutipOperator
from qalma.operators.states.arithmetic import MixtureDensityOperator
from qalma.operators.states.product import ProductDensityOperator
from qalma.operators.states.qutip import QutipDensityOperator

from ._types import (
    ANY_OPERATOR_TYPES,
    COMPLEX_NUMERIC_TYPES,
    REAL_NUMERIC_TYPES,
)


def _wrapper(qutip_density, prefactor=1) -> QutipOperator:
    """
    Discard the prefactor and build a new qutip operator
    """
    result = QutipOperator(
        qutip_density.operator,
        qutip_density.system,
        qutip_density.site_names,
        prefactor=prefactor,
    )
    return result


@Operator.register_add_handler(
    [
        (
            QutipDensityOperator,
            type_num,
        )
        for type_num in REAL_NUMERIC_TYPES
    ]
)
def _(x_op: QutipDensityOperator, y_op: float):
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
            QutipDensityOperator,
        )
        for type_num in REAL_NUMERIC_TYPES
    ]
)
def _(
    y_op: float,
    x_op: QutipDensityOperator,
):
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
            QutipDensityOperator,
            type_num,
        )
        for type_num in COMPLEX_NUMERIC_TYPES
    ]
)
def _(x_op: QutipDensityOperator, y_op: complex):
    if y_op.imag == 0:
        return x_op + y_op.real
    return _wrapper(x_op) + y_op


@Operator.register_add_handler(
    [
        (
            type_num,
            QutipDensityOperator,
        )
        for type_num in COMPLEX_NUMERIC_TYPES
    ]
)
def _(
    y_op: complex,
    x_op: QutipDensityOperator,
):
    if y_op.imag == 0:
        return x_op + y_op.real
    return _wrapper(x_op) + y_op


@Operator.register_add_handler(
    (
        QutipDensityOperator,
        QutipDensityOperator,
    )
)
def _(x_op: QutipDensityOperator, y_op: QutipDensityOperator):
    return MixtureDensityOperator(
        (
            x_op,
            y_op,
        ),
        x_op.system * y_op.system,
    )


#### Multiply by numbers


@Operator.register_mul_handler(
    [
        (
            num_type,
            QutipDensityOperator,
        )
        for num_type in REAL_NUMERIC_TYPES
    ]
)
def _(x_op: float, y_op: QutipDensityOperator):
    if x_op == 0:
        return ProductDensityOperator({}, system=y_op.system, weight=0.0)
    if x_op > 0:
        return QutipDensityOperator(
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
            QutipDensityOperator,
            num_type,
        )
        for num_type in REAL_NUMERIC_TYPES
    ]
)
def _(x_op: QutipDensityOperator, y_op: float):
    if y_op == 0:
        return ProductDensityOperator({}, system=x_op.system, weight=0.0)
    if y_op > 0:
        return QutipDensityOperator(
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
            QutipDensityOperator,
        )
        for num_type in COMPLEX_NUMERIC_TYPES
    ]
)
def _(x_op: complex, y_op: QutipDensityOperator):
    if x_op.imag == 0:
        return y_op * x_op.real

    return _wrapper(y_op, x_op)


@Operator.register_mul_handler(
    [
        (
            QutipDensityOperator,
            num_type,
        )
        for num_type in COMPLEX_NUMERIC_TYPES
    ]
)
def _(x_op: QutipDensityOperator, y_op: complex):
    if y_op.imag == 0:
        return x_op * y_op.real

    return _wrapper(x_op, y_op)


# QutipDensityOperator * QutipOperator


@Operator.register_mul_handler(
    (
        QutipDensityOperator,
        QutipDensityOperator,
    )
)
def _(x_op: QutipDensityOperator, y_op: QutipDensityOperator):
    if x_op is y_op:
        x_op_qutip = _wrapper(x_op)
        return x_op_qutip * x_op_qutip
    return _wrapper(x_op) * _wrapper(y_op)


@Operator.register_mul_handler(
    [
        (
            QutipDensityOperator,
            type_op,
        )
        for type_op in ANY_OPERATOR_TYPES
        if type_op is not QutipDensityOperator
    ]
)
def _(x_op: QutipDensityOperator, y_op: Operator):
    return _wrapper(x_op) * y_op


@Operator.register_mul_handler(
    [
        (
            type_op,
            QutipDensityOperator,
        )
        for type_op in ANY_OPERATOR_TYPES
        if type_op is not QutipDensityOperator
    ]
)
def _(
    x_op: Operator,
    y_op: QutipDensityOperator,
):
    return x_op * _wrapper(y_op)
