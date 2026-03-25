"""
Arithmetic handlers involving non-product density operators:
DensityOperatorMixin, QutipDensityOperator, GibbsDensityOperator.

The common strategy for all handlers here is to unwrap the density operator
into a plain operator via `.to_qutip_operator()`, then delegate to the
corresponding operator arithmetic.

Registered operations:
  - any density state      + any density state       ->  MixtureDensityOperator
  - BasicOperator/SumOp    + NonProductDensity        ->  (unwrap, then add)
  - BasicOperator/SumOp    * NonProductDensity        ->  (unwrap, then mul)
  - NonProductDensity      * BasicOperator/SumOp      ->  (unwrap, then mul)
  - NonProductDensity      + SumOp                    ->  (unwrap, then add)
"""

from qalma.operators.arithmetic import SumOperator
from qalma.operators.basic import Operator
from qalma.operators.qutip import QutipOperator
from qalma.operators.states.arithmetic import MixtureDensityOperator
from qalma.operators.states.basic import (
    DensityOperatorMixin,
)
from qalma.operators.states.qutip import QutipDensityOperator

from ._types import (
    BASIC_OPERATOR_TYPES,
    DENSITY_OPERATOR_BASIC_TYPES,
    NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES,
)


def _unwrap(y_op: DensityOperatorMixin) -> Operator:
    """Convert a density operator to a plain Operator for arithmetic."""
    y_op_basic: Operator = y_op.to_qutip_operator()
    if isinstance(y_op_basic, QutipDensityOperator):
        y_op_basic = QutipOperator(
            y_op_basic.operator,
            y_op_basic.system,
            y_op_basic.site_names,
            prefactor=1,
        )
    return y_op_basic


#@Operator.register_add_handler(
#    [
#        (type_1, type_2)
#        for type_1 in DENSITY_OPERATOR_BASIC_TYPES
#        for type_2 in DENSITY_OPERATOR_BASIC_TYPES
#    ]
#)
def add_generic_states_(x_op, y_op):
    system = x_op.system * y_op.system
    return MixtureDensityOperator((x_op, y_op), system)


@Operator.register_add_handler(
    [
        (type_1, type_2)
        for type_1 in BASIC_OPERATOR_TYPES
        for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES
    ]
)
@Operator.register_add_handler(
    [(SumOperator, type_2) for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES]
)
def sum_add_npdo_(x_op: Operator, y_op: DensityOperatorMixin):
    return x_op + _unwrap(y_op)


#@Operator.register_mul_handler(
#    [
#        (type_1, type_2)
#        for type_1 in BASIC_OPERATOR_TYPES
#        for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES
#    ]
#)
#@Operator.register_mul_handler(
#    [(SumOperator, type_2) for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES]
#)
def _(x_op: Operator, y_op: DensityOperatorMixin):
    return x_op * _unwrap(y_op)


#@Operator.register_mul_handler(
#    [
#        (type_2, type_1)
#        for type_1 in BASIC_OPERATOR_TYPES
#        for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES
#    ]
#)
#@Operator.register_mul_handler(
#    [(type_2, SumOperator) for type_2 in NON_PRODUCT_DENSITY_OPERATOR_BASIC_TYPES]
#)
def _(y_op: DensityOperatorMixin, x_op: Operator):
    return _unwrap(y_op) * x_op
