"""
Arithmetic handlers involving GibbsProductDensityOperator.

All handlers delegate to `.to_product_state()`, which converts a Gibbs state
into a plain ProductDensityOperator before dispatching to the product handlers.

Registered operations:
  - GibbsProductDensityOperator + BasicOperator/SumOp         ->  (convert, then add)
  - GibbsProductDensityOperator * numbers                     ->  (mul or convert)
  - GibbsProductDensityOperator * NonProductBasicOp/SumOp     ->  (convert, then mul)
  - NonProductBasicOp/SumOp * GibbsProductDensityOperator     ->  (convert, then mul)
  - GibbsProductDensityOperator * GibbsProductDensityOperator               ->  (convert both, then mul)
  - GibbsProductDensityOperator * ScalarOp/LocalOp/ProductOp  ->  (convert, then mul)
  - ScalarOp/LocalOp/ProductOp * GibbsProductDensityOperator  ->  (convert, then mul)
"""

from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.product import ProductOperator, ScalarOperator
from qalma.operators.states.arithmetic import MixtureDensityOperator
from qalma.operators.states.basic import DensityOperatorMixin
from qalma.operators.states.gibbs import GibbsProductDensityOperator
from qalma.operators.states.product import ProductDensityOperator

from ._types import (
    BASIC_OPERATOR_TYPES,
    COMPLEX_NUMERIC_TYPES,
    NON_PRODUCT_BASIC_OPERATOR_TYPES,
    REAL_NUMERIC_TYPES,
)


@Operator.register_add_handler(
    [(GibbsProductDensityOperator, type_1) for type_1 in REAL_NUMERIC_TYPES]
)
def gpdo_add_real_(x_op: GibbsProductDensityOperator, y_op: float):
    if y_op == 0:
        return x_op
    system = x_op.system
    if y_op > 0:
        return MixtureDensityOperator(
            (ProductDensityOperator({}, y_op, system), x_op), system
        )
    return x_op.to_product_state() + y_op


@Operator.register_add_handler(
    [(GibbsProductDensityOperator, type_1) for type_1 in COMPLEX_NUMERIC_TYPES]
)
def gpdo_add_complex_(x_op: GibbsProductDensityOperator, y_op: float):
    if y_op.imag == 0:
        return x_op + y_op.real
    if y_op == 0:
        return x_op
    return x_op.to_product_state() + y_op


@Operator.register_add_handler(
    [
        (
            GibbsProductDensityOperator,
            GibbsProductDensityOperator,
        ),
        (
            GibbsProductDensityOperator,
            ProductDensityOperator,
        ),
    ]
)
def _gpd_add_gpd_(x_op: GibbsProductDensityOperator, y_op: DensityOperatorMixin):
    if x_op is y_op:
        return GibbsProductDensityOperator(
            x_op.k_by_site,
            x_op.system,
            prefactor=2 * x_op.prefactor,
            normalized=True,
        )
    return MixtureDensityOperator((x_op, y_op), x_op.system * y_op.system)


@Operator.register_add_handler(
    [(GibbsProductDensityOperator, type_op) for type_op in BASIC_OPERATOR_TYPES]
)
@Operator.register_add_handler((GibbsProductDensityOperator, SumOperator))
@Operator.register_add_handler((GibbsProductDensityOperator, OneBodyOperator))
def _(x_op: GibbsProductDensityOperator, y_op: Operator):
    return x_op.to_product_state() + y_op


@Operator.register_add_handler(
    [
        (GibbsProductDensityOperator, MixtureDensityOperator),
    ]
)
def _(x_op: GibbsProductDensityOperator, y_op: MixtureDensityOperator):
    return MixtureDensityOperator((x_op,) + y_op.terms, x_op.system * y_op.system)


@Operator.register_add_handler(
    [
        (
            MixtureDensityOperator,
            GibbsProductDensityOperator,
        ),
    ]
)
def _(y_op: MixtureDensityOperator, x_op: GibbsProductDensityOperator):
    return MixtureDensityOperator((x_op,) + y_op.terms, x_op.system * y_op.system)


# ####### Multiplication ###################


# ## with numbers


@Operator.register_mul_handler(
    [(GibbsProductDensityOperator, type_op) for type_op in REAL_NUMERIC_TYPES]
)
def _(x_op: GibbsProductDensityOperator, y_op: float):
    if y_op == 0:
        return ProductDensityOperator({}, weight=0, system=x_op.system, normalized=True)
    if y_op == 1:
        return x_op
    if 0 < y_op < 1:
        return GibbsProductDensityOperator(
            x_op.k_by_site,
            x_op.system,
            prefactor=x_op.prefactor * y_op,
            normalized=True,
        )
    # Generic
    x_op_prod = x_op.to_product_state()
    return x_op_prod * y_op


@Operator.register_mul_handler(
    [(type_op, GibbsProductDensityOperator) for type_op in REAL_NUMERIC_TYPES]
)
def _(y_op: float, x_op: GibbsProductDensityOperator):
    if y_op == 0:
        return ProductDensityOperator({}, weight=0, system=x_op.system, normalized=True)
    if y_op == 1:
        return x_op
    if 0 < y_op < 1:
        return GibbsProductDensityOperator(
            x_op.k_by_site,
            x_op.system,
            prefactor=x_op.prefactor * y_op,
            normalized=True,
        )
    # Generic
    return x_op.to_product_state() * y_op


@Operator.register_mul_handler(
    [(GibbsProductDensityOperator, type_op) for type_op in COMPLEX_NUMERIC_TYPES]
)
def _(x_op: GibbsProductDensityOperator, y_op: float):
    return x_op.to_product_state() * y_op


@Operator.register_mul_handler(
    [(type_op, GibbsProductDensityOperator) for type_op in COMPLEX_NUMERIC_TYPES]
)
def _(y_op: float, x_op: GibbsProductDensityOperator):
    return x_op.to_product_state() * y_op


# With other DensityOperators:


@Operator.register_mul_handler(
    (GibbsProductDensityOperator, GibbsProductDensityOperator)
)
def _(x_op: GibbsProductDensityOperator, y_op: GibbsProductDensityOperator):
    return x_op.to_product_state() * y_op.to_product_state()


@Operator.register_mul_handler(
    [
        (GibbsProductDensityOperator, type_op)
        for type_op in (
            DensityOperatorMixin,
            ProductDensityOperator,
#            MixtureDensityOperator,
        )
    ]
)
def _(x_op: GibbsProductDensityOperator, y_op: DensityOperatorMixin):
    return x_op.to_product_state() * y_op


@Operator.register_mul_handler(
    [
        (type_op, GibbsProductDensityOperator)
        for type_op in (
            DensityOperatorMixin,
            ProductDensityOperator,
#            MixtureDensityOperator,
        )
    ]
)
def _(x_op: DensityOperatorMixin, y_op: GibbsProductDensityOperator):
    return x_op * y_op.to_product_state()


## With Basic Operators


@Operator.register_mul_handler(
    [
        (GibbsProductDensityOperator, type_op)
        for type_op in NON_PRODUCT_BASIC_OPERATOR_TYPES
    ]
)
@Operator.register_mul_handler((GibbsProductDensityOperator, SumOperator))
def _(x_op: GibbsProductDensityOperator, y_op: Operator):
    return x_op.to_product_state() * y_op


@Operator.register_mul_handler(
    [
        (type_op, GibbsProductDensityOperator)
        for type_op in NON_PRODUCT_BASIC_OPERATOR_TYPES
    ]
)
@Operator.register_mul_handler((SumOperator, GibbsProductDensityOperator))
def _(x_op: Operator, y_op: GibbsProductDensityOperator):
    return x_op * y_op.to_product_state()


@Operator.register_mul_handler((GibbsProductDensityOperator, ScalarOperator))
@Operator.register_mul_handler((GibbsProductDensityOperator, LocalOperator))
@Operator.register_mul_handler((GibbsProductDensityOperator, ProductOperator))
def _(x_op: GibbsProductDensityOperator, y_op: Operator):
    return x_op.to_product_state() * y_op


@Operator.register_mul_handler((ScalarOperator, GibbsProductDensityOperator))
@Operator.register_mul_handler((LocalOperator, GibbsProductDensityOperator))
@Operator.register_mul_handler((ProductOperator, GibbsProductDensityOperator))
def _(x_op: Operator, y_op: GibbsProductDensityOperator):
    return x_op * y_op.to_product_state()
