"""
Arithmetic handlers involving GibbsDensityOperator.

All handlers delegate to `.to_qutip_operator()`, which converts a Gibbs state
into a plain ProductDensityOperator before dispatching to the product handlers.

Registered operations:
  - Gibbs + numbers                     ->  (mul or convert)
  - Gibbs + BasicOperator/SumOp         ->  (convert, then add)
  - Gibbs * numbers                     ->  (mul or convert)
  - Gibbs * NonProductBasicOp/SumOp     ->  (convert, then mul)
  - NonProductBasicOp/SumOp * Gibbs     ->  (convert, then mul)
  - GibbsProduct * Gibbs               ->  (convert both, then mul)
  - Gibbs * GibbsProduct               ->  (convert both, then mul)
  - Gibbs * Gibbs               ->  (convert both, then mul)
  - Gibbs * ScalarOp/LocalOp/ProductOp  ->  (convert, then mul)
  - ScalarOp/LocalOp/ProductOp * Gibbs  ->  (convert, then mul)
"""

from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.product import ProductOperator, ScalarOperator
from qalma.operators.states.arithmetic import MixtureDensityOperator
from qalma.operators.states.basic import DensityOperatorMixin
from qalma.operators.states.gibbs import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
)
from qalma.operators.states.product import ProductDensityOperator

from ._types import (
    BASIC_OPERATOR_TYPES,
    COMPLEX_NUMERIC_TYPES,
    NON_PRODUCT_BASIC_OPERATOR_TYPES,
    REAL_NUMERIC_TYPES,
)


@Operator.register_add_handler(
    [(GibbsDensityOperator, type_1) for type_1 in REAL_NUMERIC_TYPES]
)
def gdo_add_real_(x_op: GibbsDensityOperator, y_op: float):
    if y_op == 0:
        return x_op
    system = x_op.system
    if y_op > 0:
        return MixtureDensityOperator(
            (ProductDensityOperator({}, y_op, system), x_op), system
        )
    return x_op.to_qutip_operator() + y_op


@Operator.register_add_handler(
    [(GibbsDensityOperator, type_1) for type_1 in COMPLEX_NUMERIC_TYPES]
)
def gdo_add_complex_(x_op: GibbsDensityOperator, y_op: float):
    if y_op.imag == 0:
        return x_op + y_op.real
    if y_op == 0:
        return x_op
    return x_op.to_qutip_operator() + y_op


@Operator.register_add_handler(
    [
        (GibbsDensityOperator, DensityOperatorMixin),
        (GibbsDensityOperator, ProductDensityOperator),
        (GibbsDensityOperator, GibbsDensityOperator),
        (GibbsDensityOperator, GibbsProductDensityOperator),
    ]
)
def _(x_op: GibbsDensityOperator, y_op: DensityOperatorMixin):
    if x_op is y_op:
        return GibbsDensityOperator(
            x_op.k,
            x_op.system,
            prefactor=2 * x_op.prefactor,
            normalized=True,
            meanfield=x_op._meanfield,
            symmetry_projections=x_op.symmetry_projections,
        )

    return MixtureDensityOperator((x_op, y_op), x_op.system * y_op.system)


@Operator.register_add_handler(
    [
        (GibbsDensityOperator, MixtureDensityOperator),
    ]
)
def _(x_op: GibbsDensityOperator, y_op: MixtureDensityOperator):
    return MixtureDensityOperator((x_op,) + y_op.terms, x_op.system * y_op.system)


@Operator.register_add_handler(
    [
        (
            MixtureDensityOperator,
            GibbsDensityOperator,
        ),
    ]
)
def _(y_op: MixtureDensityOperator, x_op: GibbsDensityOperator):
    return MixtureDensityOperator((x_op,) + y_op.terms, x_op.system * y_op.system)


@Operator.register_add_handler(
    [(GibbsDensityOperator, type_op) for type_op in BASIC_OPERATOR_TYPES]
)
@Operator.register_add_handler((GibbsDensityOperator, SumOperator))
@Operator.register_add_handler((GibbsDensityOperator, OneBodyOperator))
def add_gdo_sum_(x_op: GibbsDensityOperator, y_op: Operator):

    result = x_op.to_qutip_operator() + y_op
    return result


# ####### Multiplication ###################


# ## with numbers


@Operator.register_mul_handler(
    [(GibbsDensityOperator, type_op) for type_op in REAL_NUMERIC_TYPES]
)
def _(x_op: GibbsDensityOperator, y_op: float):
    if y_op == 0:
        return GibbsDensityOperator(
            ScalarOperator(0, x_op.system), x_op.system, prefactor=0, normalized=False
        )
    if y_op == 1:
        return x_op
    if 0 < y_op < 1:
        return GibbsDensityOperator(
            x_op.k,
            x_op.system,
            prefactor=x_op.prefactor * y_op,
            normalized=True,
        )
    # Generic
    print("    converting to qutip density operator")
    return x_op.to_qutip_operator() * y_op


@Operator.register_mul_handler(
    [(type_op, GibbsDensityOperator) for type_op in REAL_NUMERIC_TYPES]
)
def _(y_op: float, x_op: GibbsDensityOperator):
    if y_op == 0:
        return GibbsDensityOperator(
            ScalarOperator(0, x_op.system), x_op.system, prefactor=0, normalized=False
        )
    if y_op == 1:
        return x_op
    if 0 < y_op < 1:
        return GibbsDensityOperator(
            x_op.k,
            x_op.system,
            prefactor=x_op.prefactor * y_op,
            normalized=x_op.normalized,
        )
    # Generic
    return x_op.to_qutip_operator() * y_op


@Operator.register_mul_handler(
    [(GibbsDensityOperator, type_op) for type_op in COMPLEX_NUMERIC_TYPES]
)
def _(x_op: GibbsDensityOperator, y_op: float):
    return x_op.to_qutip_operator() * y_op


@Operator.register_mul_handler(
    [(type_op, GibbsDensityOperator) for type_op in COMPLEX_NUMERIC_TYPES]
)
def _(y_op: float, x_op: GibbsDensityOperator):
    return x_op.to_qutip_operator() * y_op


# With other DensityOperators:


@Operator.register_mul_handler((GibbsDensityOperator, GibbsDensityOperator))
def _(x_op: GibbsDensityOperator, y_op: GibbsDensityOperator):
    return x_op.to_qutip_operator() * y_op.to_qutip_operator()


@Operator.register_mul_handler((GibbsProductDensityOperator, GibbsDensityOperator))
def _(x_op: GibbsProductDensityOperator, y_op: GibbsDensityOperator):
    return x_op.to_qutip_operator() * y_op.to_qutip_operator()


@Operator.register_mul_handler((GibbsDensityOperator, GibbsProductDensityOperator))
def _(x_op: GibbsDensityOperator, y_op: GibbsProductDensityOperator):
    return x_op.to_qutip_operator() * y_op.to_qutip_operator()


@Operator.register_mul_handler(
    [
        (GibbsDensityOperator, type_op)
        for type_op in (
            DensityOperatorMixin,
            ProductDensityOperator,
#            MixtureDensityOperator,
        )
    ]
)
def _(x_op: GibbsDensityOperator, y_op: DensityOperatorMixin):
    return x_op.to_qutip_operator() * y_op


@Operator.register_mul_handler(
    [
        (type_op, GibbsDensityOperator)
        for type_op in (
            DensityOperatorMixin,
            ProductDensityOperator,
#            MixtureDensityOperator,
        )
    ]
)
def _(x_op: DensityOperatorMixin, y_op: GibbsDensityOperator):
    return x_op * y_op.to_qutip_operator()


## With Basic Operators


@Operator.register_mul_handler(
    [
        (GibbsDensityOperator, type_op) for type_op in NON_PRODUCT_BASIC_OPERATOR_TYPES
          if type_op is not SumOperator
     ]
)
@Operator.register_mul_handler((GibbsDensityOperator, SumOperator))
def _(x_op: GibbsDensityOperator, y_op: Operator):
    return x_op.to_qutip_operator() * y_op


@Operator.register_mul_handler(
    [(type_op, GibbsDensityOperator) for type_op in NON_PRODUCT_BASIC_OPERATOR_TYPES
     if type_op is not SumOperator
     ]
)
@Operator.register_mul_handler((SumOperator, GibbsDensityOperator))
def _(x_op: Operator, y_op: GibbsDensityOperator):
    return x_op * y_op.to_qutip_operator()


@Operator.register_mul_handler((GibbsDensityOperator, ScalarOperator))
@Operator.register_mul_handler((GibbsDensityOperator, LocalOperator))
@Operator.register_mul_handler((GibbsDensityOperator, ProductOperator))
def _(x_op: GibbsDensityOperator, y_op: Operator):
    return x_op.to_qutip_operator() * y_op


@Operator.register_mul_handler((ScalarOperator, GibbsDensityOperator))
@Operator.register_mul_handler((LocalOperator, GibbsDensityOperator))
@Operator.register_mul_handler((ProductOperator, GibbsDensityOperator))
def _(x_op: Operator, y_op: GibbsDensityOperator):
    return x_op * y_op.to_qutip_operator()
