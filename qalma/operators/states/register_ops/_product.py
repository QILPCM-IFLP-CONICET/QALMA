"""Arithmetic handlers involving ProductDensityOperator.

The strategy is to treat a ProductDensityOperator as a plain ProductOperator
for arithmetic purposes, delegating via `ProductOperator(y_op.site_factors, ...)`.

Registered operations:
  - number                 + ProductDensity            ->  (cast, then add)
  - BasicOperator/SumOp    + ProductDensity            ->  (cast, then add)
  - BasicOperator          * ProductDensity            ->  (cast, then mul)
  - ProductDensity         * BasicOperator             ->  (cast, then mul)
  - ProductDensity         * ProductDensity            ->  site-wise product
  - ProductDensity         * SumOp                    ->  distribute over terms
  - SumOp                  * ProductDensity            ->  distribute over terms
  - ProductDensity         + ScalarOperator            ->  simplify to local/sum
"""

from typing import cast

from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.product import ProductOperator, ScalarOperator
from qalma.operators.states.arithmetic import MixtureDensityOperator
from qalma.operators.states.product import ProductDensityOperator

from ._types import BASIC_OPERATOR_TYPES, COMPLEX_NUMERIC_TYPES, REAL_NUMERIC_TYPES
from ._wrappers import _wrapper_product as _as_product_op


@Operator.register_add_handler((ProductDensityOperator, ProductDensityOperator))
def _(x_op: ProductDensityOperator, y_op: ProductDensityOperator):
    """Multiply two product operators."""
    return MixtureDensityOperator((x_op, y_op), x_op.system * y_op.system)


@Operator.register_add_handler((ProductDensityOperator, ScalarOperator))
def _(x_op: ProductDensityOperator, y_op: ScalarOperator):
    """Multiply a product operator with an scalar."""
    site_op = x_op.site_factors.copy()
    prefactor = x_op.prefactor
    system = x_op.system or y_op.system
    if len(site_op) == 0:
        return ScalarOperator(prefactor + y_op.prefactor, system)
    if len(site_op) == 1:
        first_site, first_loc_op = next(iter(site_op.items()))
        return LocalOperator(
            first_site, first_loc_op * prefactor + y_op.prefactor, system
        )
    return SumOperator((x_op, y_op), system)


@Operator.register_add_handler(
    [(ProductDensityOperator, type_1) for type_1 in REAL_NUMERIC_TYPES]
)
def pdo_add_real_(x_op: ProductDensityOperator, y_op: float):
    """Multiply a product operator with an scalar."""
    if y_op == 0:
        return x_op
    system = x_op.system
    if y_op > 0:
        return MixtureDensityOperator(
            (ProductDensityOperator({}, y_op, system), x_op), system
        )
    return _as_product_op(x_op) + y_op


@Operator.register_add_handler(
    [(ProductDensityOperator, type_1) for type_1 in COMPLEX_NUMERIC_TYPES]
)
def prd_add_complex_(x_op: ProductDensityOperator, y_op: float):
    """Multiply a product operator with a complex number."""
    if y_op.imag == 0:
        return x_op + y_op.real
    if y_op == 0:
        return x_op
    return _as_product_op(x_op) + y_op


@Operator.register_add_handler(
    [(type_1, ProductDensityOperator) for type_1 in BASIC_OPERATOR_TYPES]
)
@Operator.register_add_handler((OneBodyOperator, ProductDensityOperator))
@Operator.register_add_handler((SumOperator, ProductDensityOperator))
def _(x_op: Operator, y_op: ProductDensityOperator):
    """Multiply a product operator sum operators."""
    return x_op + _as_product_op(y_op)


#  ####### MUL


@Operator.register_mul_handler(
    [(type_1, ProductDensityOperator) for type_1 in REAL_NUMERIC_TYPES]
)
def _(x_op: float, y_op: ProductDensityOperator):
    """Multiply a product operator with real numbers."""
    if x_op == 0:
        return ProductDensityOperator(
            {},
            weight=x_op,
            system=y_op.system,
            normalized=True,
        )
    if x_op > 0:
        return ProductDensityOperator(
            y_op.site_factors,
            weight=cast(float, y_op.prefactor) * x_op,
            system=y_op.system,
            normalized=True,
        )
    return _as_product_op(y_op, x_op)


@Operator.register_mul_handler(
    [(ProductDensityOperator, type_1) for type_1 in REAL_NUMERIC_TYPES]
)
def _(y_op: ProductDensityOperator, x_op: float):
    """Multiply a product operator with real numbers."""
    if x_op == 0:
        return ProductDensityOperator(
            {},
            weight=x_op,
            system=y_op.system,
            normalized=True,
        )
    if x_op > 0:
        return ProductDensityOperator(
            y_op.site_factors,
            weight=cast(float, y_op.prefactor) * x_op,
            system=y_op.system,
            normalized=True,
        )
    print("using as product op")
    return _as_product_op(y_op, x_op)


@Operator.register_mul_handler(
    [(type_1, ProductDensityOperator) for type_1 in COMPLEX_NUMERIC_TYPES]
)
def _(x_op: complex, y_op: ProductDensityOperator):
    """Multiply a product operator with complex numbers."""
    if x_op.imag == 0.0:
        return y_op * x_op.real
    return _as_product_op(y_op) * x_op


@Operator.register_mul_handler(
    [
        (
            ProductDensityOperator,
            type_1,
        )
        for type_1 in COMPLEX_NUMERIC_TYPES
    ]
)
def _(y_op: ProductDensityOperator, x_op: complex):
    """Multiply a product operator with complex numbers."""
    if x_op.imag == 0:
        return y_op * x_op.real
    return _as_product_op(y_op) * x_op


@Operator.register_mul_handler(
    [(type_1, ProductDensityOperator) for type_1 in BASIC_OPERATOR_TYPES]
)
def _(x_op: Operator, y_op: ProductDensityOperator):
    """Multiply a product operator with basic operators."""
    return x_op * _as_product_op(y_op)


@Operator.register_mul_handler(
    [(ProductDensityOperator, type_1) for type_1 in BASIC_OPERATOR_TYPES]
)
def _(y_op: ProductDensityOperator, x_op: Operator):
    """Multiply a product operator with basic operators."""
    return _as_product_op(y_op) * x_op


@Operator.register_mul_handler((ProductDensityOperator, ProductDensityOperator))
def _(x_op: ProductDensityOperator, y_op: ProductDensityOperator):
    """Multiply a product operator with a product density operator."""
    system = x_op.system * y_op.system if x_op.system else y_op.system
    site_factors = x_op.site_factors.copy()
    for site, factor in y_op.site_factors.items():
        if site in site_factors:
            site_factors[site] *= factor
        else:
            site_factors[site] = factor
    return ProductOperator(site_factors, 1, system)


@Operator.register_mul_handler((ProductDensityOperator, OneBodyOperator))
@Operator.register_mul_handler((ProductDensityOperator, SumOperator))
def _(x_op: ProductDensityOperator, y_op: SumOperator):
    """Product density operator with a sum operator."""
    system = x_op.system * y_op.system if x_op.system else y_op.system
    return SumOperator(
        tuple(x_op * term for term in y_op.terms),
        system,
    )


@Operator.register_mul_handler((SumOperator, ProductDensityOperator))
@Operator.register_mul_handler((OneBodyOperator, ProductDensityOperator))
def _(x_op: SumOperator, y_op: ProductDensityOperator):
    """Product density operator with a sum operator."""
    system = x_op.system * y_op.system if x_op.system else y_op.system
    return SumOperator(
        tuple(term * y_op for term in x_op.terms),
        system,
    )
