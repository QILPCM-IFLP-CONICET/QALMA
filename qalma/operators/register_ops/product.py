"""
Arithmetic bindings for ProductOperator.

Covers:
  - ProductOperator * ProductOperator
  - ProductOperator * Number  (and reversed)
  - ProductOperator * ScalarOperator  (and reversed)
  - ProductOperator + Number
  - ProductOperator + ProductOperator / ScalarOperator  (simplified via simplify())
  - ProductOperator + LocalOperator
"""

from qalma.operators.arithmetic import SumOperator
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.product import ProductOperator, ScalarOperator

from ._common import NUMERIC_TYPES


@Operator.register_mul_handler((ProductOperator, ProductOperator))
def _(x_op: ProductOperator, y_op: ProductOperator):
    system = x_op.system.union(y_op.system)
    site_op = x_op.sites_op.copy()
    for site, op_local in y_op.sites_op.items():
        site_op[site] = site_op[site] @ op_local if site in site_op else op_local
    prefactor = x_op.prefactor * y_op.prefactor
    if len(site_op) == 0 or prefactor == 0:
        return ScalarOperator(prefactor, system)
    if len(site_op) == 1:
        site, array_op_local = next(iter(site_op.items()))
        # Pass ndarray directly — LocalOperator.__init__ calls _to_array internally
        return LocalOperator(site, array_op_local * prefactor, system)
    return ProductOperator(site_op, prefactor, system)


@Operator.register_mul_handler([(ProductOperator, t) for t in NUMERIC_TYPES])
def _(x_op: ProductOperator, y_value: complex) -> Operator:
    if y_value:
        return ProductOperator(x_op.sites_op, x_op.prefactor * y_value, x_op.system)
    return ScalarOperator(0, x_op.system)


@Operator.register_mul_handler([(t, ProductOperator) for t in NUMERIC_TYPES])
def _(y_value: complex, x_op: ProductOperator):
    if y_value:
        return ProductOperator(x_op.sites_op, x_op.prefactor * y_value, x_op.system)
    return ScalarOperator(0, x_op.system)


@Operator.register_mul_handler((ProductOperator, ScalarOperator))
def _(x_op: ProductOperator, y_op: ScalarOperator):
    prefactor = y_op.prefactor
    if prefactor:
        return ProductOperator(x_op.sites_op, x_op.prefactor * prefactor, x_op.system)
    return ScalarOperator(0, x_op.system)


@Operator.register_mul_handler((ScalarOperator, ProductOperator))
def _(y_op: ScalarOperator, x_op: ProductOperator):
    prefactor = y_op.prefactor
    if prefactor:
        return ProductOperator(x_op.sites_op, x_op.prefactor * prefactor, x_op.system)
    return ScalarOperator(0, x_op.system)


@Operator.register_add_handler([(ProductOperator, t) for t in NUMERIC_TYPES])
def _(x_op: ProductOperator, y_value: complex):
    prefactor = x_op.prefactor
    system = x_op.system
    sites_op = x_op.sites_op
    if len(sites_op) == 0:
        return ScalarOperator(prefactor + y_value, system)
    if len(sites_op) == 1:
        first_site, first_loc_op = next(iter(x_op.sites_op.items()))
        return LocalOperator(first_site, first_loc_op * prefactor + y_value, system)
    return SumOperator((x_op, ScalarOperator(y_value, system)), system)


@Operator.register_add_handler(
    [
        (ProductOperator, ProductOperator),
        (ScalarOperator, ProductOperator),
        (ProductOperator, ScalarOperator),
    ]
)
def _(x_op: ProductOperator, y_op: ProductOperator):
    system = x_op.system or y_op.system
    if len(x_op.sites_op) > 1 or len(y_op.sites_op) > 1:
        return SumOperator((x_op, y_op), system)
    return x_op.simplify() + y_op.simplify()


@Operator.register_add_handler((ProductOperator, LocalOperator))
def _(x_op: ProductOperator, y_op: LocalOperator):
    system = x_op.system or y_op.system
    if len(x_op.sites_op) > 1:
        return SumOperator((x_op, y_op), system)
    return x_op.simplify() + y_op.simplify()
