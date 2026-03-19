"""
Arithmetic bindings for LocalOperator.

Covers:
  - LocalOperator + Number
  - LocalOperator + ScalarOperator
  - LocalOperator + LocalOperator
  - LocalOperator * Number  (and reversed)
  - LocalOperator * ScalarOperator  (and reversed)
  - LocalOperator * LocalOperator
  - LocalOperator * ProductOperator  (and reversed)
"""

import numpy as np

from qalma.operators.arithmetic import OneBodyOperator
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.product import ProductOperator, ScalarOperator

from ._common import NUMERIC_TYPES


@Operator.register_add_handler([(LocalOperator, t) for t in NUMERIC_TYPES])
def _(x_op: LocalOperator, y_val: complex) -> Operator:
    new_op = x_op.operator.copy()
    np.fill_diagonal(new_op, new_op.diagonal() + y_val)
    return LocalOperator(x_op.site, new_op, x_op.system)


@Operator.register_add_handler((LocalOperator, ScalarOperator))
def _(x_op: LocalOperator, y_op: ScalarOperator) -> Operator:
    system = x_op.system.union(y_op.system)
    new_op = x_op.operator.copy()
    np.fill_diagonal(new_op, new_op.diagonal() + y_op.prefactor)
    return LocalOperator(x_op.site, new_op, system)


@Operator.register_add_handler((LocalOperator, LocalOperator))
def _(x_op: LocalOperator, y_op: LocalOperator):
    system = x_op.system.union(y_op.system)
    if x_op.site == y_op.site:
        return LocalOperator(x_op.site, x_op.operator + y_op.operator, system)
    return OneBodyOperator((x_op, y_op), system)


@Operator.register_mul_handler([(LocalOperator, t) for t in NUMERIC_TYPES])
def _(x_op: LocalOperator, y_val: complex):
    return LocalOperator(x_op.site, x_op.operator * y_val, x_op.system)


@Operator.register_mul_handler([(t, LocalOperator) for t in NUMERIC_TYPES])
def _(y_val: complex, x_op: LocalOperator):
    return LocalOperator(x_op.site, x_op.operator * y_val, x_op.system)


@Operator.register_mul_handler((LocalOperator, ScalarOperator))
def _(x_op: LocalOperator, y_op: ScalarOperator):
    return LocalOperator(
        x_op.site, x_op.operator * y_op.prefactor, x_op.system or y_op.system
    )


@Operator.register_mul_handler((ScalarOperator, LocalOperator))
def _(y_op: ScalarOperator, x_op: LocalOperator):
    return LocalOperator(
        x_op.site, x_op.operator * y_op.prefactor, x_op.system or y_op.system
    )


@Operator.register_mul_handler((LocalOperator, LocalOperator))
def _(x_op: LocalOperator, y_op: LocalOperator):
    site_x = x_op.site
    site_y = y_op.site
    system = x_op.system or y_op.system
    if site_x == site_y:
        return LocalOperator(site_x, x_op.operator @ y_op.operator, system)
    return ProductOperator(
        sites_operators={site_x: x_op.operator, site_y: y_op.operator},
        prefactor=1,
        system=system,
    )


@Operator.register_mul_handler((ProductOperator, LocalOperator))
def _(x_op: ProductOperator, y_op: LocalOperator):
    site = y_op.site
    op_local = y_op.operator
    system = x_op.system * y_op.system if x_op.system else y_op.system
    sites_op = x_op.site_factors.copy()
    if site in sites_op:
        op_local = sites_op[site] @ op_local
    sites_op[site] = op_local
    if len(sites_op) == 1:
        site, array_op_local = next(iter(sites_op.items()))
        return LocalOperator(site, array_op_local * x_op.prefactor, system)
    return ProductOperator(sites_op, x_op.prefactor, system)


@Operator.register_mul_handler((LocalOperator, ProductOperator))
def _(y_op: LocalOperator, x_op: ProductOperator):
    site = y_op.site
    op_local = y_op.operator
    system = x_op.system * y_op.system if x_op.system else y_op.system
    sites_op = x_op.site_factors.copy()
    if site in sites_op:
        op_local = op_local @ sites_op[site]
    sites_op[site] = op_local
    if len(sites_op) == 1:
        site, array_op_local = next(iter(sites_op.items()))
        return LocalOperator(site, array_op_local * x_op.prefactor, system)
    return ProductOperator(sites_op, x_op.prefactor, system)
