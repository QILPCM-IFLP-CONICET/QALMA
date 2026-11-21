"""Arithmetic bindings for BlockOperator.

Covers:
  - BlockOperator + any Operator
  - BlockOperator + BlockOperator
  - BlockOperator + Number  (and reversed)
  - BlockOperator * Number  (and reversed)
  - BlockOperator * BlockOperator
  - BlockOperator * ScalarOperator  (and reversed)
  - BlockOperator * Operator / LocalOperator / ProductOperator  (and reversed)
  - LocalOperator + BlockOperator
"""

from typing import Union

from qutip import Qobj

from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import LocalOperator, Operator
from qalma.operators.block import BlockOperator
from qalma.operators.product import ProductOperator, ScalarOperator

from ._common import NUMERIC_TYPES


@Operator.register_add_handler(
    [
        (BlockOperator, op_type)
        for op_type in (
            Operator,
            ScalarOperator,
            LocalOperator,
            ProductOperator,
            OneBodyOperator,
        )
    ]
)
def _(x_op: BlockOperator, y_op: Operator):
    system = x_op.system.union(y_op.system)
    return SumOperator((x_op, y_op), system)


@Operator.register_add_handler((BlockOperator, BlockOperator))
def _(x_op: BlockOperator, y_op: BlockOperator):
    system = x_op.system.union(y_op.system)
    x_names = x_op.site_names
    y_names = y_op.site_names
    if x_names == y_names:
        x_qutip = x_op.operator
        y_qutip = y_op.operator
        if x_qutip is y_qutip:
            return BlockOperator(
                x_qutip,
                system,
                names=x_names,
                prefactor=x_op.prefactor + y_op.prefactor,
            )
        return BlockOperator(
            x_qutip * x_op.prefactor + y_qutip * y_op.prefactor,
            system,
            names=x_names,
            prefactor=1,
        )
    block_set = set(x_names)
    block_set.update(y_names)
    if len(block_set) <= max(len(x_names), len(y_names)):
        block = sorted(block_set)
        qutip_sum = x_op.to_qutip(tuple(block)) + y_op.to_qutip(tuple(block))
        return BlockOperator(
            qutip_sum,
            system,
            names={site: i for i, site in enumerate(block)},
            prefactor=1,
        )
    return SumOperator((x_op, y_op), system)


@Operator.register_add_handler([(t, BlockOperator) for t in NUMERIC_TYPES])
def _(y_val: complex, x_op: BlockOperator):
    return BlockOperator(
        x_op.operator * x_op.prefactor + y_val,
        x_op.system,
        names=x_op.site_names,
        prefactor=1,
    )


@Operator.register_add_handler([(BlockOperator, t) for t in NUMERIC_TYPES])
def _(x_op: BlockOperator, y_val: Union[complex, Qobj]):
    return BlockOperator(
        x_op.operator * x_op.prefactor + y_val,
        x_op.system,
        names=x_op.site_names,
        prefactor=1,
    )


@Operator.register_mul_handler([(BlockOperator, t) for t in NUMERIC_TYPES])
def _(x_op: BlockOperator, y_val: Union[complex, Qobj]):
    return BlockOperator(
        x_op.operator,
        x_op.system,
        names=x_op.site_names,
        prefactor=x_op.prefactor * y_val,
    )


@Operator.register_mul_handler([(t, BlockOperator) for t in NUMERIC_TYPES])
def _(y_val: complex, x_op: BlockOperator):
    return BlockOperator(
        x_op.operator,
        x_op.system,
        names=x_op.site_names,
        prefactor=x_op.prefactor * y_val,
    )


@Operator.register_add_handler([(LocalOperator, BlockOperator)])
def _(x_op: Operator, y_qutip_op: BlockOperator):
    return x_op.to_qutip_operator() + y_qutip_op


@Operator.register_mul_handler(
    [(t, BlockOperator) for t in (Operator, LocalOperator, ProductOperator)]
)
def _(x_op: Operator, y_qutip_op: BlockOperator):
    if x_op.acts_over():
        return x_op.to_qutip_operator() * y_qutip_op
    return y_qutip_op * x_op.prefactor


@Operator.register_mul_handler(
    [(BlockOperator, t) for t in (Operator, LocalOperator, ProductOperator)]
)
def _(x_qutip_op: BlockOperator, y_op: Operator):
    if y_op.acts_over():
        return x_qutip_op * y_op.to_qutip_operator()
    return x_qutip_op * y_op.prefactor


@Operator.register_mul_handler((BlockOperator, BlockOperator))
def _(x_op: BlockOperator, y_op: BlockOperator):
    x_names = x_op.site_names
    y_names = y_op.site_names
    if not x_names:
        if not y_names:
            system = x_op.system.union(y_op.system)
            return BlockOperator(
                1,
                names=x_names,
                prefactor=x_op.prefactor * y_op.prefactor,
                system=system,
            )
        return y_op * x_op.prefactor
    if not y_names:
        return x_op * y_op.prefactor
    system = x_op.system.union(y_op.system)
    if x_names == y_names:
        return BlockOperator(
            x_op.operator * y_op.operator,
            system,
            names=x_names,
            prefactor=x_op.prefactor * y_op.prefactor,
        )
    names_set = set(x_names)
    names_set.update(y_names)
    block = tuple(sorted(names_set))
    if x_op.system is not system:
        x_op = BlockOperator(x_op.operator, system, x_op.site_names, x_op.prefactor)
    if y_op.system is not system:
        y_op = BlockOperator(y_op.operator, system, y_op.site_names, y_op.prefactor)
    operator_qutip = x_op.to_qutip(block) * y_op.to_qutip(block)
    return BlockOperator(
        operator_qutip,
        system,
        names={site: i for i, site in enumerate(block)},
        prefactor=1,
    )


@Operator.register_mul_handler((ScalarOperator, BlockOperator))
def _(x_op: ScalarOperator, y_op: BlockOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    return BlockOperator(
        y_op.operator,
        names=y_op.site_names,
        prefactor=x_op.prefactor * y_op.prefactor,
        system=system,
    )


@Operator.register_mul_handler((BlockOperator, ScalarOperator))
def _(y_op: BlockOperator, x_op: ScalarOperator):
    system = x_op.system * y_op.system if x_op.system else y_op.system
    return BlockOperator(
        y_op.operator,
        names=y_op.site_names,
        prefactor=x_op.prefactor * y_op.prefactor,
        system=system,
    )
