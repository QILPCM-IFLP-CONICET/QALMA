"""
Functions for operators.
"""

from numbers import Complex, Real

# from collections.abc import Iterable
# from typing import Callable, List, Optional, Tuple
from typing import Tuple

from numpy import imag, real

from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import (
    LocalOperator,
    Operator,
    ProductOperator,
    ScalarOperator,
)
from qalma.operators.qutip import QutipOperator


def compute_dagger(operator):
    """
    Compute the adjoint of an `operator.
    If `operator` is a number, return its complex conjugate.
    """
    if isinstance(operator, Real):
        return operator
    if isinstance(operator, Complex):
        if operator.imag == 0:
            return operator.real
        return operator.conj()
    return operator.dag()


def hermitian_and_antihermitian_parts(operator: Operator) -> Tuple[Operator, Operator]:
    """Decompose an operator Q as A + i B with
    A and B self-adjoint operators
    """
    from qalma.operators.quadratic import QuadraticFormOperator

    system = operator.system
    if operator.isherm:
        return operator, ScalarOperator(0, system)

    operator = operator.simplify()
    if isinstance(operator, OneBodyOperator):

        terms = [hermitian_and_antihermitian_parts(term) for term in operator.terms]
        herm_terms = tuple(term[0] for term in terms)
        antiherm_terms = tuple(term[1] for term in terms)
        return (
            OneBodyOperator(herm_terms, system, isherm=True).simplify(),
            OneBodyOperator(antiherm_terms, system, isherm=True).simplify(),
        )

    if isinstance(operator, SumOperator):
        terms = [hermitian_and_antihermitian_parts(term) for term in operator.terms]
        herm_terms = tuple(term[0] for term in terms)
        antiherm_terms = tuple(term[1] for term in terms)
        return (
            SumOperator(herm_terms, system, isherm=True).simplify(),
            SumOperator(antiherm_terms, system, isherm=True).simplify(),
        )

    if isinstance(operator, QuadraticFormOperator):
        weights = operator.weights
        basis = operator.basis
        system = operator.system
        offset = operator.offset
        linear_term = operator.linear_term
        if offset is None:
            real_offset, imag_offset = (None, None)
        else:
            real_offset, imag_offset = hermitian_and_antihermitian_parts(offset)

        if linear_term is None:
            real_linear_term, imag_linear_term = (None, None)
        else:
            real_linear_term, imag_linear_term = hermitian_and_antihermitian_parts(
                linear_term
            )

        weights_re, weights_im = tuple((real(w) for w in weights)), tuple(
            (imag(w) for w in weights)
        )
        return (
            QuadraticFormOperator(
                basis,
                weights_re,
                system=system,
                offset=real_offset,
                linear_term=real_linear_term,
            ).simplify(),
            QuadraticFormOperator(
                basis,
                weights_im,
                system=system,
                offset=imag_offset,
                linear_term=imag_linear_term,
            ).simplify(),
        )

    if isinstance(operator, ProductOperator):
        sites_op = operator.sites_op
        system = operator.system
        if len(operator.sites_op) == 1:
            site, loc_op = next(iter(sites_op.items()))
            loc_op = loc_op * 0.5
            loc_op_dag = loc_op.dag()
            return (
                LocalOperator(site, loc_op + loc_op_dag, system),
                LocalOperator(site, loc_op * 1j - loc_op_dag * 1j, system),
            )

    elif isinstance(operator, (LocalOperator, QutipOperator)):
        operator = operator * 0.5
        op_dagger = compute_dagger(operator)
        return (
            (operator + op_dagger).simplify(),
            (op_dagger - operator).simplify() * 1j,
        )

    operator = operator * 0.5
    operator_dag = compute_dagger(operator)
    return (
        SumOperator(
            (
                operator,
                operator_dag,
            ),
            system,
            isherm=True,
        ).simplify(),
        SumOperator(
            (
                operator_dag * 1j,
                operator * (-1j),
            ),
            system,
            isherm=True,
        ).simplify(),
    )
