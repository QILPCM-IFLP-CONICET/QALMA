"""Spectral-related functions for operators."""

# from collections.abc import Iterable
# from typing import Callable, List, Optional, Tuple
import logging

from numpy import inf, ndarray, real
from scipy.linalg import eigvals as scp_eigvals, norm as scp_norm

from qalma.operators.arithmetic import OneBodyOperator
from qalma.operators.basic import (
    LocalOperator,
    Operator,
)
from qalma.operators.product import (
    ProductOperator,
    ScalarOperator,
)

# from qalma.operators.simplify import simplify_sum_operator


def eigenvalues(
    operator: Operator,
    sparse: bool = False,
    sort: str = "low",
    eigvals: int = 0,
    tol: float = 0.0,
    maxiter: int = 100000,
) -> ndarray:
    """Compute the eigenvalues of operator"""
    qutip_op = operator.to_qutip() if isinstance(operator, Operator) else operator
    if eigvals > 0 and qutip_op.data.shape[0] < eigvals:
        sparse = False
        eigvals = 0

    return qutip_op.eigenenergies(sparse, sort, eigvals, tol, maxiter)


def spectral_norm(operator: Operator) -> float:
    """Compute the spectral norm of the operator `op`"""
    if isinstance(operator, ScalarOperator):
        return abs(operator.prefactor)
    if isinstance(operator, LocalOperator):
        if operator.isherm:
            return max(abs(scp_eigvals(operator.operator)))
        return scp_norm(operator.operator, ord=inf)
    if isinstance(operator, ProductOperator):
        result = abs(operator.prefactor)
        for loc_op in operator.site_factors.values():
            result *= scp_norm(loc_op, ord=inf)
        return real(result)

    if operator.isherm:
        if isinstance(operator, OneBodyOperator):
            operator = operator.simplify()
            if hasattr(operator, "terms"):
                return sum(spectral_norm(term) for term in operator.terms)
        return max(abs(eigenvalues(operator)))
    return max(eigenvalues(operator.dag() * operator)) ** 0.5


def log_op(operator: Operator) -> Operator:
    """The logarithm of an operator"""
    assert isinstance(operator, Operator)
    if hasattr(operator, "logm"):
        return operator.logm()
    return operator.to_qutip_operator().logm()


def relative_entropy(rho: Operator, sigma: Operator) -> float:
    """Compute the relative entropy"""
    log_rho = log_op(rho)
    log_sigma = log_op(sigma)
    delta_log = (log_rho - log_sigma).simplify()

    if hasattr(rho, "expect"):
        result = real(rho.expect(delta_log))
    else:
        result = real((rho * delta_log).tr())
    if result < 0:
        logging.warning("S(rho|sigma)=%.4f<0", result)
    return max(0, result)
