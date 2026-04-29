"""
Fidelity and related functions.
"""

import numpy as np
from scipy.linalg import eigvals as scp_eigvals

from qalma.operators import ProductOperator


def fidelity(rho1, rho2) -> float:
    """
    Compute the fidelity between two states.
    Following qutip, we compute the  Bhattacharyya's
    quantum coefficient (the square root of the fidelity), which
    is the maximum  absolute value of the overlap between
    all the possible purifications of the states.
    """

    if isinstance(rho1, ProductOperator):
        if isinstance(rho2, ProductOperator):
            return fidelity_product_states(rho1, rho2)

    radicand = (rho1 * rho2).to_qutip_operator()
    return sum(abs(radicand.eigenenergies()) ** 0.5)


def fidelity_product_states(rho1: ProductOperator, rho2: ProductOperator) -> float:
    """
    Compute the fidelity between two product states.
    Following qutip, we compute the  Bhattacharyya's
    quantum coefficient (the square root of the fidelity), which
    is the maximum  absolute value of the overlap between
    all the possible purifications of the states.
    """

    radicand = rho1 * rho2

    result = 0.5 * np.log(radicand.prefactor)
    for factor in radicand.site_factors.values():
        result += np.log(sum(abs(scp_eigvals(factor)) ** 0.5))

    return np.exp(result)
