"""
Fidelity and related functions.
"""

import numpy as np

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

    # print("general routine for fidelity")
    radicand = (rho1 * rho2).to_qutip_operator()
    # print("rho1=", operator_to_wolfram(rho1),";")
    # print("rho2=", operator_to_wolfram(rho2),";")
    # print("radicand=", operator_to_wolfram(radicand),";")

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
    for factor in radicand.sites_op.values():
        result += np.log(sum(abs(factor.eigenenergies()) ** 0.5))

    return np.exp(result)
