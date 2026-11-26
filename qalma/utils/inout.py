"""Utility functions for export objects into different formats."""

import numbers
from typing import Union

import numpy as np
import qutip  # type: ignore[import-untyped]


def operator_to_wolfram(operator) -> str:
    """WL representation of the operator.

    Produce a string with a Wolfram Mathematica expression representing the
    operator.
    """
    # pylint: disable=import-outside-toplevel
    from qutip import Qobj

    from qalma.operators.arithmetic import SumOperator
    from qalma.operators.basic import LocalOperator, Operator
    from qalma.operators.product import ProductOperator

    def get_site_identity(site_name):
        """The identity operator of the site."""
        site_spec = sites[site_name]
        if "operators" in site_spec:
            return site_spec["operators"]["identity"]
        dim = dimensions[site_name]
        result = qutip.qeye(dim)
        site_spec["operators"] = {"identity": result}
        return result

    if hasattr(operator, "to_wolfram"):
        return operator.to_wolfram()

    if isinstance(operator, Qobj):
        data = operator.data
        if hasattr(data, "toarray"):
            array = data.toarray()
        elif hasattr(data, "to_array"):
            array = data.to_array()
        else:
            raise TypeError((f"Do not know how to convert {type(data)} into a ndarray"))

        assert len(array.shape) == 2, f"the shape  {array.shape} is not a matrix"
        return matrix_to_wolfram(array)

    if isinstance(operator, SumOperator):
        assert all(isinstance(term, Operator) for term in operator.terms)
        terms = [operator_to_wolfram(term) for term in operator.terms]
        terms = [term for term in terms if term != "0"]
        return "(" + " + ".join(terms) + ")"

    sites = operator.system.sites
    dimensions = operator.system.dimensions
    prefactor = operator.prefactor
    if prefactor == 0:
        return "0"

    prefix = "KroneckerProduct["
    if prefactor != 1:
        prefix = f"({prefactor}) * " + prefix

    if isinstance(operator, LocalOperator):
        local_site = operator.site
        factors = [
            operator.operator_qutip if site == local_site else get_site_identity(site)
            for site in sorted(dimensions)
        ]
        factors_str = [operator_to_wolfram(factor) for factor in factors]

        return prefix + ", ".join(factors_str) + "]"

    if isinstance(operator, ProductOperator):
        factors = [
            operator.site_factors_qutip.get(site, get_site_identity(site))
            for site in sorted(dimensions)
        ]
        factors_str = [operator_to_wolfram(factor) for factor in factors]

        return prefix + ", ".join(factors_str) + "]"

    if hasattr(operator, "prefactor"):
        return (
            "("
            + str(prefactor)
            + ") * "
            + operator_to_wolfram((operator / prefactor).to_qutip())
        )

    return operator_to_wolfram(operator.to_qutip())


def _number_to_wolfram(num) -> str:
    """WL representation of an integer, real or complex number.

    Integers are exported as WL integers (``3``), reals as machine reals
    (``3.0``, ``1.5*^-10``) and complex numbers using ``I`` for the imaginary
    unit.
    """
    assert isinstance(num, numbers.Number) and not isinstance(
        num, (bool, np.bool_)
    ), f"{type(num)} {num} is not an integer, float or complex."

    if isinstance(num, numbers.Integral):
        return str(int(num))
    if isinstance(num, numbers.Real):
        return str(num).replace("e", "*^")
    if num.imag == 0:
        return str(num.real).replace("e", "*^")
    return (
        str(num).replace("(", "").replace(")", "").replace("e", "*^").replace("j", "I")
    )


def matrix_to_wolfram(matr: Union[np.ndarray, numbers.Number]) -> str:
    """Produce a string representing the data in the matrix.

    Parameters
    ----------
    matr: numpy.ndarray or number
        A two-dimensional array of integer, real or complex numbers, or a
        single number.

    Returns
    -------
    str
        A WL expression: a list of lists for an array, or the number.
        Integer entries are exported as WL integers, and the remaining real
        entries as machine reals.

    """
    assert isinstance(
        matr, (np.ndarray, numbers.Number)
    ), f"{type(matr)} is not ndarray or number"

    if isinstance(matr, np.ndarray):
        assert matr.ndim == 2, f"the shape {matr.shape} is not a matrix"
        rows = [
            "{" + (", ".join(_number_to_wolfram(elem) for elem in row)) + "}"
            for row in matr
        ]
        return "{\n" + ",\n".join(rows) + "\n}"
    return _number_to_wolfram(matr)
