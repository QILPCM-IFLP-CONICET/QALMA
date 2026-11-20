"""Gram's matrix computation routines.

Routines to build the associated to a scalar product and an
operator basis.
"""

import logging
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray

from qalma.scalarprod.utils import find_linearly_independent_rows

logger = logging.getLogger(__name__)

# from datetime import datetime

__all__ = [
    "gram_matrix",
    "merge_gram_matrices",
]


def gram_matrix(basis, sp: Callable) -> NDArray:
    """Compute compute the Gram matrix for an operator basis (serial implementation).

    Computes the Gram matrix of a given operator basis using a scalar product.

    The Gram matrix is symmetric and defined as:

    .. code-block:: python

        Gij = sp(op1, op2)

    where ``sp`` is the scalar product function and `op1, op2` are operators from
    the basis.

    Parameters
    ----------
    basis: List[Operator]
        A list of basis operators.
    sp: Callable
        A callable that defines a scalar product function between two
    operators.

    Return
    ------

    result: NDArray
        A symmetric NumPy array representing the Gram matrix, with entries
        rounded to 14 decimal places.

    """
    if hasattr(sp, "compute_gram_matrix"):  # and all(b_i.isherm for b_i in basis):
        return sp.compute_gram_matrix(basis)

    size = len(basis)
    result = np.zeros([size, size], dtype=float)

    for i, op1 in enumerate(basis):
        for j, op2 in enumerate(basis):
            if j < i:
                continue  # Use symmetry: Gij = Gji.

            entry = np.real(sp(op1, op2))
            if i == j:
                result[i, i] = entry  # Diagonal elements.
            else:
                result[i, j] = result[j, i] = entry  # Off-diagonal elements.

    return result.round(14)


def merge_gram_matrices(
    g11: NDArray, g11_inv: NDArray, g22: NDArray, g12: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray, Tuple[int]]:
    """Build the gram and gram_inv tensors from the available information.

    Given two basis, their Gram matrices, the inverse
    of the first and the cross gram matrix, this function finds
    a set of linearly independent basis elements that span
    the union of both basis, the Gram matrix associated to the new
    basis and its inverse, and the Gram matrices of the reduced blocks.

    Parameters
    ----------
    g11: NDArray
         The upper-left block of the Gram Matrix
    g11_inv: NDArray
         The inverse of the upper-left block of the Gram Matrix
    g22: NDArray
         The bottom-right block of the Gram Matrix
    g12: NDArray
         The cross block of the Gram Matrix

    Return values
    -------------

    gram_full: NDArray
        The gram matrix of the reduced basis

    gram_full_inv: NDArray
         The inverse of the gram matrix of the reduced basis

    g11, g22: NDArray
         The sub-blocks of g11 and g22 associated to the
         reduced common basis

    li_indices: Tuple[int]
       the list of indices of the elements in the original basis
       that span the linearly independent common basis.

    """
    # Build the new Gram matrix
    n_1 = len(g11)
    n_2 = len(g22)
    n_total = n_1 + n_2

    g21 = g12.T
    gram_full = np.block([[g11, g12], [g21, g22]])

    # If gram is singular, reduce it and remove the
    # linearly dependent elements.
    li_indices = find_linearly_independent_rows(gram_full)
    li_1_indices = tuple(i for i in li_indices if i < n_1)
    if len(li_1_indices) != n_1:
        raise ValueError("It looks like basis_1 were singular.")

    if len(li_indices) != n_total:
        n_total = len(li_indices)
        if n_total == n_1:
            return g11, g11_inv, g11, g11, li_indices
    n_2 = n_total - n_1
    gram_full = gram_full[li_indices, :][:, li_indices]
    g12 = gram_full[:n_1, n_1:]
    g21 = g12.T
    g22 = gram_full[n_1:, n_1:]

    # --- Gram inverse (block inversion, Schur complement) ---
    # Should not be singular, because we ensure that gram is not
    # singular...
    shur = g22 - g21 @ g11_inv @ g12

    try:
        shur_inv = np.linalg.inv(shur)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("Gram matrix is singular after merging bases") from e

    # Build the inverse
    top_left = g11_inv + g11_inv @ g12 @ shur_inv @ g21 @ g11_inv
    top_right = -g11_inv @ g12 @ shur_inv
    bottom_left = -shur_inv @ g21 @ g11_inv
    bottom_right = shur_inv
    gram_full_inv = np.block([[top_left, top_right], [bottom_left, bottom_right]])

    return gram_full, gram_full_inv, g11, g22, li_indices
