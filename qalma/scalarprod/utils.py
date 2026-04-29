"""
Util functions used in scalar product related functions.
"""

from typing import Generator, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr

from qalma.settings import QALMA_TOLERANCE


def find_linearly_independent_rows(
    mat: NDArray, tol: float = QALMA_TOLERANCE
) -> Tuple[int]:
    """
    Find indices of a maximal subset of linearly independent columns of the matrix.
    """
    tol = min(tol, min(row[i] for i, row in enumerate(mat)) * 0.25)
    r, inds = qr(mat, mode="r", pivoting=True)
    rank = np.linalg.matrix_rank(r, tol=tol)
    # The first `rank` indices are linearly independent columns
    return tuple(sorted(inds[:rank]))


def iterator_transverse(tn_1: list, tn_2: list) -> Generator:
    """
    Given two lists `tn_1`, `tn_2`, generate a list of
    indices `i`,`j` of the matrix `m[i,j]=tn_1[i]tn_2[j]`
    from the lower-right corner to the upper-left corner,
    going through (anti)-diagonals in alternating directions.
    """
    m1 = len(tn_1)
    m2 = len(tn_2)
    for s in range(m1 + m2 - 2, -1, -1):
        lower_bound_i = max(0, s - m2 + 1)
        upper_bound_i = min(s, m1 - 1)
        if s % 2 == 0:
            for i in range(lower_bound_i, upper_bound_i + 1):
                j = s - i
                yield (
                    (
                        i,
                        j,
                    )
                )
        else:
            for i in range(upper_bound_i, lower_bound_i - 1, -1):
                j = s - i
                yield (
                    (
                        i,
                        j,
                    )
                )


def iterator_transverse_upper(tn_1: list) -> Generator:
    """
    Given a list `tn_1`, generate a list of
    indices `i`,`j` of the matrix `m[i,j]=tn_1[i]tn_1[j]`
    from the lower-right corner to the upper-left corner,
    going through (anti)-diagonals in alternating directions,
    covering the strict upper-triangular submatrix.
    """
    m = len(tn_1)
    for s in range(2 * m - 3, 0, -1):
        lower_bound_i = max(0, s - m + 1)
        upper_bound_i = min(m - 1, s // 2)
        if s % 2 == 0:
            for i in range(lower_bound_i, upper_bound_i + 1):
                j = s - i
                if i == j:
                    continue
                yield (
                    (
                        i,
                        j,
                    )
                )
        else:
            for i in range(upper_bound_i, lower_bound_i - 1, -1):
                j = s - i
                if i == j:
                    continue
                yield (
                    (
                        i,
                        j,
                    )
                )
