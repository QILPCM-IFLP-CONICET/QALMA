from typing import Tuple

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
    print("type mat", mat)
    r, inds = qr(np.array(mat), mode="r", pivoting=True)
    rank = np.linalg.matrix_rank(r, tol=tol)
    print("rank", tuple(sorted(inds[:rank])))
    # The first `rank` indices are linearly independent columns
    return tuple(sorted(inds[:rank]))
