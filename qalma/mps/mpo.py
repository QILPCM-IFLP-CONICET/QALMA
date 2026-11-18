r"""Matrix-product operators built from sums of few-body terms.

The Hamiltonians of QALMA are sums of terms, each acting on a small set of
sites. Given the position of every site along the chain, each term is split
by successive SVDs (an *operator-Schmidt decomposition*) into a small chain of
local factors, and all the terms are assembled in a single MPO with the usual
finite-state-machine structure: every term that spans a bond contributes its
own set of channels to that bond.

The resulting bond dimension is ``2 + sum of the operator-Schmidt ranks of the
terms that cross the bond``. It is not minimal for Hamiltonians with many
long-range terms, but it is exact and needs no assumption about the terms.

Convention: ``W[p]`` has indices ``(left, right, row, column)``, with
``<s| O |t> = W[..., s, t]``. The first tensor has ``left = 1`` and the last
one ``right = 1``.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = ["build_mpo", "mpo_to_dense", "term_to_factors"]

_START = "S"
_END = "E"


def term_to_factors(
    matrix: np.ndarray, dims: Sequence[int], tol: float = 1e-12
) -> List[np.ndarray]:
    """Operator-Schmidt decomposition of a ``k``-site operator.

    Parameters
    ----------
    matrix : ndarray
        Dense matrix of shape ``(D, D)``, ``D = prod(dims)``, in the tensor
        product order of ``dims``.
    dims : sequence of int
        Local dimensions of the ``k`` sites.
    tol : float, optional
        Relative threshold of the singular values kept at every cut.

    Returns
    -------
    list of ndarray
        ``k`` tensors ``B[j]`` of shape ``(r_{j-1}, d_j, d_j, r_j)`` with
        ``r_{-1} = r_{k-1} = 1`` such that the operator is the sum over the
        internal indices of the product of the ``B[j]``.
    """
    dims = [int(d) for d in dims]
    k = len(dims)
    tensor = np.asarray(matrix, dtype=complex).reshape(tuple(dims) * 2)
    order = [axis for j in range(k) for axis in (j, k + j)]
    rest = tensor.transpose(order).reshape(1, -1)
    factors: List[np.ndarray] = []
    r_prev = 1
    for dim in dims[:-1]:
        mat = rest.reshape(r_prev * dim * dim, -1)
        u_mat, s_vals, vh_mat = np.linalg.svd(mat, full_matrices=False)
        keep = max(1, int(np.sum(s_vals > tol * s_vals[0])))
        factors.append(u_mat[:, :keep].reshape(r_prev, dim, dim, keep))
        rest = s_vals[:keep, None] * vh_mat[:keep]
        r_prev = keep
    factors.append(rest.reshape(r_prev, dims[-1], dims[-1], 1))
    return factors


def build_mpo(
    terms: Sequence[Tuple[Sequence[int], np.ndarray]],
    dims: Sequence[int],
    zero_tol: float = 1e-14,
) -> List[np.ndarray]:
    """Assemble the MPO of a sum of few-body terms.

    Parameters
    ----------
    terms : sequence of (positions, matrix)
        Every term is given by the **ascending** positions along the chain
        where it acts and its dense matrix, written in the tensor-product
        order of those positions.
    dims : sequence of int
        Local dimension at every position.
    zero_tol : float, optional
        Terms whose matrix has all its entries below this value are skipped.

    Returns
    -------
    list of ndarray
        The MPO tensors ``W[p]`` of shape ``(D_p, D_{p+1}, d_p, d_p)``.
    """
    dims = [int(d) for d in dims]
    n_sites = len(dims)

    # Decompose terms and find the channels crossing every bond.
    decomposed = []
    # channels[b]: keys of the channels present in bond b (between b-1 and b)
    channels: List[List[object]] = [[] for _ in range(n_sites + 1)]
    channels[0] = [_START]
    channels[n_sites] = [_END]
    for bond in range(1, n_sites):
        channels[bond] = [_START, _END]

    for t_idx, (positions, matrix) in enumerate(terms):
        positions = tuple(int(p) for p in positions)
        matrix = np.asarray(matrix, dtype=complex)
        if np.max(np.abs(matrix), initial=0.0) < zero_tol:
            continue
        if list(positions) != sorted(set(positions)):
            raise ValueError("term positions must be strictly ascending")
        factors = term_to_factors(matrix, [dims[p] for p in positions])
        decomposed.append((t_idx, positions, factors))
        for j in range(len(positions) - 1):
            rank = factors[j].shape[3]
            for bond in range(positions[j] + 1, positions[j + 1] + 1):
                channels[bond].extend((t_idx, j, alpha) for alpha in range(rank))

    index: List[Dict[object, int]] = [
        {key: i for i, key in enumerate(keys)} for keys in channels
    ]
    mpo = [
        np.zeros(
            (len(channels[p]), len(channels[p + 1]), dims[p], dims[p]), dtype=complex
        )
        for p in range(n_sites)
    ]

    # Identity channels
    for pos in range(n_sites):
        ident = np.eye(dims[pos])
        for key in (_START, _END):
            if key in index[pos] and key in index[pos + 1]:
                mpo[pos][index[pos][key], index[pos + 1][key]] = ident

    # Terms
    for t_idx, positions, factors in decomposed:
        k = len(positions)
        if k == 1:
            pos = positions[0]
            mpo[pos][index[pos][_START], index[pos + 1][_END]] += factors[0][0, :, :, 0]
            continue
        for j, pos in enumerate(positions):
            fac = factors[j]
            for alpha in range(fac.shape[0]):
                left_key = _START if j == 0 else (t_idx, j - 1, alpha)
                if j == 0 and alpha > 0:
                    break
                for beta in range(fac.shape[3]):
                    right_key = _END if j == k - 1 else (t_idx, j, beta)
                    mpo[pos][index[pos][left_key], index[pos + 1][right_key]] += fac[
                        alpha, :, :, beta
                    ]
            # Identity along the channels that jump over the sites in between
            if j < k - 1:
                for mid in range(pos + 1, positions[j + 1]):
                    ident = np.eye(dims[mid])
                    for alpha in range(fac.shape[3]):
                        key_tuple = (
                            t_idx,
                            j,
                            alpha,
                        )
                        mpo[mid][index[mid][key_tuple], index[mid + 1][key]] += ident
    return mpo


def mpo_to_dense(mpo: Sequence[np.ndarray]) -> np.ndarray:
    """Contract an MPO into a dense matrix. Exponential in the length."""
    result = mpo[0][0]  # (D_right, row, col)
    for tensor in mpo[1:]:
        result = np.einsum("aIJ,abst->bIsJt", result, tensor, optimize=True)
        shape = result.shape
        result = result.reshape(shape[0], shape[1] * shape[2], shape[3] * shape[4])
    return result[0]
