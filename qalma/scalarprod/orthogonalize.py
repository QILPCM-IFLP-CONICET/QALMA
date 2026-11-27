"""Orthogonalization of bases of operators.

Routines to orthogonalize sets of operators with respect to a scalar product
(see :mod:`qalma.scalarprod.build`), to build bases of hermitian operators,
and to compute the components of an operator in an orthonormal basis.

"""

# from datetime import datetime
from typing import Callable, List

import numpy as np
from numpy.linalg import (
    LinAlgError as _LinAlgError,
    cholesky as _cholesky,
    inv as _inv,
    norm as _norm,
    svd as _svd,
)
from scipy.linalg import sqrtm as _sqrtm

from qalma.operators import Operator
from qalma.scalarprod.gram import gram_matrix

__all__ = ["build_hermitian_basis", "orthogonalize_basis", "operator_components"]


def build_hermitian_basis(basis, sp=lambda x, y: ((x.dag() * y).tr())):
    r"""Build a basis of hermitian operators.

    Build an orthonormal basis of hermitian operators that generates a set of
    operators, and the coefficients for the expansion of the original operators
    in terms of the new basis.

    Each hermitian operator of ``basis`` contributes with itself. A non-hermitian
    operator :math:`b` contributes with its hermitian and anti-hermitian parts,
    :math:`b+b^\dagger` and :math:`i(b-b^\dagger)`. The hermitian operators
    obtained in this way are then orthonormalized with respect to ``sp``,
    dropping the linearly dependent ones. Null operators, and the null hermitian
    or anti-hermitian parts of the operators, do not contribute to the new basis.

    Parameters
    ----------
    basis: List[Operator]
        The operators to be expanded.
    sp: Callable, optional
        The scalar product used to orthonormalize the new basis. By default, the
        Hilbert-Schmidt scalar product :math:`\mathrm{Tr}(x^\dagger y)`.

    Returns
    -------
    new_basis: List[Operator]
        An orthonormal basis of hermitian operators that spans the hermitian
        and anti-hermitian parts of the elements of ``basis``.
    coeffs: numpy.ndarray
        An array of shape ``(len(basis), len(new_basis))``, such that
        ``basis[i] = sum(coeffs[i, j] * new_basis[j] for j)``. The row of a
        null operator is zero.

    """
    # First, find a basis of hermitian operators that generates
    # basis.
    new_basis = []
    indx = 0
    # indices is a list that keeps the connection between the original
    # basis and the hermitian basis
    indices = []
    for b in basis:
        indices.append([])
        if b.isherm:
            if b:
                new_basis.append(b)
                indices[-1].append(
                    (
                        indx,
                        1.0,
                    )
                )
                indx += 1
        else:
            op = b + b.dag()  # .simplify()
            if op:
                new_basis.append(op)
                indices[-1].append(
                    (
                        indx,
                        0.5,
                    )
                )
                indx += 1
            op = 1j * b - 1j * b.dag()  # .simplify()
            if op:
                new_basis.append(1j * (b - b.dag()))
                indices[-1].append(
                    (
                        indx,
                        -0.5j,
                    )
                )
                indx += 1

    if not new_basis:
        # All the operators are null.
        return [], np.zeros((len(basis), 0))

    # Now, we work with the hermitian basis.
    # The first step is to build the Gram's matrix
    gram_mat = gram_matrix(new_basis, sp)

    # Now, we construct the SVD of the Gram's matrix
    u_mat, s_mat, vd_mat = _svd(gram_mat, full_matrices=False, hermitian=True)
    # And find a change of basis to an orthonormalized basis
    t = np.array([row * s ** (-0.5) for row, s in zip(vd_mat, s_mat) if s > 1e-10])
    # and build the hermitian, orthogonalized basis
    new_basis = [
        sum(c * op for c, op in zip(row, new_basis)) for row, s in zip(t, s_mat)
    ]
    # Then, we build the change back to the hermitian basis
    q = np.array([row * s ** (0.5) for row, s in zip(u_mat.T, s_mat) if s > 1e-10]).T
    # Finally, we apply the change of basis to the original (non-hermitian)
    # basis
    # (null operators have no entries in ``row``, and give a zero row).
    coeffs = np.array(
        [
            sum((factor * q[idx] for idx, factor in row), np.zeros(q.shape[1]))
            for row in indices
        ]
    )

    return new_basis, coeffs


def operator_components(op: Operator, orthogonal_basis: List[Operator], sp: Callable):
    r"""Compute the components of ``op`` in an orthonormal basis.

    Get the components of the orthogonal projection of an operator onto the span
    of an orthonormal basis, with respect to a scalar product:

    .. math::

        c_i = \langle e_i, \mathrm{op}\rangle,

    so that :math:`\sum_i c_i e_i` is the projection of ``op``. If ``op`` is in
    the span of the basis, the projection is ``op`` itself.

    Parameters
    ----------
    op: Operator
        The operator to be projected.
    orthogonal_basis: List[Operator]
        A list of operators :math:`e_i`, orthonormal with respect to ``sp``,
        like the ones returned by :func:`~qalma.scalarprod.orthogonalize.orthogonalize_basis`. If the elements
        are orthogonal but not normalized, the results are the components
        multiplied by the norm of each element.
    sp: Callable
        A callable that defines the scalar product function between
        two operators. It must be antilinear in the first argument.

    Returns
    -------
    numpy.ndarray
        A NumPy array with the projection coefficients, where the i-th
        coefficient is the component of ``op`` along the i-th element of
        ``orthogonal_basis``.

    """
    return np.array([sp(op2, op) for op2 in orthogonal_basis])


def orthogonalize_basis(basis, sp: Callable, tol: float = 1.0e-5):
    """Orthogonalize a given basis of operators using the default method.

    The default method is Gram-Schmidt: the operators are processed in order.
    Each one is normalized, the projections over the previous elements of the
    result are subtracted, and the remainder is normalized again. The k-th
    element of the result is a linear combination of the first operators of
    ``basis``.

    The alternatives, based on the Cholesky decomposition and on the symmetric
    orthogonalization, are available as ``orthogonalize_basis_cholesky`` and
    ``orthogonalize_basis_svd`` in this module. Unlike Gram-Schmidt, they raise a
    ``LinAlgError`` if the basis is linearly dependent.

    Parameters
    ----------
    basis: List[Operator]
        A list of operators to be orthogonalized.
    sp: Callable
        A callable that defines the scalar product function between two
        operators.
    tol: float
        A tolerance value (default: 1e-5). An operator is discarded if its norm
        is below ``tol``, either initially (it is null) or after subtracting the
        projections over the previous operators (it is linearly dependent on
        them). Overlaps below ``tol`` are neglected.

    Returns
    -------
    orth_basis: List[Operator]
        A list of orthogonalized operators, normalized with respect
        to the scalar product `sp`. Null operators and operators that are linear
        combinations of the previous ones are discarded, so it can be
        shorter than ``basis``.

    """
    return orthogonalize_basis_gs(basis, sp, tol)


def orthogonalize_basis_gs(basis, sp: Callable, tol: float = 1.0e-5):
    """Orthogonalize basis using by Gram-Schmidt.

    Orthogonalizes a given basis of operators using a scalar product and the
    Gram-Schmidt method.

    The operators are processed in order. Each one is normalized, the projections
    over the previous elements of the result are subtracted, and the remainder is
    normalized again. The k-th element of the result is a linear combination of
    the first operators of ``basis``.

    Parameters
    ----------
    basis: List[Operator]
        A list of operators (or matrices) to be orthogonalized.
    sp: Callable
        A callable that defines the scalar product function between two
        operators.
    tol: float
        A tolerance value (default: 1e-5). An operator is discarded if its norm
        is below ``tol``, either initially (it is null) or after subtracting the
        projections over the previous operators (it is linearly dependent on
        them). Overlaps below ``tol`` are neglected.

    Returns
    -------
    orth_basis: List[Operator]
        A list of orthogonalized operators, normalized with respect
        to the scalar product `sp`. It is shorter than ``basis`` if some
        operators were discarded.

    """
    orth_basis: list = []
    for op_orig in basis:
        op_norm: float = abs(sp(op_orig, op_orig)) ** 0.5
        if op_norm < tol:
            continue
        changed = False
        new_op = op_orig / op_norm
        for prev_op in orth_basis:
            overlap = sp(prev_op, new_op)
            if abs(overlap) > tol:
                new_op -= prev_op * overlap
                changed = True
        if changed:
            op_norm = abs(sp(new_op, new_op)) ** 0.5
            if op_norm < tol:
                continue
            new_op = new_op / op_norm
        orth_basis.append(new_op)
    return orth_basis


def orthogonalize_basis_cholesky(basis, sp: Callable, tol: float = 1.0e-5):
    r"""Orthogonalize an operator basis using the Cholesky method.

    Orthogonalizes a given basis of operators using a scalar product and the
    Cholesky decomposition of the Gram matrix :math:`G=LL^T`. The operators of
    the result are
    :math:`e_i=\sum_{s\le i} (L^{-1})_{is}\, b_s`: as in Gram-Schmidt, each
    one is a linear combination of the first operators of ``basis``.

    Parameters
    ----------
    basis: List[Operator]
        A list of operators (or matrices) to be orthogonalized.
    sp: Callable
        A callable that defines the scalar product function between two
        operators.
    tol: float
        A tolerance value (default: 1e-5). It is used to detect linearly
        dependent operators (diagonal elements of :math:`L` below ``tol``), and
        to verify the orthonormality of the resulting basis.

    Returns
    -------
    orth_basis: List[Operator]
        A list of orthogonalized operators, normalized with respect
        to the scalar product `sp`. It has the same length as ``basis``.

    Raises
    ------
    numpy.linalg.LinAlgError:
        If the operators in ``basis`` are linearly dependent.
    AssertionError:
        If the orthogonalized basis does not satisfy
        orthonormality within the specified tolerance.

    """
    local_basis = basis

    # Compute the Gram matrix for the given basis
    gram = gram_matrix(basis=local_basis, sp=sp)

    try:
        l_gram = _cholesky(gram)
    except _LinAlgError as exc:
        raise _LinAlgError(
            "The Gram matrix is not positive definite: the basis is "
            "linearly dependent."
        ) from exc
    if np.abs(np.diag(l_gram)).min(initial=np.inf) < tol:
        raise _LinAlgError(
            "The Gram matrix is singular within the tolerance: the basis is "
            "linearly dependent."
        )

    l_inv = _inv(l_gram)

    # Construct the orthogonalized basis by linear combinations of
    # the original basis
    orth_basis = [
        sum(local_basis[s] * l_inv[i, s] for s in range(i + 1))
        for i in range(len(local_basis))
    ]
    # Verify the orthogonality by checking that the Gram matrix is
    # approximately the identity matrix
    assert (
        _norm(gram_matrix(basis=orth_basis, sp=sp) - np.identity(len(orth_basis))) < tol
    ), f"Error: Basis not correctly orthogonalized\n{gram_matrix(basis=orth_basis, sp=sp)}"

    return orth_basis


def orthogonalize_basis_svd(basis, sp: Callable, tol: float = 1.0e-5):
    r"""Orthogonalize an operator basis using the symmetric method.

    Orthogonalizes a given basis of operators using a scalar product and the
    symmetric (Löwdin) orthogonalization. The operators of the result are
    :math:`e_i=\sum_j (G^{-1/2})_{ji}\, b_j`, where :math:`G` is the Gram matrix.
    Among the orthonormal bases that can be built from linear combinations of
    ``basis``, it is the closest to the original one, in the least-squares
    sense. The matrix
    :math:`G^{-1/2}` is computed with :func:`scipy.linalg.sqrtm`.

    Parameters
    ----------
    basis: List[Operator]
        A list of operators (or matrices) to be orthogonalized.
    sp: Callable
        A callable that defines the scalar product function between two
        operators.
    tol: float
        A tolerance value (default: 1e-5) to verify the orthonormality of the
        resulting basis.

    Returns
    -------
    orth_basis: List[Operator]
        A list of orthogonalized operators, normalized with respect
        to the scalar product `sp`. It has the same length as ``basis``.

    Raises
    ------
    numpy.linalg.LinAlgError:
        If the operators in ``basis`` are linearly dependent.
    AssertionError:
        If the orthogonalized basis does not satisfy
        orthonormality within the specified tolerance.

    """
    local_basis = basis

    # Compute the inverse Gram matrix for the given basis
    inv_gram_matrix = _inv(gram_matrix(basis=local_basis, sp=sp))

    # Construct the orthogonalized basis by linear combinations of
    # the original basis
    orth_basis = [
        sum(
            _sqrtm(inv_gram_matrix)[j][i] * local_basis[j]
            for j in range(len(local_basis))
        )
        for i in range(len(local_basis))
    ]

    # Verify the orthogonality by checking that the Gram matrix is
    # approximately the identity matrix
    assert (
        _norm(gram_matrix(basis=orth_basis, sp=sp) - np.identity(len(orth_basis))) < tol
    ), "Error: Basis not correctly orthogonalized"

    return orth_basis
