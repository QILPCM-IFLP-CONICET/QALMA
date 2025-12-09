"""
Functions implementing the Covariance Scalar Product
"""

# from datetime import datetime
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from numpy import real
from numpy.typing import NDArray

from qalma.operators import Operator, iterable_to_operator
from qalma.operators.functions import anticommutator
from qalma.operators.states import ProductDensityOperator
from qalma.settings import QALMA_TOLERANCE

#  ### Functions that build the scalar products ###


class CovariantScalarProductFunction:
    """
    A callable object that computes the Covariance scalar
    product of two operators, relative to a given
    reference state sigma.
    """

    def __init__(self, state):
        if hasattr(state, "to_product_state"):
            state = state.to_product_state()
        self.sigma = state

    def __call__(self, op1, op2):
        sigma = self.sigma

        if hasattr(sigma, "terms"):
            return compute_cov_mix_sp(self.sigma, op1, op2)

        if isinstance(sigma, ProductDensityOperator):
            result = compute_cov_prod_sp(self.sigma, op1, op2)
            return result

        # The remaining code computes the scalar product for generic states.
        if op1 is op2:
            op1 = op1.simplify()
            op1_herm = op1.isherm
            return (
                abs(sigma.expect(op1 * op1))
                if op1_herm
                else abs(0.5 * sigma.expect(anticommutator(op1.dag(), op1).simplify()))
            )

        op1 = op1.simplify()
        op1_herm = op1.isherm
        op2 = op2.simplify()
        op2_herm = op2.isherm

        if op1_herm:
            if op2_herm:
                o1o2 = (op1 * op2).simplify()
                return real(sigma.expect(o1o2))
            op1_dag = op1
        else:
            op1_dag = op1.dag()
        if op1_dag is op2:
            return sigma.expect((op1_dag * op2).simplify())
        return 0.5 * sigma.expect(anticommutator(op1_dag, op2).simplify())

    def compute_cross_gram_matrix(
        self, basis_1: Tuple[Operator, ...], basis_2: Tuple[Operator, ...]
    ) -> NDArray:
        """
        Compute the cross gram matrix for basis basis_1 and basis_2.
        Operators are assumed to be hermitician.
        """
        sigma = self.sigma
        basis_1_size = len(basis_1)
        basis_2_size = len(basis_2)
        cross_gram_matrix = np.zeros(
            (
                basis_1_size,
                basis_2_size,
            ),
            dtype=float,
        )
        if isinstance(sigma, ProductDensityOperator):
            for i in range(basis_1_size):
                for j in range(basis_2_size):
                    cross_gram_matrix[i, j] = self(basis_1[i], basis_2[j])
            return cross_gram_matrix
        if hasattr(sigma, "terms"):
            return sum(
                (
                    CovariantScalarProductFunction(term).compute_cross_gram_matrix(
                        basis_1, basis_2
                    )
                    * term.prefactor
                    for term in sigma.terms
                ),
                cross_gram_matrix,
            )

        operators_dict = {}
        for i in range(basis_1_size):
            for j in range(basis_2_size):
                operators_dict[
                    (
                        i,
                        j,
                    )
                ] = (basis_1[i] * basis_2[j]).simplify()

        coeffs_dict = self.sigma.expect(operators_dict)
        cross_gram_matrix = np.zeros(
            (
                basis_1_size,
                basis_2_size,
            ),
            dtype=float,
        )
        for pos, val in coeffs_dict.items():
            i, j = pos
            cross_gram_matrix[i, j] = np.real(val)
        return cross_gram_matrix

    def compute_gram_matrix(self, basis: Tuple[Operator, ...]) -> NDArray:
        """
        Compute the gram matrix associated to the hermitician operators
        specified in `basis`.

        """
        basis_size = len(basis)
        sigma = self.sigma
        gram_matrix = np.zeros(
            (
                basis_size,
                basis_size,
            ),
            dtype=float,
        )
        if isinstance(sigma, ProductDensityOperator):
            for i in range(basis_size):
                for j in range(basis_size):
                    if i > j:
                        continue
                    value = self(basis[i], basis[j])
                    gram_matrix[i, j] = value
                    if i < j:
                        gram_matrix[j, i] = value
            return gram_matrix

        if hasattr(sigma, "terms"):
            return sum(
                (
                    CovariantScalarProductFunction(term).compute_gram_matrix(basis)
                    * term.prefactor
                    for term in sigma.terms
                ),
                gram_matrix,
            )

        operators_dict = {}
        for i in range(basis_size):
            for j in range(i + 1):
                operators_dict[
                    (
                        i,
                        j,
                    )
                ] = (basis[i] * basis[j]).simplify()

        coeffs_dict = self.sigma.expect(operators_dict)
        gram_matrix = np.zeros(
            (
                basis_size,
                basis_size,
            )
        )

        for pos, val in coeffs_dict.items():
            i, j = pos
            if i == j:
                gram_matrix[i, i] = np.abs(val)
            else:
                val = np.real(val)
                gram_matrix[i, j] = gram_matrix[j, i] = val
        return gram_matrix


def compute_cov_sp_sqnorm(rho: ProductDensityOperator, op1: Operator) -> complex:
    """
    Efficiently computes the squared norm associated to the  covariant scalar
    product for Hermitian operators op1 and op2 when the state rho is
    a ProductDensityOperator.


    Parameters
    ----------
    rho : ProductDensityOperator
        State defining the scalar product.
    op1 : Operator
        operator.

    Returns
    -------
    complex
        covar-induced norm.

    """
    result = 0.0
    av_1: Dict[int, complex] = {}
    op1 = op1.simplify()
    terms_1 = op1.terms if hasattr(op1, "terms") else [op1]
    for i, t1 in enumerate(terms_1):
        for j, t2 in enumerate(terms_1):
            if j > i:
                continue
            if i == j:
                result += 0.5 * cast(
                    float,
                    np.real(cast(complex, rho.expect(t1.dag() * t2 + t2 * t1.dag()))),
                )
            else:
                overlap = t1.acts_over().intersection(t2.acts_over())
                if overlap:
                    t1_red = t1.reduce(overlap, rho)
                    t2_red = t2.reduce(overlap, rho)
                    contrib = cast(
                        float,
                        np.real(
                            cast(
                                complex,
                                rho.expect(
                                    t1_red.dag() * t2_red + t2_red * t1_red.dag()
                                ),
                            )
                        ),
                    )
                else:
                    if i not in av_1:
                        av_1[i] = cast(complex, rho.expect(t1))
                    if j not in av_1:
                        av_1[j] = cast(complex, rho.expect(t2))

                    contrib = 2.0 * cast(
                        float,
                        np.real(np.conj(av_1[i]) * av_1[j]),
                    )
                result += contrib
    return np.abs(result)


def compute_cov_sp_prod(
    rho: ProductDensityOperator, op1: Operator, op2: Operator
) -> complex:
    """
    Efficiently computes the covariant scalar product for Hermitian operators
    op1 and op2 when the state rho is a ProductDensityOperator.
    """
    result: complex = 0.0
    av_1: Dict[int, complex] = {}
    av_2: Dict[int, complex] = {}
    if op1 is op2:
        return compute_cov_sp_sqnorm(rho, op1)

    op1 = op1.simplify()
    op2 = op2.simplify()
    for i, t1 in enumerate(op1.terms if hasattr(op1, "terms") else [op1]):
        for j, t2 in enumerate(op2.terms if hasattr(op2, "terms") else [op2]):
            if i == j:
                contrib = 0.5 * cast(complex, rho.expect(t1.dag() * t2 + t2 * t1.dag()))
            else:
                overlap = t1.acts_over().intersection(t2.acts_over())
                if not overlap:
                    if i not in av_1:
                        av_1[i] = cast(complex, rho.expect(t1))
                    if j not in av_2:
                        av_2[j] = cast(complex, rho.expect(t2))
                    contrib = np.real(np.conj(av_1[i]) * av_2[j])
                else:
                    t1_red = t1.reduce(overlap, rho)
                    t2_red = t2.reduce(overlap, rho)
                    contrib = 0.5 * cast(
                        complex,
                        rho.expect(t1_red.dag() * t2_red + t2_red * t1_red.dag()),
                    )
            result += contrib

    return np.real(result)


def compute_cov_mix_sp(rho, op1: Operator, op2: Operator) -> complex:
    """
    Compute the covariance scalar product relative to a
    mixture state.
    """
    return sum(
        CovariantScalarProductFunction(term)(op1, op2) * term.prefactor
        for term in rho.terms
    )


def compute_cov_sqnorm(rho: ProductDensityOperator, op1: Operator) -> complex:
    """
    Compute the square of the covar operator norm of `op1` associated
    to the state `rho`.

    Parameters
    ----------
    rho : ProductDensityOperator
        The state.
    op1 : Operator
        operator.

    Returns
    -------
    complex
        the covar-induced operator norm.

    """
    if op1.isherm:
        return _compute_cov_prod_normsq_h(rho, op1)
    return _compute_cov_prod_sp_hg(rho, op1.dag(), op1)


def compute_cov_prod_sp(
    rho: ProductDensityOperator, op1: Operator, op2: Operator
) -> complex:
    """
    Compute the covariance scalar product
    associated to a product state.
    """
    if op1.isherm:
        if op1 is op2:
            result = _compute_cov_prod_normsq_h(rho, op1)
            return result
        if op2.isherm:
            result = _compute_cov_prod_sp_hh(rho, op1, op2)
            return result
        return _compute_cov_prod_sp_hg(rho, op1, op2)
    if op2.isherm:
        op1, op2 = op2, op1
        return np.conj(_compute_cov_prod_sp_hg(rho, op1, op2))
    return _compute_cov_prod_sp_hg(rho, op1.dag(), op2)


def compute_list_terms_with_norms(
    rho: ProductDensityOperator, terms=Tuple[Operator, ...]
) -> List[Tuple[float, Operator]]:
    """
    Return a list of terms that decompose operator with their corresponding squared
    covar norms relative to rho, sorted in norm decreasing order.
    """
    return sorted(
        [
            (
                np.real(cast(complex, rho.expect(term.dag() * term))),
                term,
            )
            for term in terms
        ],
        key=lambda x: -x[0],
    )


def remove_under_tolerance_terms(
    terms_with_norms=Tuple[Operator, ...], tol=QALMA_TOLERANCE
) -> List[Tuple[float, Operator]]:
    """
    Return a list of terms that decompose operator with their corresponding squared
    covar norms relative to rho, sorted in norm decreasing order.
    """
    n = len(terms_with_norms)
    error = 0.0
    while error < tol and n > 1:
        n = n - 1
        last_norm = terms_with_norms[n][0] ** 0.5
        error += last_norm

    return terms_with_norms[:n]


def trim_terms_by_tolerance(
    rho: ProductDensityOperator, operator: Operator, tol: float = QALMA_TOLERANCE
) -> Operator:
    """Compute squared norms of each term, and remove those with smaller norm"""

    isherm = operator.isherm
    system = operator.system
    terms = operator.flat().terms if hasattr(operator, "terms") else (operator,)
    terms_with_norms: List[Tuple[float, Operator]] = compute_list_terms_with_norms(
        rho, terms
    )
    terms_with_norms = remove_under_tolerance_terms(terms_with_norms, tol)
    return iterable_to_operator(
        (term[1] for term in terms_with_norms), system, isherm=isherm
    )


def _compute_cov_prod_normsq_h(
    rho: ProductDensityOperator, op1: Operator, tol: float = QALMA_TOLERANCE
) -> float:
    """
    Compute the square of the induced norm by
    the covariance scalar product associated to
    a product state for hermitician operators.
    """
    result: float = 0.0
    av_cache: Dict[Operator, complex] = {}
    op1 = op1.simplify().flat()
    terms_1 = op1.terms if hasattr(op1, "terms") else [op1]
    norms_sq = [np.real(cast(complex, rho.expect(t1.dag() * t1))) for t1 in terms_1]

    for i, t1 in enumerate(terms_1):
        for j, t2 in enumerate(terms_1):
            if j > i:
                continue
            if i == j:
                result += norms_sq[i]
            else:
                result += 2 * _term_sp_cov_prod_h(rho, t1, t2, av_cache)
    result = np.abs(result)
    return result


def _compute_cov_prod_normsq_g(
    rho: ProductDensityOperator, op1: Operator, tol: float = QALMA_TOLERANCE
) -> float:
    """
    Compute the square of the induced norm by
    the covariance scalar product associated to
    a product state for general operators.
    """
    result: float = 0.0
    av_cache: Dict[Operator, complex] = {}
    op1 = op1.simplify().flat()
    terms_1 = op1.terms if hasattr(op1, "terms") else [op1]

    for i, t1 in enumerate(terms_1):
        for j, t2 in enumerate(terms_1):
            if j > i:
                continue
            if i == j:
                result += np.real(_term_sp_cov_prod_g(rho, t1, t2, av_cache))
            else:
                result += 2 * np.real(_term_sp_cov_prod_g(rho, t1, t2, av_cache))
    return result


def _expect_value_cache(
    rho: ProductDensityOperator,
    op: Operator,
    cache: Optional[Dict[Operator, complex]] = None,
) -> complex:
    if cache is None:
        return cast(complex, rho.expect(op))
    try:
        return cache[op]
    except KeyError:
        cache[op] = val = cast(complex, rho.expect(op))
        return val


def _term_sp_cov_prod_g(
    rho: ProductDensityOperator, op1, op2, av_cache=None
) -> complex:
    """
    Compute the cov scalar product assocated to rho for
    two hermitician single-term operators.
    """
    overlap = op1.acts_over().intersection(op2.acts_over())
    if overlap:
        op1_red = op1.reduce(overlap, rho).dag()
        op2_red = op2.reduce(overlap, rho)
        return 0.5 * (cast(complex, rho.expect(op1_red * op2_red + op2_red * op1_red)))

    av_1 = _expect_value_cache(rho, op1, av_cache)
    av_2 = _expect_value_cache(rho, op2, av_cache)
    return np.conj(av_1) * av_2


def _term_sp_cov_prod_h(rho: ProductDensityOperator, op1, op2, av_cache=None) -> float:
    """
    Compute the cov scalar product assocated to rho for
    two hermitician single-term operators.
    """
    overlap = op1.acts_over().intersection(op2.acts_over())
    if overlap:
        op1_red = op1.reduce(overlap, rho)
        op2_red = op2.reduce(overlap, rho)
        return np.real(cast(complex, rho.expect(op1_red * op2_red)))

    av_1 = _expect_value_cache(rho, op1, av_cache)
    av_2 = _expect_value_cache(rho, op2, av_cache)
    return np.real(av_1 * av_2)


def _compute_cov_prod_sp_g(
    rho: ProductDensityOperator,
    op1: Operator,
    op2: Operator,
    tol: float = QALMA_TOLERANCE,
) -> complex:
    """
    Compute the covariance scalar product
    associated to a product state for two
    general operators.
    """
    error: float = 0.0
    result: complex = 0.0
    av_cache: Dict[Operator, complex] = {}

    terms_1 = op1.simplify().flat().terms if hasattr(op1, "terms") else (op1,)
    terms_2 = op2.simplify().flat().terms if hasattr(op2, "terms") else (op2,)

    terms_norms_1 = compute_list_terms_with_norms(rho, terms_1)
    terms_norms_2 = compute_list_terms_with_norms(rho, terms_2)
    rem_num_terms = len(terms_norms_1) * len(terms_norms_2)

    for i, (norm_1, t1) in enumerate(reversed(terms_norms_1)):
        for j, (norm_2, t2) in enumerate(reversed(terms_norms_2)):
            # If the norm of (t1,t2) is small enough, we can
            # skip the term without harm:
            mag = norm_1 * norm_2
            if (tol - error) > mag * rem_num_terms:
                error += mag
                continue
            rem_num_terms -= 1
            # Determine the overlap
            result += _term_sp_cov_prod_g(rho, op1, op2, av_cache)
    return result


def _compute_cov_prod_sp_h(
    rho: ProductDensityOperator,
    op1: Operator,
    op2: Operator,
    tol: float = QALMA_TOLERANCE,
) -> float:
    """
    Compute the covariance scalar product
    associated to a product state for two
    hermitician operators.
    """
    error: float = 0.0
    result: float = 0.0
    av_cache: Dict[Operator, complex] = {}

    terms_1 = op1.simplify().flat().terms if hasattr(op1, "terms") else (op1,)
    terms_2 = op2.simplify().flat().terms if hasattr(op2, "terms") else (op2,)

    terms_norms_1 = compute_list_terms_with_norms(rho, terms_1)
    terms_norms_2 = compute_list_terms_with_norms(rho, terms_2)
    rem_num_terms = len(terms_norms_1) * len(terms_norms_2)

    for i, (norm_1, t1) in enumerate(reversed(terms_norms_1)):
        for j, (norm_2, t2) in enumerate(reversed(terms_norms_2)):
            # If the norm of (t1,t2) is small enough, we can
            # skip the term without harm:
            mag = norm_1 * norm_2
            if (tol - error) > mag * rem_num_terms:
                error += mag
                continue
            rem_num_terms -= 1
            # Determine the overlap
            result += _term_sp_cov_prod_h(rho, op1, op2, av_cache)
    return result


def _compute_cov_prod_sp_hg(
    rho: ProductDensityOperator, op1: Operator, op2: Operator
) -> complex:
    """
    Compute the covariance scalar product
    associated to a product state for
    an hermitician operator and a general operator.
    """
    result: complex = 0.0
    av_1: Dict[int, complex] = {}
    av_2: Dict[int, complex] = {}

    op1 = op1.simplify()
    op2 = op2.simplify()
    terms_1 = op1.terms if hasattr(op1, "terms") else [op1]
    terms_2 = op2.terms if hasattr(op2, "terms") else [op2]
    for i, t1 in enumerate(terms_1):
        for j, t2 in enumerate(terms_2):
            overlap = t1.acts_over().intersection(t2.acts_over())
            if not overlap:
                if i not in av_1:
                    av_1[i] = cast(complex, rho.expect(t1))
                if j not in av_2:
                    av_2[j] = cast(complex, rho.expect(t2))
                result += av_1[i] * av_2[j]
            else:
                t1_red = t1.reduce(overlap, rho)
                t2_red = t2.reduce(overlap, rho)
                result += (
                    cast(
                        complex,
                        rho.expect(t1_red * t2_red + t2_red * t1_red),
                    )
                    * 0.5
                )
    return result


def _compute_cov_prod_sp_hh(
    rho: ProductDensityOperator,
    op1: Operator,
    op2: Operator,
    tol: float = QALMA_TOLERANCE,
) -> float:
    """
    Compute the covariance scalar product
    associated to a product state for two
    hermitician operators.
    """
    error: float = 0.0
    result: float = 0.0
    av_1: Dict[int, complex] = {}
    av_2: Dict[int, complex] = {}
    cache_reduce_1: Dict[Tuple[int, frozenset], Operator] = {}
    cache_reduce_2: Dict[Tuple[int, frozenset], Operator] = {}

    op1 = op1.simplify()
    op2 = op2.simplify()

    terms_norms_1 = sorted(
        (
            (
                np.abs(cast(complex, rho.expect(term.dag() * term))) ** 0.5,
                term,
            )
            for term in (op1.terms if hasattr(op1, "terms") else (op1,))
        ),
        key=lambda x: x[0],
    )
    terms_norms_2 = sorted(
        (
            (
                abs(cast(complex, rho.expect(term.dag() * term))) ** 0.5,
                term,
            )
            for term in (op2.terms if hasattr(op2, "terms") else (op2,))
        ),
        key=lambda x: x[0],
    )

    rem_num_terms = len(terms_norms_1) * len(terms_norms_2)

    for i, (norm_1, t1) in enumerate(terms_norms_1):
        for j, (norm_2, t2) in enumerate(terms_norms_2):
            # If the norm of (t1,t2) is small enough, we can
            # skip the term without harm:
            mag = norm_1 * norm_2
            if (tol - error) > mag * rem_num_terms:
                error += mag
                continue
            rem_num_terms -= 1
            # Determine the overlap
            overlap = t1.acts_over().intersection(t2.acts_over())
            if not overlap:
                if i not in av_1:
                    av_1[i] = cast(complex, rho.expect(t1))
                if j not in av_2:
                    av_2[j] = cast(complex, rho.expect(t2))
                result += np.real(av_1[i] * av_2[j])
            else:
                key = (i, overlap)
                if key in cache_reduce_1:
                    t1_red = cache_reduce_1[key]
                else:
                    t1_red = cache_reduce_1[key] = t1.reduce(overlap, rho)
                key = (j, overlap)
                if key in cache_reduce_2:
                    t2_red = cache_reduce_2[key]
                else:
                    t2_red = cache_reduce_2[key] = t2.reduce(overlap, rho)

                result += np.real(cast(complex, rho.expect(t1_red * t2_red)))
    return result
