"""Functions implementing the Covariance Scalar Product."""

# from datetime import datetime
from typing import Callable, Dict, Generator, List, Optional, Tuple, cast

import numpy as np
from numpy import real
from numpy.typing import NDArray

from qalma.operators import Operator, iterable_to_operator
from qalma.operators.functions import anticommutator
from qalma.operators.states import ProductDensityOperator
from qalma.settings import QALMA_TOLERANCE

#  ### Functions that build the scalar products ###


class ErrorCummulator:
    """
    Error Cummulator.

    A class to track the accumulated error introduced by discarding terms
    into a sum.
    """

    tol: float
    rem_terms: int
    error: float

    def __init__(self, tol, num_terms):
        self.tol = tol
        self.rem_terms = num_terms
        self.error = 0.0
        self.margin = tol

    def query(self, mag, steps=1) -> bool:
        """
        Check if a the magnitud of a term affects the norm of the sum.

        Check if mag times the remaining number of terms is below the
        difference between the tolerance and the accumulated error.
        """
        margin = self.margin
        result = margin > mag and margin > mag * self.rem_terms
        if result:
            self.error += mag
            self.margin -= mag
        self.rem_terms -= steps
        return result


class CovariantScalarProductFunction:
    """
    Covariant Scalar Product function.

    A callable object that computes the Covariance scalar product of two
    operators, relative to a given reference state sigma.
    """

    def __init__(self, state):
        if hasattr(state, "to_product_state"):
            state = state.to_product_state()
        self.sigma = state

    def __call__(self, op1, op2):
        """Evaluate the scalar product."""
        sigma = self.sigma

        if hasattr(sigma, "terms"):
            return compute_cov_mix_sp(
                self.sigma, op1.simplify().flat(), op2.simplify().flat()
            )

        if isinstance(sigma, ProductDensityOperator):
            result = compute_cov_prod_sp(
                self.sigma, op1.simplify().flat(), op2.simplify().flat()
            )
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
        """Compute the cross gram matrix for basis basis_1 and basis_2.

        Operators are assumed to be hermitian.
        """
        sigma = self.sigma
        basis_1 = tuple(b.simplify().flat() for b in basis_1)
        basis_2 = tuple(b.simplify().flat() for b in basis_2)
        isherm = all(all(b.isherm for b in basis) for basis in (basis_1, basis_2))
        basis_1_size = len(basis_1)
        basis_2_size = len(basis_2)
        cross_gram_matrix = np.zeros(
            (
                basis_1_size,
                basis_2_size,
            ),
            dtype=float if isherm else complex,
        )
        if isinstance(sigma, ProductDensityOperator):
            av_cache: Dict[Operator, complex] = {}

            def _sp(op1, op2):
                return compute_cov_prod_sp(self.sigma, op1, op2, av_cache)

            for i in range(basis_1_size):
                for j in range(basis_2_size):
                    cross_gram_matrix[i, j] = _sp(basis_1[i], basis_2[j])
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
        Compute the Gram's matrix for hermitian basis.

        Compute the gram matrix associated to the hermitian operators
        specified in `basis`.
        """
        basis_size = len(basis)
        sigma = self.sigma

        basis = tuple(b.simplify().flat() for b in basis)
        isherm = all(b.isherm for b in basis)
        gram_matrix = np.zeros(
            (
                basis_size,
                basis_size,
            ),
            dtype=float if isherm else complex,
        )
        if isinstance(sigma, ProductDensityOperator):
            av_cache: Dict[Operator, complex] = {}

            def _sp(op1, op2):
                return compute_cov_prod_sp(self.sigma, op1, op2, av_cache)

            for i in range(basis_size):
                for j in range(basis_size):
                    if i > j:
                        continue
                    value = _sp(basis[i], basis[j])
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


def compute_cov_mix_sp(rho, op1: Operator, op2: Operator) -> complex:
    """Compute the covariance scalar product relative to a mixture state."""
    return sum(
        CovariantScalarProductFunction(term)(op1, op2) * term.prefactor
        for term in rho.terms
    )


def compute_cov_sqnorm(
    rho: ProductDensityOperator,
    operator: Operator,
    av_cache: Optional[Dict[Operator, complex]] = None,
) -> complex:
    """
    Compute the square of the covariance norm of an operator.

    Compute the square of the covar operator norm of `op1` associated to the
    state `rho`.

    Parameters
    ----------
    rho : ProductDensityOperator
        The state that defines the scalar product.
    operator : Operator
        the operator argument of the norm.

    Returns
    -------
    complex
        the covar-induced operator norm.

    """
    if av_cache is None:
        av_cache = {}
    # TODO:
    # The following routines are optimized for handling products of
    # large sums of operators. Consider branch this to handle
    # special cases.
    if operator.isherm:
        return _compute_cov_prod_normsq(
            rho, operator, term_sp=_term_sp_cov_prod_h, av_cache=av_cache
        )
    return _compute_cov_prod_normsq(rho, operator, av_cache=av_cache)


def compute_cov_prod_sp(
    rho: ProductDensityOperator,
    op1: Operator,
    op2: Operator,
    av_cache: Optional[Dict[Operator, complex]] = None,
) -> complex:
    """Compute the covariance scalar product associated to a product state."""
    if av_cache is None:
        av_cache = {}
    if op1 is op2:
        if op1.isherm:
            return _compute_cov_prod_normsq(
                rho, op1, term_sp=_term_sp_cov_prod_h, av_cache=av_cache
            )
        return _compute_cov_prod_normsq(rho, op1, av_cache=av_cache)

    if op1.isherm:
        if op2.isherm:
            return _compute_cov_prod_sp_h(rho, op1, op2, av_cache=av_cache)
    return _compute_cov_prod_sp_g(rho, op1, op2, av_cache=av_cache)


def compute_list_terms_with_norms(
    rho: ProductDensityOperator, terms=Tuple[Operator, ...]
) -> List[Tuple[float, Operator]]:
    """
    Compute the list of terms and norms.

    Return a list of terms that decompose operator with their corresponding
    squared covar norms relative to rho, sorted in norm decreasing order.
    """
    return sorted(
        [
            (
                np.abs(cast(complex, rho.expect(term.dag() * term))) ** 0.5,
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
    Return a list of terms of the decomposition of an operator, ordered by norm.

    Return a list of terms that decompose operator with their corresponding
    squared covar norms relative to rho, sorted in norm decreasing order.
    """
    n = len(terms_with_norms)
    error = 0.0
    while error < tol and n > 1:
        n = n - 1
        last_norm = terms_with_norms[n][0]
        error += last_norm

    return terms_with_norms[:n]


def trim_terms_by_tolerance(
    rho: ProductDensityOperator, operator: Operator, tol: float = QALMA_TOLERANCE
) -> Operator:
    """
    Compute the square of covariance norm, and trim contributions under a threshold.

    Compute squared norms of each term, and remove those with smaller
    norm.
    """
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
    rho: ProductDensityOperator,
    op1: Operator,
    op2: Operator,
    av_cache: Optional[Dict[Operator, complex]] = None,
) -> complex:
    """
    Compute the covariance scalar product for single term hermitian operators.

    Compute the cov scalar product assocated to rho for two hermitian
    single-term operators.
    """
    overlap = op1.acts_over().intersection(op2.acts_over())
    if overlap:
        op1_red = op1.reduce(overlap, rho).dag()
        op2_red = op2.reduce(overlap, rho)
        return 0.5 * (cast(complex, rho.expect(op1_red * op2_red + op2_red * op1_red)))

    av_1 = _expect_value_cache(rho, op1, av_cache)
    av_2 = _expect_value_cache(rho, op2, av_cache)
    return np.conj(av_1) * av_2


def _term_sp_cov_prod_h(
    rho: ProductDensityOperator,
    op1: Operator,
    op2: Operator,
    av_cache: Optional[Dict[Operator, complex]] = None,
) -> float:
    """
    Compute the square of covariance norm for single term operators.

    Compute the cov scalar product assocated to rho for two hermitian
    single-term operators.
    """
    overlap = op1.acts_over().intersection(op2.acts_over())
    if overlap:
        op1_red = op1.reduce(overlap, rho)
        op2_red = op2.reduce(overlap, rho)
        return np.real(cast(complex, rho.expect(op1_red * op2_red)))

    av_1 = _expect_value_cache(rho, op1, av_cache)
    av_2 = _expect_value_cache(rho, op2, av_cache)
    return np.real(av_1 * av_2)


def _compute_cov_prod_normsq(
    rho: ProductDensityOperator,
    op1: Operator,
    tol: float = QALMA_TOLERANCE,
    term_sp: Callable = _term_sp_cov_prod_g,
    av_cache: Optional[Dict[Operator, complex]] = None,
) -> float:
    """
    Compute the square of covariance norm for general operators.

    Compute the square of the induced norm by the covariance scalar product
    associated to a product state for general operators.
    """
    terms_1 = op1.terms if hasattr(op1, "terms") else [op1]
    len_terms = len(terms_1)
    discard_check = ErrorCummulator(tol, len_terms * (len_terms - 1))
    terms_with_norms = compute_list_terms_with_norms(rho, terms_1)
    result: float = sum(n**2 for n, t in terms_with_norms)
    if av_cache is None:
        av_cache = {}

    def iterator():
        for j in range(len_terms - 1, -1, -1):
            for i in range(j - 1, -1, -1):
                yield (
                    (
                        i,
                        j,
                    )
                )

    for i, j in iterator():
        norm1, t1 = terms_with_norms[i]
        norm2, t2 = terms_with_norms[j]
        if discard_check.query(2 * norm1 * norm2, 2):
            continue
        result += 2 * np.real(term_sp(rho, t1, t2, av_cache))
    return result


def _compute_cov_prod_sp_h(
    rho: ProductDensityOperator,
    op1: Operator,
    op2: Operator,
    tol: float = QALMA_TOLERANCE,
    av_cache: Optional[Dict[Operator, complex]] = None,
) -> float:
    """
    Compute the covariance scalar product for hermitian operators.

    Compute the covariance scalar product associated to a product state for
    two hermitian operators.
    """
    if av_cache is None:
        av_cache = {}

    result: float = 0.0
    term_sp: Callable = _term_sp_cov_prod_h

    terms_1 = op1.flat().terms if hasattr(op1, "terms") else (op1,)
    terms_2 = op2.flat().terms if hasattr(op2, "terms") else (op2,)

    if len(terms_2) < len(terms_1):
        terms_1, terms_2 = terms_2, terms_1

    terms_norms_1 = compute_list_terms_with_norms(rho, terms_1)
    terms_norms_2 = compute_list_terms_with_norms(rho, terms_2)
    discard_check = ErrorCummulator(tol, len(terms_norms_1) * len(terms_norms_2))

    # Go over terms with the smaller norms, and try to discard them
    def iterator():
        m_1 = len(terms_norms_1) - 1
        m_2 = len(terms_norms_2) - 1
        for j in range(m_2, m_1, -1):
            for i in range(m_1, -1, -1):
                yield (
                    (
                        i,
                        j,
                    )
                )
        for j in range(m_1, -1, -1):
            yield (
                (
                    j,
                    j,
                )
            )
            for i in range(j - 1, -1, -1):
                yield (
                    (
                        i,
                        j,
                    )
                )
                yield (
                    (
                        j,
                        i,
                    )
                )

    # Go over terms with the smaller norms, and try to discard them
    for i, j in iterator():
        norm1, t1 = terms_norms_1[i]
        norm2, t2 = terms_norms_2[j]
        if discard_check.query(norm1 * norm2, 1):
            continue
        result += term_sp(rho, t1, t2, av_cache)

    return result


def _compute_cov_prod_sp_g(
    rho: ProductDensityOperator,
    op1: Operator,
    op2: Operator,
    tol: float = QALMA_TOLERANCE,
    av_cache: Optional[Dict[Operator, complex]] = None,
) -> complex:
    """
    Compute the covariance scalar product for hermitian operators.

    Compute the covariance scalar product associated to a product state for
    two hermitian operators.
    """
    if av_cache is None:
        av_cache = {}

    conjugate: bool = False
    result: complex = 0.0
    term_sp: Callable = _term_sp_cov_prod_g

    terms_1 = op1.flat().terms if hasattr(op1, "terms") else (op1,)
    terms_2 = op2.flat().terms if hasattr(op2, "terms") else (op2,)

    if len(terms_2) < len(terms_1):
        conjugate = True
        terms_1, terms_2 = terms_2, terms_1

    terms_norms_1 = compute_list_terms_with_norms(rho, terms_1)
    terms_norms_2 = compute_list_terms_with_norms(rho, terms_2)

    discard_check = ErrorCummulator(tol, len(terms_norms_1) * len(terms_norms_2))
    # Go over terms with the smaller norms, and try to discard them

    def iterator() -> Generator:
        m_1 = len(terms_norms_1) - 1
        m_2 = len(terms_norms_2) - 1
        for j in range(m_2, m_1, -1):
            for i in range(m_1, -1, -1):
                yield (
                    (
                        i,
                        j,
                    )
                )
        for j in range(m_1, -1, -1):
            yield (
                (
                    j,
                    j,
                )
            )
            for i in range(j - 1, -1, -1):
                yield (
                    (
                        i,
                        j,
                    )
                )
                yield (
                    (
                        j,
                        i,
                    )
                )

    # Go over terms with the smaller norms, and try to discard them
    for i, j in iterator():
        norm1, t1 = terms_norms_1[i]
        norm2, t2 = terms_norms_2[j]
        if discard_check.query(norm1 * norm2, 1):
            continue
        result += term_sp(rho, t1, t2, av_cache)

    return np.conj(result) if conjugate else result
