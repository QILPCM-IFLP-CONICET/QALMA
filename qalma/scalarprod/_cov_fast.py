r"""
Fast kernel for the covariance scalar product relative to a product state.

For a product state :math:`\sigma=\bigotimes_i\sigma_i` and operators
:math:`A=\sum_\alpha A_\alpha`, :math:`B=\sum_\beta B_\beta` given as sums of
terms, the covariance product is a sum over pairs of terms. When the supports
of :math:`A_\alpha` and :math:`B_\beta` are disjoint,
:math:`\langle A_\alpha^\dagger B_\beta\rangle =
\langle A_\alpha\rangle^* \langle B_\beta\rangle`, so

.. math::

    (A, B) = \sum_{\alpha\beta:\ \mathrm{overlapping}}
             \big[k(A_\alpha, B_\beta) - d(A_\alpha, B_\beta)\big]
             + D(A, B),

where :math:`k` is the kernel for one pair, :math:`d` its value for disjoint
supports (a product of means) and :math:`D(A,B)` the same expression with the
total means. Only pairs of terms with overlapping supports are visited, and
no operator is centered: the terms keep their structure.

Two kinds of terms are supported:

* *product* terms (:class:`LocalOperator`, :class:`ProductOperator`,
  :class:`ScalarOperator`): a pair of products is evaluated site by site,
  :math:`\prod_i \mathrm{Tr}[\sigma_i a_i b_i]`, without building any
  many-site matrix;
* *dense* terms (:class:`BlockOperator` acting on at most
  :data:`MAX_DENSE_SITES` sites): they are reduced onto the overlap,
  :math:`\mathrm{Tr}_{S\setminus O}[\sigma\,A]`, with NumPy tensor
  contractions; the reductions are cached.

The functions return ``None`` when some term is of another type, so that the
caller can fall back on the general implementation.

Unlike the general implementation, this kernel does not discard small
contributions: the result is exact up to round-off.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from qalma.operators.basic import LocalOperator
from qalma.operators.product import ProductOperator
from qalma.operators.qutip import BlockOperator
from qalma.operators.states import ProductDensityOperator

#: Largest support (number of sites) of a dense term handled by the kernel.
MAX_DENSE_SITES = 6

#: The cache is cleared when it holds more entries than this.
MAX_CACHE_ENTRIES = 200_000


class _ProductTerm:
    """A product term: prefactor and local factors on its sites."""

    __slots__ = ("prefactor", "factors")

    def __init__(self, prefactor: complex, factors: Dict[str, NDArray]):
        self.prefactor = prefactor
        self.factors = factors


class _DenseTerm:
    """A dense term: tensor with row and column indices, one per site."""

    __slots__ = ("support", "tensor")

    def __init__(self, support: Tuple[str, ...], tensor: NDArray):
        self.support = support
        self.tensor = tensor


class DenseTermCache:
    """Cache of term representations, means and reductions for one product state.

    Entries are keyed by ``id(term)``; the cache keeps a reference to every
    term it stores, so that the ids remain unique while the entries exist.
    """

    def __init__(self, rho: ProductDensityOperator):
        self.rho = rho
        self._local: Dict[str, NDArray] = {}
        self._terms: Dict[int, Tuple[object, object]] = {}
        self._means: Dict[int, complex] = {}
        self._reduced: Dict[Tuple[int, Tuple[str, ...]], NDArray] = {}
        self._blocks: Dict[Tuple[str, ...], NDArray] = {}

    def clear(self):
        """Drop all the cached data."""
        self._terms.clear()
        self._means.clear()
        self._reduced.clear()

    def _check_size(self):
        if len(self._terms) + len(self._reduced) > MAX_CACHE_ENTRIES:
            self.clear()

    def sigma(self, site: str) -> NDArray:
        """Return the local density matrix of the reference state on ``site``."""
        try:
            return self._local[site]
        except KeyError:
            pass
        factors = self.rho.site_factors_qutip
        if site in factors:
            mat = np.asarray(factors[site].full())
        else:
            dim = self.rho.system.dimensions[site]
            mat = np.eye(dim) / dim
        self._local[site] = mat
        return mat

    def term(self, term) -> Optional[object]:
        """Return the internal representation of ``term``, or None if unsupported."""
        key = id(term)
        try:
            return self._terms[key][1]
        except KeyError:
            pass
        rep = _make_representation(term)
        if rep is None:
            return None
        self._check_size()
        self._terms[key] = (term, rep)
        return rep

    def mean(self, term) -> complex:
        """Return ``<term>`` in the reference state."""
        key = id(term)
        try:
            return self._means[key]
        except KeyError:
            pass
        rep = self.term(term)
        if isinstance(rep, _ProductTerm):
            value = complex(rep.prefactor)
            for site, factor in rep.factors.items():
                value *= np.trace(self.sigma(site) @ factor)
        else:
            value = complex(self.reduce(term, ()).item())
        self._means[key] = value
        return value

    def reduce(self, term, overlap: Tuple[str, ...]) -> NDArray:
        r"""Return ``Tr_{S \ O}[sigma_{S \ O} term]`` as a matrix on ``overlap``.

        ``overlap`` is a sorted tuple of sites; the rows and columns of the
        result follow that order.
        """
        key = (id(term), overlap)
        try:
            return self._reduced[key]
        except KeyError:
            pass
        rep = self.term(term)
        if isinstance(rep, _ProductTerm):
            scalar = complex(rep.prefactor)
            matrix = np.ones((1, 1), dtype=complex)
            for site, factor in rep.factors.items():
                if site not in overlap:
                    scalar *= np.trace(self.sigma(site) @ factor)
            for site in overlap:
                local = rep.factors.get(site)
                if local is None:
                    local = np.eye(self.rho.system.dimensions[site])
                matrix = np.kron(matrix, local)
            result = scalar * matrix
        else:
            result = self._reduce_dense(cast(_DenseTerm, rep), overlap)
        self._reduced[key] = result
        return result

    def _reduce_dense(self, rep: _DenseTerm, overlap: Tuple[str, ...]) -> NDArray:
        tensor = rep.tensor
        support = list(rep.support)
        for pos in range(len(support) - 1, -1, -1):
            site = support[pos]
            if site in overlap:
                continue
            n_sites = len(support)
            # Tr[sigma_s T] = sum_{ab} T[.., a, .., b, ..] sigma_s[b, a]
            tensor = np.tensordot(
                tensor, self.sigma(site).T, axes=([pos, n_sites + pos], [0, 1])
            )
            support.pop(pos)
        n_sites = len(support)
        if n_sites == 0:
            return np.array([[complex(tensor)]])
        order = [support.index(site) for site in overlap]
        tensor = np.transpose(tensor, order + [n_sites + p for p in order])
        dim = int(np.prod(tensor.shape[:n_sites]))
        return tensor.reshape(dim, dim)

    def sigma_block(self, overlap: Tuple[str, ...]) -> NDArray:
        """Return the reference state on the sites of ``overlap`` (kron order).

        The blocks are cached: the overlaps of the terms of local operators
        are a few pairs and triples of sites, shared by many pairs of terms.
        """
        try:
            return self._blocks[overlap]
        except KeyError:
            pass
        matrix = np.ones((1, 1))
        for site in overlap:
            matrix = np.kron(matrix, self.sigma(site))
        self._blocks[overlap] = matrix
        return matrix


def _make_representation(term) -> Optional[object]:
    """Build the internal representation of a term, or None if unsupported."""
    if isinstance(term, LocalOperator):
        return _ProductTerm(1.0, {term.site: np.asarray(term.operator)})
    if isinstance(term, ProductOperator):
        factors = {
            site: np.asarray(
                factor.full() if hasattr(factor, "full") else factor, dtype=complex
            )
            for site, factor in term.site_factors.items()
        }
        return _ProductTerm(complex(term.prefactor), factors)
    if isinstance(term, BlockOperator):
        names = term.site_names
        if len(names) > MAX_DENSE_SITES:
            return None
        support = tuple(sorted(names, key=lambda name: names[name]))
        matrix = np.asarray(term.operator.full()) * term.prefactor
        dims = list(term.operator.dims[0])
        return _DenseTerm(support, matrix.reshape(dims + dims))
    return None


def _flat_terms(operator) -> Sequence:
    return operator.terms if hasattr(operator, "terms") else (operator,)


def _pair_value(cache, t_1, t_2, overlap, hermitian) -> complex:
    """Kernel ``k(t_1, t_2)`` for two terms with overlapping supports."""
    rep_1, rep_2 = cache.term(t_1), cache.term(t_2)
    if isinstance(rep_1, _ProductTerm) and isinstance(rep_2, _ProductTerm):
        sites = set(rep_1.factors) | set(rep_2.factors)
        direct = complex(rep_1.prefactor * rep_2.prefactor)
        if hermitian:
            for site in sites:
                a = rep_1.factors.get(site)
                b = rep_2.factors.get(site)
                prod = b if a is None else (a if b is None else a @ b)
                direct *= np.trace(cache.sigma(site) @ prod)
            return complex(np.real(direct))
        direct = np.conj(rep_1.prefactor) * rep_2.prefactor
        reverse = direct
        for site in sites:
            a = rep_1.factors.get(site)
            b = rep_2.factors.get(site)
            sig = cache.sigma(site)
            a_dag = None if a is None else a.conj().T
            if a_dag is None:
                value = np.trace(sig @ b)
                direct *= value
                reverse *= value
            elif b is None:
                value = np.trace(sig @ a_dag)
                direct *= value
                reverse *= value
            else:
                direct *= np.trace(sig @ a_dag @ b)
                reverse *= np.trace(sig @ b @ a_dag)
        return 0.5 * (direct + reverse)

    red_1 = cache.reduce(t_1, overlap)
    red_2 = cache.reduce(t_2, overlap)
    sig = cache.sigma_block(overlap)
    if hermitian:
        return complex(np.real(np.trace(sig @ red_1 @ red_2)))
    red_1 = red_1.conj().T
    return 0.5 * (np.trace(sig @ red_1 @ red_2) + np.trace(sig @ red_2 @ red_1))


def fast_cov_prod_sp(
    rho, op_1, op_2, hermitian: bool, cache: Optional[DenseTermCache] = None
) -> Optional[complex]:
    """Covariance scalar product of ``op_1`` and ``op_2`` relative to ``rho``.

    Parameters
    ----------
    rho : ProductDensityOperator
        The reference product state.
    op_1, op_2 : Operator
        Flattened operators (sums of terms, or single terms).
    hermitian : bool
        If True, compute ``Re <op_1 op_2>`` (both operators Hermitian);
        otherwise ``<{op_1^dagger, op_2}>/2``.
    cache : DenseTermCache, optional
        Cache bound to ``rho``; a temporary one is used if not given.

    Returns
    -------
    complex or None
        The scalar product, or None if some term is not supported.
    """
    if cache is None or cache.rho is not rho:
        cache = DenseTermCache(rho)
    terms_1 = _flat_terms(op_1)
    terms_2 = _flat_terms(op_2)
    reps_2: List[object] = []
    for term in terms_1:
        if cache.term(term) is None:
            return None
    for term in terms_2:
        rep = cache.term(term)
        if rep is None:
            return None
        reps_2.append(rep)

    means_1 = [cache.mean(term) for term in terms_1]
    means_2 = [cache.mean(term) for term in terms_2]
    if hermitian:
        result = complex(np.real(sum(means_1) * sum(means_2)))
    else:
        result = np.conj(sum(means_1)) * sum(means_2)

    index: Dict[str, List[int]] = defaultdict(list)
    supports_2 = [_support(rep) for rep in reps_2]
    for pos, support in enumerate(supports_2):
        for site in support:
            index[site].append(pos)

    for pos_1, term_1 in enumerate(terms_1):
        support_1 = _support(cache.term(term_1))
        partners: Set[int] = set()
        for site in support_1:
            partners.update(index.get(site, ()))
        for pos_2 in partners:
            overlap = tuple(sorted(support_1.intersection(supports_2[pos_2])))
            value = _pair_value(cache, term_1, terms_2[pos_2], overlap, hermitian)
            if hermitian:
                disjoint = np.real(means_1[pos_1] * means_2[pos_2])
            else:
                disjoint = np.conj(means_1[pos_1]) * means_2[pos_2]
            result += value - disjoint
    return float(np.real(result)) if hermitian else complex(result)


def _support(rep) -> frozenset:
    if isinstance(rep, _ProductTerm):
        return frozenset(rep.factors)
    return frozenset(rep.support)
