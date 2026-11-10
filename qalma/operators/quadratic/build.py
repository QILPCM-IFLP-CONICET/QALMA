r"""Factory and helpers for building :class:`QuadraticFormOperator` objects.

Given an arbitrary two-body quantum operator :math:`T`, this module provides
the machinery to decompose it as

.. math::

    T = L + \sum_\alpha w_\alpha Q_\alpha^2 + \delta T

by diagonalising the coupling matrix of :math:`T` in a Hilbert–Schmidt
orthonormal local basis.  The main entry point is
:func:`build_quadratic_form_from_operator`.

Pipeline overview
-----------------
The decomposition proceeds in five steps:

1. **Classify** (:func:`classify_terms`): split :math:`T` into two-site
   blocks, a one-body (linear) part, and a high-body remainder
   :math:`\delta T` using :func:`~qalma.projections.n_body_projection`.
2. **Local basis** (:func:`build_local_basis`, :func:`orthonormal_hs_local_basis`):
   build a traceless, HS-orthonormal basis of local operators on each site
   from the one-body factors that appear in the two-site blocks.
3. **Coupling matrix** (:func:`build_quadratic_form_matrix`,
   :func:`fill_array_from_block`): assemble the real symmetric matrix
   :math:`M_{\mu\nu}` whose entries are the coefficients of the coupling
   of each pair of basis elements.
4. **Diagonalise** (:func:`decompose_matrix`): compute :math:`(w_\alpha, v_\alpha) = \text{eigh}(M)`,
   optionally sort and truncate, and build one-body operators
   :math:`Q_\alpha = \sum_i (v_\alpha)_i \hat{e}_i`.
5. **Normalise** (:func:`basis_and_weights`): rescale each :math:`Q_\alpha`
   to spectral norm 1 and absorb the squared norm into :math:`w_\alpha`.

Optional: if a reference state ``sigma_ref`` is provided,
:func:`zero_expectation_value_basis` shifts every basis element so that
:math:`\langle Q_\alpha \rangle_{\sigma_{\rm ref}} = 0` before the
diagonalisation step.  This is the decomposition relative to
:math:`\sigma_{\rm ref}` used in the self-consistent MFA loop.
"""

from typing import Callable, Dict, Generator, List, Optional, Tuple, cast

import numpy as np
from numpy.linalg import eigh
from qutip import Qobj

from qalma.model import SystemDescriptor
from qalma.operators.arithmetic import OneBodyOperator, SumOperator
from qalma.operators.basic import (
    LocalOperator,
    Operator,
)
from qalma.operators.product import (
    ProductOperator,
    ScalarOperator,
)
from qalma.operators.qutip import QutipOperator
from qalma.settings import QALMA_TOLERANCE

from .quadratic import QuadraticFormOperator

LocalBasisDict = Dict[str, List[Qobj]]
BlockTermsDict = Dict[Tuple[str, ...], List[Operator]]


__all__ = ["build_quadratic_form_from_operator"]


def build_local_basis(
    terms_by_block: BlockTermsDict,
) -> LocalBasisDict:
    r"""Build an HS-orthonormal local operator basis from two-site blocks.

    Extracts the one-body factors of each two-site product operator in
    ``terms_by_block`` and assembles a Hilbert–Schmidt orthonormal basis of
    traceless Hermitian operators on each site via
    :func:`orthonormal_hs_local_basis`.

    Parameters
    ----------
    terms_by_block : dict[tuple[str, str], list[Operator]]
        Mapping from pairs of site labels to the list of two-body
        :class:`~qalma.operators.product.ProductOperator` terms acting on
        those sites.  Each key must be a 2-tuple of site names.

    Returns
    -------
    dict[str, list[Qobj]]
        Mapping from site label to the list of HS-orthonormal traceless
        Hermitian ``Qobj`` operators forming the local basis on that site.
    """
    basis_by_site: Dict[str, List[Qobj]] = {}
    # First, collect the one-body factors
    for sites, terms_list in terms_by_block.items():
        assert len(sites) == 2, sites
        product_terms: List[ProductOperator] = []
        for term in terms_list:
            # If a term is a QutipOperator, decompose
            # it first as a sum of product operators
            if hasattr(term, "as_sum_of_products"):
                term_ = term.as_sum_of_products()
                if hasattr(term_, "terms"):
                    product_terms.extend(term_.terms)
                else:
                    product_terms.append(term_)
            elif isinstance(term, ProductOperator):
                product_terms.append(term)
            else:
                raise TypeError(f"type {type(term)} should not be here.")

        for term in product_terms:
            site1, site2 = sites
            basis_by_site.setdefault(site1, []).append(term.site_factors_qutip[site1])
            basis_by_site.setdefault(site2, []).append(term.site_factors_qutip[site2])

    return orthonormal_hs_local_basis(basis_by_site)


def orthonormal_hs_local_basis(local_generators_dict: LocalBasisDict) -> LocalBasisDict:
    r"""Gram–Schmidt orthonormalise a set of local operators under the HS inner product.

    For each site, takes the supplied generators, splits non-Hermitian ones
    into Hermitian and anti-Hermitian parts, subtracts the trace to enforce
    tracelessness, and orthonormalises the result via Gram–Schmidt with
    respect to the Hilbert–Schmidt scalar product
    :math:`\langle A, B \rangle = \operatorname{Tr}[A B]`.
    Operators whose squared norm falls below :data:`~qalma.settings.QALMA_TOLERANCE`
    are discarded as linearly dependent.

    Parameters
    ----------
    local_generators_dict : dict[str, list[Qobj]]
        Mapping from site label to the list of local ``Qobj`` operators that
        seed the basis on that site.

    Returns
    -------
    dict[str, list[Qobj]]
        Mapping from site label to the orthonormal traceless Hermitian basis.
        Each returned ``Qobj`` satisfies ``b.isherm is True`` and
        :math:`\operatorname{Tr}[b_\mu b_\nu] = \delta_{\mu\nu}`.
    """
    basis_dict: Dict[str, List[Qobj]] = {}
    for site, generators in local_generators_dict.items():
        basis: List[Qobj] = []
        # Now, go over each local basis:
        for generator in generators:
            # Split in hermitian and antihermitian parts:
            components = (
                (generator,)
                if generator.isherm
                else (
                    generator + generator.dag(),
                    generator * 1j - generator.dag() * 1j,
                )
            )
            # GS orthogonalization of each component regarding the existent base.
            # If the norm is under the tolerance, discard the element.
            for hcomponent in components:
                # Ensure that components are tagged as hermitian.
                hcomponent = hcomponent - np.real(
                    hcomponent.tr() / hcomponent.dims[0][0]
                )
                hcomponent = hcomponent - sum(
                    (hcomponent * b_op).tr() * b_op for b_op in basis
                )
                normsq = abs((hcomponent * hcomponent).tr())
                if normsq <= 0 or normsq < QALMA_TOLERANCE:
                    continue
                hcomponent = hcomponent * abs(normsq ** (-0.5))
                hcomponent.isherm = True
                basis.append(hcomponent)
        #
        basis_dict[site] = basis
    return basis_dict


def zero_expectation_value_basis(basis: LocalBasisDict, sigma_ref):
    r"""Shift each basis element so that its expectation value in ``sigma_ref`` is zero.

    Replaces every local basis operator :math:`e_\mu` on each site by

    .. math::

        \tilde{e}_\mu = \operatorname{herm}\!\bigl(e_\mu
                         - \operatorname{Tr}[\sigma_{\rm ref}^{(i)} e_\mu]\bigr),

    where :math:`\sigma_{\rm ref}^{(i)}` is the single-site reduced density
    matrix and :math:`\operatorname{herm}(\cdot)` projects onto the Hermitian
    part.  After this shift, the zero-th order contribution of each
    :math:`Q_\alpha = \sum_i c_{\alpha i} \tilde{e}_i` to the free-energy
    gradient vanishes, making the subsequent optimisation better conditioned.

    Parameters
    ----------
    basis : dict[str, list[Qobj]]
        Local basis obtained from :func:`build_local_basis` or
        :func:`orthonormal_hs_local_basis`.
    sigma_ref : ProductDensityOperator
        Reference product state.  Its ``site_factors_qutip`` attribute
        must return a mapping from site label to local ``Qobj`` density matrix.

    Returns
    -------
    dict[str, list[Qobj]]
        Shifted basis with the same structure as ``basis``.

    Notes
    -----
    This function does **not** re-orthonormalise the shifted basis.  If
    orthonormality is required after the shift, pass the result back through
    :func:`orthonormal_hs_local_basis`.
    """
    local_sigmas = sigma_ref.site_factors_qutip

    def hermitian_part(qobj):
        """project a Qobj to its hermitian part"""
        if qobj.isherm:
            return qobj
        qobj = qobj * 0.5
        return qobj + qobj.dag()

    new_basis = {}
    for site, _ in basis.items():
        local_sigma = local_sigmas[site]
        new_basis_site = [
            hermitian_part(elem - (elem * local_sigma).tr()) for elem in basis[site]
        ]
        new_basis[site] = new_basis_site
    return new_basis


def classify_terms(
    operator: Operator, sigma_ref
) -> Tuple[BlockTermsDict, List[Operator], List[Operator]]:
    r"""Split ``operator`` into two-site blocks, a one-body part, and a remainder.

    Decomposes ``operator`` as

    .. math::

        T = \underbrace{\sum_{i<j} \sum_a q_{a,ij}}_{\text{two-site blocks}}
            + \underbrace{\sum_b L_b}_{\text{one-body}}
            + \underbrace{\sum_c \delta_c}_{n \geq 3\text{-body remainder}},

    where every :math:`q_{a,ij}` is a two-site product operator and each
    :math:`L_b` is a local or one-body operator.

    For each two-site ``ProductOperator``, the function subtracts the
    mean-field one-body contribution induced by the reference state
    :math:`\sigma_{\rm ref}` (or the maximally mixed state if
    ``sigma_ref is None``), so the two-body part is *centred* around
    :math:`\sigma_{\rm ref}`.  Terms acting on more than two sites are first
    projected onto the two-body sector using
    :func:`~qalma.projections.n_body_projection`; the residual goes into the
    offset.

    Parameters
    ----------
    operator : Operator
        The operator to classify.  Must be already simplified.
    sigma_ref : ProductDensityOperator or None
        Reference product state defining the one-body subtraction.  If
        ``None``, the maximally mixed state :math:`\mathbb{1}/d` is used on
        each site.

    Returns
    -------
    terms_by_block : dict[tuple[str, str], list[ProductOperator]]
        Two-site product operators grouped by the pair of sites they act on.
        Keys are alphabetically sorted 2-tuples of site labels.
    linear_terms : list[Operator]
        One-body contributions (including the mean-field shifts extracted
        from the two-site terms).
    offset_terms : list[Operator]
        High-body remainder terms (:math:`n \geq 3`) that could not be
        captured by the two-body projection.
    """
    local_sigmas = (
        sigma_ref.site_factors_qutip
        if sigma_ref is not None
        else {
            site: 1 / dimension
            for site, dimension in operator.system.dimensions.items()
        }
    )

    def decompose_two_body_product_operator(prod_op) -> Tuple[Operator, Operator]:
        prefactor = prod_op.prefactor
        system = prod_op.system
        assert isinstance(prod_op, ProductOperator)
        sites_op = prod_op.site_factors_qutip
        assert len(sites_op) == 2
        averages = {
            site: (
                (loc_op * local_sigmas[site]).tr()
                if isinstance(loc_op, Qobj)
                else loc_op
            )
            for site, loc_op in sites_op.items()
        }
        sites_op = {
            site: (loc_op - averages[site]) for site, loc_op in sites_op.items()
        }
        site1, site2 = sites_op
        one_body_term: Operator = OneBodyOperator(
            (
                LocalOperator(
                    site1, sites_op[site1] * (averages[site2] * prefactor), system
                ),
                LocalOperator(
                    site2, sites_op[site2] * (averages[site1] * prefactor), system
                ),
            ),
            system,
        ) + averages[site1] * (averages[site2] * prefactor)
        one_body_term = one_body_term.simplify()
        return ProductOperator(sites_op, prefactor, system).simplify(), one_body_term

    terms_by_block: BlockTermsDict = {}
    offset_terms: List[Operator] = []
    linear_terms: List[Operator] = []

    if isinstance(operator, OneBodyOperator):
        return terms_by_block, [operator], offset_terms

    operator = operator.flat()
    # If is a sum, collect contributions of each term:
    if isinstance(operator, SumOperator):
        for term in operator.terms:
            sub_terms_by_block, sub_linear_terms, sub_offset_terms = classify_terms(
                term, sigma_ref
            )
            linear_terms.extend(sub_linear_terms)
            offset_terms.extend(sub_offset_terms)
            for key, val in sub_terms_by_block.items():
                assert len(key) == 2
                terms_by_block.setdefault(key, []).extend(val)
        return terms_by_block, linear_terms, offset_terms

    acts_over = operator.acts_over()
    if acts_over is None or len(acts_over) > 2:
        # pylint: disable=import-outside-toplevel
        from qalma.projections import n_body_projection

        two_body_part = n_body_projection(operator, 2, sigma_ref)
        if isinstance(operator, QutipOperator):
            operator = (operator - two_body_part).to_qutip_operator()
        else:
            operator = (operator - two_body_part).simplify()

        if operator:
            offset_terms.append(operator)
        terms_by_block, linear_terms, _ = classify_terms(two_body_part, sigma_ref)
        return terms_by_block, linear_terms, offset_terms
    if len(acts_over) < 2:
        return terms_by_block, [operator], offset_terms

    # operator acts exactly on two sites
    if isinstance(operator, QutipOperator):
        return classify_terms(operator.as_sum_of_products(), sigma_ref)
    if isinstance(operator, ProductOperator):
        operator, linear_term = decompose_two_body_product_operator(operator)
        terms_by_block[tuple(sorted(acts_over))] = [operator]
        assert len(operator.acts_over()) == 2
        return terms_by_block, ([] if linear_term.is_zero else [linear_term]), []

    raise ValueError(f"operator of type {type(operator)} cannot be processed.")


def build_quadratic_form_matrix(
    terms_by_block: BlockTermsDict, local_basis: LocalBasisDict
) -> Tuple[np.ndarray, Dict[str, int]]:
    r"""Assemble the real symmetric coupling matrix of the quadratic form.

    Concatenates the local bases from all sites into a single global index
    and fills a real symmetric matrix :math:`M` such that

    .. math::

        \sum_{(i,j)} \sum_a q_{a,ij}
        = \sum_{\mu,\nu} M_{\mu\nu}\, \hat{e}_\mu \otimes \hat{e}_\nu,

    where :math:`\hat{e}_\mu` are the HS-orthonormal basis elements on their
    respective sites and the sum runs over all site pairs.

    Parameters
    ----------
    terms_by_block : dict[tuple[str, str], list[ProductOperator]]
        Output of :func:`classify_terms`.
    local_basis : dict[str, list[Qobj]]
        HS-orthonormal local basis, output of :func:`build_local_basis`.

    Returns
    -------
    qf_array : np.ndarray, shape (N, N)
        Real symmetric coupling matrix, where :math:`N` is the total number
        of local basis elements across all sites.
    positions : dict[str, int]
        Offset of each site's block in the global index, i.e.
        ``positions[site]`` is the row/column index where site ``site``
        starts.
    """
    positions: Dict[str, int]
    full_size: int
    positions, full_size = build_positions_and_full_size(local_basis)

    result_array: np.ndarray = np.zeros(
        (
            full_size,
            full_size,
        )
    )
    block: Tuple[str, ...]
    terms: List[Operator]
    for block, terms in terms_by_block.items():
        fill_array_from_block(result_array, local_basis, positions, block, terms)

    return result_array, positions


def build_quadratic_form_from_operator(
    operator: Operator,
    simplify: bool = True,
    isherm: Optional[bool] = None,
    sigma_ref=None,
    sort_fn: Optional[Callable[[float], float]] = None,
    sort_imag_fn: Optional[Callable[[float], float]] = None,
    count: Optional[int] = None,
) -> QuadraticFormOperator:
    r"""Decompose ``operator`` into a :class:`QuadraticFormOperator`.

    Main factory function of the module.  Converts an arbitrary operator
    into the structured form

    .. math::

        T = L + \sum_\alpha w_\alpha Q_\alpha^2 + \delta T

    by classifying its terms, building a local HS-orthonormal basis,
    assembling the coupling matrix :math:`M`, and diagonalising it.

    Parameters
    ----------
    operator : Operator
        The operator to decompose.  May be any concrete subclass of
        :class:`~qalma.operators.basic.Operator`.
    simplify : bool, optional
        If ``True`` (default), simplify ``operator`` before decomposing.
    isherm : bool or None, optional
        If ``True``, force the output to be Hermitian: take the real parts
        of the weights and project ``linear_term`` and ``offset`` onto their
        Hermitian parts.  If ``None``, inferred from ``operator.isherm``.
    sigma_ref : ProductDensityOperator or None, optional
        Reference product state.  When provided, the two-body part is
        *centred* around :math:`\sigma_{\rm ref}`, so that each
        :math:`Q_\alpha` has zero expectation value in :math:`\sigma_{\rm ref}`.
        This is the form required by the variational mean-field loop in
        :mod:`qalma.meanfield.variational`.
    sort_fn : callable or None, optional
        Sorting key applied to the *real* eigenvalues :math:`w_\alpha` before
        truncation.  Use ``lambda x: x`` to sort ascending (most negative
        first), which selects the modes that lower the variational free energy
        the most.  If ``None``, eigenvalues are returned in the order produced
        by :func:`numpy.linalg.eigh` (ascending).
    sort_imag_fn : callable or None, optional
        Sorting key for the imaginary part of the eigenvalues when processing
        the anti-Hermitian component of a non-Hermitian operator.  Ignored
        when ``isherm=True``.
    count : int or None, optional
        Maximum number of quadratic terms to retain after sorting.  Corresponds
        to the ``numfields`` parameter of
        :func:`~qalma.meanfield.variational.variational_quadratic_mfa`.
        If ``None``, all terms above the numerical tolerance are kept.

    Returns
    -------
    QuadraticFormOperator
        The decomposed operator with at most ``count`` quadratic terms.

    Notes
    -----
    For non-Hermitian operators the function splits into Hermitian and
    anti-Hermitian parts,

    .. math::

        T = T_H + i T_{aH}, \quad
        T_H = \tfrac{1}{2}(T + T^\dagger), \quad
        T_{aH} = \tfrac{1}{2i}(T - T^\dagger),

    processes each with ``isherm=True`` and ``sort_imag_fn``, and sums the
    results as :math:`T_H \cdot 1 + T_{aH} \cdot i`.

    Examples
    --------
    Build the quadratic form for the two-body projection of an Ising
    Hamiltonian, keeping only the 6 modes with the most negative eigenvalues:

    .. code-block:: python

        from qalma.operators.quadratic import build_quadratic_form_from_operator
        from qalma.projections import n_body_projection

        ham_2b = n_body_projection(ham, n_max=2, sigma=sigma).hermitian_part()
        qf = build_quadratic_form_from_operator(
            ham_2b,
            isherm=True,
            sigma_ref=sigma,
            sort_fn=lambda x: x,
            count=6,
        )
        print(qf.weights)     # 6 most-negative eigenvalues of the coupling matrix
        print(len(qf.basis))  # 6
    """
    if simplify:
        operator = operator.simplify()

    if sigma_ref is not None:
        if hasattr(sigma_ref, "to_product_state"):
            sigma_ref = sigma_ref.to_product_state()

    system = operator.system
    # Trivial cases
    if isinstance(operator, ScalarOperator):
        if isherm and not operator.isherm:
            operator = ScalarOperator(operator.prefactor.real, system)
        assert (
            not isherm or isherm == operator.isherm
        ), f"{operator} -> {isherm}!={operator.isherm}"
        return QuadraticFormOperator(tuple(), tuple(), system, operator, None)

    if (
        isinstance(operator, (LocalOperator, OneBodyOperator))
        or len(operator.acts_over()) < 2
    ):
        if isherm and not operator.isherm:
            operator = operator + operator.dag()
        return QuadraticFormOperator(
            tuple(), tuple(), system, operator.simplify(), None
        )

    # Already a quadratic form:
    if isinstance(operator, QuadraticFormOperator):
        if isherm and not operator.isherm:
            operator = QuadraticFormOperator(
                operator.basis,
                tuple((np.real(w) for w in operator.weights)),
                system,
                force_hermitic_t(operator.linear_term),
                force_hermitic_t(operator.offset),
            )
        return operator

    # SumOperators, and operators acting on at least size 2 blocks:
    isherm = isherm or operator.isherm

    # For non-hermitian, convert the hermitian
    # and the anti-hermitian parts, and sum both.
    if not isherm:
        return sum(
            build_quadratic_form_from_operator(
                op,
                simplify=True,
                isherm=True,
                sigma_ref=sigma_ref,
                sort_fn=sfn,
                count=count,
            )
            * w
            for op, w, sfn in (
                (operator.hermitian_part(), 1.0, sort_fn),
                ((operator * (-1j)).hermitian_part(), 1.0j, sort_imag_fn),
            )
        )

    # Process hermitian operators
    # Classify terms
    system = operator.system
    terms_by_2body_block, linear_terms, offset_terms = classify_terms(
        operator, sigma_ref
    )
    linear_term = cast(Operator, sum(linear_terms)).simplify() if linear_terms else None
    offset = cast(Operator, sum(offset_terms)).simplify() if offset_terms else None

    if isherm:
        linear_term = force_hermitic_t(linear_term)
        offset = force_hermitic_t(offset)

    # Build the basis
    local_basis: Dict[str, List[Qobj]] = build_local_basis(terms_by_2body_block)
    # Build the matrix of the quadratic form
    qf_array, local_basis_offsets = build_quadratic_form_matrix(
        terms_by_2body_block, local_basis
    )
    if sigma_ref is not None:
        local_basis = zero_expectation_value_basis(local_basis, sigma_ref)

    weights, qf_basis = basis_and_weights(
        decompose_matrix(
            qf_array, local_basis, local_basis_offsets, system, sort_fn, count
        )
    )

    return QuadraticFormOperator(
        basis=qf_basis,
        weights=weights,
        system=operator.system,
        linear_term=linear_term,
        offset=offset,
    )


def basis_and_weights(qf_basis_list: List[List[Operator]]):
    r"""Normalise generators to spectral norm 1 and absorb the norm into the weights.

    Given a list of ``(raw_weight, raw_generator)`` pairs produced by
    :func:`decompose_matrix`, rescales each generator :math:`G_\alpha` as

    .. math::

        Q_\alpha = G_\alpha / \|G_\alpha\|_\infty, \quad
        w_\alpha = w_\alpha^{\rm raw} \cdot \|G_\alpha\|_\infty^2,

    so that every :math:`Q_\alpha` has spectral norm 1 and the product
    :math:`w_\alpha Q_\alpha^2` is unchanged.

    Parameters
    ----------
    qf_basis_list : list of [raw_weight, raw_generator]
        Pairs ``[w, G]`` as returned by :func:`decompose_matrix`.

    Returns
    -------
    weights : tuple[complex, ...]
        Rescaled weights :math:`w_\alpha`.
    basis : tuple[OneBodyOperator, ...]
        Normalised generators :math:`Q_\alpha` with spectral norm 1.
    """

    def spectral_norm(ob_op: Operator) -> complex:
        if isinstance(ob_op, ScalarOperator):
            return ob_op.prefactor
        if isinstance(ob_op, OneBodyOperator):
            return sum(spectral_norm(term) for term in ob_op.simplify().terms)
        if isinstance(ob_op, LocalOperator):
            return max((ob_op.operator_qutip**2).eigenenergies()) ** 0.5
        raise TypeError(f"spectral_norm can not be computed for {type(ob_op)}")

    spectral_norms: Generator = (
        spectral_norm(weight_generator[1]) for weight_generator in qf_basis_list
    )
    qf_basis_and_weight = tuple(
        (
            weight_generator[0] * sn**2,
            weight_generator[1] / sn,
        )
        for sn, weight_generator in zip(spectral_norms, qf_basis_list)
    )
    return (
        tuple((weight_generator[0] for weight_generator in qf_basis_and_weight)),
        tuple((weight_generator[1] for weight_generator in qf_basis_and_weight)),
    )


def decompose_matrix(
    qf_array: np.ndarray,
    local_basis,
    local_basis_offsets,
    system: SystemDescriptor,
    sort_fn: Optional[Callable[[float], float]] = None,
    count: Optional[int] = None,
):
    r"""Diagonalise the coupling matrix and build one-body mode operators.

    Computes the eigendecomposition :math:`M = V \Lambda V^\top` via
    :func:`numpy.linalg.eigh`, optionally sorts and truncates the modes, and
    constructs the corresponding one-body operators

    .. math::

        G_\alpha = \sum_i (v_\alpha)_i\, \hat{e}_i,

    where :math:`\hat{e}_i` are the HS-orthonormal local basis elements and
    the index :math:`i` runs over all sites in the concatenated global basis.

    Parameters
    ----------
    qf_array : np.ndarray, shape (N, N)
        Real symmetric coupling matrix from :func:`build_quadratic_form_matrix`.
    local_basis : dict[str, list[Qobj]]
        HS-orthonormal local basis from :func:`build_local_basis`.
    local_basis_offsets : dict[str, int]
        Starting column of each site's block in the global index, as returned
        by :func:`build_quadratic_form_matrix`.
    system : SystemDescriptor
        System descriptor propagated to the constructed operators.
    sort_fn : callable or None, optional
        Sorting key for the eigenvalues before truncation.  ``None`` keeps the
        default ascending order from :func:`numpy.linalg.eigh`.
    count : int or None, optional
        Maximum number of modes to return.  ``None`` returns all modes whose
        eigenvalue exceeds :data:`~qalma.settings.QALMA_TOLERANCE`.

    Returns
    -------
    list of [float, OneBodyOperator]
        Pairs ``[w/2, G]`` sorted by ``sort_fn(w)`` and truncated to
        ``count`` entries.  The factor :math:`1/2` appears because
        :math:`w_\alpha Q_\alpha^2 = \tfrac{w_\alpha}{2}(Q_\alpha + Q_\alpha^\dagger)^2/2`
        in the Hermitian convention used here.
    """
    e_vals, e_vecs = eigh(qf_array)
    e_vecs = e_vecs.T

    if sort_fn is not None:
        sorted_eval_evec = sorted(
            tuple(zip(e_vals, e_vecs)), key=lambda x: sort_fn(x[0])
        )
        e_vals = np.array([e_val for e_val, e_vec in sorted_eval_evec])
        e_vecs = np.array([e_vec for e_val, e_vec in sorted_eval_evec])

    if count is not None and len(e_vals) > count:
        e_vals = e_vals[:count]
        e_vecs = e_vecs[:count]

    return [
        (
            0.5 * e_val,
            OneBodyOperator(
                tuple(
                    LocalOperator(
                        site,
                        sum(
                            local_op * e_vec[mu + local_basis_offsets[site]]
                            for mu, local_op in enumerate(local_base)
                        ),
                        system,
                    )
                    for site, local_base in local_basis.items()
                ),
                system,
            ),
        )
        for e_val, e_vec in zip(e_vals, e_vecs)
        if abs(e_val) > QALMA_TOLERANCE
    ]


def force_hermitic_t(t):
    """Project ``t`` onto its Hermitian part, or return ``None`` unchanged.

    Replaces ``t`` by ``(t + t.dag()) * 0.5`` when ``t`` is not already
    Hermitian, then simplifies.  If ``t`` is ``None`` the function is a
    no-op and returns ``None``.

    Parameters
    ----------
    t : Operator or None
        Operator to project.

    Returns
    -------
    Operator or None
        Hermitian projection of ``t``, or ``None``.
    """
    if t is None:
        return t
    if not t.isherm:
        t = (t * 0.5 + t.dag() * 0.5).simplify()
    return t


def build_positions_and_full_size(
    local_basis: LocalBasisDict,
) -> Tuple[Dict[str, int], int]:
    """Compute the global-index offset of each site's local basis block.

    Given a local basis ``{site: [e_0, e_1, ...]}`` for each site, builds a
    mapping from site name to the starting column in the concatenated
    (alphabetically sorted) global index, and returns the total number of
    basis elements.

    Parameters
    ----------
    local_basis : dict[str, list[Qobj]]
        Local basis as produced by :func:`build_local_basis`.

    Returns
    -------
    positions : dict[str, int]
        Mapping ``site -> start_index`` in the global basis.
    full_size : int
        Total number of local basis elements, i.e. the dimension of the
        coupling matrix produced by :func:`build_quadratic_form_matrix`.
    """
    sizes: Dict[str, int] = {
        site: len(local_base) for site, local_base in local_basis.items()
    }
    sorted_sites: List[str] = sorted(sizes)
    return (
        {
            site: sum(sizes[site_] for site_ in sorted_sites[:pos])
            for pos, site in enumerate(sorted_sites)
        },
        sum(sizes.values()),
    )


def fill_array_from_block(
    result_array: np.ndarray,
    local_basis: LocalBasisDict,
    positions: Dict[str, int],
    block: Tuple[str, ...],
    terms: List[Operator],
) -> None:
    r"""Accumulate the coupling-matrix entries from a single two-site block.

    For each product operator ``prefactor * A_{site1} ⊗ B_{site2}`` in
    ``terms``, adds its coefficients in the local HS basis to
    ``result_array``:

    .. math::

        M_{\mu,\nu} \mathrel{+}= \operatorname{prefactor}
            \cdot \operatorname{Tr}[A\, b_\mu]\,\operatorname{Tr}[B\, b_\nu],

    and symmetrises as :math:`M_{\nu,\mu} = M_{\mu,\nu}`.  The function
    operates **in-place** on ``result_array``.

    Parameters
    ----------
    result_array : np.ndarray, shape (N, N)
        Coupling matrix to update in-place.
    local_basis : dict[str, list[Qobj]]
        HS-orthonormal local basis on each site.
    positions : dict[str, int]
        Global-index offsets from :func:`build_positions_and_full_size`.
    block : tuple[str, str]
        The two site labels ``(site1, site2)`` of this block.
    terms : list[ProductOperator]
        Two-site product operators acting on ``block``.
    """
    site1, site2 = block
    position_1 = positions[site1]
    position_2 = positions[site2]
    for term in terms:
        assert isinstance(term, ProductOperator)
        op1, op2 = (term.site_factors_qutip[site] for site in block)
        for mu, b1 in enumerate(local_basis[site1]):
            for nu, b2 in enumerate(local_basis[site2]):
                i = position_1 + mu
                j = position_2 + nu
                result_array[i, j] += np.real(
                    term.prefactor * (op1 * b1).tr() * (op2 * b2).tr()
                )
                result_array[j, i] = result_array[i, j]
