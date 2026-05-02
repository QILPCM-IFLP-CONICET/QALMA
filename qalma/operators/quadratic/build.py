r"""QuadraticForm Operators.

Quadratic Form Operators provides a representation for quantum operators
of the form

Q= L + \sum_a w_a M_a ^2 + \delta Q

with L and M_a one-body operators, w_a certain weights and \delta Q a
*remainder* as a sum of n-body terms.
"""

# from numbers import Number
from typing import Dict, Generator, List, Optional, Tuple, cast

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
# from typing import Union


__all__ = ["build_quadratic_form_from_operator"]


def build_local_basis(
    terms_by_block: BlockTermsDict,
) -> LocalBasisDict:
    """Build a local basis from a list of two-body operators.

    Build a local basis of operators from a list of two-body operators on
    each pair of sites.
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
    """Build a HS orthonormal basis of local operators.

    From a set of operators associated to each site, build an
    orthonormalized basis of hermitian operators regarding the HS scalar
    product on each site.
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
                if normsq < QALMA_TOLERANCE:
                    continue
                hcomponent = hcomponent * abs(normsq ** (-0.5))
                hcomponent.isherm = True
                basis.append(hcomponent)
        #
        basis_dict[site] = basis
    return basis_dict


def zero_expectation_value_basis(basis: LocalBasisDict, sigma_ref):
    """Build a zero expectation value basis.

    Add an offset of each element of the local basis in a way that each
    operator have zero mean regarding sigma_ref.
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
    """Decompose ``operator`` as a sum of two-site operators.

    Decompose `operator` as list of terms
    associated to each pairs of sites,
    and offset terms
    operator = sum_{ij} sum_a q_ija +  sum_{b} offset_{b}.

    If sigma_ref is None, two-body and many-body terms
    have zero trace. Otherwise, operators have zero expectation
    value relative to sigma_ref.
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
        assert isinstance(operator, ProductOperator)
        sites_op = operator.site_factors_qutip
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
    """Build the matrix associated to the quadratic form in a given basis."""
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
) -> QuadraticFormOperator:
    """Build a QuadraticFormOperator from `operator`."""
    if simplify:
        operator = operator.simplify()

    if sigma_ref is not None:
        if hasattr(sigma_ref, "to_product_state"):
            sigma_ref = sigma_ref.to_product_state()
        # assert isinstance(
        #    sigma_ref, ProductDensityOperator
        # ), f"sigma_ref must be a ProductDensityOperator. Got {type(sigma_ref)}"

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
            )
            * w
            for op, w in (
                (operator.hermitian_part(), 1.0),
                ((operator * (-1j)).hermitian_part(), 1.0j),
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
        decompose_matrix(qf_array, local_basis, local_basis_offsets, system)
    )

    return QuadraticFormOperator(
        basis=qf_basis,
        weights=weights,
        system=operator.system,
        linear_term=linear_term,
        offset=offset,
    )


def basis_and_weights(qf_basis_list: List[List[Operator]]):
    """Build basis and weights."""

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
    qf_array: np.ndarray, local_basis, local_basis_offsets, system: SystemDescriptor
):
    """Decompose the array into."""
    e_vals, e_vecs = eigh(qf_array)

    return sorted(
        [
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
            for e_val, e_vec in zip(e_vals, e_vecs.T)
            if abs(e_val) > QALMA_TOLERANCE
        ],
        key=lambda x: x[0],
    )


def force_hermitic_t(t):
    """Force `t` to be hermitian."""
    if t is None:
        return t
    if not t.isherm:
        t = (t + t.dag()).simplify()
        t = t * 0.5
    return t


def build_positions_and_full_size(
    local_basis: LocalBasisDict,
) -> Tuple[Dict[str, int], int]:
    """Build the positions."""
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
    """Full result_array with the coefficients obtained from the local basis."""
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
