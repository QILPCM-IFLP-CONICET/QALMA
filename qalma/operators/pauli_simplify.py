r"""Symbolic simplification of sums with Pauli strings and tensor products.

The generic simplification of :class:`SumOperator` groups the terms by the
block of sites they act on and adds them as dense (QuTiP) operators. For
sums of Pauli strings and :class:`TensorProductOperator` this destroys the
symbolic structure, and the blocks of terms with Jordan-Wigner strings can
be as large as the system.

:func:`simplify_pauli_sum` keeps the structure instead. Every term is
written as :math:`\lambda\, P(c) \otimes \bigotimes_{s\in L} A_s`, and

* terms without local factors are added in a single
  :class:`PauliSumOperator`;
* local factors on two-level sites are expanded in Pauli matrices, so that
  they join the Pauli sum (e.g. spin-1/2 operators in a Kondo lattice);
* terms with the same code and a single local site :math:`s` are added as
  one term with the sum of the local factors; with ``c = 0`` they are
  local operators;
* terms with the same code and the same local sites are merged when their
  factors differ on at most one site,
  :math:`P\otimes A\otimes B_1 + P\otimes A\otimes B_2 =
  P\otimes A\otimes(B_1+B_2)`;
* local factors proportional to the identity are absorbed into the
  prefactor, and the trace of the factor of single-site terms goes to the
  Pauli part, so that the terms that remain are linearly independent
  (except for products on several sites that cannot be merged).

Any other term (e.g. a :class:`BlockOperator`) is simplified by the
generic method and kept beside the others.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from qalma.model import SystemDescriptor
from qalma.settings import QALMA_TOLERANCE

from .arithmetic import SumOperator
from .basic import LocalOperator, Operator
from .pauli import PAULI_MATRICES, PauliStringOperator, _site_index
from .pauli_sum import PauliSumOperator, as_pauli_sum
from .product import ProductOperator, ScalarOperator
from .tensor_product import TensorProductOperator

__all__ = ["has_pauli_terms", "simplify_pauli_sum"]

_PAULI_FAMILY = (PauliStringOperator, PauliSumOperator, TensorProductOperator)

# A term as (code, {site: factor}, prefactor), in the site order of a system.
_Term = Tuple[int, Dict[str, np.ndarray], complex]


def has_pauli_terms(operator: Operator) -> bool:
    """``True`` if a (possibly nested) sum has Pauli strings or tensor products."""
    if isinstance(operator, _PAULI_FAMILY):
        return True
    if isinstance(operator, SumOperator):
        return any(has_pauli_terms(term) for term in operator.terms)
    return False


def _as_term(operator: Operator, system: SystemDescriptor) -> Optional[_Term]:
    """Write ``operator`` as ``(code, local factors, prefactor)``, if possible."""
    from qalma.qutip_tools.tools import _to_array

    if isinstance(operator, TensorProductOperator):
        return (
            operator.code_in(system),
            dict(operator.local_factors),
            operator.prefactor,
        )
    if isinstance(operator, PauliStringOperator):
        return operator.code_in(system), {}, operator.prefactor
    if isinstance(operator, ScalarOperator):
        return 0, {}, operator.prefactor
    if isinstance(operator, ProductOperator):
        return 0, dict(operator.site_factors), operator.prefactor
    if isinstance(operator, LocalOperator):
        return 0, {operator.site: _to_array(operator.operator)}, 1.0
    return None


def _flatten(operator: Operator, pauli_parts: list, others: list, terms: list, system):
    """Split the terms of a (nested) sum into Pauli sums, structured terms and the rest."""
    if isinstance(operator, (int, float, complex, np.number)):
        if operator:
            terms.append((0, {}, complex(operator)))
        return
    if isinstance(operator, PauliSumOperator):
        pauli_parts.append(operator)
        return
    if isinstance(operator, SumOperator):
        for sub_term in operator.terms:
            _flatten(sub_term, pauli_parts, others, terms, system)
        return
    term = _as_term(operator, system)
    if term is None:
        others.append(operator)
    elif term[2]:
        terms.append(term)


def _expand_two_level_factors(term: _Term, system: SystemDescriptor) -> List[_Term]:
    """Expand the local factors on two-level sites in Pauli matrices."""
    code, factors, prefactor = term
    dimensions = system.dimensions
    two_level = [site for site in factors if dimensions[site] == 2]
    if not two_level:
        return [term]
    position = _site_index(system).position
    rest = {site: f for site, f in factors.items() if dimensions[site] != 2}
    expanded = [(code, prefactor)]
    for site in two_level:
        factor = factors[site]
        components = [
            (label, complex(np.trace(PAULI_MATRICES[label] @ factor) / 2))
            for label in range(4)
        ]
        components = [
            (label, value) for label, value in components if abs(value) > 1e-15
        ]
        shift = 2 * position[site]
        # The site is free in the Pauli part: the codes do not overlap.
        expanded = [
            (c | (label << shift), p * value)
            for c, p in expanded
            for label, value in components
        ]
    return [(c, dict(rest), p) for c, p in expanded]


def _drop_identities(term: _Term) -> _Term:
    """Absorb the local factors proportional to the identity into the prefactor."""
    code, factors, prefactor = term
    kept = {}
    for site, factor in factors.items():
        diagonal = factor.diagonal()
        value = diagonal[0]
        if np.allclose(factor, value * np.eye(factor.shape[0]), atol=1e-14, rtol=1e-12):
            prefactor = prefactor * value
        else:
            kept[site] = factor
    return code, kept, prefactor


def _normalize(factor: np.ndarray) -> Tuple[complex, np.ndarray]:
    """Split ``factor`` as ``scale * normalized`` with the largest entry of ``normalized`` equal to 1."""
    flat = factor.ravel()
    index = int(np.argmax(np.abs(flat)))
    scale = complex(flat[index])
    if scale == 0:
        return 0j, factor
    return scale, factor / scale


def _merge_group(group: List[Tuple[complex, Dict[str, np.ndarray]]]):
    """Merge terms with the same sites whose factors differ on at most one site."""
    # Normalize the factors, moving the scales to the prefactors.
    items: List[Tuple[complex, Dict[str, np.ndarray]]] = []
    for prefactor, factors in group:
        normalized = {}
        for site, factor in factors.items():
            scale, factor = _normalize(factor)
            prefactor *= scale
            normalized[site] = factor
        if prefactor:
            items.append((prefactor, normalized))
    merged = True
    while merged:
        merged = False
        for i, (p_i, f_i) in enumerate(items):
            for j in range(i + 1, len(items)):
                p_j, f_j = items[j]
                different = [
                    site
                    for site in f_i
                    if not np.allclose(f_i[site], f_j[site], atol=1e-14, rtol=1e-12)
                ]
                if len(different) > 1:
                    continue
                if not different:
                    items[i] = (p_i + p_j, f_i)
                else:
                    site = different[0]
                    scale, combined = _normalize(p_i * f_i[site] + p_j * f_j[site])
                    items[i] = (scale, {**f_i, site: combined})
                del items[j]
                merged = True
                break
            if merged:
                break
    return [(p, f) for p, f in items if abs(p) > QALMA_TOLERANCE]


def simplify_pauli_sum(operator: SumOperator) -> Operator:
    """Simplify a sum keeping Pauli strings and tensor products symbolic.

    See the module documentation for the rules. The result is a
    :class:`PauliSumOperator`, a single term, or a :class:`SumOperator`
    marked as simplified whose terms are a :class:`PauliSumOperator`,
    :class:`TensorProductOperator`, local operators, products and the
    terms that are not products.
    """
    system = operator.system
    pauli_parts: List[PauliSumOperator] = []
    others: List[Operator] = []
    raw_terms: List[_Term] = []
    _flatten(operator, pauli_parts, others, raw_terms, system)

    pauli: Dict[int, complex] = {}
    for part in pauli_parts:
        converted = as_pauli_sum(part)
        assert converted is not None
        for code, coeff in (
            PauliSumOperator._new(converted.codes_in(system), converted.coeffs, system)
            .to_dict()
            .items()
        ):
            pauli[code] = pauli.get(code, 0) + coeff

    single: Dict[Tuple[int, str], np.ndarray] = {}
    groups: Dict[Tuple[int, frozenset], list] = {}
    for raw in raw_terms:
        for code, factors, prefactor in _expand_two_level_factors(
            _drop_identities(raw), system
        ):
            if not factors:
                pauli[code] = pauli.get(code, 0) + prefactor
            elif len(factors) == 1:
                ((site, factor),) = factors.items()
                key = (code, site)
                value = prefactor * factor
                single[key] = single[key] + value if key in single else value
            else:
                groups.setdefault((code, frozenset(factors)), []).append(
                    (prefactor, factors)
                )

    result: List[Operator] = []
    local_terms: List[Operator] = []
    for (code, site), factor in single.items():
        # P (x) A = tr(A)/d P + P (x) (A - tr(A)/d): the traceless parts of
        # different terms, and the Pauli strings, are linearly independent.
        dim = factor.shape[0]
        identity_part = np.trace(factor) / dim
        if abs(identity_part) > QALMA_TOLERANCE:
            pauli[code] = pauli.get(code, 0) + identity_part
            factor = factor - identity_part * np.eye(dim)
        if np.abs(factor).max() < QALMA_TOLERANCE:
            continue
        if code == 0:
            local_terms.append(LocalOperator(site, factor, system))
        else:
            local_terms.append(
                TensorProductOperator._new(code, {site: factor}, 1.0, system)
            )
    for (code, _), group in groups.items():
        for prefactor, factors in _merge_group(group):
            if code == 0:
                local_terms.append(ProductOperator(factors, prefactor, system))
            else:
                local_terms.append(
                    TensorProductOperator._new(code, factors, prefactor, system)
                )

    pauli = {
        code: coeff for code, coeff in pauli.items() if abs(coeff) > QALMA_TOLERANCE
    }
    if pauli:
        result.append(PauliSumOperator.from_dict(pauli, system).collapse())
    result.extend(local_terms)
    if others:
        rest = others[0] if len(others) == 1 else SumOperator(tuple(others), system)
        rest = rest.simplify()
        result.extend(rest.terms if isinstance(rest, SumOperator) else (rest,))
    if not result:
        return ScalarOperator(0, system)
    if len(result) == 1:
        return result[0]
    isherm = getattr(operator, "_isherm", None)
    return SumOperator(tuple(result), system, isherm=isherm, simplified=True)
