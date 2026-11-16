r"""Products of a Pauli string and local operators on disjoint sites.

A :class:`TensorProductOperator` represents

.. math::

    \lambda\; P \otimes \bigotimes_{s \in L} A_s ,

where :math:`P` is a Pauli string (stored as its code, see
:mod:`qalma.operators.pauli`) acting on two-level sites, and :math:`A_s`
are dense local factors on a set :math:`L` of sites disjoint from the
support of :math:`P`. The sites in :math:`L` can have any dimension.

This is the form of the operators of models whose sites are split into
fermionic qubits plus a non-fermionic subsite (e.g. the t-J model, or Kondo
lattices with :math:`S > 1/2`): the fermionic part carries the
Jordan-Wigner strings, and the rest is local.

Because the two parts act on disjoint sites they commute, and

.. math::

    (P_1 \otimes A_1)(P_2 \otimes A_2) = (P_1 P_2) \otimes (A_1 A_2),

whenever the Pauli support of each factor is disjoint from the local
support of the other. The Pauli product is computed on the codes (so the
Jordan-Wigner strings cancel symbolically, without leaving identity
factors), and only the local factors are multiplied as matrices. When the
supports overlap, the product falls back to the dense product of
:class:`ProductOperator`.
"""

from functools import cached_property
from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np

from qalma.model import SystemDescriptor
from qalma.qutip_tools.tools import _to_array
from qalma.settings import QALMA_TOLERANCE

from .basic import LocalOperator, Operator
from .pauli import (
    _I_POWERS,
    PAULI_LABELS,
    PAULI_MATRICES,
    PauliStringOperator,
    _product_phase,
    _remap_code,
    _site_index,
)
from .product import ProductOperator, ScalarOperator

__all__ = ["TensorProductOperator", "tensor_product"]


def _pauli_sites(code: int, system: SystemDescriptor) -> Dict[str, int]:
    """Mapping ``site -> Pauli code`` of the nontrivial factors of ``code``."""
    order = _site_index(system).order
    result = {}
    k = 0
    while code:
        local = code & 3
        if local:
            result[order[k]] = local
        code >>= 2
        k += 1
    return result


class TensorProductOperator(ProductOperator):
    r"""Product :math:`\lambda\, P \otimes \bigotimes_s A_s` with disjoint supports.

    Subclass of :class:`ProductOperator`: ``site_factors`` (the Pauli
    matrices of :math:`P` together with the local factors) is built on
    demand, so every routine written for products accepts it.

    Parameters
    ----------
    code : int
        The Pauli string, encoded in the site order of ``system``.
    local_factors : dict[str, np.ndarray]
        The local factors :math:`A_s`, on sites where ``code`` is the
        identity.
    prefactor : complex
        The global prefactor :math:`\lambda`.
    system : SystemDescriptor
        The system.
    """

    code: int
    local_factors: Dict[str, np.ndarray]

    def __init__(
        self,
        code: int,
        local_factors: Mapping[str, np.ndarray],
        prefactor: complex = 1.0,
        system: Optional[SystemDescriptor] = None,
    ):
        # ProductOperator.__init__ is not called: ``site_factors`` is
        # derived on demand from the two parts.
        assert system is not None
        overlap = set(_pauli_sites(code, system)) & set(local_factors)
        if overlap:
            raise ValueError(
                f"the Pauli string and the local factors share the sites {sorted(overlap)}."
            )
        self._init_fast(code, local_factors, prefactor, system)

    def _init_fast(self, code, local_factors, prefactor, system):
        self.code = code
        # complex arrays, as in ProductOperator (the QuTiP conversion relies on it)
        self.local_factors = {
            site: (
                factor
                if isinstance(factor, np.ndarray) and factor.dtype == np.complex128
                else np.asarray(factor, dtype=np.complex128)
            )
            for site, factor in local_factors.items()
        }
        self.prefactor = prefactor
        self._to_qutip_cache = {}
        self._set_system_attrs(system)

    def _set_system_attrs(self, system: Optional[SystemDescriptor]):
        self.system = system  # type: ignore[assignment]
        if system is not None:
            self.size = len(system.sites)
            self.dimensions = system.dimensions

    @classmethod
    def _new(cls, code, local_factors, prefactor, system) -> "TensorProductOperator":
        result = cls.__new__(cls)
        result._init_fast(code, local_factors, prefactor, system)
        return result

    # ------------------------------------------------------------------
    # Parts and representations
    # ------------------------------------------------------------------

    @cached_property
    def paulis(self) -> Dict[str, int]:
        """Mapping ``site -> code`` of the nontrivial factors of the Pauli part."""
        return _pauli_sites(self.code, self.system)

    @property
    def pauli_part(self) -> PauliStringOperator:
        """The Pauli string :math:`P` (with prefactor 1)."""
        return PauliStringOperator._new(self.code, 1.0, self.system)

    @property
    def local_part(self) -> ProductOperator:
        """The product of local factors :math:`\\bigotimes_s A_s` (with prefactor 1)."""
        return ProductOperator(dict(self.local_factors), 1.0, self.system)

    @cached_property
    def site_factors(self) -> Dict[str, np.ndarray]:  # type: ignore[override]
        """All the factors, for compatibility with :class:`ProductOperator`."""
        result = {site: PAULI_MATRICES[c].copy() for site, c in self.paulis.items()}
        result.update(self.local_factors)
        return result

    def __repr__(self):
        paulis = " ".join(f"{PAULI_LABELS[c]}@{s}" for s, c in self.paulis.items())
        local = " ".join(f"A@{s}" for s in self.local_factors)
        return f"{self.prefactor} * {paulis or 'I'} (x) {local or 'I'}"

    def acts_over(self) -> frozenset:
        """Sites with a factor."""
        return frozenset(self.paulis) | frozenset(self.local_factors)

    def code_in(self, system: SystemDescriptor) -> int:
        """The Pauli code in the site order of ``system``."""
        return _remap_code(self.code, self.system, system)

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def __bool__(self):
        return bool(self.prefactor)

    def __neg__(self):
        return self._new(self.code, self.local_factors, -self.prefactor, self.system)

    def dag(self):
        """Adjoint: the parts commute, so each one is conjugated in place."""
        prefactor = self.prefactor
        if isinstance(prefactor, complex):
            prefactor = prefactor.conjugate()
        local = {site: op.conj().T for site, op in self.local_factors.items()}
        return self._new(self.code, local, prefactor, self.system)

    def flat(self):
        """Return the simplest equivalent operator."""
        return self.simplify()

    def simplify(self) -> Operator:
        """Collapse to a Pauli string or a plain product when one part is trivial."""
        system = self.system
        if not self.prefactor:
            return ScalarOperator(0, system)
        if not self.local_factors:
            if self.code == 0:
                return ScalarOperator(self.prefactor, system)
            return PauliStringOperator._new(self.code, self.prefactor, system)
        if self.code == 0:
            return _plain_product(dict(self.local_factors), self.prefactor, system)
        return self

    def tr(self) -> complex:
        """Trace: zero unless the Pauli part is the identity."""
        if self.code:
            return 0
        dims = self.system.dimensions
        result = complex(self.prefactor)
        for site, dim in dims.items():
            factor = self.local_factors.get(site, None)
            result *= dim if factor is None else np.trace(factor)
        return result

    def tidyup(self, atol=None):
        """Drop a negligible prefactor."""
        if atol is None:
            atol = QALMA_TOLERANCE
        if abs(self.prefactor) < atol:
            return ScalarOperator(0, self.system)
        return self

    def to_product_operator(self) -> ProductOperator:
        """Return the equivalent plain :class:`ProductOperator`."""
        return ProductOperator(dict(self.site_factors), self.prefactor, self.system)

    def _set_system_(self, system=None):
        """Change the system, keeping the meaning of the operator."""
        if system is not None and self.system is not None:
            self.code = _remap_code(self.code, self.system, system)
        for key in ("paulis", "site_factors", "site_factors_qutip"):
            self.__dict__.pop(key, None)
        self._set_system_attrs(system)
        return self

    def __getstate__(self):
        """Pickle the parts and the system, not the caches."""
        return {
            "code": self.code,
            "local_factors": self.local_factors,
            "prefactor": self.prefactor,
            "system": self.system,
        }

    def __setstate__(self, state):
        self._init_fast(
            state["code"], state["local_factors"], state["prefactor"], state["system"]
        )


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

_Parts = Tuple[int, Dict[str, np.ndarray], complex]


def _plain_product(
    factors: Dict[str, np.ndarray], prefactor: complex, system: SystemDescriptor
) -> Operator:
    """A product of local factors, as the simplest operator type."""
    if not prefactor:
        return ScalarOperator(0, system)
    if not factors:
        return ScalarOperator(prefactor, system)
    if len(factors) == 1:
        ((site, factor),) = factors.items()
        return LocalOperator(site, factor * prefactor, system)
    return ProductOperator(factors, prefactor, system)


def _parts(operator: Operator, system: SystemDescriptor) -> Optional[_Parts]:
    """Split ``operator`` into ``(code, local factors, prefactor)`` in ``system``."""
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


def _dense_product(x_op: Operator, y_op: Operator, system) -> Operator:
    """Fallback: multiply as plain products of dense factors."""

    def as_product(op):
        if isinstance(op, LocalOperator):
            return ProductOperator({op.site: _to_array(op.operator)}, 1.0, system)
        return ProductOperator(dict(op.site_factors), op.prefactor, system)

    factors = dict(as_product(x_op).site_factors)
    for site, factor in as_product(y_op).site_factors.items():
        factors[site] = factors[site] @ factor if site in factors else factor
    prefactor = (x_op.prefactor if not isinstance(x_op, LocalOperator) else 1.0) * (
        y_op.prefactor if not isinstance(y_op, LocalOperator) else 1.0
    )
    return _plain_product(factors, prefactor, system)


def tensor_product(x_op: Operator, y_op: Operator) -> Operator:
    """Product of two operators among Pauli strings, tensor products and products.

    If the Pauli support of each operand is disjoint from the local support
    of the other, the Pauli parts are multiplied on their codes and the
    local parts as matrices. Otherwise both are multiplied as dense
    products.
    """
    system = x_op.system.union(y_op.system)
    x_parts, y_parts = _parts(x_op, system), _parts(y_op, system)
    if x_parts is None or y_parts is None:
        raise TypeError(f"cannot multiply {type(x_op)} and {type(y_op)} here.")
    x_code, x_local, x_pref = x_parts
    y_code, y_local, y_pref = y_parts
    x_paulis = _pauli_sites(x_code, system)
    y_paulis = _pauli_sites(y_code, system)
    if any(site in y_local for site in x_paulis) or any(
        site in x_local for site in y_paulis
    ):
        return _dense_product(x_op, y_op, system)
    k, code = _product_phase(x_code, y_code, _site_index(system).mask)
    prefactor = x_pref * y_pref * _I_POWERS[k]
    local = dict(x_local)
    for site, factor in y_local.items():
        local[site] = local[site] @ factor if site in local else factor
    if not prefactor:
        return ScalarOperator(0, system)
    if not local:
        if code == 0:
            return ScalarOperator(prefactor, system)
        return PauliStringOperator._new(code, prefactor, system)
    if code == 0:
        return _plain_product(local, prefactor, system)
    return TensorProductOperator._new(code, local, prefactor, system)


def tensor_scale(operator: TensorProductOperator, value: Union[complex, float]):
    """Return ``value * operator``."""
    if not value:
        return ScalarOperator(0, operator.system)
    return TensorProductOperator._new(
        operator.code,
        operator.local_factors,
        operator.prefactor * value,
        operator.system,
    )
