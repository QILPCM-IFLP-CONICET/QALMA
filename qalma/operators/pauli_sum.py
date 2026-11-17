r"""Linear combinations of Pauli strings, stored as arrays.

A :class:`PauliSumOperator` represents

.. math::

    O = \sum_{a=1}^{m} c_a P_a ,

with :math:`P_a` Pauli strings encoded as in
:mod:`qalma.operators.pauli` (two bits per site, ``00, 01, 10, 11`` for
:math:`1, \sigma^x, \sigma^y, \sigma^z`). The codes are stored in an array
``codes`` of shape ``(m, W)`` and dtype ``uint64`` (``W = ceil(2n/64)``
words per string, least significant word first), and the coefficients in an
array ``coeffs`` of shape ``(m,)``. The rows are sorted and unique, and
negligible coefficients are dropped.

Products, commutators and anticommutators between sums are computed for
all the pairs of terms at once with bitwise operations and population
counts, so they cost :math:`O(m_1 m_2 W)` word operations, followed by a
sort to merge equal strings.
"""

from functools import cached_property
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import qutip  # type: ignore[import-untyped]
from scipy import sparse

from qalma.model import SystemDescriptor
from qalma.settings import QALMA_TOLERANCE

from .arithmetic import SumOperator
from .basic import Operator
from .pauli import (
    PAULI_LABELS,
    PauliStringOperator,
    _remap_code,
    _site_index,
)
from .product import ScalarOperator

__all__ = [
    "PauliSumOperator",
    "as_pauli_sum",
    "permute_codes",
    "to_pauli_sum_operator",
    "words_for",
]

_WORD_BITS = 64
_LOW_BITS = np.uint64(0x5555_5555_5555_5555)
_I_POWERS_ARRAY = np.array([1.0, 1j, -1.0, -1j], dtype=complex)
# Number of pair products evaluated at once in products of sums.
_CHUNK_SIZE = 1 << 20

_BYTE_POPCOUNT = np.array([bin(k).count("1") for k in range(256)], dtype=np.uint8)


def _popcount_bytes(array: np.ndarray) -> np.ndarray:
    """Number of set bits of each element, with a lookup table (numpy < 2.0)."""
    array = np.ascontiguousarray(array)
    counts = _BYTE_POPCOUNT[array.view(np.uint8)]
    return counts.reshape(array.shape + (array.itemsize,)).sum(axis=-1, dtype=np.int64)


# Number of set bits of each element.
_popcount = getattr(np, "bitwise_count", _popcount_bytes)


def words_for(num_sites: int) -> int:
    """Number of 64-bit words of the code of a string on ``num_sites`` sites."""
    return max(1, -(-2 * num_sites // _WORD_BITS))


def _num_words(system: SystemDescriptor) -> int:
    return words_for(len(system.sites))


def permute_codes(
    codes: np.ndarray,
    src_order: Sequence[str],
    dst_position: Mapping[str, int],
    dst_words: int,
) -> np.ndarray:
    """Rewrite codes from the site order ``src_order`` to the positions ``dst_position``.

    ``codes`` has shape ``(m, W)``: the bits ``(2k+1, 2k)`` refer to the site
    ``src_order[k]``. Raises ``ValueError`` if a string acts on a site that
    is not in ``dst_position``, or has bits beyond ``len(src_order)`` sites.
    """
    result = np.zeros((len(codes), dst_words), dtype=np.uint64)
    if len(codes) == 0:
        return result
    codes = np.asarray(codes, dtype=np.uint64).reshape(len(codes), -1)
    src_order = list(src_order)
    # Bits beyond the last site.
    used_bits = 2 * len(src_order)
    for word in range(codes.shape[1]):
        low = word * _WORD_BITS
        if used_bits >= low + _WORD_BITS:
            continue
        valid = 0 if used_bits <= low else (1 << (used_bits - low)) - 1
        if (codes[:, word] & np.uint64(~valid & 0xFFFF_FFFF_FFFF_FFFF)).any():
            raise ValueError(f"the codes act on more than {len(src_order)} sites.")
    same_order = all(dst_position.get(site) == k for k, site in enumerate(src_order))
    if same_order:
        width = min(codes.shape[1], dst_words)
        result[:, :width] = codes[:, :width]
        return result
    for k, site in enumerate(src_order):
        word, shift = divmod(2 * k, _WORD_BITS)
        local = (codes[:, word] >> np.uint64(shift)) & np.uint64(3)
        if not local.any():
            continue
        position = dst_position.get(site, None)
        if position is None:
            raise ValueError(
                f"the codes act on the site <<{site}>>, which is not available."
            )
        new_word, new_shift = divmod(2 * position, _WORD_BITS)
        result[:, new_word] |= local << np.uint64(new_shift)
    return result


def _int_to_words(code: int, num_words: int) -> np.ndarray:
    return np.array(
        [(code >> (_WORD_BITS * w)) & 0xFFFF_FFFF_FFFF_FFFF for w in range(num_words)],
        dtype=np.uint64,
    )


def _words_to_int(row: np.ndarray) -> int:
    result = 0
    for w, word in enumerate(row):
        result |= int(word) << (_WORD_BITS * w)
    return result


def _symplectic(codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split interleaved codes into the ``x`` and ``z`` bit masks."""
    high = (codes >> np.uint64(1)) & _LOW_BITS
    return (codes ^ high) & _LOW_BITS, high


def _count(array: np.ndarray) -> np.ndarray:
    """Population count summed over the last (word) axis."""
    return _popcount(array).sum(axis=-1, dtype=np.int64)


def _reduce_rows(
    codes: np.ndarray, coeffs: np.ndarray, scale: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge equal rows: sort, add the coefficients, keep the largest scale."""
    if len(codes) == 0:
        return codes, coeffs, scale
    if codes.shape[1] == 1:
        unique, inverse = np.unique(codes[:, 0], return_inverse=True)
        unique = unique.reshape(-1, 1)
    else:
        unique, inverse = np.unique(codes, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    size = len(unique)
    summed = np.bincount(
        inverse, weights=coeffs.real, minlength=size
    ) + 1j * np.bincount(inverse, weights=coeffs.imag, minlength=size)
    new_scale = np.zeros(size)
    np.maximum.at(new_scale, inverse, scale)
    return unique, summed, new_scale


def _drop_small(
    codes: np.ndarray, coeffs: np.ndarray, scale: np.ndarray, rtol: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop the coefficients that vanish up to rounding."""
    keep = np.abs(coeffs) > rtol * scale
    keep &= coeffs != 0
    return codes[keep], coeffs[keep]


class PauliSumOperator(SumOperator):
    r"""Sum :math:`\sum_a c_a P_a` of Pauli strings with complex coefficients.

    Subclass of :class:`SumOperator`: the attribute ``terms`` (a tuple of
    :class:`PauliStringOperator` and, for the identity, a
    :class:`ScalarOperator`) is built on demand, so every routine written
    for sums accepts it. Products, sums, commutators, anticommutators,
    adjoints, traces, partial traces, expectation values over product
    states and the conversion to QuTiP are computed directly on the arrays.

    Parameters
    ----------
    codes : array of shape (m, W) and dtype uint64, or sequence of int
        The strings, either as words (least significant first) or as
        Python integers.
    coeffs : array of shape (m,)
        The coefficients.
    system : SystemDescriptor
        The system. Its sites must be two-level sites.
    simplify : bool
        If ``True`` (default), merge equal strings and drop negligible
        coefficients. Pass ``False`` only for arrays already in canonical
        form.
    """

    codes: np.ndarray
    coeffs: np.ndarray

    def __init__(
        self,
        codes,
        coeffs,
        system: SystemDescriptor,
        simplify: bool = True,
    ):
        # SumOperator.__init__ is not called: ``terms`` is derived on demand.
        assert system is not None
        num_words = _num_words(system)
        coeffs = np.asarray(coeffs, dtype=complex).reshape(-1)
        if len(coeffs) == 0:
            codes = np.zeros((0, num_words), dtype=np.uint64)
        elif isinstance(codes, np.ndarray) and codes.dtype == np.uint64:
            codes = codes.reshape(
                len(coeffs), codes.shape[-1] if codes.ndim > 1 else -1
            )
            if codes.shape[1] < num_words:
                codes = np.hstack(
                    [
                        codes,
                        np.zeros((len(codes), num_words - codes.shape[1]), np.uint64),
                    ]
                )
        else:
            codes = np.array(
                [_int_to_words(int(c), num_words) for c in codes], dtype=np.uint64
            ).reshape(len(coeffs), num_words)
        if simplify:
            codes, coeffs, scale = _reduce_rows(codes, coeffs, np.abs(coeffs))
            codes, coeffs = _drop_small(codes, coeffs, scale, QALMA_TOLERANCE)
        self.codes = codes
        self.coeffs = coeffs
        self.system = system
        self._isherm = None
        self._isdiagonal = None
        self._simplified = True

    # ------------------------------------------------------------------
    # Constructors and conversions
    # ------------------------------------------------------------------

    @classmethod
    def _new(cls, codes, coeffs, system) -> "PauliSumOperator":
        return cls(codes, coeffs, system, simplify=False)

    @classmethod
    def from_dict(
        cls, coefficients: Mapping[int, complex], system: SystemDescriptor
    ) -> "PauliSumOperator":
        """Build from a mapping ``code -> coefficient``."""
        return cls(list(coefficients.keys()), list(coefficients.values()), system)

    @classmethod
    def from_operator(cls, operator: Operator) -> "PauliSumOperator":
        """Expand ``operator`` in Pauli strings (see :func:`to_pauli_sum`)."""
        result = as_pauli_sum(operator)
        if result is None:
            from .pauli import to_pauli_sum

            result = as_pauli_sum(to_pauli_sum(operator))
        assert result is not None
        return result

    def to_dict(self) -> Dict[int, complex]:
        """Return the mapping ``code -> coefficient``."""
        return {
            _words_to_int(row): complex(c) for row, c in zip(self.codes, self.coeffs)
        }

    def codes_in(self, system: SystemDescriptor) -> np.ndarray:
        """Return the codes in the site order (and word size) of ``system``."""
        codes = self.codes
        num_words = _num_words(system)
        if self.system is system or (
            _site_index(self.system).order == _site_index(system).order
        ):
            if codes.shape[1] == num_words:
                return codes
            if codes.shape[1] < num_words:
                return np.hstack(
                    [
                        codes,
                        np.zeros((len(codes), num_words - codes.shape[1]), np.uint64),
                    ]
                )
        return np.array(
            [
                _int_to_words(
                    _remap_code(_words_to_int(row), self.system, system), num_words
                )
                for row in codes
            ],
            dtype=np.uint64,
        ).reshape(len(codes), num_words)

    def _local_codes(self, position: int) -> np.ndarray:
        """Pauli code (0..3) of every term on the site at ``position``."""
        word, shift = divmod(2 * position, _WORD_BITS)
        return ((self.codes[:, word] >> np.uint64(shift)) & np.uint64(3)).astype(
            np.int64
        )

    @cached_property
    def terms(self) -> Tuple[Operator, ...]:  # type: ignore[override]
        """The terms, as Pauli strings (and a scalar for the identity)."""
        system = self.system
        result: List[Operator] = []
        for row, coeff in zip(self.codes, self.coeffs):
            code = _words_to_int(row)
            if code:
                result.append(PauliStringOperator._new(code, complex(coeff), system))
            else:
                result.append(ScalarOperator(complex(coeff), system))
        return tuple(result)

    def __len__(self):
        return len(self.coeffs)

    def num_terms(self) -> int:
        """Number of strings in the sum."""
        return len(self.coeffs)

    def __bool__(self):
        return len(self.coeffs) > 0

    @property
    def is_zero(self) -> bool:
        """``True`` if the sum has no terms."""
        return len(self.coeffs) == 0

    def __repr__(self):
        order = _site_index(self.system).order
        lines = []
        for row, coeff in zip(self.codes, self.coeffs):
            code, k, factors = _words_to_int(row), 0, []
            while code:
                if code & 3:
                    factors.append(f"{PAULI_LABELS[code & 3]}@{order[k]}")
                code >>= 2
                k += 1
            lines.append(f"  ({complex(coeff)}) * {' '.join(factors) or 'I'}")
        return "PauliSumOperator(\n" + "\n".join(lines) + "\n)"

    def collapse(self) -> Operator:
        """Return a :class:`ScalarOperator` or a :class:`PauliStringOperator` if possible."""
        size = len(self.coeffs)
        if size == 0:
            return ScalarOperator(0, self.system)
        if size == 1:
            return self.terms[0]
        return self

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------

    def _binary(self, other: "PauliSumOperator", mode: str) -> Operator:
        """Product (``mode='mul'``), commutator or anticommutator with ``other``."""
        system = self.system.union(other.system)
        codes_1, codes_2 = self.codes_in(system), other.codes_in(system)
        coeffs_1, coeffs_2 = self.coeffs, other.coeffs
        if len(coeffs_1) == 0 or len(coeffs_2) == 0:
            return ScalarOperator(0, system)
        x_2, z_2 = _symplectic(codes_2)
        self_2 = _count(x_2 & z_2)
        factor = 1.0 if mode == "mul" else 2.0
        rows_per_chunk = max(1, _CHUNK_SIZE // len(coeffs_2))
        pieces = []
        for start in range(0, len(coeffs_1), rows_per_chunk):
            c_1 = codes_1[start : start + rows_per_chunk]
            a_1 = coeffs_1[start : start + rows_per_chunk]
            x_1, z_1 = _symplectic(c_1)
            self_1 = _count(x_1 & z_1)
            x_1, z_1 = x_1[:, None, :], z_1[:, None, :]
            cross = _count(z_1 & x_2[None, :, :])
            if mode == "mul":
                selected = None
            else:
                anticommute = (cross + _count(x_1 & z_2[None, :, :])) & 1
                selected = anticommute == (1 if mode == "commutator" else 0)
                if not selected.any():
                    continue
            codes_3 = c_1[:, None, :] ^ codes_2[None, :, :]
            x_3, z_3 = _symplectic(codes_3)
            phase = (
                self_1[:, None] + self_2[None, :] - _count(x_3 & z_3) + 2 * cross
            ) & 3
            values = factor * a_1[:, None] * coeffs_2[None, :] * _I_POWERS_ARRAY[phase]
            if selected is None:
                codes_3 = codes_3.reshape(-1, codes_3.shape[-1])
                values = values.reshape(-1)
            else:
                codes_3, values = codes_3[selected], values[selected]
            pieces.append(_reduce_rows(codes_3, values, np.abs(values)))
        if not pieces:
            return ScalarOperator(0, system)
        codes = np.concatenate([p[0] for p in pieces])
        values = np.concatenate([p[1] for p in pieces])
        scale = np.concatenate([p[2] for p in pieces])
        codes, values, scale = _reduce_rows(codes, values, scale)
        codes, values = _drop_small(codes, values, scale, QALMA_TOLERANCE)
        return PauliSumOperator._new(codes, values, system).collapse()

    def mul(self, other: "PauliSumOperator") -> Operator:
        """Operator product ``self * other``."""
        return self._binary(other, "mul")

    def commutator(self, other: "PauliSumOperator") -> Operator:
        r"""Commutator :math:`[\mathrm{self}, \mathrm{other}]`."""
        return self._binary(other, "commutator")

    def anticommutator(self, other: "PauliSumOperator") -> Operator:
        r"""Anticommutator :math:`\{\mathrm{self}, \mathrm{other}\}`."""
        return self._binary(other, "anticommutator")

    def add(self, other: "PauliSumOperator") -> Operator:
        """Sum ``self + other``."""
        system = self.system.union(other.system)
        codes = np.concatenate([self.codes_in(system), other.codes_in(system)])
        coeffs = np.concatenate([self.coeffs, other.coeffs])
        return PauliSumOperator(codes, coeffs, system).collapse()

    def scale(self, value: complex) -> Operator:
        """Return ``value * self``."""
        if not value:
            return ScalarOperator(0, self.system)
        return PauliSumOperator._new(self.codes, self.coeffs * value, self.system)

    def __neg__(self):
        return PauliSumOperator._new(self.codes, -self.coeffs, self.system)

    def __pow__(self, exp) -> Operator:
        if isinstance(exp, (int, np.integer)) and exp >= 1:
            result: Operator = self
            for _ in range(exp - 1):
                result = result * self
            return result
        return super().__pow__(exp)

    def dag(self):
        """Adjoint: Pauli strings are hermitian, so only the coefficients change."""
        return PauliSumOperator._new(self.codes, self.coeffs.conj(), self.system)

    def hermitian_part(self):
        """Hermitian part: the real part of the coefficients."""
        return PauliSumOperator(self.codes, self.coeffs.real, self.system).collapse()

    @property
    def isherm(self) -> bool:  # type: ignore[override]
        """``True`` if every coefficient is real (up to ``QALMA_TOLERANCE``)."""
        scale = max(1.0, float(np.abs(self.coeffs).max(initial=0.0)))
        return bool(np.all(np.abs(self.coeffs.imag) <= QALMA_TOLERANCE * scale))

    @property
    def isdiagonal(self) -> bool:  # type: ignore[override]
        """``True`` if no term has a :math:`\\sigma^x` or :math:`\\sigma^y` factor."""
        x_mask, _ = _symplectic(self.codes)
        return not x_mask.any()

    def acts_over(self) -> frozenset:
        """Sites where some term acts nontrivially."""
        if len(self.coeffs) == 0:
            return frozenset()
        support = np.bitwise_or.reduce(self.codes, axis=0)
        code = _words_to_int(support)
        order = _site_index(self.system).order
        result, k = [], 0
        while code:
            if code & 3:
                result.append(order[k])
            code >>= 2
            k += 1
        return frozenset(result)

    def flat(self):
        """Return ``self``: the sum is already flat."""
        return self

    def simplify(self):
        """Return the simplest equivalent type (scalar, string or ``self``)."""
        return self.collapse()

    def tidyup(self, atol=None):
        """Drop coefficients (or their real/imaginary parts) below ``atol``."""
        if atol is None:
            atol = QALMA_TOLERANCE
        coeffs = self.coeffs
        real = np.where(np.abs(coeffs.real) < atol, 0.0, coeffs.real)
        imag = np.where(np.abs(coeffs.imag) < atol, 0.0, coeffs.imag)
        return PauliSumOperator(self.codes, real + 1j * imag, self.system).collapse()

    def tr(self) -> complex:
        """Trace: the coefficient of the identity times the dimension."""
        identity = ~self.codes.any(axis=1)
        return complex(self.coeffs[identity].sum()) * 2 ** len(self.system.sites)

    def norm(self, ord=None):
        """Frobenius norm computed from the coefficients; other norms fall back."""
        if ord == "fro":
            dim = 2.0 ** len(self.system.sites)
            return float(np.sqrt(dim * np.sum(np.abs(self.coeffs) ** 2)))
        return super().norm(ord)

    def _support_mask(self, sites: Iterable[str]) -> np.ndarray:
        """Codes with ones on the two bits of every site in ``sites``."""
        position = _site_index(self.system).position
        code = 0
        for site in sites:
            code |= 3 << (2 * position[site])
        return _int_to_words(code, self.codes.shape[1])

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Partial trace over the sites not in ``sites``."""
        if isinstance(sites, SystemDescriptor):
            subsystem = sites
            sites = frozenset(sites.sites)
        else:
            subsystem = self.system.subsystem(sites)
        traced = [s for s in self.system.sites if s not in sites]
        outside = self._support_mask(traced)
        keep = ~((self.codes & outside[None, :]).any(axis=1))
        factor = 2 ** len(traced)
        reduced = PauliSumOperator._new(
            self.codes[keep], self.coeffs[keep] * factor, self.system
        )
        if not reduced:
            return ScalarOperator(0, subsystem)
        return PauliSumOperator._new(
            reduced.codes_in(subsystem), reduced.coeffs, subsystem
        ).collapse()

    def reduce(self, sites: Iterable, state=None):
        """Normalized partial trace (``state=None``), or the generic reduction."""
        if state is not None:
            return SumOperator.reduce(self, sites, state)
        sites = frozenset(sites)
        outside = self._support_mask(s for s in self.system.sites if s not in sites)
        keep = ~((self.codes & outside[None, :]).any(axis=1))
        return PauliSumOperator._new(
            self.codes[keep], self.coeffs[keep], self.system
        ).collapse()

    def expect_product(self, local_states: Mapping[str, np.ndarray]) -> complex:
        r"""Expectation value over a product state :math:`\bigotimes_i \rho_i`.

        ``local_states`` maps each site to its (2x2) local density matrix;
        missing sites are taken as the maximally mixed state.
        """
        from .pauli import PAULI_MATRICES

        values = self.coeffs.copy()
        position = _site_index(self.system).position
        for site in self.acts_over():
            rho = local_states.get(site, None)
            if rho is None:
                table = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
            else:
                rho = np.asarray(rho)
                table = np.array([complex((rho * p.T).sum()) for p in PAULI_MATRICES])
            values = values * table[self._local_codes(position[site])]
        return complex(values.sum())

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Sparse QuTiP representation, built directly from the codes.

        Each string is a signed permutation matrix:
        :math:`X^x Z^z |j\\rangle = (-1)^{|j \\wedge z|} |j \\oplus x\\rangle`.
        """
        system = self.system
        sites = system.sites
        acts_over = self.acts_over()
        if block is None:
            block = tuple(sorted(sites))
        else:
            block = tuple(s for s in block if s in sites) + tuple(
                sorted(s for s in acts_over if s not in block)
            )
        size = len(block)
        if size == 0:
            return complex(self.coeffs[~self.codes.any(axis=1)].sum())
        dimensions = system.dimensions
        if any(dimensions[site] != 2 for site in block):
            # Sites of other dimensions (e.g. the rest subsites of a split
            # site) are outside the strings: build the matrix on the sites
            # the strings act on, and add identities on the others.
            acted = tuple(site for site in block if site in acts_over)
            others = tuple(site for site in block if site not in acts_over)
            identities = [qutip.qeye(dimensions[site]) for site in others]
            if not acted:
                scalar = complex(self.coeffs[~self.codes.any(axis=1)].sum())
                return scalar * qutip.tensor(*identities)
            result = qutip.tensor(self.to_qutip(acted), *identities)
            order = acted + others
            return result.permute([order.index(site) for site in block])
        if size > 62:
            raise ValueError("too many sites for a matrix representation.")
        position = _site_index(system).position
        x_mask = np.zeros(len(self.coeffs), dtype=np.int64)
        z_mask = np.zeros(len(self.coeffs), dtype=np.int64)
        for index, site in enumerate(block):
            if site not in acts_over:
                continue
            local = self._local_codes(position[site])
            bit = np.int64(1) << np.int64(size - 1 - index)
            x_mask |= np.where((local == 1) | (local == 2), bit, 0)
            z_mask |= np.where((local == 2) | (local == 3), bit, 0)
        dim = 1 << size
        columns = np.arange(dim, dtype=np.int64)
        phase = _I_POWERS_ARRAY[_popcount(x_mask & z_mask).astype(np.int64) & 3]
        weights = self.coeffs * phase
        matrix = sparse.csr_matrix((dim, dim), dtype=complex)
        terms_per_chunk = max(1, _CHUNK_SIZE // dim)
        for start in range(0, len(weights), terms_per_chunk):
            stop = start + terms_per_chunk
            rows = columns[None, :] ^ x_mask[start:stop, None]
            signs = 1 - 2 * (
                _popcount(columns[None, :] & z_mask[start:stop, None]).astype(np.int64)
                & 1
            )
            data = weights[start:stop, None] * signs
            cols = np.broadcast_to(columns, rows.shape)
            matrix = matrix + sparse.csr_matrix(
                (data.ravel(), (rows.ravel(), cols.ravel())), shape=(dim, dim)
            )
        return qutip.Qobj(matrix, dims=[[2] * size, [2] * size])

    def _set_system_(self, system=None):
        """Change the system, keeping the meaning of the strings."""
        if system is not None and self.system is not None:
            self.codes = self.codes_in(system)
        self.__dict__.pop("terms", None)
        self.system = system
        return self

    def __getstate__(self):
        """Pickle only the arrays and the system, not the cached terms."""
        state = self.__dict__.copy()
        state.pop("terms", None)
        return state


def to_pauli_sum_operator(operator) -> PauliSumOperator:
    """Like :func:`as_pauli_sum`, but raise ``TypeError`` instead of returning ``None``."""
    result = as_pauli_sum(operator)
    if result is None:
        raise TypeError(f"{type(operator).__name__} is not a sum of Pauli strings.")
    return result


def as_pauli_sum(operator) -> Optional[PauliSumOperator]:
    """Return ``operator`` as a :class:`PauliSumOperator`, or ``None``.

    Accepts Pauli strings, scalars, Pauli sums and (nested) sums of them.
    It does not expand other operators: use :func:`to_pauli_sum` for that.
    """
    if isinstance(operator, PauliSumOperator):
        return operator
    system = operator.system
    if isinstance(operator, PauliStringOperator):
        return PauliSumOperator([operator.code], [operator.prefactor], system)
    if isinstance(operator, ScalarOperator):
        return PauliSumOperator([0], [operator.prefactor], system)
    if isinstance(operator, SumOperator):
        parts = [as_pauli_sum(term) for term in operator.terms]
        if any(part is None for part in parts):
            return None
        num_words = _num_words(system)
        codes = np.concatenate(
            [p.codes_in(system) for p in parts]  # type: ignore[union-attr]
        ).reshape(-1, num_words)
        coeffs = np.concatenate([p.coeffs for p in parts])  # type: ignore[union-attr]
        return PauliSumOperator(codes, coeffs, system)
    return None
