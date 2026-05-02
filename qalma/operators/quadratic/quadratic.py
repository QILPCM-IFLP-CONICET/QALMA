r"""QuadraticForm Operators.

Quadratic Form Operators provides a representation for quantum operators
of the form

Q= L + \sum_a w_a M_a ^2 + \delta Q

with L and M_a one-body operators, w_a certain weights and \delta Q a
*remainder* as a sum of n-body terms.
"""

from numbers import Number

# from numbers import Number
from typing import Callable, Iterable, Optional, Set, Tuple, Union, cast

import numpy as np

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
from qalma.settings import QALMA_TOLERANCE

# from typing import Union

__all__ = ["QuadraticFormOperator"]


class QuadraticFormOperator(Operator):
    r"""Represent a two-body operator as a sum of squares with coefficients.

    The operator ``T`` is represented in terms of operators $Q_\alpha$ and
    weights $w_\alpha$ s.t.

    .. math::

        T = sum_alpha w_alpha * Q_alpha^2

    with $Q_alpha$ a local operator or a one body operator.
    """

    system: SystemDescriptor
    terms: list
    weights: list
    offset: Optional[Operator]

    def __init__(self, basis, weights, system=None, linear_term=None, offset=None):
        r"""Initializae a QuadraticFormOperator.

        Parameters
        ----------
        basis : tuple[Operator, ...]
            Tuple of Hermitian one-body operators :math:`Q_\\alpha`. Each
            contributes a term :math:`w_\\alpha Q_\\alpha^2` to the operator.
        weights : tuple[complex, ...]
            Scalar weights :math:`w_\\alpha`, one per basis element.
        system : SystemDescriptor or None, optional
            Descriptor of the full lattice system. Inferred from ``basis``
            if not provided.
        linear_term : OneBodyOperator, LocalOperator, ScalarOperator, or None, optional
            The one-body part :math:`L` of the operator. Default is ``None``.
        offset : Operator or None, optional
            Additional remainder term :math:`\\delta Q` not captured by the
            quadratic or linear parts. Default is ``None``.

        """
        # If the system is not given, infer it from the terms
        if offset:
            offset = offset.simplify()
        if linear_term:
            linear_term = linear_term.simplify()
            assert (
                isinstance(linear_term, OneBodyOperator)
                or len(linear_term.acts_over()) < 2
            )
        self._isherm = None
        assert isinstance(basis, tuple)
        assert isinstance(weights, tuple)
        for pos, gen in enumerate(basis):
            assert (
                gen.isherm
            ), f"Operator at pos {pos} got {gen.isherm}\n{gen}"  # TODO: REMOVE ME
        assert (
            isinstance(linear_term, (OneBodyOperator, LocalOperator, ScalarOperator))
            or linear_term is None
        ), f"{type(offset)} should be a LocalOperator or a OneBodyOperator"
        if system is None:
            for term in basis:
                if system is None:
                    system = term.system
                else:
                    system = system.union(term.system)

        # If check_and_simplify, ensure that all the terms are
        # one-body operators and try to use the simplified forms
        # of the operators.

        self.weights = weights
        self.basis = basis
        self.system = system
        self.offset = offset
        self.linear_term = linear_term
        self._simplified = False

    def __bool__(self):
        """Convert to bool."""
        for term in (self.linear_term, self.offset):
            if term is not None:
                if not term.is_zero:
                    return True
        return len(self.weights) > 0 and any(self.weights) and any(self.basis)

    def __add__(self, other):
        """Add with another operator."""
        # TODO: remove me and fix the sums
        if not bool(other):
            return self
        if isinstance(other, Number):
            other = ScalarOperator(other, system=self.system)

        assert isinstance(other, Operator), "other must be an operator."
        system = self.system or other.system
        if isinstance(other, QuadraticFormOperator):
            basis = self.basis + other.basis
            weights = self.weights + other.weights
            offset = self.offset
            linear_term = self.linear_term
            if offset is None:
                offset = other.offset
            else:
                if other.offset is not None:
                    offset = offset + other.offset

            if linear_term is None:
                offset = other.linear_term
            else:
                if other.linear_term is not None:
                    linear_term = linear_term + other.linear_term
            return QuadraticFormOperator(basis, weights, system, linear_term, offset)
        if isinstance(
            other,
            (
                ScalarOperator,
                LocalOperator,
                OneBodyOperator,
            ),
        ):
            linear_term = self.linear_term
            linear_term = (
                other if linear_term is None else (linear_term + other).simplify()
            )
            basis = self.basis
            weights = self.weights
            return QuadraticFormOperator(
                basis, weights, system, linear_term, offset=None
            )
        return SumOperator(
            (
                self,
                other,
            ),
            system,
        )

    def __mul__(self, other):
        """Multiply with another operator by the right."""
        system = self.system
        if isinstance(other, ScalarOperator):
            other = other.prefactor
            system = system or other.system
        if isinstance(other, (float, complex)):
            offset = self.offset
            if offset is not None:
                offset = offset * other
            linear_term = self.linear_term
            if linear_term is not None:
                linear_term = (linear_term * other).simplify()

            return QuadraticFormOperator(
                self.basis,
                tuple(w * other for w in self.weights),
                system,
                linear_term=linear_term,
                offset=offset,
            )
        standard_repr = self.as_sum_of_products(False).simplify()
        return standard_repr * other

    def __neg__(self):
        """Represent the additive opposite."""
        offset = self.offset
        if offset is not None:
            offset = -offset
        linear_term = self.linear_term
        if linear_term is not None:
            linear_term = -linear_term
        return QuadraticFormOperator(
            self.basis,
            tuple(-w for w in self.weights),
            self.system,
            linear_term,
            offset,
        )

    def _set_system_(self, system=None):
        # pylint: disable=protected-access
        self.system = system
        for basis_elem in self.basis:
            basis_elem._set_system_(system)

        offset = self.offset
        linear_term = self.linear_term
        if offset is not None:
            offset._set_system_(system)
        if linear_term is not None:
            linear_term._set_system_(system)
        return self

    def acts_over(self) -> frozenset:
        """Set of sites over the state acts."""
        result: Set[str] = set()
        for term in self.basis:
            try:
                result = result.union(term.acts_over())
            except TypeError:
                return frozenset(self.system.sites)

        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            try:
                result = result.union(term.acts_over())
            except TypeError:
                return frozenset(self.system.sites)
        return frozenset(result)

    def as_sum_of_products(
        self, simplify: bool = True
    ) -> ProductOperator | LocalOperator | SumOperator:
        """Convert to a linear combination of two-body operators."""
        isherm = self._isherm
        isdiag = self.isdiagonal
        if all(b_op.isherm for b_op in self.basis):
            terms = tuple(
                (
                    ((op_term.dag() * op_term) * w)
                    for w, op_term in zip(self.weights, self.basis)
                )
            )
        else:
            terms = tuple(
                (
                    ((op_term.dag() * op_term) * w)
                    for w, op_term in zip(self.weights, self.basis)
                )
            )

        for term in (self.offset, self.linear_term):
            if term is not None:
                terms = terms + (term,)
        if len(terms) == 0:
            return ScalarOperator(0, self.system)
        if len(terms) == 1:
            return terms[0]
        result = SumOperator(terms, self.system, isherm, isdiag)
        if simplify:
            return result.simplify()
        return result

    def dag(self):
        r"""Return the adjoint :math:`O^\\dagger`.

        Conjugates the weights and takes the adjoint of the linear term
        and offset. The basis elements are assumed Hermitian so they are
        unchanged.

        Returns
        -------
        QuadraticFormOperator
            The adjoint operator.

        """
        # pylint: disable=protected-access
        linear_term = self.linear_term
        linear_term = None if linear_term is None else linear_term.dag()
        offset = self.offset
        offset = None if offset is None else offset.dag()
        result = QuadraticFormOperator(
            self.basis,
            tuple((np.conj(w) for w in self.weights)),
            self.system,
            linear_term,
            offset,
        )
        result._simplified = self._simplified
        return result

    def flat(self):
        """Convert to a flat sum of product operators.

        Delegates to :meth:`as_sum_of_products` and then flattens the
        resulting :class:`~qalma.operators.arithmetic.SumOperator`.

        Returns
        -------
        Operator
            A flat :class:`~qalma.operators.arithmetic.SumOperator` with
            no nested sums.

        """
        return self.as_sum_of_products().flat()

    @property
    def isdiagonal(self):
        """True if the operator is diagonal in the product basis."""
        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            isdiagonal = term.isdiagonal
            if not isdiagonal:
                return isdiagonal

        if all(term.isdiagonal for term in self.basis):
            return True
        return False

    @property
    def isherm(self):
        """``True`` if the operator is Hermitian.

        Checks that all weights are real and that the linear term and
        offset (if present) are Hermitian. If inconclusive, converts to
        a sum of products and delegates the check. The result is cached
        in ``_isherm``.
        """
        isherm = self._isherm
        if isherm is not None:
            return isherm

        # We start assumig that the operator is hermitian
        isherm = True
        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            isherm = (isherm and term.isherm) or False

        # Now, let's check the weights
        weights = self.weights
        if len(weights) == 0:
            self._isherm = isherm
            return isherm
        if isherm:
            isherm = all(abs(np.imag(weight)) < QALMA_TOLERANCE for weight in weights)
            if isherm is not None:
                if isherm or len(weights) == 1:
                    self._isherm = isherm
                    return isherm
        # A more drastic approach: convert it to a sum of products
        isherm = self.as_sum_of_products().simplify().isherm or False
        self._isherm = isherm
        return isherm

    def n_body_sector(self) -> int:
        """Return the n-body sector of the operator.

        Returns
        -------
        int
            Always at least ``2`` (the quadratic terms). Returns
            ``max(2, offset.n_body_sector())`` if an offset is present.

        """
        if self.offset is None:
            return 2
        return max(2, self.offset.n_body_sector())

    def num_terms(self) -> int:
        """Return the total number of terms in the operator.

        Counts the quadratic terms plus any terms in the linear part and
        offset.

        Returns
        -------
        int
            Total number of summands.

        """
        num_terms = len(self.weights)
        if self.linear_term:
            num_terms += self.linear_term.num_terms()
        if self.offset:
            num_terms += self.offset.num_terms()
        return num_terms

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        r"""Compute the partial trace over the complement of ``sites``.

        Traces out the linear term and offset analytically. For the
        quadratic terms :math:`w_\\alpha Q_\\alpha^2`, expands each as a
        product and takes the partial trace term by term.

        Parameters
        ----------
        sites : frozenset[str] or SystemDescriptor
            Sites to *keep*. All other sites are traced out.

        Returns
        -------
        Operator
            The reduced operator on the subsystem defined by ``sites``.

        """
        if not isinstance(sites, SystemDescriptor):
            sites = self.system.subsystem(sites)

        result = None
        for term in (self.offset, self.linear_term):
            if term is None:
                continue
            if result:
                tpt = term.partial_trace(sites)
                assert isinstance(tpt, Operator)
                result = result + tpt
            else:
                result = term.partial_trace(sites)
                assert isinstance(
                    result, Operator
                ), f"partial trace of {type(term)} returns {type(result)}"

        if len(self.basis) == 0 and result is None:
            return ScalarOperator(0, sites)

        # TODO: Implement me to return a quadratic operator
        #
        #  (Sum_a  w_a(sum_i L_ai)^2).ptrace = Sum_a w_a ((sum_i L_ai)^2).ptrace
        #  (Sum_i L_ai)^2 = Sum_i (La_i L_aj).ptrace= (La_i1)^2*Tr[1_2] + I Tr(La_i2)^2+...
        #
        terms = tuple(
            w * (op_term * op_term).partial_trace(sites)
            for w, op_term in zip(self.weights, self.basis)
        )
        if result is not None:
            terms = terms + (result,)
        terms = tuple(terms)
        return SumOperator(
            terms,
            sites,
        ).simplify()

    def reduce(self, sites: Iterable, state=None) -> Operator:
        """Compute the reduced operator relative to ``state``.

        The reduced operator is the partial trace of the product of this
        operator and the density operator acting on the complementary
        subsystem. If no state is provided, the result is the partial trace
        divided by the dimension of the traced-out subsystem.

        Parameters
        ----------
        sites : iterable
            The sites to keep.
        state : DensityOperatorProtocol, optional
            The state relative to which the reduction is performed.

        Returns
        -------
        Operator
            The reduced operator acting on the subsystem specified by ``sites``.

        """
        # An alternative would be to build a new QuadraticOperator
        # by reducing the quadratic terms.
        return self.as_sum_of_products().reduce(sites, state)

    def simplify(self):
        """Simplify the operator.

        Build a new representation with a smaller basis.
        """
        # pylint: disable=protected-access
        if self._simplified:
            return self

        result = simplify_quadratic_form(self, hermitic=False)
        result._simplified = True
        return result

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Return a qutip ``Qobj`` object.

        The Qobj acts over the sites listed in `block`.
        By default (`block=None`) returns a qutip object
        acting over all the sites, in lexicographical order.
        """
        sites = self.system.sites
        if block is None:
            block = tuple(sorted(sites))
        else:
            block = block + tuple(
                (site for site in self.acts_over() if site not in block)
            )

        result = sum(
            (op_term.dag() * op_term * w).to_qutip(block)
            for w, op_term in zip(self.weights, self.basis)
        )
        for term in (self.offset, self.linear_term):
            if term is not None:
                result += term.to_qutip(block)
        return result


def one_body_operator_hermitian_hs_sp(x_op: OneBodyOperator, y_op: OneBodyOperator):
    """Hilbert Schmidt scalar product optimized for OneBodyOperators."""
    result: complex = 0
    terms_x: Tuple[ScalarOperator | LocalOperator] = cast(
        Tuple[ScalarOperator | LocalOperator],
        (x_op.terms if isinstance(x_op, OneBodyOperator) else (x_op,)),
    )
    terms_y: Tuple[ScalarOperator | LocalOperator] = cast(
        Tuple[ScalarOperator | LocalOperator],
        (y_op.terms if isinstance(y_op, OneBodyOperator) else (y_op,)),
    )

    for t_1 in terms_x:
        for t_2 in terms_y:
            if isinstance(t_1, ScalarOperator):
                result += t_2.tr() * t_1.prefactor
            elif isinstance(t_2, ScalarOperator):
                result += t_1.tr() * t_2.prefactor
            elif t_1.site == t_2.site:
                result += (t_1.operator @ t_2.operator).trace()
            else:
                result += t_1.operator.trace() * t_2.operator.trace()
    return result


def simplify_quadratic_form(
    operator: QuadraticFormOperator,
    hermitic: bool = True,
    scalar_product: Callable = one_body_operator_hermitian_hs_sp,
):
    """Take a 2-body operator and returns lists weights, ops.

    The original operator is spanned as

    .. code-block:: python

        sum(w * op**2 for w,op in zip(weights,ops))
    """
    from .build import build_quadratic_form_from_operator

    changed = False
    system = operator.system
    if not operator.isherm and hermitic:
        changed = True

    def simplify_other_terms(term):
        """Simplify ``term``, optionally projecting onto its Hermitian part.

        Parameters
        ----------
        term : Operator or None
            The term to simplify. Returns ``None`` unchanged.

        Returns
        -------
        Operator or None
            The simplified (and optionally Hermitian-projected) term.

        """
        nonlocal changed
        if term is None:
            return term
        new_term = term
        if hermitic and not term.isherm:
            new_term = (new_term + new_term.dag()) * 0.5
        new_term = new_term.simplify()
        if term is not new_term:
            changed = True
        return new_term

    # First, rebuild the quadratic form.
    qf_op = QuadraticFormOperator(operator.basis, operator.weights, system)
    new_qf_op = build_quadratic_form_from_operator(
        qf_op.as_sum_of_products(), True, hermitic
    )
    # If the new basis is larger and the hermitian character haven´t changed, keep the older.
    if changed or len(new_qf_op.basis) < len(qf_op.basis):
        qf_op = new_qf_op
        changed = True

    # Now, work on the offset and the linear term

    linear_term = simplify_other_terms(operator.linear_term)
    offset = simplify_other_terms(operator.offset)

    if not changed:
        return operator

    if qf_op.linear_term:
        linear_term = (
            (linear_term + qf_op.linear_term).simplify()
            if linear_term
            else qf_op.linear_term
        )

    if qf_op.offset:
        offset = (
            (offset + qf_op.offset).simplify() if offset is not None else qf_op.offset
        )

    return QuadraticFormOperator(
        qf_op.basis, qf_op.weights, system, linear_term, offset
    )
