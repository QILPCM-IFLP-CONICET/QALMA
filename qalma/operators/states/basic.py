"""Density operator classes."""

import logging
import pickle
from typing import Dict, Iterable, Optional, Protocol, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from qalma.model import SystemDescriptor
from qalma.operators.arithmetic import SumOperator
from qalma.operators.basic import (
    Operator,
)
from qalma.operators.quadratic import QuadraticFormOperator


class DensityOperatorMixin:
    """Base class for Density Operators.

    DensityOperatorMixin is a Mixing class that
    contributes operator subclasses with the method
    ``expect``.

    Notice that the `prefactor` attribute of these classes
    is only taken into account when density operators are combined
    into  a mixture by adding them, and when we do operations with
    positive numbers.

    In other operations, like multiplication with other operators,
    density operators are handled as positive operators of trace 1.

    So, for example,

    .. code-block:: python

        rho = .3 * ProductDensityOperator({"site1": qutip.qeye(2) + qutip.sigmax(),
                                       "site2": qutip.qeye(2)})


    acts under operations with other operators like

    .. code-block:: python

        rho = ProductOperator({"site1": .5*(qutip.qeye(2) + qutip.sigmax()),
                           "site2": .5*qutip.qeye(2)})


    If now we introduce a ``sigma`` operator

    .. code-block:: python

        sigma = .7 ProductDensityOperator({"site1": qutip.qeye(2),
                                       "site2": qutip.qeye(2) +
                                       qutip.sigmax()})


    the mixture

    .. code-block:: python

        mix = rho + sigma


    and another operator ``A``

    .. code-block:: python

        A = ProductOperator({"site1": qutip.sigmax(), "site2": qutip.sigmax()})


    we obtain the equality

    .. code-block:: python

        (mix * A).tr() == .3 * (A * rho).tr() + .7 * (A* sigma)


    Notice that algebraic operations does not check if the prefactors
    of all the terms adds to 1.
    To be sure about the normalization, use the method ``expect``:

    .. code-block:: python

        mix.expect(A) == (mix * A).tr()/sum([t.prefactor for t in A.terms])


    """

    prefactor: complex
    system: SystemDescriptor

    def __neg__(self):
        """Multiply the operator by -1."""
        logging.warning("Negate a DensityOperator leads to a regular operator.")
        return -self.to_qutip_operator()

    def __getstate__(self):
        """Get the state for persistency."""
        if hasattr(self, "_serialized"):
            return self._serialized
        state = self.__dict__.copy()
        self._serialized = pickle.dumps(state)
        return self._serialized

    def __setstate__(self, state):
        """Set the state from persistency."""
        state_dict = pickle.loads(state)
        self.__dict__.update(state_dict)

    def dag(self) -> Operator:
        """Build the adjoint operator."""
        return cast(Operator, self)

    def eigenstates(self) -> list:
        """Compute the eigendecomposition."""
        if isinstance(self, Operator):
            return super().eigenstates()  # type: ignore[misc]
        raise NotImplementedError

    def expect(
        self,
        obs_objs: Union[Operator, Iterable],
        _local_states: Optional[Dict[frozenset, "DensityOperatorProtocol"]] = None,
    ) -> Union[NDArray, dict, complex]:
        """Compute the expectation value of an observable."""
        # TODO: explode that expectation values of operators just requires the
        # state where the operators acts.
        from qalma.operators.states.utils import (
            collect_local_states,
            reduced_state_by_block,
        )

        _local_states = collect_local_states(
            obs_objs, self, _local_states=_local_states
        )
        # local_states = {None: self}

        def do_evaluate_expect(obs):
            """Inner function to evaluate expectation values.

            This method keeps track of the states of the subsystems
            required in the evaluation, which in typical cases is the
            most expensive part of the evaluation.
            """
            nonlocal _local_states

            if isinstance(obs, dict):
                return {
                    name: do_evaluate_expect(operator) for name, operator in obs.items()
                }

            if isinstance(obs, (tuple, list)):
                return np.array([do_evaluate_expect(operator) for operator in obs])

            if isinstance(obs, QuadraticFormOperator):
                obs = obs.as_sum_of_products()

            obs = obs.simplify()
            if isinstance(obs, SumOperator):
                return sum(do_evaluate_expect(term) for term in obs.terms)

            acts_over = obs.acts_over()
            if acts_over is not None and len(acts_over) == 0:
                if hasattr(obs, "prefactor"):
                    return obs.prefactor

            # if the argument matches with the argument of expect,
            # it means that we already try with the implementation of the
            # subclasses. Then, let's rely in the generic implementation:
            # convert everything to qutip and evaluate the trace:
            local_state_acts_over = reduced_state_by_block(obs, _local_states)
            if obs_objs is obs:
                block = tuple(sorted(acts_over))
                return (
                    local_state_acts_over.to_qutip(block) * obs.to_qutip(block)
                ).tr()

            # If obs comes from an internal call, then try to use
            # the specific method of the subclass.
            return local_state_acts_over.expect(obs)

        return do_evaluate_expect(obs_objs)

    @property
    def isherm(self):
        """Evaluate the isherm property."""
        return True

    def simplify(self):
        """Build an operator in a simplified representation."""
        # DensityOperator's are considered "simplified".
        return self

    def to_qutip_operator(self):
        """Convert to the QutipOperator representation."""
        from qalma.operators.states import QutipDensityOperator

        prefactor = getattr(self, "prefactor", 1.0)
        block = tuple(sorted(self.system.sites))
        if not block:
            return QutipDensityOperator(1, system=self.system, prefactor=prefactor)

        names = {name: pos for pos, name in enumerate(block)}
        rho_qutip = self.to_qutip(block)
        return QutipDensityOperator(
            rho_qutip, names=names, system=self.system, prefactor=prefactor
        )

    def tr(self):
        """Compute the trace of the operator."""
        return 1


class DensityOperatorProtocol(Protocol):
    """Minimal interface of DensityOperators."""

    prefactor: complex
    system: SystemDescriptor

    def acts_over(self) -> frozenset:
        """Return a list of sites over which this operator acts."""

    def __add__(self, other):
        """Compute the sum with another number or operator."""

    def __radd__(self, other):
        """Compute the sum with another number or operator."""

    def __mul__(self, other):
        """Compute the product."""

    def __rmul__(self, other):
        """Compute the product from left."""

    def expect(
        self,
        obs: Union[Operator, Iterable],
        _local_states: Optional[Dict[frozenset, "DensityOperatorProtocol"]] = None,
    ) -> Union[np.ndarray, dict, complex]:
        """Compute expectation values."""

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        """Compute the partial trace."""

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Return a Qobj representation acting over block."""

    def to_qutip_operator(self):
        """Convert to QutipOperator representation."""
