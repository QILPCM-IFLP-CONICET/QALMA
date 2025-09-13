"""
Qutip representation for density operators.

Be careful: just use this class for states of small systems.
"""

import logging
from numbers import Number, Real
from typing import Optional, Tuple, Union, cast

import numpy as np
from qutip import Qobj, tensor as tensor_qutip  # type: ignore[import-untyped]

from qalma.model import SystemDescriptor
from qalma.operators.basic import Operator, ScalarOperator
from qalma.operators.qutip import QutipOperator
from qalma.operators.states.basic import (
    DensityOperatorMixin,
    DensityOperatorProtocol,
)


class QutipDensityOperator(DensityOperatorMixin, QutipOperator):
    """
    Qutip representation of a density operator
    """

    def __init__(
        self,
        qoperator: Qobj,
        system: Optional[SystemDescriptor] = None,
        names=None,
        prefactor=1,
        normalized=False,
    ):
        self._normalized = normalized
        super().__init__(qoperator, system, names, prefactor)
        self.normalize()

    def __add__(self, operand) -> Operator:
        if isinstance(operand, (int, float, np.float64)):
            if operand >= 0:
                return QutipDensityOperator(
                    self.operator * self.prefactor + operand,
                    self.system,
                )
            logging.warning(
                f"Adding {
                    operand} to a DensityOperator produces a generic operator."
            )
            return QutipOperator(
                self.operator * self.prefactor + operand,
                self.system,
            )
        if isinstance(operand, (complex, np.complex128)):
            logging.warning(
                f"Adding {
                    operand} to a DensityOperator produces a generic operator."
            )
            return QutipOperator(
                self.operator * self.prefactor + operand,
                self.system,
            )

        # TODO: check me again
        assert operand.system is self.system
        block = tuple(sorted(self.system.sites))
        names = {name: pos for pos, name in enumerate(block)}

        if hasattr(operand, "expect"):
            p1 = self.prefactor
            p2 = operand.prefactor
            prefactor = p1 + p2

            result_qutip = self.to_qutip(block) * (p1 / prefactor) + operand.to_qutip(
                block
            ) * (p2 / prefactor)
            return QutipDensityOperator(result_qutip, self.system, names, prefactor)
        result_qutip = self.to_qutip(block) * self.prefactor + operand.to_qutip(block)

        return QutipOperator(result_qutip, self.system, names)

    def __mul__(self, operand) -> Operator:
        try:
            return self.join_states(operand)
        except ValueError:
            pass
        self.normalize()
        op_b = QutipOperator(self.operator, names=self.site_names, system=self.system)
        return op_b * operand

    def __neg__(self):
        logging.warning("Negate a DensityOperator leads to a regular operator.")
        self.normalize()
        return QutipOperator(self.operator, self.system, self.site_names, -1)

    def __radd__(self, operand) -> Operator:
        if isinstance(operand, (int, float, np.float64)):
            if operand >= 0:
                return QutipDensityOperator(
                    self.operator * self.prefactor + operand,
                    self.system,
                )
            return QutipOperator(
                self.operator * self.prefactor + operand,
                self.system,
            )
        if isinstance(operand, (complex, np.complex128)):
            logging.warning(
                f"Adding {
                    operand} to a DensityOperator produces a generic operator."
            )
            return QutipOperator(
                self.operator * self.prefactor + operand,
                self.system,
            )

        # TODO: check me again
        op_qo = operand.to_qutip()
        if isinstance(operand, DensityOperatorMixin):
            op_qo = op_qo * self.prefactor
            return QutipDensityOperator(op_qo, self.system or op_qo.system)
        return QutipOperator(op_qo, self.system or op_qo.system)

    def __rmul__(self, operand):
        try:
            return self.join_states(operand)
        except ValueError:
            pass
        self.normalize()
        op_b = QutipOperator(self.operator, names=self.site_names, system=self.system)
        return operand * op_b

    def join_states(self, other: DensityOperatorProtocol | Number):
        """
        Combine the states of two disjoint systems to produce the state
        of the union of both systems.
        """
        if isinstance(other, Real):
            if other < 0:
                raise ValueError
            return QutipDensityOperator(
                self.operator,
                system=self.system,
                names=self.site_names,
                prefactor=self.prefactor * other,
            )
        if isinstance(other, Number):
            raise ValueError("operand is not a positive number")
        rho: DensityOperatorProtocol = other
        if not hasattr(rho, "expect"):
            raise ValueError
        if not rho.prefactor:
            return QutipDensityOperator(
                self.operator, system=self.system, names=self.site_names, prefactor=0
            )
        system_a = self.system
        system_b = rho.system
        if set(system_a.sites).intersection(system_b.sites):
            raise ValueError("Systems have overlap")

        system = system_a.union(system_b)
        acts_over_b = rho.acts_over()
        if len(acts_over_b) == 0:
            return QutipDensityOperator(
                self.operator,
                system=self.system,
                names=self.site_names,
                prefactor=cast(Real, self.prefactor) * cast(Real, rho.prefactor),
            )
        acts_over_a = self.acts_over()
        if len(acts_over_a) == 0:
            return rho * self.prefactor

        block_a = tuple(acts_over_a)
        block_b = tuple(acts_over_b)
        names = {site: pos for pos, site in enumerate(block_a + block_b)}
        qutip_block = tensor_qutip(self.to_qutip(block_a), rho.to_qutip(block_b))
        prefactor = self.prefactor * rho.prefactor
        return QutipDensityOperator(
            qutip_block, names=names, system=system, prefactor=prefactor
        )

    def logm(self):
        self.normalize()
        operator = self.operator
        evals, evecs = operator.eigenstates()
        evals[abs(evals) < 1.0e-30] = 1.0e-30
        log_op = sum(
            np.log(e_val) * e_vec * e_vec.dag() for e_val, e_vec in zip(evals, evecs)
        )
        return QutipOperator(log_op, self.system, self.site_names)

    def normalize(self):
        """Normalize the operator"""
        if self._normalized:
            return self
        qoperator = self.operator
        tr_op = qoperator.tr()
        if tr_op != 1:
            qoperator = qoperator / tr_op
        self.operator = qoperator
        self._normalized = True
        return self

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        self.normalize()
        self_pt = super().partial_trace(sites)
        if isinstance(self_pt, ScalarOperator):
            return self_pt

        return QutipDensityOperator(
            self_pt.operator,
            names=self_pt.site_names,
            system=self_pt.system,
            prefactor=self.prefactor,
        )

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        self.normalize()
        # set the prefactor temporarily to 1, because it should
        # not be taken into account in the conversion of a state.
        prefactor = self.prefactor
        self.prefactor = 1
        qutip_op = super().to_qutip(block)
        # setting back the value
        self.prefactor = prefactor
        return qutip_op
