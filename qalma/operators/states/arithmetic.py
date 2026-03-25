"""
Arithmetic operations with states.

Essentially, arithmetic operations with states involves just mixing of operators,
implemented though the class MixtureDensityOperator.

"""

import logging
import pickle
from typing import Iterable, Optional, Set, Tuple, Union, cast

import numpy as np

from qalma.model import SystemDescriptor
from qalma.operators.arithmetic import SumOperator
from qalma.operators.basic import (
    Operator,
)
from qalma.operators.product import (
    ScalarOperator,
)
from qalma.operators.states.basic import (
    DensityOperatorMixin,
)


class MixtureDensityOperator(DensityOperatorMixin, SumOperator):
    """
    A mixture of density operators
    """

    terms: Tuple[Operator]

    def __init__(self, terms: tuple, system: Optional[SystemDescriptor] = None):
        super().__init__(terms, system, True)

    def __neg__(self):
        logging.warning("Negate a DensityOperator leads to a regular operator.")
        new_terms = tuple(((-t) * (t.prefactor) for t in self.terms))
        return SumOperator(new_terms, self.system, isherm=True)

    def acts_over(self) -> frozenset:
        """
        Return a set with the name of the
        sites where the operator nontrivially acts
        """
        sites: Set[str] = set()
        for term in self.terms:
            acts_over = cast(Operator, term).acts_over()
            sites.update(acts_over)
        return frozenset(sites)

    def expect(
        self, obs_objs: Union[Operator, Iterable]
    ) -> Union[np.ndarray, dict, complex]:

        def compute_results(curr_obs, sub_averages, prefactors):
            if isinstance(curr_obs, dict):
                result = {}
                for key in curr_obs:
                    content = curr_obs[key]
                    result[key] = compute_results(
                        content,
                        tuple(contrib[key] for contrib in sub_averages),
                        prefactors,
                    )
                return result
            # Operator, list or tuple, just return the linear combination, because exp_eval
            # is a tuple of Operator or ndarray objects.
            return sum(
                exp_val * p_refactor
                for exp_val, p_refactor in zip(sub_averages, prefactors)
            )

        averages = tuple(
            cast(DensityOperatorMixin, term).expect(obs_objs) for term in self.terms
        )
        prefactors = tuple(term.prefactor for term in self.terms)
        return compute_results(obs_objs, averages, prefactors)

    def partial_trace(self, sites: Union[frozenset, SystemDescriptor]):
        new_terms = tuple(cast(Operator, t).partial_trace(sites) for t in self.terms)
        subsystem = new_terms[0].system
        return MixtureDensityOperator(new_terms, subsystem)

    def simplify(self):
        # DensityOperator's are considered "simplified".
        return self

    def __setstate__(self, state):
        state = pickle.loads(state)
        self.__dict__.update(state)
        self._set_system_(self.system)

    def to_qutip(self, block: Optional[Tuple[str, ...]] = None):
        """Produce a qutip compatible object"""
        if len(self.terms) == 0:
            return ScalarOperator(0, self.system).to_qutip()

        acts_over = self.acts_over()
        if block is None or acts_over is None:
            block = tuple(sorted(self.system.sites))
        else:
            block = block + tuple(
                (site for site in sorted(acts_over) if site not in block)
            )

        # TODO: find a more efficient way to avoid element-wise
        # multiplications
        terms = (
            (
                cast(Operator, term).to_qutip(block),
                term.prefactor,
            )
            for term in self.terms
        )
        return sum(term[0] * term[1] for term in terms)
