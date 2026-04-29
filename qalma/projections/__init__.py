"""Routines to implement different kind of linear and non-linear projections
on operator objects.

"""

from .nbody import (
    ProjectingOperatorFunction,
    n_body_projection,
    one_body_from_qutip_operator,
    project_operator_to_n_body,
)

__all__ = [
    "ProjectingOperatorFunction",
    "one_body_from_qutip_operator",
    "n_body_projection",
    "project_operator_to_n_body",
]
