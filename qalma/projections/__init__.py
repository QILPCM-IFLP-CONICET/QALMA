"""
Projection routines.

Routines to implement different kind of linear and non-linear projections on
operator objects.
"""

from .nbody import (
    ProjectingOperatorFunction,
    estimate_log_of_partial_trace,
    n_body_projection,
    one_body_from_qutip_operator,
    project_k_to_sep,
    project_operator_to_n_body,
)

__all__ = [
    "ProjectingOperatorFunction",
    "estimate_log_of_partial_trace",
    "one_body_from_qutip_operator",
    "n_body_projection",
    "project_k_to_sep",
    "project_operator_to_n_body",
]
