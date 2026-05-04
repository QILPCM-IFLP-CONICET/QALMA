"""Meanfield module.

This module include functions used for implement different flavors of
the meanfield approximation.
"""

from .meanfield import project_meanfield
from .self_consistent_projections import self_consistent_project_meanfield
from .variational import compute_free_energy, compute_t_score, variational_quadratic_mfa

__all__ = [
    "compute_t_score",
    "compute_free_energy",
    "project_meanfield",
    "self_consistent_project_meanfield",
    "variational_quadratic_mfa",
]
