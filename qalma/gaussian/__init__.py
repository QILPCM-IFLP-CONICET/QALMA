"""Gaussian states of bosonic modes.

:class:`BosonicGaussianSystem` represents a Gaussian state of bosonic modes
with vanishing first moments. It is built from a quadratic generator
(:func:`build_boson_system_from_quadratic_form`) or from its matrix of
contractions (:func:`build_boson_system_from_correlations`), and gives the
covariance matrices, entropies and negativities of subsets of modes. The
Bogoliubov transformations, including the removal of zero modes, are in
:mod:`qalma.gaussian.bogoliubov`.
"""

from .bogoliubov import (
    ZERO_MODE_RTOL,
    ZeroModeData,
    bogoliubov_reduced,
    bogoliubov_transformation,
    quadrature_map,
    symplectic_spectrum,
)
from .system import (
    BosonicGaussianSystem,
    build_boson_system_from_correlations,
    build_boson_system_from_quadratic_form,
)

__all__ = [
    "BosonicGaussianSystem",
    "ZERO_MODE_RTOL",
    "ZeroModeData",
    "bogoliubov_transformation",
    "bogoliubov_reduced",
    "build_boson_system_from_correlations",
    "build_boson_system_from_quadratic_form",
    "quadrature_map",
    "symplectic_spectrum",
]
