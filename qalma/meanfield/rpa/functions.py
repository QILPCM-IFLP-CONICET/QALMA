"""Entropies and negativities of the bosonized RPA."""

from typing import Optional

from .rpa_structure import RPAStructure

__all__ = ["rpa_entanglement_entropy", "rpa_negativity"]


def _rpa_boson_system(structure: RPAStructure):
    """Return the bosonized RPA, rejecting it if it has zero modes."""
    bosons = structure.boson_system()
    if bosons.n_zero_modes:
        raise ValueError(
            f"The bosonic RPA of the whole system has {bosons.n_zero_modes} zero "
            "mode(s) (a broken continuous symmetry): the covariance matrix of a "
            "subsystem is not defined without restoring the symmetry. Use the "
            "hybrid reduction instead."
        )
    return bosons


def rpa_entanglement_entropy(structure: RPAStructure, sites: frozenset) -> float:
    r"""Bosonic RPA entropy of a subsystem, in bits.

    It is the von Neumann entropy of the reduced Gaussian state of the
    bosons of ``sites`` (see
    :meth:`~qalma.gaussian.BosonicGaussianSystem.entropy`). For a pure state
    (zero temperature) it is the entanglement entropy of the bipartition
    (``sites``, complement); otherwise it also includes the thermal entropy.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset
        Sites of the subsystem.

    Returns
    -------
    float
        The entropy in bits.
    """
    bosons = _rpa_boson_system(structure)
    return bosons.entropy(bosons.modes_of(sites))


def rpa_negativity(
    structure: RPAStructure, sites_b: frozenset, sites_c: Optional[frozenset] = None
) -> float:
    r"""Bosonic RPA negativity between two subsystems.

    See :meth:`~qalma.gaussian.BosonicGaussianSystem.negativity`.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites_b, sites_c : frozenset
        Sites of the two subsystems. If ``sites_c`` is ``None``, the
        complement of ``sites_b`` is used.

    Returns
    -------
    float
        The negativity.
    """
    bosons = _rpa_boson_system(structure)
    if sites_c is None:
        sites_c = frozenset(bosons.labels) - frozenset(sites_b)
    return bosons.negativity(bosons.modes_of(sites_b), bosons.modes_of(sites_c))
