r"""Reduced states of one or two sites from the bosonized RPA.

The thermal state of the whole system is approximated by the Gaussian state
of the bosonized RPA (:func:`~qalma.meanfield.rpa.rpa_structure.build_rpa_boson_system`).
Local operators are mapped to the bosons including the quadratic (population)
terms: in the eigenbasis :math:`|k\rangle` of the local mean-field state, with
populations :math:`p_i \ge p_j` for :math:`i<j`,

.. math::

    |i\rangle\langle j| \to \sqrt{p_i-p_j}\, a_{ij}, \qquad
    |j\rangle\langle i| \to \sqrt{p_i-p_j}\, a^\dagger_{ij}, \qquad
    |j\rangle\langle j| - |i\rangle\langle i| \to
        2(a^\dagger_{ij} a_{ij} - n^0_{ij}) + \dots

that is, creating a boson in the pair :math:`(i, j)` moves population from
:math:`i` to :math:`j`, relative to its occupation
:math:`n^0_{ij} = p_j/(p_i-p_j)` in the local state. For an operator
:math:`O` (:math:`O_{ij}` in that basis),

.. math::

    O \to \langle O\rangle_{\rm MF}
        + \sum_{i<j} \sqrt{p_i-p_j}\,(O_{ij} a_{ij} + O_{ji} a^\dagger_{ij})
        + \sum_{i<j} (O_{jj} - O_{ii})(a^\dagger_{ij} a_{ij} - n^0_{ij}) .

Expectation values of products of operators of two sites follow from Wick's
theorem. This is the bosonization of J. M. Matera's thesis (chapter 6,
eq. 6.9): the one-body averages and the correlations of the linear parts
coincide with those obtained deriving the RPA partition function with
respect to the couplings (the mean field plus RPA of Canosa, Matera and
Rossignoli), and the Gaussian state adds terms of higher order in
:math:`1/N` (the product of the population corrections and the Wick
contraction of the quadratic parts).

Since the bosonization of the whole system is computed once and stored in
the :class:`~qalma.meanfield.rpa.rpa_structure.RPAStructure`, the state of
each pair costs only a few contractions, much less than a dynamical or
hybrid reduction. The resulting matrices are Hermitian and normalized, but
not necessarily positive: the truncation can leave negative eigenvalues of
the order of the neglected terms.
"""

from typing import Dict, Tuple

import numpy as np
from qutip import Qobj

from .rpa_structure import RPAStructure
from .utils import _boson_pairs, _is_dynamic, _ordered_spectrum

__all__ = ["rpa_bosonized_state"]


def _site_map(
    structure: RPAStructure, site: str, modes: list, contractions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the matrix units ``|b><a|`` of ``site`` to the bosons.

    Returns ``(mean, linear, quadratic)`` with shapes ``(d, d)``,
    ``(d, d, 2 n)`` and ``(d, d, m)``, indexed by ``[a, b]``: the corrected
    mean value, the coefficients of ``Phi = (a, a^dag)`` and the coefficients
    of the centred occupations of the ``m`` modes of the site.
    """
    n_all = contractions.shape[0] // 2
    p, vecs = _ordered_spectrum(structure.local_spectrum(site))
    dim = len(p)
    pairs = [
        (i, j)
        for i, j, gap in _boson_pairs(p)
        if gap >= structure.static_gap and _is_dynamic(gap, structure.static_gap)
    ]
    if len(pairs) != len(modes):
        raise RuntimeError(
            f"Site {site}: {len(pairs)} bosonic pairs but {len(modes)} modes."
        )
    # units[a, b] = <k|b><a|k'> in the eigenbasis: vecs^dag |b><a| vecs
    units = np.einsum("bk,al->abkl", vecs.conj(), vecs)
    mean = np.einsum("abkk,k->ab", units, p).astype(complex)
    linear = np.zeros((dim, dim, 2 * n_all), dtype=complex)
    quadratic = np.zeros((dim, dim, len(modes)), dtype=complex)
    for col, (mode, (i, j)) in enumerate(zip(modes, pairs)):
        weight = np.sqrt(p[i] - p[j])
        linear[:, :, mode] = weight * units[:, :, i, j]
        linear[:, :, n_all + mode] = weight * units[:, :, j, i]
        quadratic[:, :, col] = units[:, :, j, j] - units[:, :, i, i]
        occupation = np.real(contractions[n_all + mode, n_all + mode])
        reference = p[j] / (p[i] - p[j])
        mean += quadratic[:, :, col] * (occupation - reference)
    return mean, linear, quadratic


def rpa_bosonized_state(structure: RPAStructure, sites: frozenset) -> Qobj:
    r"""Reduced state of one or two sites from the bosonized RPA.

    See the module documentation for the map of the local operators to the
    bosons. Only the dynamical (non-static) bosons are included.

    Parameters
    ----------
    structure : RPAStructure
        Subsystem-independent data of the RPA.
    sites : frozenset
        One or two sites.

    Returns
    -------
    qutip.Qobj
        The reduced density matrix, with the sites in sorted order. It is
        Hermitian and normalized, but not necessarily positive.

    Raises
    ------
    ValueError
        If the bosonized RPA has zero modes.
    NotImplementedError
        If more than two sites are requested.
    """
    block = tuple(sorted(sites))
    if not 1 <= len(block) <= 2:
        raise NotImplementedError(
            "Bosonized reduced states are implemented for one or two sites."
        )
    bosons = structure.boson_system()
    if bosons.n_zero_modes:
        raise ValueError(
            f"The bosonic RPA of the whole system has {bosons.n_zero_modes} zero "
            "mode(s): the bosonized states are not defined. Use the dynamical "
            "or hybrid reductions with symmetry_projections instead."
        )
    contractions = bosons.contractions
    n_all = bosons.n_modes
    maps: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    modes = {}
    for site in block:
        modes[site] = bosons.modes_of([site])
        maps[site] = _site_map(structure, site, modes[site], contractions)

    dims = [maps[site][0].shape[0] for site in block]
    if len(block) == 1:
        rho = maps[block[0]][0]
    else:
        (mean_i, lin_i, quad_i), (mean_j, lin_j, quad_j) = (
            maps[block[0]],
            maps[block[1]],
        )
        # <Phi_alpha Phi_beta> = (G X)_{alpha beta}
        swap = np.concatenate([np.arange(n_all, 2 * n_all), np.arange(n_all)])
        g_x = contractions[:, swap]
        # Wick: <(a_k^dag a_k)(a_l^dag a_l)>_c = |<a_k^dag a_l>|^2 + |<a_k a_l>|^2
        idx_i, idx_j = np.array(modes[block[0]]), np.array(modes[block[1]])
        hop = contractions[np.ix_(n_all + idx_i, n_all + idx_j)]
        pair = g_x[np.ix_(idx_i, idx_j)]
        wick = np.abs(hop) ** 2 + np.abs(pair) ** 2
        rho = (
            np.einsum("ab,cd->acbd", mean_i, mean_j)
            + np.einsum("abx,xy,cdy->acbd", lin_i, g_x, lin_j)
            + np.einsum("abk,kl,cdl->acbd", quad_i, wick, quad_j)
        )
        rho = rho.reshape(dims[0] * dims[1], dims[0] * dims[1])
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)
    return Qobj(rho, dims=[dims, dims])
