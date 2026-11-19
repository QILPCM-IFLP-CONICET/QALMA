"""Tests for the Gaussian states of bosonic modes."""

import numpy as np
import pytest

from qalma.gaussian import (
    BosonicGaussianSystem,
    bogoliubov_reduced,
    build_boson_system_from_correlations,
    build_boson_system_from_quadratic_form,
    quadrature_map,
)


def _thermal_entropy(occupation):
    """Entropy, in bits, of a thermal mode of mean occupation ``occupation``."""
    f = np.asarray(occupation, dtype=float)
    return float(np.sum((1 + f) * np.log2(1 + f) - f * np.log2(f)))


def _quadratic_form(a_mat, b_mat):
    """Matrix ``[[A, B], [B*, A*]]`` of a bosonic quadratic form."""
    a_mat, b_mat = np.asarray(a_mat, dtype=complex), np.asarray(b_mat, dtype=complex)
    return np.block([[a_mat, b_mat], [b_mat.conj(), a_mat.conj()]])


def _two_mode_squeezed_contractions(r):
    """Contractions of the two-mode squeezed vacuum of squeezing ``r``."""
    c, s = np.cosh(r), np.sinh(r)
    # Phi = (b1, b2, b1^dag, b2^dag); <b1 b2> = <b2 b1> = c s
    return np.array(
        [
            [c**2, 0, 0, c * s],
            [0, c**2, c * s, 0],
            [0, c * s, s**2, 0],
            [c * s, 0, 0, s**2],
        ],
        dtype=complex,
    )


def _random_stable_form(n_modes, seed):
    """Random positive definite quadratic form of ``n_modes`` modes."""
    rng = np.random.default_rng(seed)
    a_mat = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(
        size=(n_modes, n_modes)
    )
    a_mat = a_mat @ a_mat.conj().T + 2 * n_modes * np.eye(n_modes)
    b_mat = 0.3 * rng.normal(size=(n_modes, n_modes))
    b_mat = b_mat + b_mat.T
    return _quadratic_form(a_mat, b_mat)


def test_thermal_mode_from_quadratic_form():
    """A single mode ``K = omega b^dag b`` is thermal."""
    omega = 0.7
    system = build_boson_system_from_quadratic_form(_quadratic_form([[omega]], [[0]]))
    occupation = 1.0 / np.expm1(omega)
    assert np.allclose(system.contractions, np.diag([occupation + 1, occupation]))
    assert np.allclose(system.covariance_matrix(), (occupation + 0.5) * np.eye(2))
    assert abs(system.entropy() - _thermal_entropy(occupation)) < 1e-10
    assert system.n_zero_modes == 0 and system.labels == [0]


def test_two_mode_squeezed_vacuum():
    """Entropy and negativity of the two-mode squeezed vacuum."""
    r = 0.4
    system = build_boson_system_from_correlations(
        _two_mode_squeezed_contractions(r), labels=["a", "b"]
    )
    assert system.modes_of(["b"]) == [1]
    assert (
        abs(system.entropy(system.modes_of(["a"])) - _thermal_entropy(np.sinh(r) ** 2))
        < 1e-10
    )
    assert abs(system.entropy()) < 1e-8
    assert abs(system.negativity([0], [1]) - 0.5 * np.expm1(2 * r)) < 1e-10
    assert abs(system.negativity([0]) - system.negativity([0], [1])) < 1e-12
    with pytest.raises(ValueError):
        system.negativity([0], [0, 1])


def test_quadratic_form_and_correlations_agree():
    """The contractions of a thermal state rebuild the same state."""
    h_mat = _random_stable_form(3, seed=7)
    from_form = build_boson_system_from_quadratic_form(h_mat, labels="xyz")
    from_corr = build_boson_system_from_correlations(
        from_form.contractions, labels="xyz"
    )
    assert from_corr.lam is None and from_corr.t_mat is None
    for modes in ([0], [1, 2], [0, 1, 2]):
        assert abs(from_form.entropy(modes) - from_corr.entropy(modes)) < 1e-10
    assert abs(from_form.negativity([0], [1]) - from_corr.negativity([0], [1])) < 1e-12
    # the entropy of the whole state is the thermal entropy of the normal modes
    omega = np.abs(from_form.lam[from_form.lam > 0])
    assert abs(from_form.entropy() - _thermal_entropy(1.0 / np.expm1(omega))) < 1e-8


def test_low_temperature_limit_is_pure():
    """At low temperature the thermal state is the pure Bogoliubov vacuum."""
    system = build_boson_system_from_quadratic_form(
        60.0 * _random_stable_form(2, seed=3)
    )
    assert abs(system.entropy()) < 1e-8
    assert system.entropy([0]) > 1e-6
    assert abs(system.entropy([0]) - system.entropy([1])) < 1e-8


def test_invalid_inputs_are_rejected():
    """Matrices that are not quadratic forms or contractions are rejected."""
    with pytest.raises(ValueError):
        build_boson_system_from_quadratic_form(np.eye(3))
    with pytest.raises(ValueError):  # not of the form [[A, B], [B*, A*]]
        build_boson_system_from_quadratic_form(np.diag([1.0, 2.0]))
    with pytest.raises(ValueError):  # unstable
        build_boson_system_from_quadratic_form(_quadratic_form([[1.0]], [[2.0]]))
    with pytest.raises(ValueError):  # violates the commutation relations
        build_boson_system_from_correlations(np.diag([1.0, 1.0]))
    with pytest.raises(ValueError):  # violates the uncertainty relations
        build_boson_system_from_correlations(np.diag([0.5, -0.5]))
    with pytest.raises(ValueError):
        BosonicGaussianSystem(np.eye(2), labels=["a", "b"])


def test_zero_modes_are_reported():
    """The free particle has a zero mode: the entropies are not defined."""
    system = build_boson_system_from_quadratic_form(
        _free_particle_hamiltonian(1.7, 2.3, 0.0)
    )
    assert system.n_zero_modes == 1
    with pytest.raises(ValueError, match="zero mode"):
        system.entropy()
    with pytest.raises(ValueError, match="zero mode"):
        system.negativity([0])


# ### Bogoliubov transformations with zero modes ###


def _free_particle_hamiltonian(omega, kappa, drive):
    """Oscillator plus a free particle ``Q`` whose charge ``P`` drives it.

    ``H = omega (x1^2 + p1^2)/2 + kappa p2^2/2 + drive x1 p2``, with
    quadratures ordered as ``(x1, x2, p1, p2)``; ``x2`` does not appear.
    """
    m_real = np.zeros((4, 4))
    m_real[0, 0], m_real[2, 2], m_real[3, 3] = omega, omega, kappa
    m_real[0, 3] = m_real[3, 0] = drive
    w_map = quadrature_map(2)
    return w_map @ m_real @ w_map.conj().T


def test_bogoliubov_reduced_removes_zero_mode():
    """The zero mode is removed and the rest of the modes are exact."""
    omega, kappa, drive = 1.7, 2.3, 0.6
    lam, t_mat, n_zero, data = bogoliubov_reduced(
        _free_particle_hamiltonian(omega, kappa, drive)
    )
    assert n_zero == 1
    assert np.allclose(np.sort(lam), [-omega, omega])
    assert t_mat.shape == (4, 2)
    assert abs(data.kappa[0, 0] - kappa) < 1e-10
    # the response of the oscillator to the drive renormalizes the charge
    assert abs(data.kappa_eff[0, 0] - (kappa - drive**2 / omega)) < 1e-10


def test_bogoliubov_reduced_without_zero_modes_is_standard():
    """Without zero modes the result is the usual Bogoliubov transformation."""
    m_real = np.diag([1.2, 0.8, 2.0, 1.5])
    w_map = quadrature_map(2)
    h_mat = w_map @ m_real @ w_map.conj().T
    lam, t_mat, n_zero, data = bogoliubov_reduced(h_mat)
    assert n_zero == 0 and data is None and t_mat.shape == (4, 4)
    assert np.allclose(
        np.sort(np.abs(lam)), np.sort(np.sqrt([1.2 * 2.0, 0.8 * 1.5]).repeat(2))
    )


def test_bogoliubov_reduced_rejects_unstable_hamiltonians():
    """A negative eigenvalue is an instability, not a zero mode."""
    m_real = np.diag([1.0, -0.5, 1.0, 1.0])
    w_map = quadrature_map(2)
    try:
        bogoliubov_reduced(w_map @ m_real @ w_map.conj().T)
    except ValueError as error:
        assert "Unstable" in str(error)
    else:
        raise AssertionError("A ValueError was expected")
