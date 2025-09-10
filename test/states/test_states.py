"""
Basic unit test for states.
"""

from test.helper import (
    OBSERVABLE_CASES,
    SUBSYSTEMS,
    SZ_TOTAL,
    TEST_CASES_STATES,
    alert,
    check_equality,
    expect_from_qutip,
)

import pytest

from qalma.operators import OneBodyOperator
from qalma.settings import QALMA_TOLERANCE

# from qalma.settings import VERBOSITY_LEVEL

QT_TEST_CASES = {
    name: operator.to_qutip() for name, operator in TEST_CASES_STATES.items()
}


@pytest.mark.parametrize(
    ["name_rho", "name_sigma"],
    [
        (
            name_rho,
            name_sigma,
        )
        for name_rho in TEST_CASES_STATES
        for name_sigma in TEST_CASES_STATES
    ],
)
def test_binary_mixtures(name_rho, name_sigma):
    rho_coeff = 0.99
    sigma_coeff = 0.01
    rho = TEST_CASES_STATES[name_rho]
    sigma = TEST_CASES_STATES[name_sigma]
    print(
        f"{rho_coeff}*",
        name_rho,
        f"[{type(rho)}] + {sigma_coeff} * ",
        name_sigma,
        f"[{type(sigma)}]",
    )
    mixture = rho_coeff * rho + sigma_coeff * sigma
    qutip_mixture = (
        rho_coeff * QT_TEST_CASES[name_rho] + sigma_coeff * QT_TEST_CASES[name_sigma]
    )
    assert check_equality(rho.tr(), 1)
    assert check_equality(sigma.tr(), 1)
    assert check_equality(mixture.tr(), 1)
    assert check_equality(qutip_mixture.tr(), 1)
    print("mixture:\n", mixture)
    print("qutip mixture:\n", qutip_mixture)
    check_equality(mixture.to_qutip(), qutip_mixture)


@pytest.mark.parametrize(
    ["name_rho", "name_sigma", "name_theta"],
    [
        (
            name_rho,
            name_sigma,
            name_theta,
        )
        for name_rho in TEST_CASES_STATES
        for name_sigma in TEST_CASES_STATES
        for name_theta in TEST_CASES_STATES
    ],
)
def no_test_ternary_mixtures(name_rho, name_sigma, name_theta):
    rho_coeff = 0.5
    sigma_coeff = 0.3
    theta_coeff = 0.2
    rho = TEST_CASES_STATES[name_rho]
    sigma = TEST_CASES_STATES[name_sigma]
    theta = TEST_CASES_STATES[name_theta]

    print(
        f"{rho_coeff}*",
        name_rho,
        f"[{type(rho)}] + {sigma_coeff} * ",
        name_sigma,
        f"[{type(sigma)}]",
        name_theta,
        f"[{type(theta)}]",
    )
    mixture = rho_coeff * rho + sigma_coeff * sigma + theta_coeff * theta
    qutip_mixture = (
        rho_coeff * QT_TEST_CASES[name_rho]
        + sigma_coeff * QT_TEST_CASES[name_sigma]
        + theta_coeff * QT_TEST_CASES[name_theta]
    )
    assert check_equality(rho.tr(), 1, QALMA_TOLERANCE)
    assert check_equality(sigma.tr(), 1, QALMA_TOLERANCE)
    assert check_equality(theta.tr(), 1, QALMA_TOLERANCE)
    assert check_equality(mixture.tr(), 1, QALMA_TOLERANCE)
    assert check_equality(qutip_mixture.tr(), 1, QALMA_TOLERANCE)
    print("mixture:\n", mixture)
    print("qutip mixture:\n", qutip_mixture)
    check_equality(mixture.to_qutip(), qutip_mixture, QALMA_TOLERANCE)


@pytest.mark.parametrize(["name", "rho"], list(TEST_CASES_STATES.items()))
def test_states(name, rho):
    """Tests for state objects"""
    # enumerate the name of each subsystem
    print(80 * "=", "\n")
    print("test states")
    print(80 * "=", "\n")
    assert isinstance(SZ_TOTAL, OneBodyOperator)

    print("\n     ", 120 * "@", "\n testing", name, f"({type(rho)})", "\n", 100 * "@")
    assert abs(rho.tr() - 1) < 1.0e-10, "la traza de rho no es 1"
    assert abs(1 - QT_TEST_CASES[name].tr()) < 1.0e-10, "la traza de rho.qutip no es 1"

    for subsystem in SUBSYSTEMS:
        print("   subsystem", subsystem)
        local_rho = rho.partial_trace(frozenset(subsystem))
        print(" type", local_rho)
        assert check_equality(local_rho.tr(), 1), "la traza del operador local no es 1"

    # Check Expectation Values
    print(" ??????????????? testing expectation values")
    print(rho.expect)
    expectation_values = rho.expect(OBSERVABLE_CASES)
    qt_expectation_values = expect_from_qutip(QT_TEST_CASES[name], OBSERVABLE_CASES)

    assert isinstance(expectation_values, dict)
    assert isinstance(qt_expectation_values, dict)
    for obs in expectation_values:
        alert(0, "\n     ", 80 * "*", "\n     ", name, " over ", obs)
        alert(0, "Native", expectation_values)
        alert(0, "QTip", qt_expectation_values)
        try:
            assert check_equality(
                expectation_values[obs], qt_expectation_values[obs], QALMA_TOLERANCE
            )
        except AssertionError:
            assert (
                False
            ), f"the expectation value for the observable{obs} relative to {name} do not match ((result)={expectation_values[obs]} !=  {qt_expectation_values[obs]} (qutip))."


# test_load()
# test_all()
# test_eval_expr()
