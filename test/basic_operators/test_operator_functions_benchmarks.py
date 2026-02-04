"""
Tests and benchmarks for `gram_matrix`.
"""

import os
from test.helper import HAMILTONIAN, SX_A, SX_TOTAL, SZ_TOTAL, TEST_CASES_STATES

import pytest

from qalma.operators import ProductOperator
from qalma.operators.functions import commutator, fidelity


@pytest.mark.skipif(
    not os.environ.get("BENCHMARKS", 0), reason="only run for benchmarks."
)
@pytest.mark.parametrize(
    [
        "case",
        "op1",
        "op2",
        "deep",
    ],
    [
        (
            "[H+Sz_Total, Sx_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SX_TOTAL,
            2,
        ),
        (
            "[H+Sz_Total, Sz_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SZ_TOTAL,
            2,
        ),
        (
            "[H+Sz_Total, SX_A]",
            HAMILTONIAN,
            SX_A,
            2,
        ),
        (
            "[H+Sz_Total, Sx_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SX_TOTAL,
            6,
        ),
        (
            "[H+Sz_Total, Sz_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SZ_TOTAL,
            6,
        ),
        (
            "[H+Sz_Total, SX_A]",
            HAMILTONIAN,
            SX_A,
            6,
        ),
        (
            "[H+Sz_Total, Sx_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SX_TOTAL,
            8,
        ),
        (
            "[H+Sz_Total, Sz_Total]",
            HAMILTONIAN + SZ_TOTAL,
            SZ_TOTAL,
            8,
        ),
        (
            "[H+Sz_Total, SX_A]",
            HAMILTONIAN,
            SX_A,
            8,
        ),
    ],
)
def test_iterated_commutator_benchmark(benchmark, case, op1, op2, deep):
    """
    Benchmark scalar products
    """

    def impl():
        result = op2
        for i in range(deep):
            result = commutator(op1, result)
        return result

    benchmark.pedantic(impl, rounds=3, iterations=1)


@pytest.mark.skipif(
    not os.environ.get("BENCHMARKS", 0), reason="only run for benchmarks."
)
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
def test_fidelity(benchmark, name_rho, name_sigma):

    rho = TEST_CASES_STATES[name_rho]
    sigma = TEST_CASES_STATES[name_sigma]
    if not (isinstance(rho, ProductOperator) and isinstance(sigma, ProductOperator)):
        return

    def impl():
        fidelity1 = fidelity(rho, sigma)
        assert 0 <= fidelity1 <= 1.000000000001
        return fidelity1

    benchmark.pedantic(impl, rounds=3, iterations=1)
