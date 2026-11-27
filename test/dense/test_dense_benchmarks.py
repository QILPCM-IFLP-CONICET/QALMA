"""Benchmarks of the code paths that rely on dense linear algebra.

The benchmarks of ``make benchmark`` act mostly on symbolic operators over
small local spaces, where BLAS/LAPACK never use more than one thread. The
cases here exercise matrices and tensors large enough for the threading of
the underlying library to matter, so they are the ones to use to look at
the effect of ``OPENBLAS_NUM_THREADS`` / ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS``.

Run with::

    make benchmark-dense

Compare the effect of the number of threads with::

    OPENBLAS_NUM_THREADS=1 make benchmark-dense BENCHMARK_FILE=bench_st
    make benchmark-dense BENCHMARK_FILE=bench_mt
    python tools/compare_benchmarks.py <st json> <mt json>

Every benchmark stores in ``extra_info["blas"]`` the thread pools that were
actually active (needs ``threadpoolctl``; otherwise the environment variables
are stored), so the saved files tell which configuration they correspond to.

The sizes can be changed through environment variables with comma separated
lists: ``DENSE_RAW_SIZES`` (matrix dimensions), ``DENSE_SITES`` (sites of the
exact Gibbs states), ``DENSE_RPA_SITES``, ``DENSE_DMRG`` (``sites:chi`` pairs).
"""

import os

import numpy as np
import pytest
import scipy.linalg

from qalma import build_system
from qalma.meanfield import variational_quadratic_mfa
from qalma.meanfield.rpa import RPAGibbsDensityOperator
from qalma.meanfield.symmetry_breaking import symmetry_breaking_mfa
from qalma.mps import MPSGSOperator
from qalma.operators.states import GibbsDensityOperator

BENCHMARKS = bool(os.environ.get("BENCHMARKS", 0))

pytestmark = [
    pytest.mark.skipif(not BENCHMARKS, reason="run only in benchmarks"),
    pytest.mark.benchmark(group="bench_dense"),
]


def _int_list(name, default):
    return [int(x) for x in os.environ.get(name, default).split(",") if x.strip()]


def _dmrg_cases():
    raw = os.environ.get("DENSE_DMRG", "12:32,16:32,16:64")
    return [tuple(int(v) for v in item.split(":")) for item in raw.split(",")]


RAW_SIZES = _int_list("DENSE_RAW_SIZES", "128,256,512,1024")
GIBBS_SITES = _int_list("DENSE_SITES", "8,10")
RPA_SITES = _int_list("DENSE_RPA_SITES", "8,12,16")
DMRG_CASES = _dmrg_cases()

ROUNDS = 5
ENV_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _annotate(benchmark):
    """Record in the benchmark which thread pools were active."""
    info = {name: os.environ.get(name) for name in ENV_VARS}
    try:
        from threadpoolctl import threadpool_info

        info["pools"] = [
            {
                "api": pool.get("user_api"),
                "lib": pool.get("internal_api"),
                "threads": pool.get("num_threads"),
            }
            for pool in threadpool_info()
        ]
    except ImportError:  # pragma: no cover
        info["pools"] = "threadpoolctl not installed"
    benchmark.extra_info["blas"] = info


def _run(benchmark, setup, func, rounds=ROUNDS):
    """Time ``func(*args)`` with ``args`` rebuilt by ``setup`` on each round.

    A fresh state per round keeps cached results (normalization, RPA
    structure, DMRG solution) from hiding the cost that is being measured.
    """
    _annotate(benchmark)
    return benchmark.pedantic(
        func,
        setup=lambda: (setup(), {}),
        rounds=rounds,
        iterations=1,
        warmup_rounds=1,
    )


def _spin_chain(length, **params):
    defaults = {"a": 1, "J": 1, "Gamma": 0.5, "Jz0": 1.0, "Jxy0": 1.0}
    defaults.update(params)
    return build_system("chain lattice", "spin", L=length, **defaults)


# ---------------------------------------------------------------------------
# Raw dense linear algebra. Positive control: these must react to the number
# of BLAS threads if the threading has any effect on the machine.
# ---------------------------------------------------------------------------


def _random_matrices(size, hermitian=False):
    rng = np.random.default_rng(size)
    mat = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    other = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    if hermitian:
        mat = 0.5 * (mat + mat.conj().T)
    return mat, other


@pytest.mark.parametrize("size", RAW_SIZES)
def test_raw_matmul(benchmark, size):
    _run(benchmark, lambda: _random_matrices(size), lambda a, b: a @ b)


@pytest.mark.parametrize("size", RAW_SIZES)
def test_raw_eigh(benchmark, size):
    _run(
        benchmark,
        lambda: (_random_matrices(size, hermitian=True)[0],),
        np.linalg.eigh,
    )


@pytest.mark.parametrize("size", RAW_SIZES)
def test_raw_expm(benchmark, size):
    _run(
        benchmark,
        lambda: (0.01 * _random_matrices(size, hermitian=True)[0],),
        scipy.linalg.expm,
    )


# ---------------------------------------------------------------------------
# Exact (dense) Gibbs states.
# ---------------------------------------------------------------------------


def _gibbs_state(length):
    system = _spin_chain(length)
    hamiltonian = system.global_operator("Hamiltonian")
    return system, hamiltonian, GibbsDensityOperator(2.0 * hamiltonian, system)


@pytest.mark.parametrize("length", GIBBS_SITES)
def test_exact_diagonalization(benchmark, length):
    system, hamiltonian, _ = _gibbs_state(length)
    block = tuple(sorted(system.sites))

    def setup():
        return (hamiltonian.to_qutip(block).full(),)

    _run(benchmark, setup, np.linalg.eigh)


@pytest.mark.parametrize("length", GIBBS_SITES)
def test_gibbs_to_qutip(benchmark, length):
    system, hamiltonian, _ = _gibbs_state(length)
    _run(
        benchmark,
        lambda: (GibbsDensityOperator(2.0 * hamiltonian, system),),
        lambda state: state.to_qutip(),
    )


@pytest.mark.parametrize("length", GIBBS_SITES)
def test_gibbs_expect(benchmark, length):
    system, hamiltonian, _ = _gibbs_state(length)
    _run(
        benchmark,
        lambda: (GibbsDensityOperator(2.0 * hamiltonian, system),),
        lambda state: state.expect(hamiltonian),
    )


@pytest.mark.parametrize("length", GIBBS_SITES)
def test_gibbs_exact_pair_reduction(benchmark, length):
    system, hamiltonian, _ = _gibbs_state(length)
    pair = frozenset(sorted(system.sites)[:2])
    _run(
        benchmark,
        lambda: (GibbsDensityOperator(2.0 * hamiltonian, system),),
        lambda state: state.to_qutip_operator().partial_trace(pair),
    )


# ---------------------------------------------------------------------------
# RPA.
# ---------------------------------------------------------------------------

RPA_VARIANTS = {
    "static": {},
    "dynamic": {"dynamic": True},
    "global": {"environment": "global"},
}


@pytest.mark.parametrize("variant", list(RPA_VARIANTS))
@pytest.mark.parametrize("length", RPA_SITES)
def test_rpa_pair_reduction(benchmark, length, variant):
    system = _spin_chain(length, Jxy0=0.5, Gamma=1.0)
    generator = 1.0 * system.global_operator("Hamiltonian")
    pair = frozenset(sorted(system.sites)[:2])
    options = RPA_VARIANTS[variant]

    def setup():
        return (RPAGibbsDensityOperator(generator, system, **options),)

    _run(benchmark, setup, lambda state: state.partial_trace(pair))


@pytest.mark.parametrize("length", [n for n in RPA_SITES if n <= 8] or [8])
def test_rpa_hybrid_three_sites(benchmark, length):
    system = _spin_chain(length, Jxy0=0.5, Gamma=1.0)
    generator = 1.0 * system.global_operator("Hamiltonian")
    block = frozenset(sorted(system.sites)[:3])

    def setup():
        return (RPAGibbsDensityOperator(generator, system, hybrid=True),)

    _run(benchmark, setup, lambda state: state.partial_trace(block))


# ---------------------------------------------------------------------------
# Mean field with symmetry breaking (dense optimization over the fields).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", [8])
def test_symmetry_breaking_mfa(benchmark, length):
    system = _spin_chain(length, Gamma=0.0)
    generator = 3.0 * system.global_operator("Hamiltonian")

    def run():
        variational_quadratic_mfa(generator, numfields=0, max_self_consistent_steps=100)
        return symmetry_breaking_mfa(
            generator,
            system,
            numfields=4,
            epsilon=1e-3,
            n_attempts=2,
            seed=0,
            max_self_consistent_steps=50,
        )

    _run(benchmark, lambda: (), run, rounds=3)


# ---------------------------------------------------------------------------
# DMRG.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length,chi", DMRG_CASES)
def test_dmrg_ground_state(benchmark, length, chi):
    system = build_system("chain lattice", "spin", L=length)
    hamiltonian = system.global_operator("Hamiltonian")

    def setup():
        return (MPSGSOperator(hamiltonian, chi_max=chi),)

    _run(benchmark, setup, lambda state: state.solve())
