"""Tests for the parallel routines (``QALMA_USE_PARALLEL``).

The settings are read when QALMA is imported, so the parallel paths are run in
a subprocess with the environment variables set. The subprocess counts the
executors it starts, to make sure that the parallel paths are really taken,
and compares their results with the serial ones.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

from qalma.parallel import optimal_workers

SCRIPT = textwrap.dedent("""
    import json
    import numpy as np
    import qalma.parallel as P
    import qalma.scalarprod.gram as G

    started = {"executors": 0}
    for module in (P, G):
        for name in ("ProcessPoolExecutor", "ThreadPoolExecutor"):
            original = getattr(module, name, None)
            if original is None:
                continue

            class Counted(original):
                def __init__(self, *args, **kwargs):
                    started["executors"] += 1
                    super().__init__(*args, **kwargs)

            setattr(module, name, Counted)
    # Force two workers, whatever the size of the tasks.
    P.optimal_workers = lambda n_tasks, scale, max_workers: 2

    from qalma import build_system
    from qalma.operators.functions.commutators import commutator_qalma_serial
    from qalma.operators.states import GibbsProductDensityOperator
    from qalma.projections.nbody import _project_monomial
    from qalma.scalarprod import fetch_covar_scalar_product
    from qalma.scalarprod.gram import gram_matrix

    rng = np.random.default_rng(0)
    system = build_system("open chain lattice", "spin", L=6, J=1.0)
    sites = tuple(system.sites)
    ops = [
        [system.site_operator(f"S{c}@{s}") for s in sites] for c in ("x", "y", "z")
    ]
    ham = system.global_operator("Hamiltonian")
    kappa = sum(rng.normal() * ops[c][i] for c in range(3) for i in range(6))
    sigma = GibbsProductDensityOperator(kappa.simplify()).to_product_state()

    def dense(op):
        return op.to_qutip(sites).full()

    # Projection of many three-body products.
    terms = []
    for _ in range(60):
        i = rng.integers(0, 4)
        a, b, c = rng.integers(0, 3, size=3)
        terms.append(rng.normal() * ops[a][i] * ops[b][i + 1] * ops[c][i + 2])
    terms = tuple(terms)
    before = started["executors"]
    parallel = P.parallel_process_non_dispatched_terms(terms, 2, sigma, max_workers=2)
    used_projection = started["executors"] - before
    serial = tuple(_project_monomial(t, 2, sigma) for t in terms)
    diff_projection = max(
        float(np.abs(dense(p) - dense(s)).max()) for p, s in zip(parallel, serial)
    )

    # Commutator of two sums of products.
    op_1 = sum(terms[:30]).simplify()
    op_2 = (ham + sum(terms[30:])).simplify()
    before = started["executors"]
    comm_parallel = P.commutator_qalma_parallel(op_1, op_2, num_workers=2)
    used_commutator = started["executors"] - before
    comm_serial = commutator_qalma_serial(op_1, op_2)
    diff_commutator = float(np.abs(dense(comm_parallel) - dense(comm_serial)).max())

    # The Gram matrix of the covariance product uses its own fast routine.
    sp = fetch_covar_scalar_product(sigma)
    basis = (op_1, op_2, ham)
    before = started["executors"]
    gram = gram_matrix(basis, sp)
    used_gram = started["executors"] - before
    diff_gram = float(np.abs(gram - sp.compute_gram_matrix(basis)).max())

    print(json.dumps({
        "use_parallel": P.USE_PARALLEL,
        "use_threads": P.USE_THREADS,
        "used_projection": used_projection,
        "used_commutator": used_commutator,
        "used_gram": used_gram,
        "diff_projection": diff_projection,
        "diff_commutator": diff_commutator,
        "diff_gram": diff_gram,
    }))
    """)


def run_with_environment(**variables):
    """Run SCRIPT in a subprocess with the given environment variables."""
    env = dict(os.environ, **variables)
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "n_tasks,scale,max_workers,expected",
    [
        (0, 0.1, 8, 1),
        (10, 0.1, 8, 1),
        (40, 0.1, 8, 2),
        (1000, 0.1, 8, 8),
        (1000, 0.1, 4, 4),
        (4000, 0.001, 8, 2),
        (10**6, 0.001, 1, 1),
    ],
)
def test_optimal_workers(n_tasks, scale, max_workers, expected):
    """The number of workers grows as sqrt(n_tasks), between 1 and max_workers."""
    assert optimal_workers(n_tasks, scale, max_workers) == expected


@pytest.mark.parametrize("threads", ["0", "1"], ids=["processes", "threads"])
def test_parallel_paths_match_serial(threads):
    """The parallel projection and commutator agree with the serial ones."""
    out = run_with_environment(
        QALMA_USE_PARALLEL="1",
        QALMA_MAX_WORKERS="2",
        QALMA_PARALLEL_USE_THREADS=threads,
    )
    assert out["use_parallel"]
    assert out["use_threads"] == (threads == "1")
    assert out["used_projection"] == 1
    assert out["used_commutator"] == 1
    assert out["diff_projection"] < 1e-12
    assert out["diff_commutator"] < 1e-12


def test_gram_matrix_keeps_the_fast_covariance_routine():
    """With parallel routines, the covariance Gram matrix still uses its own method."""
    out = run_with_environment(QALMA_USE_PARALLEL="1", QALMA_MAX_WORKERS="2")
    assert out["used_gram"] == 0
    assert out["diff_gram"] < 1e-12
