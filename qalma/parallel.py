"""Parallel routines."""

import logging
from functools import partial
from typing import Any, Dict, Optional, Tuple

import qalma.settings as qalma_settings
from qalma.operators import Operator
from qalma.operators.arithmetic import iterable_to_operator
from qalma.operators.simplify import _commutator_term_pairs, collect_nbody_terms
from qalma.operators.states.gibbs import (
    GibbsProductDensityOperator,
)
from qalma.operators.states.product import (
    ProductDensityOperator,
)

USE_PARALLEL = qalma_settings.USE_PARALLEL
MAX_WORKERS = qalma_settings.PARALLEL_MAX_WORKERS
USE_THREADS = qalma_settings.PARALLEL_USE_THREADS


DISPATCH_PROJECTION_METHOD_PARALLEL: Dict[Any, Any] = {}


if qalma_settings.USE_PARALLEL:
    try:
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

        logging.info("Using parallel routines for large objects.")
    except ModuleNotFoundError:
        USE_PARALLEL = False
        logging.warning(
            "ProcessPoolExecutor/ThreadPoolExecutor cannot be loaded. Using serial routines."
        )
        MAX_WORKERS = 1
        USE_THREADS = False
else:
    logging.info("Using serial routines for large objects.")
    MAX_WORKERS = 1
    USE_THREADS = False


def _commutator_term_worker(entries):
    """Compute the commutator [hi, kj] = hi * kj - kj * hi and simplify the result.

    The `system` attribute of the result is set to `None` to reduce the
    serializing cost.

    Parameters
    ----------
    entries : tuple
        A tuple (hi, kj) of SumOperator objects.

    Returns
    -------
    Operator
        The simplified commutator operator with small terms removed (threshold 1e-5).

    """
    op_1, op_2 = entries
    return (op_1 * op_2 - op_2 * op_1).simplify()._set_system_()


def commutator_qalma_parallel(
    op_1: Operator,
    op_2: Operator,
    use_threads: bool = USE_THREADS,
    num_workers: int = MAX_WORKERS,
) -> Operator:
    """Compute the commutator of two Operator objects `op_1` and `op_2`.

    Parallel implementation.

    Parameters
    ----------
    op_1 : Operator
        First operator.
    op_2 : Operator
        Second operator.
    use_threads : bool, optional
        Whether to use threads instead of processes.
    num_workers : int, optional
        Number of parallel workers.

    Returns
    -------
    Operator
        The commutator [op_1, op_2] = op_1 * op_2 - op_2 * op_1, simplified.

    """
    system = op_1.system.union(op_2.system)
    op_1_terms = collect_nbody_terms(op_1.flat())
    op_2_terms = collect_nbody_terms(op_2.flat())

    terms_pairs = tuple(_commutator_term_pairs(op_1_terms, op_2_terms))
    len_terms_pairs = len(terms_pairs)
    # The wall-time estimated for a task can be estimated as
    # WallTime =   N_TASKS / NUM_PROC * TIME_SINGLE_TASK + NUM_PROC * PARALLELIZING_OVERHEAD_TIME
    # The optimal number of processes is given by
    #  NUM_PROC = sqrt( N_TASKS *  TIME_SINGLE_TASK/PARALLELIZING_OVERHEAD_TIME )

    num_workers = min(1, max(num_workers, int((0.001 * len_terms_pairs) ** 0.5)))
    if num_workers == 1 or not USE_PARALLEL:
        terms = tuple(op_1 * op_2 - op_2 * op_1 for op_1, op_2 in terms_pairs)
        return iterable_to_operator(terms, system).simplify()

    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    chunksize = max(1, int(len_terms_pairs / num_workers))
    with executor_cls(max_workers=num_workers) as executor:
        terms = tuple(
            (
                val
                for val in executor.map(
                    _commutator_term_worker, terms_pairs, chunksize=chunksize
                )
                if val is not None
            )
        )
    return iterable_to_operator(terms, system)._set_system_(system).simplify()


def _project_monomial_worker(operator, nmax, sigma):
    """Project a single operator to the nmax subspace relative to state sigma."""
    return (
        DISPATCH_PROJECTION_METHOD_PARALLEL[type(operator)](operator, nmax, sigma)
        .simplify()
        ._set_system_(None)
    )


def parallel_process_non_dispatched_terms(
    terms: Tuple[Operator, ...],
    nmax: int,
    sigma: Optional[ProductDensityOperator | GibbsProductDensityOperator] = None,
    use_threads=USE_THREADS,
    max_workers=MAX_WORKERS,
) -> Tuple[Operator, ...]:
    """Project each operator in `terms` to the nmax subspace, relative to `sigma`.

    Parameters
    ----------
    terms : tuple of Operator
        Operators to project.
    nmax : int
        Maximum body order of the projection subspace.
    sigma : ProductDensityOperator or GibbsProductDensityOperator, optional
        Reference state for the projection.
    use_threads : bool, optional
        Whether to use threads instead of processes.
    max_workers : int, optional
        Maximum number of parallel workers.

    Returns
    -------
    tuple of Operator
        Projected operators with the system attribute restored.

    """
    system = terms[0].system
    non_dispatched_length = len(terms)
    project_worker = partial(_project_monomial_worker, nmax=nmax, sigma=sigma)
    executor_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    # The wall-time estimated for a task can be estimated as
    # WallTime =   N_TASKS / NUM_PROC * TIME_SINGLE_TASK + NUM_PROC * PARALLELIZING_OVERHEAD_TIME
    # The optimal number of processes is given by
    #  NUM_PROC = sqrt( N_TASKS *  TIME_SINGLE_TASK/PARALLELIZING_OVERHEAD_TIME )

    num_workers = min(1, max(max_workers, int((0.1 * non_dispatched_length) ** 0.5)))
    if num_workers > 1:
        chunksize = max(1, int(non_dispatched_length / num_workers))
        with executor_cls(max_workers=num_workers) as executor:
            terms = tuple(
                term
                for term in executor.map(project_worker, terms, chunksize=chunksize)
            )
    else:
        terms = tuple(_project_monomial_worker(term, nmax, sigma) for term in terms)
    for term in terms:
        term._set_system_(system)
    return terms
