import multiprocessing as mp
import pickle

import pytest

from qalma.geometry import GraphDescriptor
from qalma.model import SystemDescriptor
from qalma.operators import Operator

from .helper import (
    FULL_TEST_CASES,
    OPERATOR_TYPE_CASES,
    SYSTEM,
    TEST_CASES_STATES,
    check_operator_equality,
)

# Use forkserver for safety, but we will solve the speed issue
# by reusing the processes.
MP_CONTEXT_TYPE = "forkserver"


@pytest.fixture(scope="module")
def pool():
    """
    Creates a persistent pool of workers for this test module.
    The 'forkserver' overhead is paid only once when the pool starts.
    """
    ctx = mp.get_context(MP_CONTEXT_TYPE)
    # Preload your library in the server to speed up worker creation
    try:
        mp.set_forkserver_preload(["qalma"])
    except (AttributeError, ImportError):
        pass

    p = ctx.Pool(processes=4)
    yield p
    p.terminate()
    p.join()


def test_serialize_graph():
    print("test serialize graph")
    graph = SYSTEM.spec["graph"]
    a = pickle.dumps(graph)
    reconstructed_graph = pickle.loads(a)
    assert isinstance(reconstructed_graph, GraphDescriptor)
    assert graph == reconstructed_graph


def test_serialize_system():
    print("test serialize system")
    a = pickle.dumps(SYSTEM)
    reconstructed_system = pickle.loads(a)
    assert isinstance(reconstructed_system, SystemDescriptor)
    assert SYSTEM == reconstructed_system


@pytest.mark.parametrize(["name", "operator"], list(FULL_TEST_CASES.items()))
def test_serialize_operator(name, operator):
    print("test serialize", name)
    a = pickle.dumps(operator)
    reconstructed_operator = pickle.loads(a)
    assert isinstance(reconstructed_operator, Operator)
    reconstructed_operator._set_system_(operator.system)
    assert check_operator_equality(operator, reconstructed_operator, tolerance=1e-8)


# --- Worker Functions (Must be top-level for pickling) ---


def worker_add_task(args):
    """Logic for the 'add number' test moved to a standalone function."""
    operator, number = args
    if hasattr(operator, "terms"):
        system = operator.system
        for t in operator.terms:
            assert t.system is system
    return operator + number


def worker_expect_task(args):
    """Logic for the 'expect' test moved to a standalone function."""
    state, obs = args
    return state.expect(obs)


# --- Modified Multiprocess Tests ---


@pytest.mark.parametrize(["name", "operator"], list(FULL_TEST_CASES.items()))
def test_process_add_number(name, operator, pool):
    print("test process add number", name)

    if hasattr(operator, "terms"):
        system = operator.system
        for t in operator.terms:
            assert t.system is system

    # Use the pool to execute the task instead of starting a new Process
    result_worker = pool.apply(worker_add_task, ((operator, 1.0),))

    result_mine = operator + 1.0
    result_worker._set_system_(operator.system)
    result_mine._set_system_(operator.system)

    assert check_operator_equality(result_worker, result_mine, tolerance=1e-8)


@pytest.mark.parametrize(
    ["state_name", "operator_name"],
    [
        (state_name, operator_name)
        for state_name in TEST_CASES_STATES
        for operator_name in OPERATOR_TYPE_CASES
    ],
)
def test_process_expect(state_name, operator_name, pool):
    print("test process expect", state_name, operator_name)

    state = TEST_CASES_STATES[state_name]
    operator = OPERATOR_TYPE_CASES[operator_name]

    # Send work to the existing pool
    result_worker = pool.apply(worker_expect_task, ((state, operator),))

    result_mine = state.expect(operator)
    assert abs(result_worker - result_mine) < 1e-9
