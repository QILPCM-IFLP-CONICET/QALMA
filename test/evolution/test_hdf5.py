from test.helper import check_equality

import numpy as np

from qalma.evolution import Simulation, qutip_me_solve
from qalma.model import build_system
from qalma.operators.states import GibbsDensityOperator


def test_store_and_recover_dicts():
    """Test stored dict routines"""
    from h5py import File
    from numpy import array as np_array, ndarray

    from qalma.evolution.simulation import load_hdf5_dict, store_hdf5_dict

    test_dict = {
        "str": "string",
        "bool": True,
        "float": 2.1,
        "int": 32,
        "tuple empty": tuple(),
        "tuple of numbers": (
            1,
            2,
            3,
        ),
        "tuple of bools": (
            True,
            False,
        ),
        "tuple of strings": (
            "s1",
            "s2",
            "hola!",
        ),
        "list empty": list(),
        "list of numbers": (1, 2, 3),
        "list of bools": [True, False],
        "list of strings": ["s1", "s2", "hola!"],
        "set of strings": set(("Hola", "no estoy")),
        "1d np array": np_array([1.0, 2.0]),
        "2d np array": np_array([[1.0, 2.0], [3.0, 4.0]]),
        "sub dict": {"a": 1, "b": 2.0},
    }
    with File("/tmp/test_dict.h5", "w") as f5:
        group = f5.create_group("test_dict")
        store_hdf5_dict(group, test_dict)
    with File("/tmp/test_dict.h5", "r") as f5:
        stored_dict = load_hdf5_dict(f5["test_dict"])

    def check_dict_equality(d1, d2):
        assert isinstance(d1, dict), "d1 is not a dict"
        assert isinstance(d2, dict), "d2 is not a dict"
        assert all(
            key in d2 for key in d1
        ), f"keys {','.join([k for k in d1 if k not in d2])} not in d2."
        assert all(
            key in d1 for key in d2
        ), f"keys {','.join([k for k in d2 if k not in d1])} not in d1."
        for key, val in d1.items():
            if isinstance(val, dict):
                check_dict_equality(val, d2[key])
            elif isinstance(val, ndarray):
                assert (val == d2[key]).all(), f"{val}!={d2[key]}"
            else:
                print(type(val), type(d2[key]))
                assert (
                    val == d2[key]
                ), f"key {key} have different values ({val}!={d2[key]})"

    check_dict_equality(stored_dict, test_dict)


def test_empty():
    sim_empty = Simulation(
        parameters={}, stats={}, time_span=[], expect_ops={}, states=[]
    )
    sim_empty.save_hdf5("/tmp/sol.h5", mode="w")
    sim = Simulation.load_hdf5("/tmp/sol.h5")
    compare_sim_objects(sim_empty, sim)
    assert len(sim.parameters) == 0
    assert len(sim.stats) == 0
    assert len(sim.time_span) == 0
    assert len(sim.expect_ops) == 0
    assert len(sim.states) == 0


def test_qutip_me_solve_states():
    system = build_system(
        parms={
            "L": 2,
            "a": 1,
            "h": 2,
            "J": 1,
        }
    )
    t_list = np.linspace(0, 3, 10)
    hamiltonian = system.global_operator("Hamiltonian")
    rho0 = GibbsDensityOperator(system.site_operator("Sz", "1[0]"))
    solution = qutip_me_solve(hamiltonian, rho0, t_list)

    solution.save_hdf5("/tmp/sol.h5", mode="w")
    stored = Simulation.load_hdf5("/tmp/sol.h5")
    compare_sim_objects(solution, stored)


def test_qutip_me_solve_expect():
    system = build_system(
        parms={
            "L": 2,
            "a": 1,
            "h": 2,
            "J": 1,
        }
    )
    t_list = np.linspace(0, 3, 10)
    hamiltonian = system.global_operator("Hamiltonian")
    rho0 = GibbsDensityOperator(system.site_operator("Sz", "1[0]"))
    solution = qutip_me_solve(
        hamiltonian,
        rho0,
        t_list,
        e_ops={
            "Sz_1": system.site_operator("Sz", "1[0]"),
            "Sz_2": system.site_operator("Sz", "1[1]"),
        },
    )
    solution.save_hdf5("/tmp/sol.h5", mode="w")
    assert len(solution.states) == 0 or (
        solution.states[0].system is system
    ), "Check that the system was not modified after serialization."
    stored = Simulation.load_hdf5("/tmp/sol.h5")
    compare_sim_objects(solution, stored)


def compare_sim_objects(solution, stored):
    assert len(solution.time_span) == len(stored.time_span)
    assert np.allclose(solution.time_span, stored.time_span)
    sol_stats = solution.stats
    sto_stats = stored.stats
    for key in sol_stats:
        assert key in sto_stats
        assert (
            sto_stats[key] == sol_stats[key]
        ), f"{key}  have different values {sto_stats[key]}!={sol_stats[key]}"

    for key in solution.parameters:
        assert key in stored.parameters
        assert stored.parameters[key] == solution.parameters[key], key

    system = solution.states[0].system if solution.states else None
    for state in stored.states:
        assert str(system) == str(state.system), f"{system}!={state.system}"
    assert all(
        key in stored.expect_ops for key in solution.expect_ops
    ), f"{key} is missing in the stored object."
    for key, val in solution.expect_ops.items():
        stored_val = stored.expect_ops[key]
        sim_val = np.array(val)
        try:
            assert check_equality(sim_val, stored_val)
        except AssertionError:
            print(f"{sim_val}!={stored_val} for {key}")
            print(type(sim_val), "=?=", type(stored_val))
            print(
                [
                    (
                        type(x),
                        type(y),
                    )
                    for x, y in zip(sim_val, stored_val)
                ]
            )
            raise

    for rho, sigma in zip(solution.states, stored.states):
        check_equality(rho, sigma)
