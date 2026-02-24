from test.helper import check_equality

import numpy as np

from qalma.evolution import Simulation, qutip_me_solve
from qalma.model import build_system
from qalma.operators.states import GibbsDensityOperator


def test_empty():
    sim_empty = Simulation(
        parameters={}, stats={}, time_span=[], expect_ops={}, states=[]
    )
    sim_empty.save_hdf5("/tmp/sol.h5")
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

    solution.save_hdf5("/tmp/sol.h5")
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
    solution.save_hdf5("/tmp/sol.h5")
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
