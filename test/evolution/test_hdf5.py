from test.helper import check_equality

import numpy as np

from qalma.evolution import Simulation, qutip_me_solve
from qalma.model import build_system
from qalma.operators.states import GibbsDensityOperator


def test_qutip_me_solve():
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
    assert len(t_list) == len(stored.time_span)
    assert np.allclose(t_list, stored.time_span)
    sol_stats = solution.stats
    sto_stats = stored.stats
    for key in sol_stats:
        assert key in sto_stats
        assert (
            sto_stats[key] == sol_stats[key]
        ), f"{key}  have different values {sto_stats[key]}!={sol_stats[key]}"

    for key in solution.parameters:
        print(
            "comparing",
            key,
            [(type(x), x) for x in [stored.parameters[key], solution.parameters[key]]],
        )
        assert key in stored.parameters
        assert stored.parameters[key] == solution.parameters[key], key

    assert all(str(system) == str(state.system) for state in stored.states)
    assert stored.expect_ops == solution.expect_ops
    for rho, sigma in zip(solution.states, stored.states):
        check_equality(rho, sigma)
