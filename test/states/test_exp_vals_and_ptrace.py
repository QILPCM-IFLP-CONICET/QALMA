"""
Basic unit test for states.
"""

from test.helper import (
    EXPECTATION_VALUE_TOLERANCE,
    OPERATOR_TYPE_CASES,
    TEST_CASES_STATES,
)

import pytest

from qalma.operators import QutipOperator

# from qalma.settings import VERBOSITY_LEVEL


@pytest.mark.parametrize(
    ["name_rho", "rho", "name_obs", "obs"],
    [
        (
            name_rho,
            rho,
            name_obs,
            obs,
        )
        for name_rho, rho in TEST_CASES_STATES.items()
        for name_obs, obs in OPERATOR_TYPE_CASES.items()
    ],
)
def test_pt_and_exp_vals(name_rho, rho, name_obs, obs):
    if len(rho.system.sites) > 8:
        return
    acts_over = obs.acts_over()
    if rho is None or acts_over is None or len(acts_over) == 0:
        return
    if len(acts_over) > 6:
        return

    acts_over_tuple = tuple(acts_over)
    acts_over_subsystem = frozenset(acts_over)
    rho_pt = rho.partial_trace(acts_over_subsystem)
    qutip_obs = obs if isinstance(obs, QutipOperator) else obs.to_qutip_operator()
    print(f"rho {type(rho)}:\n", rho.to_qutip())
    print(f"rho_pt {type(rho_pt)}:\n", rho_pt.to_qutip())

    print(f"obs {type(obs)}:\n", qutip_obs.to_qutip())

    print("acts_over_tuple from the observable:", acts_over_tuple)
    print("rho_pt", type(rho_pt), "acts_over", rho_pt.acts_over())
    print("qutip_obs.acts_over", qutip_obs.acts_over())

    comparisons = {
        "full_qutip": (rho.to_qutip() * qutip_obs.to_qutip()).tr(),
        "block_qutip": (
            rho_pt.to_qutip(acts_over_tuple) * qutip_obs.to_qutip(acts_over_tuple)
        ).tr(),
        "rho.expect": rho.expect(obs),
        "rho_pt.expect(obs)": rho_pt.expect(obs),
        "rho_pt.expect(qutip)": rho_pt.expect(obs.to_qutip_operator()),
    }
    if qutip_obs is not obs:
        comparisons["rho.expect(qutip_obs)"] = rho.expect(qutip_obs)
        comparisons["rho_pt.expect(qutip_obs)"] = rho_pt.expect(qutip_obs)

    for key, val in comparisons.items():
        abs_error = abs(comparisons["rho.expect"] - val)
        rel_error = abs_error / (1 + abs(val))
        assert (
            rel_error < EXPECTATION_VALUE_TOLERANCE
        ), f"For {key}, does not match: {val}!=  {comparisons['rho.expect']} + O({EXPECTATION_VALUE_TOLERANCE})"
