from test.helper import (
    HAMILTONIAN,
    SX_TOTAL,
    SY_TOTAL,
    SZ_TOTAL,
    TEST_CASES_STATES,
    check_equality,
)

import numpy as np

from qalma.evolution import (
    heisenberg_solve,
    qutip_me_solve,
)

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)


def test_rabi_evolution():
    rho0 = TEST_CASES_STATES[
        "ProductGibbs from local operator, hermitician"
    ].to_product_state()
    H = SY_TOTAL
    ts = np.linspace(0, 5, 100)
    e_ops = {
        "sxtotal": SX_TOTAL,
        "sytotal": SY_TOTAL,
        "sztotal": SZ_TOTAL,
        "Hamiltonian": HAMILTONIAN,
    }
    result = heisenberg_solve(H, rho0, ts, e_ops=e_ops, deep=4).expect_ops["sxtotal"]
    result_qutip = qutip_me_solve(H, rho0, ts, e_ops=e_ops).expect_ops["sxtotal"]
    print("len result=", len(result), "len qutip=", len(result_qutip))
    print("result", np.real(result))
    print("result_qutip", np.array(result_qutip))
    check_equality(np.real(result), np.array(result_qutip), tolerance=1e-5)
