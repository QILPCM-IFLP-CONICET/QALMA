import os
from test.helper import (
    OPERATOR_TYPE_CASES,
    TEST_CASES_STATES,
    check_equality,
)

import pytest

from qalma.scalarprod.build import covar_scalar_product
from qalma.settings import QALMA_TOLERANCE


@pytest.mark.skipif(
    not os.environ.get("QALMA_HEAVY_LOWLEVELTESTS", 0),
    reason="run only if explicitly required",
)
@pytest.mark.parametrize(["rho_name"], [(rho_name,) for rho_name in TEST_CASES_STATES])
def test_sp(rho_name):
    rho = TEST_CASES_STATES[rho_name]
    sp = covar_scalar_product(rho)
    rho_qutip = rho.to_qutip()
    for a_name in OPERATOR_TYPE_CASES:
        a = OPERATOR_TYPE_CASES[a_name]
        ad_qutip = a.dag().to_qutip()
        for b_name in OPERATOR_TYPE_CASES:
            b = OPERATOR_TYPE_CASES[b_name]
            b_qutip = b.to_qutip()
            qutip_sp = (
                rho_qutip * (ad_qutip * b_qutip + b_qutip * ad_qutip)
            ).tr() * 0.5

            assert check_equality(sp(a, b), qutip_sp, 1e2 * QALMA_TOLERANCE)
