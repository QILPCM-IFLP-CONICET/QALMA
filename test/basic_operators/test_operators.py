"""
Basic unit test.
"""

from test.helper import (
    CHAIN_SIZE,
    FULL_TEST_CASES,
    HAMILTONIAN,
    SITES,
    SX_A as local_SX_A,
    SY_A,
    SY_B,
    SZ_A,
    SZ_C,
    SZ_TOTAL,
    check_operator_equality,
)

import numpy as np
import pytest

from qalma.operators import (
    LocalOperator,
    OneBodyOperator,
    ProductOperator,
    QutipOperator,
    SumOperator,
)
from qalma.operators.states import DensityOperatorMixin

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)


QUTIP_OPS_CACHE = {}

SX_A = ProductOperator({local_SX_A.site: local_SX_A.operator}, 1.0, local_SX_A.system)
SX_A2 = SX_A * SX_A
SX_ASY_B = SX_A * SY_B
SX_AsyB_times_2 = 2 * SX_ASY_B
OP_GLOBAL = SZ_C + SX_AsyB_times_2

FULL_CHAIN = {f"1[{s}]" for s in range(CHAIN_SIZE)}
ACTS_OVER_RESULTS = {
    "scalar, zero": set(),
    "product, zero": set(),
    "product, 1": set(),
    "product, 2": set(),
    "scalar, real": set(),
    "scalar, complex": set(),
    "local operator, hermitician": {"1[0]"},
    "local operator, non hermitician": {"1[0]"},
    "One body, diagonal": FULL_CHAIN,
    "One body, hermitician": FULL_CHAIN,
    "One body, non hermitician": FULL_CHAIN,
    "three body, hermitician": {"1[0]", "1[1]", "1[2]"},
    "three body, non hermitician": FULL_CHAIN,
    "product operator, hermitician": {"1[0]", "1[1]"},
    "product operator, hermitician, twice": {"1[0]", "1[1]"},
    "product operator, non hermitician": {"1[0]", "1[1]"},
    "sum operator, hermitician": {"1[0]", "1[1]"},
    "sum operator, hermitician from non hermitician": {"1[0]", "1[1]"},
    "sum operator, anti-hermitician": {"1[0]", "1[1]"},
    "hermitician quadratic operator": FULL_CHAIN,
    "non hermitician quadratic operator": FULL_CHAIN,
    "fully mixed": FULL_CHAIN,
    "z semipolarized": FULL_CHAIN,
    "x semipolarized": FULL_CHAIN,
    "first full polarized": FULL_CHAIN,
    "gibbs_sz_as_product": FULL_CHAIN,
    "qutip operator": FULL_CHAIN,
    "qutip operator twice": FULL_CHAIN,
    "gibbs_sz": FULL_CHAIN,
    "gibbs_sz_bar": FULL_CHAIN,
    "gibbs_H": FULL_CHAIN,
    "mixture": FULL_CHAIN,
    "mixture of first and second partially polarized": FULL_CHAIN,
    "log unitary": FULL_CHAIN,
    "single interaction term": {"1[1]", "1[0]"},
    "sum local operators": {"1[0]"},
    "sum local qutip operators": {"1[0]"},
    "sum local qutip operator and local operator": {"1[0]", "1[1]"},
    "sum two-body qutip operators": {"1[0]", "1[1]"},
}


@pytest.mark.parametrize(["name", "operator"], list(FULL_TEST_CASES.items()))
def test_acts_over(name, operator):
    """Check acts_over method"""
    print(name)
    acts_over = operator.acts_over()
    print("    acts over ", acts_over)
    assert acts_over == ACTS_OVER_RESULTS[name]


@pytest.mark.parametrize(
    ("name1", "operator1", "name2", "operator2"),
    [
        (
            name1,
            operator1,
            name2,
            operator2,
        )
        for name1, operator1 in FULL_TEST_CASES.items()
        for name2, operator2 in FULL_TEST_CASES.items()
    ],
)
def test_product_and_trace(name1, operator1, name2, operator2):
    """Check acts_over method"""
    skip_cases = {
        "hermitician quadratic operator",
        "non hermitician quadratic operator",
    }
    if name1 in skip_cases or name2 in skip_cases:
        return
    op_qutip_1 = QUTIP_OPS_CACHE.get(name1, None)
    if op_qutip_1 is None:
        op_qutip_1 = operator1.to_qutip()
        QUTIP_OPS_CACHE[name1] = op_qutip_1

    if name1 == name2:
        print("checking the trace of ", name1)

        if not abs(operator1.tr() - op_qutip_1.tr()) < 1.0e-10:
            assert False, ("   failed:", "\033[91mtraces should match.\033[0m")

    # Mix of two
    print("checking the trace of the product of ", name1, "and", name2)
    op_qutip_2 = QUTIP_OPS_CACHE.get(name2, None)
    if op_qutip_2 is None:
        op_qutip_2 = operator2.to_qutip()
        QUTIP_OPS_CACHE[name2] = op_qutip_2
    # The trace of the products should match
    prod = operator1 * operator2
    if isinstance(prod, DensityOperatorMixin):
        assert False, (
            "\033[91m  failed: \033[0m",
            f"Product of {type(operator1)}*{type(operator2)} is a density matrix.",
        )

    alps_trace = (prod).tr()
    qutip_trace = (op_qutip_1 * op_qutip_2).tr()
    if not abs(alps_trace - qutip_trace) < 1.0e-10:
        assert False, (
            "\033[91m  failed:\033[0m",
            f"the traces of the products should match. {type(operator1)}*{type(operator2)}->{type(prod)}, {alps_trace}!={qutip_trace}",
        )
    print("\033[92m  passed.\033[0m")


def test_build_hamiltonian():
    """build ham"""
    assert SZ_TOTAL is not None
    assert HAMILTONIAN is not None
    hamiltonian_with_field = HAMILTONIAN + SZ_TOTAL
    assert check_operator_equality(
        (hamiltonian_with_field).to_qutip(),
        (HAMILTONIAN.to_qutip() + SZ_TOTAL.to_qutip()),
    )


def test_type_operator():
    """Tests for operator types"""
    assert isinstance(SX_A, ProductOperator)
    assert isinstance(SY_B, LocalOperator)
    assert isinstance(SZ_C, LocalOperator)
    assert isinstance(2 * SY_B, LocalOperator)
    assert isinstance(SY_B * 2, LocalOperator)
    assert isinstance(SX_A + SY_B, OneBodyOperator)
    assert isinstance(
        SX_A + SY_B + SZ_C, OneBodyOperator
    ), f"{type(SX_A + SY_B + SZ_C)} is not OneBodyOperator"
    assert isinstance(SX_A + SY_B + SX_A * SZ_C, SumOperator)
    assert isinstance((SX_A + SY_B), OneBodyOperator)
    assert isinstance(
        (SX_A + SY_B) * 3, OneBodyOperator
    ), f"{type((SX_A + SY_B)*3.)} is not OneBodyOperator"
    assert isinstance(
        3.0 * (SX_A + SY_B), OneBodyOperator
    ), f"{type(3.0*(SX_A + SY_B))} is not OneBodyOperator"

    assert isinstance((SX_A + SY_B) * 2.0, OneBodyOperator)
    assert isinstance(SY_B + SX_A, OneBodyOperator)
    assert isinstance(SX_ASY_B, ProductOperator)
    assert len(SX_ASY_B.sites_op) == 2
    assert isinstance(SX_AsyB_times_2, ProductOperator)
    assert isinstance(OP_GLOBAL, SumOperator)
    assert isinstance(SX_A + SY_B, SumOperator)
    assert len(SX_AsyB_times_2.sites_op) == 2

    OP_GLOBAL.prefactor = 2
    assert SX_A2.prefactor == 1
    assert OP_GLOBAL.prefactor == 2

    assert check_operator_equality(SX_A, SX_A.to_qutip())
    terms = [SX_A, SY_A, SZ_A]
    assert check_operator_equality(sum(terms), sum(t.to_qutip() for t in terms))
    assert check_operator_equality(SX_A.inv(), SX_A.to_qutip().inv())
    OP_GLOBAL_offset = OP_GLOBAL + 1.3821
    assert check_operator_equality(
        OP_GLOBAL_offset.inv(), OP_GLOBAL_offset.to_qutip().inv()
    )


def test_inv_operator():
    """test the exponentiation of different kind of operators"""
    sx_a_inv = SX_A.inv()
    assert isinstance(sx_a_inv, LocalOperator)
    assert check_operator_equality(sx_a_inv.to_qutip(), SX_A.to_qutip().inv())

    sx_obl = SX_A + SY_B + SZ_C
    sx_obl_inv = sx_obl.inv()
    assert isinstance(sx_obl_inv, QutipOperator)
    assert check_operator_equality(sx_obl_inv.to_qutip(), sx_obl.to_qutip().inv())

    s_prod = SX_A * SY_B * SZ_C
    s_prod_inv = s_prod.inv()
    assert isinstance(s_prod, ProductOperator)
    assert check_operator_equality(s_prod_inv.to_qutip(), s_prod.to_qutip().inv())

    OP_GLOBAL_offset = OP_GLOBAL + 1.3821
    OP_GLOBAL_offset_inv = OP_GLOBAL_offset.inv()
    assert isinstance(OP_GLOBAL_offset_inv, QutipOperator)
    assert check_operator_equality(
        OP_GLOBAL_offset_inv.to_qutip(), OP_GLOBAL_offset.to_qutip().inv()
    )


def test_exp_operator():
    """test the exponentiation of different kind of operators"""
    SX_A_exp = SX_A.expm()
    assert isinstance(SX_A_exp, LocalOperator)
    assert check_operator_equality(SX_A_exp.to_qutip(), SX_A.to_qutip().expm())

    sx_obl = SX_A + SY_B + SZ_C
    print("type sx_obl", type(sx_obl))
    print("sx_obl:\n", sx_obl)
    print("sx_obl->qutip:", sx_obl.to_qutip())

    sx_obl_exp = sx_obl.expm()
    assert isinstance(sx_obl_exp, ProductOperator)
    print("exponential and conversion:")
    print(sx_obl_exp.to_qutip())
    print("conversion and then exponential:")
    print(sx_obl.to_qutip().expm())
    assert check_operator_equality(sx_obl_exp.to_qutip(), sx_obl.to_qutip().expm())

    OP_GLOBAL_exp = OP_GLOBAL.expm()
    assert isinstance(OP_GLOBAL_exp, QutipOperator)
    assert check_operator_equality(
        OP_GLOBAL_exp.to_qutip(), OP_GLOBAL.to_qutip().expm()
    )


def test_local_operator():
    """Tests for local operators"""
    assert (SX_A * SX_A).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)
    assert (SZ_A * SZ_A).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)

    print("product * local", type(SX_A * SY_A))
    print("local * product", type(SY_A * SX_A))
    print("commutator:", type(SX_A * SY_A - SY_A * SX_A))
    print(
        ((SX_A * SY_A - SY_A * SX_A) * SZ_A).tr(),
        -1j * 0.5 * 2 ** (CHAIN_SIZE - 1),
    )
    assert ((SX_A * SY_A - SY_A * SX_A) * SZ_A).tr() == (
        -1j * 0.5 * 2 ** (CHAIN_SIZE - 1)
    )
    assert (SZ_A * (SX_A * SY_A - SY_A * SX_A)).tr() == (
        -1j * 0.5 * 2 ** (CHAIN_SIZE - 1)
    )

    assert (SZ_A * SY_B * SZ_A * SY_B).tr() == 0.25 * 2 ** (CHAIN_SIZE - 2)
    assert (
        (SX_A * SY_A * SY_B - SY_A * SX_A * SY_B) * (SZ_A * SY_B)
    ).tr() == -1j * 0.25 * 2 ** (CHAIN_SIZE - 2)
    assert (
        (SZ_A * SY_B) * (SX_A * SY_A * SY_B - SY_A * SX_A * SY_B)
    ).tr() == -1j * 0.25 * 2 ** (CHAIN_SIZE - 2)

    assert SX_A.tr() == 0.0
    assert (SX_A2).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)
    assert (SZ_C * SZ_C).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)

    SX_A_qt = SX_A.to_qutip()
    SX_A2_qt = SX_A2.to_qutip()

    assert SX_A_qt.tr() == 0
    assert (SX_A2_qt).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)
    assert SX_A.partial_trace(frozenset((SITES[0],))).tr() == 0.0
    assert SX_A.partial_trace(frozenset((SITES[1],))).tr() == 0.0
    assert SX_A.partial_trace(frozenset((SITES[0], SITES[1]))).tr() == 0.0
    assert SX_A.partial_trace(frozenset((SITES[1], SITES[2]))).tr() == 0.0


def test_product_operator():
    """Tests for product operators"""

    assert (SX_ASY_B * SX_A * SY_B).tr() == 0.25 * 2 ** (CHAIN_SIZE - 2)
    assert (OP_GLOBAL * SX_A * SY_B).tr() == 0.5 * 2 ** (CHAIN_SIZE - 2)
    assert (SX_AsyB_times_2 * SX_AsyB_times_2).tr() == 2 ** (CHAIN_SIZE - 2)
    assert (OP_GLOBAL * OP_GLOBAL).tr() == 2 ** (CHAIN_SIZE - 2) * 2

    SX_A_qt = SX_A.to_qutip()
    syB_qt = SY_B.to_qutip()
    szC_qt = SZ_C.to_qutip()

    SX_AsyB_qt = SX_ASY_B.to_qutip()
    SX_AsyB_times_2_qt = SX_AsyB_times_2.to_qutip()
    OP_GLOBAL_qt = OP_GLOBAL.to_qutip()

    assert (SX_AsyB_qt * SX_A_qt * syB_qt).tr() == 0.25 * 2 ** (CHAIN_SIZE - 2)
    assert (OP_GLOBAL_qt * SX_A_qt * syB_qt).tr() == 0.5 * 2 ** (CHAIN_SIZE - 2)
    assert (szC_qt * szC_qt).tr() == 0.5 * 2 ** (CHAIN_SIZE - 1)
    assert (SX_AsyB_times_2_qt * SX_AsyB_times_2_qt).tr() == 2 ** (CHAIN_SIZE - 2)
    assert (OP_GLOBAL_qt * OP_GLOBAL_qt).tr() == 2 ** (CHAIN_SIZE - 2) * 2
