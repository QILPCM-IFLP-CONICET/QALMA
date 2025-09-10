"""
Basic unit test.
"""

import numpy as np
import pytest
from qutip import Qobj, create, jmat, qeye, sigmax, sigmay, sigmaz, tensor

from qalma.operators import Operator, ProductOperator, QutipOperator, ScalarOperator
from qalma.qutip_tools.tools import (
    data_element_iterator,
    data_get_type,
    data_is_diagonal,
    data_is_scalar,
    data_is_zero,
    decompose_qutip_operator,
    decompose_qutip_operator_hermitician,
    norm,
    reduce_to_proper_spaces,
    schmidt_dec_first_rest_qutip_operator,
    schmidt_dec_first_rest_qutip_operator_hermitician,
    schmidt_dec_rest_last_qutip_operator,
    schmidt_dec_rest_last_qutip_operator_hermitician,
)
from qalma.settings import QALMA_TOLERANCE

from .helper import (
    CHAIN_SIZE,
    OPERATOR_TYPE_CASES,
    SITES,
    SX_A as LOCAL_SX_A,
    SY_B,
    SZ_C,
    check_equality,
    check_operator_equality,
)

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)


SX_A = ProductOperator({LOCAL_SX_A.site: LOCAL_SX_A.operator}, 1.0, LOCAL_SX_A.system)
SX_A2 = SX_A * SX_A
SX_A_SY_B = SX_A * SY_B
SX_A_SY_B_TIMES_2 = 2 * SX_A_SY_B
OP_GLOBAL = SZ_C + SX_A_SY_B_TIMES_2


SX_A_QT = SX_A.to_qutip_operator()
SX_A2_QT = SX_A_QT * SX_A_QT

SY_B_QT = SY_B.to_qutip_operator()
SZ_C_QT = SZ_C.to_qutip_operator()
SX_A_SY_B_QT = SX_A_QT * SY_B_QT
SX_A_SY_B_TIMES_2_QT = 2 * SX_A_SY_B_QT
OP_GLOBAL_QT = SZ_C_QT + SX_A_SY_B_TIMES_2_QT


SX_A_QT_NATIVE = SX_A_QT.to_qutip()
SX_A2_QT_NATIVE = SX_A_QT_NATIVE * SX_A_QT_NATIVE

SY_B_QT_NATIVE = SY_B_QT.to_qutip()
SZ_C_QT_NATIVE = SZ_C_QT.to_qutip()
SX_A_SY_B_QT_NATIVE = SX_A_QT_NATIVE * SY_B_QT_NATIVE
SX_A_SY_B_TIMES_2_QT_NATIVE = 2 * SX_A_SY_B_QT_NATIVE
OP_GLOBAL_QT_NATIVE = SZ_C_QT_NATIVE + SX_A_SY_B_TIMES_2_QT_NATIVE


ID_2_QUTIP = qeye(2)
ID_3_QUTIP = qeye(3)
SX_QUTIP, SY_QUTIP, SZ_QUTIP = jmat(0.5)
LX_QUTIP, LY_QUTIP, LZ_QUTIP = jmat(1.0)


SUBSYSTEMS = [
    frozenset((SITES[0],)),
    frozenset((SITES[1],)),
    frozenset((SITES[3],)),
    frozenset(
        (
            SITES[0],
            SITES[1],
        )
    ),
    frozenset(
        (
            SITES[1],
            SITES[2],
        )
    ),
]


QUTIP_TEST_CASES = {
    "hermitician quadratic operator": {
        "operator": OPERATOR_TYPE_CASES["hermitician quadratic operator"].to_qutip()
        * 2,
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "product_scalar": {
        "operator": 3 * tensor(ID_2_QUTIP, ID_3_QUTIP),
        "diagonal": True,
        "scalar": True,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "product_diagonal": {
        "operator": tensor(SZ_QUTIP, LZ_QUTIP),
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "product_non_diagonal": {
        "operator": tensor(SX_QUTIP, LX_QUTIP),
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "product_zero": {
        "operator": 0 * tensor(ID_2_QUTIP, ID_3_QUTIP),
        "diagonal": True,
        "scalar": True,
        "zero": True,
        "type": np.dtype("complex128"),
    },
    "scalar": {
        "operator": 3 * ID_2_QUTIP,
        "diagonal": True,
        "scalar": True,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "diagonal": {
        "operator": SZ_QUTIP + 0.5 * ID_2_QUTIP,
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "non_diagonal": {
        "operator": LX_QUTIP,
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "zero": {
        "operator": 0 * ID_2_QUTIP,
        "diagonal": True,
        "scalar": True,
        "zero": True,
        "type": np.dtype("complex128"),
    },
    "complex": {
        "operator": SY_QUTIP,
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "dense": {
        "operator": SX_QUTIP.expm(),
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "diagonal dense": {
        "operator": SZ_QUTIP.expm(),
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "tensor dense": {
        "operator": tensor(SX_QUTIP, LX_QUTIP).expm(),
        "diagonal": False,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "tensor diagonal dense": {
        "operator": tensor(SZ_QUTIP, SZ_QUTIP).expm(),
        "diagonal": True,
        "scalar": False,
        "zero": False,
        "type": np.dtype("complex128"),
    },
    "diagonal dense zero": {
        "operator": (-10000 * (ID_2_QUTIP + SZ_QUTIP)).expm(),
        "diagonal": True,
        "scalar": True,
        "zero": True,
        "type": np.dtype("complex128"),
    },
}


QUTIP_DECOMPOSITION_TEST_CASES = {
    "zero operator AB": tensor(ID_2_QUTIP, ID_3_QUTIP) * 0,
    "zero operator ABC": tensor(ID_2_QUTIP, ID_3_QUTIP, ID_2_QUTIP) * 0,
    "identity operator AB": tensor(ID_2_QUTIP, ID_3_QUTIP),
    "identity operator ABC": tensor(ID_2_QUTIP, ID_3_QUTIP, ID_2_QUTIP),
    "product operator AB": tensor(create(2), create(3)),
    "product operator ABC": tensor(create(2), ID_3_QUTIP, create(3)),
    "product operator complex ABC": tensor(create(2), ID_3_QUTIP, 1j * create(3)),
    "product operator hermitician AB": tensor(SY_QUTIP, LX_QUTIP),
    "product operator hermitician ABC": tensor(SX_QUTIP, ID_3_QUTIP, LX_QUTIP),
    "sum of product operator hermitician AB": tensor(SX_QUTIP, LX_QUTIP)
    + tensor(SY_QUTIP, LY_QUTIP),
    "sum of product operator hermitician ABC": (
        tensor(SX_QUTIP, ID_3_QUTIP, LX_QUTIP) + tensor(SY_QUTIP, ID_3_QUTIP, LY_QUTIP)
    ),
    "sum of product operator hermitician with diagonal terms ABC": (
        tensor(SX_QUTIP, ID_3_QUTIP, LX_QUTIP)
        + tensor(SY_QUTIP, ID_3_QUTIP, LY_QUTIP)
        + tensor(SZ_QUTIP, ID_3_QUTIP, LZ_QUTIP)
    ),
    "complex sum of product operator hermitician with diagonal terms ABC": (
        tensor(SX_QUTIP, ID_3_QUTIP, LX_QUTIP)
        + tensor(SY_QUTIP, ID_3_QUTIP, LY_QUTIP)
        + tensor(SZ_QUTIP, ID_3_QUTIP, LZ_QUTIP)
        + tensor(SX_QUTIP, LX_QUTIP, ID_3_QUTIP)
        + tensor(SY_QUTIP, LY_QUTIP, ID_3_QUTIP)
        + tensor(SZ_QUTIP, LZ_QUTIP, ID_3_QUTIP)
    ),
}


@pytest.mark.parametrize(("case", "operator_spec"), list(QUTIP_TEST_CASES.items()))
def test_data_element_iterator(case, operator_spec):

    operator = operator_spec["operator"]
    dtype = operator_spec["type"]
    data = operator.data
    array = data.to_array()

    def build_array(op, dims, dtype):
        result = np.zeros(dims, dtype=dtype)
        for i, j, val in data_element_iterator(op):
            result[i, j] = val

        return result

    reconstructed = build_array(data, data.shape, dtype)
    print(case, type(operator.data))
    print("array:", type(array), "\n", array.real)
    print("reconstructed:", type(reconstructed), "\n", reconstructed.real)
    print("diferencia:\n", array.real - reconstructed.real)

    assert (reconstructed == array).all()


@pytest.mark.parametrize(["case", "data"], list(QUTIP_TEST_CASES.items()))
def test_qutip_properties(case, data):
    print("testing ", case)
    operator_data = data["operator"].data
    assert data["diagonal"] == data_is_diagonal(operator_data)
    assert data["scalar"] == data_is_scalar(operator_data)
    assert data["zero"] == data_is_zero(operator_data)
    assert data["type"] is data_get_type(operator_data)


@pytest.mark.parametrize(
    ["name", "qutip_operator"], list(QUTIP_DECOMPOSITION_TEST_CASES.items())
)
def test_schmidt_dec_first_rest_qutip_operator(name, qutip_operator):
    """
    test decomposition of qutip operators
    as sums of product operators
    """
    print("decomposing ", name)
    print("qutip operator:\n", qutip_operator)
    terms = schmidt_dec_first_rest_qutip_operator(qutip_operator)
    if terms[0] or not data_is_zero(qutip_operator.data):
        reconstructed = sum(tensor(*t) for t in zip(*terms))
        assert check_operator_equality(
            qutip_operator, reconstructed
        ), "reconstruction does not match with the original."

    if not qutip_operator.isherm:
        return

    print("check hermitician decomposition")
    terms = schmidt_dec_first_rest_qutip_operator_hermitician(qutip_operator)
    if terms[0] or not data_is_zero(qutip_operator.data):
        reconstructed = sum(tensor(*t) for t in zip(*terms))
        print("reconstructed:\n", reconstructed)
        assert check_operator_equality(
            qutip_operator, reconstructed
        ), "hermitician reconstruction does not match with the original."


@pytest.mark.parametrize(
    ["name", "qutip_operator"], list(QUTIP_DECOMPOSITION_TEST_CASES.items())
)
def test_schmidt_dec_rest_last_qutip_operator(name, qutip_operator):
    """
    test decomposition of qutip operators
    as sums of product operators
    """
    print("decomposing ", name)
    print("qutip operator:\n", qutip_operator)
    terms = schmidt_dec_rest_last_qutip_operator(qutip_operator)
    if terms[0] or not data_is_zero(qutip_operator.data):
        reconstructed = sum((tensor(*t) for t in zip(*terms)), 0)
        assert check_operator_equality(
            qutip_operator, reconstructed
        ), "reconstruction does not match with the original."

    if not qutip_operator.isherm:
        return

    print("check hermitician decomposition")
    terms = schmidt_dec_rest_last_qutip_operator_hermitician(qutip_operator)
    if terms[0] or not data_is_zero(qutip_operator.data):
        reconstructed = sum(tensor(*t) for t in zip(*terms))
        assert check_operator_equality(
            qutip_operator, reconstructed
        ), "hermitician reconstruction does not match with the original."


@pytest.mark.parametrize(
    ["name", "qutip_operator"], list(QUTIP_DECOMPOSITION_TEST_CASES.items())
)
def test_decompose_qutip_operators(name, qutip_operator):
    """
    test decomposition of qutip operators
    as sums of product operators
    """
    print("decomposing ", name)

    print("qutip operator:\n", qutip_operator)
    terms = decompose_qutip_operator(qutip_operator)
    print("terms:", type(terms), terms)
    reconstructed = sum(tensor(*t) for t in terms)
    assert check_operator_equality(
        qutip_operator, reconstructed
    ), "reconstruction does not match with the original."

    if not qutip_operator.isherm:
        return

    print("check hermitician decomposition")
    terms = decompose_qutip_operator_hermitician(qutip_operator)
    print("terms:\n", 20 * "-")
    for term in terms:
        print(term)
        print("+")
    print(20 * "-")
    reconstructed = sum(tensor(*t) for t in terms)
    print("reconstructed:\n", reconstructed)
    assert check_operator_equality(
        qutip_operator, reconstructed
    ), "hermitician reconstruction does not match with the original."


@pytest.mark.parametrize(
    ("case", "op_case", "expected_value", "op_native"),
    [
        ("SX_A", SX_A_QT, 0.0, SX_A_QT_NATIVE),
        ("sy_B", SY_B_QT, 0.0, SY_B_QT_NATIVE),
        ("SX_A^2", SX_A2_QT, 0.25 * 2 ** (CHAIN_SIZE), SX_A2_QT_NATIVE),
        (
            "overlap (sxsy, sx*sy)",
            SX_A_SY_B_QT * SX_A_QT * SY_B_QT,
            0.25**2 * 2 ** (CHAIN_SIZE),
            SX_A_SY_B_QT_NATIVE * SX_A_QT_NATIVE * SY_B_QT_NATIVE,
        ),
        (
            "overlap (global, sx*sy)",
            OP_GLOBAL_QT * SX_A_QT * SY_B_QT,
            2 * (0.25**2) * 2 ** (CHAIN_SIZE),
            OP_GLOBAL_QT_NATIVE * SX_A_QT_NATIVE * SY_B_QT_NATIVE,
        ),
        (
            "Sz_C^2",
            SZ_C_QT * SZ_C_QT,
            0.25 * 2 ** (CHAIN_SIZE),
            SZ_C_QT_NATIVE * SZ_C_QT_NATIVE,
        ),
        (
            "sxsy^2",
            SX_A_SY_B_TIMES_2_QT * SX_A_SY_B_TIMES_2_QT,
            4 * (0.25**2) * (2**CHAIN_SIZE),
            SX_A_SY_B_TIMES_2_QT_NATIVE * SX_A_SY_B_TIMES_2_QT_NATIVE,
        ),
        (
            "global_qt^2",
            OP_GLOBAL_QT * OP_GLOBAL_QT,
            (0.25 + 4 * 0.25**2) * (2 ** (CHAIN_SIZE)),
            OP_GLOBAL_QT_NATIVE * OP_GLOBAL_QT_NATIVE,
        ),
        (
            "global * sx_A",
            OP_GLOBAL_QT * SX_A_QT,
            0.0,
            OP_GLOBAL_QT_NATIVE * SX_A_QT_NATIVE,
        ),
        (
            "sx_A * global",
            SX_A_QT * OP_GLOBAL_QT,
            0.0,
            SX_A_QT_NATIVE * OP_GLOBAL_QT_NATIVE,
        ),
        (
            "global^2",
            OP_GLOBAL * OP_GLOBAL,
            (0.25 + 4 * 0.25**2) * (2 ** (CHAIN_SIZE)),
            OP_GLOBAL_QT_NATIVE * OP_GLOBAL_QT_NATIVE,
        ),
        (
            "global * global_qt",
            OP_GLOBAL * OP_GLOBAL_QT,
            (0.25 + 4 * 0.25**2) * (2**CHAIN_SIZE),
            OP_GLOBAL_QT_NATIVE * OP_GLOBAL_QT_NATIVE,
        ),
        (
            "global_qt * global",
            OP_GLOBAL_QT * OP_GLOBAL,
            (0.25 + 4 * 0.25**2) * (2**CHAIN_SIZE),
            OP_GLOBAL_QT_NATIVE * OP_GLOBAL_QT_NATIVE,
        ),
    ],
)
def test_qutip_operators(
    case: str, op_case: Operator, expected_value: complex, op_native: Qobj
):
    """Test for the qutip representation"""
    print("testing case", case, expected_value, "=?=", op_native.tr())
    site_map = {site_name: i for i, site_name in enumerate(op_case.system.sites)}
    failed_tr = {}
    failed_pt = {}
    for subsystem in SUBSYSTEMS:
        ptoperator = (op_case).partial_trace(subsystem)
        site_indices = [site_map[site_name] for site_name in subsystem]
        native_pt = op_native.ptrace(site_indices)
        if not check_operator_equality(ptoperator.to_qutip(), native_pt):
            print("\n######  Ptrace failed for", (subsystem), "states are different:")
            failed_pt[subsystem] = (
                f"PT Operator:\n{ptoperator}\n\nToQutip:\n  {ptoperator.to_qutip()}\n\n native result:\n {native_pt}"
            )
            print(failed_pt[subsystem])

        if ptoperator.tr() != expected_value:
            print(
                "\n######  Ptrace failed for ",
                (subsystem),
                f"{ptoperator.tr()} != {expected_value}",
            )
            failed_tr[subsystem] = (
                f"value:{ptoperator.tr()}\n expected: {expected_value}\n\n Operator:\n {op_case}\n\n PTOperator:\n {ptoperator}"
            )
            print(failed_tr[subsystem])
    assert len(failed_pt) == 0, f"partial traces does not match for {failed_pt}"
    assert len(failed_tr) == 0, f"traces does not match for {failed_tr}"


@pytest.mark.parametrize(["name", "operator_case"], list(OPERATOR_TYPE_CASES.items()))
def test_as_sum_of_products(name, operator_case):
    """
    Convert qutip operators into product
    operators back and forward
    """
    print("testing QutipOperator.as_sum_of_products")
    print("   operator", name, "of type", type(operator_case))
    qutip_op = 3 * (operator_case.to_qutip_operator())

    # TODO: support handling hermitician operators
    if not qutip_op.isherm:
        return
    reconstructed = qutip_op.as_sum_of_products()
    qutip_op2 = reconstructed.to_qutip_operator()
    assert qutip_op.system == qutip_op2.system
    print("operator case: \n", 3 * operator_case.to_qutip())
    print("Qutip form:\n", qutip_op.to_qutip())
    print("reconstructed:\n", reconstructed.to_qutip())
    print("qutip form reconstructed:\n", qutip_op2.to_qutip())
    assert qutip_op.to_qutip() == qutip_op2.to_qutip()


def test_detached_operators():
    """Check operators not coming from a system"""
    # Tests for QutipOperators defined without a system
    test_op = SX_A_SY_B_TIMES_2
    system = test_op.system
    test_op_tr = test_op.tr()
    test_op_sq_tr = (test_op * test_op).tr()
    qutip_repr = test_op.to_qutip(tuple(system.sites))
    assert test_op_tr == qutip_repr.tr()
    assert test_op_sq_tr == (qutip_repr * qutip_repr).tr()

    # Now, build a detached operator
    detached_qutip_operator = QutipOperator(qutip_repr)
    assert test_op_tr == detached_qutip_operator.tr()
    assert test_op_sq_tr == (detached_qutip_operator * detached_qutip_operator).tr()

    # sites with names
    detached_qutip_operator = QutipOperator(
        qutip_repr, names={s: i for i, s in enumerate(SITES)}
    )
    assert test_op_tr == detached_qutip_operator.tr()
    assert test_op_sq_tr == (detached_qutip_operator * detached_qutip_operator).tr()
    assert (
        test_op_tr == detached_qutip_operator.partial_trace(frozenset(SITES[0:1])).tr()
    )
    assert (
        test_op_tr == detached_qutip_operator.partial_trace(frozenset(SITES[0:2])).tr()
    )


@pytest.mark.parametrize(["name", "spec"], list(QUTIP_TEST_CASES.items()))
def test_norm(name, spec):
    print("testing norm on", name)
    operator = spec["operator"]
    svlist = np.array(
        [x**0.5 for x in (operator.dag() * operator).eigenenergies() if x > 0]
    )
    frobenius_norm = (operator.dag() * operator).tr() ** 0.5
    nuclear_norm = sum(svlist)
    spectral_norm = max(svlist) if len(svlist) else 0.0
    value = norm(operator, ord="fro")
    assert (
        abs(value - frobenius_norm) < QALMA_TOLERANCE**0.5
    ), f"Frobenius norm failed for {name}: {value}!=  {frobenius_norm}."
    value = norm(operator, ord="nuc")
    assert (
        abs(value - nuclear_norm) < QALMA_TOLERANCE**0.25
    ), f"Nuclear norm failed for {name}: {value}!={nuclear_norm}."
    value = norm(operator, ord=2)
    assert (
        abs(value - spectral_norm) < QALMA_TOLERANCE**0.5
    ), f"spectral norm failed for {name}:  {value}!={spectral_norm}."


def test_to_qutip_operator():
    # special cases
    expected_types_to_qutip = {
        "scalar, zero": ScalarOperator,
        "scalar, real": ScalarOperator,
        "scalar, complex": ScalarOperator,
        "product, zero": ScalarOperator,
        "product, 1": ScalarOperator,
        "product, 2": ScalarOperator,
    }
    for name, op_case in OPERATOR_TYPE_CASES.items():
        expected_type = expected_types_to_qutip.get(name, QutipOperator)
        op_tqo = op_case.to_qutip_operator()
        assert isinstance(
            op_tqo, expected_type
        ), f"<<{name}>> to qutip operator results in {type(op_tqo)} instead of {expected_type}"


def test_reduce_to_proper_spaces():
    """
    Test the projections to proper spaces.
    """
    observable_x = tensor(sigmax(), ID_2_QUTIP) + tensor(ID_2_QUTIP, sigmax())
    observable_z = tensor(sigmaz(), ID_2_QUTIP) + tensor(ID_2_QUTIP, sigmaz())
    observable_h = observable_x + observable_z  # Hadamard
    corr_x = tensor(sigmax(), sigmax())
    corr_y = tensor(sigmay(), sigmay())

    global_id = tensor(ID_2_QUTIP, ID_2_QUTIP)
    full_mixture = 0.25 * global_id

    state = 0.25 * (global_id + corr_x)
    # The state and the observable are diagonalizable in the same basis (sx_total):
    reduced_state = reduce_to_proper_spaces(state, observable_x)
    assert check_operator_equality(reduced_state, state, 1e-9)

    # The observable is diagonal, and not in the same basis (sz_total)
    reduced_state = reduce_to_proper_spaces(state, observable_z)
    expected = full_mixture + 0.125 * corr_x + 0.125 * corr_y
    assert check_operator_equality(reduced_state, expected, 1e-9)

    # Non diagonal observable which does not commute with the state (Hadamard)
    reduced_state = reduce_to_proper_spaces(state, observable_h)
    # the reduced state should commute with the observable:
    assert check_operator_equality(
        reduced_state * observable_h, observable_h * reduced_state, 1e-9
    )
    # the spectrum for this case is 0, 1/4, 1/4, 1/2
    spectrum = reduced_state.eigenenergies()
    assert check_equality(
        spectrum, np.array([0, 0.25, 0.375, 0.375])
    ), f"{reduced_state.eigenenergies()}"


def test_trivial_QutipOperator():
    b = QutipOperator(sigmax(), prefactor=0.5)
    system = b.system
    a = QutipOperator(1, prefactor=2, system=system)
    ab = a * b
    print("a:", a)
    print("b:", b)
    print("ab:", ab)
    check_equality(ab, 2 * b)
    check_equality(a.tr(), 4.0)
    check_equality((a * a).tr(), 8.0)
    check_equality((b * b).tr(), 0.5)
