"""Tests for ``qalma.utils``: export to Wolfram, drawing, and convex fits."""

from test.helper import SITES, SX_A, SX_B, SYSTEM, SZ_B, SZ_C

import numpy as np
import pytest
import qutip

from qalma.operators import ProductOperator
from qalma.utils import (
    convex_fit_derivative,
    draw_operator,
    matrix_to_wolfram,
    operator_to_wolfram,
)

# ---------------------------------------------------------------------------
# matrix_to_wolfram / operator_to_wolfram
# ---------------------------------------------------------------------------


def test_matrix_to_wolfram_real():
    """Every row of the matrix is exported."""
    matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert matrix_to_wolfram(matrix) == "{\n{0.0, 1.0},\n{1.0, 0.0}\n}"


def test_matrix_to_wolfram_complex():
    """Complex entries use the ``I`` of Mathematica."""
    matrix = np.array([[1.0, 2j], [0.5, -1 + 1j]])
    assert matrix_to_wolfram(matrix) == "{\n{1.0, 2I},\n{0.5, -1+1I}\n}"


def test_matrix_to_wolfram_exponents():
    """Scientific notation is translated to Mathematica's ``*^``."""
    matrix = np.array([[1.0e-10, 0.0], [0.0, 1.5e-10]])
    assert matrix_to_wolfram(matrix) == "{\n{1*^-10, 0.0},\n{0.0, 1.5*^-10}\n}"


@pytest.mark.parametrize("n_rows", [1, 2, 5])
def test_matrix_to_wolfram_number_of_rows(n_rows):
    """The output has one ``{...}`` row per matrix row."""
    matrix = np.arange(n_rows * 3, dtype=float).reshape(n_rows, 3)
    lines = matrix_to_wolfram(matrix).split("\n")
    assert lines[0] == "{" and lines[-1] == "}"
    assert len(lines) == n_rows + 2
    assert all(line.startswith("{") for line in lines[1:-1])


@pytest.mark.parametrize("dtype", [int, np.int32, np.uint8])
def test_matrix_to_wolfram_integers(dtype):
    """Integer matrices are exported with WL integers."""
    matrix = np.array([[1, 0], [3, 2]], dtype=dtype)
    assert matrix_to_wolfram(matrix) == "{\n{1, 0},\n{3, 2}\n}"


def test_matrix_to_wolfram_negative_integers():
    """The sign of the integers is kept."""
    assert matrix_to_wolfram(np.array([[-1, 2]])) == "{\n{-1, 2}\n}"


def test_matrix_to_wolfram_mixed_entries():
    """Integers, reals and complex numbers can be mixed in an object array."""
    matrix = np.array([[1, 0.5], [2j, 1 + 0j]], dtype=object)
    assert matrix_to_wolfram(matrix) == "{\n{1, 0.5},\n{2I, 1.0}\n}"


def test_matrix_to_wolfram_single_precision():
    """Other floating point types are accepted."""
    matrix = np.array([[0.5]], dtype=np.float32)
    assert matrix_to_wolfram(matrix) == "{\n{0.5}\n}"


@pytest.mark.parametrize(
    ["number", "expected"],
    [
        (3, "3"),
        (-3, "-3"),
        (2.5, "2.5"),
        (1.0e-10, "1*^-10"),
        (1 + 2j, "1+2I"),
        (2 + 0j, "2.0"),
    ],
)
def test_matrix_to_wolfram_number(number, expected):
    """Scalars are exported using the same conventions as the entries."""
    assert matrix_to_wolfram(number) == expected


@pytest.mark.parametrize(
    "wrong_input",
    [
        "not a matrix",
        None,
        np.array([1.0, 2.0]),  # vector, not matrix
        np.array([[True, False]]),
        np.array([["a", "b"]]),
    ],
    ids=["string", "None", "vector", "booleans", "strings"],
)
def test_matrix_to_wolfram_rejects_wrong_inputs(wrong_input):
    """Inputs that are not numeric matrices are rejected."""
    with pytest.raises(AssertionError):
        matrix_to_wolfram(wrong_input)


def test_operator_to_wolfram_qobj():
    """A ``Qobj`` is exported as a matrix."""
    assert operator_to_wolfram(qutip.sigmax()) == "{\n{0.0, 1.0},\n{1.0, 0.0}\n}"


def test_operator_to_wolfram_local_operator():
    """A local operator becomes a Kronecker product with one factor per site."""
    result = operator_to_wolfram(SX_A)
    assert result.startswith("KroneckerProduct[") and result.endswith("]")
    assert result.count("{\n{") == len(SITES)
    # The only non-trivial factor is the first one.
    assert result.count("{0.0, 0.5}") == 1


def test_operator_to_wolfram_sum_operator():
    """A sum of operators is exported as a sum of Kronecker products."""
    result = operator_to_wolfram(SX_A + SZ_B)
    assert result.startswith("(KroneckerProduct[") and result.endswith("])")
    assert result.count(" + ") == 1
    assert result.count("KroneckerProduct[") == 2


def test_operator_to_wolfram_zero_prefactor():
    """A product operator with a null prefactor is exported as ``0``."""
    assert operator_to_wolfram(ProductOperator({}, prefactor=0.0, system=SYSTEM)) == "0"


# ---------------------------------------------------------------------------
# draw_operator
# ---------------------------------------------------------------------------


@pytest.fixture(name="axis")
def fixture_axis():
    """An empty matplotlib axis (matplotlib is an optional dependency)."""
    pytest.importorskip("matplotlib")
    # pylint: disable=import-outside-toplevel
    from matplotlib.figure import Figure

    return Figure().add_subplot()


def test_draw_operator_one_site(axis):
    """A single-site operator is a disk, and nothing else."""
    assert draw_operator(SX_A, axis) is axis
    assert len(axis.patches) == 1
    assert type(axis.patches[0]).__name__ == "Circle"
    assert len(axis.lines) == 0


def test_draw_operator_two_sites(axis):
    """A two-site operator is an ellipse around both sites."""
    assert draw_operator(SX_A * SX_B, axis) is axis
    assert len(axis.patches) == 1
    assert type(axis.patches[0]).__name__ == "Ellipse"
    assert len(axis.lines) == 0


def test_draw_operator_many_sites(axis):
    """Operators acting on three or more sites are drawn as a closed line."""
    assert draw_operator(SX_A * SX_B * SZ_C, axis) is axis
    assert len(axis.patches) == 0
    assert len(axis.lines) == 1
    assert len(axis.lines[0].get_xdata()) == 4  # three sites + closing point


def test_draw_operator_sum(axis):
    """Each term of a sum operator is drawn."""
    draw_operator(SX_A + SZ_B + SX_A * SX_B, axis)
    assert len(axis.patches) == 3


# ---------------------------------------------------------------------------
# convex_fit_derivative
# ---------------------------------------------------------------------------

X_DATA = np.linspace(0.0, 2.0, 30)


@pytest.mark.parametrize(
    ["convexity", "func", "deriv"],
    [
        (-1.0, lambda x: -(x**2), lambda x: -2 * x),
        (1.0, np.exp, np.exp),
    ],
    ids=["concave", "convex"],
)
def test_convex_fit_derivative_exact_data(convexity, func, deriv):
    """If the data satisfy the constraint, the fit recovers the derivative."""
    data = list(zip(X_DATA, func(X_DATA)))
    result = convex_fit_derivative(data, convexity=convexity)
    assert isinstance(result, list)
    assert len(result) == len(data)
    assert np.abs(np.array(result) - deriv(X_DATA)).max() < 5.0e-2


def test_convex_fit_derivative_default_is_concave():
    """By default, the fit is concave: the derivative decreases."""
    data = list(zip(X_DATA, -(X_DATA**2)))
    assert convex_fit_derivative(data) == convex_fit_derivative(data, convexity=-1.0)


@pytest.mark.parametrize("convexity", [-1.0, 1.0])
def test_convex_fit_derivative_enforces_convexity(convexity):
    """The result is monotonic with the sign set by ``convexity``.

    The data oscillate around a parabola, so they violate the constraint
    for both signs of the convexity.
    """
    data = list(zip(X_DATA, -(X_DATA**2) + 0.05 * np.sin(8 * X_DATA)))
    result = np.array(convex_fit_derivative(data, convexity=convexity))
    assert np.all(convexity * np.diff(result) > -1.0e-6)


def test_convex_fit_derivative_keeps_the_order_of_the_data():
    """The input does not need to be sorted, and the output follows its order."""
    rng = np.random.default_rng(1234)
    permutation = rng.permutation(len(X_DATA))
    x_shuffled = X_DATA[permutation]
    data = list(zip(x_shuffled, -(x_shuffled**2)))
    result = np.array(convex_fit_derivative(data))
    # Each derivative corresponds to its own x...
    assert np.abs(result + 2 * x_shuffled).max() < 5.0e-2
    # ...and the fit does not depend on the order of the data.
    sorted_result = np.array(convex_fit_derivative(list(zip(X_DATA, -(X_DATA**2)))))
    assert np.allclose(result, sorted_result[permutation], rtol=0.0, atol=1.0e-10)
