"""Basic unit test."""

from test.helper import (
    FULL_TEST_CASES,
    check_operator_equality,
)

import pytest


@pytest.mark.parametrize(["name"], [(name,) for name in FULL_TEST_CASES])
def test_hermitician_part(name):
    """Test for the function that checks if the operator is equivalent to 0."""
    operator = FULL_TEST_CASES[name]
    if operator is None:
        return
    h_part = operator.hermitician_part()
    explicit = (operator + operator.dag()) * 0.5
    assert h_part.isherm, "hermitician part should be hermitician..."
    assert check_operator_equality(h_part, explicit, tolerance=1e-8)
    assert check_operator_equality(h_part.to_qutip(), explicit.to_qutip())
