"""Tests for the Max-Ent solvers against the exact evolution."""

from test.helper import HAMILTONIAN, SX_AB, SZ_TOTAL

import numpy as np
import qutip

from qalma.evolution import adaptive_maxent_evolution, series_evolution
from qalma.operators.functions import spectral_norm

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter={"float": lambda x: "%.3g" % x}
)


def compare_solutions(sol, sol_qutip, t_span, order, coeff_bound, tol=1.0e-8):
    """Compare two solutions."""
    diff_sols = [
        spectral_norm(k1.to_qutip() - k2_qutip) for k1, k2_qutip in zip(sol, sol_qutip)
    ]

    broken = {
        t: (
            error,
            coeff_bound * t**order + tol,
        )
        for t, error in zip(t_span, diff_sols)
        if error >= coeff_bound * t**order + tol
    }

    assert len(broken) == 0, (
        "Some of the errors in " f"{broken}" " are larger from the expected bound."
    )


def test_evolution():
    """Compare the series and single-basis solutions with the exact one."""
    t_span = np.linspace(0, 0.1, 10)
    k0 = SX_AB
    qutip_solution = qutip.mesolve(HAMILTONIAN.to_qutip(), k0.to_qutip(), t_span).states
    for order in range(1, 5):
        as_series_solution = series_evolution(HAMILTONIAN, k0, t_span, order).states
        compare_solutions(as_series_solution, qutip_solution, t_span, order, 0.5)
        single_basis_solution = adaptive_maxent_evolution(
            HAMILTONIAN,
            k0,
            t_span,
            order,
            n_body=order + 1,
            adaptive=False,
            memory_time=None,
        ).states
        compare_solutions(single_basis_solution, qutip_solution, t_span, order, 0.55)


def test_adaptive():
    """Adaptive evolution with memory closure, against the exact one."""
    t_span = np.linspace(0, 2, 10)
    k0 = SX_AB
    ham = 0.1 * HAMILTONIAN + SZ_TOTAL
    qutip_solution = qutip.mesolve(ham.to_qutip(), k0.to_qutip(), t_span).states
    for memory_time in ("auto", None):
        solution = adaptive_maxent_evolution(
            ham, k0, t_span, 4, 2, memory_time=memory_time
        ).states
        compare_solutions(
            solution, qutip_solution, t_span, order=0, coeff_bound=0.0, tol=2
        )
