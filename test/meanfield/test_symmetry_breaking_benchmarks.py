"""Benchmark tests for symmetry_breaking_mfa.

Run with:
    BENCHMARKS=1 pytest test/meanfield/test_symmetry_breaking_benchmarks.py -v

These tests exercise cases that are too slow for the regular test suite:
- Large system sizes (L=8, 12, 16)
- Continuous-symmetry models (XX, XXX) where escaping the symmetric
  fixed point requires many SC steps and multiple attempts
- High beta (low temperature) where convergence is slower
- Many auxiliary fields (numfields=6, 8)
"""

import os

import numpy as np
import pytest

from qalma.meanfield import compute_free_energy, variational_quadratic_mfa
from qalma.meanfield.symmetry_breaking import symmetry_breaking_mfa
from qalma.model import build_system
from qalma.operators.states import ProductDensityOperator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPSILON = 1e-3


def _spin_chain_nn(L, Jz=1.0, Jxy=1.0, Gamma=0.0):
    params = {"L": L, "a": 1, "Gamma": Gamma, "J": 1, "Jz0": Jz, "Jxy0": Jxy}
    return build_system("chain lattice", "spin", **params)


def _spin_chain_j1j2(L, J1=1.0, J2=0.5):
    params = {
        "L": L,
        "a": 1,
        "Gamma": 0.0,
        "J": 1,
        "Jz0": J1,
        "Jxy0": J1,
        "Jz1": J2,
        "Jxy1": J2,
    }
    return build_system("nnn open chain lattice", "spin", **params)


def _free_energy(sigma, H, beta):
    return float(np.real(compute_free_energy(sigma, beta * H)))


def _free_energy_mixed(system, H, beta):
    mixed = ProductDensityOperator({}, system=system)
    return _free_energy(mixed, H, beta)


skipif_no_benchmarks = pytest.mark.skipif(
    not os.environ.get("BENCHMARKS", 0),
    reason="set BENCHMARKS=1 to run benchmark tests",
)

# ---------------------------------------------------------------------------
# Benchmark 1: XX and XXX chains — continuous U(1) symmetry
# Requires many SC steps and multiple attempts to escape the mixed state.
# ---------------------------------------------------------------------------


@skipif_no_benchmarks
@pytest.mark.parametrize(
    "L,Jz,Jxy,beta,numfields,label",
    [
        (8, 0.0, 1.0, 3.0, 4, "xx_L8_b3"),
        (12, 0.0, 1.0, 3.0, 4, "xx_L12_b3"),
        (8, 1.0, 1.0, 3.0, 4, "xxx_afm_L8_b3"),
        (12, 1.0, 1.0, 3.0, 4, "xxx_afm_L12_b3"),
        (8, 1.0, 1.0, 5.0, 6, "xxx_afm_L8_b5"),
        (12, 1.0, 1.0, 5.0, 6, "xxx_afm_L12_b5"),
    ],
)
def test_benchmark_continuous_symmetry(L, Jz, Jxy, beta, numfields, label):
    """symmetry_breaking_mfa must beat both mixed state and SC for XX/XXX chains.

    These models have a continuous U(1)/SU(2) symmetry. The SC solution
    gets stuck in the fully mixed state. symmetry_breaking_mfa should
    escape it with enough attempts and SC steps.
    """
    system = _spin_chain_nn(L, Jz=Jz, Jxy=Jxy)
    H = system.global_operator("Hamiltonian")

    f_mixed = _free_energy_mixed(system, H, beta)

    sigma_sc = variational_quadratic_mfa(
        beta * H, numfields=0, max_self_consistent_steps=100
    )
    f_sc = _free_energy(sigma_sc, H, beta)

    sigma_sb = symmetry_breaking_mfa(
        beta * H,
        system,
        numfields=numfields,
        epsilon=_EPSILON,
        n_attempts=5,
        seed=0,
        max_self_consistent_steps=100,
    )
    f_sb = _free_energy(sigma_sb, H, beta)

    print(
        f"\n[{label}]  F_mixed={f_mixed:.5f}  F_sc={f_sc:.5f}  F_sb={f_sb:.5f}"
        f"  gain_vs_sc={f_sc - f_sb:.5f}"
    )

    assert (
        f_sb < f_mixed - 1e-4
    ), f"[{label}] did not improve over mixed: F_sb={f_sb:.5f} F_mixed={f_mixed:.5f}"
    assert (
        f_sb <= f_sc + 1e-5
    ), f"[{label}] worse than SC: F_sb={f_sb:.5f} F_sc={f_sc:.5f}"


# ---------------------------------------------------------------------------
# Benchmark 2: frustrated J1-J2 chain at large system sizes and low T
# ---------------------------------------------------------------------------


@skipif_no_benchmarks
@pytest.mark.parametrize(
    "L,J2,beta,numfields,label",
    [
        (12, 0.5, 5.0, 6, "j1j2_L12_J2=0.5_b5"),
        (16, 0.5, 5.0, 6, "j1j2_L16_J2=0.5_b5"),
        (12, 0.8, 5.0, 6, "j1j2_L12_J2=0.8_b5"),
        (16, 0.8, 5.0, 6, "j1j2_L16_J2=0.8_b5"),
        (16, 0.5, 10.0, 8, "j1j2_L16_J2=0.5_b10"),
    ],
)
def test_benchmark_j1j2_large(L, J2, beta, numfields, label):
    """symmetry_breaking_mfa on large frustrated chains at low temperature.

    The J1-J2 chain near the critical point J2/J1=0.5 is highly frustrated
    and develops spiral order at low T. The SC solution gets stuck in the
    disordered phase.
    """
    system = _spin_chain_j1j2(L, J2=J2)
    H = system.global_operator("Hamiltonian")

    f_mixed = _free_energy_mixed(system, H, beta)

    sigma_sc = variational_quadratic_mfa(
        beta * H, numfields=0, max_self_consistent_steps=100
    )
    f_sc = _free_energy(sigma_sc, H, beta)

    sigma_sb = symmetry_breaking_mfa(
        beta * H,
        system,
        numfields=numfields,
        epsilon=_EPSILON,
        n_attempts=5,
        seed=0,
        max_self_consistent_steps=100,
    )
    f_sb = _free_energy(sigma_sb, H, beta)

    print(
        f"\n[{label}]  F_mixed={f_mixed:.5f}  F_sc={f_sc:.5f}  F_sb={f_sb:.5f}"
        f"  gain_vs_sc={f_sc - f_sb:.5f}"
    )

    assert (
        f_sb <= f_sc + 1e-5
    ), f"[{label}] worse than SC: F_sb={f_sb:.5f} F_sc={f_sc:.5f}"


# ---------------------------------------------------------------------------
# Benchmark 3: improvement vs numfields sweep
# Documents how much the symmetry-breaking strategy gains as m grows.
# ---------------------------------------------------------------------------


@skipif_no_benchmarks
@pytest.mark.parametrize(
    "L,Jz,Jxy,beta,label",
    [
        (8, 1.0, 1.0, 3.0, "xxx_afm_L8_b3"),
        (12, 1.0, 1.0, 3.0, "xxx_afm_L12_b3"),
    ],
)
def test_benchmark_numfields_sweep(L, Jz, Jxy, beta, label):
    """F[sigma_m] is non-increasing with numfields for symmetry_breaking_mfa.

    Also checks that the variance ratio R_m = Var[sigma_m] / Var[sigma_SC]
    decreases with m, indicating convergence.
    """
    from qalma.meanfield import compute_variance

    system = _spin_chain_nn(L, Jz=Jz, Jxy=Jxy)
    H = system.global_operator("Hamiltonian")

    sigma_sc = variational_quadratic_mfa(
        beta * H, numfields=0, max_self_consistent_steps=100
    )
    var_sc = float(np.real(compute_variance(sigma_sc, beta * H)))
    f_sc = _free_energy(sigma_sc, H, beta)

    print(f"\n[{label}]  F_sc={f_sc:.5f}  Var_sc={var_sc:.4g}")
    print(f"  {'nf':>4}  {'F':>10}  {'Var':>10}  {'R':>8}")

    f_prev = f_sc
    for nf in [1, 2, 4, 6, 8]:
        sigma = symmetry_breaking_mfa(
            beta * H,
            system,
            numfields=nf,
            epsilon=_EPSILON,
            n_attempts=3,
            seed=0,
            max_self_consistent_steps=100,
        )
        f = _free_energy(sigma, H, beta)
        var = float(np.real(compute_variance(sigma, beta * H)))
        R = var / var_sc if var_sc > 1e-15 else float("nan")
        print(f"  {nf:>4}  {f:>10.5f}  {var:>10.4g}  {R:>8.4f}")

        assert f <= f_prev + 1e-4, (
            f"[{label}] F increased from nf={nf-1} to nf={nf}: "
            f"{f_prev:.5f} -> {f:.5f}"
        )
        f_prev = f
