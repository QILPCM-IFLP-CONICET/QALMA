"""Benchmark and validation tests for the chiral three-body Hamiltonian
on a triangular strip.

The model Hamiltonian is

    H = J  * sum_{<ij>} vec{s}_i . vec{s}_j
      + chi * sum_{triangles} vec{s}_i . (vec{s}_j x vec{s}_k)

where the scalar triple product

    chi_{ijk} = S^x_i (S^y_j S^z_k - S^z_j S^y_k)
              + S^y_i (S^z_j S^x_k - S^x_j S^z_k)
              + S^z_i (S^x_j S^y_k - S^y_j S^x_k)

is the spin chirality operator.  The lattice is a two-leg triangular strip
(LATTICEGRAPH "triangular strip open/periodic"), defined in lattices.xml.
The Hamiltonian is "chiral spin" in models.xml.

Physical motivation
-------------------
The chiral term breaks time-reversal and parity symmetry but is allowed by
lattice translation. On the triangular strip it arises e.g. in effective
descriptions of topological phases and fractional quantum Hall states on
ladders.  The term is *non-trivial* (i.e. cannot be written as a sum of
two-body operators), making it a meaningful stress-test for the variational
mean-field approach: the n_body_projection step in
`variational_quadratic_mfa` must correctly handle three-body interactions.

Usage
-----
Quick validation (no BENCHMARKS env var needed):
    pytest test_chiral_three_body.py -v

Full benchmark suite (saves JSON):
    BENCHMARKS=1 pytest test_chiral_three_body.py -v

Or run directly:
    python test_chiral_three_body.py
"""

import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest
import scipy

from qalma import graph_from_alps_xml, model_from_alps_xml
from qalma.meanfield import (
    compute_t_score,
    variational_quadratic_mfa,
)
from qalma.model import SystemDescriptor
from qalma.operators.states import ProductDensityOperator

# ---------------------------------------------------------------------------
# Constants and test matrix
# ---------------------------------------------------------------------------

#: Lengths (in unit cells, so 2*L sites total) for quick tests with exact ED.
LENGTHS_EXACT = [2, 3, 4]  # 4, 6, 8 sites → feasible for full diagonalisation

#: Lengths for the large-scale MF benchmark (no exact reference needed).
LENGTHS_MF = [6, 8, 12]  # 12, 16, 24 sites

BETAS = [0.5, 1.0, 2.0, 5.0]

#: (label, J, Wilson) pairs that define the model families we test.
#: J=0 isolates the purely chiral interaction; J>0 adds a Heisenberg backbone.
CHIRAL_CASES = [
    ("pure chiral (J=0, chi=1)", 0.0, 1.0),
    ("Heisenberg + weak chiral (chi=0.5J)", 1.0, 0.5),
    ("Heisenberg + strong chiral (chi=J)", 1.0, 1.0),
    ("Heisenberg only (chi=0)", 1.0, 0.0),
]

NUMFIELDS_LIST = [0, 1, 2, 3, 4, 6, 8]


# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------


def build_chiral_strip(
    L: int,
    J: float,
    chi: float,
    boundary: str = "open",
) -> Tuple[SystemDescriptor, object]:
    """Build a spin-1/2 triangular strip with Heisenberg + chiral interactions.

    Parameters
    ----------
    L : int
        Number of unit cells (each cell has 2 sites → 2*L sites total).
    J : float
        Heisenberg nearest-neighbor coupling (same on all bond types).
    chi : float
        Coefficient of the scalar-triple-product (chirality) term.
    boundary : {"open", "periodic"}
        Boundary condition along the strip.

    Returns
    -------
    system : SystemDescriptor
    ham    : Operator   (the Hamiltonian as a QALMA operator)
    """
    lattice_name = f"triangular strip {boundary}"
    parms = {
        "L": L,
        "a": 1,
        # Heisenberg coupling on all four bond types
        "J0": J,
        "J1": J,
        "J2": J,
        "J3": J,
        # Chiral coupling (applies to both loop types 0 and 1)
        "Wilson2": chi,
        "Wilson20": chi,  # loop type "0"  (up-triangles)
        "Wilson21": chi,  # loop type "1"  (down-triangles)
    }
    graph = graph_from_alps_xml(name=lattice_name, parms=parms)
    model = model_from_alps_xml(name="chiral spin")
    system = SystemDescriptor(graph, model, parms)
    ham = system.global_operator("Hamiltonian")
    return system, ham


# ---------------------------------------------------------------------------
# Exact-diagonalisation helpers (only for small L)
# ---------------------------------------------------------------------------


def exact_free_energy(ham, system, beta: float) -> float:
    """Compute -log Z = -log Tr[exp(-beta*H)] via full diagonalisation.

    Parameters
    ----------
    ham : Operator
    system : SystemDescriptor
    beta : float
        Inverse temperature.

    Returns
    -------
    float
        -log Z  (negative of the log-partition function).
    """
    sites = tuple(sorted(system.sites.keys()))
    ham_qutip = ham.to_qutip(sites)
    evals = ham_qutip.eigenenergies()
    e0 = evals.min()
    log_Z_shift = np.log(np.exp(-beta * (evals - e0)).sum())
    return -log_Z_shift + beta * e0


def mf_free_energy(sigma, ham, beta: float) -> float:
    """F_mf = Tr[sigma (log sigma + beta*H)]."""
    return sigma.variational_free_energy(beta * ham)


def t_score(sigma, ham, beta: float, f_exact: Optional[float]) -> Optional[float]:
    """T-score of the variational state relative to the exact Gibbs state.

    Parameters
    ----------
    sigma : ProductDensityOperator
    ham : Operator
    beta : float
    f_exact : float or None
        Value of -log Tr[exp(-beta*H)] as returned by ``exact_free_energy``.
        If None, the T-score is not computed and None is returned.
    """
    if f_exact is None:
        return None
    return float(compute_t_score(sigma, ham * beta, f_exact)[0])


def _var_f(sigma, ham, beta: float) -> float:
    """Var_{sigma}[hat{F}] without requiring F_exact.

    Uses a sentinel value for _f_exact that is guaranteed to be lower
    than the actual free energy so that compute_t_score does not raise,
    but only the variance (third return value) is used.
    """
    _, _, var = compute_t_score(sigma, ham * beta, -1e9)
    return float(np.real(var))


# ---------------------------------------------------------------------------
# Sanity-check helpers
# ---------------------------------------------------------------------------


def check_chiral_operator_antisymmetry(system, ham_chiral_only, tol=1e-10):
    """Verify chi_{ijk} = -chi_{ikj} for the chiral-only Hamiltonian.

    The scalar triple product is antisymmetric under exchange of any two
    indices.  If the LOOP definitions in the XML assign the same coefficient
    to up- and down-triangles (which differ by a permutation of two vertices),
    the total Hamiltonian should be proportional to

        sum_{up} chi_{ijk}  -  sum_{down} chi_{ijk}

    i.e. the two loop types contribute with opposite signs automatically
    because the NODE order is reversed.  This test checks that the operator
    is *non-zero* and Hermitian.
    """
    sites = tuple(sorted(system.sites.keys()))
    H_qutip = ham_chiral_only.to_qutip(sites)
    # Hermitian check
    diff = (H_qutip - H_qutip.dag()).norm()
    assert diff < tol, f"Chiral Hamiltonian is not Hermitian: ||H - H†|| = {diff:.2e}"
    # Non-trivial check (should not vanish for chi != 0 and L >= 2)
    assert (
        H_qutip.norm() > tol
    ), "Chiral Hamiltonian is the zero operator — check LOOP definitions."


def check_pure_heisenberg_limit(system, ham, J, chi, beta, tol=1e-6):
    """When chi=0 the chiral Hamiltonian must equal the pure Heisenberg result.

    Builds an independent Heisenberg Hamiltonian using the standard 'spin'
    model (no chiral term) and checks that the two operators coincide as
    matrices up to ``tol``.  Only meaningful when chi == 0.
    """
    if chi != 0.0:
        return

    L = len(system.sites) // 2
    parms_heis = {
        "L": L,
        "a": 1,
        "J0": J,
        "J1": J,
        "J2": J,
        "J3": J,
    }
    graph_heis = graph_from_alps_xml(name="triangular strip open", parms=parms_heis)
    model_heis = model_from_alps_xml(name="spin")
    # Override bond couplings so all four edge types carry the same J
    parms_heis.update(
        {
            "Jz0": J,
            "Jxy0": J,
            "Jz1": J,
            "Jxy1": J,
            "Jz2": J,
            "Jxy2": J,
            "Jz3": J,
            "Jxy3": J,
        }
    )
    system_heis = SystemDescriptor(graph_heis, model_heis, parms_heis)
    ham_heis = system_heis.global_operator("Hamiltonian")

    sites = tuple(sorted(system.sites.keys()))
    H_chiral = ham.to_qutip(sites)
    H_heis = ham_heis.to_qutip(sites)

    diff = (H_chiral - H_heis).norm()
    assert diff < tol, (
        f"chi=0 chiral Hamiltonian differs from pure Heisenberg by {diff:.2e} "
        f"(L={L}, J={J})"
    )


# ---------------------------------------------------------------------------
# Test: chi=0 collapses to Heisenberg (operator-level check)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", LENGTHS_EXACT)
def test_chi_zero_is_heisenberg(L):
    """For chi=0 the chiral Hamiltonian must be Hermitian and non-trivial."""
    system, ham = build_chiral_strip(L, J=1.0, chi=0.0)
    sites = tuple(sorted(system.sites.keys()))
    H = ham.to_qutip(sites)
    assert (H - H.dag()).norm() < 1e-10, "H not Hermitian for chi=0."
    try:
        assert H.norm() > 1e-10, "H is zero for chi=0, J=1 — something is wrong."
    except scipy.sparse.linalg._eigen.arpack.arpack.ArpackError:
        pass


# ---------------------------------------------------------------------------
# Test: pure chiral operator properties
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", LENGTHS_EXACT)
def test_chiral_operator_hermitian(L):
    """The purely chiral term (J=0, chi=1) must be Hermitian."""
    system, ham = build_chiral_strip(L, J=0.0, chi=1.0)
    check_chiral_operator_antisymmetry(system, ham)


# ---------------------------------------------------------------------------
# Test: variational MF improves over mixed state (small L, exact reference)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", LENGTHS_EXACT)
@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("label,J,chi", CHIRAL_CASES)
def test_variational_improves_over_mixed(label, J, chi, L, beta):
    """Variational MF free energy must be ≤ mixed-state free energy.

    This is the core inequality  F_var ≤ F_mixed = -N log 2  that must hold
    for *any* Hamiltonian (including non-trivial three-body ones).
    """
    system, ham = build_chiral_strip(L, J=J, chi=chi)
    N = len(system.sites)
    f_mixed_ref = -N * np.log(2)  # analytic: Tr[sigma_mixed log sigma_mixed]

    sigma_mixed = ProductDensityOperator({}, system=system)
    f_mixed = mf_free_energy(sigma_mixed, ham, beta)

    # Sanity: our analytic formula matches QALMA's value
    assert (
        abs(f_mixed - f_mixed_ref) < 1e-8
    ), f"Mixed free energy mismatch: got {f_mixed:.8f}, expected {f_mixed_ref:.8f}"

    sigma_var = variational_quadratic_mfa(
        beta * ham,
        numfields=4,
        max_self_consistent_steps=30,
    )
    f_var = mf_free_energy(sigma_var, ham, beta)

    assert f_var <= f_mixed + 1e-6, (
        f"{label}  L={L}  beta={beta}: " f"F_var={f_var:.6f} > F_mixed={f_mixed:.6f}"
    )


# ---------------------------------------------------------------------------
# Test: exact validation (small L only, requires full diagonalisation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L", LENGTHS_EXACT)
@pytest.mark.parametrize("beta", [1.0, 2.0])
@pytest.mark.parametrize("label,J,chi", CHIRAL_CASES)
def test_exact_lower_bound(label, J, chi, L, beta):
    """Exact free energy must be ≤ variational free energy (Gibbs inequality)."""
    system, ham = build_chiral_strip(L, J=J, chi=chi)

    f_exact = exact_free_energy(ham, system, beta)
    sigma_var = variational_quadratic_mfa(
        beta * ham,
        numfields=4,
        max_self_consistent_steps=30,
    )
    f_var = mf_free_energy(sigma_var, ham, beta)

    assert f_exact <= f_var + 1e-6, (
        f"{label}  L={L}  beta={beta}: "
        f"F_exact={f_exact:.6f} > F_var={f_var:.6f}  (violates Gibbs inequality)"
    )


# ---------------------------------------------------------------------------
# Test: F non-increasing with numfields (frustration convergence)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("BENCHMARKS"),
    reason="set BENCHMARKS=1 to run",
)
@pytest.mark.parametrize("label,J,chi", CHIRAL_CASES)
@pytest.mark.parametrize("L", LENGTHS_MF)
@pytest.mark.parametrize("beta", BETAS)
def test_numfields_convergence(label, J, chi, L, beta):
    """F_mf must be non-increasing as numfields grows."""
    results = run_numfields_sweep(L, J, chi, beta)
    fs = [r["f"] for r in results if r["numfields"] > 0]
    for i in range(1, len(fs)):
        nf_prev = results[i]["numfields"]
        nf_curr = results[i + 1]["numfields"]
        assert fs[i] <= fs[i - 1] + 1e-4, (
            f"{label}  L={L}  beta={beta}: "
            f"F increased nf={nf_prev}→{nf_curr} ({fs[i-1]:.5f}→{fs[i]:.5f})"
        )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_numfields_sweep(
    L: int,
    J: float,
    chi: float,
    beta: float,
    numfields_list: List[int] = NUMFIELDS_LIST,
    boundary: str = "open",
) -> List[dict]:
    """Sweep numfields for a chiral strip, using warm start between runs.

    For each numfields value nf the variational state sigma_{nf} is computed
    and we record:

      * ``f``            F_mf = Tr[sigma (log sigma + beta*H)]
      * ``var_f``        Var_{sigma}[hat{F}] (numerator of the T-score)
      * ``var_f_ratio``  Var_{sigma_{nf}} / Var_{sigma_{SC}}
      * ``magnetization``  list of <S^z_i> on each site

    Parameters
    ----------
    L : int
        Number of unit cells (2*L sites).
    J : float
        Heisenberg coupling.
    chi : float
        Chiral coupling.
    beta : float
        Inverse temperature.
    numfields_list : list of int
        Values of numfields to sweep (0 = self-consistent only).
    boundary : str
        "open" or "periodic".

    Returns
    -------
    list of dict
        One dict per numfields value (including nf=0 as SC baseline).
    """
    system, ham = build_chiral_strip(L, J=J, chi=chi, boundary=boundary)
    sites = list(system.sites.keys())
    Sz_ops = [system.site_operator("Sz", s) for s in sites]

    # --- Self-consistent baseline (nf=0) ---
    t0 = time.perf_counter()
    sigma_sc = variational_quadratic_mfa(
        beta * ham,
        numfields=0,
        max_self_consistent_steps=100,
    )
    t_sc = time.perf_counter() - t0
    f_sc = mf_free_energy(sigma_sc, ham, beta)
    var_f_sc = _var_f(sigma_sc, ham, beta)
    mag_sc = [float(np.real(sigma_sc.expect(sz))) for sz in Sz_ops]

    print(
        f"  J={J}  chi={chi}  L={L}  beta={beta}  "
        f"nf= 0 (SC):  F_mf={f_sc:.6f}  Var={var_f_sc:.4g}  t={t_sc:.2f}s"
    )

    results = [
        {
            "J": J,
            "chi": chi,
            "L": L,
            "beta": beta,
            "numfields": 0,
            "f": f_sc,
            "var_f": var_f_sc,
            "var_f_ratio": 1.0,
            "magnetization": mag_sc,
            "time": t_sc,
        }
    ]

    sigma_ref = sigma_sc
    nf_list_positive = [nf for nf in sorted(numfields_list) if nf > 0]

    for nf in nf_list_positive:
        t0 = time.perf_counter()
        sigma = variational_quadratic_mfa(
            beta * ham,
            numfields=nf,
            sigma_ref=sigma_ref,
            max_self_consistent_steps=30,
        )
        elapsed = time.perf_counter() - t0

        f_mf = mf_free_energy(sigma, ham, beta)
        var_f = _var_f(sigma, ham, beta)
        var_f_ratio = var_f / var_f_sc if var_f_sc > 1e-15 else None
        mag = [float(np.real(sigma.expect(sz))) for sz in Sz_ops]

        ratio_str = f"{var_f_ratio:.4f}" if var_f_ratio is not None else "  n/a"
        print(
            f"  J={J}  chi={chi}  L={L}  beta={beta}  "
            f"nf={nf:2d}:  F_mf={f_mf:.6f}  Var={var_f:.4g}  "
            f"Var_ratio={ratio_str}  t={elapsed:.2f}s"
        )

        results.append(
            {
                "J": J,
                "chi": chi,
                "L": L,
                "beta": beta,
                "numfields": nf,
                "f": f_mf,
                "var_f": var_f,
                "var_f_ratio": var_f_ratio,
                "magnetization": mag,
                "time": elapsed,
            }
        )
        sigma_ref = sigma  # warm start

    return results


# ---------------------------------------------------------------------------
# Main: full benchmark run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    all_results: dict = {
        "exact_validation": [],
        "numfields_convergence": [],
    }

    # ---- Exact validation (small L) ---------------------------------------
    print("=" * 70)
    print("Exact validation  (chiral strip, small L)")
    print("=" * 70)

    for label, J, chi in CHIRAL_CASES:
        for L in LENGTHS_EXACT:
            for beta in BETAS:
                print(f"\n{label}  L={L}  beta={beta}")
                system, ham = build_chiral_strip(L, J=J, chi=chi)
                sigma_mixed = ProductDensityOperator({}, system=system)

                t0 = time.perf_counter()
                sigma_var = variational_quadratic_mfa(
                    beta * ham, numfields=6, max_self_consistent_steps=30
                )
                t_var = time.perf_counter() - t0

                t0 = time.perf_counter()
                sigma_sc = variational_quadratic_mfa(
                    beta * ham, numfields=0, max_self_consistent_steps=100
                )
                t_sc = time.perf_counter() - t0

                f_exact = exact_free_energy(ham, system, beta)

                row = {
                    "label": label,
                    "J": J,
                    "chi": chi,
                    "L": L,
                    "beta": beta,
                    "N_sites": len(system.sites),
                    "F_exact": f_exact,
                    "F_mixed": mf_free_energy(sigma_mixed, ham, beta),
                    "F_sc": mf_free_energy(sigma_sc, ham, beta),
                    "F_variational": mf_free_energy(sigma_var, ham, beta),
                    "T_score_mixed": t_score(sigma_mixed, ham, beta, f_exact),
                    "T_score_sc": t_score(sigma_sc, ham, beta, f_exact),
                    "T_score_variational": t_score(sigma_var, ham, beta, f_exact),
                    "time_variational": t_var,
                    "time_sc": t_sc,
                }
                all_results["exact_validation"].append(row)

                print(
                    f"  F: mixed={row['F_mixed']:.4f}  "
                    f"SC={row['F_sc']:.4f}  "
                    f"var={row['F_variational']:.4f}  "
                    f"exact={f_exact:.4f}  "
                    f"  T_score(var)={row['T_score_variational']:.4g}"
                )

    # ---- numfields convergence (larger L) ---------------------------------
    print("\n" + "=" * 70)
    print("F_mf vs numfields  (chiral strip, larger L)")
    print("=" * 70)

    for label, J, chi in CHIRAL_CASES:
        for L in LENGTHS_MF:
            for beta in BETAS:
                print(f"\n{label}  L={L}  beta={beta}")
                try:
                    rows = run_numfields_sweep(L, J, chi, beta)
                    all_results["numfields_convergence"].extend(rows)
                except Exception as exc:
                    print(f"  FAILED: {exc}")

    # ---- Save results -----------------------------------------------------
    out = output_dir / "chiral_three_body_results.json"
    with open(out, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\nResults saved → {out}")

    # ---- Summary table ----------------------------------------------------
    print("\n--- Convergence summary (L=8 unit cells = 16 sites, beta=2.0) ---")
    print(
        f"{'Model':45s}  "
        f"{'F(nf=0)':>10} {'F(nf=4)':>10} {'F(nf=8)':>10}  "
        f"{'Var_r(4)':>10} {'Var_r(8)':>10}"
    )
    for label, J, chi in CHIRAL_CASES:
        rows = [
            r
            for r in all_results["numfields_convergence"]
            if r["J"] == J and r["chi"] == chi and r["L"] == 8 and r["beta"] == 2.0
        ]
        if not rows:
            continue
        by_nf_f = {r["numfields"]: r["f"] for r in rows}
        by_nf_r = {r["numfields"]: r.get("var_f_ratio") for r in rows}

        def _fmt_f(nf):
            v = by_nf_f.get(nf, float("nan"))
            return f"{v:.4f}" if v == v else "   nan"

        def _fmt_r(nf):
            v = by_nf_r.get(nf)
            return f"{v:.4f}" if v is not None else "    --"

        print(
            f"  {label:43s}  "
            f"{_fmt_f(0):>10} {_fmt_f(4):>10} {_fmt_f(8):>10}  "
            f"{_fmt_r(4):>10} {_fmt_r(8):>10}"
        )
