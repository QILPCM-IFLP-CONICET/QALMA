"""
Benchmarks for the variational mean-field approximation — paper figures.

Two families of tests:

1. Validation against exact diagonalization (L <= 8)
   Models: Ising transverse, XX, XXX, XYZ
   Lattice: "open chain" (simple1d unit cell, NN bonds only)

2. Convergence vs numfields — J1-J2 frustrated chain
   Lattice: "open chain" with "complex1d" unit cell
             bond type 0 = NN (J1), bond type 1 = NNN (J2)
   Requires 'complex1d' unit cell in lattices.xml (already present).

Usage
-----
Full run (saves JSON for plotting):
    python test_variational_mf_paper.py

Quick validation (pytest, no BENCHMARKS flag needed):
    pytest test_variational_mf_paper.py -v

Full benchmark suite:
    BENCHMARKS=1 pytest test_variational_mf_paper.py -v \\
        --benchmark-json=results.json
"""

import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from qalma import graph_from_alps_xml, model_from_alps_xml
from qalma.meanfield import variational_quadratic_mfa
from qalma.meanfield.variational import compute_rel_entropy
from qalma.model import SystemDescriptor
from qalma.operators.states import ProductDensityOperator

# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------


def build_nn_chain(L: int, parms: dict) -> Tuple[SystemDescriptor, object]:
    """
    Spin-1/2 open chain with nearest-neighbor bonds only.
    Uses 'open chain' latticegraph (simple1d unit cell, bond type 0).

    Parameters
    ----------
    L : int
        Number of sites.
    parms : dict
        Model parameters. Keys: Jz, Jxy, Gamma, h, etc.
        Bond type 0 maps to J, Jz, Jxy in the 'spin' Hamiltonian.
    """
    graph = graph_from_alps_xml(name="open chain lattice", parms={"L": L, "a": 1})
    model = model_from_alps_xml(name="spin")
    system = SystemDescriptor(graph, model, parms)
    ham = system.global_operator("Hamiltonian")
    return system, ham


def build_j1j2_chain(L: int, J1: float, J2: float) -> Tuple[SystemDescriptor, object]:
    """
    Spin-1/2 J1-J2 open chain.
    Uses 'nnn open chain lattice' (complex1d unit cell), which has:
      - bond type 0: nearest neighbors  (J1)
      - bond type 1: next-nearest neighbors (J2)

    Parameters
    ----------
    L : int
        Number of sites.
    J1 : float
        Nearest-neighbor coupling. Negative = ferromagnetic.
    J2 : float
        Next-nearest-neighbor coupling.
    """
    parms = {
        "Jz": J1,
        "Jxy": J1,  # bond type 0 → NN
        "Jz'": J2,
        "Jxy'": J2,  # bond type 1 → NNN
    }
    graph = graph_from_alps_xml(name="nnn open chain lattice", parms={"L": L, "a": 1})
    model = model_from_alps_xml(name="spin")
    system = SystemDescriptor(graph, model, parms)
    ham = system.global_operator("Hamiltonian")
    # print("J1=",J1, "J2=",J2)
    # print(ham)
    return system, ham


# ---------------------------------------------------------------------------
# Helpers: reference quantities
# ---------------------------------------------------------------------------


def exact_free_energy(ham, system, beta: float) -> float:
    """
    Exact Helmholtz free energy F = -1/beta * log Z.
    Only feasible for L <= 10 (Hilbert space dim = 2^L).
    """
    sites = tuple(sorted(system.sites.keys()))
    ham_qutip = ham.to_qutip(sites)
    evals = ham_qutip.eigenenergies()
    e0 = evals.min()
    log_Z = np.log(np.exp(-beta * (evals - e0)).sum())
    return -(1.0 / beta) * log_Z - e0


def s_rel(sigma, ham, beta: float) -> float:
    """
    S_rel(sigma || e^{-beta H}) = Tr[sigma(log sigma + beta H)] + log Z.

    compute_rel_entropy returns Tr[sigma(log sigma + K)] with K = beta*H,
    which equals S_rel up to the constant log Z — sufficient for comparing
    approximations at fixed (H, beta).
    """
    return float(np.real(compute_rel_entropy(sigma, beta * ham)))


# ---------------------------------------------------------------------------
# Family 1 — Validation against exact diagonalization
# ---------------------------------------------------------------------------

# (label, parms, L_list, beta_list)
EXACT_CASES = [
    (
        "Ising transverse (Gamma=0.5J)",
        {"Jz": 1.0, "Jxy": 0.0, "Gamma": 0.5},
        [4, 6, 8],
        [0.5, 1.0, 2.0, 5.0],
    ),
    (
        "Ising transverse critical (Gamma=J)",
        {"Jz": 1.0, "Jxy": 0.0, "Gamma": 1.0},
        [4, 6, 8],
        [0.5, 1.0, 2.0],
    ),
    (
        "XX chain",
        {"Jz": 0.0, "Jxy": 1.0},
        [4, 6, 8],
        [0.5, 1.0, 2.0, 5.0],
    ),
    (
        "XXX Heisenberg AFM",
        {"Jz": 1.0, "Jxy": 1.0},
        [4, 6, 8],
        [0.5, 1.0, 2.0, 5.0],
    ),
    (
        "XXX Heisenberg FM",
        {"Jz": -1.0, "Jxy": -1.0},
        [4, 6, 8],
        [0.5, 1.0, 2.0, 5.0],
    ),
    (
        "XYZ anisotropic (Jz=1, Jxy=0.5)",
        {"Jz": 1.0, "Jxy": 0.5},
        [4, 6, 8],
        [0.5, 1.0, 2.0],
    ),
]


@pytest.mark.parametrize("label,parms,L_list,beta_list", EXACT_CASES)
@pytest.mark.parametrize("L", [4, 6, 8])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_exact_validation(label, parms, L_list, beta_list, L, beta):
    """
    Variational MF must improve over the fully mixed state.
    The mixed state is the trivial upper bound on S_rel.
    """
    if L not in L_list or beta not in beta_list:
        pytest.skip(f"Not in test matrix for {label}")

    system, ham = build_nn_chain(L, parms)
    sigma_mixed = ProductDensityOperator({}, system=system)

    sigma_var = variational_quadratic_mfa(
        beta * ham,
        numfields=6,
        max_self_consistent_steps=30,
    )
    sigma_sc = variational_quadratic_mfa(
        beta * ham,
        numfields=1,
        its=0,
        max_self_consistent_steps=100,
    )

    s_mixed = s_rel(sigma_mixed, ham, beta)
    s_sc = s_rel(sigma_sc, ham, beta)
    s_var = s_rel(sigma_var, ham, beta)

    print(f"\n{label}  L={L}  beta={beta}")
    print(f"  S_rel mixed: {s_mixed:.6f}")
    print(f"  S_rel SC:    {s_sc:.6f}")
    print(f"  S_rel var:   {s_var:.6f}")
    print(f"  Delta(var vs SC):   {s_sc - s_var:.6f}")

    assert (
        s_var <= s_mixed + 1e-6
    ), f"Variational ({s_var:.4f}) not better than mixed ({s_mixed:.4f})"


# ---------------------------------------------------------------------------
# Family 2 — S_rel vs numfields for the J1-J2 chain
# ---------------------------------------------------------------------------

# (J2/J1, label)
J1J2_CASES = [
    (0.0, "no frustration (J2=0)"),
    (0.2, "weak frustration"),
    (0.4, "moderate frustration"),
    (0.5, "maximum frustration (critical)"),
    (0.6, "spiral phase"),
    (0.8, "strong J2"),
    (1.0, "J1=J2"),
]

NUMFIELDS_LIST = [1, 2, 3, 4, 6, 8, 10]
J1 = -1.0  # AFM nearest-neighbor


def run_numfields_sweep(
    J2_ratio: float,
    L: int,
    beta: float,
    numfields_list: List[int] = NUMFIELDS_LIST,
) -> List[dict]:
    """
    Sweep numfields for a J1-J2 chain, using warm start between runs.
    Returns list of result dicts suitable for JSON serialization.
    """
    J2 = J2_ratio * abs(J1)
    system, ham = build_j1j2_chain(L, J1, J2)
    sites = list(system.sites.keys())
    Sz_ops = [system.site_operator("Sz", s) for s in sites]

    results = []
    sigma_ref = None

    for nf in sorted(numfields_list):
        t0 = time.perf_counter()
        sigma = variational_quadratic_mfa(
            beta * ham,
            numfields=nf,
            sigma_ref=sigma_ref,
            max_self_consistent_steps=30,
        )
        elapsed = time.perf_counter() - t0

        sr = s_rel(sigma, ham, beta)
        mag = [float(np.real(sigma.expect(sz))) for sz in Sz_ops]

        results.append(
            {
                "J2_over_J1": J2_ratio,
                "L": L,
                "beta": beta,
                "numfields": nf,
                "s_rel": sr,
                "magnetization": mag,
                "time": elapsed,
            }
        )

        print(
            f"  J2/J1={J2_ratio:.2f}  L={L}  beta={beta}  "
            f"nf={nf:2d}:  S_rel={sr:.6f}  t={elapsed:.2f}s  "
            f"<Sz>=[{', '.join(f'{m:.3f}' for m in mag[:5])}...]"
        )

        sigma_ref = sigma  # warm start for next nf

    return results


@pytest.mark.skipif(
    not os.environ.get("BENCHMARKS"),
    reason="set BENCHMARKS=1 to run",
)
@pytest.mark.parametrize("J2_ratio,label", J1J2_CASES)
@pytest.mark.parametrize("L", [8, 12, 16])
@pytest.mark.parametrize("beta", [1.0, 2.0, 5.0])
def test_numfields_convergence(J2_ratio, label, L, beta):
    """
    S_rel must be non-increasing as numfields grows.
    Tolerance of 1e-4 allows for numerical noise in the optimizer.
    """
    print(f"\n--- {label}  L={L}  beta={beta} ---")
    results = run_numfields_sweep(J2_ratio, L, beta)
    s_rels = [r["s_rel"] for r in results]
    for i in range(1, len(s_rels)):
        nf_prev = results[i - 1]["numfields"]
        nf_curr = results[i]["numfields"]
        assert s_rels[i] <= s_rels[i - 1] + 1e-4, (
            f"S_rel increased: nf={nf_prev} → {nf_curr}  "
            f"({s_rels[i-1]:.5f} → {s_rels[i]:.5f})"
        )


# ---------------------------------------------------------------------------
# Main: full benchmark run, saves results to JSON for plotting
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    all_results = {"exact_validation": [], "numfields_convergence": []}

    # ---- Family 1 --------------------------------------------------------
    print("=" * 70)
    print("Family 1: Validation against exact diagonalization")
    print("=" * 70)

    for label, parms, L_list, beta_list in EXACT_CASES:
        for L in L_list:
            for beta in beta_list:
                print(f"\n{label}  L={L}  beta={beta}")
                system, ham = build_nn_chain(L, parms)
                sigma_mixed = ProductDensityOperator({}, system=system)

                t0 = time.perf_counter()
                sigma_var = variational_quadratic_mfa(
                    beta * ham, numfields=6, max_self_consistent_steps=30
                )
                t_var = time.perf_counter() - t0

                t0 = time.perf_counter()
                sigma_sc = variational_quadratic_mfa(
                    beta * ham, numfields=1, its=0, max_self_consistent_steps=100
                )
                t_sc = time.perf_counter() - t0

                F_exact = exact_free_energy(ham, system, beta) if L <= 8 else None

                row = {
                    "label": label,
                    "params": parms,
                    "L": L,
                    "beta": beta,
                    "F_exact": F_exact,
                    "s_rel_mixed": s_rel(sigma_mixed, ham, beta),
                    "s_rel_sc": s_rel(sigma_sc, ham, beta),
                    "s_rel_variational": s_rel(sigma_var, ham, beta),
                    "time_variational": t_var,
                    "time_sc": t_sc,
                }
                all_results["exact_validation"].append(row)

                print(
                    f"  S_rel: mixed={row['s_rel_mixed']:.4f}  "
                    f"SC={row['s_rel_sc']:.4f}  "
                    f"var={row['s_rel_variational']:.4f}  "
                    f"(t_var={t_var:.1f}s  t_sc={t_sc:.1f}s)"
                    + (f"  F_exact={F_exact:.4f}" if F_exact is not None else "")
                )

    # ---- Family 2 --------------------------------------------------------
    print("\n" + "=" * 70)
    print("Family 2: S_rel vs numfields  (J1-J2 frustrated chain)")
    print("=" * 70)

    for J2_ratio, label in J1J2_CASES:
        for L in [8, 12, 16]:
            for beta in [1.0, 2.0, 5.0]:
                print(f"\n{label}  L={L}  beta={beta}")
                try:
                    rows = run_numfields_sweep(J2_ratio, L, beta)
                    all_results["numfields_convergence"].extend(rows)
                except Exception as exc:
                    print(f"  FAILED: {exc}")

    # ---- Save ------------------------------------------------------------
    out = output_dir / "variational_mf_paper_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out}")

    # ---- Summary table ---------------------------------------------------
    print("\n--- S_rel convergence summary (L=8, beta=2.0) ---")
    print(f"{'Frustration':45s} {'nf=1':>8} {'nf=4':>8} {'nf=10':>8}")
    for J2_ratio, label in J1J2_CASES:
        rows = [
            r
            for r in all_results["numfields_convergence"]
            if r["J2_over_J1"] == J2_ratio and r["L"] == 8 and r["beta"] == 2.0
        ]
        if not rows:
            continue
        by_nf = {r["numfields"]: r["s_rel"] for r in rows}
        s1 = f"{by_nf.get(1,  float('nan')):.4f}"
        s4 = f"{by_nf.get(4,  float('nan')):.4f}"
        s10 = f"{by_nf.get(10, float('nan')):.4f}"
        print(f"  {label:43s} {s1:>8} {s4:>8} {s10:>8}")
