#!/usr/bin/env python
import argparse
import logging
import pickle
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pympler.asizeof import asizeof

from qalma.evolution import (
    Simulation,
    qutip_me_solve,
    series_evolution as series_solver,
)
from qalma.evolution.maxent_evol import (
    adaptive_projected_evolution,
    projected_evolution,
)
from qalma.model import build_system
from qalma.operators import ScalarOperator
from qalma.operators.states import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
)
from qalma.projections import n_body_projection
from qalma.scalarprod import fetch_covar_scalar_product
from qalma.scalarprod.basis import HierarchicalOperatorBasis

logging.basicConfig(level=logging.INFO)

UUID_CALL = f"{uuid.uuid4()}"


def update_basis_callback(state):
    """
    Function called each time the basis is updated.
    """
    print("  update basis:\n    now:", datetime.now())
    print("    t=", state["t"], "t_ref=", state["t_ref"], "t_last", state["last_t"])
    print(
        "    Delta t=",
        state["t"] - state["t_ref"],
        "x speed",
        state["max_error_speed"],
        "=",
        (state["t"] - state["t_ref"]) * state["max_error_speed"],
    )

    print("    phi_0 =", state["phi_0"])
    print("    basis time cost=", state["basis time cost"])
    print("    basis memory cost=", [asizeof(b) for b in state["basis"].operator_basis])
    print(
        "    basis nbody sector=",
        [b.n_body_sector() for b in state["basis"].operator_basis],
    )
    print(
        "    basis num terms=", [b.num_terms() for b in state["basis"].operator_basis]
    )
    print("    curr_n_body=", state["curr_n_body"])
    print("    len basis=", len(state["basis"].operator_basis))
    print("    away from mean field", state["away"])
    print("    cummulated error", state["cummulated error"])
    print("    basis.errors", state["basis"].errors)


np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)

# PARAMETERS
TIME_SPAN = np.linspace(0, 8, 500)


ALPHA = 0.61  #   jy=.9 jx
JX = 0.662743 / (1 - ALPHA) ** 0.5  # 1.75  -> vLR=1
JY = (1 - ALPHA) * JX
PHI_0 = [0, 0.25, 0.25, 1]


def build_system_objects(args):
    global L, W, SYSTEM, HAMILTONIAN, SZ_TOTAL, HALF_LEN_COMM, SITES, GLOBAL_IDENTITY, K0, TRACK_OBSERVABLES
    SYSTEM = build_system(
        geometry_name="open square lattice",
        model_name="spin",
        **{"L": L, "W": W, "Jz": 0, "Jxy": JX, "Alpha": ALPHA},
    )
    HAMILTONIAN = SYSTEM.global_operator("Hamiltonian").simplify()
    SZ_TOTAL = SYSTEM.global_operator("Sz")

    SITES = tuple(SYSTEM.sites.keys())
    # Other operators
    GLOBAL_IDENTITY = ScalarOperator(1.0, SYSTEM)

    SX_A = SYSTEM.site_operator(f"Sx@{SITES[0]}")
    SY_A = SYSTEM.site_operator(f"Sy@{SITES[0]}")
    SZ_A = SYSTEM.site_operator(f"Sz@{SITES[0]}")
    K0 = SX_A * PHI_0[1] + SY_A * PHI_0[2] + SZ_A * PHI_0[3]
    RHO_0 = GibbsProductDensityOperator(K0)
    K0 = -RHO_0.logm()

    track_list = args.track
    if track_list.lower() == "none":
        track_list = ""
    elif track_list.lower() == "all":
        track_list = "SZ_TOTAL,SZ_TOTAL_SQ,H,H_SQ"
    track_observables = []
    for obs_t in track_list.split(","):
        if obs_t.strip() == "SZ_TOTAL":
            track_observables.append(SZ_TOTAL)
        elif obs_t.strip() == "SZ_TOTAL":
            track_observables.append(SZ_TOTAL)
        elif obs_t.strip() == "SZ_TOTAL_SQ":
            track_observables.append(SZ_TOTAL**2)
        elif obs_t.strip() == "H":
            track_observables.append(HAMILTONIAN)
        elif obs_t.strip() == "H_SQ":
            track_observables.append(HAMILTONIAN * HAMILTONIAN)
    TRACK_OBSERVABLES = tuple(track_observables)


def run_series(axis):
    k_0 = K0 * BETA
    hamiltonian = HAMILTONIAN
    print("Start series:", datetime.now())
    series_sol = series_solver(hamiltonian, k_0, TIME_SPAN, 30)
    series = []
    for k in series_sol.states:
        try:
            series.append(GibbsDensityOperator(k).to_qutip_operator())
        except Exception:
            break

    print("Plot observables")
    series_expect = [np.real(rho.expect(SZ_TOTAL)) for rho in series]
    series_sol.expect_ops[0] = series_expect

    with open(f"series_L={L}_W={W}_beta={BETA}_{UUID_CALL}.pkl", "wb") as f:
        pickle.dump(series_sol, f)

    axis.plot(TIME_SPAN[: len(series)], series_expect, label="series")
    print("   done")


def run_projected(axis):
    k_0 = K0 * BETA
    hamiltonian = HAMILTONIAN
    print("Start exact:", datetime.now())
    exact_sol = qutip_me_solve(hamiltonian, k_0, TIME_SPAN)
    with open(f"exact_L={L}_beta={BETA}_{UUID_CALL}.pkl", "wb") as f:
        pickle.dump(exact_sol, f)

    exact_k = exact_sol.states
    exact = [GibbsDensityOperator(k).to_qutip_operator() for k in exact_k]

    sigma_0 = GibbsProductDensityOperator(K0)
    sp = fetch_covar_scalar_product(sigma_0)

    def projection_function(op_b):
        print("projection function", datetime.now())
        new_op = n_body_projection(op_b, nmax=MAX_M, sigma=sigma_0)
        if new_op is not op_b:
            if hasattr(new_op, "terms"):
                print("  projection has", len(new_op.terms))

        return new_op

    basis = (
        HierarchicalOperatorBasis(
            k_0,
            HAMILTONIAN,
            ELL,
            sp,
            n_body_projection=projection_function,
        )
        + TRACK_OBSERVABLES
    )
    print("projecting using basis", basis, datetime.now())

    projected = []
    for t, k in zip(TIME_SPAN, exact_k):
        print(f"projected K({t})")
        try:
            projected.append(
                GibbsDensityOperator(basis.project_onto(k)).to_qutip_operator()
            )
        except Exception:
            break

    print("Plot observables", datetime.now())
    exact_expect = [np.real(rho.expect(SZ_TOTAL)) for rho in exact]
    axis.set_ylim(min(-max(exact_expect), min(exact_expect)), max(exact_expect))
    axis.plot(TIME_SPAN, exact_expect, label="exact")
    projected_expect = [np.real(rho.expect(SZ_TOTAL)) for rho in projected]
    axis.plot(
        TIME_SPAN[: len(projected_expect)], projected_expect, ls="-.", label="projected"
    )

    parameters = exact_sol.parameters.copy()
    parameters["basis size"] = 10
    parameters["n_body_projection"] = MAX_M
    parameters["ell"] = 10
    parameters["beta"] = BETA
    with open(f"projected_exact_L={L}_beta={BETA}_{UUID_CALL}.pkl", "wb") as f:
        pickle.dump(
            Simulation(
                parameters=parameters,
                stats={"errors": basis.errors},
                time_span=exact_sol.time_span,
                expect_ops={0: projected_expect},
                states=[],
            ),
            f,
        )
    print("   done")


def run_simulation_adaptive(basis_depth, n_body, tolerance, axis):
    k_0 = K0 * BETA
    hamiltonian = HAMILTONIAN
    print(
        f"                   Start max ent ell={basis_depth},m={n_body},tol={tolerance}:",
        datetime.now(),
    )
    try:
        adaptative_sol = adaptive_projected_evolution(
            hamiltonian,
            k_0,
            TIME_SPAN,
            basis_depth,
            n_body,
            tol=tolerance,
            e_ops=[SZ_TOTAL],
            on_update_basis_callback=update_basis_callback,
            include_one_body_projection=True,
            extra_observables=TRACK_OBSERVABLES,
        )

        with open(
            f"adaptative3__L={L}_beta={BETA}_nbody={n_body}_deep={basis_depth}_{UUID_CALL}.pkl",
            "wb",
        ) as f:
            pickle.dump(adaptative_sol, f)

        # plt.scatter(ts[:len(max_ent)], [np.real(rho.expect(k_0)) for rho in max_ent], label=f"$\\ell={basis_depth}$, m={n_body}, tol={tolerance}")
        plt.scatter(
            TIME_SPAN[: len(adaptative_sol.time_span)],
            [np.real(ex_val) for ex_val in adaptative_sol.expect_ops[0]],
            label=f"c->$\\ell={basis_depth}$, m={n_body}, tol={tolerance}",
        )
        print("len:", len(adaptative_sol.time_span))
    except Exception as e:
        print("                   EXCEPTION ")
        print(type(e), e)
        raise


def run_simulation_projected(basis_depth, n_body, tolerance, axis):
    k_0 = K0 * BETA
    hamiltonian = HAMILTONIAN
    print(
        f"                   Start max ent ell={basis_depth},m={n_body},tol={tolerance}:",
        datetime.now(),
    )
    try:
        max_ent = [
            GibbsDensityOperator(k)
            for k in projected_evolution(hamiltonian, k_0, TIME_SPAN, 100, 100)
        ]
        # plt.scatter(ts[:len(max_ent)], [np.real(rho.expect(k_0)) for rho in max_ent], label=f"$\\ell={basis_depth}$, m={n_body}, tol={tolerance}")
        plt.scatter(
            TIME_SPAN[: len(max_ent)],
            [np.real(rho.expect(SZ_TOTAL)) for rho in max_ent],
            ls="-.",
            label=f"proyected-> $\\ell={basis_depth}$, m={n_body}, tol={tolerance}",
        )
        print("len:", len(max_ent))
    except Exception as e:
        print("                   EXCEPTION ")
        print(type(e), e)
        raise


def set_parameters():
    global BETA, FULL, L, W, MAX_M, ELL, TOL

    argparser = argparse.ArgumentParser(
        prog="run_dynamics",
        usage="%(prog)s [options] [FILE]",
        add_help=False,
        description="Simulate the dynamics of a spin chain with different approaches.",
        epilog="""Just with experimental porpouse.""",
    )
    argparser.add_argument(
        "--full", help="compute exact and projected dynamics too.", action="store_true"
    )
    argparser.add_argument(
        "--length", "-L", type=int, default=4, help="length of the spin chain"
    )
    argparser.add_argument(
        "--width", "-w", type=int, default=-1, help="width of the spin chain"
    )
    argparser.add_argument(
        "--beta", type=float, default=0.001, help="inverse temperature"
    )
    argparser.add_argument(
        "--n_body", "-M", type=int, default=4, help="max n_body sector"
    )
    argparser.add_argument(
        "--deep",
        "-D",
        type=int,
        default=2,
        help="deep of the Hiearchical basis/order the perturbative series.",
    )
    argparser.add_argument("--tol", type=float, default=0.001, help="tolerance")
    argparser.add_argument(
        "--track",
        type=str,
        default="None",
        help=(
            "track observables. One or a comma separated list with elements in"
            "`SZ_TOTAL`\n`SZ_TOTAL_SQ`\n `H`\n`H_SQ`. Special keywords are "
            "`None`: (default)->no track observables and \n`All`: Track all the observables."
        ),
    )
    argparser.add_argument(
        "--help", "-help", "-h", help="show this help message and exit", action="help"
    )

    args, ns = argparser.parse_known_args()
    print("ns", ns)
    print(type(args), args)
    L = args.length
    W = args.width if args.width > 0 else L
    TOL = args.tol
    ELL = args.deep
    BETA = args.beta
    MAX_M = args.n_body
    FULL = args.full
    build_system_objects(args)


def run_simulations():
    fig, axis = plt.subplots()
    if FULL:
        run_projected(axis)
        run_series(axis)
    run_simulation_adaptive(ELL, MAX_M, TOL, axis)
    axis.legend()
    # axis.set_title(f"Max-Ent evolution, beta={BETA} tolerance={tolerance}")
    fig.savefig(f"output_{UUID_CALL}.svg")


if __name__ == "__main__":
    set_parameters()
    run_simulations()
