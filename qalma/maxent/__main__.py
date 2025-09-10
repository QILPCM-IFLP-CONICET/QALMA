"""
Run a simulation
"""

# custom library including basic linear algebra functions

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

# import qutip
from scipy import linalg
from scipy.optimize import fsolve

import qalma.maxent.restricted_maxent_toolkit as me
from qalma.alpsmodels import model_from_alps_xml
from qalma.geometry import graph_from_alps_xml
from qalma.model import SystemDescriptor
from qalma.operators.states import GibbsDensityOperator, GibbsProductDensityOperator
from qalma.operators.states.meanfield import (
    one_body_from_qutip_operator,
    project_to_n_body_operator,
)

# function used to safely and robustly map K-states to states
# from qalma.proj_evol import safe_exp_and_normalize

# long term ev

# Configuration du style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,  # Taille de police
        "axes.labelsize": 14,  # Taille des labels des axes
        "axes.titlesize": 16,  # Taille des titres
        "legend.fontsize": 12,  # Taille des légendes
        "xtick.labelsize": 12,  # Taille des labels des ticks sur l'axe X
        "ytick.labelsize": 12,  # Taille des labels des ticks sur l'axe Y
        "font.family": "serif",  # Police de type "serif" pour un
        # rendu professionnel
        "axes.linewidth": 1.5,  # Largeur des bordures des axes
        "grid.alpha": 0.5,  # Transparence des grilles
    }
)

# CONSTANTS


# functions used for Mean-Field projections


def build_system(params):
    """
    Build the system and observables from the parms dict
    """
    system = SystemDescriptor(
        model=model_from_alps_xml(params["models_lib_file"], "spin"),
        graph=graph_from_alps_xml(
            params["lattice_lib_file"],
            "open chain lattice",
            parms={"L": params["size"], "a": 1},
        ),
        parms={"h": 0, "J": params["Jx"]},
    )

    sites = list(system.sites)
    sx_ops = [
        system.site_operator("Sx", "1[" + str(a) + "]")
        for a in range(len(system.sites))
    ]
    sy_ops = [
        system.site_operator("Sy", "1[" + str(a) + "]")
        for a in range(len(system.sites))
    ]
    sz_ops = [
        system.site_operator("Sz", "1[" + str(a) + "]")
        for a in range(len(system.sites))
    ]

    hamiltonian = (
        params["Jx"] * sum(sx_ops[i] * sx_ops[i + 1] for i in range(params["size"] - 1))
        + params["Jy"]
        * sum(sy_ops[i] * sy_ops[i + 1] for i in range(params["size"] - 1))
        + params["Jz"]
        * sum(sz_ops[i] * sz_ops[i + 1] for i in range(params["size"] - 1))
    )
    idop = system.site_operator("identity", sites[0])
    return {
        "system": system,
        "sites": sites,
        "idop": idop,
        "H": hamiltonian,
        "sx_ops": sx_ops,
        "sy_ops": sy_ops,
        "sz_ops": sz_ops,
    }


def run_restricted_simulation(params, system_data, k_0, sigma_0):
    """
    Run the simulation
    """
    print("run restricted simulation")
    hamiltonian = system_data["H"]
    obs_sz = sum(sz for sz in system_data["sz_ops"])
    sigma_act = sigma_0
    print("obs_sz is", type(obs_sz))
    print("k_0=", k_0)

    current_simulation = {
        "params": params,
        "saved_cut_times_index_ell": [0],
        "no_acts_ell": [0],
        "local_bound_error_ell": [],
        "spectral_norm_Hij_tensor_ell": [],
        "instantaneous_w_errors": [],
    }

    chosen_depth = params["chosen_depth"]
    eps_tol = params["eps"]
    max_bodies = params["max_bodies"]

    # Initialize variables to track errors, saved cut times,
    # expectation values, and commutators

    saved_cut_times_index_ell = current_simulation["saved_cut_times_index_ell"]
    no_acts_ell = current_simulation["no_acts_ell"]
    number_of_commutators_ell = [chosen_depth]
    current_simulation["number_of_commutators_ell"] = number_of_commutators_ell

    # to be used in storing the values of the partial sum at each time
    local_bound_error_ell = current_simulation["local_bound_error_ell"]
    # to be used in storing the spectral norm of the Hij tensor
    # at each actualization of the (orthonormalized) basis
    spectral_norm_hij_tensor_ell = current_simulation["spectral_norm_Hij_tensor_ell"]
    # Norm of the orthogonal component of the commutators
    instantaneous_w_errors = current_simulation["instantaneous_w_errors"]

    # Start the computation

    ev_obs_maxent_act_partial_sum_ell = [sigma_0.expect(obs_sz)]
    current_simulation["ev_obs_maxent_act_partialSum_ell"] = (
        ev_obs_maxent_act_partial_sum_ell
    )

    # Compute the scalar product operator used for orthogonalization
    sp_local = me.fetch_covar_scalar_product(sigma=sigma_0)
    local_t_value = 0.0

    # Build the initial Krylov basis and orthogonalize it
    print("hamiltonian", type(hamiltonian), "k_0", type(k_0))
    print("build the hierarchical basis:")
    hbb_ell_act = me.build_Hierarch(
        generator=hamiltonian, seed_op=k_0, deep=chosen_depth
    )
    for i, b in enumerate(hbb_ell_act):
        print(f"b_{i} =", type(b))

    print("orthogonalize the basis")
    orth_basis_act = me.orthogonalize_basis(basis=hbb_ell_act, sp=sp_local)
    for i, b in enumerate(orth_basis_act):
        print(f"b_{i} =", type(b))

    # Compute the Hamiltonian tensor for the basis
    print("hamiltonian tensor")
    hij_tensor_act, w_errors = me.fn_Hij_tensor_with_errors(
        generator=hamiltonian, basis=orth_basis_act, sp=sp_local
    )
    instantaneous_w_errors.append(w_errors)

    print("spectral norm")
    spectral_norm_hij_tensor_ell.append(linalg.norm(hij_tensor_act))

    print("project k_0")
    # Initial condition
    phi0_proj_act = me.project_op(k_0, orth_basis_act, sp_local)

    # Initialize lists to store time-evolved values
    # phi_at_timet = [phi0_proj_act]
    # K_at_timet = [K0.to_qutip()]
    # sigma_at_timet = [me.safe_expm_and_normalize(K_at_timet[0])]

    # Iterate through the time steps
    timespan = params["timespan"]
    for curr_t in timespan[1:]:
        print("curr_t", curr_t)
        # Evolve the state phi(t) for a small time window
        print("hij_tensor_act\n", hij_tensor_act)
        phi_proj = np.real(
            linalg.expm(hij_tensor_act * (curr_t - local_t_value)) @ phi0_proj_act
        )

        # Compute the new K-state from the orthogonal basis and phi(t)
        k_proj = me.Kstate_from_phi_basis(phi=-phi_proj, basis=orth_basis_act)
        print("k_proj is", type(k_proj), k_proj.acts_over())

        # Normalize to obtain the updated density matrix sigma(t)
        sigma_proj = GibbsDensityOperator(k_proj)

        # Record expectation values of the observable
        ev_obs_maxent_act_partial_sum_ell.append(sigma_proj.expect(obs_sz))

        # Calculate the local error bound using partial sums
        local_bound_error_ell.append(
            me.m_th_partial_sum(phi=phi_proj, m=2)
            / me.m_th_partial_sum(phi=phi_proj, m=0)
        )

        # Check if the local error exceeds the threshold
        if abs(local_bound_error_ell[-1]) >= eps_tol:
            # If positive, perform actualization
            no_acts_ell.append(no_acts_ell[-1] + 1)

            # Log errors at specific intervals for debugging
            if list(timespan).index(curr_t) % 50 == 0:
                print("error", local_bound_error_ell[-1])

            # Update the local time value and save the cut time index
            local_t_value = curr_t
            saved_cut_times_index_ell.append(list(timespan).index(curr_t))

            # Map the K-local state onto a Mean-Field state,
            # retaining only its one-body correlations, to be used in sp
            _, sigma_act = me.mft_state_it(k_proj, sigma_act, max_it=10)

            # Recompute the scalar product using the MF state
            sp_local = me.fetch_covar_scalar_product(sigma=sigma_act)

            # The new basis is spanned from the K_proj state
            hbb_ell_act = me.build_Hierarch(
                generator=hamiltonian, seed_op=k_proj, deep=chosen_depth
            )

            # The growth in complexity of the basis is arrested by projecting
            # this basis onto simpler basis
            # composed of $nmax$-body observables only, with $nmax$ much
            # smallaller than the size of the system.
            hbb_ell_act = [
                project_to_n_body_operator(
                    one_body_from_qutip_operator(op), nmax=max_bodies
                )
                for op in hbb_ell_act
            ]

            orth_basis_act = me.orthogonalize_basis(basis=hbb_ell_act, sp=sp_local)

            # Recompute the Hamiltonian tensor and project the state
            hij_tensor_act, w_errors = me.fn_Hij_tensor_with_errors(
                generator=hamiltonian, basis=orth_basis_act, sp=sp_local
            )
            instantaneous_w_errors.append(w_errors)
            spectral_norm_hij_tensor_ell.append(linalg.norm(hij_tensor_act))
            phi0_proj_act = me.project_op(k_proj, orth_basis_act, sp_local)
        else:
            # If error is below the threshold, retain the current basis and sp
            number_of_commutators_ell.append(number_of_commutators_ell[-1])
            no_acts_ell.append(no_acts_ell[-1])

    return current_simulation


def load_parameters():
    """
    Load the command line parameters
    """
    parser = argparse.ArgumentParser(add_help=False)

    parser = argparse.ArgumentParser(
        description="Run the MF-Adaptive Max-Ent simulation."
    )
    parser.add_argument(
        "--size", type=int, required=True, help="Length of the spin chain"
    )
    parser.add_argument(
        "--ell", type=int, required=True, help="Depth of the Hierarchical bases"
    )
    parser.add_argument(
        "--tolerance", type=float, required=True, help="Tolerance of Partial Sum"
    )
    parser.add_argument(
        "--nmax",
        type=int,
        required=True,
        help="$m$-body components retained only via a MF proj",
    )
    args = parser.parse_args()

    params = {}
    params["chosen_depth"] = args.ell
    params["max_bodies"] = args.nmax
    params["eps"] = args.tolerance
    params["size"] = args.size

    params["models_lib_file"] = "../qalma/lib/models.xml"
    params["lattice_lib_file"] = "../qalma/lib/lattices.xml"
    params["Jx"] = 1.0
    params["Jy"] = 0.75 * params["Jx"]
    params["Jz"] = 1.05 * params["Jx"]

    coeffs = [
        1,
        0,
        -params["Jx"] * params["Jy"]
        + params["Jx"] * params["Jy"]
        + params["Jy"] * params["Jz"],
        -2 * params["Jx"] * params["Jy"] * params["Jz"],
    ]
    f_factor = np.real(max(np.roots(np.poly1d(coeffs))))
    chi_y = fsolve(
        lambda x, y: x * np.arcsinh(x) - np.sqrt(x**2 + 1) - y, 1e-1, args=(0,)
    )[0]
    params["vLR"] = 4 * f_factor * chi_y
    params["timespan"] = np.linspace(0.0, 650.1 / params["vLR"], 75)
    return params


def post_process(simulation_dict):
    """
    Generate graphics from the results
    """
    # Acá poner las rutinas que hacen los gráficos....
    params = simulation_dict["params"]
    plt.figure(figsize=(8, 6))
    plt.plot(
        simulation_dict["params"]["timespan"],
        simulation_dict["ev_obs_maxent_act_partialSum_ell"],
        label="MF-Adaptive Max-Ent",
        color="red",
        linestyle="--",
    )
    plt.xlabel("Time [arb. units]")
    plt.ylabel("Observable Expectation")
    plt.title(
        (
            f'MF-Adaptive Max-Ent Dynamics (L={params["size"]}, '
            f'ell={params["chosen_depth"]}, nmax={params["max_bodies"]})'
        )
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def initial_state(system_data):
    """
    Build the initial state and its generator from
    the system data.
    """
    system = system_data["system"]
    sites = system_data["sites"]
    site = sites[0]
    print("initializing on ", site[0])
    idop = system_data["idop"]
    hbb_0 = [
        idop,
        system.site_operator("Sx", "1[0]"),
        system.site_operator("Sy", "1[0]"),
        system.site_operator("Sz", "1[0]"),
    ]

    phi_0 = np.array([0.0, 0.25, 0.25, -10.0])
    k_0 = me.Kstate_from_phi_basis(phi_0, hbb_0)
    print("initial state", k_0)
    sigma_0 = GibbsProductDensityOperator(k_0)
    k_0 = -(sigma_0.logm())
    print("   sigma=", sigma_0)
    print("   k_0=", k_0)
    return k_0, sigma_0


def main(parms):
    """
    Build the system and run the simulation
    """
    system_data = build_system(parms)
    k_0, sigma_0 = initial_state(system_data)
    print("sigma_0", type(sigma_0))
    sim = run_restricted_simulation(parms, system_data, k_0, sigma_0)
    post_process(sim)


if __name__ == "__main__":
    main(load_parameters())
    sys.exit(0)
