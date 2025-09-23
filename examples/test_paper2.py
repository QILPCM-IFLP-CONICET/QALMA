from datetime import datetime
from qalma.scalarprod import fetch_covar_scalar_product
from qalma.evolution.maxent_evol import (
    adaptive_projected_evolution,
    projected_evolution,
)
from qalma.evolution import qutip_me_solve, series_evolution as series_solver
import numpy as np
import matplotlib.pyplot as plt
from qalma.scalarprod.basis import HierarchicalOperatorBasis

from qalma.model import build_system


from qalma.operators import ScalarOperator
from qalma.operators.states import (
    GibbsDensityOperator,
    GibbsProductDensityOperator,
)

import logging


logging.basicConfig(level=logging.INFO)

def update_basis_callback(state):
    """
    Function called each time the basis is updated.
    """
    print("  now:", datetime.now())
    print("  Updating basis at t=",state["t"])
    print("  rebuild the basis took ", state["basis time cost"])

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)

# PARAMETERS
ts = np.linspace(0,1280,180)
BETA = .01


L = 8
JX=0.02890 # 1.75  -> vLR=1
ALPHA = .61 #   jy=.9 jx
JY=(1-ALPHA)*JX
PHI_0=[0,.25,.25,1]
SYSTEM=build_system(geometry_name="chain lattice", model_name="spin",
                    **{"L": L, "Jz":0, "Jxy":JX, "Alpha":ALPHA})
HAMILTONIAN = SYSTEM.global_operator("Hamiltonian").simplify()
SZ_TOTAL = SYSTEM.global_operator("Sz")

SITES = tuple(SYSTEM.sites.keys())
# Other operators
GLOBAL_IDENTITY = ScalarOperator(1.0, SYSTEM)


SX_A = SYSTEM.site_operator(f"Sx@{SITES[0]}")
SX_B = SYSTEM.site_operator(f"Sx@{SITES[1]}")
SX_AB = 0.7 * SX_A + 0.3 * SX_B


SY_A = SYSTEM.site_operator(f"Sy@{SITES[0]}")
SY_B = SYSTEM.site_operator(f"Sy@{SITES[1]}")

SPLUS_A = SYSTEM.site_operator(f"Splus@{SITES[0]}")
SPLUS_B = SYSTEM.site_operator(f"Splus@{SITES[1]}")
SMINUS_A = SYSTEM.site_operator(f"Sminus@{SITES[0]}")
SMINUS_B = SYSTEM.site_operator(f"Sminus@{SITES[1]}")


SZ_A = SYSTEM.site_operator(f"Sz@{SITES[0]}")
SZ_B = SYSTEM.site_operator(f"Sz@{SITES[1]}")
SZ_C = SYSTEM.site_operator(f"Sz@{SITES[2]}")
SZ_AB = 0.7 * SZ_A + 0.3 * SZ_B

K0= SX_A*PHI_0[1]+ SY_A*PHI_0[2]+ SZ_A*PHI_0[3]
RHO_0 = GibbsProductDensityOperator(K0)
K0 = -RHO_0.logm()


TRACK_OBSERVABLES = (SZ_TOTAL,) #(SZ_TOTAL, SZ_TOTAL*SZ_TOTAL,HAMILTONIAN,HAMILTONIAN**2,)



def run_exact(axis):
    k_0 = K0*BETA
    hamiltonian = HAMILTONIAN
    print("Start exact:", datetime.now())
    exact =[GibbsDensityOperator(k).to_qutip_operator() for k in qutip_me_solve(hamiltonian, k_0, ts)]
    print("Plot observables")
    exact_expect = [np.real(rho.expect(SZ_TOTAL)) for rho in exact]
    axis.set_ylim(min(-max(exact_expect), min(exact_expect)), max(exact_expect))
    axis.plot(ts, exact_expect,label="exact")
    axis.plot(ts, [exact_expect[0]*np.cos(1.4142*BETA*JX*ALPHA*t) for t in ts],label="2nd order Ehrenfest",ls="-.")
    axis.plot(ts, [exact_expect[0]*(1 + 2 * ALPHA**2/(1+3 *ALPHA**2)*(np.cos(JX*BETA*t*(1+3*ALPHA**2)**.5)-1)) for t in ts],label="3nd order Ehrenfest",ls="-.")
    print("   done")




def run_series(axis):
    k_0 = K0*BETA
    hamiltonian = HAMILTONIAN
    print("Start series:", datetime.now())
    series =[GibbsDensityOperator(k).to_qutip_operator() for k in series_solver(hamiltonian, k_0, ts, 30).states]
    print("Plot observables")
    series_expect = [np.real(rho.expect(SZ_TOTAL)) for rho in series]
    axis.plot(ts, series_expect,label="series")
    print("   done")



def run_projected(axis):
    k_0 = K0*BETA
    hamiltonian = HAMILTONIAN
    print("Start exact:", datetime.now())
    exact_k = qutip_me_solve(hamiltonian, k_0, ts).states
    exact =[GibbsDensityOperator(k).to_qutip_operator() for k in exact_k]

    sigma_0 = GibbsProductDensityOperator(K0)
    sp = fetch_covar_scalar_product(sigma_0)
    basis = HierarchicalOperatorBasis(k_0, HAMILTONIAN,50,sp)
    print("projecting using basis", basis)
    projected =[GibbsDensityOperator(basis.project_onto(k)).to_qutip_operator() for k in exact_k]
    
    print("Plot observables")
    exact_expect = [np.real(rho.expect(SZ_TOTAL)) for rho in exact]
    axis.set_ylim(min(-max(exact_expect), min(exact_expect)), max(exact_expect))
    axis.plot(ts, exact_expect,label="exact")
    projected_expect = [np.real(rho.expect(SZ_TOTAL)) for rho in projected]
    axis.plot(ts, projected_expect,label="projected")
    print("   done")


def run_simulation_adaptive(basis_depth, n_body, tolerance, axis):
    k_0 = K0*BETA
    hamiltonian = HAMILTONIAN
    print(f"                   Start max ent L={basis_depth},m={n_body},tol={tolerance}:", datetime.now())
    try:
        max_ent =[GibbsDensityOperator(k) for k in
                    adaptive_projected_evolution(
                    hamiltonian,
                    k_0,
                    ts,
                    basis_depth,
                    n_body,
                    tol=tolerance,
                    on_update_basis_callback=update_basis_callback,
                    include_one_body_projection=True,
                    extra_observables=TRACK_OBSERVABLES,
                    ).states]
        #plt.scatter(ts[:len(max_ent)], [np.real(rho.expect(k_0)) for rho in max_ent], label=f"$\\ell={basis_depth}$, m={n_body}, tol={tolerance}")
        plt.scatter(ts[:len(max_ent)], [np.real(rho.expect(SZ_TOTAL)) for rho in max_ent], label=f"c->$\\ell={basis_depth}$, m={n_body}, tol={tolerance}")
        print("len:", len(max_ent))
    except Exception as e:
        print("                   EXCEPTION ")
        print(type(e), e)
        raise


def run_simulation_projected(basis_depth, n_body, tolerance, axis):
    k_0 = K0*BETA
    hamiltonian = HAMILTONIAN
    print(f"                   Start max ent L={basis_depth},m={n_body},tol={tolerance}:", datetime.now())
    try:
        max_ent =[GibbsDensityOperator(k) for k in
                projected_evolution(
                    hamiltonian,
                    k_0,
                    ts,
                    100,
                    100
                    )]
        #plt.scatter(ts[:len(max_ent)], [np.real(rho.expect(k_0)) for rho in max_ent], label=f"$\\ell={basis_depth}$, m={n_body}, tol={tolerance}")
        plt.scatter(ts[:len(max_ent)], [np.real(rho.expect(SZ_TOTAL)) for rho in max_ent], label=f"proyected-> $\\ell={basis_depth}$, m={n_body}, tol={tolerance}")
        print("len:", len(max_ent))
    except Exception as e:
        print("                   EXCEPTION ")
        print(type(e), e)
        raise


def run_simulations():
    fig, axis = plt.subplots()
    run_projected(axis)
    run_series(axis)
    run_simulation_adaptive(6,4,.025,axis)
    axis.legend()
    # axis.set_title(f"Max-Ent evolution, beta={BETA} tolerance={tolerance}")
    fig.savefig("output_ab.svg")


if __name__=="__main__":
    run_simulations()
