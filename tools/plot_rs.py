#!/usr/bin/env python

import matplotlib as mpl

mpl.use("module://mpl_ascii")
import glob
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt

from qalma.evolution.simulation import Simulation
from qalma.operators.functions import relative_entropy
from qalma.operators.states import DensityOperatorMixin, GibbsDensityOperator


def find_exact_sim(sim):
    """Find a simulation containing the exact evolution"""
    parms = sim.parameters
    L = parms["L"]
    BETA = parms["beta"]
    exact_sim = None
    print("look for", f"exact_*L={L}_beta={BETA}*.h5")
    for candidate in glob.glob(f"exact_*L={L}_beta={BETA}*.h5"):
        print("candidate:", candidate)
        exact_sim = Simulation.load_hdf5(candidate)
        if exact_sim.states[0].system == sim.states[0].system:
            print("candidate found:", candidate)
            break
        print("systems are not equal. Try the next...")

    if exact_sim is None:
        print("No candidate found. Run `run_dynamics.py` with the `--full` parameter.")
    return exact_sim


def populate_rs(sim) -> bool:
    """
    Compute the relative entropy of the states relative
    to the exact evolution.
    """
    exact_sim = find_exact_sim(sim)
    rs_values = []
    if exact_sim is None:
        print("No exact simulation available")
        return False
    for sigma, rho in zip(sim.states, exact_sim.states):
        if not isinstance(rho, DensityOperatorMixin):
            rho = GibbsDensityOperator(rho)
        if not isinstance(sigma, DensityOperatorMixin):
            sigma = GibbsDensityOperator(sigma)
        rs_values.append(relative_entropy(sigma, rho))
    sim.expect_ops["relative entropy"] = rs_values
    return True


for idx, filename in enumerate([fn for pat in sys.argv[1:] for fn in glob.glob(pat)]):
    print(filename)
    is_hdf5 = True
    obs_key = "relative entropy"
    if filename[-4:] == ".pkl":
        is_hdf5 = False
        with open(filename, "rb") as f:
            simulation = pickle.load(f)
    elif filename[-3:] == ".h5":
        simulation = Simulation.load_hdf5(filename)
    else:
        print("wrong file extension", filename)
        exit(-1)

    if obs_key not in simulation.expect_ops:
        populate_rs(simulation)
        if is_hdf5:
            simulation.save_hdf5(filename)
        else:
            with open(filename, "wb") as f:
                pickle.dump(simulation, f)

    (plt.plot if idx == 0 else plt.scatter)(
        simulation.time_span[: len(simulation.expect_ops[obs_key])],
        [np.real(x) for x in simulation.expect_ops[obs_key]],
        label=filename[:20],
    )

plt.legend()
plt.savefig("relative_entropy_evol.svg")
plt.show()
