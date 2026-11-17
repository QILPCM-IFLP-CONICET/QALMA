#!/usr/bin/env python

import pickle
import sys

from qalma.evolution.simulation import Simulation

filename = sys.argv[1]
print(filename)

if filename[-4:] == ".pkl":
    with open(filename, "rb") as f:
        simulation = pickle.load(f)
elif filename[-3:] == ".h5":
    simulation = Simulation.load_hdf5(filename)
else:
    print("unknown filename extension", filename)
    exit(-1)


print("parms:\n")
for key, val in simulation.parameters.items():
    print(key, ":", val)
