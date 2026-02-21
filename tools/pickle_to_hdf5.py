#!/usr/bin/env python

import pickle
import sys

for filename in sys.argv[1:]:
    if filename[-4:] == ".pkl":
        out_filename = filename[:-4] + ".h5"
    else:
        out_filename = filename + ".h5"

    print(filename, "->", out_filename)
    with open(filename, "rb") as f:
        sim = pickle.load(f)
    sim.save_hdf5(out_filename)
