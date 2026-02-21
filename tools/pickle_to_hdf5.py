#!/usr/bin/env python

import pickle
import sys

for filename in sys.argv[1:]:
    out_filename = filename.split(".")[0] + ".h5"
    print(filename, "->", out_filename)
    with open(filename, "rb") as f:
        sim = pickle.load(f)
    sim.save_hdf5(out_filename)
