#!/usr/bin/env python

import pickle
import sys

filename = sys.argv[1]
print(filename)
with open(filename, "rb") as f:
    data = pickle.load(f)
print("parms:\n")
for key, val in data.parameters.items():
    print(key, ":", val)
