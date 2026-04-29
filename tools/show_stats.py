#!/usr/bin/env python

import pickle
import sys

filename = sys.argv[1]
print(filename)
with open(filename, "rb") as f:
    data = pickle.load(f)
print("stats:")
for key, val in data.stats.items():
    print(key, ":", val)
