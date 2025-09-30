#!/usr/bin/env python

import matplotlib as mpl

mpl.use("module://mpl_ascii")
import pickle
import sys

import numpy as np
from matplotlib import pyplot as plt

from qalma.operators.states import GibbsDensityOperator

for filename in sys.argv[1:]:
    print(filename)
    with open(filename, "rb") as f:
        data = pickle.load(f)

    if not data.expect_ops:
        if data.states:
            print("build expect from states")
            system = data.states[0].system
            sz_total = sum(system.site_operator("Sz", s) for s in system.sites)
            data.expect_ops[0] = [
                GibbsDensityOperator(k).to_qutip_operator().expect(sz_total)
                for k in data.states
            ]
            with open(filename, "wb") as f:
                pickle.dump(data, f)
                print("data are now stored...")
        else:
            print(" does not contain information. Skip.")
            continue

    plt.plot(
        data.time_span[: len(data.expect_ops[0])],
        [np.real(x) for x in data.expect_ops[0]],
        label=filename,
    )

plt.legend()
plt.savefig("sz_evol.svg")
plt.show()
