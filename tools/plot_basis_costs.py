#!/usr/bin/env python

# import matplotlib as mpl
# mpl.use("module://mpl_ascii")

import pickle
import sys

from matplotlib import pyplot as plt

fig, axs = plt.subplots(2, 2, sharex=True)

for filename in sys.argv[1:]:
    print(filename)
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(data.stats.keys())
    axs[0][0].plot(data.time_span, data.stats["away_from_ref"], label=filename[:20])
    axs[0][1].plot(data.time_span, data.stats["n_body_sector"], label=filename[:20])
    axs[1][0].plot(data.time_span, data.stats["occupation factor"], label=filename[:20])
    axs[1][1].plot(
        data.stats["basis update times"],
        data.stats["basis time costs"],
        label=filename[:20],
    )

axs[0][0].set_title("away from ref")
axs[0][1].set_title("n body sector")
axs[1][0].set_title("occupation factor")
axs[1][1].set_title("basis time costs")
axs[1][1].legend()
plt.savefig("costs_basis.svg")
plt.show()
