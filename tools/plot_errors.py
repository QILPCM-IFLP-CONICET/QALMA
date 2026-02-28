#!/usr/bin/env python

import matplotlib as mpl

mpl.use("module://mpl_ascii")
import pickle
import sys
import glob

from matplotlib import pyplot as plt

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 8))


for idx, filename in enumerate(
        [
        fn
        for pat in sys.argv[1:]
        for fn in glob.glob(pat)
        ]):
    print(filename)
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(data.stats.keys())
    axs[0].plot(data.time_span, data.stats["away_from_ref"], label=filename[:20])
    axs[1].plot(data.time_span, data.stats["errors"], label=filename[:20])
    axs[2].plot(
        data.stats["basis update times"],
        data.stats["basis time costs"],
        label=filename[:20],
    )

axs[0].set_title("away from ref")
axs[1].set_title("error")
axs[2].set_title("basis costs")
axs[2].legend()
plt.savefig("errors.svg")
plt.show()
