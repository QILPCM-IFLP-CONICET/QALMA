#!/usr/bin/env python
import logging
import pickle
import sys
from datetime import datetime

import numpy as np

from qalma.meanfield.variational import compute_rel_entropy, variational_quadratic_mfa

logging.basicConfig(level=logging.INFO)


def run_meanfield_projection(filename):
    out_filename = "meanfield_" + filename[5:]
    with open(filename, "rb") as f:
        result = pickle.load(f)

    rel_s_vals = []
    varmf = []
    sigma_mf = None
    system = result.states[0].system
    sz_total = sum(system.site_operator("Sz", site) for site in system.sites)
    assert sz_total is not None
    for t, state_ln in zip(result.time_span, result.states):
        logging.info(
            "\t\t\t\t\t\t\t\tprojecting rho[%s] at %s", str(t), str(datetime.now())
        )
        sigma_mf = variational_quadratic_mfa(state_ln, sigma_ref=sigma_mf)
        rel_s = compute_rel_entropy(sigma_mf, state_ln)
        varmf.append(sigma_mf)
        rel_s_vals.append(rel_s)
        with open(f"_{out_filename}", "wb") as f:
            pickle.dump(
                {
                    "relative entropy": rel_s_vals,
                    "states": varmf,
                    "time_span": result.time_span[: len(varmf)],
                },
                f,
            )

    result.states = varmf
    result.expect_ops[0] = [np.real(rho.expect(sz_total)) for rho in varmf]
    result.expect_ops["relative entropy"] = rel_s_vals

    with open(out_filename, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        assert filename[-4:] == ".pkl", "pickle file expected."
        print(filename)
        run_meanfield_projection(filename)
