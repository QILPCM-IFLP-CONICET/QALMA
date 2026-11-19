#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Tue Sep 12 19:07:05 2023.

@author: mmatera <matera@fisica.unlp.edu.ar>
"""

import os
import os.path as osp

# import sys
# from pathlib import Path

# import pkg_resources


def get_srcdir():
    """Get the root directory of the source code."""
    filename = osp.normcase(osp.dirname(osp.abspath(__file__)))
    return osp.realpath(filename)


def _env_flag(*names: str) -> bool:
    """Read a boolean flag from the first of the environment variables set.

    Empty values, "0", "false", "no" and "off" (in any case) are False.
    """
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return value.strip().lower() not in ("", "0", "false", "no", "off")
    return False


ROOT_DIR = get_srcdir()
LATTICE_LIB_FILE = f"{ROOT_DIR}/lib/lattices.xml"
MODEL_LIB_FILE = f"{ROOT_DIR}/lib/models.xml"

# set the level of verbosity in the warnings and error messages
VERBOSITY_LEVEL = 2
QALMA_INFER_ARITHMETICS = False
QALMA_ALLOW_OVERWRITE_BINDINGS = False
QALMA_TOLERANCE = float(os.environ.get("QALMA_TOLERANCE", "1.0e-14"))
USE_PARALLEL = _env_flag("QALMA_USE_PARALLEL", "USE_PARALLEL")
PARALLEL_MAX_WORKERS = int(os.environ.get("QALMA_MAX_WORKERS", 8))
PARALLEL_USE_THREADS = _env_flag("QALMA_PARALLEL_USE_THREADS", "PARALLEL_USE_THREADS")


# Mean field approximation parameters.
DEFAULT_MAX_NUMBER_OF_FIELDS = 6
MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE = 8
