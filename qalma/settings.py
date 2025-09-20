#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 19:07:05 2023

@author: mmatera <matera@fisica.unlp.edu.ar>
"""

import os
import os.path as osp

# import sys
# from pathlib import Path

# import pkg_resources


def get_srcdir():
    """Get the root directory of the source code"""
    filename = osp.normcase(osp.dirname(osp.abspath(__file__)))
    print("__file__", __file__)
    return osp.realpath(filename)


ROOT_DIR = get_srcdir()
FIGURES_DIR = f"{ROOT_DIR}/../docs/figures"
LATTICE_LIB_FILE = f"{ROOT_DIR}/lib/lattices.xml"
MODEL_LIB_FILE = f"{ROOT_DIR}/lib/models.xml"

# set the level of verbosity in the warnings and error messages
VERBOSITY_LEVEL = 2
QALMA_INFER_ARITHMETICS = False
QALMA_ALLOW_OVERWRITE_BINDINGS = False
QALMA_TOLERANCE = 1.0e-14
USE_PARALLEL = bool(os.environ.get("USE_PARALLEL", 0))
PARALLEL_MAX_WORKERS = int(os.environ.get("QALMA_MAX_WORKERS", 8))
PARALLEL_USE_THREADS = False


MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE = 8
