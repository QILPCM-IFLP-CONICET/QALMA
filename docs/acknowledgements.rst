.. _acknowledgements:

Acknowledgements
================

QALMA is built on top of the open-source scientific Python ecosystem, and it
would not exist without it.

* The numerical core relies on **NumPy** :cite:p:`NumPy2020` for array
  computing and on **SciPy** :cite:p:`SciPy2020` for linear algebra, matrix
  exponentials, sparse matrices and optimization.
* Dense and sparse blocks, and the exact reference solvers used in the tests
  and examples, come from **QuTiP** :cite:p:`QuTiP2012,QuTiP2013,QuTiP5`.
* Simulations are stored through **h5py** :cite:p:`h5py`, the Python interface
  to the HDF5 format.
* The lattice and model libraries in ``qalma/lib/*.xml`` come from the
  **ALPS** project :cite:p:`ALPS2011,GAENKO2017235`, and keep their original
  license (Boost Software License 1.0).

If you use QALMA in your research, please also cite the works that
implement the methods you used; they are listed in :ref:`references`.
