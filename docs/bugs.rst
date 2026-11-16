.. index:: bugs
.. _bugs:

Known limitations and reporting bugs
====================================

Known limitations
-----------------

* **Fermionic models.** Models whose fermionic quantum numbers take two
  values (``spinless fermions``, ``fermion Hubbard``, ``t-J``,
  ``Kondo lattice``, ...) are split into fermionic qubits plus a rest, and
  their fermionic operators are mapped to Pauli strings through the
  Jordan-Wigner transformation, so the signs are exact (see
  :doc:`user/systems`). For constrained bases like ``t-J``, the results
  must be taken in the sector without unphysical sites. If the split is
  disabled (``qubit_split=False``), or a fermionic quantum number takes more
  than two values, operators on different sites commute, the results are
  those of the corresponding hard-core particles, and building the system
  emits a :class:`qalma.FermionicSignWarning`.

* **Exact methods scale exponentially.** Operations that need the full
  matrix representation of an operator (``to_qutip``, exact
  exponentials and spectra) are limited to small systems.

* **RPA reductions.** Only the Gaussian truncation of the bosonization is
  implemented. At intermediate temperatures, and for strong couplings, the
  results are qualitative, and the generator corrections are not reliable
  at low temperatures (see :doc:`user/rpa`).

* **Simulation files.** HDF5 files written by QALMA 1.0 or newer do not
  contain Python pickles (see :doc:`user/hdf5_format`). Files written by
  older versions do: they are only loaded with ``allow_pickle=True``, may
  not be readable by other versions of QALMA and, as with any pickle,
  should only be opened if you trust them. Functions cannot be stored, so
  Gibbs states are stored without their symmetry projections.

Reporting bugs
--------------

Bugs and feature requests are tracked in the
`issue tracker <https://github.com/QILPCM-IFLP-CONICET/QALMA/issues>`_.
A useful report includes:

* a short, self-contained script that reproduces the problem;
* what you got (the full traceback or output) and what you expected;
* the versions involved, which you can get with

  .. code-block:: python

     import qalma, qutip, numpy, scipy
     print(qalma.__version__, qutip.__version__, numpy.__version__, scipy.__version__)

If you already know how to fix the problem, pull requests are welcome: see
``CONTRIBUTING.rst`` in the repository.
