.. index:: bugs
.. _bugs:

Known limitations and reporting bugs
====================================

Known limitations
-----------------

* **Fermionic models.** The signs coming from the anticommutation of
  fermionic operators on different sites are not implemented: operators on
  different sites always commute. Building a system from a model with
  fermionic operators (``fermion Hubbard``, ``spinless fermions``, ``t-J``,
  ``Kondo lattice``, ...) emits a :class:`qalma.FermionicSignWarning`. The
  results are those of the corresponding hard-core particles, and agree
  with the fermionic ones only when the signs cancel out, e.g. for
  nearest-neighbour hopping of a single species on an open chain.

* **Exact methods scale exponentially.** Operations that need the full
  matrix representation of an operator (``to_qutip``, exact
  exponentials and spectra) are limited to small systems.

* **Simulation files.** HDF5 files written by
  :class:`qalma.evolution.simulation.Simulation` store the system and the
  states as Python pickles inside the file. As with any pickle, only open
  files that you trust, and note that files may not be readable by other
  versions of QALMA.

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
