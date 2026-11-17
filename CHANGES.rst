Changelog
=========

All notable changes to QALMA are documented in this file. Version numbers
follow `Semantic Versioning <https://semver.org>`_.


1.0.0 (unreleased)
------------------

First public release.

Packaging
+++++++++

* QALMA is now distributed under the LGPL-3.0 (previously GPL-3.0). The ALPS
  model and lattice libraries in ``qalma/lib`` keep the Boost Software
  License 1.0, whose text (``qalma/lib/LICENSE_1_0.txt``) is installed with
  the package.
* Python 3.11 or newer is required.
* Runtime dependencies are declared with tested lower bounds:
  ``numpy>=1.26``, ``scipy>=1.11``, ``qutip>=5.1``, ``h5py>=3.10`` and
  ``packaging``.
* ``matplotlib`` is now an optional dependency, needed only by the plotting
  helpers. Install it with ``pip install "qalma[plot]"``.
* ``pyproject.toml`` is the only source of package metadata; ``setup.py``
  was removed.
* The wheel installs only the ``qalma`` package. Previously it also
  installed a top-level ``test`` package.
* ``qalma.__version__`` is available and is the single source of the
  version number.

Operators
+++++++++

* New classes ``PauliStringOperator`` (a product of Pauli matrices, stored
  as a bit string), ``PauliSumOperator`` (linear combinations of them,
  stored as arrays) and ``TensorProductOperator`` (a Pauli string times
  local factors on other sites). Products, commutators and
  anticommutators between them are computed symbolically on the bit
  strings, and ``to_pauli_sum`` expands operators on two-level sites.
  The three classes have native HDF5 formats.
* ``SumOperator.simplify`` keeps sums with Pauli strings or tensor
  products symbolic: the strings are collected in a single
  ``PauliSumOperator``, tensor products with the same string and local
  site are added, and products that differ on a single site are merged,
  instead of converting the terms into dense blocks.

Mean-field and RPA
++++++++++++++++++

* New module ``qalma.meanfield.rpa``: RPA reduction of Gibbs states. The
  reduced generator is the mean-field one plus the contribution of the
  Gaussian fluctuations of the environment, with a static or a dynamical
  (bosonization) correction. It is requested with the ``rpa`` and
  ``dynamic`` options of ``GibbsDensityOperator.partial_trace``.
* ``partial_trace(..., rpa=True, hybrid=True)`` keeps the subsystem exact and
  treats only the environment in the RPA (non perturbative in the coupling;
  see ``hybrid_options``).
* ``RPAGibbsDensityOperator`` is a Gibbs state whose reductions include the
  RPA and which stores the subsystem-independent data of the RPA
  (``rpa_structure``). It also gives the entropies and negativities of the
  bosonic RPA state (``rpa_entropy``, ``rpa_negativity``) and handles zero
  modes (``rpa_zero_modes``).
* Only the Gaussian truncation is implemented; see :ref:`rpa` for its
  limitations.

HDF5 files
++++++++++

* ``Simulation.save_hdf5`` writes a new format (format 1) without Python
  pickles: the system, the operators and the states are stored as numbers,
  strings and arrays. The files can be read from other languages, and
  opening a file does not run code stored in it. The format is documented
  in the user guide and has a version number; QALMA 1.x reads every version
  up to its own.
* New package ``qalma.io``, with ``save_operators`` and ``load_operators``
  to store collections of operators and states, and the functions that
  store systems, operators and dictionaries in groups of an HDF5 file.
* HDF5 files written by earlier versions contain pickles. They are now
  read-only and need ``Simulation.load_hdf5(path, allow_pickle=True)``;
  without it, the error explains how to open and convert them.
  ``tools/pickle_to_hdf5.py`` converts old HDF5 files and pickles of
  ``Simulation`` objects to the new format.
* ``Simulation.load`` raises an error instead of returning ``None`` when
  the file cannot be loaded, and needs ``allow_pickle=True`` to load a
  pickle file.
* ``Simulation.save_hdf5`` raises the errors (for instance,
  ``PermissionError``) instead of logging a warning, and writes through a
  temporary file, so a failure does not leave an incomplete file.
* The groups ``expect obs`` and ``time span`` are now ``expect_ops`` and
  ``time_span``. Keys of ``expect_ops`` that are not strings (like the
  positions of a list of observables) are kept: they used to be loaded as
  strings.
* Gibbs states are stored without their ``symmetry_projections``, which
  are functions, with a ``qalma.io.IncompleteStateWarning``. Objects in
  ``parameters``, ``stats`` or ``expect_ops`` that the format does not
  support raise a ``TypeError``, unless ``allow_pickle=True`` is given.
* Fixed: ``append`` and ``extend`` on the states of a ``SimulationHDF5``
  stored the states under a wrong name.
* ``adaptive_projected_evolution`` no longer writes a pickle checkpoint
  (``__adaptative_*.pkl``) in the working directory after each step. Pass
  ``checkpoint_file`` to get an HDF5 checkpoint in the file of your choice.
* The expressions of the models (``qalma.utils.eval_expr``) are restricted
  to arithmetic: numbers, parameters, operators and function calls. Other
  Python syntax raises a ``SyntaxError`` and Python builtins are not
  available, so neither an ALPS library nor a simulation file can run
  arbitrary code.

Max-Ent evolution
+++++++++++++++++

* New, experimental
  :func:`~qalma.evolution.leakage_controlled_evolution`: restricted Max-Ent
  evolution whose basis is rebuilt when the leakage rate
  ``||pi_perp(-i[H, K~])||`` exceeds a multiple of its value right after the
  last rebuild, and whose reference state is rebuilt when its local (per
  site) mean-field drift exceeds a tolerance. Both criteria are intensive.
  It reports a Galerkin error estimate that keeps the cancellations between
  basis elements, and a second-order estimate of the entropy production of
  the instantaneous Max-Ent projection.
* New helpers :func:`~qalma.evolution.leakage_gram`,
  :func:`~qalma.evolution.leakage_rate`,
  :func:`~qalma.evolution.leakage_integrals` and
  :func:`~qalma.evolution.local_mean_field_drift`.

Bug fixes
+++++++++

* ``remove_under_tolerance_terms`` (used by
  :func:`~qalma.scalarprod.trim_terms_by_tolerance`) always removed the
  smallest term, whatever its norm. Since the hierarchical bases of
  :func:`~qalma.evolution.update_basis` trim every level, each element lost
  a term, and the generator matrix of the restricted evolution differed from
  the Galerkin one by 10-30 % whenever the n-body projection acted.
* :func:`~qalma.evolution.update_basis` and ``update_basis_heavy`` appended
  the mean-field reference generator to the basis even when it was already
  in its span (always the case for one-body generators). The Gram matrix
  became singular, and for deep bases the evolved coefficients were
  corrupted. The reference is now appended only if it is linearly
  independent.

Behaviour changes
+++++++++++++++++

* Importing ``qalma`` no longer calls ``logging.basicConfig``. All messages
  are emitted through loggers under the ``"qalma"`` name, so INFO and DEBUG
  messages are hidden unless the application configures logging, e.g.::

      import logging
      logging.basicConfig(level=logging.INFO)

* Diagnostic ``print`` calls in the library were removed or turned into
  ``DEBUG`` log messages.
* Fermionic models are split into fermionic qubits (one per fermionic
  quantum number) plus a rest subsite, and their fermionic operators are
  mapped to Pauli strings through the Jordan-Wigner transformation, so the
  signs are exact: ``spinless fermions``, the Hubbard models, ``t-J`` and
  ``alternative t-J`` (with the unphysical local states counted by the
  global operator ``unphysical_sites``) and ``Kondo lattice``. The sites of
  these systems are the subsites (``1[0]:Nup``, ``1[0]:Sz``, ...). The
  split can be forced or disabled with ``build_system(..., qubit_split=...)``;
  when it is disabled, or not possible, building the system emits a
  ``qalma.FermionicSignWarning``.
* The ``USE_PARALLEL`` environment variable is now parsed as a boolean:
  ``USE_PARALLEL=0`` used to *enable* the parallel routines. The preferred
  name is ``QALMA_USE_PARALLEL``.
* ``qalma.settings.FIGURES_DIR`` was removed; it pointed outside the
  installed package.
* The code paths for QuTiP 4 were removed (QuTiP >= 5.1 is required).
* Fixed: building a system whose lattice has loops with a model whose sites
  do not define the operators of a loop term (for instance ``boson Hubbard``
  on ``square lattice with loop``) failed with an ``AssertionError``. The
  global operator with that loop term is now dropped, as global operators
  with bond terms already were.
