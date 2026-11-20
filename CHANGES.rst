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

Covariance scalar product
+++++++++++++++++++++++++

* The covariance scalar product relative to a product state
  (``fetch_covar_scalar_product``, and through it every Gram matrix,
  projection and basis built from it) has a fast kernel. Pairs of terms with
  disjoint supports are not visited: their contribution reduces to a product
  of the total means. Pairs of product terms are evaluated site by site, and
  dense terms on up to ``MAX_DENSE_SITES`` sites are reduced onto the overlap
  with NumPy, with a cache bound to the reference state. For a uniform XXZ
  chain with 30 sites, the Gram matrix of a depth-4 hierarchical basis and
  its commutators is 15 times faster, and ``update_basis`` 4 times faster.
  Operators with other kinds of terms use the general implementation.
  Set ``qalma.scalarprod.covar.USE_FAST_KERNEL = False`` to disable it.
* The fast kernel is exact up to round-off; the general implementation
  discards small contributions, so results can change at the level of
  ``QALMA_TOLERANCE``. In particular, nearly linearly dependent elements of
  an operator basis are now detected more reliably.
* Bug fix: the squared norm of operators with non-normal terms (such as
  ``Splus @ Sz``) used ``<T^dagger T>`` instead of
  ``<{T^dagger, T}>/2`` for the diagonal contributions.

Mean-field and RPA
++++++++++++++++++

* New module ``qalma.meanfield.rpa``: RPA reduction of Gibbs states. The
  reduced generator is the mean-field one plus the contribution of the
  Gaussian fluctuations of the environment, with a static or a dynamical
  (bosonization) correction.
* ``RPAGibbsDensityOperator`` is a Gibbs state whose reductions include the
  RPA and which stores the subsystem-independent data of the RPA
  (``rpa_structure``). The options ``rpa``, ``dynamic`` and ``hybrid`` of its
  ``partial_trace`` select the reduction; ``hybrid=True`` keeps the
  subsystem exact and treats only the environment in the RPA (non
  perturbative in the coupling; see ``hybrid_options``).
  ``reduced_generator`` returns the approximate reduced generator.
  ``GibbsDensityOperator.partial_trace`` only provides the mean-field
  reduction.
* ``partial_trace`` of Gibbs states takes ``max_exact_sites``: reductions
  that involve at most that number of sites are computed exactly. Its
  default, ``qalma.settings.MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE``, is read when
  the reduction is computed; ``0`` always uses the approximation. It also gives the entropies and negativities of the
  bosonic RPA state (``rpa_entropy``, ``rpa_negativity``) and handles zero
  modes (``rpa_zero_modes``).
* The entropies and negativities of the bosonized RPA (``rpa_entropy``,
  ``rpa_negativity``) describe the quantum fluctuations: the static
  (classical) modes are left out. With zero modes they raise a
  ``ValueError``; the entanglement is then computed from the hybrid reduced
  states with ``symmetry_projections``, as in the mean field plus RPA
  treatment of a broken continuous symmetry (orientation integrated exactly,
  Gaussian intrinsic variables).
* Only the Gaussian truncation is implemented; see :ref:`rpa` for its
  limitations.

Gaussian bosonic states
+++++++++++++++++++++++

* New module ``qalma.gaussian``. ``BosonicGaussianSystem`` represents a
  Gaussian state of bosonic modes through its matrix of contractions, and
  gives covariance matrices, entropies and negativities of subsets of modes.
  It is built from a quadratic generator
  (``build_boson_system_from_quadratic_form``, the thermal state, with the
  removal of zero modes) or from the contractions
  (``build_boson_system_from_correlations``, which checks the commutation and
  uncertainty relations). The full bosonization of the RPA
  (``build_rpa_boson_system``) uses it.

Matrix product states (experimental)
++++++++++++++++++++++++++++++++++++

* New package ``qalma.mps`` with ``MPSGSOperator``, a density operator that
  stores the ground state of a Hamiltonian as a matrix product state. The
  state is computed with a two-site DMRG written with numpy and SciPy only,
  or provided with ``MPSGSOperator.from_mps``. ``expect`` and
  ``partial_trace`` contract the MPS, so they work for systems beyond exact
  diagonalization; the reduced states are ``QutipDensityOperator`` objects.
  The ground state is assumed to be non-degenerate. The class is a sibling
  of ``GibbsDensityOperator`` (a pure state has no generator ``k``), with
  the same arithmetic for scalars and mixtures.
* The argument ``permutation`` sets the order of the sites along the chain
  (the lexicographic order of the names is rarely a good one).
* ``MPS``, ``build_mpo`` (an MPO from a sum of few-body terms of any range)
  and ``dmrg`` are also available.
* ``MPSGSOperator`` can be stored in HDF5 files (type ``mps_gs``), without
  pickles: the tensors of the MPS, the order of the sites and the
  Hamiltonian are saved. A state that was not computed yet is stored with
  its Hamiltonian only.

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
* The adaptive Max-Ent evolution no longer writes a pickle checkpoint
  (``__adaptative_*.pkl``) in the working directory after each step. Pass
  ``checkpoint_file`` to get an HDF5 checkpoint in the file of your choice.
* The expressions of the models (``qalma.utils.eval_expr``) are restricted
  to arithmetic: numbers, parameters, operators and function calls. Other
  Python syntax raises a ``SyntaxError`` and Python builtins are not
  available, so neither an ALPS library nor a simulation file can run
  arbitrary code.

Max-Ent evolution
+++++++++++++++++

* A single function, :func:`~qalma.evolution.adaptive_maxent_evolution`,
  computes the restricted Max-Ent evolution. It replaces
  ``adaptive_projected_evolution``, ``projected_evolution`` and the
  ``update_basis_light`` and ``update_basis_heavy`` recipes, which were
  removed (as was the ``basis_update_callback`` option):

  * the basis is rebuilt when the leakage rate ``||pi_perp(-i[H, K~])||``
    exceeds a multiple of its value right after the last rebuild, and the
    mean-field reference when its local (per site) drift exceeds a
    tolerance; both criteria are intensive. ``adaptive=False`` uses a single
    basis built from ``K(0)``;
  * a memory term with an exponentially decaying kernel, built from the
    leakage Gram matrix (projection of the second-order equation of motion,
    Mori-Zwanzig closure), is included by default; its lifetime is estimated
    from the dynamics (``memory_time="auto"``), and ``memory_time=None``
    gives the first-order evolution;
  * the velocity is continuous across basis rebuilds, and a
    ``RuntimeWarning`` is issued when the dynamics leaves the n-body sector;
  * the statistics include a Galerkin error estimate that keeps the
    cancellations between basis elements, and an estimate of the entropy
    production of the instantaneous Max-Ent projection.
* The restricted Max-Ent evolution works on fermionic models, whose
  operators are Pauli strings (Jordan-Wigner). Three components rejected
  Pauli sums acting on single sites, which are one-body operators with a
  different representation: the linear term of
  :class:`~qalma.operators.quadratic.QuadraticFormOperator` (now converted
  to a one-body operator), the one-body generators of
  :class:`~qalma.operators.states.GibbsProductDensityOperator`, and the
  dispatch of :func:`~qalma.projections.n_body_projection`, which now
  searches the class hierarchy (a Pauli string is projected as a product
  operator). The fast covariance kernel already evaluates Pauli strings
  site by site, as product operators.
* New diagnostics :func:`~qalma.evolution.leakage_gram`,
  :func:`~qalma.evolution.leakage_rate`,
  :func:`~qalma.evolution.leakage_integrals`,
  :func:`~qalma.evolution.local_mean_field_drift`,
  :func:`~qalma.evolution.estimate_memory_time` and
  :func:`~qalma.evolution.sector_leakage_fraction`.
* All the solvers follow the same conventions (:mod:`qalma.evolution.common`):
  ``solver(ham, k0 | rho0, t_span, *, e_ops=None, store_states=None, ...)``;
  ``e_ops`` may be a dict, a list or a callable ``e_ops(t, rho)``; states are
  stored by default only without ``e_ops``; ``parameters`` contains
  ``"method"``, ``"system"`` and ``"states"`` (``"generator"``,
  ``"density"`` or ``"none"``), and ``stats`` contains ``"method"`` and
  ``"wall time"``. In consequence, ``qutip_me_solve`` and ``heisenberg_solve``
  take ``ham`` and ``t_span`` instead of ``H`` and ``tlist``,
  ``heisenberg_solve`` takes ``order`` instead of ``deep`` and no longer has
  ``options``, and the options and statistics of QuTiP are stored under
  ``parameters["qutip options"]`` and ``stats["qutip"]``.
* Removed the ``qalma.maxent`` command-line program and the modules
  ``qalma.evolution.hierarchical_basis`` and ``qalma.evolution.tools``, which
  duplicated :class:`~qalma.scalarprod.HierarchicalOperatorBasis` or were
  only used by that program.

Bug fixes
+++++++++

* ``remove_under_tolerance_terms`` (used by
  :func:`~qalma.scalarprod.trim_terms_by_tolerance`) always removed the
  smallest term, whatever its norm. Since the hierarchical bases of
  :func:`~qalma.evolution.update_basis` trim every level, each element lost
  a term, and the generator matrix of the restricted evolution differed from
  the Galerkin one by 10-30 % whenever the n-body projection acted.
* :func:`~qalma.evolution.update_basis` appended the mean-field reference
  generator to the basis even when it was already in its span (always the
  case for one-body generators). The Gram matrix became singular, and for
  deep bases the evolved coefficients were corrupted. The reference is now
  appended only if it is linearly independent.

Performance
+++++++++++

* Operator bases store the commutators of their elements with the
  generator, which they compute while being built. Appending elements to a
  basis (as :func:`~qalma.evolution.update_basis` does with the mean-field
  reference and the velocity) and :func:`~qalma.evolution.leakage_gram` reuse
  them, instead of computing each commutator up to three times. The
  restricted Max-Ent evolution of an XXZ chain with twelve sites is about
  30 % faster, with the same results.
* The n-body projection of dense blocks relative to a product state is
  computed directly, as a combination of partial reductions
  (inclusion-exclusion over the subsets of the support), instead of expanding
  the block into products of local operators: about 60 times faster for
  three- and four-site blocks.
* The fast covariance kernel caches the reference state on each overlap of
  supports, which was rebuilt for every pair of terms.
* :func:`~qalma.evolution.leakage_gram` computes the Gram matrix of the
  commutators once per pair (it is symmetric), reuses the matrix ``h`` stored
  in the basis, and caches the commutators of the basis elements, which
  :func:`~qalma.evolution.estimate_memory_time` and
  :func:`~qalma.evolution.sector_leakage_fraction` reuse when the generator
  lies in the span of the basis.
* Together, these changes make
  :func:`~qalma.evolution.adaptive_maxent_evolution` about 3.4 times faster
  for an extensive generator (XXZ chain with eight sites), with the same
  results.

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
* Removed the parallel routines (``qalma.parallel``, the parallel Gram
  matrix, and the ``QALMA_USE_PARALLEL`` and ``QALMA_MAX_WORKERS``
  settings). They never ran in parallel, because their number of workers was
  always 1; once fixed, benchmarks of the restricted Max-Ent evolution with
  up to 100 sites showed no speed-up: they parallelized tasks that are now
  cheap, while most of the time is spent in covariance products. With the
  option enabled, the Gram matrices of the covariance product were also
  computed without the caches of the fast kernel, about 40 % slower. To use
  several cores, run independent simulations as separate processes.
* ``qalma.settings.FIGURES_DIR`` was removed; it pointed outside the
  installed package.
* The code paths for QuTiP 4 were removed (QuTiP >= 5.1 is required).
* Fixed: building a system whose lattice has loops with a model whose sites
  do not define the operators of a loop term (for instance ``boson Hubbard``
  on ``square lattice with loop``) failed with an ``AssertionError``. The
  global operator with that loop term is now dropped, as global operators
  with bond terms already were.
