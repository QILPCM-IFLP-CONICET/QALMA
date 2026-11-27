Changelog
=========

All notable changes to QALMA are documented in this file. Version numbers
follow `Semantic Versioning <https://semver.org>`_.


Unreleased
------------------

Benchmarks
++++++++++

* New ``make benchmark-dense`` target, with benchmarks of dense linear algebra
  (raw ``matmul``/``eigh``/``expm``, exact Gibbs states, RPA, symmetry-breaking
  mean field and DMRG). Each benchmark stores the active BLAS thread pools
  in ``extra_info``.

Bug fixes
+++++++++

* The RPA correction was silently skipped when the mean field of the
  environment makes the boundary terms project to zero (for instance, an
  exactly vanishing magnetization in the saturated paramagnetic phase of the
  transverse-field Ising model): ``gibbs_meanfield_generator`` reported
  ``has_boundary=False`` after discarding the projected terms, although the
  subsystem is still coupled to the environment through the fluctuations. The
  reduced state was then the mean-field one, and the result jumped to the RPA
  one as soon as the magnetization was above the tolerance (about
  ``1e-15``). ``has_boundary`` now refers to the terms of the generator before
  the projection.
* The RPA failed with ``KeyError`` (a site name) when a site of the structure
  had no local mean-field generator: all the terms of the generator that reach
  it project to zero with the mean-field state, as for the sites at the end of
  the support of an evolved generator in models with a symmetry that makes the
  mean values vanish (for instance, a local ``Sz`` perturbation of an XX
  chain). Its local state is maximally mixed, and that is the spectrum that
  :meth:`~qalma.meanfield.rpa.rpa_structure.RPAStructure.local_spectrum` now
  returns.
* The RPA failed with ``IndexError: too many indices for array`` when a site
  of the mean-field structure had an empty local basis, which happens when the
  generator reaches the site only through terms below the tolerance (for
  instance, the far end of a chain at the end of an evolution that started
  from a local perturbation). ``local_boson_modes`` now returns no bosons for
  a site without fluctuation operators.
* The local basis of quadratic forms (``orthonormal_hs_local_basis``, used by
  :func:`~qalma.operators.quadratic.build_quadratic_form_from_operator`, the
  variational mean field and the RPA) could contain non-Hermitian elements,
  which made ``QuadraticFormOperator`` fail with "Basis operator ... is not
  Hermitian" when evaluating expectation values of correlated Gibbs states
  with more than eight sites. Gram-Schmidt used the complex round-off of the
  Hilbert-Schmidt products, and normalizing nearly dependent residuals turned
  that round-off into spurious, non-Hermitian directions (61 basis elements
  for a chain of ten spins 1/2, which admits at most 30); in addition,
  ``Qobj.dag()`` trusts a possibly stale ``isherm`` flag. The basis is now
  built with dense arrays, with real Gram-Schmidt coefficients, Hermitian
  residuals and a relative tolerance for linear dependence.
* ``matrix_to_wolfram`` (and ``operator_to_wolfram`` for matrices) exported
  only the first row of the matrix.
* ``fetch_kubo_scalar_product`` and
  ``fetch_kubo_integral_representation_scalar_product`` failed for every
  input. In addition, the spectral version lost precision for almost
  degenerate eigenvalues, and the integral version used ``p1`` where it
  should have used ``p2`` in the integrand, so it did not compute the
  Kubo-Mori-Bogoliubov scalar product.
* ``draw_operator`` also drew a spurious line for operators acting on a single
  site.
* ``build_hermitian_basis`` failed with ``ValueError`` when the basis contained
  null operators. They are now filtered out.
* ``self_consistent_mf`` (and the variational mean field that uses it)
  logged spurious convergence failures and reported wrong diagnostics. The
  convergence criterion was an absolute tolerance on the free energy
  (``QALMA_TOLERANCE``), which cannot be met when ``|F|`` is large (the
  rounding noise is one unit in the last place, ``~1e-12`` for
  ``|F| ~ 5000``, as in a low temperature state of a few dozen sites): it is
  now relative, ``QALMA_TOLERANCE * max(1, |F|)``. The warning printed
  ``Last Delta S_rel= 0.0`` whenever ``max_steps`` was exhausted and counted
  one step less than the done ones; it now gives the number of steps and the
  last change of the free energy, and tells apart the case in which the free
  energy increases. ``max_steps=0`` raised ``UnboundLocalError`` and now
  returns the initial state.
* ``operator_to_wolfram`` exported the local and product operators of systems
  with more than 10 sites with the factors of the ``KroneckerProduct`` sorted
  alphabetically by site name (``1[10]`` before ``1[2]``), instead of in the
  order of the sites of the system.
* ``draw_operator`` failed with ``IndexError`` for operators that act on no
  site, like scalars.
* The RPA was wrong for states with a local gap ``beta * Delta`` above
  ``~690``, and became unstable (``Unstable RPA``) at low enough temperature.
  The local populations, ``exp(-beta * Delta)``, were clipped at ``1e-300`` and
  the gaps ``ln(p_i/p_j)`` that give the frequencies of the bosonic modes and
  the Kubo-Mori susceptibilities saturated at ``ln(1e300)``: the static
  susceptibility stopped decreasing as ``1/beta`` and ``1 + chi0 M`` lost its
  positivity. The gaps are now computed from the logarithms of the
  populations (``RPAStructure.local_log_populations``, and the optional
  ``log_populations`` of ``local_boson_modes`` and
  ``kubo_mori_susceptibility``). The adaptive imaginary-time quadrature of the
  dynamical RPA now starts with enough nodes to resolve the largest
  frequency, since two coarse quadratures can agree on missing the same
  sharp features of the kernel.
* The reduced state of the dynamical RPA drifted with the temperature even
  for a gapped state (``~7e-4`` between ``beta = 1000`` and ``2000`` in a
  3x3 ladder, growing as ``beta``). The populations that the coupling with the
  environment admixes into the excited levels were dressed with the diagonal
  of the secular generator, while its exponential uses its eigenvalues; the
  level repulsion that separates them is of fourth order in the coupling but
  grows as ``beta``. They are now dressed with the eigenvalues, and the
  low-temperature limit is reached (and stays) as the state becomes gapped.

Changes
+++++++

* ``orthogonalize_basis_cholesky`` raises ``numpy.linalg.LinAlgError`` if the
  operators are linearly dependent, as ``orthogonalize_basis_svd`` does. It
  used to return an empty list or a single operator.
* ``orthogonalize_basis_gs``, ``orthogonalize_basis_cholesky`` and
  ``orthogonalize_basis_svd`` are no longer exported by ``qalma.scalarprod``.
  They remain available in ``qalma.scalarprod.orthogonalize``.
* ``matrix_to_wolfram`` exports integers as WL integers, and complex scalars
  with ``I``.
* ``convex_fit_derivative`` returns the derivatives in the order of the input
  data, instead of sorted by ``x``.

Documentation
+++++++++++++

* New pages of the user guide: *Scalar products and operator bases*, and
  *Utilities: export, drawing and fits*.
* The docstrings of the Kubo-Mori-Bogoliubov and Hilbert-Schmidt scalar
  products, of ``orthogonalize_basis`` and the functions of its module, and of
  the functions of ``qalma.utils`` document their parameters, return values
  and exceptions. The docstring of the Gram-Schmidt orthogonalization
  described a check of the orthogonality and an ``AssertionError`` that it
  does not perform; ``tol`` is the threshold to discard null and dependent
  operators.


0.9.0
------------------

November 24, 2026.

First public pre-release. The interface of the modules marked as
*experimental* may change before the 1.0.0 release.

Core
++++

* Lattices and models are read from the ALPS XML libraries
  (``graph_from_alps_xml``, ``model_from_alps_xml``) and assembled with
  ``build_system``. Fermionic models (spinless fermions, Hubbard, t-J,
  Kondo lattice) are mapped to qubits with the Jordan-Wigner
  transformation.
* Operators are represented algebraically, as sums of local and few-body
  terms (``LocalOperator``, ``ProductOperator``, ``OneBodyOperator``,
  ``SumOperator``, ``BlockOperator``, ``QuadraticFormOperator``) and as
  Pauli strings and their linear combinations (``PauliStringOperator``,
  ``PauliSumOperator``, ``TensorProductOperator``). Arithmetic,
  commutators, traces and partial traces keep the most structured
  representation available.
* States keep the exponential form :math:`e^{-K}` unevaluated
  (``GibbsDensityOperator``, ``GibbsProductDensityOperator``), together with
  product and block density operators and mixtures.
* Any operator or state can be converted to QuTiP objects (``to_qutip``).

Scalar products and projections
+++++++++++++++++++++++++++++++

* Hilbert-Schmidt, Kubo-Mori-Bogoliubov and quantum covariance scalar
  products, with a fast kernel for the covariance product relative to a
  product state. Operator bases, Gram matrices and hierarchical bases.
* Orthogonal projections onto the :math:`n`-body sectors relative to a
  product state, Max-Ent projection, and projections onto symmetry sectors
  (:math:`\mathbb{Z}_2`, :math:`U(1)`, time reversal).

Evolution
+++++++++

* ``adaptive_maxent_evolution``: restricted Max-Ent evolution in the
  covariance geometry relative to an adaptively recomputed mean-field
  reference state, with leakage-based error control and an optional memory
  closure.
* Truncated power-series evolution, truncated Heisenberg evolution of
  observables, and exact evolution through QuTiP, all returning
  ``Simulation`` objects.

Mean field (experimental)
+++++++++++++++++++++++++

* Self-consistent and variational mean-field approximations for Gibbs
  states, with symmetry-breaking restarts and symmetry restoration.
* Reduced states of Gibbs states with an RPA correction
  (``qalma.meanfield.rpa``, ``RPAGibbsDensityOperator``) and Gaussian
  bosonic states (``qalma.gaussian``).

Matrix product states (experimental)
++++++++++++++++++++++++++++++++++++

* ``MPSGSOperator``: ground state of a Hamiltonian stored as a matrix
  product state, computed with a built-in two-site DMRG that needs only
  NumPy and SciPy. It assumes a non-degenerate ground state, and accepts
  an explicit ordering of the sites along the chain.

Input and output
++++++++++++++++

* Systems, operators, states and simulations are saved in a documented,
  versioned HDF5 format that does not use Python pickles, including
  ``MPSGSOperator``.

Packaging
+++++++++

* QALMA is distributed under the LGPL-3.0. The ALPS model and lattice
  libraries in ``qalma/lib`` keep the Boost Software License 1.0, whose text
  is installed with the package.
* Python 3.11 or newer is required (tested on 3.11 to 3.14). Runtime
  dependencies: ``numpy>=1.26``, ``scipy>=1.11``, ``qutip>=5.1``,
  ``h5py>=3.10`` and ``packaging``. ``matplotlib`` is optional
  (``pip install "qalma[plot]"``).
* ``qalma.__version__`` is the single source of the version number.
