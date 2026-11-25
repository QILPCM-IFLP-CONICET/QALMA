Changelog
=========

All notable changes to QALMA are documented in this file. Version numbers
follow `Semantic Versioning <https://semver.org>`_.


Unreleased
------------------

Bug fixes
+++++++++

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
