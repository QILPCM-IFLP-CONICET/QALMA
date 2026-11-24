What is QALMA?
~~~~~~~~~~~~~~

QALMA (*Quantum Algebras for Lattice Models and Applications*) is a Python
library for representing and manipulating operators and states of quantum
lattice systems, and for approximating their equilibrium and dynamical
properties through maximum-entropy (Max-Ent) projections.

The project started as a way of translating models defined with the ALPS XML
libraries into QuTiP objects. That translation is still there, but it is now
just the entry point of a larger toolbox, which includes its own operator
algebra, its own representation of states, and its own solvers.

What QALMA provides
===================

Models and lattices
   Lattices and models are read from the ALPS XML libraries
   :cite:p:`ALPS2011`: spin, Heisenberg, Bose-Hubbard, Hubbard, t-J and
   Kondo-lattice models on chains, ladders and two-dimensional lattices, plus
   custom models. Fermionic operators are mapped to Pauli strings with the
   Jordan-Wigner transformation :cite:p:`JordanWigner1928,Lieb.1961`, so that
   signs are exact.

Algebraic operators and states
   Operators are not stored as matrices, but as sums of local and few-body
   terms (including Pauli strings and their sums), together with the algebra
   needed to add, multiply, commute, trace and simplify them. States keep
   the exponential form :math:`\rho \propto e^{-\mathbf{K}}` of Gibbs states
   unevaluated until a number is needed.

   For example, in an Ising chain in a transverse field,

   .. math::

      {\mathbf H}=\sum_{i=1}^{N} b\, \sigma_{x,i} - \sum_{i=1}^{N-1} J\, \sigma_{z,i}\sigma_{z,i+1},

   even if the Hamiltonian is stored as a sparse matrix, the number of
   non-vanishing elements scales exponentially with the size of the chain. In
   terms of the elements of the algebra, the same operator is described by a
   number of parameters that scales linearly with the size of the system.
   This kind of decomposition is also what makes it possible to implement
   semi-analytical methods, like mean-field approaches or spin-wave-like
   expansions.

Scalar products and projections
   Orthogonal projections onto sets of relevant operators are available for
   the Hilbert-Schmidt, Kubo-Mori-Bogoliubov and quantum covariance scalar
   products :cite:p:`Kubo1957,PetzToth1993,PerezMatera2024`, as well as
   projections onto the :math:`n`-body sector relative to product states.

Solvers
   On top of these layers, QALMA implements:

   * self-consistent and variational mean-field approximations for Gibbs
     states of large systems, with symmetry-breaking restarts and symmetry
     restoration :cite:p:`Bartolome2026`;
   * restricted Max-Ent time evolution
     :cite:p:`Balian1986,PerezMatera2024`, with error control and a memory
     closure;
   * reduced states of small subsystems with a random-phase-approximation
     (RPA) correction :cite:p:`Matera2010RPA,Canosa2010SPA`, with Gaussian
     bosonic states as a building block (experimental);
   * matrix product ground states of one-dimensional Hamiltonians, computed
     with a built-in two-site DMRG :cite:p:`White1992,Schollwock2011`
     (experimental);
   * series, Heisenberg-picture and exact (through QuTiP) evolutions, which
     share the same conventions.

Interoperability and storage
   Any QALMA operator or state can be converted to a QuTiP object
   :cite:p:`QuTiP2012,QuTiP2013,QuTiP5`, which gives access to the QuTiP
   solvers and is also used to produce the exact references used in the tests.
   Systems, operators, states and simulations are stored in a documented HDF5
   format that does not use Python pickles.

Most of this functionality implements methods already reported in the
literature, which are cited in the corresponding pages of this documentation.
Two algorithms are, to our knowledge, not available in other packages: the
self-consistent mean-field solver for generators with arbitrary :math:`n`-body
terms, and the Max-Ent restricted evolution in the covariance geometry relative
to an adaptively recomputed mean-field reference state. They are described in
the companion paper.

Relation to other tools
=======================

QALMA does not try to replace the tools it builds on, nor the many excellent
packages for other regimes: exact diagonalization
:cite:p:`QuSpin2017`, tensor networks :cite:p:`ITensor2022,TeNPy2018` or neural
quantum states :cite:p:`NetKet2019,NetKet3`. Its focus is on the
description of the state of a many-body system through a reduced set of
relevant observables, for which the operator algebra and the projections above
are the natural language.

ALPS
   The ALPS (Algorithms and Libraries for Physics Simulations) project is an
   open-source initiative that provides libraries and tools for simulating
   quantum many-body systems, including quantum Monte Carlo, exact
   diagonalization and DMRG codes :cite:p:`ALPS2011`. QALMA uses its model and
   lattice libraries. For more information, visit the ALPSCore website:
   https://alpscore.org (see :cite:t:`GAENKO2017235`).

QuTiP
   The Quantum Toolbox in Python is an open-source library for simulating the
   dynamics of open quantum systems, with tools to create and manipulate
   quantum objects, evolve them under unitary and non-unitary dynamics, and
   visualize the results :cite:p:`QuTiP2012,QuTiP2013,QuTiP5`. For more
   information, visit the QuTiP documentation: http://qutip.org and the GitHub
   repository: https://github.com/qutip/qutip

Where to go next
================

* :doc:`quickstart` explains how to install QALMA and run a first example.
* The :doc:`user guide <user/index>` describes systems, operators, states, and
  each of the solvers.
* The references cited in this documentation are collected in
  :ref:`references`.
