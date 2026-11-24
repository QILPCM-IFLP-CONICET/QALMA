.. _correlations:

Quantum correlations
====================

.. contents::
   :depth: 2
   :local:

Overview
--------

The subpackage :mod:`qalma.correlations` computes entanglement and
correlation estimators between two subsystems :math:`A` and :math:`B` of a
state. Every estimator builds the dense reduced state :math:`\rho_{AB}` of
the two subsystems and evaluates a closed formula on it, so the cost is set
by the dimension of :math:`A\cup B`, not by the size of the whole system.

.. list-table::
   :header-rows: 1
   :widths: 32 20 48

   * - Function
     - Subsystems
     - Quantity
   * - :func:`~qalma.correlations.bipartite.negativity`
     - any :math:`A|B`
     - :math:`\mathcal{N}=(\|\rho_{AB}^{T_B}\|_1-1)/2`
   * - :func:`~qalma.correlations.bipartite.log_negativity`
     - any :math:`A|B`
     - :math:`E_N=\log_2\|\rho_{AB}^{T_B}\|_1`
   * - :func:`~qalma.correlations.two_qubit.concurrence`
     - two qubits
     - Wootters concurrence :math:`C`
   * - :func:`~qalma.correlations.two_qubit.entanglement_of_formation`
     - two qubits
     - :math:`E_F=h\big((1+\sqrt{1-C^2})/2\big)`
   * - :func:`~qalma.correlations.bipartite.entanglement_entropy`
     - :math:`A`
     - :math:`S(\rho_A)`
   * - :func:`~qalma.correlations.bipartite.mutual_information`
     - any :math:`A|B`
     - :math:`I(A{:}B)=S_A+S_B-S_{AB}`

Entropies are in bits by default (``base=2``). The negativity follows the
convention of :meth:`~qalma.gaussian.BosonicGaussianSystem.negativity`, so
exact values can be compared directly with
:func:`~qalma.meanfield.rpa.rpa_negativity`.

Usage
-----

The subsystems are site names, or iterables of site names. For a Heisenberg
ring at inverse temperature :math:`\beta` (recall that
:class:`~qalma.operators.states.gibbs.GibbsDensityOperator` takes the generator
:math:`K` of :math:`\rho\propto e^{-K}`, so :math:`K=\beta H`):

.. code-block:: python

   from qalma.correlations import concurrence, log_negativity, pairwise
   from qalma.model import build_spin_chain
   from qalma.operators.states import GibbsDensityOperator

   system = build_spin_chain(6)
   sites = sorted(system.sites)
   bonds = list(zip(sites, sites[1:] + sites[:1]))
   ham = sum(
       system.site_operator(f"S{a}@{i}") * system.site_operator(f"S{a}@{j}")
       for i, j in bonds
       for a in "xyz"
   )
   state = GibbsDensityOperator(2.0 * ham, system)

   concurrence(state, sites[0], sites[1])      # nearest neighbours
   log_negativity(state, sites[:3])            # half chain vs. the rest
   pairwise(state, "concurrence")              # every bond of the lattice
   pairwise(state, "negativity", pairs="all")  # every pair of sites

When ``subsystem_b`` is omitted in :func:`~qalma.correlations.bipartite.negativity`
and :func:`~qalma.correlations.bipartite.log_negativity`, :math:`B` is the
complement of :math:`A`. Sets of sites are sorted; lists and tuples keep
their order, which only matters for the layout of
:func:`~qalma.correlations.reduction.reduced_state`.

The same functions accept a :class:`qutip.Qobj` (a ket or a density
operator); the subsystems are then given by the positions of its tensor
factors:

.. code-block:: python

   import qutip
   from qalma.correlations import concurrence

   ghz = (qutip.tensor([qutip.basis(2, 0)] * 3)
          + qutip.tensor([qutip.basis(2, 1)] * 3)).unit()
   concurrence(ghz, 0, 1)   # 0: the reduced pair of a GHZ state is separable

Which estimator to use
----------------------

* The **negativity** and the **logarithmic negativity**
  :cite:p:`VidalWerner2002` are computable for any bipartition and any
  local dimension. A positive value certifies entanglement; a zero value
  does not exclude it, except for :math:`2\times2` and :math:`2\times3`
  systems, where the partial transpose criterion :cite:p:`Peres1996` is
  also sufficient.
* The **concurrence** and the **entanglement of formation**
  :cite:p:`Wootters1998` are only available for pairs of two-level sites.
  For spin-1 or larger sites use the negativity.
* The **entanglement entropy** measures the entanglement between :math:`A`
  and its complement only for pure global states.
* The **mutual information** counts quantum and classical correlations. It
  bounds every connected correlation function,
  :math:`\langle O_AO_B\rangle_c^2\le2\|O_A\|^2\|O_B\|^2\,I(A{:}B)\ln2`
  :cite:p:`Wolf2008`.

Reduction of large Gibbs states
-------------------------------

The estimators need the exact reduced state. A
:class:`~qalma.operators.states.gibbs.GibbsDensityOperator` is reduced exactly
when the system (or the support of its generator) has at most
``max_exact_sites`` sites, or when the generator does not couple the
subsystem to the rest (see
:func:`~qalma.meanfield.gibbs_partial_trace.gibbs_exact_partial_trace`).
Otherwise the estimators raise :class:`ValueError` instead of silently
falling back on the mean-field reduction, which in general underestimates
the correlations between the subsystems. There are two ways forward:

* raise ``max_exact_sites``, at the cost of building the dense state;
* pass ``allow_approximate=True`` to use the approximate reduction of
  :meth:`~qalma.operators.states.gibbs.GibbsDensityOperator.partial_trace`
  (mean field, or the RPA for
  :class:`~qalma.meanfield.rpa.RPAGibbsDensityOperator`). An
  :class:`~qalma.correlations.reduction.ApproximateReductionWarning` is emitted.

Product states (:class:`~qalma.operators.states.product.ProductDensityOperator`,
:class:`~qalma.operators.states.gibbs.GibbsProductDensityOperator`) give zero
without building any matrix.

Limits and caveats
------------------

* The dense reduced state is limited to
  :data:`~qalma.correlations.reduction.DEFAULT_MAX_DENSE_DIMENSION`. Pass
  ``max_dimension`` to change the limit for one call.
* **Fermions.** The partial transpose is the one of the qubit (hard-core
  boson) representation. Between fermionic modes it is not the fermionic
  partial transpose of :cite:t:`Shapourian2017`, so the resulting
  negativity is not a fermionic entanglement measure. A
  :class:`~qalma.model.FermionicSignWarning` is emitted when the subsystems
  contain fermionic sites.
* A physical site that the system represents by several qubits is expanded
  into its qubits, so ``concurrence`` of two such sites is rejected (their
  reduced state is not two qubits); use the negativity instead.
