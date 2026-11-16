.. _rpa:

RPA reduction of Gibbs states
=============================

.. contents::
   :depth: 2
   :local:

Overview
--------

For a Gibbs state :math:`\rho = e^{-K}/Z` of a large system, the reduced
state on a small subsystem :math:`A` cannot be computed exactly. QALMA
approximates it with a self-consistent mean field (the default of
:meth:`~qalma.operators.states.gibbs.GibbsDensityOperator.partial_trace`),
which treats the environment as a product state and neglects its
fluctuations.

The RPA reduction improves this approximation. Splitting
:math:`K = K_A + K_E + W_{AE}`, the reduced generator is approximated by

.. math::

   K_A' = K_{A,\mathrm{MF}} + K_{A,\mathrm{RPA}},

where :math:`K_{A,\mathrm{MF}}` is the mean-field generator and
:math:`K_{A,\mathrm{RPA}}` accounts for the Gaussian fluctuations of the
environment. Since :math:`K_A'` is Hermitian, :math:`e^{-K_A'}` is a
positive operator. The method is meant for Hamiltonians with quasi-local
couplings and small subsystems. The mathematical details are in the
documentation of :mod:`qalma.meanfield.rpa`.

Three reductions are available:

``dynamic=False``
   *Static* correction of the generator (the default of ``rpa=True``).
   It is the cheapest one.

``dynamic=True``
   *Dynamical* correction. The environment is bosonized in the eigenbasis
   of the local mean-field states, which is equivalent to summing all the
   Matsubara frequencies of the Hubbard-Stratonovich fluctuations.

``hybrid=True``
   The subsystem is kept exact and only the environment is treated in the
   RPA. It is non perturbative in the coupling between the subsystem and
   the environment, and it is intended for low temperatures.

Basic usage
-----------

The reductions are requested through the options of
:meth:`~qalma.operators.states.gibbs.GibbsDensityOperator.partial_trace`:

.. code-block:: python

   from qalma import build_system
   from qalma.operators.states.gibbs import GibbsDensityOperator

   system = build_system("chain lattice", "spin", L=8)
   H = system.global_operator("Hamiltonian")
   Sz = system.global_operator("Sz")

   beta = 0.8
   k = (beta * (H - 0.5 * Sz)).simplify()
   state = GibbsDensityOperator(k)

   sites = frozenset({"1[0]", "1[1]"})
   corr = (system.site_operator("Sz", "1[0]")
           * system.site_operator("Sz", "1[1]")).simplify()

   rho_mf = state.partial_trace(sites, exact=False)
   rho_static = state.partial_trace(sites, rpa=True, exact=False)
   rho_dynamic = state.partial_trace(sites, rpa=True, dynamic=True, exact=False)

   print(rho_dynamic.expect(corr))

.. note::

   By default (``exact=True``) the reduction of a small system is computed
   exactly and the RPA is not used. This is the case of the chain of 8
   sites above, so ``exact=False`` is needed to see the effect of the
   approximation.

For this chain (:math:`\beta = 0.8`) the nearest-neighbour correlation
:math:`\langle S^z_0 S^z_1\rangle` is:

===============================  ========
Method                           Value
===============================  ========
Exact (QuTiP)                    -0.04906
Mean field                       -0.05177
RPA, static                      -0.05222
RPA, dynamic                     -0.04940
RPA, hybrid (``n_gauss=3``)      -0.04977
===============================  ========

This is a single example of a small system, not a benchmark: the static
correction does not improve on the mean field here, while the dynamical and
hybrid reductions do.

Reusing the RPA data
--------------------

The part of the RPA that does not depend on the subsystem (the mean-field
state, the coupling matrix, the local bases and bosonic modes) is stored by
:class:`~qalma.meanfield.rpa.RPAGibbsDensityOperator`, so it is computed once
when several subsystems are reduced:

.. code-block:: python

   from qalma.meanfield.rpa import RPAGibbsDensityOperator

   state = RPAGibbsDensityOperator(k, dynamic=True)
   rho_a = state.partial_trace(frozenset({"1[0]", "1[1]"}), exact=False)
   rho_b = state.partial_trace(frozenset({"1[4]", "1[5]"}), exact=False)  # reuses state.rpa_structure

In an ``RPAGibbsDensityOperator`` the options ``rpa=True`` and the default
values of ``dynamic`` and ``hybrid`` are those given to the constructor.

Hybrid reduction
----------------

The hybrid reduction diagonalizes exactly a Hamiltonian that contains the
subsystem and a small number of effective bosons that represent the
environment. Its cost grows quickly with the number of effective bosons and
of Fock states. The options are passed in ``hybrid_options`` (see
:func:`~qalma.meanfield.rpa.rpa_hybrid_state`):

.. code-block:: python

   rho = state.partial_trace(
       sites, rpa=True, hybrid=True, exact=False,
       hybrid_options={"n_gauss": 3, "n_fock": 4},
   )

If the problem exceeds the limits ``max_dim`` or ``max_nodes``, a
``ValueError`` is raised and suggests which option to reduce. With the
default options this happens, for example, for the chain above at
:math:`\beta = 0.8` (``n_gauss=3`` is enough) and at :math:`\beta = 3`.

Entropies and negativities
--------------------------

If the subsystem is also bosonized, the whole system is a quadratic bosonic
Hamiltonian and its thermal state is Gaussian. The reduced covariance
matrix of any set of sites gives the entropy and the negativity of the
bosonic RPA state:

.. code-block:: python

   s = state.rpa_entropy(sites)
   neg = state.rpa_negativity(frozenset({"1[0]"}), frozenset({"1[1]"}))
   state.rpa_zero_modes(sites)      # number of zero modes

These are quantities of the bosonic RPA state, not of the spin state, and
they are not defined when the mean-field state breaks a continuous symmetry
(zero modes).

Limitations
-----------

* Only the Gaussian (linear) truncation of the bosonization is implemented.
  The quartic terms (hopping and pairing contributions in a Wick
  decoupling) are neglected. They are controlled only at low temperatures
  with not too strong couplings, or at high temperatures, so at
  intermediate temperatures the results should be regarded as qualitative.
* The generator corrections (``dynamic=False`` and ``True``) are
  perturbative and are not reliable at low temperatures. In the chain above
  at :math:`\beta = 3` both give worse values than the mean field, and the
  hybrid reduction is the one intended for that regime.
* If the bosonic Hamiltonian is not positive definite, the RPA is unstable
  and a ``ValueError`` is raised.
* If the mean-field state breaks a continuous symmetry, the zero modes are
  removed and the symmetry has to be restored with the
  ``symmetry_projections`` of the state (for instance
  ``project_conserved_quantity`` for a U(1) symmetry).

See also :doc:`meanfield_variational` and the API reference
:doc:`/api/meanfield`.
