.. _rpa:

RPA reduction of Gibbs states
=============================

.. contents::
   :depth: 2
   :local:

Overview
--------

.. note::

   The RPA reduction is experimental: its interface may change in future
   releases.

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
couplings and small subsystems. The mathematical details are given in
:ref:`rpa-formulation`.

The RPA reductions follow the mean field plus RPA treatment of thermal and
ground-state entanglement of
:cite:t:`Matera2010RPA,Matera2011RPA,Canosa2010SPA`, which builds on the
standard formulation of the RPA in :cite:t:`RingSchuck1980`; the
bosonization of the whole system is the one of chapter 6 of
:cite:t:`MateraThesis`.

Three reductions are available:

``dynamic=False``
   *Static* correction of the generator (the default).
   It is the cheapest one.

``dynamic=True``
   *Dynamical* correction. The environment is bosonized in the eigenbasis
   of the local mean-field states, which is equivalent to summing all the
   Matsubara frequencies of the Hubbard-Stratonovich fluctuations
   :cite:p:`Stratonovich1957,Hubbard1959`.

``hybrid=True``
   The subsystem is kept exact and only the environment is treated in the
   RPA. It is non perturbative in the coupling between the subsystem and
   the environment, and it is intended for low temperatures.

Basic usage
-----------

The reductions are computed by
:class:`~qalma.meanfield.rpa.RPAGibbsDensityOperator`, a Gibbs state whose
:meth:`~qalma.meanfield.rpa.RPAGibbsDensityOperator.partial_trace` includes
the RPA. A plain :class:`~qalma.operators.states.gibbs.GibbsDensityOperator`
only provides the mean-field reduction.

.. code-block:: python

   from qalma import build_system
   from qalma.meanfield.rpa import RPAGibbsDensityOperator

   system = build_system("chain lattice", "spin", L=8)
   H = system.global_operator("Hamiltonian")
   Sz = system.global_operator("Sz")

   beta = 0.8
   k = (beta * (H - 0.5 * Sz)).simplify()
   state = RPAGibbsDensityOperator(k)

   sites = frozenset({"1[0]", "1[1]"})
   corr = (system.site_operator("Sz", "1[0]")
           * system.site_operator("Sz", "1[1]")).simplify()

   rho_mf = state.partial_trace(sites, rpa=False, max_exact_sites=0)
   rho_static = state.partial_trace(sites, max_exact_sites=0)
   rho_dynamic = state.partial_trace(sites, dynamic=True, max_exact_sites=0)

   print(rho_dynamic.expect(corr))

.. note::

   Reductions that involve at most ``max_exact_sites`` sites are computed
   exactly, and then the RPA is not used. The default is
   ``qalma.settings.MAXIMUM_GIBBS_EXACT_PARTIAL_TRACE`` (8 sites), which
   includes the chain above, so ``max_exact_sites=0`` is needed to see the
   effect of the approximation. The same option is available in
   :meth:`~qalma.operators.states.gibbs.GibbsDensityOperator.partial_trace`.

The approximate generator :math:`K_A'` itself is returned by
:meth:`~qalma.meanfield.rpa.RPAGibbsDensityOperator.reduced_generator`.

For this chain (:math:`\beta = 0.8`) the nearest-neighbour correlation
:math:`\langle S^z_0 S^z_1\rangle` is:

===============================  ========
Method                           Value
===============================  ========
Exact (QuTiP)                    -0.04906
Mean field                       -0.05177
RPA, static                      -0.05222
RPA, dynamic                     -0.04951
RPA, hybrid (``n_gauss=3``)      -0.04977
===============================  ========

This is a single example of a small system, not a benchmark: the static
correction does not improve on the mean field here, while the dynamical and
hybrid reductions do.

Reusing the RPA data
--------------------

The part of the RPA that does not depend on the subsystem (the mean-field
state, the coupling matrix, the local bases and bosonic modes) is stored by
:class:`~qalma.meanfield.rpa.RPAGibbsDensityOperator` in ``rpa_structure``,
so it is computed once when several subsystems are reduced:

.. code-block:: python

   state = RPAGibbsDensityOperator(k, dynamic=True)
   rho_a = state.partial_trace(frozenset({"1[0]", "1[1]"}), max_exact_sites=0)
   rho_b = state.partial_trace(frozenset({"1[4]", "1[5]"}), max_exact_sites=0)  # reuses state.rpa_structure

The default values of ``dynamic``, ``hybrid`` and ``hybrid_options`` are
those given to the constructor, and can be overridden in each call.

Hybrid reduction
----------------

The hybrid reduction diagonalizes exactly a Hamiltonian that contains the
subsystem and a small number of effective bosons that represent the
environment. Its cost grows quickly with the number of effective bosons and
of Fock states. The options are passed in ``hybrid_options`` (see
:func:`~qalma.meanfield.rpa.rpa_hybrid_state`):

.. code-block:: python

   rho = state.partial_trace(
       sites, hybrid=True, max_exact_sites=0,
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

These are quantities of the bosonic RPA state, not of the spin state. Two
kinds of modes are left out of it:

* The *static* modes (the diagonal operators of each site, and the pairs of
  degenerate local levels) are classical fluctuations: as bosons of vanishing
  frequency their entropy would diverge. They are always excluded, so that
  the entropy and the negativity describe only the quantum (interband)
  fluctuations around the mean-field state. The static modes are included
  in the reductions of a subsystem, which only need their static response.
* The *zero modes* (Goldstone modes of a broken continuous symmetry) are
  removed from the bosonic problem. Without them the Gaussian state is not
  a physical state of the local modes (its reduced covariance matrices
  violate the uncertainty relations), so the bosonic entropy and negativity
  raise a ``ValueError`` when there are zero modes. In that case the
  entanglement is computed from the reduced states of the spins, as
  described in :ref:`rpa-zero-modes`.

The bosonic state is a :class:`~qalma.gaussian.BosonicGaussianSystem`
(``state.rpa_structure.boson_system()``), whose modes are labelled by their
sites. The same class represents any Gaussian bosonic state, built from a
quadratic generator or from its matrix of contractions (see
:mod:`qalma.gaussian`).

Free energy
-----------

:meth:`~qalma.meanfield.rpa.RPAGibbsDensityOperator.rpa_free_energy` estimates
:math:`F=-\ln Z` (in units of the generator, :math:`K=\beta H`) as the free
energy of the mean field plus the Gaussian fluctuations around it
:cite:p:`MateraThesis,Canosa2010SPA`:

.. math::

    F_\mathrm{RPA} = F_\mathrm{MF}
    + \sum_k\ln\frac{\sinh(\Omega_k/2)}{\sinh(\Lambda_k/2)}
    + \frac12\ln\frac{\det(1+\chi^0 M)}{\det(1+\chi^0_\mathrm{dyn} M)}
    - \ln g .

The second term is the free energy of the bosonized system (normal-mode
frequencies :math:`\Omega_k`) minus that of the bosonized mean field (local
frequencies :math:`\Lambda_k`); the third is the Gaussian integral over the
static fluctuations that the bosons do not describe; and the last one restores
the discrete symmetries that the mean field breaks, summing over the :math:`g`
equivalent mean-field solutions::

    from functools import partial
    import qutip
    from qalma.meanfield import symmetry_breaking_mfa
    from qalma.meanfield.rpa import RPAGibbsDensityOperator
    from qalma.projections import project_parity_like

    # sigma^x on each site generates the spin-flip parity
    system.spec["model"].site_basis["0"]["operators"]["ParityX"] = qutip.sigmax()
    parity = (partial(project_parity_like, op_name="ParityX"),)

    meanfield = symmetry_breaking_mfa(k, system, numfields=6, n_attempts=3)
    state = RPAGibbsDensityOperator(k, system, meanfield=meanfield,
                                    symmetry_projections=parity)
    estimate = state.rpa_free_energy()
    estimate.total        # F_RPA, also given by state.free_energy
    estimate.mean_field   # F_MF, an upper bound of F
    estimate.broken       # ['parity generated by ParityX'] in the ordered phase

For an :class:`~qalma.meanfield.rpa.RPAGibbsDensityOperator`, the property
``free_energy`` returns this estimate, instead of normalizing the state
exactly as :class:`~qalma.operators.states.gibbs.GibbsDensityOperator` does;
the exact value, for systems small enough, is
``GibbsDensityOperator(k).free_energy``. The estimate refers to the generator
the state was built with, also after the state has been normalized.

The symmetries are those declared in ``symmetry_projections``. Whether the
mean-field state breaks each of them is checked site by site: a parity-like
symmetry (the local operator must be an involution) or time reversal adds
:math:`-\ln 2` when broken. If the mean field breaks a symmetry that is not
declared, the estimate misses the corresponding term. Projections that cannot
be classified issue a warning.

A broken continuous symmetry, declared with
:func:`~qalma.projections.project_conserved_quantity` for a conserved
quantity :math:`P=\sum_iq_i`, produces a zero mode of the RPA. The zero mode
is removed from the dynamic and static terms, and replaced by the number of
distinguishable mean-field states along the orbit :math:`e^{i\theta P}`,
estimated by its length in the Bures metric:

.. math::

    \Delta F_\mathrm{sym} = -\ln\max\!\left(1,\sqrt{2\pi F_Q/4}\right),\qquad
    F_Q=\sum_iF_Q(\sigma_i,q_i),

where :math:`F_Q` is the quantum Fisher information of the mean-field state
for :math:`P` (for pure local states :math:`F_Q/4` is the variance of
:math:`P`, and it vanishes for a state that commutes with :math:`P`), for
local charges with integer-spaced spectra (period :math:`2\pi`)::

    from qalma.projections import project_conserved_quantity

    u1 = (partial(project_conserved_quantity, op_name="Sz"),)
    state = RPAGibbsDensityOperator(k, system, meanfield=meanfield,
                                    symmetry_projections=u1)
    state.free_energy

The Fisher information makes the term robust: a mean field that breaks the
symmetry only slightly (for instance, a spurious solution of the
symmetry-breaking search) adds almost nothing. This classical estimate holds
while many charge sectors are populated; in finite systems at low
temperatures, where the charge sectors are resolved, it overestimates the
correction. Zero modes without a declared broken continuous symmetry raise a
``ValueError``.

The result also reports the stability margin of the RPA, the smallest
eigenvalue of :math:`1+\chi^0M` without the zero modes
(``estimate.stability``). Close to an instability of the mean field the
Gaussian fluctuations diverge and the estimate overcorrects; a warning is
issued below ``STABILITY_WARNING = 0.25``. In the tests, margins above 0.4
gave errors several times smaller than the mean field, and margins below 0.2
gave large overcorrections, notably in one-dimensional chains near their
mean-field transitions.

On an Ising chain in a transverse field with :math:`L=8`, the error of the
estimate is :math:`5\times10^{-4}` at high temperature, against
:math:`3\times10^{-2}` for the mean field, and about 0.01 deep in the ordered
phase, against 0.7. As any Gaussian approximation, it slightly overestimates
the correlation energy at low temperatures.

Correlations of the environment
-------------------------------

The static and dynamical corrections couple the operators of the subsystem
to the correlations of the environment,
:math:`K_{A,RPA} \sim \sum_{\mu\nu} \langle Q^E_\mu Q^E_\nu\rangle P^A_\mu
P^A_\nu`. The option ``environment`` selects where these correlations come
from:

* ``"isolated"`` (default): the environment is bosonized with the subsystem
  removed. This needs a Bogoliubov transformation for each subsystem.
* ``"global"``: the correlations are those of the Gaussian state of the
  bosonization of the whole system, as in chapter 6 of :cite:p:`MateraThesis`.
  The Bogoliubov transformation is computed once, stored in
  ``state.rpa_structure``, and shared by all the subsystems.

.. code-block:: python

   state = RPAGibbsDensityOperator(k, dynamic=True, environment="global")
   rho = state.partial_trace(sites, max_exact_sites=0)

In both cases :math:`K_{A,RPA}` is Hermitian, so the reduced state
:math:`e^{-K_{A,MF}-K_{A,RPA}}` is positive. The global correlations include
the fluctuations of the subsystem itself fed back through the environment,
so they are somewhat less accurate. In the normal phase of the fully
connected models with :math:`n = 6`, the trace distance of the dynamical
pair state to the exact one is :math:`\sim 2\times 10^{-3}` with the
global environment and :math:`\sim 10^{-4}` with the isolated one, and the
difference grows close to an instability (:math:`1.4\times 10^{-2}` against
:math:`9\times 10^{-4}`); the concurrences differ by a few percent at most.
In return, the cost per subsystem grows much more slowly with the size of the
system: for the dynamical reduction of a pair in a fully connected chain,
about 20 ms (global) against 40 ms (isolated) with 80 spins, and 50 ms against
200 ms with 160 spins. The hybrid reduction always uses the isolated
environment.

Bosonized reduced states
------------------------

The Gaussian state of the full bosonization also gives the reduced states of
one or two sites, mapping the local operators to the bosons with their
quadratic (population) terms: creating a boson in the pair of local levels
:math:`(i, j)` moves population from :math:`i` to :math:`j`. The averages of
products of operators of two sites then follow from Wick's theorem:

.. code-block:: python

   rho_pair = state.bosonized_state(frozenset({"1[0]", "1[1]"}))

This is the bosonization of chapter 6 of :cite:t:`MateraThesis`. Its
one-body averages and the correlations of the linear parts coincide with
those obtained deriving the RPA partition function with respect to the
couplings, as in the mean field plus RPA of :cite:t:`Canosa2010SPA`;
the Gaussian state adds terms of higher order in :math:`1/n` (the products of
the population corrections and the Wick contraction of the quadratic parts),
which in the fully connected models bring the correlations closer to the
exact ones. Without the quadratic terms, the correlations of the diagonal
operators would keep their mean-field values.

The bosonization of the whole system is computed once (it is stored in
``state.rpa_structure``), so each pair costs a few contractions: in a fully
connected chain of 20 spins, about 0.5 ms per pair, against about 10 ms for
a dynamical reduction. In return, the subsystem is also treated in the
Gaussian approximation, so the dynamical reduction, which keeps it exact, is
more accurate (in the normal phase of the fully connected models with
:math:`n = 6`, the trace distance to the exact pair state is
:math:`\sim 5\times 10^{-3}` for the bosonized state and
:math:`\sim 10^{-4}` for the dynamical one). Other limitations:

* The matrices are Hermitian and normalized, but not necessarily positive:
  the truncation leaves negative eigenvalues of the order of the neglected
  terms (:math:`O(1/n^2)` in the fully connected models).
* Close to an instability, the occupation of the soft modes becomes of
  order one and the population corrections exceed the populations
  themselves; the result is then meaningless.
* As in the entropies and negativities, the static modes are left out, and
  states with zero modes raise a ``ValueError``.

.. _rpa-formulation:

Formulation
-----------

Given :math:`\rho = e^{-K}/Z` and a subsystem :math:`A`, the generator is
split as :math:`K = K_A + K_E + W_{AE}` and the reduced generator is
approximated by :math:`K_A' = K_{A,\mathrm{MF}} + K_{A,\mathrm{RPA}}`.
Here :math:`K_{A,\mathrm{MF}}` is the mean-field projection of the boundary terms
(see :mod:`qalma.meanfield.gibbs_partial_trace`), and :math:`K_{A,RPA}`
accounts for the Gaussian fluctuations of the environment around its
mean-field state. Let :math:`e_\mu` be a Hilbert-Schmidt orthonormal basis
of local operators and
:math:`\delta e_\mu = e_\mu - \langle e_\mu\rangle_{MF}` the centred basis.
The centred two-body coupling matrix :math:`M` is defined by
:math:`K_2 = \frac12\sum_{\mu\nu} M_{\mu\nu}\,\delta e_\mu\delta e_\nu`.

Static correction
^^^^^^^^^^^^^^^^^

.. math::

    K_{A,RPA} = -\frac12 \sum_{\mu\nu} \Sigma_{\mu\nu}\,
                \delta e_\mu \delta e_\nu,
    \qquad
    \Sigma = M_{AE}\,\chi_E\,M_{EA},
    \qquad
    \chi_E = (1 + \chi^0 M_{EE})^{-1}\chi^0 ,

where :math:`\chi^0` is the connected Kubo-Mori susceptibility of the
single-site mean-field states.

Dynamical correction (bosonization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The static form replaces the imaginary-time evolved operators :math:`\delta
e_\mu(s) = e^{sK_{A,MF}}\delta e_\mu e^{-sK_{A,MF}}` by their value at
:math:`s=0`. To keep the full imaginary-time kernel
:math:`\Sigma_{\mu\nu}(\tau) = \langle B_\mu(\tau) B_\nu\rangle`, with
:math:`B_\mu = \sum_{\nu\in E} M_{\mu\nu}\delta e_\nu`, the local algebra of
every site of the environment is bosonized in the eigenbasis of its
mean-field state :math:`\sigma = \sum_i p_i |i\rangle\langle i|`
(populations sorted as :math:`p_i \ge p_j` for :math:`i<j`):

.. math::

    |i\rangle\langle j| \to \sqrt{p_i - p_j}\; a_{ij},
    \qquad
    |j\rangle\langle i| \to \sqrt{p_i - p_j}\; a^\dagger_{ij},

so that :math:`[|i\rangle\langle j|, |j\rangle\langle i|]` has expectation
value :math:`p_i-p_j`, as required by :math:`[a,a^\dagger]=1`. Keeping only
the part of :math:`\delta e_\mu` that is linear in the bosons,

.. math::

    \delta e_\mu \simeq \sum_k \left(g_{\mu k} a_k + g^*_{\mu k}
    a_k^\dagger\right),
    \qquad
    g_{\mu,(ij)} = \sqrt{p_i-p_j}\,(e_\mu)_{ij},

the environment becomes a quadratic bosonic Hamiltonian
:math:`H_b = \sum_k \Delta_k a_k^\dagger a_k + \frac12\sum M_{\mu\nu}
\delta e_\mu\delta e_\nu` with :math:`\Delta_k = \ln(p_i/p_j)`, which is
diagonalized by a Bogoliubov transformation. This is equivalent to
integrating the Hubbard-Stratonovich fields :cite:p:`Stratonovich1957,Hubbard1959`
analytically and summing over
all Matsubara frequencies: with normal modes :math:`\Psi_m` of signed
frequency :math:`\lambda_m` and :math:`B_\mu = \sum_m c_{\mu m}\Psi_m`,

.. math::

    \Sigma_{\mu\nu}(\tau) = \sum_m c_{\mu m} c^*_{\nu m}\,
    \frac{\eta_m\, e^{-\lambda_m \tau}}{1 - e^{-\lambda_m}},
    \qquad \eta_m = \operatorname{sign}\lambda_m .

The diagonal operators :math:`|i\rangle\langle i|` are bilinear in the
bosons, so they are not part of the linear truncation. Their static
(intraband) response, :math:`\sum_i p_i (e_\mu)_{ii}(e_\nu)_{ii} -
\langle e_\mu\rangle\langle e_\nu\rangle`, and the one of pairs with
:math:`p_i = p_j`, are time independent. They are included as bosons of
vanishing frequency ``static_gap`` whose couplings are scaled so that
:math:`\chi(\omega=0)` is exact.

With :math:`\Sigma(\tau)` at hand, the second-order contribution to the
reduced state is

.. math::

    Y = \int_{0<s'<s<1} ds\,ds'\; \Sigma_{\mu\nu}(s-s')\,
        e^{-K_{A,MF}}\,\delta e_\mu(s)\,\delta e_\nu(s') ,

so that :math:`e^{-K_A'} \simeq e^{-K_{A,MF}} + Y` at second order in the
coupling. In the eigenbasis of :math:`K_{A,MF}` (eigenvalues
:math:`\epsilon_m`), each term of :math:`Y_{ij}` goes through an
intermediate level :math:`k`, and the terms are of two kinds:

* *Secular* terms, with :math:`\epsilon_k \ge \min(\epsilon_i,
  \epsilon_j)`. They contain the shifts of the levels, which grow with
  :math:`\beta`, and must be exponentiated. They give a correction of the
  generator, inverting the derivative of the matrix exponential
  (Kubo-Mori): :math:`\delta K_{ij} = -Y^{\rm sec}_{ij}/w_{ij}`, with
  :math:`w_{ij} = (e^{-\epsilon_j}-e^{-\epsilon_i})/(\epsilon_i-\epsilon_j)`.
* *Admixture* terms, with :math:`\epsilon_k < \min(\epsilon_i,
  \epsilon_j)`. They are the populations that the coupling to the
  environment transfers from the level :math:`k` to higher levels. At low
  temperature they are not small compared with :math:`e^{-\epsilon_i}`, so
  they cannot be written as a small correction of the generator; they are
  added to the density, with the weight of :math:`k` dressed by its
  secular shift.

The reduced generator is then
:math:`K_A' = -\log[e^{-(K_{A,MF}+\delta K)} + \sum_k e^{-\epsilon'_k}
R_k]`, with :math:`R_k = e^{\epsilon_k} Y^{\rm adm, k}` and
:math:`\epsilon'_k = \epsilon_k + \delta K_{kk}`. Both forms coincide at
second order. At high temperature the admixture is negligible and the
correction is the first-order (Kubo-Mori) correction of the generator; at
low temperature the admixture gives the populations of the excited states
of :math:`A` due to its entanglement with the environment, so that the
zero-temperature limit is reached. Correcting the generator at first order
in all the terms instead makes :math:`\delta K` diverge as
:math:`e^{\epsilon_m}` at low temperature, while resumming all of them in
the logarithm compresses the level shifts and drives the result back to
mean field as :math:`T \to 0`. In the limit where :math:`\delta e_\mu`
commutes with :math:`K_{A,MF}` the static expression is recovered.

The integral over :math:`u` is computed with a Gauss-Legendre quadrature of
``n_tau`` nodes, which are doubled until the correction converges: at low
temperature :math:`\Sigma(\tau)` varies on scales :math:`1/\omega` much
shorter than the imaginary-time interval.

Hybrid reduction
^^^^^^^^^^^^^^^^

Instead of correcting the generator at first order, the environment is
bosonized with the subsystem removed, its normal modes are compressed to a
few effective bosons (block Lanczos, a Gauss quadrature of the spectral
density), the modes with large thermal occupation are taken as static
classical Gaussian fields, and the resulting hybrid Hamiltonian (subsystem
exact, truncated bosons) is diagonalized exactly
(:func:`~qalma.meanfield.rpa.rpa_hybrid_state`). It is non perturbative in
the coupling between the subsystem and the environment, so it includes the
back-reaction of the subsystem, and it is valid at low temperatures, where
the generator corrections fail. The symmetry restoration is the one of the
Gibbs state (``symmetry_projections``), which for the parity is the mixture
of the two broken-symmetry branches.

.. _rpa-zero-modes:

Zero modes
^^^^^^^^^^

If the mean-field state breaks a continuous symmetry, the bosonic
Hamiltonian has zero modes (Goldstone modes): the Hamiltonian does not
depend on a collective coordinate :math:`Q` (the orientation of the order
parameter), and the conjugate charge :math:`P` is conserved. They are
detected and removed (see :func:`qalma.gaussian.bogoliubov_reduced`): the
orientation is fixed, and the symmetry has to be restored averaging over its
orbit, with the ``symmetry_projections`` of the state (for instance,
``project_conserved_quantity`` for a U(1) symmetry). The thermal
fluctuations of the charge :math:`P` are included in the hybrid reduction as
a static Gaussian field that couples to the subsystem and drives the bath.
A mean-field state that breaks the symmetry in the subsystem pins the
orientation of the environment, so that in many cases the environment has
only a soft mode. :func:`~qalma.meanfield.rpa.environment.rpa_zero_modes`
counts the zero modes.

This is the treatment of the mean field plus RPA of :cite:t:`Canosa2010SPA`
for a broken continuous symmetry: the Gaussian (RPA) approximation is applied only to the intrinsic
variables, the orientation is integrated exactly, and the conserved charge
has Gaussian thermal fluctuations. The RPA correction stays finite when the
frequency of the zero mode vanishes. The three ingredients are combined in
the hybrid reduction with the ``symmetry_projections`` of the state, whose
reduced states of the spins give the entanglement measures:

.. code-block:: python

   from functools import partial

   from qalma.projections.symmetries import project_conserved_quantity

   state = RPAGibbsDensityOperator(
       k, symmetry_projections=[partial(project_conserved_quantity, op_name="Sz")]
   )
   rho = state.partial_trace(
       sites, hybrid=True, max_exact_sites=0, hybrid_options={"n_gauss": 3}
   )

On a subsystem, the average over the orbit of the symmetry is the average
over the rotations of the subsystem, which is what the projection computes.
The full bosonization has no orientation integral: removing the zero mode
fixes the orientation sharply, and the resulting Gaussian state is not a
physical state of the local modes. Its entropies and negativities are
therefore not defined with zero modes.

Full bosonization
^^^^^^^^^^^^^^^^^

If the subsystem is also bosonized, the whole system is a quadratic bosonic
Hamiltonian whose thermal state is Gaussian (a
:class:`~qalma.gaussian.BosonicGaussianSystem`, see
:func:`~qalma.meanfield.rpa.rpa_structure.build_rpa_boson_system`). The
reduced covariance matrix of any subset of sites gives the entropy and the
negativity of the bosonic RPA state (see
:func:`~qalma.meanfield.rpa.functions.rpa_entanglement_entropy` and
:func:`~qalma.meanfield.rpa.functions.rpa_negativity`), as in the RPA of
:cite:t:`Matera2010RPA`.
Only the bosons with a non-vanishing frequency are included: the static
modes are classical fluctuations and are left out, so these quantities
describe the quantum fluctuations around the mean-field state.
The populations of the local mean-field states are obtained from the local
mean-field generators, so that the zero temperature limit can be reached.

Limitations
-----------

* Only the Gaussian (linear) truncation of the bosonization is implemented.
  The quartic terms (hopping and pairing contributions in a Wick
  decoupling) are neglected. They are controlled only at low temperatures
  with not too strong couplings, or at high temperatures. At intermediate
  temperatures the thermal occupations of the bosonic modes become large
  and the bosons explore regions of the Fock space that have no counterpart
  in the local Hilbert space, so the results should be regarded as
  qualitative.
* The static correction is perturbative and is not reliable at low
  temperatures. The dynamical correction reaches the zero-temperature limit
  (in the fully connected models with :math:`n = 6` sites it reproduces the
  pair entanglement of the ground state, provided the broken symmetries are
  restored), but, as any Gaussian approximation, it fails close to the
  critical temperature of the mean field, where it can predict spurious
  entanglement.
* If the bosonic Hamiltonian is not positive definite, the RPA is unstable
  and a ``ValueError`` is raised.
* If the mean-field state breaks a continuous symmetry, the zero modes are
  removed and the symmetry has to be restored with the
  ``symmetry_projections`` of the state (for instance
  ``project_conserved_quantity`` for a U(1) symmetry). The entropies and
  negativities of the full bosonization are then not defined; the
  entanglement is computed from the hybrid reduced states (see
  :ref:`rpa-zero-modes`).
* The entropies and negativities of the full bosonization leave out the
  static (classical) modes.

See also the notebook :doc:`/examples/example_rpa`, :doc:`meanfield_variational`
and the API reference :doc:`/api/meanfield`.
