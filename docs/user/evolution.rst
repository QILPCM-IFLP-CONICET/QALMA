Simulating Quantum Evolution
============================

.. contents::
   :depth: 2
   :local:

Overview
--------

QALMA implements the restricted Max-Ent evolution
:cite:p:`Balian1986,PerezMatera2024`, based on the maximum-entropy
principle :cite:p:`Jaynes1957a,Jaynes1957b`: instead of evolving the
full density matrix :math:`\rho(t)\propto e^{-K(t)}`, it projects the equation
of motion of the generator

.. math::

   \frac{dK}{dt} = -i [H, K]

onto a finite-dimensional operator basis :math:`\{Q_a\}`, yielding a system
of ODEs for the coefficients :math:`\phi_a(t)` in

.. math::

   K(t) = \sum_a \phi_a(t)\, Q_a.

The basis is built from :math:`K` by iterated commutators with :math:`H`,
projected onto the sector of operators of at most ``n_body`` sites, and is
rebuilt along the evolution. The hierarchical basis is a Krylov space on the
operator side, analogous to the recursion method
:cite:p:`ViswanathMuller1994,Parker2019`. The scalar product is the covariance
product proposed in :cite:t:`PerezMatera2024` as a computable substitute for
the Kubo-Mori-Bogoliubov one, evaluated here relative to an adaptively
recomputed mean-field reference state.

The restricted evolution approximates the instantaneous Max-Ent projection
:math:`\Pi_n\rho(t)` of the exact state onto the states with ``n_body``
generators. Its distance to the exact state,
:math:`S(\rho\|\Pi_n\rho) = S(\Pi_n\rho)-S(\rho)`, is an entropy production
that no ``n_body`` ansatz can remove. :math:`\Pi_n\rho` reproduces all the
expectation values of at most ``n_body`` sites, so the error of local
observables and pair correlations is entirely due to the method.

Adaptive Max-Ent evolution
--------------------------

The main entry point is :func:`~qalma.evolution.adaptive_maxent_evolution`::

    import numpy as np
    from qalma.evolution import adaptive_maxent_evolution

    t_span = np.linspace(0, 5, 100)
    sim = adaptive_maxent_evolution(
        ham,          # Hamiltonian
        k0,           # initial generator K(0)
        t_span,
        order=4,      # depth of the hierarchical basis
        n_body=2,     # many-body sector of the basis
        e_ops={"E": ham, "Sz": Sz_total},
    )

``order`` controls the depth of the iterated commutators
:math:`[H, [H, \ldots [H, K]\ldots]]`; ``n_body`` the size of the sector, and
with it the irreducible error and the cost.

Memory closure
~~~~~~~~~~~~~~

The first-order restricted evolution, :math:`\dot\phi = A\phi`, discards the
feedback of the part of the dynamics that leaves the span of the basis. It is
almost isentropic, and cannot follow the entropy production of
:math:`\Pi_n\rho(t)`. By default the evolution includes that feedback as a
memory term with an exponentially decaying kernel, in the spirit of the
Mori-Zwanzig projection formalism
:cite:p:`Nakajima1958,Zwanzig1961,Mori1965`,

.. math::

    \dot\phi = A\phi + m, \qquad
    \dot m = -G^{+}E\,\phi - m/\tau ,

where :math:`E` is the leakage Gram matrix
(:func:`~qalma.evolution.leakage_gram`). For :math:`\tau\to\infty` this is
the projection of :math:`\ddot K = L^2K` onto the basis, and for
:math:`\tau\to 0` the first-order evolution.

``memory_time="auto"`` (default)
    Estimate :math:`\tau=\sqrt{\pi/2}\,\|R\|/\|QLR\|` at each rebuild, with
    :math:`R=(1-\pi)L\tilde K` the leaked part of the dynamics
    (:func:`~qalma.evolution.estimate_memory_time`).
a number
    Fix the memory time.
``None``
    Disable the memory term (first-order evolution).

At each rebuild the previous velocity is added to the new basis and the
memory is chosen so that :math:`\dot\phi` is continuous: observables have no
kinks at the rebuilds (``continuous_velocity=True``).

Rebuild criteria
~~~~~~~~~~~~~~~~

The basis is rebuilt from the current :math:`\tilde K` when the leakage rate
:math:`\|\pi_\perp(-i[H,\tilde K])\|` exceeds ``max(rate_factor * floor,
rate_min)``, where ``floor`` is its value right after the last rebuild (the
leakage out of the n-body sector, which no rebuild can remove). The basis
and the mean-field reference are rebuilt when the local drift of the
reference, :math:`\max_i S(\sigma'_i\|\sigma_{0,i})`, exceeds ``drift_tol``
(:func:`~qalma.evolution.local_mean_field_drift`). Both criteria are
intensive, so they need not be rescaled with the system size.
``adaptive=False`` uses a single basis built from :math:`K(0)`.

A ``RuntimeWarning`` is issued when almost all of :math:`[H,\tilde K]` lies
outside the n-body sector (:func:`~qalma.evolution.sector_leakage_fraction`):
the basis cannot represent the dynamics, and the memory then produces
spurious entropy. Increase ``n_body`` in that case.

Diagnostics
~~~~~~~~~~~

Per output time, ``sim.stats`` has ``"errors"`` (accumulated Galerkin
estimate of :math:`\|K-\tilde K\|`), ``"irreducible entropy estimate"`` (a
second-order estimate of :math:`S(\rho\|\Pi_n\rho)`), ``"leakage rate"`` and
``"local drift"``. Per rebuild, it has ``"basis update times"``,
``"update reasons"`` (``"initial"``, ``"leakage"`` or ``"drift"``),
``"leakage floors"``, ``"memory times"``, ``"sector leakage fraction"``,
``"velocity jump"`` and ``"basis time costs"``.

Other solvers
-------------

:func:`~qalma.evolution.series_evolution` ``(ham, k0, t_span, order)``
    Power series of :math:`K(t)` truncated at ``order``.
:func:`~qalma.evolution.heisenberg_solve` ``(ham, rho0, t_span, e_ops=..., order=2)``
    Heisenberg evolution of the observables in ``e_ops`` in hierarchical
    bases; it returns only expectation values.
:func:`~qalma.evolution.qutip_me_solve` ``(ham, rho0, t_span, c_ops=None)``
    Exact evolution of the density operator with ``qutip.mesolve``, also
    with collapse operators.

Common conventions and the Simulation object
--------------------------------------------

All the solvers return a :class:`~qalma.evolution.simulation.Simulation` and
follow the conventions of :mod:`qalma.evolution.common`:

* the signature is ``solver(ham, k0 | rho0, t_span, *, e_ops=None,
  store_states=None, ...)``, with every option given by keyword;
* ``e_ops`` is a dict of operators, a list of operators (stored with the keys
  ``0, 1, ...``), or a callable ``e_ops(t, rho)`` that receives the state;
* ``store_states=None`` stores the states only when ``e_ops`` is not given.

The attributes of the simulation are:

``sim.time_span``
    Times at which the solution was recorded.
``sim.states``
    The states: generators :math:`K(t_i)` for the Max-Ent and series solvers,
    density operators for ``qutip_me_solve``, nothing for
    ``heisenberg_solve``; ``sim.parameters["states"]`` says which
    (``"generator"``, ``"density"`` or ``"none"``).
``sim.expect_ops``
    Dict of expectation values, one list per observable in ``e_ops``.
``sim.stats``
    Dict with ``"method"``, ``"wall time"`` (in seconds) and the diagnostics
    of each method.
``sim.parameters``
    Dict with ``"method"``, ``"system"``, ``"states"`` and the options of the
    simulation.

Saving and loading results
--------------------------

:class:`~qalma.evolution.simulation.Simulation` objects are stored as HDF5
files::

    sim.save_hdf5("my_simulation.h5")

    from qalma.evolution import Simulation
    sim2 = Simulation.load_hdf5("my_simulation.h5")

``load_hdf5`` returns a
:class:`~qalma.evolution.simulation.SimulationHDF5`, which loads each state
when it is accessed, avoiding memory saturation for long simulations.
States can also be added to the file as they are computed::

    sim2.states.append(k_t)          # written to the file
    sim2.time_span.append(t)
    sim2.save_hdf5()                 # update parameters, stats, expect_ops, time_span

The files do not contain Python pickles: the system, the operators and the
states are stored as numbers, strings and arrays, so the files can be read
from other languages, inspected with ``h5dump``, and opened safely even if
they come from someone else. The layout is described in :doc:`hdf5_format`.

Some things cannot be stored that way:

* The ``symmetry_projections`` of a
  :class:`~qalma.operators.states.GibbsDensityOperator` are functions. The
  state is stored without them, with a
  :class:`qalma.io.IncompleteStateWarning`.
* Arbitrary Python objects in ``parameters``, ``stats`` or ``expect_ops``
  raise a ``TypeError``. Numbers, strings, arrays, operators, and lists,
  tuples and dicts of them are supported. ``save_hdf5(..., allow_pickle=True)``
  stores any other object as a pickle; the file then needs
  ``load_hdf5(..., allow_pickle=True)``.

:func:`~qalma.evolution.adaptive_maxent_evolution` can write the partial
result after each step, with ``checkpoint_file="run.h5"``.

Files written by older versions of QALMA
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HDF5 files written before QALMA 1.0 store the system and the states as
pickles. They are read-only, and have to be opened explicitly with::

    sim = Simulation.load_hdf5("old.h5", allow_pickle=True)

Loading a pickle can run arbitrary code, so do it only with files that you
trust. ``tools/pickle_to_hdf5.py`` in the repository converts those files,
and pickles of ``Simulation`` objects, to the current format; so does
``sim.save_hdf5("new.h5")``.

.. seealso::

   :doc:`hdf5_format`, :doc:`/api/evolution`, :doc:`/api/io`,
   :doc:`/api/scalarprod`
