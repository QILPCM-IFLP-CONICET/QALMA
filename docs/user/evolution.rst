Simulating Quantum Evolution
============================

.. contents::
   :depth: 2
   :local:

Overview
--------

QALMA implements the MaxEnt projected Schrödinger equation: instead of
evolving the full density matrix, it projects the equation of motion

.. math::

   \frac{dK}{dt} = -i [H, K]

onto a finite-dimensional operator basis :math:`\{Q_a\}`, yielding a system
of ODEs for the coefficients :math:`\phi_a(t)` in

.. math::

   K(t) = \sum_a \phi_a(t)\, Q_a.

The basis is chosen adaptively to control the approximation error.

Adaptive projected evolution
----------------------------

The main entry point is
:func:`~qalma.evolution.maxent_evol.adaptive_projected_evolution`::

    import numpy as np
    from qalma.evolution import adaptive_projected_evolution

    t_span = np.linspace(0, 5, 100)
    sim = adaptive_projected_evolution(
        ham,          # Hamiltonian operator
        k0,           # initial generator K(0)
        t_span,
        order=2,      # order of the hierarchical basis
        n_body=2,     # project onto 2-body sector
        tol=1e-3,     # accuracy goal
        e_ops={"E": ham, "Sz": Sz_total},
    )

The ``order`` parameter controls the depth of iterated commutators
:math:`[H, [H, \ldots [H, K]\ldots]]` used to build the basis. Higher order
captures faster dynamics at greater computational cost.

Accuracy control
~~~~~~~~~~~~~~~~

The ``tol`` parameter bounds the cumulative approximation error. When the
current basis can no longer represent the evolved state within ``tol``, the
basis is rebuilt automatically from the current :math:`K(t)`. The
``update_condition`` argument controls this behavior:

``"adaptive"`` (default)
    Rebuild when the error bound is saturated.
``"always"``
    Rebuild at every time step.
``"never"``
    Fix the basis from the start (fastest, least accurate).

Leakage-controlled evolution (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~qalma.evolution.leakage_controlled_evolution` controls the basis
with the *leakage* of the dynamics out of its span,
:math:`r_a = \pi_\perp(-i[H, Q_a])`, through the matrix
:math:`E_{ab} = (r_a, r_b)` (:func:`~qalma.evolution.leakage_gram`),
computed once per basis:

* the basis is rebuilt when the leakage rate
  :math:`\|\pi_\perp(-i[H, \tilde K])\| = \sqrt{\phi^T E \phi}` exceeds
  ``rate_factor`` times its value right after the last rebuild. That value,
  :math:`\|(1 - \pi^{(n)})[H, \tilde K]\|`, is the leakage out of the
  n-body sector, which no rebuild can remove;
* the basis and the mean-field reference are rebuilt when the local drift
  of the reference, :math:`\max_i S(\sigma'_i \| \sigma_{0,i})`, exceeds
  ``drift_tol`` (:func:`~qalma.evolution.local_mean_field_drift`).

Both criteria are intensive, so their thresholds do not need to be scaled
with the system size::

    from qalma.evolution import leakage_controlled_evolution

    sim = leakage_controlled_evolution(
        ham, k0, t_span, order=4, n_body=2,
        rate_factor=2.0, drift_tol=0.01, e_ops=observables,
    )
    sim.stats["errors"]                        # accumulated Galerkin estimate
    sim.stats["irreducible entropy estimate"]  # ~ S(rho || Pi_n rho)
    sim.stats["update reasons"]                # "initial", "leakage", "drift"

The restricted evolution approximates the Max-Ent projection
:math:`\Pi_n\rho(t)` of the exact state, not the state itself. The
difference :math:`S(\rho\|\Pi_n\rho) = S(\Pi_n\rho) - S(\rho)` is the
entropy produced by the projection, which no n-body generator can remove;
``"irreducible entropy estimate"`` is its second-order estimate
:math:`\tfrac12\|R\|^2`, with :math:`R = \int \pi_\perp(-i[H,\tilde K])\,dt`
the accumulated leakage. It is accurate at short times and overestimates
once :math:`S(\rho\|\Pi_n\rho)` saturates.

This function is experimental, and its interface may change.

Basis update strategies
-----------------------

Three basis-update callbacks are available:

:func:`~qalma.evolution.maxent_evol.update_basis`
    Default. Builds a hierarchical basis seeded from the full :math:`K(t)`,
    applying n-body projection at each level.

:func:`~qalma.evolution.maxent_evol.update_basis_light`
    Uses the mean-field approximation of :math:`K(t)` as the seed.
    Cheaper to build, may need more frequent updates.

:func:`~qalma.evolution.maxent_evol.update_basis_heavy`
    Two-pass: builds the full hierarchical basis first, then applies the
    n-body projection in a second pass reusing the Gram matrix.

Pass the chosen callback via ``basis_update_callback``::

    from qalma.evolution import adaptive_projected_evolution, update_basis_light

    sim = adaptive_projected_evolution(
        ham, k0, t_span, order=2, n_body=2,
        basis_update_callback=update_basis_light,
    )

Static projected evolution
--------------------------

For a fixed basis (no adaptive updates), use
:func:`~qalma.evolution.maxent_evol.projected_evolution`::

    from qalma.evolution import projected_evolution

    sim = projected_evolution(ham, k0, t_span, order=3, n_body=2)

The basis is built once from :math:`K(0)` and held fixed throughout.

The Simulation object
---------------------

Both functions return a :class:`~qalma.evolution.simulation.Simulation`
dataclass with the following attributes:

``sim.time_span``
    List of times at which the solution was recorded.
``sim.states``
    List of generator operators :math:`K(t_i)`.
``sim.expect_ops``
    Dict of expectation value arrays, one per observable passed to ``e_ops``.
``sim.stats``
    Dict with diagnostics: per-step errors, basis update times, occupation
    factors.
``sim.parameters``
    Dict of simulation parameters (order, n_body, tol, etc.).

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

:func:`~qalma.evolution.adaptive_projected_evolution` can write the partial
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
