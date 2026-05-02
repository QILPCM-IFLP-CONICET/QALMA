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

:class:`~qalma.evolution.simulation.Simulation` supports HDF5 serialization::

    sim.save_hdf5("my_simulation.h5")

    from qalma.evolution.simulation import Simulation
    sim2 = Simulation.load("my_simulation.h5")

States are stored compressed and loaded on demand via the
:class:`~qalma.evolution.simulation.SimulationHDF5` interface, avoiding
memory saturation for long simulations.

.. seealso::

   :doc:`/api/evolution`, :doc:`/api/scalarprod`
