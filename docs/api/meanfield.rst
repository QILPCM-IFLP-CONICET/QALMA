qalma.meanfield
===============

.. automodule:: qalma.meanfield
   :members:
   :show-inheritance:
   :no-index:

qalma.meanfield.meanfield
--------------------------

.. automodule:: qalma.meanfield.meanfield
   :members:
   :undoc-members:
   :show-inheritance:

qalma.meanfield.self\_consistent\_projections
----------------------------------------------

.. automodule:: qalma.meanfield.self_consistent_projections
   :members:
   :undoc-members:
   :show-inheritance:

qalma.meanfield.variational
----------------------------

.. automodule:: qalma.meanfield.variational
   :members:
   :undoc-members:
   :show-inheritance:

qalma.meanfield.gibbs\_partial\_trace
--------------------------------------

.. automodule:: qalma.meanfield.gibbs_partial_trace
   :members:
   :undoc-members:
   :show-inheritance:

qalma.meanfield.rpa
--------------------

The RPA reduction adds to the mean-field generator of a reduced Gibbs state
the contribution of the Gaussian fluctuations of the environment. It can be
requested with the ``rpa`` (and ``dynamic``) options of
:meth:`~qalma.operators.states.gibbs.GibbsDensityOperator.partial_trace`.

The class :class:`~qalma.meanfield.rpa.RPAGibbsDensityOperator` is a Gibbs
state whose partial traces (and therefore the expectation values computed
from them) include the RPA correction by default, and which stores the data
of the RPA that do not depend on the subsystem
(:class:`~qalma.meanfield.rpa.RPAStructure`) to reuse them in successive
reductions:

.. code-block:: python

    from qalma.meanfield.rpa import RPAGibbsDensityOperator

    state = RPAGibbsDensityOperator(k, dynamic=True)
    rho_a = state.partial_trace(sites)     # mean field + RPA
    rho_b = state.partial_trace(other)     # reuses state.rpa_structure

Besides the correction of the generator (``dynamic=False`` or ``True``), the
option ``hybrid=True`` keeps the subsystem exact and treats only the
environment in the RPA, solving a non perturbative problem in the coupling
(:func:`~qalma.meanfield.rpa.rpa_hybrid_state`); the full bosonization of the
system gives the entropies and negativities of the bosonic RPA state
(:meth:`~qalma.meanfield.rpa.RPAGibbsDensityOperator.rpa_entropy`,
:meth:`~qalma.meanfield.rpa.RPAGibbsDensityOperator.rpa_negativity`).

.. automodule:: qalma.meanfield.rpa
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
