States and Density Operators
=============================

.. contents::
   :depth: 2
   :local:

Overview
--------

QALMA represents mixed quantum states as density operators :math:`\rho`
through a hierarchy of classes that exploit structure (product states,
Gibbs states) to avoid full matrix storage.

All density operator classes mix in
:class:`~qalma.operators.states.basic.DensityOperatorMixin`, which adds the
:meth:`~qalma.operators.states.basic.DensityOperatorMixin.expect` method for
computing expectation values.

Gibbs states
------------

The central state type in QALMA is the Gibbs density operator

.. math::

   \rho = \frac{e^{-K}}{\mathrm{Tr}(e^{-K})}

represented by :class:`~qalma.operators.states.gibbs.GibbsDensityOperator`.
The state is stored implicitly through its generator :math:`K`, and the
partition function is computed lazily on first access::

    from qalma.operators.states import GibbsDensityOperator

    rho = GibbsDensityOperator(k, system)
    rho.normalize()           # computes Tr(e^{-K}), shifts K
    F = rho.free_energy       # -log Z

Product Gibbs states
--------------------

When :math:`K` is a one-body operator :math:`K = \sum_i K_i`, the Gibbs
state factorizes:

.. math::

   \rho = \bigotimes_i \frac{e^{-K_i}}{\mathrm{Tr}(e^{-K_i})}

This is represented by
:class:`~qalma.operators.states.gibbs.GibbsProductDensityOperator`, which
stores one local ``qutip.Qobj`` per site and computes expectation values
efficiently using the product structure::

    from qalma.operators.states import GibbsProductDensityOperator

    sigma = GibbsProductDensityOperator(k_one_body, system)
    val = sigma.expect(observable)

Product density operators
-------------------------

:class:`~qalma.operators.states.product.ProductDensityOperator` stores an
explicit product state as a dict of local density matrices, one per site.
It is returned by :meth:`~qalma.operators.states.gibbs.GibbsProductDensityOperator.to_product_state`
and used internally as the mean-field reference state.

Expectation values
------------------

All state objects support :meth:`~qalma.operators.states.basic.DensityOperatorMixin.expect` for computing
:math:`\langle O \rangle_\rho = \mathrm{Tr}(\rho\, O)`::

    val = rho.expect(observable)               # single operator → complex
    vals = rho.expect([op1, op2, op3])         # list → list of complex
    vals = rho.expect({"energy": H, "Sz": Sz}) # dict → dict

For product states, expectation values of product operators are computed
site-by-site without constructing the full many-body matrix.

Partial traces and reductions
------------------------------

The :meth:`~qalma.operators.states.basic.DensityOperatorProtocol.partial_trace` method returns the reduced state on a subsystem::

    rho_sub = rho.partial_trace(frozenset({"site0", "site1"}))

For :class:`~qalma.operators.states.gibbs.GibbsDensityOperator`, the partial
trace uses the mean-field Gibbs approximation
(:doc:`/api/meanfield`), returning a
:class:`~qalma.operators.states.gibbs.GibbsProductDensityOperator`.

The ``reduce`` method computes the effective operator on a subsystem
weighted by a reference state:

.. math::

   O_{\text{eff}} = \mathrm{Tr}_{\bar{S}}(\rho_{\bar{S}}\, O)

where :math:`\bar{S}` is the complement of the kept sites.

.. seealso::

   :doc:`/api/states`, :doc:`/api/meanfield`
