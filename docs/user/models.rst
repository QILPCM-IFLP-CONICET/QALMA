Models and Hamiltonians
=======================

.. contents::
   :depth: 2
   :local:

Overview
--------

Once a :class:`~qalma.model.SystemDescriptor` is available, QALMA can
assemble many-body Hamiltonians and other global operators from the term
descriptors stored in the ALPS model definition.

Building the Hamiltonian
------------------------

Use :func:`~qalma.model.build_system` or call the model-building helpers
directly::

    from qalma import build_system, model_from_alps_xml, graph_from_alps_xml
    from qalma.model import build_operator

    model = model_from_alps_xml(name="spin", parms={"S": "1/2", "J": 1.0})
    graph = graph_from_alps_xml(name="open chain", parms={"L": 6})
    system = build_system(model, graph)

    ham = build_operator("Hamiltonian", system)

The returned object is a :class:`~qalma.operators.arithmetic.SumOperator`
— a linear combination of :class:`~qalma.operators.product.ProductOperator`
terms, one per bond or site term in the Hamiltonian definition.

Site and bond operators
-----------------------

Individual site and bond operators from the ALPS model can be built with
:func:`~qalma.model.build_operator`::

    Sz_total = build_operator("Sz", system)      # sum of local Sz
    Sz_0 = build_operator("Sz", system, site="site0")  # single-site

Bond operators require specifying the bond::

    bond_op = build_operator("Exchange", system, bond=("site0", "site1"))

Quantum numbers and symmetries
-------------------------------

The local basis at each site is labeled by quantum numbers defined in the
ALPS model (e.g. :math:`S^z` for spin models, particle number for bosonic
or fermionic models). These labels are stored in
``system.sites[site]["localstates"]`` as a list of tuples.

QALMA does not impose global symmetry sectors automatically, but the
:mod:`qalma.projections.symmetries` module provides projectors onto
fixed quantum number sectors::

    from qalma.projections.symmetries import symmetry_projector

    proj = symmetry_projector(system, {"Sz": 0})  # total Sz = 0 sector

.. seealso::

   :mod:`qalma.model`, :mod:`qalma.alpsmodels`, :mod:`qalma.projections`
