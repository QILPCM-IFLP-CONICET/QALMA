Systems and Geometries
======================

.. contents::
   :depth: 2
   :local:

Overview
--------

A QALMA simulation starts by defining the lattice geometry and the local
Hilbert space at each site. These two ingredients together produce a
:class:`~qalma.model.SystemDescriptor`, which is the object passed to every
operator and evolution routine.

Defining a system
-----------------

The primary entry point is :func:`~qalma.model.build_system`::

    from qalma import build_system, model_from_alps_xml, graph_from_alps_xml

    model = model_from_alps_xml(name="spin", parms={"S": "1/2"})
    graph = graph_from_alps_xml(name="open chain", parms={"L": 4})
    system = build_system(model, graph)

:func:`~qalma.model.build_system` combines the local basis descriptors from
the model with the connectivity information from the graph and returns a
:class:`~qalma.model.SystemDescriptor`.

The SystemDescriptor
--------------------

:class:`~qalma.model.SystemDescriptor` stores all site-level information:

* ``system.sites`` — ordered dict of site names to their local data
  (dimension, identity operator, available local operators).
* ``system.dimensions`` — mapping from site name to local Hilbert space
  dimension :math:`d_i`.
* ``system.bonds`` — list of ``(site_a, site_b, bond_type)`` tuples
  describing the lattice connectivity.

You can inspect the system directly::

    print(system.sites)      # {'site0': {...}, 'site1': {...}, ...}
    print(system.dimensions) # {'site0': 2, 'site1': 2, ...}

Subsystems
----------

QALMA supports partial traces and reductions over subsystems. A subsystem
is created with::

    sub = system.subsystem(frozenset({"site0", "site1"}))

which returns a new :class:`~qalma.model.SystemDescriptor` containing only
the specified sites, preserving all local data.

Loading geometries and models from ALPS
---------------------------------------

QALMA ships with the standard ALPS model and lattice library. You can
list available models and geometries::

    from qalma import list_models_in_alps_xml, list_geometries_in_alps_xml

    print(list_models_in_alps_xml())
    print(list_geometries_in_alps_xml())

Custom parameters are passed via the ``parms`` dictionary::

    model = model_from_alps_xml(name="Heisenberg", parms={"S": "1", "Jz": 1.0})
    graph = graph_from_alps_xml(name="square lattice", parms={"L": 4, "W": 4})

Fermionic models and qubits
---------------------------

When a model has fermionic quantum numbers, and each of them takes two
values (``N`` in ``spinless fermions``; ``Nup`` and ``Ndown`` in ``fermion
Hubbard``), every site is split into subsites: one qubit per fermionic
quantum number and, if the site has other quantum numbers that are not
functions of those, a single subsite with the rest::

    system = build_system("chain lattice", "fermion Hubbard", L=4, U=2)
    list(system.sites)        # ['1[0]:Nup', '1[0]:Ndown', '1[1]:Nup', ...]
    system.site_qubits["1[0]"]  # ('1[0]:Nup', '1[0]:Ndown')

The fermionic modes are mapped to Pauli strings through the Jordan-Wigner
transformation :cite:p:`JordanWigner1928`, in the order of ``system.sites``, so they anticommute also
within a site. Site operators are still requested by the name of the node
(``system.site_operator("c_up", "1[0]")``). Without a rest subsite they
are, like the global operators, sums of Pauli strings
(:class:`~qalma.operators.PauliSumOperator`); with one, they are sums of
:class:`~qalma.operators.TensorProductOperator` (a Pauli string times local
factors on the rest subsites). A site that is not split keeps the name of
the node.

The local matrices of the ALPS library are read with the following
convention: an operator that changes a single fermionic quantum number
(``c_up``, ``cdag_down``, ...) is the corresponding mode operator, with its
Jordan-Wigner string; an operator that does not change the fermion parity
of the site (``n``, ``Sz``, ``Splus`` :math:`= c^\dagger_\uparrow
c_\downarrow`, ``Sminus`` :math:`= c^\dagger_\downarrow c_\uparrow`, ...)
is the local matrix itself.

Constrained sites: the t-J model
++++++++++++++++++++++++++++++++

The local basis of ``t-J`` has three states, an empty site and an electron
with spin up or down, and the quantum numbers ``N`` (fermionic), ``S``
:math:`= N/2` and ``Sz``. The site is split into a qubit for ``N`` and a
three-level subsite for ``Sz``::

    system = build_system("chain lattice", "t-J", L=4, t=1, J=0.4)
    list(system.sites)        # ['1[0]:N', '1[0]:Sz', '1[1]:N', ...]
    system.constrained        # True

The product of the two subsites has six states, three of them not
physical: an empty site with :math:`S_z = \pm 1/2`, and an occupied site
with :math:`S_z = 0`. The operators of the model vanish on those states and
do not mix them with the physical ones, so the physical states are a sector
of the operator

.. math::

   Q = \sum_i (1 - \Pi_i),

with :math:`\Pi_i` the projector of the site :math:`i` onto its physical
states, available as ``system.operators["global_operators"]["unphysical_sites"]``.
:math:`Q` is diagonal and commutes with every operator of the model; the
physical states are those with :math:`Q = 0`. Spectra, ground states and
Gibbs states must be taken in that sector, for instance by adding
:math:`\Lambda Q` with a large :math:`\Lambda` to the generator of a Gibbs
state (the error is :math:`O(e^{-\Lambda})`). The same holds for
``alternative t-J``, split into two qubits ``Nup`` and ``Ndown`` with the
doubly occupied state excluded.

Product states over the subsites can impose :math:`Q = 0` only in trivial
cases: for mean-field methods, the subsites of a node must be in the same
block.

``build_system(..., qubit_split=True)`` forces the representation for any
model whose sites are products of two-level quantum numbers (e.g. spin 1/2
chains, as qubits without strings), and ``qubit_split=False`` disables it.

.. seealso::

   :mod:`qalma.model`, :mod:`qalma.geometry`, :mod:`qalma.alpsmodels`
