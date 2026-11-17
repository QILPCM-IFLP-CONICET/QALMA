Three-body Operators and Loop Terms
====================================

.. contents::
   :depth: 2
   :local:

Overview
--------

Beyond site terms and two-body bond terms, QALMA supports **loop terms** —
operators that act on an ordered tuple of three or more sites forming a
closed loop in the lattice.  The canonical example is the scalar spin
chirality

.. math::

   \chi_{ijk} = \vec{S}_i \cdot (\vec{S}_j \times \vec{S}_k)
              = S^x_i(S^y_j S^z_k - S^z_j S^y_k)
              + S^y_i(S^z_j S^x_k - S^x_j S^z_k)
              + S^z_i(S^x_j S^y_k - S^y_j S^x_k),

which appears in effective descriptions of frustrated magnets, topological
phases, and fractional quantum Hall states on ladders.  It is a genuinely
irreducible three-body interaction: it cannot be written as a sum of
two-body operators.

Defining loops in the lattice XML
----------------------------------

A loop is defined in the ``UNITCELL`` block of ``lattices.xml`` using a
``<LOOP>`` element.  Each ``<NODE>`` inside specifies a vertex of the
unit cell and its unit-cell offset:

.. code-block:: xml

   <UNITCELL name="triangular strip" dimension="1" vertices="2">
     <!-- site vertices -->
     <VERTEX><COORDINATE>0   0</COORDINATE></VERTEX>
     <VERTEX><COORDINATE>0   1</COORDINATE></VERTEX>

     <!-- bond terms -->
     <EDGE type="0"><SOURCE vertex="1" offset="0"/><TARGET vertex="1" offset="1"/></EDGE>
     <EDGE type="1"><SOURCE vertex="2" offset="0"/><TARGET vertex="2" offset="1"/></EDGE>
     <EDGE type="2"><SOURCE vertex="1" offset="0"/><TARGET vertex="2" offset="0"/></EDGE>
     <EDGE type="3"><SOURCE vertex="1" offset="0"/><TARGET vertex="2" offset="1"/></EDGE>

     <!-- up-triangle: 1[i] → 2[i] → 1[i+1] -->
     <LOOP type="0">
       <NODE vertex="1" offset="0"/>
       <NODE vertex="2" offset="0"/>
       <NODE vertex="1" offset="1"/>
     </LOOP>

     <!-- down-triangle: 1[i] → 2[i+1] → 2[i] -->
     <LOOP type="1">
       <NODE vertex="1" offset="0"/>
       <NODE vertex="2" offset="1"/>
       <NODE vertex="2" offset="0"/>
     </LOOP>
   </UNITCELL>

The **order of the** ``NODE`` **elements matters**: it determines which
site maps to index ``i``, ``j``, and ``k`` in the ``LOOPTERM`` expression
(see below).  Reversing the node order is equivalent to changing the sign
of :math:`\chi_{ijk}`, because the scalar triple product is antisymmetric
under permutation of any two indices.

Defining the loop Hamiltonian in the model XML
----------------------------------------------

Loop terms are added to a ``HAMILTONIAN`` block using a ``<LOOPTERM>``
element.  The ``indices`` attribute names the three site labels used in
the algebraic expression:

.. code-block:: xml

   <HAMILTONIAN name="chiral spin">
     <PARAMETER name="J"      default="0"/>
     <PARAMETER name="Wilson" default="0"/>
     <BASIS ref="spin"/>

     <!-- two-body Heisenberg term on all bonds -->
     <BONDTERM source="i" target="j">
       <PARAMETER name="J#" default="J"/>
       J# * exchange(i, j)
     </BONDTERM>

     <!-- three-body scalar chirality on all loops -->
     <LOOPTERM indices="i j k">
       <PARAMETER name="Wilson2#" default="Wilson2"/>
       Wilson2# * (
         Sx(i)*(Sy(j)*Sz(k) - Sz(j)*Sy(k))
       + Sy(i)*(Sz(j)*Sx(k) - Sx(j)*Sz(k))
       + Sz(i)*(Sx(j)*Sy(k) - Sy(j)*Sx(k))
       )
     </LOOPTERM>
   </HAMILTONIAN>

The ``#`` suffix on parameter names (e.g. ``Wilson2#``) enables
per-loop-type overrides: ``Wilson20`` sets the coupling for loop type 0,
``Wilson21`` for loop type 1, and so on.

Example: chiral spin Hamiltonian on a triangular strip
------------------------------------------------------

The ``"chiral spin"`` model and the ``"triangular strip open/periodic"``
lattices are included in QALMA's built-in library.  Building the system
requires only four lines:

.. code-block:: python

   from qalma import graph_from_alps_xml, model_from_alps_xml
   from qalma.model import SystemDescriptor

   L   = 6          # number of unit cells  →  2*L sites
   J   = 1.0        # Heisenberg NN coupling
   chi = 0.5        # scalar chirality coupling

   graph  = graph_from_alps_xml(name="triangular strip open",
                                parms={"L": L, "a": 1})
   model  = model_from_alps_xml(name="chiral spin")
   system = SystemDescriptor(graph, model,
                             {"J": J, "Wilson2": chi})
   ham    = system.global_operator("Hamiltonian")

The resulting ``ham`` is a
:class:`~qalma.operators.arithmetic.SumOperator` that includes both the
Heisenberg bond terms and the chiral loop terms on every up- and
down-triangle of the strip.

You can inspect which loops were generated:

.. code-block:: python

   print(graph.loops)
   # {'0': [['1[0]','2[0]','1[1]'], ['1[1]','2[1]','1[2]'], ...],
   #  '1': [['1[0]','2[1]','2[0]'], ['1[1]','2[2]','2[1]'], ...]}

Physical properties of the chiral term
---------------------------------------

The scalar chirality operator has several important symmetry properties
that QALMA's loop-term implementation preserves:

**Hermiticity.**
:math:`\chi_{ijk}` is Hermitian (as an operator on the spin Hilbert space),
so the full Hamiltonian remains Hermitian for any real coupling.

**Time-reversal breaking.**
Under time reversal, :math:`\vec{S} \to -\vec{S}`, so
:math:`\chi_{ijk} \to -\chi_{ijk}`.  A non-zero ``Wilson2`` therefore
breaks time-reversal symmetry.

**Antisymmetry.**
:math:`\chi_{ijk} = -\chi_{ikj}`.  Swapping any two site indices changes
the sign.  The node ordering in the ``LOOP`` definition controls this sign:
up- and down-triangles in the strip carry opposite chiralities automatically
because their node orders are related by a transposition.

**Spectrum of a single triangle.**
On a single triangle of spin-1/2 sites the chiral operator has eigenvalues

.. math::

   \left\{0^{\times 4},\; +\tfrac{\sqrt{3}}{4}^{\times 2},\;
   -\tfrac{\sqrt{3}}{4}^{\times 2}\right\},

and partition function

.. math::

   Z_\triangle(\beta, K)
       = 4 + 4\cosh\!\left(\tfrac{\sqrt{3}}{4}\,\beta K\right).

These analytic results are used in the test suite to verify the correctness
of the LOOPTERM implementation.

.. seealso::

   :doc:`meanfield_variational` — variational mean-field approximation
   for Hamiltonians with loop terms, including the chiral strip example.

   :doc:`models` — how site and bond operators are defined.

   :doc:`systems` — building lattice systems from ALPS XML.
