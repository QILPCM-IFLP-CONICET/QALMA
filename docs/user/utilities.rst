.. _utilities:

Utilities: export, drawing and fits
===================================

.. contents::
   :depth: 2
   :local:

The module :mod:`qalma.utils` collects helper functions to inspect and
post-process the objects of QALMA. This page describes the ones that export
operators to Mathematica, draw the support of an operator, and fit
the output of optimizations.

Exporting to Mathematica
------------------------

:func:`~qalma.utils.operator_to_wolfram` returns a string with a Wolfram
Language expression that represents an operator, which can be pasted into a
Mathematica notebook. Local and product operators are exported as a
``KroneckerProduct`` with one factor for each site of the system, in the order
of the sites of the system; sums are exported as sums.

.. code-block:: python

   from qalma.model import build_spin_chain
   from qalma.utils import operator_to_wolfram

   system = build_spin_chain(2)
   sx0 = system.site_operator("Sx", "1[0]")
   print(operator_to_wolfram(sx0))

Output::

   KroneckerProduct[{
   {0.0, 0.5},
   {0.5, 0.0}
   }, {
   {1.0, 0.0},
   {0.0, 1.0}
   }]

The matrix of a ``qutip.Qobj`` or of any other operator is exported as a list
of lists, using :func:`~qalma.utils.matrix_to_wolfram`. This function writes
integers as WL integers, the remaining real numbers as machine reals, and
complex numbers with ``I``:

.. code-block:: python

   import numpy as np

   from qalma.utils import matrix_to_wolfram

   print(matrix_to_wolfram(np.array([[1, 0], [0, -2]])))
   print(matrix_to_wolfram(np.array([[1.5, 2j]])))

Output::

   {
   {1, 0},
   {0, -2}
   }
   {
   {1.5, 2I}
   }

Drawing operators
-----------------

:func:`~qalma.utils.draw_operator` draws an operator over the graph of its
system. A term acting on one site is drawn as a disk on the site, a term
acting on two sites as an ellipse around both, and a term acting on more sites
as a closed line through them. The terms of a sum are drawn one by one. The
function needs `matplotlib` (see :doc:`configuration`).

.. code-block:: python

   import matplotlib.pyplot as plt

   from qalma.utils import draw_operator

   system = build_spin_chain(6)
   site = lambda name, i: system.site_operator(name, f"1[{i}]")
   op = (
       site("Sz", 0)
       + site("Sx", 1) * site("Sx", 2)
       + site("Sx", 3) * site("Sx", 4) * site("Sx", 5)
   )

   fig, ax = plt.subplots()
   draw_operator(op, ax)
   system.spec["graph"].draw(ax)
   plt.show()

The function returns the axis, and it does not draw the graph itself: the
nodes and the edges are drawn with the ``draw`` method of the graph, as in the
:doc:`../quickstart`.

Convex fits
-----------

:func:`~qalma.utils.convex_fit_derivative` estimates the derivative of a
function that is known to be convex or concave from noisy samples, for
instance the magnetization as the derivative of the free energy with respect to
a field. It fits a cubic spline whose second derivative has a given sign, so
that the derivative that it returns is monotonic. The samples do not need to
be sorted, and the derivatives are returned in the order of the samples.

.. code-block:: python

   import numpy as np

   from qalma.utils import convex_fit_derivative

   rng = np.random.default_rng(0)
   fields = np.linspace(0.0, 2.0, 40)
   free_energy = -np.log(2 * np.cosh(fields / 2)) + 1e-3 * rng.normal(size=40)

   # The free energy is concave as a function of the field (convexity=-1).
   slope = convex_fit_derivative(list(zip(fields, free_energy)), convexity=-1)
   print(bool(np.all(np.diff(slope) < 1e-6)))  # the slope is decreasing

   exact = -0.5 * np.tanh(fields / 2)
   print(bool(np.abs(np.array(slope) - exact).max() < 0.05))

Output::

   True
   True

With the default ``convexity=-1`` the fit is concave, and with
``convexity=1`` it is convex.
