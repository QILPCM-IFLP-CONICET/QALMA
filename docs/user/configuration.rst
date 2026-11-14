Configuration
=============

Environment variables
---------------------

QALMA reads a few settings from environment variables **when the package is
imported**, so they have to be set before ``import qalma``:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Variable
     - Default
     - Meaning
   * - ``QALMA_TOLERANCE``
     - ``1.0e-14``
     - Numerical tolerance used to decide when coefficients and operators
       are zero or equal (:data:`qalma.settings.QALMA_TOLERANCE`).
   * - ``QALMA_USE_PARALLEL``
     - off
     - Enable the parallel routines for large operators. Empty values,
       ``0``, ``false``, ``no`` and ``off`` disable it; any other value
       enables it. The unprefixed name ``USE_PARALLEL`` is also accepted.
   * - ``QALMA_MAX_WORKERS``
     - ``8``
     - Maximum number of workers used by the parallel routines.

For example:

.. code-block:: bash

   QALMA_USE_PARALLEL=1 QALMA_MAX_WORKERS=4 python my_simulation.py

The test suite uses two more variables: ``QALMA_ALLTESTS=1`` enables the
slower tests, and ``BENCHMARKS=1`` the performance benchmarks.

Logging
-------

QALMA does not configure logging. See the *Logging* section of the
:doc:`../quickstart` to enable its messages.

Optional dependencies
---------------------

The plotting helpers (:meth:`qalma.geometry.GraphDescriptor.draw` and
:func:`qalma.utils.draw_operator`) need `matplotlib`, which is installed
with the ``plot`` extra:

.. code-block:: bash

   pip install "qalma[plot]"

Public API and versioning
-------------------------

QALMA follows `semantic versioning <https://semver.org>`_. The public API
is made of the names exported by ``qalma`` and its subpackages (those listed
in their ``__all__``). Names starting with an underscore, and the contents
of :mod:`qalma.settings` other than the values documented above, are
implementation details and can change in any release.
