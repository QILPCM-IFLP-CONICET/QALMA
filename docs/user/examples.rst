Examples
========

The ``docs/examples/`` directory in the repository contains Jupyter notebooks
that demonstrate QALMA's main features:

* :doc:`../examples/examples`: getting started. Building a system, global and
  site operators, operator algebra, spectra and exponentials, QuTiP
  integration, larger systems and other geometries.
* :doc:`../examples/example_heisenberg_evolution`: simulation of a Heisenberg
  evolution.
* :doc:`../examples/example_maxent_evolution`: evolution restricted to a
  maximum-entropy manifold.
* :doc:`../examples/example_chiral_variational_mf`: quadratic variational
  mean field for the chiral spin model (see also
  :doc:`meanfield_variational`).

.. note::

   The notebooks are not executed during the documentation build
   (``nbsphinx_execute = "never"``). To run them locally, install QALMA
   with its optional dependencies and launch Jupyter from the repository
   root.

.. toctree::
   :maxdepth: 1

   ../examples/examples
   ../examples/example_heisenberg_evolution
   ../examples/example_maxent_evolution
   ../examples/example_chiral_variational_mf

Running the examples locally
----------------------------

.. code-block:: bash

   pip install "qalma[plot]" notebook
   cd docs/examples
   jupyter notebook

.. seealso::

   :doc:`../quickstart` for a self-contained introduction without notebooks.
