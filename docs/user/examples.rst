Examples
========

The ``examples/`` directory in the repository contains Jupyter notebooks
that demonstrate QALMA's main features. Below is a short description of
each; click the link to open the rendered notebook.

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

   pip install qalma[dev]
   cd examples
   jupyter notebook

.. seealso::

   :doc:`../quickstart` for a self-contained introduction without notebooks.
