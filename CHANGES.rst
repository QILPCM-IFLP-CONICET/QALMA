Changelog
=========

All notable changes to QALMA are documented in this file. Version numbers
follow `Semantic Versioning <https://semver.org>`_.


1.0.0 (unreleased)
------------------

First public release.

Packaging
+++++++++

* QALMA is now distributed under the LGPL-3.0 (previously GPL-3.0). The ALPS
  model and lattice libraries in ``qalma/lib`` keep the Boost Software
  License 1.0.
* Python 3.11 or newer is required.
* Runtime dependencies are declared with tested lower bounds:
  ``numpy>=1.26``, ``scipy>=1.11``, ``qutip>=5.1``, ``h5py>=3.10``,
  ``matplotlib>=3.8`` and ``packaging``.
* ``pyproject.toml`` is the only source of package metadata; ``setup.py``
  was removed.
* The wheel installs only the ``qalma`` package. Previously it also
  installed a top-level ``test`` package.
* ``qalma.__version__`` is available and is the single source of the
  version number.

Behaviour changes
+++++++++++++++++

* Importing ``qalma`` no longer calls ``logging.basicConfig``. All messages
  are emitted through loggers under the ``"qalma"`` name, so INFO and DEBUG
  messages are hidden unless the application configures logging, e.g.::

      import logging
      logging.basicConfig(level=logging.INFO)

* Diagnostic ``print`` calls in the library were removed or turned into
  ``DEBUG`` log messages.
