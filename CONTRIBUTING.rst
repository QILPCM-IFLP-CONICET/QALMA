Contributing to QALMA
=====================

Bug reports, feature requests and pull requests are welcome in the
`GitHub repository <https://github.com/QILPCM-IFLP-CONICET/QALMA>`_.

Development setup
-----------------

QALMA needs Python 3.11 or newer. In a virtual environment::

    git clone https://github.com/QILPCM-IFLP-CONICET/QALMA.git
    cd QALMA
    pip install -e ".[dev]"

Running the tests
-----------------

::

    python -m pytest test                    # default suite (a few minutes)
    QALMA_ALLTESTS=1 python -m pytest test   # full suite, as run by the CI

The benchmarks need the ``bench`` extra and are run through the
``make benchmark-*`` targets. ``make benchmark-dense`` (not part of
``make benchmark``) covers the code paths that use dense linear algebra, and
it is the one sensitive to the number of BLAS threads. See the docstring of
``test/dense/test_dense_benchmarks.py`` to compare runs with and without
``OPENBLAS_NUM_THREADS=1``.

Code conventions
----------------

The CI checks formatting with ``isort`` and ``black``, lints with ``ruff``
and type-checks with ``mypy``. ``make conventions`` applies the formatters,
and ``make mypy`` runs the type checker. To run the checks automatically
before each commit::

    pip install pre-commit
    pre-commit install

Public functions and classes need a docstring (numpy style); the coverage is
checked with ``interrogate``.

Pull requests
-------------

* Open pull requests against ``master``.
* Add or update tests for the behaviour you change.
* Add an entry to ``CHANGES.rst`` for user-visible changes.
* Documentation lives in ``docs/``; build it with
  ``pip install -e ".[docs]"`` and ``make -C docs html``.
* Commit the notebooks in ``docs/examples/`` without outputs: the
  documentation build executes them. Store the outputs only for a notebook
  that is too slow to run in every build. The CI executes all of them
  (``python -m pytest --nbmake docs/examples``), so keep each one running
  in a few minutes, and do not write files from them.

Releasing
---------

1. Check that the CI is green on ``master``.
2. Set the version in ``qalma/version.py`` (e.g. ``1.0.0``), and the version
   and release date in ``CHANGES.rst`` and ``CITATION.cff``. Commit.
3. Create a GitHub release with the tag ``vX.Y.Z``. The *Release* workflow
   checks that the tag matches the package version, builds the
   distributions and publishes them on PyPI.
4. Zenodo archives the release and mints a DOI. Add it to ``CITATION.cff``
   and to the README.
5. Bump ``qalma/version.py`` to the next development version
   (e.g. ``1.1.0.dev0``).

One-time setup before the first release: register the repository and the
``release.yml`` workflow as a trusted publisher of the ``qalma`` project on
PyPI, create the ``pypi`` environment in the repository settings, and enable
the repository in Zenodo.
