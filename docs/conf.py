import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "QALMA"
copyright = "2025, QILPCM-IFLP-CONICET"
author = "QILPCM-IFLP-CONICET"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "myst_parser",
]

templates_path = ["_templates"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"] if os.path.isdir("_static") else []
html_logo = "logo.svg"
html_theme_options = {
    "logo_only": False,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

# -- Autodoc settings --------------------------------------------------------

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
add_module_names = False

# -- Napoleon settings (NumPy docstrings) ------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Autosummary -------------------------------------------------------------

autosummary_generate = True

# -- Todo extension ----------------------------------------------------------

todo_include_todos = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "qutip": ("https://qutip.org/docs/latest", None),
}

# -- nbsphinx ----------------------------------------------------------------

nbsphinx_execute = "never"  # cambiar a "auto" cuando RTD tenga ALPS instalado
nbsphinx_allow_errors = False

# -- MathJax -----------------------------------------------------------------

mathjax3_config = {
    "tex": {
        "macros": {
            "ket": [r"\left|#1\right\rangle", 1],
            "bra": [r"\left\langle#1\right|", 1],
            "tr": r"\mathrm{Tr}",
            "expect": [r"\left\langle#1\right\rangle", 1],
        }
    }
}
