import os
import sys
import warnings

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


# -- Autodoc -----------------------------------------------------------------

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
add_module_names = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "imported-members": False,
}

# -- Napoleon (NumPy docstrings) ---------------------------------------------

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# -- Autosummary -------------------------------------------------------------

autosummary_generate = True

# -- Todo --------------------------------------------------------------------

todo_include_todos = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "qutip": ("https://qutip.readthedocs.io/en/stable/", None),
}

# -- nbsphinx ----------------------------------------------------------------

nbsphinx_execute = "never"  # set to "auto" once RTD has ALPS installed
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

# -- Warning suppression -----------------------------------------------------

# Duplicate object descriptions arise from __init__.py re-exporting symbols
# that autodoc already indexed in their submodules. Listed explicitly until
# the __init__.py imports are refactored.
nitpick_ignore = [
    # Duplicate re-exports from __init__.py (intentional public API surface)
    ("py:attr", "qalma.operators.arithmetic.SumOperator.terms"),
    ("py:attr", "qalma.operators.product.ProductOperator.site_factors"),
    ("py:attr", "qalma.operators.product.ProductOperator.system"),
    ("py:attr", "qalma.operators.states.gibbs.GibbsDensityOperator.k"),
    ("py:attr", "qalma.operators.states.gibbs.GibbsDensityOperator.normalized"),
    ("py:attr", "qalma.operators.states.gibbs.GibbsProductDensityOperator.isherm"),
    (
        "py:attr",
        "qalma.operators.states.gibbs.GibbsProductDensityOperator.free_energies",
    ),
    ("py:attr", "qalma.operators.states.gibbs.GibbsProductDensityOperator.k_by_site"),
    # numpy.random.RandomState.rand docstring inlined by autodoc — not our code
    ("py:class", "ndarray"),
    ("py:class", "shape ``(d0"),
    ("py:class", "d1"),
    ("py:class", "dn)``"),
    ("py:obj", "Ellipsis"),
    ("py:obj", "random"),
    ("py:obj", "Coefficient"),
    # Shape fragments from numpy.linalg / scipy.linalg docstrings
    ("py:class", "(..."),
    ("py:class", "(2"),
]

suppress_warnings = ["ref.python", "ref.ref", "misc.highlighting_failure", "app.add_directive", "autosummary.import_cycle"]

# Suppress unresolvable type-hint cross-references that arise from bare Python
# typing syntax (List[X], Callable[...], np.array, etc.) in NumPy-style
# docstrings. These are annotation strings, not actual Sphinx cross-references,
# and cannot be resolved without full stub packages available at build time.
nitpick_ignore_regex = [
    # --- Standard-library / typing generics (not resolvable without stubs) ---
    ("py:class", r"Callable.*"),
    ("py:class", r"List\[.*"),
    ("py:class", r"Tuple\[.*"),
    ("py:class", r"Dict\[.*"),
    ("py:class", r"Iterable\[.*"),
    ("py:class", r"Optional\[.*"),
    ("py:class", r"Union\[.*"),
    ("py:class", r"Sequence\[.*"),
    ("py:class", r"Set\[.*"),
    # --- NumPy / SciPy types (no intersphinx inventory available) ---
    ("py:class", r"np\..*"),
    ("py:class", r"array.like"),
    ("py:class", r"array-like"),
    ("py:class", r"ndarray"),
    ("py:class", r"NDArray.*"),
    ("py:class", r"numpy\._typing.*"),
    ("py:class", r"numpy\.dtype"),
    ("py:class", r"float64"),
    ("py:class", r"scalar"),
    ("py:class", r"matrix"),
    ("py:class", r"complex ndarray"),
    ("py:class", r"int ndarray"),
    ("py:class", r"\(M.*"),           # shape fragments like (M, N)
    ("py:class", r"\(\.\.\. .*"),
    ("py:class", r"\(2 .*"),
    ("py:class", r"M\) .*"),
    ("py:class", r"N\) .*"),
    ("py:obj",   r"angle"),
    ("py:obj",   r"imag"),
    ("py:obj",   r"real_if_close"),
    ("py:obj",   r"scipy\.linalg\.cholesky.*"),
    ("py:obj",   r"scipy\.linalg\.cho_factor"),
    ("py:obj",   r"numpy\.linalg\.cond"),
    ("py:obj",   r"numpy\.linalg\.svd"),
    ("py:func",  r"eig$"),
    ("py:func",  r"eigvalsh$"),
    ("py:func",  r"eigvals_banded"),
    ("py:func",  r"eigvalsh_tridiagonal"),
    ("py:func",  r"scipy\.linalg\.expm"),
    ("py:func",  r"scipy\.linalg\.logm"),
    # --- qutip (no intersphinx) ---
    ("py:class", r"qutip\.Qobj"),
    ("py:class", r"Qobj"),
    ("py:obj",   r"QobjEvo"),
    # --- h5py (no intersphinx) ---
    ("py:class", r"h5py.*"),
    # --- matplotlib ---
    ("py:class", r"mpl\.Axis"),
    # --- packaging ---
    ("py:class", r"packaging\.version\.Version"),
    ("py:exc",   r"InvalidVersion"),
    # --- Closing-bracket fragments from generic aliases ---
    ("py:class", r".*\]$"),
    ("py:class", r"str \(case.*"),
    # --- QALMA unqualified class names in Returns/Parameters ---
    # These appear unqualified in many docstrings; the canonical cross-reference
    # is the fully-qualified one used in class/function signatures.
    ("py:class", r"Operator"),
    ("py:class", r"LocalOperator"),
    ("py:class", r"SumOperator"),
    ("py:class", r"ScalarOperator"),
    ("py:class", r"ProductOperator"),
    ("py:class", r"OneBodyOperator"),
    ("py:class", r"QutipOperator"),
    ("py:class", r"QutipDensityOperator"),
    ("py:class", r"DensityOperator"),
    ("py:class", r"ProductDensityOperator"),
    ("py:class", r"GibbsDensityOperator"),
    ("py:class", r"GibbsProductDensityOperator"),
    ("py:class", r"MixtureDensityOperator"),
    ("py:class", r"QuadraticFormOperator"),
    ("py:class", r"HierarchicalOperatorBasis"),
    ("py:class", r"OperatorBasis"),
    ("py:class", r"DensityOperatorProtocol"),
    ("py:class", r"ProjectingOperatorFunction"),
    ("py:class", r"Simulation"),
    ("py:class", r"SystemDescriptor"),
    # --- dict/tuple fragments that Sphinx splits on the bracket ---
    ("py:class", r"dict\[.*"),
    ("py:class", r"tuple\[.*"),
    ("py:class", r"list\[.*"),
    ("py:class", r"frozenset.*"),
    ("py:class", r"iterable"),
    # --- Prose accidentally parsed as type refs ---
    ("py:class", r"The reduced operator\."),
    ("py:class", r"The partial trace.*"),
    ("py:class", r"the operator\."),
    ("py:class", r"A new `QuadraticFormOperator.*"),
    ("py:class", r"The relative entropy.*"),
    ("py:class", r"and the corresponding relative entropy\."),
    ("py:class", r"num"),
    ("py:class", r"terms in the partial sum.*"),
    ("py:class", r"value"),
    ("py:class", r"Return"),
    ("py:class", r"for the __new__ method.*"),
    ("py:exc",   r"AssertionError:"),
    # --- Unqualified meth/func refs ---
    ("py:meth",  r"load_hdf5"),
    ("py:meth",  r"as_sum_of_products"),
    ("py:meth",  r"partial_trace"),
    ("py:meth",  r"tidyup"),
    ("py:meth",  r"to_qutip"),
    ("py:meth",  r"to_product_state"),
    ("py:meth",  r"reduce"),
    ("py:meth",  r"qutip\.Qobj\.tidyup"),
    ("py:func",  r"update_basis"),
    ("py:func",  r"schmidt_dec_first_rest_qutip_operator_hermitian"),
    ("py:func",  r"_project_product_operator_recursive"),
    ("py:func",  r"_project_qutip_operator_recursive"),
    ("py:func",  r"_project_product_operator_to_one_body"),
    ("py:func",  r"_project_qutip_operator_to_one_body"),
    # --- Lowercase builtin-like type names used as informal type hints ---
    ("py:class", r"callable"),
    ("py:class", r"sequence"),
    ("py:class", r"array"),
    ("py:class", r"number"),
    # --- Single-letter shape fragments from numpy docstrings (e.g. (M, N)) ---
    ("py:class", r"^[A-Z]$"),
    ("py:class", r"^\.\.\.$"),
    ("py:class", r"^\(\.\.\. .*"),   # (... , M, N) shape fragments
    ("py:class", r"^\(2 .*"),        # (2, M) shape fragments from scipy eigvals
    # --- Additional scipy cross-references without intersphinx ---
    ("py:obj",   r"scipy\.linalg\.inv"),
    ("py:obj",   r"Ellipsis"),
    ("py:obj",   r"random"),
    ("py:class", r"collections\.abc\.Buffer"),
    ("py:class", r"numbers\.Number"),
    ("py:class", r"scipy\.linalg.*"),
    ("py:ref",   r"linalg_batch"),
]

# SyntaxWarning from LaTeX strings in notebook cell outputs (Python 3.12+)
import warnings

# Register ipython3 as an alias for ipython so nbsphinx notebooks render correctly
from pygments.lexers import get_lexer_by_name  # noqa: E402
from sphinx.highlighting import lexers  # noqa: E402

try:
    lexers["ipython3"] = get_lexer_by_name("ipython")
except Exception:
    pass

# SyntaxWarning from LaTeX strings in notebook cell outputs (Python 3.12+)
warnings.filterwarnings(
    "ignore",
    message=".*invalid escape sequence.*",
    category=SyntaxWarning,
)
