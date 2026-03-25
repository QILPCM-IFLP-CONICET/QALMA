"""
Handler registration for arithmetic operations involving density operator types.

Importing this package registers all handlers as a side effect. The submodules
are organised by the density operator type that drives each group of handlers:

  _types.py        shared operator type tuples
  _mixture.py      MixtureDensityOperator
  _non_product.py  DensityOperatorMixin, QutipDensityOperator, GibbsDensityOperator
  _product.py      ProductDensityOperator
  _gibbs_product.py        GibbsProductDensityOperator
  _gibbs.py        GibbsDensityOperator

Import order matters: _mixture and _non_product must be registered before
_product, _gibbs_product and _gibbs, because the product/Gibbs handlers delegate to the
more general ones via operator arithmetic.
"""

from . import _gibbs  # noqa: F401
from . import _gibbs_product  # noqa: F401
from . import _mixture  # noqa: F401
from . import _non_product  # noqa: F401
from . import _product  # noqa: F401
from . import _qutip  # noqa: F401

SEEN = 0

if SEEN:
    1 / 0
