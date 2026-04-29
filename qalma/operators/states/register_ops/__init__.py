"""Handler registration for arithmetic operations involving density operator types.

Importing this package registers all handlers as a side effect. The submodules
are organised by the density operator type that drives each group of handlers:

  _types.py        shared operator type tuples
  _mixture.py      MixtureDensityOperator
  _product.py      ProductDensityOperator
  _gibbs_product.py        GibbsProductDensityOperator
  _gibbs.py        GibbsDensityOperator
  _qutip.py        QutipDensityOperator

"""

import qalma.operators.basic  # noqa: F401
import qalma.operators.states.basic  # noqa: F401

from . import _gibbs  # noqa: F401
from . import _gibbs_product  # noqa: F401
from . import _mixture  # noqa: F401
from . import _product  # noqa: F401
from . import _qutip  # noqa: F401
