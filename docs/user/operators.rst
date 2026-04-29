Operators
=========

.. contents::
   :depth: 2
   :local:

Overview
--------

QALMA represents many-body operators as Python objects that carry the
algebraic structure of the lattice explicitly, avoiding the construction
of exponentially large matrices until absolutely necessary.

Operator hierarchy
------------------

All operators inherit from :class:`~qalma.operators.basic.Operator` and
fall into one of the following concrete types:

:class:`~qalma.operators.basic.LocalOperator`
    Acts non-trivially on exactly one site. Stores the local matrix as a
    NumPy array.

:class:`~qalma.operators.product.ProductOperator`
    Tensor product :math:`\lambda \bigotimes_i O_i` of local operators on
    disjoint sites.

:class:`~qalma.operators.product.ScalarOperator`
    Scalar multiple of the identity :math:`\lambda \mathbb{I}`.

:class:`~qalma.operators.arithmetic.SumOperator`
    Linear combination :math:`\sum_k O_k` of arbitrary operators.

:class:`~qalma.operators.arithmetic.OneBodyOperator`
    Restricted sum :math:`\lambda_0 \mathbb{I} + \sum_i O_i` where each
    :math:`O_i` acts on a single site. Used for mean-field generators.

Operator arithmetic
-------------------

Standard Python operators are overloaded and dispatch to the correct
implementation automatically::

    A + B      # SumOperator
    A * B      # product (dispatched by type)
    A * 0.5    # scalar multiplication
    -A         # negation
    A ** 2     # power
    A / 2.0    # scalar division
    A.dag()    # Hermitian conjugate

The dispatch table is extensible: new type pairs can be registered with
:meth:`~qalma.operators.basic.Operator.register_add_handler` and
:meth:`~qalma.operators.basic.Operator.register_mul_handler`.

Common operations
-----------------

.. code-block:: python

    op.tr()              # trace over the full system
    op.partial_trace(sites)   # partial trace, keeps `sites`
    op.reduce(sites, state)   # expectation-value reduction
    op.simplify()        # reduce to simplest equivalent type
    op.norm()            # operator norm
    op.norm("fro")       # Frobenius norm
    op.isherm            # True if Hermitian
    op.isdiagonal        # True if diagonal
    op.expm()            # matrix exponential e^O
    op.logm()            # matrix logarithm log(O)
    op.to_qutip()        # convert to qutip.Qobj

The ``acts_over()`` method returns the frozenset of sites on which the
operator acts non-trivially::

    op.acts_over()   # frozenset({'site0', 'site2'})

Converting to QuTiP
-------------------

Any operator can be converted to a :class:`qutip.Qobj` via
:meth:`~qalma.operators.basic.Operator.to_qutip`::

    qobj = op.to_qutip()                    # full system, lexicographic order
    qobj = op.to_qutip(("site0", "site1"))  # restricted to a block

The optional ``block`` argument controls the tensor-product ordering of the
output. Sites not in the operator's support contribute identity factors.

.. seealso::

   :mod:`qalma.operators`, :mod:`qalma.operators.basic`,
   :mod:`qalma.operators.arithmetic`, :mod:`qalma.operators.product`
