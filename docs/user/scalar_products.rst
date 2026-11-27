.. _scalar-products:

Scalar products and operator bases
==================================

.. contents::
   :depth: 2
   :local:

Overview
--------

Several algorithms of QALMA, such as the evolution restricted to a
hierarchical basis of generators (:doc:`evolution`), work in a vector space of
operators with a scalar product :math:`\langle A, B\rangle`. The subpackage
``qalma.scalarprod`` (:doc:`/api/scalarprod`) provides

* functions that build the scalar products
  (:mod:`qalma.scalarprod.build`),
* the Gram matrix of a set of operators
  (:func:`~qalma.scalarprod.gram.gram_matrix`),
* routines that orthonormalize a set of operators and compute the components
  of an operator in the result
  (:mod:`qalma.scalarprod.orthogonalize`), and
* classes that build and extend bases of operators
  (:class:`~qalma.scalarprod.basis.OperatorBasis` and
  :class:`~qalma.scalarprod.basis.HierarchicalOperatorBasis`).

A scalar product is a plain Python function ``sp(op1, op2)`` that returns a
complex number. All the scalar products of QALMA are antilinear in the first
argument and linear in the second one, and accept qalma operators. The
Hilbert-Schmidt and the KMB products also accept ``qutip.Qobj``.

Scalar products
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Function
     - :math:`\langle A, B\rangle`
     - Reference state
   * - :func:`~qalma.scalarprod.build.fetch_hs_scalar_product`
     - :math:`\mathrm{Tr}(A^\dagger B)`
     - none
   * - :func:`~qalma.scalarprod.build.fetch_covar_scalar_product`
     - :math:`\tfrac12\mathrm{Tr}\left(\sigma\{A^\dagger, B\}\right)`
     - :math:`\sigma`
   * - :func:`~qalma.scalarprod.build.fetch_kubo_scalar_product`
     - :math:`\int_0^1\mathrm{Tr}\left(\sigma^{1-\tau}A^\dagger\sigma^\tau B\right)d\tau`
     - :math:`\sigma`

The Hilbert-Schmidt product is not normalized by the dimension of the Hilbert
space. The covariance product only needs expectation values in :math:`\sigma`,
and it is the one used by the evolution restricted to a hierarchical basis
(:doc:`evolution`). The Kubo-Mori-Bogoliubov (KMB) product
:cite:p:`Kubo1957,PetzToth1993` is the natural metric of the manifold of Gibbs
states, but it requires the spectral decomposition of :math:`\sigma`, which is
computed densely.
:func:`~qalma.scalarprod.build.fetch_kubo_integral_representation_scalar_product`
implements the same product from its integral form, using a Gauss-Legendre rule
for the integral. It is slower, and it is useful to check the spectral
implementation.

For the same state, the three products differ only in the weight given to the
matrix elements of the operators in the eigenbasis of :math:`\sigma`: the
arithmetic mean :math:`(p_i+p_j)/2` of the eigenvalues for the covariance
product, and their logarithmic mean for the KMB product. Since the logarithmic
mean is never larger than the arithmetic one, :math:`\langle A,
A\rangle_{\mathrm{KMB}}\le\langle A, A\rangle_{\mathrm{cov}}`.

.. code-block:: python

   import numpy as np

   from qalma.model import build_spin_chain
   from qalma.operators.states import GibbsProductDensityOperator
   from qalma.scalarprod import (
       fetch_covar_scalar_product,
       fetch_hs_scalar_product,
       fetch_kubo_scalar_product,
   )

   system = build_spin_chain(3)
   site = lambda name, i: system.site_operator(name, f"1[{i}]")

   # sigma is proportional to exp(-K)
   K = sum(0.8 * system.site_operator("Sz", s) for s in system.sites)
   sigma = GibbsProductDensityOperator(K, system=system)

   hs = fetch_hs_scalar_product()
   covar = fetch_covar_scalar_product(sigma)
   kmb = fetch_kubo_scalar_product(sigma)

   sx = site("Sx", 0)
   for name, sp in [("HS", hs), ("covariance", covar), ("KMB", kmb)]:
       print(name, np.round(sp(sx, sx).real, 4))

Output::

   HS 2.0
   covariance 0.25
   KMB 0.2375

The function returned by :func:`~qalma.scalarprod.build.fetch_kubo_scalar_product`
accepts a ``threshold`` to discard the eigenvalues of :math:`\sigma` with the
smallest weights, which reduces the cost for states with a broad spectrum.

Gram matrices
-------------

:func:`~qalma.scalarprod.gram.gram_matrix` computes the matrix
:math:`G_{ij}=\langle b_i, b_j\rangle` of a list of operators. It returns a
real symmetric array, so it is meant for lists of hermitian operators.
A list of arbitrary operators can be expanded in hermitian ones with
:func:`~qalma.scalarprod.orthogonalize.build_hermitian_basis`, as explained
below.

.. code-block:: python

   from qalma.scalarprod import gram_matrix

   basis = [site("Sx", 0), site("Sx", 0) + site("Sz", 0), site("Sx", 0) * site("Sx", 1)]
   gram_matrix(basis, covar)

Output::

   array([[0.25  , 0.25  , 0.    ],
          [0.25  , 0.5   , 0.    ],
          [0.    , 0.    , 0.0625]])

Orthonormal bases
-----------------

:func:`~qalma.scalarprod.orthogonalize.orthogonalize_basis` builds an
orthonormal basis of the span of a list of operators. It uses the
Gram-Schmidt method: the operators are processed in order, and the ones that
are null, or linear combinations of the previous ones, are discarded.
:func:`~qalma.scalarprod.orthogonalize.operator_components` gives the
components of an operator in an orthonormal basis, that is, the coefficients of
its orthogonal projection.

.. code-block:: python

   from qalma.scalarprod import operator_components, orthogonalize_basis

   orth = orthogonalize_basis(basis, covar)
   gram_matrix(orth, covar)              # the identity
   op = 2 * basis[1] - basis[2]          # it is in the span of the basis
   components = operator_components(op, orth, covar)
   rebuilt = sum(c * e for c, e in zip(components, orth))
   abs((rebuilt - op).to_qutip().full()).max()   # zero, up to round-off

For an operator that is not in the span, ``sum(c * e)`` is its orthogonal
projection onto the span, with respect to the scalar product.

The functions of the module :mod:`qalma.scalarprod.orthogonalize` that
implement the other orthogonalization methods are not part of the main
API, but they can be imported from there. ``orthogonalize_basis_cholesky``
and ``orthogonalize_basis_svd`` (the symmetric, or Löwdin, orthogonalization)
need a linearly independent set of operators: they raise a
``numpy.linalg.LinAlgError`` otherwise, instead of discarding operators as
Gram-Schmidt does.

Hermitian bases
---------------

The Gram matrix, and the components, are real if the operators are hermitian.
:func:`~qalma.scalarprod.orthogonalize.build_hermitian_basis` builds, from any
list of operators, an orthonormal basis of hermitian operators that spans their
hermitian and anti-hermitian parts, together with the coefficients of the
original operators in the new basis. Null operators are filtered out.

.. code-block:: python

   from qalma.scalarprod import build_hermitian_basis

   operators = [site("Splus", 0), site("Sx", 0) * site("Sx", 1)]
   new_basis, coeffs = build_hermitian_basis(operators)
   len(new_basis), coeffs.shape       # (3, (2, 3))
   all(op.isherm for op in new_basis)  # True

The operator ``operators[i]`` is ``sum(coeffs[i, j] * new_basis[j])``. Here,
:math:`S^+` is not hermitian and contributes with two elements, and the
product :math:`S^x_0S^x_1` with one. The orthonormalization uses the
Hilbert-Schmidt product by default. A different scalar product is passed as
the second argument.

.. seealso::

   :doc:`evolution`, :doc:`/api/scalarprod`
