"""Variational Mean-field.

Build variational approximations to a Gibbsian state.
"""

import logging
from typing import Callable, Optional, Tuple, cast

import numpy as np
from numpy.random import random_sample as _random_sample
from scipy.optimize import minimize as _minimize

from qalma.operators import OneBodyOperator, Operator
from qalma.operators.quadratic import (
    QuadraticFormOperator,
    build_quadratic_form_from_operator,
)
from qalma.operators.states import ProductDensityOperator
from qalma.operators.states.gibbs import GibbsProductDensityOperator
from qalma.projections import n_body_projection
from qalma.qutip_tools.tools import safe_exp_and_normalize
from qalma.settings import DEFAULT_MAX_NUMBER_OF_FIELDS, QALMA_TOLERANCE

__all__ = [
    "compute_free_energy",
    "compute_t_score",
    "variational_quadratic_mfa",
    "self_consistent_mf",
]


def compute_t_score(
    sigma: GibbsProductDensityOperator | ProductDensityOperator,
    k: Operator,
    _f_exact: Optional[float] = None,
):
    r"""Compute the T-score of a variational mean-field state.

    Given a target state :math:`\rho=e^{-k}/Z` with :math:`Z={\rm Tr}e^{-k}`,
    and a trial state :math:`\sigma = e^{-\kappa}/Z_{trial}`, with
    :math:`Z_{trial}={\rm Tr}e^{-\kappa}`, the T-score quantifies how much
    the operator

    .. math::

       \hat{F} = \ln(\sigma)-\ln(\rho)=k-\kappa + \ln(Z/Z_{trial})

    fluctuates relative to its mean value under :math:`\sigma`:

    .. math::

        T_{\rm score}
            = \frac{\operatorname{Var}_\sigma(\hat{F})}
                   {\langle \hat{F} \rangle_\sigma^2}
            = \frac{\langle \hat{F}^2 \rangle_\sigma
                    - \langle \hat{F} \rangle_\sigma^2}
                   {\langle \hat{F} \rangle_\sigma^2}.

    The T-score is zero if and only if :math:`\hat{F}` is constant on the
    support of :math:`\sigma`, which happens precisely when
    :math:`\sigma = \rho`.  It is therefore a *complementary* diagnostic
    to the variational free energy :math:`F[\sigma]`:

    * Large :math:`F[\sigma]`, small :math:`T_{\rm score}`:
      systematic energy offset, but the shape of the distribution is
      well captured.
    * Small :math:`F[\sigma]`, large :math:`T_{\rm score}`:
      good average energy but large residual fluctuations.

    In the limit :math:`|k|, |\kappa|\rightarrow \infty`, :math:`|\kappa|/|k|\leq \infty`,
    :math:`T_{\rm score}\rightarrow V_{\rm score}=\frac{N\,\mathrm{Var}}{|E_{mf}-E_{gs}|^2}`

    Parameters
    ----------
    sigma : GibbsProductDensityOperator | ProductDensityOperator
        The trial (approximate) state.
    k : Operator
        The target generator: the exact state is :math:`\rho = e^{-k}/Z`.
    _f_exact : Optional[float]
        The value of :math:`\ln{\rm Tr}[e^{-k}]` (i.e. :math:`+\log Z_k`,
        **positive** sign convention).  If not given, it is computed exactly.

    Returns
    -------
    tscore : float
        The T-score.  Always :math:`\ge 0`.
    f_mf : float
        :math:`{\rm Tr}[\sigma(k-\kappa)]`, the mean of :math:`\hat{F}`
        under :math:`\sigma` (without the :math:`\ln Z` shift).
        Useful as a sanity check.
    var_f : float
        :math:`\operatorname{Var}_\sigma(\hat{F})`, the numerator of the
        T-score.

    See Also
    --------
    compute_free_energy : Computes :math:`F[\sigma] = \mathrm{Tr}[\sigma(k + \log\sigma)]`.
    variational_quadratic_mfa : Main variational solver.

    Examples
    --------
    >>> from qalma import build_system
    >>> from qalma.operators.states import GibbsProductDensityOperator
    >>> system = build_system("chain lattice", "spin", L=6)
    >>> ham = system.global_operator("Hamiltonian")
    >>> sz_total = system.global_operator("Sz")
    >>> sigma = GibbsProductDensityOperator(sz_total)
    >>> tscore, mean_F, var_F = compute_t_score(sigma, sz_total)
    >>> tscore
    0.0
    >>> sigma_ham = GibbsProductDensityOperator(ham)
    >>> tscore, mean_F, var_F = compute_t_score(sigma_ham, ham)
    >>> tscore
    0.0
    """
    kappa = -sigma.logm()
    f_hat = (k - kappa).simplify()
    mean_f, mean_fsq = cast(np.ndarray, sigma.expect([f_hat, f_hat**2]))
    size = len(f_hat.acts_over())
    var_f = mean_fsq - mean_f**2
    # if not given, compute log Z_k = log Tr[e^{-k}]
    if _f_exact is None:
        _, _f_exact = safe_exp_and_normalize(-k.to_qutip(tuple()))
        k_acts_over = k.acts_over()
        _f_exact += sum(
            np.log(dim)
            for site, dim in k.system.dimensions.items()
            if site not in k_acts_over
        )
    f_mf = float(np.real(mean_f))
    delta = f_mf - _f_exact
    if abs(delta) < 1e-15:
        return 0.0, f_mf, var_f
    assert delta > 0, (
        f"T-score: F_mf={f_mf:.6g} < F_exact={_f_exact:.6g} by {-delta:.2e}; "
        "this should not happen — check units or numerical precision."
    )
    tscore = size * var_f / delta**2
    return tscore, f_mf, var_f


def compute_free_energy(state: ProductDensityOperator, ham: Operator) -> float:
    r"""Estimate the variational free energy of ``ham`` from an approximate state.

    Delegates to
    :meth:`~qalma.operators.states.DensityOperatorMixin.variational_free_energy`.
    Kept as a module-level function for backwards compatibility.

    Parameters
    ----------
    state : ProductDensityOperator
        The trial (approximate) state :math:`\sigma`.  If ``None``, the
        fully mixed state is used.
    ham : Operator
        The generator :math:`H` of the target Gibbs state
        :math:`\rho \propto e^{-H}`.

    Returns
    -------
    float
        :math:`\mathrm{Tr}[\sigma\,(H + \log\sigma)]`.

    """
    if state is None:
        state = ProductDensityOperator({}, system=ham.system)
    return state.variational_free_energy(ham)


def mf_quadratic_form_exponential(
    qf_op: QuadraticFormOperator,
    num_fields: int = 1,
    method: Optional[str] = None,
    callback_optimizer: Optional[Callable] = None,
    ham: Optional[Operator] = None,
) -> ProductDensityOperator:
    """Approximate `exp(-qf_op)` as `exp(-h_mf)`.

    Here

    .. math::

        h_mf = k_0 + sum_a phi_a q_a.

    Parameters
    ----------
    qf_op : QuadraticFormOperator
        The generator of the target state state exp(-qf_op).
    num_fields : int, optional
        The number of terms of terms of qf_op to be kept in the approximation.
        The default is 1.
    method : Optional[str], optional
        The numerical optimization method. T
        he default is None.
    callback_optimizer : Callable, optional
        The callback function to be called on each evaluation.
        The default is None.

    ham: Operator
        The operator to be used as the reference.

    Returns
    -------
    ProductDensityOperator
        The approximated state.

    """

    def build_test_state(coeffs: np.ndarray) -> ProductDensityOperator:
        """Build the test state from the coefficients."""
        terms = tuple((coef * gen for coef, gen in zip(coeffs, generators)))
        if k0 is not None:
            terms = (k0,) + terms
        k = OneBodyOperator(terms, qf_op.system).tidyup().simplify()
        sigma_k = GibbsProductDensityOperator(k)
        return sigma_k.to_product_state()

    def test_state_re(coeffs: np.ndarray) -> float:
        """Target function.

        Computes the relative entropy relative to the gibbs state
        exp(-ham)
        """
        test_state: ProductDensityOperator = build_test_state(coeffs)
        return compute_free_energy(test_state, hamiltonian)

    # Trim terms with positive weights and keep at most
    # num_fields of the remaining
    # terms, in a way that exp(-qf_op')~ exp(-qf_op)

    qf_op = -reduced_quadratic_form_operator(-qf_op, num_fields)
    hamiltonian: Operator = ham or qf_op.as_sum_of_products()

    # Linear term
    k0 = qf_op.linear_term
    if k0:
        k0 = k0.tidyup() or None

    generators = qf_op.basis
    logging.info("using %s generators", len(generators))
    if len(generators) == 0:
        logging.info(
            ("No 2-body terms found. Using the linear term as reference state.")
        )
        if k0:
            return GibbsProductDensityOperator(k0).to_product_state()

        return ProductDensityOperator({}, system=qf_op.system)

    # Now, optimize the relative entropy over states of the form
    # sigma = exp(-k0 - sum_a phi_a Q_a)

    # Generate a initial guess for the coefficients.
    phis = 2 * _random_sample(len(generators)) - 1
    try:
        result = _minimize(
            test_state_re, phis, method=method, callback=callback_optimizer
        )
        phis = result.x
    except ValueError as val_exc:
        logging.info("Optimization failed with exception %s", val_exc)

    sigma_ref = build_test_state(phis)
    return sigma_ref


def reduced_quadratic_form_operator(
    qf_op: QuadraticFormOperator, num_terms: int
) -> QuadraticFormOperator:
    """Build a new quadratic form operator keeping only positive weights.

    Parameters
    ----------
    qf_op: QuadraticFormOperator
         the quadratic form.
    num_terms: int
         number of generators to keep.

    Returns
    -------
    QuadraticFormOperator
        A new :class:`~qalma.operators.quadratic.QuadraticFormOperator`
        keeping only the ``num_terms`` largest positive weights, rescaled
        so all weights equal 1.

    """
    assert num_terms > 0, f"num_terms must be an integer number >0. Got {num_terms}."
    weights, basis = qf_op.weights, qf_op.basis
    if len(weights) == 0:
        return qf_op
    num_terms = min(len(weights), num_terms)
    # If there are enough positive terms, the minimal
    # weight is the weight of the `num_terms`-th weight.
    # If that term is not positive, the minimal weight is 0.
    min_weight = max(0, sorted(weights)[-num_terms])
    generators = tuple(
        (
            basis_op * (weight**0.5)
            for weight, basis_op in zip(weights, basis)
            if weight >= min_weight
        )
    )
    return QuadraticFormOperator(
        generators,
        tuple((1 for i in generators)),
        qf_op.system,
        qf_op.linear_term,
        qf_op.offset,
    )


def self_consistent_mf(
    ham: Operator,
    sigma_ref: Optional[ProductDensityOperator] = None,
    max_steps: int = 10,
    callback: Optional[Callable] = None,
) -> Tuple[ProductDensityOperator, float]:
    """Build a self-consistent approximation of :math:`e^{-H}`.

    Starting from ``sigma_ref``, compute an approximation of :math:`e^{-H}`
    following a self-consistent mean-field algorithm.

    Parameters
    ----------
    ham : Operator
        The generator of the exact state :math:`\\rho = e^{-H}`.
    sigma_ref : ProductDensityOperator, optional
        The initial state to begin the self-consistent loop.
        Defaults to the fully mixed state.
    max_steps : int, optional
        Maximum number of self-consistent steps (default 10).
    callback : Callable, optional
        Called on each self-consistent round with signature
        ``callback(sigma, rel_entropy, step)``.

    Returns
    -------
    tuple
        A pair ``(sigma, rel_entropy)`` where ``sigma`` is the
        :class:`~qalma.operators.states.product.ProductDensityOperator`
        that approximates :math:`e^{-H}`, and ``rel_entropy`` is the
        corresponding relative entropy.

    """
    if sigma_ref is None:
        sigma_ref = ProductDensityOperator({}, system=ham.system)

    rel_entropy = compute_free_energy(sigma_ref, ham)
    converged = False
    for curr_step in range(max_steps):
        gen_sc = n_body_projection(ham, n_max=1, sigma=sigma_ref).hermitian_part()
        sigma_sc = GibbsProductDensityOperator(gen_sc).to_product_state()
        new_rel_entropy = compute_free_energy(sigma_sc, ham)
        if callback is not None:
            callback(sigma_ref, rel_entropy, curr_step)

        if abs(new_rel_entropy - rel_entropy) < QALMA_TOLERANCE:
            converged = True
            break
        if np.real(new_rel_entropy - rel_entropy) > 10 * QALMA_TOLERANCE:
            break
        rel_entropy = new_rel_entropy
        sigma_ref = sigma_sc

    if converged is False:
        msg = (
            "self consistent mean field failed to converge after "
            f"{curr_step} iterations. "
            f"Last Delta S_rel= {np.real(new_rel_entropy - rel_entropy)}."
        )
        logging.warning(msg)
    return cast(ProductDensityOperator, sigma_ref), rel_entropy


def variational_quadratic_mfa(
    ham: Operator,
    numfields: int = DEFAULT_MAX_NUMBER_OF_FIELDS,
    sigma_ref: Optional[ProductDensityOperator] = None,
    **kwargs,
) -> ProductDensityOperator:
    r"""
    Build a variational meanfield approximation 2-body projections.

    Find the Mean field approximation for the exponential
    of an operator using a variational algorithm.

    At the end, improve the solution in a self-consistent
    way, looking for a fixed point of the one-body projection.

    Decompose ham as a quadratic form

    .. code-block:: python

        ham = sum_a w_a Q_a^2 + L + delta_ham

    Then keep ``numfields`` terms of the sum with maximal weights,
    and look for a variational mean field state

    .. code-block:: python

        sigma \propto exp(-\sum_a phi_a Q_a + L)

    for real values of ``phi_a``.

    Parameters
    ----------
    ham : Operator
        The generator of the exact state rho=exp(-ham).
    numfields : int, optional
        The minimal number of *fields* $\phi_a$ to be included in the
        optimization. If there are several generators of the quadratic form
        with the same weight, ``numfields`` is extended to include all of them.
        If ``numfields`` is ``0``, the function just does a self consistent
        search.
        The default is 1.
    sigma_ref : DensityOperatorProtocol, optional
        The initial reference state to project `ham` to a quadratic form.
        The default is None.
    its : int, optional
        Maximum number of recursive rounds. If the operator is already a
        2-body operator, ``its`` is set to 1. The default is 1.
    method : Optional[str], optional
        The method used in the numeric optimization. The default is None.
    callback_optimizer : Callable, optional
        Callback function called on each evaluation of the optimizer.
        The default is None.
    max_self_consistent_steps : int, optional
        Maximum number of self-consistent steps used to improve the solution.
        The default is 10.
    callback_self_consistent_step : Callable, optional
        Function called on each self-consistent round. The default is None.

    Returns
    -------
    ProductDensityOperator
        A product operators that approximates exp(-ham).

    """
    sigma_candidate: ProductDensityOperator
    its: int = kwargs.get("its", 1)
    method: Optional[str] = kwargs.get("method", None)
    callback_optimizer = kwargs.get("callback_optimizer", None)
    max_self_consistent_steps: int = kwargs.get("max_self_consistent_steps", 10)
    callback_self_consistent_step = kwargs.get("callback_self_consistent_step", None)
    ham = ham.hermitian_part()

    if sigma_ref is not None and hasattr(sigma_ref, "to_product_state"):
        sigma_ref = sigma_ref.to_product_state()

    current_rel_entropy = (
        None if sigma_ref is None else compute_free_energy(sigma_ref, ham)
    )
    if isinstance(ham, OneBodyOperator):
        return GibbsProductDensityOperator(ham.hermitian_part()).to_product_state()

    if numfields == 0:
        sigma, _ = self_consistent_mf(
            ham, sigma_ref, max_steps=max_self_consistent_steps
        )
        return sigma

    for _ in range(its):
        # We start by projecting the generator `ham` to the two-body sector
        # relative to `sigma_ref`:
        changed = False

        ham_proj = n_body_projection(ham, n_max=2, sigma=sigma_ref).hermitian_part()
        if isinstance(ham_proj, OneBodyOperator):
            sigma_candidate = GibbsProductDensityOperator(ham_proj).to_product_state()
        else:
            # Now, write the projected operator as a QuadraticFormOperator
            # ham_proj = k_0 + sum_a w_a Q_a^2
            # with |Q_a|_{infty}=1 and
            # w_1 <= w_2 <=... <=w_l < 0 <= w_{k+1} <= ... w_n
            qf_op: QuadraticFormOperator = build_quadratic_form_from_operator(
                ham_proj,
                isherm=True,
                sigma_ref=sigma_ref,
                sort_fn=lambda x: x,
                count=numfields,
            )
            sigma_candidate = mf_quadratic_form_exponential(
                qf_op, numfields, method, callback_optimizer, ham
            )

        if current_rel_entropy is None:
            changed = True
            sigma_ref = sigma_candidate
            current_rel_entropy = compute_free_energy(sigma_ref, ham)
        else:
            rel_s = compute_free_energy(sigma_candidate, ham)
            if rel_s < current_rel_entropy:
                changed = True
                sigma_ref = sigma_candidate
                current_rel_entropy = rel_s

        # Improve the solution by a self-consistent round
        sigma_candidate, rel_s = self_consistent_mf(
            ham,
            sigma_ref,
            max_steps=max_self_consistent_steps,
            callback=callback_self_consistent_step,
        )
        if rel_s < current_rel_entropy:
            changed = True
            logging.info("  self consistent attempt-> S_rel=%.6f.", current_rel_entropy)
            sigma_ref = sigma_candidate
            current_rel_entropy = rel_s

        if ham_proj is ham or not changed:
            break

    if sigma_ref is None:
        return ProductDensityOperator({}, system=ham.system)
    if hasattr(sigma_ref, "to_product_state"):
        sigma_ref = sigma_ref.to_product_state()
    return cast(ProductDensityOperator, sigma_ref)
