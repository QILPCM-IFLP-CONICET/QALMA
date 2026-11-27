"""
Convex fit tools
----------------

Tools to postprocess results of optimizations.
"""

import logging
from typing import List, Tuple

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def convex_fit_derivative(
    data: List[Tuple[float, float]], convexity: float = -1.0
) -> List[float]:
    """Fit a convex (or concave) spline and return its first derivative.

    Fits a cubic spline with a constraint on the sign of its second
    derivative to the given data points, and evaluates its first derivative
    at the x-coordinates of the data.

    Parameters
    ----------
    data: List[Tuple[float, float]]
        The ``(x, y)`` data points. They do not need to be sorted.
    convexity: float, optional
        The sign of the convexity. ``1.`` assumes a convex function (the
        second derivative is non-negative), and ``-1.`` assumes a concave
        function (the second derivative is non-positive). By default,
        ``-1.``.

    Returns
    -------
    List[float]
        The first derivative of the fitted spline, evaluated at the
        x-coordinates of ``data``, in the same order as ``data``.

    Notes
    -----
    The spline is a cubic B-spline with ``max(4, N // 3)`` internal knots
    placed at quantiles of the x-coordinates, where ``N`` is the number of
    points. Its coefficients minimize the sum of squared residuals, using the
    SLSQP algorithm of :func:`scipy.optimize.minimize`, with the constraint on
    the sign of the second derivative imposed over 100 equally spaced points in
    the range of the data. If the optimization does not converge, a warning is
    logged, and the result of the last iteration is used.

    Examples
    --------
    The derivative of a concave function, ``-x**2``:

    >>> import numpy as np
    >>> x = np.linspace(0.0, 2.0, 30)
    >>> derivative = convex_fit_derivative(list(zip(x, -(x**2))))
    >>> bool(np.allclose(derivative, -2 * x, atol=0.05))
    True

    """
    # 1. Parse and sort data
    x_data_raw, y_data_raw = [np.array(lst) for lst in zip(*data)]

    sort_idx = np.argsort(x_data_raw)
    x = x_data_raw[sort_idx]
    y = y_data_raw[sort_idx]

    # 2. Define B-spline basis
    k = 3  # Cubic spline degree
    num_internal_knots = max(k + 1, len(x) // 3)
    # Avoid quantile error for very few points by ensuring linspace has at least 2 points
    if num_internal_knots < 1:
        internal_knots = np.array([])
    else:
        internal_knots = np.quantile(x, np.linspace(0, 1, num_internal_knots + 2)[1:-1])

    knots = np.concatenate(([x.min()] * k, internal_knots, [x.max()] * k))
    n_coeffs = len(knots) - k - 1

    def get_spline_from_coeffs(coeffs, knots, k):
        """Build a BSpline approximation."""
        return BSpline(knots, coeffs, k)

    # 3. Define the Objective Function (Mean Squared Error)
    def objective_function(coeffs, x_data, y_data, knots, k):
        """the objective function to be optimized."""
        spline_func = get_spline_from_coeffs(coeffs, knots, k)
        y_pred = spline_func(x_data)
        return np.sum((y_pred - y_data) ** 2)

    # 4. Define Convexity Constraints
    # f''(x) >= 0 for all x. So, the constraint function should return f''(x).
    x_eval_convexity = np.linspace(x.min(), x.max(), 100)  # Check at 100 points

    def convexity_constraint_func(coeffs, x_eval, knots, k):
        """Second derivative of the spline at ``x_eval``, signed by ``convexity``."""
        spline_func = get_spline_from_coeffs(coeffs, knots, k)
        spline_deriv2_func = spline_func.derivative(nu=2)
        deriv2_values = spline_deriv2_func(x_eval)
        return (
            deriv2_values * convexity
        )  # We want deriv2_values >= 0 for 'greater_equal' constraint

    constraints = [
        {
            "type": "ineq",
            "fun": convexity_constraint_func,
            "args": (x_eval_convexity, knots, k),
        }
    ]

    # Initial guess for coefficients
    initial_coeffs = np.zeros(n_coeffs)

    # 5. Run the Optimization
    result = minimize(
        objective_function,
        initial_coeffs,
        args=(x, y, knots, k),
        method="SLSQP",
        constraints=constraints,
        options={"disp": False, "maxiter": 1000},
    )

    if not result.success:
        logger.warning(f"Optimization did not converge successfully: {result.message}")

    optimized_coeffs = result.x
    convex_spline_func = get_spline_from_coeffs(optimized_coeffs, knots, k)

    # 6. Evaluate the first derivative
    spline_deriv1_func = convex_spline_func.derivative(nu=1)
    deriv_values_at_original_x = spline_deriv1_func(x)

    # The spline was evaluated over the sorted x. Restore the order of the input.
    result = np.empty_like(deriv_values_at_original_x)
    result[sort_idx] = deriv_values_at_original_x
    return result.tolist()
