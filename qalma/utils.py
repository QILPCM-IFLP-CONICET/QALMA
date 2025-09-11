"""
Utility functions to import and process ALPS specification files.
"""

import logging

import numpy as np
import qutip  # type: ignore[import-untyped]
from matplotlib.patches import Circle, Ellipse
from matplotlib.pyplot import Axes as PLTAxes
from numpy.random import rand

default_parms = {
    "pi": 3.1415926,
    "e": 2.71828183,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "rand": rand,
}


def draw_ellipse_around_points(p1, p2, ax, b_ratio=0.15):
    """
    Draw an ellipse containing p1 and p2 located over the main axis,
    symmetrically around the center.
    """
    # Compute center
    x1, y1 = p1
    x2, y2 = p2
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2

    # Compute major axis length (2a) and angle
    dx, dy = x2 - x1, y2 - y1
    a = np.hypot(dx, dy) / 2
    angle = np.degrees(np.arctan2(dy, dx))  # degrees for matplotlib

    # Minor axis length
    b = a * b_ratio

    # Plotting
    ellipse = Ellipse(
        (xc, yc),
        width=3 * a,
        height=2.1 * b,
        angle=angle,
        edgecolor=(0, 0, 1, 0.2),
        facecolor=(0, 0.3, 1, 0.2),
        lw=2,
    )

    ax.add_patch(ellipse)


def draw_operator(op, axis: PLTAxes) -> PLTAxes:
    """
    Draw the operator op over the axis.

    Parameters
    ----------

    op: Operator
      If the operator acts on a single site, draws a disk on its coordinates.
      If is a SumOperator, flatten it and draw each term.
      For many-body operators, a line is drawn.
    ax: mpl.Axis
      the axis over which the operator is going to be drawn.

    Return
    ------
    mpl.Axis
      the axis over which the operator was drawn.
    """
    # TODO: handle 3D graphs
    from qalma.operators import SumOperator

    system = op.system
    g = system.spec["graph"]
    g.complete_coordiantes()
    op = op.flat()
    if isinstance(op, SumOperator):
        for term in op.terms:
            draw_operator(term, axis)
        return axis
    acts_over = op.acts_over()
    if acts_over is not None:
        coords = [g.nodes[site]["coords"] for site in acts_over]
        coords = [(x[0], 0) if len(x) == 1 else x for x in coords]
        if len(coords) == 1:
            axis.add_artist(Circle(coords[0], 0.1))
        if len(coords) == 2:
            draw_ellipse_around_points(coords[0], coords[1], axis)
        else:
            axis.plot(
                [x[0] for x in coords] + [coords[0][0]],
                [x[1] for x in coords] + [coords[0][1]],
                lw="5",
                c="red",
            )
    return axis


def eval_expr(expr: str, parms: dict):
    """
    Evaluate the expression `expr` replacing the variables defined in `parms`.
    expr can include python`s arithmetic expressions, and some elementary
    functions.
    """
    # TODO: Improve the workflow in a way that numpy functions
    # and constants be loaded just if they are needed.

    if not isinstance(expr, str):
        return expr

    try:
        return float(expr)
    except (ValueError, TypeError):
        try:
            if expr not in ("J", "j"):
                return complex(expr)
        except (ValueError, TypeError):
            pass

    parms = {
        key.replace("'", "_prima"): val for key, val in parms.items() if val is not None
    }
    expr = expr.replace("'", "_prima")

    while expr in parms:
        expr = parms.pop(expr)
        if not isinstance(expr, str):
            return expr

    # Reduce the parameters
    p_vars = list(parms)
    while True:
        changed = False
        for k in p_vars:
            val = parms.pop(k)
            if not isinstance(val, str):
                parms[k] = val
                continue
            try:
                result = eval_expr(val, parms)
                if result is not None:
                    parms[k] = result
                if val != result:
                    changed = True
            except RecursionError:
                logging.warning(f"A recursion error happens evaluating `{val}`.")
                raise
        if not changed:
            break
    parms.update(default_parms)
    try:
        result = eval(expr, parms)
        return result
    except NameError:
        pass
    except TypeError as exc:
        logging.warning(f"Type Error. Undefined variables in [{exc}] in {expr}.")
        return None
    except SyntaxError:
        logging.error(
            (
                "expression " f"<<{expr}>>",
                f"\n   with parameters\n{parms}\n" "raised a SyntaxError",
            )
        )
        raise
    return expr


def find_ref(node, root):
    """
    Find a node in the root

    Parameters
    ----------
    node : TYPE
        the key of the node.
    root : dict
        a nested tree structure of
        dicts

    Returns
    -------
    dict
        the node corresponding to `node`.

    """
    node_items = dict(node.items())
    if "ref" in node_items:
        name_ref = node_items["ref"]
        for refnode in root.findall("./" + node.tag):
            if ("name", name_ref) in refnode.items():
                return refnode
    return node


def operator_to_wolfram(operator) -> str:
    """
    Produce a string with a Wolfram Mathematica expression
    representing the operator.
    """
    from qalma.operators.arithmetic import SumOperator
    from qalma.operators.basic import LocalOperator, Operator, ProductOperator
    from qalma.operators.qutip import Qobj

    def get_site_identity(site_name):
        site_spec = sites[site_name]
        if "operators" in site_spec:
            return site_spec["operators"]["identity"]
        dim = dimensions[site_name]
        result = qutip.qeye(dim)
        site_spec["operators"] = {"identity": result}
        return result

    if hasattr(operator, "to_wolfram"):
        return operator.to_wolfram()

    if isinstance(operator, Qobj):
        data = operator.data
        if hasattr(data, "toarray"):
            array = data.toarray()
        elif hasattr(data, "to_array"):
            array = data.to_array()
        else:
            raise TypeError((f"Do not know how to convert {type(data)} into a ndarray"))

        assert len(array.shape) == 2, f"the shape  {array.shape} is not a matrix"
        return matrix_to_wolfram(array)

    if isinstance(operator, SumOperator):
        assert all(isinstance(term, Operator) for term in operator.terms)
        terms = [operator_to_wolfram(term) for term in operator.terms]
        terms = [term for term in terms if term != "0"]
        return "(" + " + ".join(terms) + ")"

    sites = operator.system.sites
    dimensions = operator.system.dimensions
    prefactor = operator.prefactor
    if prefactor == 0:
        return "0"

    prefix = "KroneckerProduct["
    if prefactor != 1:
        prefix = f"({prefactor}) * " + prefix

    if isinstance(operator, LocalOperator):
        local_site = operator.site
        factors = [
            operator.operator if site == local_site else get_site_identity(site)
            for site in sorted(dimensions)
        ]
        factors_str = [operator_to_wolfram(factor) for factor in factors]

        return prefix + ", ".join(factors_str) + "]"

    if isinstance(operator, ProductOperator):
        factors = [
            operator.sites_op.get(site, get_site_identity(site))
            for site in sorted(dimensions)
        ]
        factors_str = [operator_to_wolfram(factor) for factor in factors]

        return prefix + ", ".join(factors_str) + "]"

    if hasattr(operator, "prefactor"):
        return (
            "("
            + str(prefactor)
            + ") * "
            + operator_to_wolfram((operator / prefactor).to_qutip())
        )

    return operator_to_wolfram(operator.to_qutip())


def matrix_to_wolfram(matr: np.ndarray):
    """Produce a string representing the data in the matrix"""
    assert isinstance(
        matr, (np.ndarray, complex, float)
    ), f"{type(matr)} is not ndarray or number"

    def process_number(num):
        assert isinstance(
            num, (float, complex)
        ), f"{type(num)} {num} is not float or complex."
        if isinstance(num, np.ndarray) and len(num) == 1:
            num = num[0]

        if isinstance(num, complex):
            if num.imag == 0:
                return str(num.real).replace("e", "*^")
        return (
            str(num)
            .replace("(", "")
            .replace(")", "")
            .replace("e", "*^")
            .replace("j", "I")
        )

    rows = [
        "{" + (", ".join(process_number(elem) for elem in row)) + "}" for row in matr
    ]
    return "{\n" + ",\n".join(rows) + "\n}"


def next_name(dictionary: dict, s: int = 1, prefix: str = "") -> str:
    """
    Produces a new key for the `dictionary` with a
    `prefix`
    """
    name = f"{prefix}{s}"
    if name in dictionary:
        return next_name(dictionary, s + 1, prefix)
    return name


def replace_variable_type(val, e_type):
    """
    if `val` is a str representing an unevaluated
    expression, replace occurrences of `#` by
    `e_type`.
    """
    if isinstance(val, str):
        return val.replace("#", f"{e_type}")
    return val
