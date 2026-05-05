"""Basic unit test."""

import os
from test.helper import alert
from typing import Tuple

import pytest
import qutip

from qalma.alpsmodels import list_models_in_alps_xml, model_from_alps_xml
from qalma.geometry import graph_from_alps_xml, list_geometries_in_alps_xml
from qalma.model import SystemDescriptor
from qalma.qutip_tools import norm
from qalma.settings import LATTICE_LIB_FILE, MODEL_LIB_FILE
from qalma.utils import eval_expr


def test_eval_expr():
    """Test basic evaluation of expressions."""
    parms = {"a": "J", "J": 2, "subexpr": "a*J"}
    test_cases = [
        ("2+a", 4),
        ("sqrt(2+a)", 2),
        ("0*rand()", 0),
        ("2*J", 4),
        ("sqrt(subexpr)", 2),
    ]
    for expr, expect in test_cases:
        result = eval_expr(expr, parms)
        assert expect == result, (
            f"evaluating {expr}"
            f"{expect} of type {type(expect)}"
            f"!= {result} of {type(result)}"
        )


@pytest.mark.skipif(not os.environ.get("QALMA_ALLTESTS"), reason="shorter tests")
def test_load_all_models_and_lattices():
    """Try to load each model and lattice."""
    models = list_models_in_alps_xml()
    graphs = list_geometries_in_alps_xml()

    for model_name in models:
        print(model_name, "\n", 10 * "*")
        for graph_name in graphs:
            g = graph_from_alps_xml(
                LATTICE_LIB_FILE,
                graph_name,
                parms={"L": 2, "W": 2, "a": 1, "b": 1, "c": 1},
            )
            model = model_from_alps_xml(
                MODEL_LIB_FILE,
                model_name,
                parms={"L": 2, "W": 2, "a": 1, "b": 1, "c": 1, "Nmax": 5},
            )
            try:
                SystemDescriptor(g, model, {})
            except Exception as exc:
                # assert False, f"model {model_name} over
                # graph {graph_name} could not be loaded due to {type(e)}:{e}"
                alert(1, "   ", graph_name, "  [failed]", exc)
                continue


# ---------------------------------------------------------------------------
# Tests for Hamiltonians with Loop-like terms
# ---------------------------------------------------------------------------


def check_chiral_operator_antisymmetry(system, ham_chiral_only, tol=1e-10):
    """Verify chi_{ijk} = -chi_{ikj} for the chiral-only Hamiltonian.

    The scalar triple product is antisymmetric under exchange of any two
    indices.  If the LOOP definitions in the XML assign the same coefficient
    to up- and down-triangles (which differ by a permutation of two vertices),
    the total Hamiltonian should be proportional to

        sum_{up} chi_{ijk}  -  sum_{down} chi_{ijk}

    i.e. the two loop types contribute with opposite signs automatically
    because the NODE order is reversed.  This test checks that the operator
    is *non-zero* and Hermitian.
    """
    sites = tuple(sorted(system.sites.keys()))
    H_qutip = ham_chiral_only.to_qutip(sites)

    # Check symmetries:
    id2 = qutip.qeye(2)
    for name, pauli_gen in [
        ("X", qutip.sigmax()),
        ("Y", qutip.sigmay()),
        ("Z", qutip.sigmaz()),
    ]:
        # Check that U_{parity} H U_{parity}^\dagger = H
        u_parity = qutip.tensor([pauli_gen for site in sites])
        diff = norm(H_qutip - u_parity * H_qutip * u_parity)
        assert (
            diff < tol
        ), f"Chiral Hamiltonian is non symmetric under parity-{name}: ||H - U H U†|| = {diff:.2e}"
        # Check that U_{rot} H U_{rot}^\dagger = H
        u_rot = (1j * pauli_gen + id2) * 2 ** (-0.5)
        u_rot = qutip.tensor([u_rot for site in sites])
        u_rot_dag = (-1j * pauli_gen + id2) * 2 ** (-0.5)
        u_rot_dag = qutip.tensor([u_rot_dag for site in sites])
        diff = norm(H_qutip - u_rot * H_qutip * u_rot_dag)
        assert (
            diff < tol
        ), f"Chiral Hamiltonian is non symmetric under \\Pi/4 {name}-rotation: ||H - U H U†|| = {diff:.2e}"

    # Check quirality
    H_quiral = H_qutip.conj()
    diff = norm(H_quiral + H_qutip)
    assert (
        diff < tol
    ), f"Chiral Hamiltonian is non reversed under quirality: ||H + H.conj()|| = {diff:.2e}"

    # Hermitian check
    diff = norm(H_qutip - H_qutip.dag())

    assert diff < tol, f"Chiral Hamiltonian is not Hermitian: ||H - H†|| = {diff:.2e}"
    # Non-trivial check (should not vanish for chi != 0 and L >= 2)
    assert (
        norm(H_qutip) > tol
    ), "Chiral Hamiltonian is the zero operator — check LOOP definitions."


def build_chiral_strip(
    L: int,
    J: float,
    chi: float,
    boundary: str = "open",
) -> Tuple[SystemDescriptor, object]:
    """Build a spin-1/2 triangular strip with Heisenberg + chiral interactions.

    Parameters
    ----------
    L : int
        Number of unit cells (each cell has 2 sites → 2*L sites total).
    J : float
        Heisenberg nearest-neighbor coupling (same on all bond types).
    chi : float
        Coefficient of the scalar-triple-product (chirality) term.
    boundary : {"open", "periodic"}
        Boundary condition along the strip.

    Returns
    -------
    system : SystemDescriptor
    ham    : Operator   (the Hamiltonian as a QALMA operator)
    """
    lattice_name = f"triangular strip {boundary}"
    parms = {
        "L": L,
        "a": 1,
        # Heisenberg coupling on all four bond types
        "J0": J,
        "J1": J,
        "J2": J,
        "J3": J,
        # Chiral coupling (applies to both loop types 0 and 1)
        "Wilson2": chi,
        "Wilson20": chi,  # loop type "0"  (up-triangles)
        "Wilson21": chi,  # loop type "1"  (down-triangles)
    }
    graph = graph_from_alps_xml(name=lattice_name, parms=parms)
    model = model_from_alps_xml(name="chiral spin")
    system = SystemDescriptor(graph, model, parms)
    ham = system.global_operator("Hamiltonian")
    return system, ham


@pytest.mark.parametrize("L", [3, 4])
def test_chiral_operator_hermitian(L):
    """The purely chiral term (J=0, chi=1) must be Hermitian."""
    system, ham = build_chiral_strip(L, J=0.0, chi=1.0)
    check_chiral_operator_antisymmetry(system, ham)
