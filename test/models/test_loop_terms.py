"""Global operators built from loop terms (``LOOPTERM`` in the ALPS models)."""

import pytest

from qalma import build_system


def test_loop_term_with_undefined_site_operators_is_dropped():
    """A loop term whose operators are not defined on the sites is skipped.

    ``loop_term`` in ``models.xml`` uses ``Sx``, ``Sy`` and ``Sz``, which the
    boson sites do not define. Like global operators built from bond terms
    in the same situation, it must be dropped, and the system must still be
    built.
    """
    system = build_system("square lattice with loop", "boson Hubbard", L=2, W=2, Nmax=1)
    global_ops = system.operators["global_operators"]
    assert "loop_term" not in global_ops
    assert "Hamiltonian" in global_ops


def test_loop_term_with_undefined_site_operators_raises_value_error():
    """Evaluating such a loop term directly raises a ``ValueError``."""
    system = build_system("square lattice with loop", "boson Hubbard", L=2, W=2, Nmax=1)
    spec = system.spec
    model = spec["model"]
    graph = spec["graph"]
    term_spec = {
        "expr": "Sx@[i]*Sy@[j]*Sz@[k]",
        "indices": ("i", "j", "k"),
    }
    with pytest.raises(ValueError, match="could not be evaluated"):
        system.loop_term_from_descriptor(term_spec, graph, model, {})


def test_loop_term_on_spin_sites():
    """On spin sites the loop term is built."""
    system = build_system("square lattice with loop", "spin", L=2, W=2)
    assert "loop_term" in system.operators["global_operators"]
