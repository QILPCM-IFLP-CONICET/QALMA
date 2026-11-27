"""Convergence criterion and warnings of ``self_consistent_mf``."""

import logging
import re

import pytest
import qutip

from qalma.meanfield import variational
from qalma.meanfield.variational import self_consistent_mf
from qalma.model import build_system
from qalma.operators.states.product import ProductDensityOperator
from qalma.settings import QALMA_TOLERANCE

LOGGER_NAME = "qalma.meanfield.variational"


def _chain(length, gamma, beta, amplitude=0.4):
    """Generator of a TFIM chain and a staggered product state."""
    system = build_system(
        "open chain lattice", "spin", L=length, a=1, Jz0=1.0, Jxy0=0.0, Gamma=gamma
    )
    names = sorted(system.sites, key=lambda name: int(name.split("[")[1][:-1]))
    generator = (beta * system.global_operator("Hamiltonian")).simplify()
    seed = ProductDensityOperator(
        {
            name: 0.5 * (qutip.qeye(2) + amplitude * (-1) ** pos * qutip.sigmaz())
            for pos, name in enumerate(names)
        },
        system=system,
    )
    return generator, seed


def _warnings(caplog):
    """Messages of the warnings logged by the mean-field module."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == LOGGER_NAME and record.levelno >= logging.WARNING
    ]


def _fake_free_energy(monkeypatch, values):
    """Replace the free energy by a fixed sequence of values."""
    iterator = iter(values)
    monkeypatch.setattr(
        variational, "compute_free_energy", lambda sigma, ham: next(iterator)
    )


def test_rounding_noise_of_a_large_free_energy_is_converged(monkeypatch, caplog):
    """A change of the size of the rounding noise of ``F`` is convergence.

    For ``|F| ~ 5000`` one unit in the last place is ``~1e-12``, much larger
    than ``QALMA_TOLERANCE``: the tolerance must be relative.
    """
    noise = 0.5 * QALMA_TOLERANCE * 5000.0
    generator, seed = _chain(2, 2.5, 10.0)
    _fake_free_energy(monkeypatch, [-5000.0, -5000.5, -5000.5 + noise])
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        _, rel_entropy = self_consistent_mf(generator, seed, max_steps=10)
    assert _warnings(caplog) == []
    assert rel_entropy == -5000.5


def test_low_temperature_state_of_many_sites_converges_without_warning(caplog):
    """A paramagnetic chain at low temperature, with ``F = -beta*Gamma*N/2``."""
    generator, seed = _chain(20, 2.5, 160.0)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        _, rel_entropy = self_consistent_mf(generator, seed, max_steps=200)
    assert _warnings(caplog) == []
    assert rel_entropy == pytest.approx(-160.0 * 2.5 * 20 / 2, rel=1e-9)


def test_increase_of_the_free_energy_is_reported(monkeypatch, caplog):
    """If ``F`` increases, the loop stops and returns the previous state."""
    generator, seed = _chain(2, 2.5, 10.0)
    _fake_free_energy(monkeypatch, [-10.0, -9.0])
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        sigma, rel_entropy = self_consistent_mf(generator, seed, max_steps=10)
    messages = _warnings(caplog)
    assert len(messages) == 1 and "increased" in messages[0]
    assert sigma is seed
    assert rel_entropy == -10.0


def test_exhausted_steps_report_the_last_change(caplog):
    """The warning counts the steps done and gives the last change of ``F``."""
    generator, seed = _chain(8, 1.0, 40.0)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        self_consistent_mf(generator, seed, max_steps=3)
    messages = _warnings(caplog)
    assert len(messages) == 1
    assert "after 3 steps" in messages[0]
    change = re.search(r"was (-?[0-9.]+e[+-][0-9]+)", messages[0])
    assert change is not None and float(change.group(1)) != 0.0


def test_zero_steps_return_the_initial_state(caplog):
    """Without steps, the initial state is returned (and a warning logged)."""
    generator, seed = _chain(4, 1.0, 10.0)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        sigma, _ = self_consistent_mf(generator, seed, max_steps=0)
    assert sigma is seed
    assert len(_warnings(caplog)) == 1
