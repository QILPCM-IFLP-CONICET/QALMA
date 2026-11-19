"""n-body projection of dense blocks relative to product states."""

import numpy as np
import pytest
import qutip

from qalma import build_system
from qalma.operators import QutipOperator
from qalma.operators.states import GibbsProductDensityOperator
from qalma.projections import n_body_projection
from qalma.projections.nbody import _project_dense_block, project_to_n_body_operator


@pytest.fixture(scope="module")
def chain():
    """A five-site spin chain and its local operators."""
    system = build_system("open chain lattice", "spin", L=5, J=1.0)
    sites = tuple(system.sites)
    sx = [system.site_operator(f"Sx@{site}") for site in sites]
    sz = [system.site_operator(f"Sz@{site}") for site in sites]
    return system, sites, sx, sz


def random_block(system, sites, rng, size):
    """A random, non-Hermitian dense block on ``size`` random sites."""
    chosen = list(rng.choice(sites, size=size, replace=False))
    dim = 2**size
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    qobj = qutip.Qobj(matrix, dims=[[2] * size, [2] * size])
    names = {site: pos for pos, site in enumerate(chosen)}
    return QutipOperator(qobj, system, names=names) * (0.7 + 0.2j)


@pytest.mark.parametrize("size,n_max", [(3, 2), (4, 2), (4, 3)])
def test_dense_projection_matches_product_expansion(chain, size, n_max):
    """The direct projection equals the projection of the product expansion."""
    system, sites, sx, sz = chain
    rng = np.random.default_rng(size + 10 * n_max)
    for _ in range(3):
        kappa = sum(rng.normal() * sx[i] + rng.normal() * sz[i] for i in range(5))
        sigma = GibbsProductDensityOperator(kappa.simplify()).to_product_state()
        block = random_block(system, sites, rng, size)
        direct = _project_dense_block(block, n_max, sigma)
        expanded = project_to_n_body_operator(block.as_sum_of_products(), n_max, sigma)
        diff = (direct - expanded).to_qutip(sites).full()
        assert np.abs(diff).max() < 1e-12
        assert direct.n_body_sector() <= n_max


def test_dense_projection_is_a_projection(chain):
    """Projecting twice gives the same operator, and n-body operators are fixed."""
    system, sites, sx, sz = chain
    rng = np.random.default_rng(3)
    sigma = GibbsProductDensityOperator((0.3 * sx[0] - 0.4 * sz[2]).simplify())
    sigma = sigma.to_product_state()
    block = random_block(system, sites, rng, 3)
    once = n_body_projection(block, n_max=2, sigma=sigma)
    twice = n_body_projection(once, n_max=2, sigma=sigma)
    assert np.abs((once - twice).to_qutip(sites).full()).max() < 1e-12


def test_dense_projection_without_reference(chain):
    """Without a reference state, the maximally mixed state is used."""
    system, sites, sx, sz = chain
    rng = np.random.default_rng(5)
    block = random_block(system, sites, rng, 3)
    mixed = GibbsProductDensityOperator({}, system).to_product_state()
    a = _project_dense_block(block, 2, None)
    b = _project_dense_block(block, 2, mixed)
    assert np.abs((a - b).to_qutip(sites).full()).max() < 1e-12
