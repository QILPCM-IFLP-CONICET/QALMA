"""
Consistency tests for the variational mean-field approximation.

Three independent properties are verified:

1. Monotonicity  — compute_free_energy(sigma_nf, H) is non-increasing in
   numfields, where free_energy(sigma, H) = Tr[sigma*(H + log sigma)]
   approximates S(sigma || e^{-H}/Z) up to the constant log Z.
2. Solvable models — variational_quadratic_mfa finds a non-trivial state that
   strictly improves the free energy over the fully mixed state for quasi-free
   models (transverse Ising, XX chain).  The transverse field must be set via
   the 'Gamma' parameter of the 'spin' model ('h' couples to Sz, not Sx).
3. Spiral order — for the frustrated J1-J2 chain with J2/J1 > 0.5 the
   variational state detects incommensurate magnetic order via the dominant
   Fourier mode of the <Sz_i> profile.
"""

import numpy as np
import pytest

from qalma.meanfield.variational import compute_free_energy, variational_quadratic_mfa
from qalma.model import build_system
from qalma.operators.states import ProductDensityOperator

# ── helpers ───────────────────────────────────────────────────────────────────


def _spin_chain_nn(L: int, Jz: float = 1.0, Jxy: float = 1.0, Gamma: float = 0.0):
    """
    SystemDescriptor for a spin-1/2 chain with nearest-neighbour bonds.

    Uses the 'chain lattice' geometry with the 'spin' model.
    Jz0, Jxy0 are the bond-0 couplings; Gamma is the transverse field
    (couples to Sx).  Note: 'h' couples to Sz (longitudinal) and should
    not be used to build transverse-field models.
    """
    params = {"L": L, "a": 1, "Gamma": Gamma, "J": 1, "Jz0": Jz, "Jxy0": Jxy}
    return build_system("chain lattice", "spin", **params)


def _spin_chain_j1j2(L: int, J1: float = 1.0, J2: float = 0.5):
    """
    SystemDescriptor for the frustrated J1-J2 open chain.

    Uses the 'nnn open chain lattice' geometry so that bond type 0 carries
    J1 (NN) and bond type 1 carries J2 (NNN).  The parameter names 'Jz1'
    and 'Jxy1' are aliases for "Jz'" and "Jxy'" in the ALPS model spec.
    """
    params = {
        "L": L,
        "a": 1,
        "Gamma": 0.0,
        "J": 1,
        "Jz0": J1,
        "Jxy0": J1,
        "Jz1": J2,
        "Jxy1": J2,
    }
    return build_system("nnn open chain lattice", "spin", **params)


def _hamiltonian(system):
    """Return the Hamiltonian operator of *system*."""
    return system.global_operator("Hamiltonian")


def _free_energy_mixed(system, H) -> float:
    """
    Free energy of the fully mixed state sigma = I/d.

    F[I/d] = Tr[(I/d) * H] + Tr[(I/d) * log(I/d)]
           = <H>_mixed - log(d)

    This is the worst-case baseline: any non-trivial variational state
    should strictly improve on this value.
    """
    mixed = ProductDensityOperator({}, system=system)
    return compute_free_energy(mixed, H)


def _sz_profile(state: ProductDensityOperator, system) -> np.ndarray:
    """Return the array [<Sz_i>]_{i=0..L-1} for *state*."""
    sites = list(system.sites.keys())
    return np.array(
        [float(np.real(state.expect(system.site_operator(f"Sz@{s}")))) for s in sites]
    )


# ── Test 1: monotonicidad de la energía libre con numfields ───────────────────


@pytest.mark.parametrize(
    "L,Jz,Jxy,Gamma,label",
    [
        (4, 1.0, 0.0, 1.0, "ising_L4"),
        (4, 0.0, 1.0, 0.0, "xx_L4"),
        (4, 1.0, 1.0, 0.0, "xxx_L4"),
        (6, 1.0, 0.0, 1.0, "ising_L6"),
    ],
)
def test_free_energy_nonincreasing_with_numfields(L, Jz, Jxy, Gamma, label):
    """
    compute_free_energy(sigma_nf, H) must be non-increasing as numfields grows.

    free_energy(sigma, H) = Tr[sigma*(H + log sigma)] equals
    S(sigma || e^{-H}/Z) - log Z, so minimising it is equivalent to
    minimising the true relative entropy.  With a larger variational family
    the minimum can only decrease or stay the same.

    We scan numfields in {1, 2, 4, L-1} and verify monotonicity with a
    small numerical slack of 1e-5 to absorb optimizer noise.
    """
    system = _spin_chain_nn(L, Jz=Jz, Jxy=Jxy, Gamma=Gamma)
    H = _hamiltonian(system)
    nf_values = sorted({1, 2, min(4, L - 1), L - 1})

    prev_fe = np.inf
    for nf in nf_values:
        sigma = variational_quadratic_mfa(_BETA * H, numfields=nf)
        fe = compute_free_energy(sigma, _BETA * H)
        assert fe <= prev_fe + 1e-5, (
            f"[{label}] Free energy NOT non-increasing: "
            f"F(numfields={nf}) = {fe:.6f} > F_prev = {prev_fe:.6f}"
        )
        prev_fe = fe


# ── Test 2: mejora sobre el estado mixto en modelos solubles ──────────────────


@pytest.mark.parametrize(
    "L,Jz,Jxy,Gamma,label",
    [
        (4, 1.0, 0.0, 1.0, "ising_L4"),
        (4, 0.0, 1.0, 0.0, "xx_L4"),
        (6, 1.0, 0.0, 1.0, "ising_L6"),
        (6, 0.0, 1.0, 0.0, "xx_L6"),
    ],
)
def test_variational_improves_over_mixed_state(L, Jz, Jxy, Gamma, label):
    """
    For quasi-free models (transverse Ising, XX chain) the variational MFA
    must strictly improve the free energy over the fully mixed state sigma=I/d.

    The fully mixed state is the worst-case baseline: it ignores all
    structure of H.  A working variational method must always beat it for
    any non-trivial Hamiltonian.

    We do not compare against the exact thermal state here — the MFA is an
    approximation and its gap to the exact solution is a physics result, not
    a correctness criterion.  The correctness criterion is that the optimizer
    finds a better state than doing nothing.
    """
    system = _spin_chain_nn(L, Jz=Jz, Jxy=Jxy, Gamma=Gamma)
    H = _hamiltonian(system)

    fe_mixed = _free_energy_mixed(system, _BETA * H)
    sigma_mf = variational_quadratic_mfa(_BETA * H, numfields=L - 1)
    fe_mf = compute_free_energy(sigma_mf, _BETA * H)

    assert fe_mf < fe_mixed - 1e-4, (
        f"[{label}] Variational MFA did not improve over the mixed state: "
        f"F_mf={fe_mf:.6f}, F_mixed={fe_mixed:.6f}"
    )


# ── Test 3: detección de orden espiral en la cadena J1-J2 ─────────────────────

_BETA = 3.0  # temperatura inversa; a beta=1 el MFA no rompe simetría en XX ni J1-J2
_POINT_FIELD = 1e-2  # amplitud del campo puntual en el sitio central
_NUMFIELDS = 6  # independiente de L: captura al menos 2 modos no triviales
# dado que H XXX tiene degeneración 3 en la matriz de acoplamientos


@pytest.mark.parametrize(
    "L,J2_over_J1,label",
    [
        (12, 0.6, "j1j2_L12_J2=0.6"),
        (12, 0.8, "j1j2_L12_J2=0.8"),
    ],
)
def test_j1j2_spiral_order_detection(L, J2_over_J1, label):
    """
    For J2/J1 > 0.5 the J1-J2 chain develops incommensurate (spiral) magnetic
    order.  We verify that variational_quadratic_mfa detects non-trivial
    oscillatory structure in the <Sz_i> profile.

    Strategy
    --------
    A small field epsilon * Sz_{L//2} is applied at the central site to break
    SU(2) and spatial reflection symmetry without imposing any particular
    wavevector.  This seeds the optimizer away from the trivial sigma ~ I/d
    minimum while leaving the system free to select its own q*.

    The Hamiltonian is scaled by beta=_BETA to work at low enough temperature
    for the MFA to find a symmetry-broken solution.

    numfields=_NUMFIELDS is chosen from the structure of H: SU(2) symmetry
    gives a 3-fold degeneracy in the coupling matrix eigenvalues, so 6 fields
    captures at least two independent non-trivial modes regardless of L.

    Since the MFA state is a product state, the connected correlator
    <Sz_i Sz_j>_c = 0 identically, so the accessible observable is the
    magnetization profile <Sz_i> itself.  We check two conditions:

    1. The profile is not homogeneous (the perturbation is felt).
    2. The profile has at least one sign change beyond simple AF order:
       the dominant Fourier mode is neither k=0 (FM) nor k=L//2 (AF, q=pi).

    For L=12 the classical spiral wavevector q* = arccos(-J1/4J2) gives:
      J2/J1=0.6: q* ~ 0.408*pi  (between k=2 and k=3, closest to k=2)
      J2/J1=0.8: q* ~ 0.450*pi  (between k=2 and k=3, closest to k=3)
    Both are safely incommensurate for L=12.
    """
    J1 = 1.0
    J2 = J2_over_J1 * J1
    system = _spin_chain_j1j2(L, J1=J1, J2=J2)
    H = _hamiltonian(system)

    # Small field at the central site: breaks SU(2) and spatial reflection
    # without imposing a wavevector
    sites = list(system.sites.keys())
    central_site = sites[L // 2]
    H_perturbed = H + _POINT_FIELD * system.site_operator(f"Sz@{central_site}")

    sigma = variational_quadratic_mfa(_BETA * H_perturbed, numfields=_NUMFIELDS)
    sz = _sz_profile(sigma, system)

    # Structure factor S(q) = |FFT(<Sz_i>)|^2
    # rfft gives modes k = 0, 1, ..., L//2  (q = 2*pi*k/L)
    sq = np.abs(np.fft.rfft(sz)) ** 2
    dominant_k = int(np.argmax(sq))
    af_k = len(sq) - 1  # k = L//2 corresponds to q = pi

    assert sz.std() > 1e-4, (
        f"[{label}] <Sz_i> profile is homogeneous (std={sz.std():.2e}); "
        f"central-site perturbation had no effect. "
        f"Profile: {sz.round(4)}"
    )
    assert dominant_k not in (0, af_k), (
        f"[{label}] Dominant mode k={dominant_k} is FM (k=0) or AF (k={af_k}), "
        f"not spiral. "
        f"S(q): {sq.round(4)}, <Sz_i>: {sz.round(4)}"
    )
