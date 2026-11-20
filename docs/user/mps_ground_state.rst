.. _mps_ground_state:

Ground states as matrix product states
======================================

.. contents::
   :depth: 2
   :local:

Overview
--------

.. note::

   The MPS ground-state operator is experimental: its interface may change
   in future releases.

:class:`~qalma.mps.gs_operator.MPSGSOperator` is a density operator
:math:`\rho = |\psi_0\rangle\langle\psi_0|`, where :math:`|\psi_0\rangle` is
the ground state of a Hamiltonian :math:`H`. The state is never built as a
dense matrix. It is stored as a matrix product state (MPS) computed with a
two-site DMRG, so it can describe systems that are far beyond exact
diagonalization, as long as their entanglement is moderate (typically
one-dimensional or quasi-one-dimensional systems).

It behaves like the other states of QALMA:
:meth:`~qalma.mps.gs_operator.MPSGSOperator.expect` and
:meth:`~qalma.mps.gs_operator.MPSGSOperator.partial_trace` work with any
operator, the arithmetic of density operators is available, and the reduced
states are :class:`~qalma.operators.states.qutip.QutipDensityOperator`
objects, so they can be used everywhere a reduced state is needed (for
instance, as a reference for the reductions of Gibbs states in :doc:`rpa`).

Two differences with :class:`~qalma.operators.states.gibbs.GibbsDensityOperator`
are worth keeping in mind:

* ``MPSGSOperator`` is a *sibling* of ``GibbsDensityOperator``, not a
  subclass. A pure state has no finite generator :math:`K` with
  :math:`\rho \propto e^{-K}`, so the attribute ``k`` does not exist and
  ``logm`` raises ``NotImplementedError``.
* The ground state is assumed to be **non-degenerate**. For a degenerate
  ground-state manifold the limit :math:`\beta\to\infty` of the Gibbs state is
  a mixture over the manifold, which this class does not represent.

Quick start
-----------

.. code-block:: python

   from qalma import build_system
   from qalma.mps import MPSGSOperator

   n_sites = 12
   system = build_system("open chain lattice", "spin", L=n_sites)
   H = system.global_operator("Hamiltonian")

   # The order of the sites along the MPS (see the next section).
   order = [f"1[{i}]" for i in range(n_sites)]

   rho = MPSGSOperator(H, permutation=order, chi_max=32)
   print(rho.energy)      # -5.14209063   (exact: -5.14209063)
   print(rho)             # MPSGSOperator(12 sites, max bond=32, E0=-5.1420906)

The DMRG runs lazily, the first time the state is needed (here, when
``rho.energy`` is read). Call :meth:`~qalma.mps.gs_operator.MPSGSOperator.solve`
to run it explicitly, or to run it again with other options.

Expectation values of operators of any support are computed from the MPS:

.. code-block:: python

   Sz = lambda i: system.site_operator("Sz", f"1[{i}]")

   rho.expect(Sz(0) * Sz(1))     # -0.21876
   rho.expect(Sz(0) * Sz(2))     #  0.06609
   rho.expect(Sz(0) * Sz(5))     # -0.04308
   rho.expect(H)                 # -5.14209063

Reduced states of a few sites are
:class:`~qalma.operators.states.qutip.QutipDensityOperator` objects:

.. code-block:: python

   reduced = rho.partial_trace(frozenset({"1[0]", "1[1]"}))
   reduced.to_qutip(("1[0]", "1[1]"))      # 4x4 density matrix

How the observables are evaluated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``expect`` splits the observable into terms and asks the state for the
reduced state of the sites each term acts on. The cost of a reduced state
grows exponentially with the number of sites kept (a dense
:math:`d^{|A|}\times d^{|A|}` matrix is built) and linearly with the distance
along the chain between the first and the last of them. Terms acting on a few
sites are therefore cheap, even if the sites are far apart, while terms acting
on many sites are not. There is no shortcut for products of many operators:
reduce the problem to few-body terms.

Order of the sites along the chain
----------------------------------

An MPS is a chain, but the sites of a QALMA system are labelled by names.
By default, the chain follows the **lexicographic** order of the names, and
this is often a poor choice: for a chain with the sites ``"1[0]"``, ...,
``"1[11]"`` the lexicographic order is ``"1[0]", "1[10]", "1[11]", "1[1]",
"1[2]"``, ..., and for a two-dimensional lattice it depends on how the
coordinates are written. The argument ``permutation`` fixes it::

   permutation[p] = i      # position p of the chain holds the i-th site
                           # (in lexicographic order of the names)

A list with the names of the sites in the desired order is also accepted,
which is usually clearer. The attribute ``order`` of the state shows the order
in use.

The order matters for the cost and the accuracy, not for the result: two
orders give the same state if the bond dimension is large enough. For an
open chain of 40 spins with ``chi_max=32``:

.. code-block:: python

   system = build_system("open chain lattice", "spin", L=40)
   H = system.global_operator("Hamiltonian")
   natural = [f"1[{i}]" for i in range(40)]

   MPSGSOperator(H, permutation=natural, chi_max=32)   # E0 = -17.54147207
   MPSGSOperator(H, chi_max=32)                        # E0 = -17.48637053

With the natural order the DMRG converges in 5 sweeps (about 8 s on a laptop)
with a truncation error of :math:`3\times10^{-8}`. With the lexicographic
order it does not converge in 10 sweeps (about 45 s), with a truncation error
of :math:`5\times10^{-4}`, and the energy is wrong in the second decimal.
The reason is that terms between sites that are neighbours in the lattice but
far apart in the chain add channels to every bond of the MPO between them, and
that the entanglement across the cuts of the chain is larger.

For lattices in two dimensions, the order should minimize the number of
lattice bonds that cross every cut of the chain. For an :math:`L\times W`
lattice, this means to run along the longer side so that a cut crosses the
shorter one. For example, for the ``"open square lattice"`` with ``L=4`` and
``W=3``, whose sites are named ``"1[x, y]"``, the lexicographic order runs over
the columns of three sites, which is a good choice, while a *snake* that runs
over the rows of four sites is slightly worse. Both give the exact energy
``-6.69168019`` with ``chi_max=64``, but with ``chi_max=32`` the lexicographic
order has an error of :math:`10^{-4}` and converges in 3 sweeps, while the
snake has an error of :math:`7\times10^{-4}` and does not converge in 30 sweeps.
When in doubt, compare a few orders on a small instance of the problem.

Controlling the accuracy
------------------------

The accuracy is controlled by the maximum bond dimension ``chi_max`` (default
64) and by the options of the DMRG, passed through ``dmrg_options`` or as
keyword arguments of ``solve``:

``max_sweeps`` (30)
   Maximum number of sweeps.

``tol`` (``1e-10``)
   The DMRG stops when the energy changes less than this between sweeps.

``cutoff`` (``1e-12``)
   Weight that can be discarded at every bond before applying ``chi_max``.

``chi_init`` (4), ``seed``
   Bond dimension and random seed of the initial state.

``mps0``
   Initial state, as an :class:`~qalma.mps.mps.MPS`.

After the DMRG runs, ``rho.energy`` is the energy of the state and
``rho.dmrg_info`` is a dictionary with ``energies`` (one per sweep),
``converged``, ``sweeps``, ``max_truncation`` (the largest discarded weight in
the last sweep) and ``bond_dims``. A warning is logged if the DMRG does not
converge.

.. code-block:: python

   rho = MPSGSOperator(H, permutation=natural, chi_max=32,
                       dmrg_options={"max_sweeps": 10, "tol": 1e-8})
   rho.solve()
   rho.dmrg_info["converged"], rho.dmrg_info["max_truncation"]

Check the truncation error and the convergence of the energy with
``chi_max`` before trusting a result. The truncation error is a good indicator
for the error of local observables, but it is not a bound for the error of the
energy in a state that did not converge. As a reference, with
``chi_max=32`` the energy of the open 4x3 lattice above has an error of
:math:`10^{-4}` and a truncation of :math:`2\times10^{-5}`; with
``chi_max=64``, which is exact for 12 spins, the error is of the order of
the round-off.

Bringing your own MPS
---------------------

The DMRG of QALMA is plain: it works with numpy only, does not use
symmetries, and is meant for moderate bond dimensions. If you have a state
computed with a dedicated code, build an :class:`~qalma.mps.mps.MPS` from its
tensors and wrap it with
:meth:`~qalma.mps.gs_operator.MPSGSOperator.from_mps`. The tensors have the
shape ``(chi_left, d, chi_right)``, with ``chi_left = 1`` in the first one and
``chi_right = 1`` in the last one, and the position along the chain follows
the same ``permutation`` as before:

.. code-block:: python

   import numpy as np
   from qalma.mps import MPS, MPSGSOperator

   tensors = [...]                       # from your DMRG code
   mps = MPS(tensors)
   rho = MPSGSOperator.from_mps(mps, system, permutation=order, hamiltonian=H)

``MPS.from_dense`` builds an MPS from a state vector, which is useful for
tests on small systems. The local dimensions of the MPS are checked against
the system.

Relation with Gibbs states
--------------------------

At low temperatures, the reduced state of a Gibbs state tends to the one of
the ground state. This can be used to validate a reduction of a Gibbs state
(see :doc:`rpa`) at large :math:`\beta`, or the other way around. For the
bond term :math:`S^z_3 S^z_4` of an open chain with 8 sites:

.. code-block:: python

   from qalma.operators.states import GibbsDensityOperator

   system = build_system("open chain lattice", "spin", L=8)
   H = system.global_operator("Hamiltonian")
   obs = system.site_operator("Sz", "1[3]") * system.site_operator("Sz", "1[4]")
   sites = frozenset({"1[3]", "1[4]"})

   gs = MPSGSOperator(H, permutation=[f"1[{i}]" for i in range(8)])
   gs.partial_trace(sites).expect(obs)               # -0.103069

   for beta in (1, 2, 4, 8, 16, 32):
       gibbs = GibbsDensityOperator((beta * H).simplify())
       gibbs.partial_trace(sites).expect(obs)

.. list-table::
   :header-rows: 1
   :widths: 20 40

   * - :math:`\beta`
     - :math:`\langle S^z_3 S^z_4\rangle`
   * - 1
     - -0.06822
   * - 2
     - -0.11350
   * - 4
     - -0.13136
   * - 8
     - -0.11351
   * - 16
     - -0.10359
   * - 32
     - -0.10307
   * - ground state
     - -0.10307

The correlation is not monotonic in :math:`\beta`: its magnitude has a
maximum around :math:`\beta = 4` and then decreases towards the
ground-state value, so a comparison at a single moderate :math:`\beta` can be
misleading.

Saving and loading
------------------

The state can be stored in HDF5 files, alone or inside mixtures and
containers, with :func:`~qalma.io.save_operators`:

.. code-block:: python

   from qalma.io import load_operators, save_operators

   save_operators("ground_state.h5", {"rho": rho, "H": H})
   loaded = load_operators("ground_state.h5")["rho"]

The file holds the tensors of the MPS, the order of the sites, the
Hamiltonian and the diagnostics of the DMRG, so the loaded state is
identical to the stored one, and no dense matrix is written: a state of 24
spins with ``chi_max=16`` takes about 240 kB. Nothing is stored as a
Python pickle, so the file can be opened without ``allow_pickle``.

A state whose DMRG did not run yet is stored with the Hamiltonian only, and the
DMRG runs when the loaded state is needed. Give a ``seed`` in
``dmrg_options`` if you want the same initial state, since otherwise it is
random. The option ``mps0`` is not stored: ``save_operators`` warns about it
with an ``IncompleteStateWarning``. The layout is described in
:doc:`hdf5_format`.

Limitations
-----------

* The ground state must be non-degenerate (see above).
* The DMRG is plain, with no symmetries, no subspace expansion and no noise.
  For orientation, an open chain of 40 spins with ``chi_max=32`` takes about
  8 s, while a 6x4 open square lattice with ``chi_max=64`` takes minutes and
  did not converge in 12 sweeps. For two-dimensional systems of that size,
  or larger, compute the state with a dedicated code and wrap it with
  ``from_mps``.
* Periodic boundary conditions are legitimate but expensive for an MPS, since
  they double the entanglement across every cut and add a long-range term
  to the MPO. Prefer open boundaries when you can.
* ``to_qutip`` and ``to_qutip_operator`` return dense matrices and refuse to
  work with more than ``MPSGSOperator.MAX_DENSE_SITES`` sites (10 by default).
  Products of the state with operators, and sums with states of other types,
  fall back to this dense representation, so they are meant for small
  systems. Scaling by a positive number, ``rho / 2`` and ``rho + rho`` keep
  the MPS.
* Only Hamiltonians written with spins have been tested. Fermionic models
  require care, since the sites are reordered along the chain.

.. seealso::

   :doc:`/api/mps`, :doc:`states`, :doc:`rpa`,
   :doc:`../examples/example_mps_ground_state`
