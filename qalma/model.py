"""System Descriptors.

A ``SystemDescriptor`` contains all the information relative to the
system to be simulated.

``SystemDescriptor``s can be generated from information in ALPS xml file
libraries by means of ``build_system``.

An alternative is just to build the system through
``build_system_from_dims`` which takes as argument a  dictionary,
specifying the names of the system sites as keys, and their
corresponding dimensions as values.

``build_spin_chain`` is a short-cut to build a spin chain lattice.
"""

import itertools
import logging
import pickle
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import qutip  # type: ignore[import-untyped]

from qalma.alpsmodels import ModelDescriptor, qutip_model_from_dims
from qalma.geometry import GraphDescriptor
from qalma.settings import LATTICE_LIB_FILE, MODEL_LIB_FILE
from qalma.utils import eval_expr, replace_variable_type

logger = logging.getLogger(__name__)


class FermionicSignWarning(UserWarning):
    """Warn that a model with fermionic operators is treated as bosonic.

    When every site of the system is a single fermionic mode (a two-level
    site with a fermionic quantum number), the operators are built through
    the Jordan-Wigner transformation and the signs are exact. Otherwise
    (e.g. sites with two spin species, or models that mix fermionic and
    non-fermionic sites), operators acting on different sites are assumed
    to commute: the signs coming from the anticommutation of fermionic
    operators are not implemented. Results are then only correct when
    those signs cancel out (e.g. nearest-neighbour hopping of a single
    species on an open chain).
    """


def _split_site(site_basis: dict, binary_as_qubits: bool) -> Optional[dict]:
    """Split the local space of a site into qubits and a rest.

    The qubits are the fermionic quantum numbers (and, if
    ``binary_as_qubits``, also the non-fermionic ones that take two
    consecutive values). The rest is given by the remaining quantum numbers
    that are not functions of the ones already chosen; it becomes a single
    subsite of arbitrary dimension. The local basis of the site must be a
    subset of the product (qubits) x (rest): if it is a proper subset, the
    site is *constrained* (e.g. the t-J basis, where the spin of an empty
    site is fixed).

    Returns a dict with the keys

    ``qubits``
        list of ``(qn_name, position, lower_value, fermionic)``,
    ``rest``
        list of ``(qn_name, position)``,
    ``rest_values``
        the distinct values of the ``rest`` quantum numbers, in order of
        appearance in the local basis (the basis of the rest subsite),
    ``constrained``
        whether the local basis is a proper subset of the product,

    or ``None`` if a fermionic quantum number does not take two consecutive
    values, or if the quantum numbers are not available.
    """
    qns = site_basis.get("qn", None)
    qn_index = site_basis.get("qn_index", None)
    states = site_basis.get("localstates", None)
    if not qns or qn_index is None or not states:
        return None
    ordered = sorted(qn_index.items(), key=lambda item: item[1])
    chosen: List[int] = []
    n_distinct = 1

    def count(positions):
        """Number of distinct values taken by the quantum numbers at ``positions``."""
        return len({tuple(state[p] for p in positions) for state in states})

    def binary_values(position):
        """Values at ``position`` if they are two consecutive numbers, else ``None``."""
        values = sorted({state[position] for state in states})
        if len(values) == 2 and abs(values[1] - values[0] - 1) < 1e-10:
            return values
        return None

    qubits = []
    for qn_name, position in ordered:
        if not qns.get(qn_name, {}).get("fermionic", False):
            continue
        values = binary_values(position)
        if values is None:
            return None
        new_count = count(chosen + [position])
        if new_count == n_distinct:
            continue  # a function of the ones already chosen
        qubits.append((qn_name, position, values[0], True))
        chosen.append(position)
        n_distinct = new_count
    if binary_as_qubits:
        for qn_name, position in ordered:
            if position in chosen or binary_values(position) is None:
                continue
            new_count = count(chosen + [position])
            if new_count > n_distinct:
                qubits.append((qn_name, position, binary_values(position)[0], False))
                chosen.append(position)
                n_distinct = new_count
    rest = []
    for qn_name, position in ordered:
        if position in chosen:
            continue
        new_count = count(chosen + [position])
        if new_count > n_distinct:
            rest.append((qn_name, position))
            chosen.append(position)
            n_distinct = new_count
    if n_distinct != len(states):
        return None
    rest_values: List[tuple] = []
    for state in states:
        value = tuple(state[p] for _, p in rest)
        if value not in rest_values:
            rest_values.append(value)
    return {
        "qubits": qubits,
        "rest": rest,
        "rest_values": rest_values if rest else [()],
        "constrained": 2 ** len(qubits) * max(1, len(rest_values)) > len(states),
    }


def _qubit_layout(physical_sites: dict, qubit_split: Optional[bool]) -> Optional[dict]:
    """Decide whether the sites are split into subsites, and how.

    Returns ``None`` if the sites keep their own local spaces. Otherwise
    returns a mapping ``node -> split``, with ``split`` as returned by
    :func:`_split_site` plus the key ``names`` (the names of the subsites:
    the qubits, then the rest) and ``whole`` (``True`` if the node is not
    split at all, i.e. it has no qubits).

    With ``qubit_split=None`` the split is used when the model has
    fermionic quantum numbers and the fermionic quantum numbers of every
    site take two values. With ``qubit_split=True``, every two-level
    quantum number becomes a qubit, and every site must be a product of
    them.
    """
    if qubit_split is False or not physical_sites:
        return None
    layout = {}
    has_fermions = False
    for node, site_basis in physical_sites.items():
        split = _split_site(site_basis, binary_as_qubits=bool(qubit_split))
        if qubit_split and (split is None or split["rest"] or split["constrained"]):
            raise ValueError(
                f"the local space of site {node} is not a product of two-level "
                "quantum numbers, and cannot be split into qubits."
            )
        if split is None:
            return None
        n_subsites = len(split["qubits"]) + (1 if split["rest"] else 0)
        if n_subsites == 1:
            names = [node]
        else:
            names = [f"{node}:{qn_name}" for qn_name, *_ in split["qubits"]]
            if split["rest"]:
                names.append(
                    f"{node}:" + "+".join(qn_name for qn_name, _ in split["rest"])
                )
        split["names"] = names
        split["whole"] = not split["qubits"]
        layout[node] = split
        has_fermions = has_fermions or any(q[3] for q in split["qubits"])
    if qubit_split is None and not has_fermions:
        return None
    return layout


def _qubit_site_basis(qn_name, lower, fermionic, parent_basis) -> dict:
    """Local basis descriptor of the qubit of ``qn_name`` in sites of ``parent_basis``.

    The same descriptor is shared by the qubits of all the sites with the
    same local basis, as the local bases of the model are shared by the
    sites of the same type. It has no reference to the node it belongs to:
    that is ``SystemDescriptor.qubit_parent``.
    """
    identity = qutip.qeye(2)
    return {
        "name": f"{parent_basis.get('name', '')}:{qn_name}",
        "qn": {qn_name: {"min": lower, "max": lower + 1, "fermionic": fermionic}},
        "qn_index": {qn_name: 0},
        "dimension": 2,
        "identity": identity,
        "operators": {"identity": identity},
        "parms": parent_basis.get("parms", {}),
        "localstates": [(lower,), (lower + 1,)],
        # marks the descriptor as the one of a qubit of a split site
        "qubit_of": qn_name,
        "fermionic": fermionic,
        # parity of the first basis state, (-1)**lower, for fermionic qubits
        "parity0": (-1.0) ** (int(round(lower)) % 2) if fermionic else 1.0,
    }


def _rest_site_basis(rest, rest_values, parent_basis) -> dict:
    """Local basis descriptor of the non-fermionic rest of a split site."""
    dimension = len(rest_values)
    identity = qutip.qeye(dimension)
    names = tuple(qn_name for qn_name, _ in rest)
    return {
        "name": f"{parent_basis.get('name', '')}:" + "+".join(names),
        "qn": {
            qn_name: {
                "min": min(v[k] for v in rest_values),
                "max": max(v[k] for v in rest_values),
                "fermionic": False,
            }
            for k, qn_name in enumerate(names)
        },
        "qn_index": {qn_name: k for k, qn_name in enumerate(names)},
        "dimension": dimension,
        "identity": identity,
        "operators": {"identity": identity},
        "parms": parent_basis.get("parms", {}),
        "localstates": list(rest_values),
        # marks the descriptor as the one of the rest of a split site
        "rest_of": names,
        "fermionic": False,
        "parity0": 1.0,
    }


class SystemDescriptor:
    """System Descriptor class.

    A SystemDescriptor represents a quantum system as a collection of
    Hilbert spaces associated to the nodes of a graph or lattice, and a
    set of parameters defining a Hamiltonian operator.
    """

    _subsystems_cache: dict
    # Jordan-Wigner string of each qubit, as (code, sign); built on demand.
    _jw_strings: Optional[Dict[str, Tuple[int, float]]]
    dimensions: dict
    name: str
    sites: dict
    spec: dict
    operators: dict

    def __init__(
        self,
        graph: GraphDescriptor,
        model: ModelDescriptor,
        parms: Optional[dict] = None,
        sites=None,
        _load_operators=True,
        qubit_split: Optional[bool] = None,
    ):
        """Build a system from a graph and a model.

        Parameters
        ----------
        graph, model, parms :
            Geometry, model and parameters.
        sites : dict, optional
            Local spaces, overriding the ones derived from ``graph`` (used
            internally by ``union`` and ``subsystem``).
        qubit_split : bool or None
            Whether to represent each site by one qubit per two-level
            quantum number, with fermionic modes mapped to Pauli strings
            through the Jordan-Wigner transformation. ``None`` (default)
            does it when the model has fermionic quantum numbers and every
            site can be split; ``True`` forces it (raising ``ValueError`` if
            some site cannot be split); ``False`` disables it.
        """
        if parms is None:
            parms = {}
        if model:
            model_parms = model.parms.copy()
            model_parms.update(parms)
            parms = model_parms

        self.spec = {"graph": graph, "model": model, "parms": parms}
        self.name = f"{self.spec['model'].name} on {self.spec['graph'].name}"
        site_basis = model.site_basis
        try:
            physical_sites = {
                node: site_basis[attr["type"]] for node, attr in graph.nodes.items()
            }
        except KeyError as ex:
            if not sites:
                raise ValueError(
                    (
                        f"Model <<{model.name}>> does not provide the specification "
                        f"for site of type <<{ex.args[0]}>> used in nodes of <<{graph.name}>>."
                    )
                ) from ex
            physical_sites = {}
        given_split = bool(sites) and any(
            "qubit_of" in basis or "rest_of" in basis for basis in sites.values()
        )
        if sites and not given_split:
            physical_sites = sites
        self.qubit_split = qubit_split
        self.physical_sites = physical_sites
        layout = _qubit_layout(physical_sites, qubit_split)
        # When the sites are split, every fermionic quantum number of every
        # site becomes a qubit, and the rest of the quantum numbers of the
        # site a single subsite. The subsites are ordered as the nodes and,
        # inside each node, as the quantum numbers (qubits first). Fermionic
        # modes are mapped to Pauli strings through the Jordan-Wigner
        # transformation in that order.
        self.qubit_representation = layout is not None
        self._site_splits = layout or {}
        if layout is None:
            self.sites = sites if sites else physical_sites
            self.site_qubits = {node: (node,) for node in self.sites}
        else:
            shared: Dict[Tuple[int, str], dict] = {}
            split_sites = {}
            for node, split in layout.items():
                parent_basis = physical_sites[node]
                if split["whole"]:
                    split_sites[node] = parent_basis
                    continue
                for name, (qn_name, _, lower, fermionic) in zip(
                    split["names"], split["qubits"]
                ):
                    key = (id(parent_basis), qn_name)
                    if key not in shared:
                        shared[key] = _qubit_site_basis(
                            qn_name, lower, fermionic, parent_basis
                        )
                    split_sites[name] = shared[key]
                if split["rest"]:
                    key = (id(parent_basis), ":rest")
                    if key not in shared:
                        shared[key] = _rest_site_basis(
                            split["rest"], split["rest_values"], parent_basis
                        )
                    split_sites[split["names"][-1]] = shared[key]
            self.sites = sites if given_split else split_sites
            site_qubits = {
                node: tuple(name for name in split["names"] if name in self.sites)
                for node, split in layout.items()
            }
            self.site_qubits = {node: qs for node, qs in site_qubits.items() if qs}
        self.qubit_parent = {
            qubit: node for node, qubits in self.site_qubits.items() for qubit in qubits
        }

        self.dimensions = {name: site["dimension"] for name, site in self.sites.items()}
        self.operators = {
            "site_operators": {},
            "bond_operators": {},
            "loop_operators": {},
            "global_operators": {},
        }
        self._subsystems_cache = {}
        self._jw_strings = None
        if _load_operators:
            if not self.qubit_representation and any(
                getattr(op, "fermionic", False)
                for site in self.physical_sites.values()
                for op in site.get("operators", {}).values()
            ):
                warnings.warn(
                    f"The model <<{model.name}>> has fermionic operators, but "
                    "fermionic signs are not implemented: operators on different "
                    "sites are assumed to commute, so in general the results "
                    "correspond to hard-core particles, not to fermions.",
                    FermionicSignWarning,
                    stacklevel=2,
                )
            self._load_site_operators()
            self._load_global_ops()

    def __eq__(self, other):
        """Check equality."""
        # To be the equal, two SystemDescriptors
        # should have the same values in the
        # `spec` attribute:
        if self is other:
            return True
        if other is None:
            return False
        assert isinstance(other, SystemDescriptor)
        for key, my_spec in self.spec.items():
            if other.spec[key] != my_spec:
                return False
        for site in other.sites:
            if site not in self.sites:
                return False

        for site, descr in self.sites.items():
            if descr != other.sites[site]:
                return False

        return True

    def __repr__(self):
        """Build repr str."""
        result = (
            "graph:"
            + repr(self.spec["graph"])
            + "\n"
            + "sites:"
            + repr(self.sites.keys())
            + "\n"
            + "dimensions:"
            + repr(self.dimensions)
        )
        return result

    def __getstate__(self):
        """Get state."""
        # Copy the object's state and exclude _subsystems_cache and operators
        if hasattr(self, "_serialized"):
            return self._serialized

        state = {
            key: {} if key in ("_subsystems_cache", "operators") else val
            for key, val in self.__dict__.items()
        }
        self._serialized = pickle.dumps(state)
        return self._serialized

    def __iter__(self):
        """Iterate over sites."""
        yield from self.sites

    def __setstate__(self, state):
        """Set state."""
        state_dict = pickle.loads(state)
        # Files written before the qubit representation was introduced.
        if "physical_sites" not in state_dict:
            sites = state_dict.get("sites", {})
            state_dict["physical_sites"] = sites
            state_dict["site_qubits"] = {node: (node,) for node in sites}
            state_dict["qubit_parent"] = {node: node for node in sites}
            state_dict["qubit_representation"] = False
            state_dict["qubit_split"] = False
        if "_site_splits" not in state_dict:
            # Files written before the rest subsites were introduced.
            layout = None
            if state_dict.get("qubit_representation", False):
                layout = _qubit_layout(
                    state_dict["physical_sites"], state_dict.get("qubit_split", None)
                )
            state_dict["_site_splits"] = layout or {}
        state_dict["_jw_strings"] = None
        state_dict["operators"] = {
            "site_operators": {},
            "bond_operators": {},
            "loop_operators": {},
            "global_operators": {},
        }
        self.__dict__.update(state_dict)

    def subsystem(self, sites: frozenset):
        """Build a subsystem including the sites listed in sites."""
        # Try to find the subsystem in the cache
        assert isinstance(sites, frozenset)
        result = self._subsystems_cache.get(sites, None)
        if result is not None:
            return result

        # To avoid circular references, the cache
        # only stores proper subsystems
        self_sites = self.sites
        if len(self_sites) == len(sites):
            if all(name in self_sites for name in sites):
                return self

        # Build a new subsystem
        parms = self.spec["parms"].copy()
        model = self.spec["model"]
        if self.qubit_representation:
            # ``sites`` are qubits: keep the nodes they belong to, and the
            # qubits in the same order as in ``self``.
            qubit_parent = self.qubit_parent
            graph = self.spec["graph"].subgraph(
                frozenset(qubit_parent[qubit] for qubit in sites)
            )
            qubits = {q: basis for q, basis in self_sites.items() if q in sites}
            result = SystemDescriptor(
                graph,
                model,
                parms,
                sites=qubits,
                _load_operators=False,
                qubit_split=self.qubit_split,
            )
            return self._subsystems_cache.setdefault(sites, result)
        graph = self.spec["graph"].subgraph(sites)
        result = SystemDescriptor(
            graph, model, parms, _load_operators=False, qubit_split=self.qubit_split
        )
        return self._subsystems_cache.setdefault(sites, result)

    def _load_site_operators(self):
        for site_name, site in self.physical_sites.items():
            for op_name in site["operators"]:
                op_site = f"{op_name}@{site_name}"
                self.site_operator(op_site)

    def _load_global_ops(self):
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from qalma.operators import LocalOperator, OneBodyOperator

        # First, load conserved quantum numbers:
        for constraint_qn in self.spec["model"].constraints:
            if self.qubit_representation:
                local_terms = tuple(
                    self.site_operator(
                        site_basis["qn"][constraint_qn]["operator"], node
                    )
                    for node, site_basis in self.physical_sites.items()
                    if constraint_qn in site_basis["qn"]
                )
                if local_terms:
                    global_qn = self.combine_split_terms(local_terms)
                    if bool(global_qn):
                        self.operators["global_operators"][constraint_qn] = global_qn
                continue
            global_qn_terms = []
            for site, site_basis in self.sites.items():
                local_qn = site_basis["qn"].get(constraint_qn, None)
                if local_qn is None:
                    continue
                op_name = local_qn["operator"]
                operator = site_basis["operators"][op_name]

                global_qn_terms.append(LocalOperator(site, operator, self))

            global_qn = OneBodyOperator(tuple(global_qn_terms), self, True).simplify()

            if bool(global_qn):
                self.operators["global_operators"][constraint_qn] = global_qn

        # Number of sites outside the physical subspace of constrained
        # split sites (e.g. t-J sites with the spin of an empty site, or a
        # spinless occupied site).
        unphysical = self.unphysical_sites_operator()
        if unphysical is not None:
            self.operators["global_operators"]["unphysical_sites"] = unphysical

        names = list(self.spec["model"].global_ops)
        for gop in names:
            self.global_operator(gop)

    def __mul__(self, system):
        """Return the union of two models."""
        if system is None or system is self:
            return self
        return self.union(system)

    def __len__(self):
        """Return the number of sites in the model."""
        return len(self.sites)

    def _repr_latex_(self):
        return "System \textbf{" + self.name + "}"

    def contains(self, system):
        """Check if the object contains `system` as a subsystem."""
        if self is system:
            return True
        block = frozenset(system.sites)
        if system is self._subsystems_cache.get(block, None):
            return True

        for subsystem in self._subsystems_cache.values():
            if subsystem.contains(system):
                self._subsystems_cache[block] = system
                return True

        logger.info("check model and graph")
        if (
            self.spec["model"] == system.spec["model"]
            and self.spec["graph"].contains(system.spec["graph"])
            and all(site in self.sites for site in system.sites)
        ):
            self._subsystems_cache[block] = system
            return True
        if not self.spec["model"] == system.spec["model"]:
            logger.debug(
                "models differ: %s vs %s", self.spec["model"], system.spec["model"]
            )
        if not self.spec["graph"].contains(system.spec["graph"]):
            logger.debug(
                "graph %s does not contain %s", self.spec["graph"], system.spec["graph"]
            )
        return False

    def union(self, system):
        """Return a SystemDescritor containing system and self."""
        if system is None or system is self:
            return self
        if all(site in self.sites for site in system.sites):
            return self
        if all(site in system.sites for site in self.sites):
            return system

        model = self.spec["model"]
        assert (
            model is system.spec["model"]
        ), "Join systems with different base models is not supported."
        parms = self.spec["parms"].copy()
        union_graph = self.spec["graph"] + system.spec["graph"]
        sites = self.sites.copy()
        sites.update(system.sites)
        # raise NotImplementedError("Union of disjoint systems are not implemented.")
        result = SystemDescriptor(
            union_graph, model, parms, sites, qubit_split=self.qubit_split
        )
        result._subsystems_cache[frozenset(self.sites)] = self
        result._subsystems_cache[frozenset(system.sites)] = system

        return result

    def site_identity(self, site: str):  # -> Qobj
        """
        Return the identity operator for the site.

        Returns the internal representation of the identity associated to
        `site`.
        """
        return self.sites[site]["identity"]

    def site_operator(self, name: str, site: str = ""):  # -> "Operator"
        """Return a site operator.

        Return a global operator representing an operator `name` acting over
        the site `site`.

        By default, the name is assumed to specify both the name and
        site in the form `"name@site"`.
        """
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from qalma.operators import LocalOperator

        if site != "":
            op_name = name
            name = f"{name}@{site}"
        else:
            op_name, site = name.split("@")

        site_op = self.operators["site_operators"].get(site, {}).get(op_name, None)
        if site_op is not None:
            return site_op

        local_op = self.physical_sites[site]["operators"].get(op_name, None)
        if local_op is None:
            return None
        if self.qubit_representation:
            result_op = self.jordan_wigner_operator(site, local_op)
        else:
            result_op = LocalOperator(site, local_op, system=self)
        self.operators["site_operators"].setdefault(site, {})
        self.operators["site_operators"][site][op_name] = result_op
        return result_op

    @property
    def jordan_wigner(self) -> bool:
        """``True`` if the fermionic modes are mapped to Pauli strings.

        That is, if the system uses the qubit representation and some qubit
        is a fermionic mode.
        """
        return self.qubit_representation and any(
            basis.get("fermionic", False) for basis in self.sites.values()
        )

    def jordan_wigner_string(self, qubit: str):
        r"""Return :math:`\prod_{q' < q} (-1)^{N_{q'}}` as a Pauli string.

        The product runs over the fermionic qubits that come before
        ``qubit`` in ``self.sites``. Only available when
        ``self.qubit_representation`` is ``True``.
        """
        # pylint: disable=import-outside-toplevel
        from qalma.operators.pauli import PauliStringOperator

        code, sign = self._jordan_wigner_strings()[qubit]
        return PauliStringOperator(code, sign, self)

    def _jordan_wigner_strings(self) -> dict:
        """Map each qubit to ``(code, sign)`` of its Jordan-Wigner string."""
        if self._jw_strings is not None:
            return self._jw_strings
        assert self.qubit_representation, "the system is not represented by qubits."
        result = {}
        code, sign = 0, 1.0
        for position, (qubit, basis) in enumerate(self.sites.items()):
            result[qubit] = (code, sign)
            if basis.get("fermionic", False):
                # (-1)^N = parity0 * sigma_z
                code |= 3 << (2 * position)
                sign *= basis["parity0"]
        self._jw_strings = result
        return result

    @property
    def constrained(self) -> bool:
        """``True`` if some split site has states outside its physical subspace."""
        return any(split["constrained"] for split in self._site_splits.values())

    def unphysical_sites_operator(self):
        r"""Return :math:`Q = \sum_i (1 - \Pi_i)`, or ``None`` if there is no constraint.

        :math:`\Pi_i` is the projector of the node ``i`` onto its physical
        local states. :math:`Q` is diagonal, commutes with every operator
        built from the model, and counts the sites outside the physical
        subspace: the physical states are those with :math:`Q = 0`.
        """
        # pylint: disable=import-outside-toplevel
        from qalma.operators import ScalarOperator

        terms = []
        for node, split in self._site_splits.items():
            if not split["constrained"] or any(
                name not in self.sites for name in split["names"]
            ):
                continue
            dimension = len(self.physical_sites[node]["localstates"])
            projector = self.jordan_wigner_operator(node, np.eye(dimension))
            terms.extend([ScalarOperator(1.0, self), -projector])
        if not terms:
            return None
        return self.combine_split_terms(terms)

    def combine_split_terms(self, terms):
        """Add operators of a split system, merging the Pauli parts symbolically.

        Sums of Pauli strings are collected in a single
        :class:`~qalma.operators.PauliSumOperator`; the other terms (tensor
        products, products and local operators on rest subsites) are kept.
        """
        # pylint: disable=import-outside-toplevel
        from qalma.operators import ScalarOperator, SumOperator
        from qalma.operators.pauli import is_pauli_sum, to_pauli_sum

        pauli_parts = []
        others = []

        def visit(term):
            """Sort ``term`` (recursively through sums) into Pauli parts and others."""
            if isinstance(term, (int, float, complex, np.number)):
                if term:
                    pauli_parts.append(ScalarOperator(term, self))
            elif is_pauli_sum(term):
                pauli_parts.append(term)
            elif isinstance(term, SumOperator):
                for sub_term in term.terms:
                    visit(sub_term)
            elif term:
                others.append(term)

        for term in terms:
            visit(term)
        result = []
        if pauli_parts:
            pauli = (
                pauli_parts[0]
                if len(pauli_parts) == 1
                else SumOperator(tuple(pauli_parts), self)
            )
            pauli = to_pauli_sum(pauli)
            if pauli:
                result.append(pauli)
        result.extend(others)
        if not result:
            return ScalarOperator(0, self)
        if len(result) == 1:
            return result[0]
        return SumOperator(tuple(result), self)

    def jordan_wigner_operator(self, site: str, local_op):
        r"""Map an operator on the node ``site`` to its split representation.

        The local space of ``site`` is embedded in the product of its
        qubits (one per fermionic quantum number) and its rest subsite (the
        other quantum numbers), by mapping each local state to the values
        of its quantum numbers. The embedded operator vanishes outside the
        physical local states.

        The operator is expanded as :math:`\sum_a \sigma_a \otimes B_a`,
        with :math:`\sigma_a` products of Pauli matrices on the qubits and
        :math:`B_a` matrices on the rest. A :math:`\sigma_a` with an even
        number of :math:`\sigma^{x,y}` on fermionic qubits is kept as it
        is. One with an odd number is read as the product of single-mode
        operators in the order of the quantum numbers, with each
        :math:`\sigma^{x,y}` on a fermionic qubit replaced by its
        Jordan-Wigner image
        :math:`\prod_{q'<q}(-1)^{N_{q'}}\,\sigma^{x,y}_q`.

        This is the convention in which the local matrices of the ALPS
        library are written: ``c_up`` and ``c_down`` act on their own
        quantum number with matrix element 1, and the operators that do not
        change the fermion parity of the site are the matrices themselves:
        ``Splus`` is :math:`c^\dagger_\uparrow c_\downarrow`, ``Sminus``
        is :math:`c^\dagger_\downarrow c_\uparrow`, and so on. The
        resulting mode operators satisfy the canonical anticommutation
        relations, also between modes of the same site.

        Returns a sum of Pauli strings if the site has no rest subsite, and
        otherwise a sum of :class:`~qalma.operators.TensorProductOperator`,
        Pauli strings and local operators on the rest subsite.
        """
        # pylint: disable=import-outside-toplevel
        from qalma.operators import LocalOperator
        from qalma.operators.pauli import (
            _I_POWERS,
            PAULI_MATRICES,
            _product_phase,
            _site_index,
            pauli_dict_to_operator,
        )
        from qalma.operators.tensor_product import TensorProductOperator
        from qalma.qutip_tools.tools import _to_array

        split = self._site_splits[site]
        if split["whole"]:
            return LocalOperator(site, local_op, system=self)
        names = split["names"]
        qubit_names = names[: len(split["qubits"])]
        rest_name = names[-1] if split["rest"] else None
        n_qubits = len(qubit_names)
        rest_values = split["rest_values"]
        rest_dim = len(rest_values)
        site_basis = self.physical_sites[site]

        # Embed the local matrix in (qubits) x (rest).
        matrix = _to_array(local_op)
        embedding = np.array(
            [
                sum(
                    int(round(state[position] - lower)) << (n_qubits - 1 - r)
                    for r, (_, position, lower, _) in enumerate(split["qubits"])
                )
                * rest_dim
                + rest_values.index(tuple(state[p] for _, p in split["rest"]))
                for state in site_basis["localstates"]
            ]
        )
        size = 2**n_qubits * rest_dim
        embedded = np.zeros((size, size), dtype=complex)
        embedded[np.ix_(embedding, embedding)] = matrix
        blocks = embedded.reshape(2**n_qubits, rest_dim, 2**n_qubits, rest_dim)

        index = _site_index(self)
        mask = index.mask
        qubit_shift = [2 * index.position[q] for q in qubit_names]
        fermionic = [entry[3] for entry in split["qubits"]]
        strings = self._jordan_wigner_strings()
        components: dict = {}
        scale = 2.0**n_qubits
        for labels in itertools.product(range(4), repeat=n_qubits):
            pauli = PAULI_MATRICES[labels[0]]
            for label in labels[1:]:
                pauli = np.kron(pauli, PAULI_MATRICES[label])
            # B_a = Tr_qubits[(sigma_a x 1) M] / 2**n_qubits
            rest_matrix = np.einsum("uv,vrus->rs", pauli, blocks) / scale
            if np.abs(rest_matrix).max() < 1e-14:
                continue
            flips = sum(
                1 for r, label in enumerate(labels) if label in (1, 2) and fermionic[r]
            )
            if flips % 2 == 0:
                # Even in the fermions of the site: the Jordan-Wigner strings
                # cancel, and the local matrix is the operator itself (e.g.
                # Splus = c^+_up c_down, Sminus = c^+_down c_up, n_up n_down).
                code = 0
                for r, label in enumerate(labels):
                    code |= label << qubit_shift[r]
                value = 1.0 + 0j
            else:
                # Odd: ordered product of the single-mode factors, each
                # sigma^{x,y} on a fermionic qubit carrying its string.
                code, value = 0, 1.0 + 0j
                for r, label in enumerate(labels):
                    if label == 0:
                        continue
                    factor_code = label << qubit_shift[r]
                    if label in (1, 2) and fermionic[r]:
                        string_code, sign = strings[qubit_names[r]]
                        factor_code |= string_code
                        value *= sign
                    k, code = _product_phase(code, factor_code, mask)
                    value *= _I_POWERS[k]
            components[code] = components.get(code, 0) + value * rest_matrix

        if rest_name is None:
            return pauli_dict_to_operator(
                {
                    code: complex(block[0, 0])
                    for code, block in components.items()
                    if abs(block[0, 0]) > 1e-14
                },
                self,
            )
        pauli_coefficients = {}
        terms: list = []
        identity = np.eye(rest_dim)
        for code, block in components.items():
            if np.abs(block).max() < 1e-14:
                continue
            if np.allclose(block, block[0, 0] * identity, atol=1e-14):
                pauli_coefficients[code] = complex(block[0, 0])
            elif code == 0:
                terms.append(LocalOperator(rest_name, block, self))
            else:
                terms.append(
                    TensorProductOperator._new(code, {rest_name: block}, 1.0, self)
                )
        if pauli_coefficients:
            terms.insert(0, pauli_dict_to_operator(pauli_coefficients, self))
        return self.combine_split_terms(terms)

    def bond_operator(self, name: str, src: str, dst: str, skip=None):  # -> "Operator":
        """Return a bond operator by name and sites."""
        result_op = self.operators["global_operators"].get(
            (
                name,
                src,
                dst,
            ),
            None,
        )
        if result_op is not None:
            return result_op
        # Try to build the bond operator from the descriptors.
        bond_op_descriptors = self.spec["model"].bond_ops
        bond_op_descriptor = bond_op_descriptors.get(name, None)
        if bond_op_descriptor is None:
            return None

        bond_dependencies = [
            bop
            for bop in bond_op_descriptors
            if bond_op_descriptor.find(bop + "@") >= 0
        ]
        bond_op_descriptor = bond_op_descriptor.replace("@", "__")
        src_operators = self.physical_sites[src]["operators"]
        dst_operators = self.physical_sites[dst]["operators"]

        # Load site operators on src and dst
        parms_and_ops = {
            f"{name_src}__src": self.site_operator(name_src, src)
            for name_src in src_operators
        }
        parms_and_ops.update(
            {
                f"{name_dst}__dst": self.site_operator(name_dst, dst)
                for name_dst in dst_operators
            }
        )
        self_parms = self.spec["parms"]
        if self_parms:
            parms_and_ops.update(self_parms)

        # Try to evaluate
        result = eval_expr(bond_op_descriptor, parms_and_ops)
        if not (result is None or isinstance(result, str)):
            self.operators["bond_operators"][
                (
                    name,
                    src,
                    dst,
                )
            ] = result
            return result
        # Now, try to include existent bond operators
        parms_and_ops.update(
            {
                f"{tup_op[0]}__src_dst": op
                for tup_op, op in self.operators["global_operators"].items()
                if tup_op[0] == src and tup_op[1] == dst
            }
        )
        parms_and_ops.update(
            {
                f"{tup_op[0]}__dst_src": op
                for tup_op, op in self.operators["global_operators"].items()
                if tup_op[0] == dst and tup_op[1] == src
            }
        )

        result = eval_expr(bond_op_descriptor, parms_and_ops)
        if not (result is None or isinstance(result, str)):
            self.operators["bond_operators"][
                tuple(
                    (
                        name,
                        src,
                        dst,
                    )
                )
            ] = result
            return result

        # Finally, try to load other operators
        if skip is None:
            skip = [name]
        else:
            skip.append(name)
        for bop in bond_dependencies:
            # Skip this name
            if bop in skip:
                continue

            # src->dst
            new_bond_op = self.bond_operator(bop, src, dst, skip)
            if new_bond_op is None:
                continue

            parms_and_ops[f"{bop}__src_dst"] = new_bond_op

            new_bond_op = self.bond_operator(bop, dst, src, skip)
            if new_bond_op is None:
                continue

            parms_and_ops[f"{bop}__dst_src"] = new_bond_op

            result = eval_expr(bond_op_descriptor, parms_and_ops)
            if result is not None and not isinstance(result, str):
                result = result.simplify()
                self.operators["bond_operators"][
                    (
                        name,
                        src,
                        dst,
                    )
                ] = result
                return result

        # If this fails after exploring all the operators, then it means that
        # the operator is not in the basis.
        # if skip[-1]==name:
        #    self.bond_operators[(name, src, dst,)] = None
        return None

    def loop_operator(self, name: str, loop: Tuple[str], _skip=None):  # -> "Operator":
        """Return a loop operator by name and sites."""
        result_op = self.operators["global_operators"].get(
            (
                name,
                loop,
            ),
            None,
        )
        if result_op is not None:
            return result_op
        loop_op_descriptors = self.spec["model"].loop_ops
        loop_op_descriptor = loop_op_descriptors.get(name, None)
        if loop_op_descriptor is None:
            return None

        # loop_dependencies = [
        #    bop
        #    for bop in loop_op_descriptors
        #    if loop_op_descriptor.find(bop + "@") >= 0
        # ]
        loop_op_descriptor = loop_op_descriptor.replace("@", "__")
        # vertex_operators  = tuple((self.sites[src]["operators"] for src in loop))
        raise ValueError(f"{loop_op_descriptor} not implemented for  {loop}.")

    def site_term_from_descriptor(self, term_spec, graph, parms):
        """Build a site term from a site term specification."""
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from qalma.operators import OneBodyOperator, ScalarOperator

        expr = term_spec["expr"]
        site_type = term_spec.get("type", None)
        term_ops = []
        t_parm = {}
        t_parm.update(term_spec.get("parms", {}))
        if parms:
            t_parm.update(parms)
        for node_name, node in graph.nodes.items():
            node_type = node.get("type", None)
            if site_type is not None and site_type != node_type:
                continue
            s_expr = expr.replace("#", node_type)
            operator_names = set(re.findall(r"\b([a-zA-Z_]+)\b@", s_expr))

            s_expr = s_expr.replace("@", "__")
            s_parm = {
                replace_variable_type(key, node_type): replace_variable_type(
                    val, node_type
                )
                for key, val in t_parm.items()
            }
            s_parm.update(
                {
                    f"{name_op}_local": local_op
                    for name_op, local_op in self.operators["site_operators"][
                        node_name
                    ].items()
                }
            )
            term_op = eval_expr(s_expr, s_parm)
            if term_op is None or isinstance(term_op, str):
                raise ValueError(
                    f"<<{s_expr}>> could not be evaluated.", operator_names
                )
            term_ops.append(term_op)

        if len(term_ops) == 0:
            return ScalarOperator(0, self)
        if len(term_ops) == 1:
            return term_ops[0]
        if self.qubit_representation:
            # Odd site terms carry Jordan-Wigner strings: they are not
            # one-body operators in the split representation.
            return self.combine_split_terms(term_ops)
        return OneBodyOperator(tuple(term_ops), self)

    def bond_term_from_descriptor(self, term_spec, graph, model, parms):
        """Build a bond term from a bond term specification."""
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from qalma.operators import ScalarOperator, SumOperator

        def process_edge(e_expr, bond, model, t_parm):
            """Evaluate the bond expression ``e_expr`` on the edge ``bond``."""
            edge_type, src, dst = bond
            e_parms = {
                replace_variable_type(key, edge_type): replace_variable_type(
                    val, edge_type
                )
                for key, val in t_parm.items()
            }
            for op_idx in ([src, "src"], [dst, "dst"]):
                e_parms.update(
                    {
                        f"{key}__{op_idx[1]}": val
                        for key, val in self.operators["site_operators"][
                            op_idx[0]
                        ].items()
                    }
                )

            # Try to compute using only site terms

            term_op = eval_expr(e_expr, e_parms)
            if not isinstance(term_op, str):
                return term_op

            # Try now adding the bond operators
            for name_bop in model.bond_ops:
                self.bond_operator(name_bop, src, dst)
                # Fermionic signs come from the Jordan-Wigner mapping when the
                # system is split into qubits (see ``FermionicSignWarning``).
                self.bond_operator(name_bop, dst, src)

            for src_idx, dst_idx in ((1, 2), (2, 1)):
                e_parms.update(
                    {
                        f"{key[0]}__src_dst": val
                        for key, val in self.operators["bond_operators"].items()
                        if key[src_idx] == src and key[dst_idx] == dst
                    }
                )

            return eval_expr(e_expr, e_parms)

        expr = term_spec["expr"]
        term_type = term_spec.get("type", None)
        t_parm = {}
        t_parm.update(term_spec.get("parms", {}))
        if parms:
            t_parm.update(parms)
        result_terms = []
        for edge_type, edges in graph.edges.items():
            if term_type is not None and term_type != edge_type:
                continue

            e_expr = expr.replace("#", edge_type)
            operator_names = set(re.findall(r"\b([a-zA-Z_]+)\b@", e_expr))
            e_expr = e_expr.replace("@", "__")
            for src, dst in edges:
                term_op = process_edge(e_expr, (edge_type, src, dst), model, t_parm)
                if isinstance(term_op, str):
                    raise ValueError(
                        f"   Bond term <<{term_op}>> could not be evaluated for {operator_names}."
                    )

                result_terms.append(term_op)

        if len(result_terms) == 0:
            return ScalarOperator(0.0, self)
        if len(result_terms) == 1:
            return result_terms[0]
        return SumOperator(tuple(result_terms), self, True)

    def loop_term_from_descriptor(self, term_spec, graph, model, parms):
        """Build loop term from aloop term specification."""
        # Import here to avoid circular dependency
        # pylint: disable=import-outside-toplevel
        from qalma.operators import ScalarOperator, SumOperator

        def process_loop(loop_expr, loop_type, vertices_map, _model, t_parm):
            """Evaluate the loop expression ``loop_expr`` on the sites of a loop."""
            loop_parms = {
                replace_variable_type(key, loop_type): replace_variable_type(
                    val, loop_type
                )
                for key, val in t_parm.items()
            }

            for key, site in vertices_map.items():
                loop_expr = loop_expr.replace(f"@[{key}]", f"__{key}_")
                loop_parms.update(
                    {
                        f"{op_key}__{key}_": val
                        for op_key, val in self.operators["site_operators"][
                            site
                        ].items()
                    }
                )

            # If some operator of the expression is not defined on the
            # sites of the loop, ``eval_expr`` returns the expression as a
            # string. The caller turns that into a ``ValueError``.
            return eval_expr(loop_expr, loop_parms)

        expr = term_spec["expr"]
        term_type = term_spec.get("type", None)
        index_names = term_spec.get("indices", None)
        t_parm = {}
        t_parm.update(term_spec.get("parms", {}))
        if parms:
            t_parm.update(parms)
        result_terms = []
        for loop_type, loops in graph.loops.items():
            if term_type is not None and term_type != loop_type:
                continue
            loop_expr = expr.replace("#", loop_type)
            operator_names = set(re.findall(r"\b([a-zA-Z_]+)\b@", loop_expr))
            for vertices in loops:
                vertices_site_map = dict(zip(index_names, vertices))
                term_op = process_loop(
                    loop_expr, loop_type, vertices_site_map, model, t_parm
                )
                if isinstance(term_op, str):
                    raise ValueError(
                        f"Loop term <<{expr}>> of type <<{loop_type}>> could "
                        f"not be evaluated on the sites {tuple(vertices)}: "
                        "some of the operators "
                        f"{sorted(operator_names)} are not defined there."
                    )

                result_terms.append(term_op)

        if len(result_terms) == 0:
            return ScalarOperator(0.0, self)
        if len(result_terms) == 1:
            return result_terms[0]
        return SumOperator(tuple(result_terms), self, True)

    def global_operator(self, name):
        """Return a global operator by its name."""
        # pylint: disable=import-outside-toplevel
        from qalma.operators import OneBodyOperator, SumOperator

        if len(self.operators["site_operators"]) == 0:
            self._load_site_operators()
            self._load_global_ops()

        result = self.operators["global_operators"].get(name, None)
        if result is not None:
            logger.info("%s already defined. Return from cache.", name)
            return result
        # Build the global_operator from the descriptor
        op_descr = self.spec["model"].global_ops.get(name, None)
        if op_descr is None:
            logger.warning(
                "%s not defined in %s.", name, str(self.spec["model"].global_ops.keys())
            )
            return None

        graph = self.spec["graph"]
        parms = self.spec["parms"]
        model = self.spec["model"]

        # Process site terms
        try:
            site_terms = (
                self.site_term_from_descriptor(term_spec, graph, parms)
                for term_spec in op_descr["site terms"]
            )
            site_terms = tuple(term for term in site_terms if term)
        except ValueError as exc:
            logger.debug("%s aborting evaluation of %s.", exc.args, name)
            model.global_ops.pop(name)
            return None

        # Process bond terms
        try:
            bond_terms = tuple(
                self.bond_term_from_descriptor(term_spec, graph, model, parms)
                for term_spec in op_descr["bond terms"]
            )
            bond_terms = tuple(term for term in bond_terms if term)

        except ValueError as exc:
            logger.debug("%s aborting evaluation of %s.", exc.args, name)
            model.global_ops.pop(name)
            return None

        # Process loop terms
        try:
            loop_terms = tuple(
                self.loop_term_from_descriptor(term_spec, graph, model, parms)
                for term_spec in op_descr["loop terms"]
            )
            loop_terms = tuple(term for term in loop_terms if term)

        except ValueError as exc:
            logger.debug("%s Aborting evaluation of %s.", exc.args, name)
            model.global_ops.pop(name)
            return None

        interaction_terms = bond_terms + loop_terms
        if self.qubit_representation:
            result = self.combine_split_terms(site_terms + interaction_terms)
            self.operators["global_operators"][name] = result
            return result
        if interaction_terms:
            result = SumOperator(site_terms + interaction_terms, self, True)
        else:
            result = OneBodyOperator(site_terms, self, True)
        result = result.simplify()
        self.operators["global_operators"][name] = result
        logger.debug("  return operator of type %s.", str(type(result)))
        return result


def build_spin_chain(length: int = 2, field=0.0):
    """Build a nn spin chain of length ``l`` in an external magnetic field."""
    parameters = {"L": length, "a": 1, "h": field, "J": 1, "Jz0": 1, "Jxy0": 1}
    return build_system("chain lattice", "spin", **parameters)


def build_system(
    geometry_name: str = "chain lattice",
    model_name: str = "spin",
    models_lib_file=MODEL_LIB_FILE,
    lattice_lib_file=LATTICE_LIB_FILE,
    qubit_split: Optional[bool] = None,
    **kwargs,
) -> SystemDescriptor:
    """Build a SystemDescriptor from the names of the geometry and the model.

    lattice_lib_file: str

    qubit_split: bool or None
        Represent the sites by qubits (one per two-level quantum number),
        with fermions mapped through the Jordan-Wigner transformation.
        See :class:`SystemDescriptor`. By default, this is done for
        fermionic models whose sites can be split into qubits.

    kwargs: Optional keyword parameters are passed to the model.
    """
    # pylint: disable=import-outside-toplevel
    from qalma.alpsmodels import model_from_alps_xml

    # pylint: disable=import-outside-toplevel
    from qalma.geometry import graph_from_alps_xml

    logger.info("loading model %s over graph %s.", model_name, geometry_name)

    parms = {"L": 4, "J": 1, "a": 1}
    parms.update(kwargs)
    model = model_from_alps_xml(models_lib_file, model_name, parms)
    graph = graph_from_alps_xml(lattice_lib_file, geometry_name, parms)

    assert model is not None
    assert graph is not None
    return SystemDescriptor(graph, model, parms, qubit_split=qubit_split)


def build_system_from_dims(dims_by_name: Dict[str, int]) -> SystemDescriptor:
    """Build a System from the dimension of each site.

    Parameters
    ----------
    dims_by_name: dict[str, int]
       a dictionary specifying the dimension of each site in the system.

    Return
    =======

    system: SystemDescriptor
        A system with the specified local sites.

    """
    model = qutip_model_from_dims(dims_by_name.values())
    sitebasis = model.site_basis
    sites = {name: sitebasis[f"dim={dim}"] for name, dim in dims_by_name.items()}
    graph = GraphDescriptor(
        "disconnected graph",
        {name: {"type": f"dim={dim}"} for name, dim in dims_by_name.items()},
        {},
        {},
    )
    return SystemDescriptor(graph, model, sites=sites)
