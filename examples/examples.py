# -*- coding: utf-8 -*-
"""
Examples
"""


from qalma.model import SystemDescriptor, build_spin_chain
from qalma.operators import (
    OneBodyOperator,
    ScalarOperator,
    SumOperator,
)
from qalma.quadratic import build_quadratic_form_from_operator

CHAIN_SIZE = 6

system: SystemDescriptor = build_spin_chain(CHAIN_SIZE)
sites: tuple = tuple(s for s in system.sites.keys())

sz_total: OneBodyOperator = system.global_operator("Sz")
sx_total: OneBodyOperator = sum(system.site_operator("Sx", s) for s in sites)
sy_total: OneBodyOperator = sum(system.site_operator("Sy", s) for s in sites)
hamiltonian: SumOperator = system.global_operator("Hamiltonian")

global_identity: ScalarOperator = ScalarOperator(1.0, system)

sx_A = system.site_operator(f"Sx@{sites[0]}")
sx_B = system.site_operator(f"Sx@{sites[1]}")
sx_AB = 0.7 * sx_A + 0.3 * sx_B


sy_A = system.site_operator(f"Sy@{sites[0]}")
sy_B = system.site_operator(f"Sy@{sites[1]}")


splus_A = system.site_operator(f"Splus@{sites[0]}")
splus_B = system.site_operator(f"Splus@{sites[1]}")
sminus_A = system.site_operator(f"Sminus@{sites[0]}")
sminus_B = system.site_operator(f"Sminus@{sites[1]}")

sz_A = system.site_operator(f"Sz@{sites[0]}")
sz_B = system.site_operator(f"Sz@{sites[1]}")
sz_C = system.site_operator(f"Sz@{sites[2]}")
sz_AB = 0.7 * sz_A + 0.3 * sz_B


sh_A = 0.25 * sx_A + 0.5 * sz_A
sh_B = 0.25 * sx_B + 0.5 * sz_B
sh_AB = 0.7 * sh_A + 0.3 * sh_B


subsystems = [
    [sites[0]],
    [sites[1]],
    [sites[2]],
    [sites[0], sites[1]],
    [sites[0], sites[2]],
    [sites[2], sites[3]],
]


observable_cases = {
    "Identity": ScalarOperator(1.0, system),
    "sz_total": sz_total,  # OneBodyOperator
    "sx_A": sx_A,  # LocalOperator
    "sy_A": sy_A,  # Local Operator
    "sz_B": sz_B,  # Diagonal local operator
    "sh_AB": sh_AB,  # ProductOperator
    "exchange_AB": sx_A * sx_B + sy_A * sy_B,  # Sum operator
    "hamiltonian": hamiltonian,  # Sum operator, hermitician
    "observable array": [[sh_AB, sh_A], [sz_A, sx_A]],
}


operator_type_cases = {
    "scalar, real": ScalarOperator(1.0, system),
    "scalar, complex": ScalarOperator(1.0 + 3j, system),
    "local operator, hermitician": sx_A,  # LocalOperator
    "local operator, non hermitician": sx_A + sy_A * 1j,
    "One body, hermitician": sz_total,
    "One body, non hermitician": sx_total + sy_total * 1j,
    "three body, hermitician": (sx_A * sy_B * sz_C),
    "three body, non hermitician": (sminus_A * sminus_B + sy_A * sy_B) * sz_total,
    "product operator, hermitician": sh_AB,
    "product operator, non hermitician": sminus_A * splus_B,
    "sum operator, hermitician": sx_A * sx_B + sy_A * sy_B,  # Sum operator
    "sum operator, hermitician from non hermitician": splus_A * splus_B
    + sminus_A * sminus_B,
    "sum operator, anti-hermitician": splus_A * splus_B - sminus_A * sminus_B,
    "qutip operator": hamiltonian.to_qutip_operator(),
    "hermitician quadratic operator": build_quadratic_form_from_operator(hamiltonian),
    "non hermitician quadratic operator": build_quadratic_form_from_operator(
        hamiltonian - sz_total * 1j
    ),
}


def test_mean_field():
    hamiltonian_qf = build_quadratic_form_from_operator(hamiltonian)


if __name__ == "__main__":
    # test_isherm_operator()
    test_call_function()
