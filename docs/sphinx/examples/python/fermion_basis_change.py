# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed, controlled and QROM-selected fermionic transformations.

Run with:  python3 fermion_basis_change.py  (with the package on PYTHONPATH)
The checks need double precision, so the example selects qpp-cpu unless
CUDAQ_DEFAULT_SIMULATOR names another simulator.
The selected example allocates sixteen qubits, including twelve angle qubits.
"""

# [Begin Documentation]
import os

import cudaq
import numpy as np

from cudaq_algorithms import fermion_basis_change
from cudaq_algorithms.primitives import QROM

# fp64 simulation: the checks below assert to about 1e-12.
cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

# A 2x2 mode matrix acts on TWO occupation qubits. Its full Fock action
# preserves vacuum, acts as U on one particle, and as det(U) on two particles.
U = np.array([[1., -1.], [1j, 1j]]) / np.sqrt(2)
fixed = fermion_basis_change(U)
apply_fixed = fixed.kernel()


@cudaq.kernel
def fixed_demo():
    system = cudaq.qvector(2)
    h(system[0])
    h(system[1])
    apply_fixed(system)


fock = np.zeros((4, 4), dtype=complex)
fock[0, 0] = 1
fock[1:3, 1:3] = U
fock[3, 3] = np.linalg.det(U)
np.testing.assert_allclose(np.asarray(cudaq.get_state(fixed_demo)),
                           fock @ (np.ones(4) / 2),
                           atol=1e-12)

# An external enable can coherently control the same fixed transformation.
# The returned wrapper expects [enable, system...]. Its angles stay classical.
controlled_fixed = fixed.controlled_kernel()


@cudaq.kernel
def controlled_demo():
    q = cudaq.qvector(3)
    h(q[0])
    x(q[1])
    controlled_fixed(q)


expected_controlled = np.zeros(8, dtype=complex)
expected_controlled[2] = 1 / np.sqrt(2)  # enable=0, mode 0 unchanged
expected_controlled[3] = U[0, 0] / np.sqrt(2)
expected_controlled[5] = U[1, 0] / np.sqrt(2)
np.testing.assert_allclose(np.asarray(cudaq.get_state(controlled_demo)),
                           expected_controlled,
                           atol=1e-12)

# Now choose I or U using a quantum address. A common QR schedule consumes
# angle words. QROM is an explicit choice of data source for those words.
bits = 3
plans = [
    fermion_basis_change(matrix, programmable=True, angle_bits=bits)
    for matrix in (np.eye(2), U)
]
program = plans[0]
assert all(p.structure == program.structure for p in plans)
assert plans[0].encoded_words == (0, 0, 0, 0)
assert plans[1].encoded_words == (0, 2, 1, 2)

# Four slots: diagonal phases, mixing angle, relative phase. Keep zero slots:
# the loaded quantum words override every slot, including host zero angles.
# With the default layout and the output view q[2:span], slot s starts at
# output bit bits*s; other layouts use parameter_registers (see the guide).
rows = [
    sum(code << (bits * slot) for slot, code in enumerate(p.encoded_words))
    for p in plans
]
table = QROM(rows,
             address_bits=1,
             output_bits=bits * program.structure.num_parameters,
             variant="select")
lookup = table.kernel
consume = program.kernel()
span = program.num_qubits  # 2 system + 4*3 angle = 14
address = span
scratch = address + table.num_address
total = scratch + table.num_ladder  # 16, including address and QROM scratch


@cudaq.kernel
def selected_preparation():
    q = cudaq.qvector(total)
    x(q[0])
    h(q[address])
    lookup(q[address:scratch], q[scratch:total], q[2:span])
    consume(q[0:span])
    lookup(q[address:scratch], q[scratch:total], q[2:span])


# |0>_addr |mode 0>/sqrt(2) + |1>_addr (|mode 0>+i|mode 1>)/2.
# Compare the complete state, so nonzero angle/scratch outputs also fail.
expected = np.zeros(1 << total, dtype=complex)
expected[1] = 1 / np.sqrt(2)
expected[(1 << address) + 1] = 0.5
expected[(1 << address) + 2] = 0.5j
state = np.asarray(cudaq.get_state(selected_preparation))
np.testing.assert_allclose(state, expected, atol=1e-11)
print("Selected preparation:",
      [(i, complex(a)) for i, a in enumerate(state) if abs(a) > 1e-10])
print("Register widths:", program.structure.num_system_qubits, "system +",
      table.num_output, "angles +", table.num_address, "address +",
      table.num_ladder, "scratch =", total)
print("Quantization bounds:", [p.quantization_error_bound for p in plans])
print("Fixed, controlled, and selected checks passed.")
