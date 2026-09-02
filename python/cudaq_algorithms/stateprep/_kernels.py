# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State-preparation device kernels (pure Python, composable).

Ports of the ``__qpu__`` kernels in ``lib/stateprep/device/``. The
``uccsd`` reproduces the C++ excitation enumeration with direct index
arithmetic (the interleaved-layout orbital indices are affine in the loop
counters, so no in-kernel lists are needed). Its parameter order and
non-interleaved gate sequences match the C++; interleaved doubles use sparse
Pauli rotations so overlapping Jordan-Wigner strings remain correct.
``uccgsd``, ``upccgsd``, and ``ceo`` consume the grouped Pauli words and
coefficients produced by the ``get_*_pauli_lists`` helpers, one parameter per
group.

``hartree_fock`` and ``hartree_fock_occupation`` prepare the reference
determinant those ansatz kernels are applied to, and
``fixed_parameter_ucc`` applies an arbitrary grouped Pauli product at
fixed amplitudes (one parameter per excitation group) on top of it; the
occupation builders and the pool-to-Pauli-list converter live in
``_hartree_fock``.

Descending CNOT ladders are written as ``range(hi, lo, -1)`` just like the
C++; guards use positive ``if`` blocks, never early ``return`` (kernel
``return`` is silently ignored by the compiler,
https://github.com/NVIDIA/cuda-quantum/issues/4845).
"""

from __future__ import annotations

import cudaq

M_PI_2 = 1.5707963267948966


@cudaq.kernel
def hartree_fock(qubits: cudaq.qview, num_electrons: int):
    """Fill the first ``num_electrons`` spin orbitals (closed-shell HF).

    For an open-shell system build the occupation with
    ``make_hartree_fock_occupation(num_qubits, num_electrons, spin)`` and
    use ``hartree_fock_occupation`` instead.
    """
    for i in range(num_electrons):
        x(qubits[i])


@cudaq.kernel
def hartree_fock_occupation(qubits: cudaq.qview, occupied_orbitals: list[int]):
    """Prepare a determinant from explicit occupied spin-orbital indices.

    The indices must be valid and unique (validate on the host with
    ``validate_hartree_fock_occupation``); the list must be non-empty
    (empty lists cannot cross the kernel boundary, cuda-quantum#4847).
    """
    for i in range(len(occupied_orbitals)):
        x(qubits[occupied_orbitals[i]])


@cudaq.kernel
def single_excitation(qubits: cudaq.qview, theta: float, p_occ: int,
                      q_virt: int):
    """exp(-0.25i * theta * (Y_p Z... X_q - X_p Z... Y_q)) via CNOT ladders."""
    # Y_p X_q
    rx(M_PI_2, qubits[p_occ])
    h(qubits[q_virt])

    for i in range(p_occ, q_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(0.5 * theta, qubits[q_virt])

    for i in range(q_virt, p_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    h(qubits[q_virt])
    rx(-M_PI_2, qubits[p_occ])

    # -X_p Y_q
    h(qubits[p_occ])
    rx(M_PI_2, qubits[q_virt])

    for i in range(p_occ, q_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(-0.5 * theta, qubits[q_virt])

    for i in range(q_virt, p_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    rx(-M_PI_2, qubits[q_virt])
    h(qubits[p_occ])


@cudaq.kernel
def _ordered_double_excitation(qubits: cudaq.qview, theta: float, p_occ: int,
                               q_occ: int, r_virt: int, s_virt: int):
    """The original ladder circuit for occupied indices below virtual ones."""
    i_occ = 0
    j_occ = 0
    a_virt = 0
    b_virt = 0
    t = theta
    if (p_occ < q_occ) and (r_virt < s_virt):
        i_occ = p_occ
        j_occ = q_occ
        a_virt = r_virt
        b_virt = s_virt
    elif (p_occ > q_occ) and (r_virt > s_virt):
        i_occ = q_occ
        j_occ = p_occ
        a_virt = s_virt
        b_virt = r_virt
    elif (p_occ < q_occ) and (r_virt > s_virt):
        i_occ = p_occ
        j_occ = q_occ
        a_virt = s_virt
        b_virt = r_virt
        t = -theta
    elif (p_occ > q_occ) and (r_virt < s_virt):
        i_occ = q_occ
        j_occ = p_occ
        a_virt = r_virt
        b_virt = s_virt
        t = -theta

    # Block 1: X_i X_j X_a Y_b
    h(qubits[i_occ])
    h(qubits[j_occ])
    h(qubits[a_virt])
    rx(M_PI_2, qubits[b_virt])

    for i in range(i_occ, j_occ):
        x.ctrl(qubits[i], qubits[i + 1])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(0.125 * t, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    rx(-M_PI_2, qubits[b_virt])
    h(qubits[a_virt])

    # Block 2: X_i X_j Y_a X_b
    rx(M_PI_2, qubits[a_virt])
    h(qubits[b_virt])

    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(0.125 * t, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    for i in range(j_occ, i_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    rx(-M_PI_2, qubits[a_virt])
    h(qubits[j_occ])

    # Block 3: X_i Y_j X_a X_b
    rx(M_PI_2, qubits[j_occ])
    h(qubits[a_virt])

    for i in range(i_occ, j_occ):
        x.ctrl(qubits[i], qubits[i + 1])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(-0.125 * t, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    h(qubits[b_virt])
    h(qubits[a_virt])

    # Block 4: X_i Y_j Y_a Y_b
    rx(M_PI_2, qubits[a_virt])
    rx(M_PI_2, qubits[b_virt])

    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(0.125 * t, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    for i in range(j_occ, i_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    rx(-M_PI_2, qubits[j_occ])
    h(qubits[i_occ])

    # Block 5: Y_i X_j Y_a Y_b
    rx(M_PI_2, qubits[i_occ])
    h(qubits[j_occ])

    for i in range(i_occ, j_occ):
        x.ctrl(qubits[i], qubits[i + 1])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(0.125 * t, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    rx(-M_PI_2, qubits[b_virt])
    rx(-M_PI_2, qubits[a_virt])

    h(qubits[a_virt])
    h(qubits[b_virt])

    # Block 6: Y_i X_j X_a X_b
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(-0.125 * t, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    for i in range(j_occ, i_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    h(qubits[b_virt])
    h(qubits[j_occ])

    # Block 7: Y_i Y_j X_a Y_b
    rx(M_PI_2, qubits[j_occ])
    rx(M_PI_2, qubits[b_virt])

    for i in range(i_occ, j_occ):
        x.ctrl(qubits[i], qubits[i + 1])
    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(-0.125 * t, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    rx(-M_PI_2, qubits[b_virt])
    h(qubits[a_virt])

    # Block 8: Y_i Y_j Y_a X_b
    rx(M_PI_2, qubits[a_virt])
    h(qubits[b_virt])

    x.ctrl(qubits[j_occ], qubits[a_virt])
    for i in range(a_virt, b_virt):
        x.ctrl(qubits[i], qubits[i + 1])

    rz(-0.125 * t, qubits[b_virt])

    for i in range(b_virt, a_virt, -1):
        x.ctrl(qubits[i - 1], qubits[i])
    x.ctrl(qubits[j_occ], qubits[a_virt])

    for i in range(j_occ, i_occ, -1):
        x.ctrl(qubits[i - 1], qubits[i])

    h(qubits[b_virt])
    rx(-M_PI_2, qubits[a_virt])
    rx(-M_PI_2, qubits[j_occ])
    rx(-M_PI_2, qubits[i_occ])


@cudaq.kernel
def _double_pauli_rotation(qubits: cudaq.qview, angle: float, i_occ: int,
                           j_occ: int, a_virt: int, b_virt: int, op_i: int,
                           op_j: int, op_a: int, op_b: int):
    """Rotate one double-excitation Pauli term.

    ``op_*`` is 0 for X and 1 for Y. Overlapping Jordan-Wigner parity
    intervals change endpoint operators and can contribute a minus sign.
    """
    local_i = op_i
    local_j = op_j
    local_a = op_a
    local_b = op_b
    phase_exponent = 0

    # Occupied endpoint operator precedes the virtual parity string.
    if (a_virt < i_occ) and (i_occ < b_virt):
        local_i = 1 - local_i
        if op_i == 0:
            phase_exponent += 3  # XZ = -iY
        else:
            phase_exponent += 1  # YZ = iX
    if (a_virt < j_occ) and (j_occ < b_virt):
        local_j = 1 - local_j
        if op_j == 0:
            phase_exponent += 3
        else:
            phase_exponent += 1

    # Occupied parity string precedes the virtual endpoint operator.
    if (i_occ < a_virt) and (a_virt < j_occ):
        local_a = 1 - local_a
        if op_a == 0:
            phase_exponent += 1  # ZX = iY
        else:
            phase_exponent += 3  # ZY = -iX
    if (i_occ < b_virt) and (b_virt < j_occ):
        local_b = 1 - local_b
        if op_b == 0:
            phase_exponent += 1
        else:
            phase_exponent += 3

    phase_sign = 1.0
    if phase_exponent % 4 == 2:
        phase_sign = -1.0

    if local_i == 0:
        h(qubits[i_occ])
    else:
        rx(M_PI_2, qubits[i_occ])
    if local_j == 0:
        h(qubits[j_occ])
    else:
        rx(M_PI_2, qubits[j_occ])
    if local_a == 0:
        h(qubits[a_virt])
    else:
        rx(M_PI_2, qubits[a_virt])
    if local_b == 0:
        h(qubits[b_virt])
    else:
        rx(M_PI_2, qubits[b_virt])

    # Accumulate the sparse Pauli parity directly on b_virt. The two
    # Jordan-Wigner strings cancel where their interiors overlap.
    for index in range(qubits.size()):
        is_endpoint = ((index == i_occ) or (index == j_occ)
                       or (index == a_virt) or (index == b_virt))
        in_occ_parity = (i_occ < index) and (index < j_occ)
        in_virt_parity = (a_virt < index) and (index < b_virt)
        in_one_parity = in_occ_parity != in_virt_parity
        if (index != b_virt) and (is_endpoint or in_one_parity):
            x.ctrl(qubits[index], qubits[b_virt])

    rz(phase_sign * angle, qubits[b_virt])

    for index in range(qubits.size()):
        is_endpoint = ((index == i_occ) or (index == j_occ)
                       or (index == a_virt) or (index == b_virt))
        in_occ_parity = (i_occ < index) and (index < j_occ)
        in_virt_parity = (a_virt < index) and (index < b_virt)
        in_one_parity = in_occ_parity != in_virt_parity
        if (index != b_virt) and (is_endpoint or in_one_parity):
            x.ctrl(qubits[index], qubits[b_virt])

    if local_b == 0:
        h(qubits[b_virt])
    else:
        rx(-M_PI_2, qubits[b_virt])
    if local_a == 0:
        h(qubits[a_virt])
    else:
        rx(-M_PI_2, qubits[a_virt])
    if local_j == 0:
        h(qubits[j_occ])
    else:
        rx(-M_PI_2, qubits[j_occ])
    if local_i == 0:
        h(qubits[i_occ])
    else:
        rx(-M_PI_2, qubits[i_occ])


@cudaq.kernel
def double_excitation(qubits: cudaq.qview, theta: float, p_occ: int,
                      q_occ: int, r_virt: int, s_virt: int):
    """Apply a double excitation for any ordering of four distinct indices."""
    i_occ = 0
    j_occ = 0
    a_virt = 0
    b_virt = 0
    t = theta
    if (p_occ < q_occ) and (r_virt < s_virt):
        i_occ = p_occ
        j_occ = q_occ
        a_virt = r_virt
        b_virt = s_virt
    elif (p_occ > q_occ) and (r_virt > s_virt):
        i_occ = q_occ
        j_occ = p_occ
        a_virt = s_virt
        b_virt = r_virt
    elif (p_occ < q_occ) and (r_virt > s_virt):
        i_occ = p_occ
        j_occ = q_occ
        a_virt = s_virt
        b_virt = r_virt
        t = -theta
    elif (p_occ > q_occ) and (r_virt < s_virt):
        i_occ = q_occ
        j_occ = p_occ
        a_virt = r_virt
        b_virt = s_virt
        t = -theta

    if j_occ < a_virt:
        _ordered_double_excitation(qubits, t, i_occ, j_occ, a_virt, b_virt)
    else:
        scale = 0.125 * t
        _double_pauli_rotation(qubits, scale, i_occ, j_occ, a_virt, b_virt, 0,
                               0, 0, 1)
        _double_pauli_rotation(qubits, scale, i_occ, j_occ, a_virt, b_virt, 0,
                               0, 1, 0)
        _double_pauli_rotation(qubits, scale, i_occ, j_occ, a_virt, b_virt, 0,
                               1, 1, 1)
        _double_pauli_rotation(qubits, scale, i_occ, j_occ, a_virt, b_virt, 1,
                               0, 1, 1)
        _double_pauli_rotation(qubits, -scale, i_occ, j_occ, a_virt, b_virt, 0,
                               1, 0, 0)
        _double_pauli_rotation(qubits, -scale, i_occ, j_occ, a_virt, b_virt, 1,
                               0, 0, 0)
        _double_pauli_rotation(qubits, -scale, i_occ, j_occ, a_virt, b_virt, 1,
                               1, 0, 1)
        _double_pauli_rotation(qubits, -scale, i_occ, j_occ, a_virt, b_virt, 1,
                               1, 1, 0)


@cudaq.kernel
def uccsd(qubits: cudaq.qview, thetas: list[float], num_electrons: int,
          spin: int):
    """UCCSD ansatz over the interleaved spin-orbital layout.

    Parameter order matches ``get_uccsd_excitations``: singles alpha,
    singles beta, mixed doubles, alpha doubles, beta doubles. The orbital
    indices are computed inline: occupied alpha 2i, virtual alpha
    2j + 2*n_occ_alpha, occupied beta 2i + 1, virtual beta
    2j + 2*n_occ_beta + 1 (for spin == 0 these reduce to the closed-shell
    formulas of the C++ implementation since 2*n_occ = num_electrons;
    they differ from the C++ only for the invalid odd-electrons/spin==0
    input, which the host API rejects). The kernel performs no input
    validation — device kernels have no error channel — so arguments
    must satisfy the ``get_uccsd_excitations`` contract, and ``thetas``
    must hold ``get_num_uccsd_parameters`` entries.

    Implementation caution: the virtual-orbital offsets must stay
    inlined in the index expressions below (``2 * n_occ_alpha`` etc.).
    Introducing dedicated offset variables that are reassigned inside
    the ``spin > 0`` branch (mirroring the C++'s ``num_electrons``
    closed-shell offsets) miscompiled on CUDA-Q 0.15: successive
    launches of a calling kernel at different qubit counts accumulated
    qubits in ``get_state``. The existing ``n_occ_alpha``/``n_occ_beta``
    reassignments just below are fine — the failing shape was
    specifically the additional offset variables.
    """
    n_spatial = qubits.size() // 2
    n_occ_alpha = num_electrons // 2
    n_occ_beta = num_electrons // 2
    if spin > 0:
        n_occ_beta = (num_electrons - spin) // 2
        n_occ_alpha = num_electrons - n_occ_beta
    n_virt_alpha = n_spatial - n_occ_alpha
    n_virt_beta = n_spatial - n_occ_beta

    counter = 0

    # Singles alpha: (occ_alpha[i], virt_alpha[j])
    for i in range(n_occ_alpha):
        for j in range(n_virt_alpha):
            single_excitation(qubits, thetas[counter], 2 * i,
                              2 * j + 2 * n_occ_alpha)
            counter += 1

    # Singles beta: (occ_beta[i], virt_beta[j])
    for i in range(n_occ_beta):
        for j in range(n_virt_beta):
            single_excitation(qubits, thetas[counter], 2 * i + 1,
                              2 * j + 2 * n_occ_beta + 1)
            counter += 1

    # Mixed doubles: (occ_alpha[i], occ_beta[j], virt_beta[k], virt_alpha[l])
    for i in range(n_occ_alpha):
        for j in range(n_occ_beta):
            for k in range(n_virt_beta):
                for l in range(n_virt_alpha):
                    double_excitation(qubits, thetas[counter], 2 * i,
                                      2 * j + 1, 2 * k + 2 * n_occ_beta + 1,
                                      2 * l + 2 * n_occ_alpha)
                    counter += 1

    # Alpha doubles: p < q occupied, r < s virtual
    for p in range(n_occ_alpha - 1):
        for q in range(p + 1, n_occ_alpha):
            for r in range(n_virt_alpha - 1):
                for s in range(r + 1, n_virt_alpha):
                    double_excitation(qubits, thetas[counter], 2 * p, 2 * q,
                                      2 * r + 2 * n_occ_alpha,
                                      2 * s + 2 * n_occ_alpha)
                    counter += 1

    # Beta doubles: p < q occupied, r < s virtual
    for p in range(n_occ_beta - 1):
        for q in range(p + 1, n_occ_beta):
            for r in range(n_virt_beta - 1):
                for s in range(r + 1, n_virt_beta):
                    double_excitation(qubits, thetas[counter], 2 * p + 1,
                                      2 * q + 1, 2 * r + 2 * n_occ_beta + 1,
                                      2 * s + 2 * n_occ_beta + 1)
                    counter += 1


@cudaq.kernel
def uccgsd(qubits: cudaq.qview, thetas: list[float],
           pauli_words_list: list[list[cudaq.pauli_word]],
           coefficients_list: list[list[float]]):
    """Generalized UCC product: one parameter per Pauli-word group."""
    for i in range(len(pauli_words_list)):
        theta = thetas[i]
        words = pauli_words_list[i]
        coefficients = coefficients_list[i]
        for j in range(len(words)):
            exp_pauli(theta * coefficients[j], qubits, words[j])


@cudaq.kernel
def upccgsd(qubits: cudaq.qview, thetas: list[float],
            pauli_words_list: list[list[cudaq.pauli_word]],
            coefficients_list: list[list[float]]):
    """Paired generalized UCC product: one parameter per group."""
    for i in range(len(pauli_words_list)):
        theta = thetas[i]
        words = pauli_words_list[i]
        coefficients = coefficients_list[i]
        for j in range(len(words)):
            exp_pauli(theta * coefficients[j], qubits, words[j])


@cudaq.kernel
def ceo(qubits: cudaq.qview, thetas: list[float],
        pauli_words_list: list[list[cudaq.pauli_word]],
        coefficients_list: list[list[float]]):
    """Coupled-exchange-operator product: one parameter per group."""
    for i in range(len(pauli_words_list)):
        theta = thetas[i]
        words = pauli_words_list[i]
        coefficients = coefficients_list[i]
        for j in range(len(words)):
            exp_pauli(theta * coefficients[j], qubits, words[j])


@cudaq.kernel
def fixed_parameter_ucc(qubits: cudaq.qview, thetas: list[float],
                        pauli_words_list: list[list[cudaq.pauli_word]],
                        coefficients_list: list[list[float]]):
    """Fixed-amplitude UCC product over an arbitrary operator pool.

    One (fixed, non-variational) parameter per excitation group; the
    grouped words and coefficients come from any pool via
    ``get_fixed_parameter_ucc_pauli_lists`` (or from the
    ``get_*_pauli_lists`` helpers directly). The qubits must already hold
    a reference determinant — prepare it with ``hartree_fock`` /
    ``hartree_fock_occupation`` first; applied to |0...0> this yields a
    physically meaningless state.
    """
    for i in range(len(pauli_words_list)):
        theta = thetas[i]
        words = pauli_words_list[i]
        coefficients = coefficients_list[i]
        for j in range(len(words)):
            exp_pauli(theta * coefficients[j], qubits, words[j])
