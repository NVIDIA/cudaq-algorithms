# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""State-preparation device kernels (pure Python, composable).

Ports of the ``__qpu__`` kernels in ``lib/stateprep/device/``. The
``uccsd`` kernel reproduces the C++ excitation enumeration with direct
index arithmetic (the interleaved-layout orbital indices are affine in the
loop counters, so no in-kernel lists are needed); gate sequences and the
parameter order match the C++ exactly. ``uccgsd``, ``upccgsd``, and
``ceo`` consume the grouped Pauli words and coefficients produced by the
``get_*_pauli_lists`` helpers, one parameter per group.

Descending CNOT ladders are written as ``range(hi, lo, -1)`` just like the
C++; guards use positive ``if`` blocks, never early ``return`` (kernel
``return`` is silently ignored by the compiler,
https://github.com/NVIDIA/cuda-quantum/issues/4845).
"""

from __future__ import annotations

import cudaq

M_PI_2 = 1.5707963267948966


@cudaq.kernel
def single_excitation(qubits: cudaq.qview, theta: float, p_occ: int,
                      q_virt: int):
    """exp(0.5i * theta * (Y_p Z... X_q - X_p Z... Y_q)) via CNOT ladders."""
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
def double_excitation(qubits: cudaq.qview, theta: float, p_occ: int,
                      q_occ: int, r_virt: int, s_virt: int):
    """The 8-term double-excitation circuit (C++ gate-for-gate port)."""
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
