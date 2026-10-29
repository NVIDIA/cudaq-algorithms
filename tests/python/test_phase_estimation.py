# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PhaseEstimation: eigenvalues recovered from a block encoding's walk.

Textbook QPE on the qubitization walk of a ``PauliLCU``. Recovered energies
are checked against the Hamiltonian's exact eigenvalues within the
phase-estimation resolution ``~ alpha * pi / 2^num_bits`` (the half-bin
rounding of the walk eigenphase), with the probability masses the input's
spectral weights predict. qpp-cpu / fp64. Each ``energies`` call costs
``2^num_bits - 1`` controlled walk steps (about 0.6 s each here), so the
instances stay at ``num_bits <= 6``.
"""

import math

import numpy as np
import pytest

import cudaq

from cudaq_algorithms import PauliLCU, PhaseEstimation


def resolution_tol(num_bits: int, alpha: float) -> float:
    # dE/dtheta = alpha*sin(theta) <= alpha; half-bin phase error pi/2^m;
    # 1.5x margin for the finite side-lobe spread.
    return 1.5 * alpha * math.pi / (1 << num_bits)


# A 1-qubit Hamiltonian H = 0.3 Z + 0.4 X: eigenvalues +-0.5, alpha = 0.7.
# The +0.5 eigenstate is ry(phi)|0> with phi = atan2(0.4, 0.3).
_PHI_PLUS = math.atan2(0.4, 0.3)


@cudaq.kernel
def prep_plus(q: cudaq.qview):
    ry(_PHI_PLUS, q[0])


@cudaq.kernel
def prep_11(q: cudaq.qview):
    x(q[0])
    x(q[1])


# ----------------------------------------------------------------------
# Energy readout formula
# ----------------------------------------------------------------------


def test_phase_to_energy_formula():
    enc = PauliLCU({"Z": 0.3, "X": 0.4})
    qpe = PhaseEstimation(enc)
    m = 4
    # E = -alpha cos(2 pi j / 2^m).
    for j in (0, 3, 7, 11):
        assert qpe.phase_to_energy(j, m) == pytest.approx(
            -0.7 * math.cos(2 * math.pi * j / (1 << m)), abs=1e-12)


# ----------------------------------------------------------------------
# Eigenvalue recovery
# ----------------------------------------------------------------------


def test_eigenstate_readout_concentrates_on_its_energy():
    enc = PauliLCU({"Z": 0.3, "X": 0.4})
    qpe = PhaseEstimation(enc)
    m = 5
    ranked = qpe.energies(m, state_prep=prep_plus)
    tol = resolution_tol(m, enc.alpha)
    # The +0.5 eigenstate reads +0.5 within resolution, with essentially all
    # the probability on readouts near that energy (the +/-theta mirror pair,
    # each possibly split across adjacent bins).
    assert ranked[0][0] == pytest.approx(0.5, abs=tol)
    # The mirror pair is folded: the top entry alone carries the mass that
    # a pure eigenstate puts on its energy, and no energy repeats.
    assert ranked[0][1] > 0.9
    energies_only = [e for e, _ in ranked]
    assert len(energies_only) == len(set(energies_only)) == (1 << (m - 1)) + 1
    assert sum(p for _, p in ranked) == pytest.approx(1.0, abs=1e-12)


def test_mixed_input_recovers_both_eigenvalues():
    enc = PauliLCU({"Z": 0.3, "X": 0.4})
    qpe = PhaseEstimation(enc)
    m = 5
    # Input |0>: 80% overlap with the +0.5 eigenstate, 20% with -0.5.
    ranked = qpe.energies(m, ket=np.array([1.0, 0.0], dtype=complex))
    tol = resolution_tol(m, enc.alpha)
    # Dominant distinct energy ~ +0.5 (the larger overlap) with ~0.8 of the
    # mass, the minority eigenvalue -0.5 with ~0.2: the spectral weights of
    # |0> = cos(phi/2)|+> - sin(phi/2)|->.
    assert ranked[0][0] == pytest.approx(0.5, abs=tol)
    pos = sum(p for e, p in ranked if abs(e - 0.5) < tol)
    neg = sum(p for e, p in ranked if abs(e + 0.5) < tol)
    assert pos == pytest.approx(0.8, abs=0.03)
    assert neg == pytest.approx(0.2, abs=0.03)


def test_two_qubit_diagonal_ground_energy():
    # H = 0.5 Z0 + 0.3 Z1 + 0.2 Z0 Z1 (diagonal); alpha = 1.0.
    # Computational eigenstates: |11> -> -0.5-0.3+0.2 = -0.6 (the ground).
    enc = PauliLCU({"ZI": 0.5, "IZ": 0.3, "ZZ": 0.2})
    assert enc.alpha == pytest.approx(1.0, abs=1e-12)
    qpe = PhaseEstimation(enc)
    m = 6
    ranked = qpe.energies(m, state_prep=prep_11)
    tol = resolution_tol(m, enc.alpha)
    assert ranked[0][0] == pytest.approx(-0.6, abs=tol)
    # This eigenphase is mid-bin, so the +/-theta lobes each split across
    # adjacent bins; the main lobe (within ~2 resolution units) holds >0.9.
    mass = sum(p for e, p in ranked if abs(e - (-0.6)) < 2 * tol)
    assert mass > 0.9


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def test_validation():
    enc = PauliLCU({"Z": 0.3, "X": 0.4})
    qpe = PhaseEstimation(enc)
    with pytest.raises(ValueError, match="num_bits"):
        qpe.kernel(0)
    with pytest.raises(ValueError, match="num_bits"):
        qpe.kernel(True)
    with pytest.raises(ValueError, match="exactly one"):
        qpe.energies(4)
    with pytest.raises(ValueError, match="exactly one"):
        qpe.energies(4, ket=np.array([1.0, 0.0]), state_prep=prep_plus)
    with pytest.raises(ValueError, match="top"):
        qpe.energies(2, ket=np.array([1.0, 0.0]), top=0)
    with pytest.raises(ValueError, match="top"):
        qpe.energies(2, ket=np.array([1.0, 0.0]), top=-1)
    assert repr(qpe) == f"PhaseEstimation({enc!r})"


# ----------------------------------------------------------------------
# Readout shapes, equivalent input forms, small and on-grid cases
# ----------------------------------------------------------------------


def test_one_bit_readout_splits_by_spectral_weight():
    # m = 1 has two readouts, j = 0 (E = -alpha) and j = 1 (E = +alpha);
    # nothing to fold. |0> on H = 0.3 Z + 0.4 X puts its +0.5 / -0.5
    # weights on the +alpha / -alpha bins in textbook-QPE proportions.
    enc = PauliLCU({"Z": 0.3, "X": 0.4})
    qpe = PhaseEstimation(enc)
    probabilities = qpe.phase_probabilities(1, ket=np.array([1.0, 0.0]))
    assert probabilities.shape == (2, )
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    ranked = qpe.energies(1, ket=np.array([1.0, 0.0]))
    assert len(ranked) == 2
    assert sorted(e for e, _ in ranked) == pytest.approx([-0.7, 0.7],
                                                         abs=1e-12)
    assert ranked[0][0] == pytest.approx(0.7, abs=1e-12)
    assert ranked[0][1] > ranked[1][1]


def test_on_grid_eigenvalue_reads_exactly():
    # H = 0.5 Z: alpha = 0.5, |1> has E = -0.5 = -alpha cos(0), i.e. the
    # readout j = 0 exactly -- a single-term encoding with an idle ancilla.
    enc = PauliLCU({"Z": 0.5})
    qpe = PhaseEstimation(enc)
    ranked = qpe.energies(3, ket=np.array([0.0, 1.0]), top=1)
    assert ranked == [(pytest.approx(-0.5,
                                     abs=1e-12), pytest.approx(1.0,
                                                               abs=1e-10))]


def test_ket_and_state_prep_inputs_agree():
    enc = PauliLCU({"Z": 0.3, "X": 0.4})
    qpe = PhaseEstimation(enc)
    plus = np.array([math.cos(_PHI_PLUS / 2),
                     math.sin(_PHI_PLUS / 2)],
                    dtype=complex)
    from_ket = qpe.phase_probabilities(4, ket=plus)
    from_state = qpe.phase_probabilities(4, ket=cudaq.State.from_data(plus))
    from_prep = qpe.phase_probabilities(4, state_prep=prep_plus)
    np.testing.assert_allclose(from_ket, from_prep, atol=1e-10)
    np.testing.assert_allclose(from_state, from_prep, atol=1e-10)
    # The folded energies are the mirror sums of the raw marginal.
    ranked = dict(qpe.energies(4, ket=plus))
    half = 1 << 3
    folded = from_ket[:half + 1].copy()
    folded[1:half] += from_ket[-1:half:-1]
    for j in range(half + 1):
        assert ranked[qpe.phase_to_energy(j, 4)] == pytest.approx(folded[j],
                                                                  abs=1e-12)


# ----------------------------------------------------------------------
# Simulation runs on a large-stack worker thread
# ----------------------------------------------------------------------


def test_run_on_large_stack_runs_the_call_on_a_worker_thread():
    import threading

    from cudaq_algorithms import sim_utils

    seen = {}

    def probe(a, b=1):
        seen["thread"] = threading.current_thread()
        return a + b

    assert sim_utils.run_on_large_stack(probe, 2, b=3) == 5
    assert seen["thread"] is not threading.main_thread()
    assert threading.stack_size() == 0  # the process default is restored

    def fail():
        raise ZeroDivisionError("propagated")

    with pytest.raises(ZeroDivisionError, match="propagated"):
        sim_utils.run_on_large_stack(fail)
    assert threading.stack_size() == 0


def test_phase_probabilities_simulates_on_the_large_stack(monkeypatch):
    # The 2^m - 1 walk steps grow the host stack by about 1 kB each on the
    # simulator (captured tables re-materialized per loop iteration), so
    # the statevector is taken on the large-stack worker.
    from cudaq_algorithms import phase_estimation as module

    calls = []
    real = module.run_on_large_stack

    def recording(function, /, *args, **kwargs):
        calls.append(function)
        return real(function, *args, **kwargs)

    monkeypatch.setattr(module, "run_on_large_stack", recording)
    qpe = PhaseEstimation(PauliLCU({"Z": 0.3, "X": 0.4}))
    direct = qpe.phase_probabilities(2, state_prep=prep_plus)
    assert calls == [cudaq.get_state]
    ket = np.array([math.cos(_PHI_PLUS / 2), math.sin(_PHI_PLUS / 2)])
    from_ket = qpe.phase_probabilities(2, ket=ket)
    assert calls == [cudaq.get_state, cudaq.get_state]
    np.testing.assert_allclose(from_ket, direct, atol=1e-12)
