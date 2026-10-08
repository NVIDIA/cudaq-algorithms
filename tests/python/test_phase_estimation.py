# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PhaseEstimation: eigenvalues recovered from a block encoding's walk.

Textbook QPE on the qubitization walk of a ``PauliLCU``. Recovered energies
are checked against the Hamiltonian's exact eigenvalues within the
phase-estimation resolution ``~ alpha * pi / 2^num_bits`` (the half-bin
rounding of the walk eigenphase). qpp-cpu / fp64.
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
    mass = sum(p for e, p in ranked if abs(e - 0.5) < tol)
    assert mass > 0.9


def test_mixed_input_recovers_both_eigenvalues():
    enc = PauliLCU({"Z": 0.3, "X": 0.4})
    qpe = PhaseEstimation(enc)
    m = 5
    # Input |0>: 80% overlap with the +0.5 eigenstate, 20% with -0.5.
    ranked = qpe.energies(m, ket=np.array([1.0, 0.0], dtype=complex))
    tol = resolution_tol(m, enc.alpha)
    # Dominant distinct energy ~ +0.5 (the larger overlap).
    assert ranked[0][0] == pytest.approx(0.5, abs=tol)
    # The minority eigenvalue -0.5 appears among the readouts.
    assert any(e == pytest.approx(-0.5, abs=tol) for e, _ in ranked)
    # Positive-energy mass dominates negative (the 80/20 split).
    pos = sum(p for e, p in ranked if e > 0)
    neg = sum(p for e, p in ranked if e < 0)
    assert pos > neg


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
    with pytest.raises(ValueError, match="exactly one"):
        qpe.energies(4)
    with pytest.raises(ValueError, match="exactly one"):
        qpe.energies(4, ket=np.array([1.0, 0.0]), state_prep=prep_plus)
