# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Representative focused test from checkout revision current-fixture-r2."""

import pytest

from qsvt_current_contract import PhaseSequence, QSVT


def test_kernel_requires_a_tagged_phase_sequence():
    transformer = QSVT()
    sequence = PhaseSequence([0.1, -0.2, 0.3], convention="qsp")

    assert transformer.kernel(sequence)[0] is sequence
    with pytest.raises(TypeError, match="PhaseSequence"):
        transformer.kernel([0.1, -0.2, 0.3])


def test_removed_factory_convention_keyword_is_rejected():
    transformer = QSVT()
    sequence = PhaseSequence([0.1, -0.2, 0.3], convention="qsp")

    with pytest.raises(TypeError):
        transformer.kernel(sequence, convention="qsp")
