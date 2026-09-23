# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E05 small-register outcome checks; execute artifacts only inside isolation.

This module is not a campaign runner or registration. Frozen consumer-source,
artifact/binding provenance, original regression and rubric checks remain the
controller's responsibility. State extraction belongs only to this evaluator.
"""
from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from ci_oracle import ATOL, assert_clean_ancillas, assert_prepared_state
from scale_adapter import AdapterUnavailable

TERMS = {"ZIII": .5, "IXII": -.3, "IIYI": .2, "IIIZ": .4}
ALPHA = 1.4


def dense_hamiltonian():
    """Independent little-endian .5 Z0 - .3 X1 + .2 Y2 + .4 Z3."""
    matrix = np.zeros((16, 16), complex)
    for column in range(16):
        matrix[column, column] = (.5 * (-1 if column & 1 else 1) + .4 *
                                  (-1 if column & 8 else 1))
        matrix[column ^ 2, column] = -.3
        matrix[column ^ 4, column] = -.2j if column & 4 else .2j
    return matrix


def assess_outputs(outputs, expected):
    """Check actual preparation, clean-ancilla identities and one-step block.

    One-step Walk shares preparation's phase, catching a consumer sign change.
    Zero-step and roundtrip identities each permit their own global phase.
    The one-step complement is norm-checked, not a full Walk-unitary comparison.
    """
    expected = np.asarray(expected, dtype=complex)
    if expected.shape != (16, ):
        raise ValueError("this native probe requires four system qubits")
    metrics = {
        "preparation": assert_prepared_state(outputs["preparation"], expected)
    }
    for name in ("zero_step", "roundtrip"):
        metrics[name] = assert_clean_ancillas(outputs[name],
                                              expected,
                                              num_ancilla=2)
    one_step = np.asarray(outputs["one_step"], dtype=complex)
    assert one_step.shape == (64, ), "wrong one-step Walk register layout"
    assert np.isfinite(one_step).all(), "nonfinite Walk output"
    assert abs(np.linalg.norm(one_step) -
               1) <= ATOL, "Walk output is not normalized"
    overlap = np.vdot(expected, outputs["preparation"])
    phase = overlap / abs(overlap)
    expected_block = -phase * dense_hamiltonian() @ expected / ALPHA
    error = float(np.max(np.abs(one_step[:16] - expected_block)))
    assert error <= ATOL, "incorrect coherent one-step Walk block"
    metrics["walk_block_max_error"] = error
    return metrics


@contextmanager
def _without_extraction():
    """Catch ordinary SDK extraction dependencies, not adversarial evasion."""
    import cudaq
    from cudaq.runtime import state

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not require state extraction")

    saved = [(module, name, getattr(module, name)) for module in (cudaq, state)
             for name in ("get_state", "get_state_async")]
    try:
        for module, name, _ in saved:
            setattr(module, name, forbidden)
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


def _inspect_preparation(preparation):
    """Inspect actual Quake operations, including compiled captured kernels.

    This first binding supports system-only unitaries. Additional internal
    allocation needs a declared layout and is UNKNOWN, not an E05 failure:
    the brief allows explicitly defined ancilla and uncomputation behavior.
    """
    # qkeModule ensures lazy compilation and preserves already-compiled kernels
    # which legitimately have no Python AST (e.g. mk_decorator(builder)).
    module = preparation.qkeModule

    def operations(operation):
        yield operation.name
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    yield from operations(child.operation)

    names = set(operations(module.operation))
    # In CUDA-Q 0.15.1 captured decorated kernels are runtime arguments, not
    # necessarily inlined in the caller module. Inspect their actual resolved
    # modules too; a helper must not hide measurement/reset or allocation.
    from cudaq.kernel.kernel_decorator import DecoratorCapture, LinkedKernelCapture
    pending = list(preparation.resolve_captured_arguments())
    while pending:
        captured = pending.pop()
        if isinstance(captured, DecoratorCapture):
            names.update(operations(captured.decorator.qkeModule.operation))
            pending.extend(captured.resolved)
        elif isinstance(captured, LinkedKernelCapture):
            names.update(operations(captured.qkeModule.operation))
    assert not names.intersection({"quake.init_state", "quake.mx", "quake.my",
                                   "quake.mz", "quake.reset", "quake.apply_noise"}), \
        "preparation must be a data-free unitary"
    if "quake.alloca" in names:
        raise AdapterUnavailable(
            "internal allocation requires an explicit ancilla-layout binding")


def probe_preparation(invoke, case_data, expected):
    """Execute a mapped four-system-qubit factory in an isolated runtime only.

    Sampling has no state arguments and runs with extraction disabled. Dense
    state extraction is evaluator-only, after construction/sampling. The SDK
    guard and Quake inspection demonstrate this bounded compilation path,
    not universal hardware portability or an adversarial security boundary.
    """
    import cudaq
    from cudaq_algorithms import PauliLCU, Walk

    expected = np.array(expected, dtype=complex, copy=True)
    assert_prepared_state(expected, expected)
    if expected.shape != (16, ):
        raise ValueError("this native probe requires four system qubits")
    with _without_extraction():
        preparation = invoke(deepcopy(case_data))
        _inspect_preparation(preparation)

        @cudaq.kernel
        def entry():
            system = cudaq.qvector(4)
            preparation(system)

        encoding = PauliLCU(TERMS)
        assert encoding.num_system == 4 and encoding.num_ancilla == 2
        assert abs(encoding.alpha - ALPHA) <= ATOL
        walk = Walk(encoding)
        kernels = {
            "preparation":
            entry,
            "zero_step":
            walk.kernel(power=0, uncompute=True, state_prep=preparation),
            "roundtrip":
            walk.roundtrip_kernel(power=2, state_prep=preparation),
            "one_step":
            walk.kernel(power=1, uncompute=True, state_prep=preparation)
        }
        counts = {
            name: sum(cudaq.sample(kernel, shots_count=32).values())
            for name, kernel in kernels.items()
        }
        assert all(count == 32
                   for count in counts.values()), "incomplete sampling"
    outputs = {
        name: np.asarray(cudaq.get_state(kernel), dtype=complex)
        for name, kernel in kernels.items()
    }
    metrics = assess_outputs(outputs, expected)
    metrics["sample_counts"] = counts
    return metrics
