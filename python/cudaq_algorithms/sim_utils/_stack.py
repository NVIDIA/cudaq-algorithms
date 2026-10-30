# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an executor on a worker thread with a large stack."""

from __future__ import annotations

__all__ = ["LARGE_STACK_BYTES", "run_on_large_stack"]

LARGE_STACK_BYTES = 512 << 20
"""Worker-thread stack size used by `run_on_large_stack` (512 MiB, reserved
lazily, so the cost is address space rather than memory)."""


def run_on_large_stack(function,
                       /,
                       *args,
                       stack_bytes=LARGE_STACK_BYTES,
                       **kwargs):
    """Call ``function(*args, **kwargs)`` on a worker thread with a
    ``stack_bytes`` stack and return its result (exceptions propagate).

    A simulated kernel runs on the calling thread's stack. When a kernel
    applies a sub-kernel that captures Python lists inside a loop, the
    compiler re-materializes those lists on the stack on every iteration
    and releases them only when the kernel returns, so the stack in use
    grows with the trip count: about 1 kB per controlled walk step for a
    `PauliLCU`, which exceeds the default 8 MiB thread stack after a few
    thousand steps and ends the process with a segmentation fault. Running
    the executor on a thread with a large stack keeps such circuits within
    bounds; `phase_probabilities` does this for the ``2^num_bits - 1`` walk
    steps of a phase-estimation circuit, and a circuit from
    `PhaseEstimation.kernel` that you sample on a simulator needs the
    same::

        counts = sim_utils.run_on_large_stack(cudaq.sample, circuit,
                                              shots_count=1000)

    If the platform refuses the requested stack size the call runs on the
    current thread.
    """
    import threading

    outcome: dict = {}

    def target():
        try:
            outcome["value"] = function(*args, **kwargs)
        except BaseException as error:  # re-raised on the caller's thread
            outcome["error"] = error

    previous = threading.stack_size()
    try:
        threading.stack_size(int(stack_bytes))
    except (ValueError, RuntimeError):
        return function(*args, **kwargs)
    try:
        worker = threading.Thread(target=target,
                                  name="cudaq-algorithms-large-stack",
                                  daemon=True)
        worker.start()
    finally:
        threading.stack_size(previous)
    worker.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]
