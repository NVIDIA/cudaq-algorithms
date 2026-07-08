# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The block-encoding protocol consumed by ``Walk`` and ``QSVT``.

``Walk`` and ``QSVT`` are generic over the encoding: they receive an
encoding object and delegate every encoding-specific circuit to it, keeping
only sequencing, control conventions, and measurement for themselves. Any
object satisfying this protocol works — conformance is structural
(``typing.Protocol``), so implementations do not inherit from anything.

The contract that makes the composition work is *data erasure at the kernel
boundary*: every factory returns a ``@cudaq.kernel`` whose signature is
fixed by this protocol (registers only), with all encoding-specific data
already captured inside the kernel at factory time. The consumers can then
call the injected kernels without knowing anything about the encoding's
internals.

Register conventions shared by all implementations:

- ``encode_kernel``-produced kernels allocate the system register from a
  ``cudaq.State`` first, ancillas after it (so the good subspace is the
  first ``2**num_system`` amplitudes; see ``sim_utils.good_subspace``).
- Controlled variants take a combined ``[control, ancilla...]`` register
  whose qubit 0 is the external control (a CUDA-Q Python control set
  cannot mix a bare qubit with a separate register); with control |0>
  they must reduce to the identity.
- The flagged block of the *walk step* is ``-H/alpha`` (the sign folded
  into the walk construction); ``Walk.moment`` and the QSVT response
  conventions rely on this.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# A compiled ``@cudaq.kernel``; CUDA-Q exposes no stable public Python type.
Kernel = Any


@runtime_checkable
class BlockEncoding(Protocol):
    """A zero-flagged block encoding ``U_A`` with ``<0|_anc U_A |0>_anc = H / alpha``.

    ``PauliLCU`` is the provided implementation; double-factorized or
    sparse-oracle encodings plug in by satisfying the same surface.
    """

    @property
    def num_system(self) -> int:
        """Number of system qubits the encoded operator acts on."""
        ...

    @property
    def num_ancilla(self) -> int:
        """Number of ancilla (signal) qubits flagging the encoded block."""
        ...

    @property
    def alpha(self) -> float:
        """The block-encoding normalization: the encoded block is H / alpha."""
        ...

    # ------------------------------------------------------------------
    # Kernel factories. Signatures of the returned kernels are fixed;
    # all encoding data is captured inside at factory time.
    # ------------------------------------------------------------------

    def prepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: PREPARE the ancilla superposition."""
        ...

    def unprepare_kernel(self) -> Kernel:
        """``(ancilla: qview)``: PREPARE dagger."""
        ...

    def apply_kernel(self) -> Kernel:
        """``(ancilla: qview, system: qview)``: the full block encoding U_A."""
        ...

    def controlled_apply_kernel(self) -> Kernel:
        """``(control_and_ancilla: qview, system: qview)``: U_A controlled
        by qubit 0 of the combined register."""
        ...

    def walk_step_kernel(self) -> Kernel:
        """``(ancilla: qview, system: qview)``: one qubitization walk step W
        (block encodes ``-H/alpha``)."""
        ...

    def adjoint_walk_step_kernel(self) -> Kernel:
        """``(ancilla: qview, system: qview)``: one adjoint walk step W†."""
        ...

    def controlled_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla: qview, system: qview)``: controlled W."""
        ...

    def controlled_adjoint_walk_step_kernel(self) -> Kernel:
        """``(control_and_ancilla: qview, system: qview)``: controlled W†."""
        ...

    # ------------------------------------------------------------------
    # Observable hooks
    # ------------------------------------------------------------------

    def select_observable(self) -> Any:
        """The odd-moment observable as a ``cudaq.SpinOperator``.

        Measured after PREPARE and ``p`` walk steps (no UNPREPARE), its
        expectation is the odd Chebyshev moment ``<T_{2p+1}(H/alpha)>``.
        The construction is encoding-specific (for an LCU it is
        ``sum_i sign_i |i><i|_anc x P_i``); the even-moment reflection
        observable ``2|0..0><0..0| - I`` needs only the register geometry,
        so ``Walk`` derives it without an encoding hook.
        """
        ...
