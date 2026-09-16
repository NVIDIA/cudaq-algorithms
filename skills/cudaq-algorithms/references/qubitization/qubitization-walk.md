# Qubitization walk kernels

Status: draft. Operation + object: **evolve** a **quantum state by a
qubitization walk**.

## Identity and classification

- Public symbol: `cudaq_algorithms.Walk`.
- Source: `python/cudaq_algorithms/qubitization.py`.
- Tests: `test_qubitization.py`, `test_walk_qsvt_orchestration.py`.
- Source provenance: current public source and tests are authoritative and must
  be rechecked at use time. [source-provenance.md](../source-provenance.md)
  records historical last-review audit context.
- Kind/role/abstraction: quantum operation, computational, composite protocol.
- Parameterization: encoding at construction; `power`, direction, control, and
  uncomputation at factory call.
- Layers: host orchestration, kernel factory, device kernel.
- Exactness: exact sequencing of the injected walk; deterministic.

## Scientific contract

`Walk(encoding)` sequences the encoding's PREPARE, walk-step, adjoint-step, and
controlled-step factories. One walk step block-encodes `-H/alpha`. When
`uncompute=True`, PREPARE is undone after the requested power; the flagged block
is `T_power(-H/alpha)`.

Use this when a circuit or controlled circuit is needed. Use
[qubitization-moments.md](qubitization-moments.md) when the desired output is a
classical Chebyshev expectation.

## Inputs and outputs

- Constructor: `Walk(encoding: BlockEncoding)`; rejects `num_ancilla == 0`.
- `kernel(power=1, uncompute=True, state_prep=None)`.
- `adjoint_kernel(...)`.
- `roundtrip_kernel(power=1, state_prep=None)`.
- `controlled_kernel(power=1, control_state=1, uncompute=True,
  state_prep=None)`.
- `controlled_roundtrip_kernel(...)`.

Without `state_prep`, emitted kernels take one `cudaq.State`. With a packaged
one-argument preparation kernel, they take no arguments: the consumer allocates
the system register first, calls preparation, then allocates ancilla or
`[control, ancilla...]`. Preparation runs once and uncontrolled.

The preparation width is required to equal `encoding.num_system`, but the
factory has no general way to verify it. Runtime behavior for a mismatch is
provider/consumer-dependent and unverified; do not promise a launch failure.

`power` must be a non-negative integer and `control_state` must be 0 or 1. The
controlled register's qubit 0 is the external control. Control `|0>` reduces to
identity up to the cancelling PREPARE pair.

## Capabilities and composition

- Requires `cudaq-algorithms.block-encoding.zero-flagged.v1`, including exact
  member signatures and walk sign.
- Optionally consumes
  `cudaq-algorithms.state-preparation.unitary.v1` through concrete factories.
- A structural `BlockEncoding` check does not prove scientific correctness or
  preparation support.
- The encoding is read-only after construction because minted kernels are
  cached.

## Accuracy, resources, and validation

There is no walk resource estimator. Exact structural quantities are
`num_system + num_ancilla` qubits (plus one for controlled kernels), one
PREPARE/UNPREPARE pair when uncomputing, and `power` walk-step calls. These are
not gate counts, depth, runtime, or memory.

Independent oracles:

- compare the flagged block with dense `T_power(-H/alpha)`;
- verify walk followed by adjoint walk returns the input;
- verify the control-0 and control-1 blocks independently;
- use a countable mock encoding to check factory call counts.

The runnable pointers are `tests/python/test_qubitization.py` and
`tests/python/test_walk_qsvt_orchestration.py`; they use dense Pauli references
and amplitude comparisons. Current public source and tests must be rechecked at
use time; this record was not freshly executed. Eval coverage includes
`qubitization-walk-moment-boundary` and
application composition cases.
