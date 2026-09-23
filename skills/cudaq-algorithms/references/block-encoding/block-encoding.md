# Block encoding — family front door

Family operation + object: **encode** an **operator as a
zero-flagged block of a unitary**.

This front door owns the shared representation and capability records. The
packaged concrete primitive is [PauliLCU](pauli-lcu.md).

## Representation record — `BlockEncoding`

- Canonical symbol: `cudaq_algorithms.BlockEncoding`, defined in
  `python/cudaq_algorithms/block_encoding.py`.
- Form: `@runtime_checkable typing.Protocol`; conformance is structural and
  does not prove signatures or scientific semantics.
- Meaning:

  ```text
  (<0|_anc x I) U_A (|0>_anc x I) = H / alpha
  ```

- Geometry: `num_system` system qubits, `num_ancilla >= 1` signal qubits, and
  scalar normalization `alpha`.
- Register order in packaged consumer kernels: system is allocated first, then
  ancilla/signal. Protocol method argument order remains `(ancilla, system)`.

The twelve protocol members are:

| Member | Returned form or meaning |
| --- | --- |
| `num_system`, `num_ancilla`, `alpha` | readable attributes |
| `prepare_kernel()` | `(ancilla: qview)` |
| `unprepare_kernel()` | `(ancilla: qview)` |
| `apply_kernel()` | `(ancilla: qview, system: qview)` |
| `controlled_apply_kernel()` | `(control_and_ancilla: qview, system: qview)` |
| `walk_step_kernel()` | `(ancilla: qview, system: qview)`; block is `-H/alpha` |
| `adjoint_walk_step_kernel()` | `(ancilla: qview, system: qview)` |
| `controlled_walk_step_kernel()` | `(control_and_ancilla: qview, system: qview)` |
| `controlled_adjoint_walk_step_kernel()` | same combined-control layout |
| `select_observable()` | odd-moment observable |

All encoding data is captured when a factory is called; emitted kernels carry
registers only. Consumers mint and cache each factory once, so treat an
injected encoding as immutable.

Known boundaries:

- `isinstance(obj, BlockEncoding)` checks member presence only.
- `select_observable` is optional in practice: example encodings can satisfy
  the runtime protocol yet raise `NotImplementedError`; only odd
  `Walk.moment` calls require it.
- `state_prep`, `encode_kernel`, and `walk_kernel` are not protocol members.
- The protocol represents only the all-zero flagged subspace.
- Controlled methods place the external control at qubit 0 of a combined
  `[control, ancilla...]` register.

Producers include packaged `PauliLCU`, the example-only
`DoubleFactorizedEncoding`, teaching/test encodings, and caller implementations.
Consumers include `Walk`, `QSVT`, `reflection_observable`, and some
simulation-only helpers; [simulation action](../simulation/simulation-action.md) is narrower
because it calls `encode_kernel`, which the protocol does not declare.

## Capability record — zero-flagged block access

- Stable ID: `cudaq-algorithms.block-encoding.zero-flagged.v1`.
- Contract type: backed by the source-level `BlockEncoding` protocol; the ID
  itself is a documentation identifier, not a public symbol.
- Owner: this family record.
- Providers/consumers: the representation record above.
- Required invariants: the `H/alpha` zero block, `num_ancilla >= 1`, fixed
  geometry, fixed kernel signatures, the walk-step `-H/alpha` sign, cached
  factory immutability, and the combined-control convention.
- Execution boundary: construction/validation on the host, returned kernels on
  the device; statevector extraction remains simulation-only.
- Unsupported/unverified: non-zero flag states are unsupported; member
  semantics cannot be inferred from structural conformance; hardware execution
  and foreign helper compatibility are unverified unless separately tested.

## Provenance and validation

- Source provenance: current public source and tests are authoritative and must
  be rechecked at use time. [Source lookup](../source-provenance.md) gives shared current-source paths.
- Contract source: `python/cudaq_algorithms/block_encoding.py`.
- Authoritative tests: `test_block_encoding_protocol.py`,
  `test_walk_qsvt_orchestration.py`, `test_df_encoding.py`.
- Independent checks: dense flagged-block extraction, controlled identity at
  control `|0>`, foreign structural provider parity, and explicit failure of
  odd moments when `select_observable` is absent.
- Verification policy: recheck current public source and tests at use time;
  this record was not freshly executed.

## Record routing

| Need | Read |
| --- | --- |
| Construct the packaged Pauli-sum encoding | [pauli-lcu.md](pauli-lcu.md) |
| Apply walk kernels | [qubitization-walk.md](../qubitization/qubitization-walk.md) |
| Measure Chebyshev moments | [qubitization-moments.md](../qubitization/qubitization-moments.md) |
| Apply a polynomial transform | [qsvt-sequence.md](../qsvt/qsvt-sequence.md) |
| Inspect simulated flagged blocks | [simulation analysis](../simulation/simulation-analysis.md) |

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| encode / Pauli-sum operator as a zero-flagged unitary block | `PauliLCU` | quantum operation; host construction + kernel factory/device kernel | provides `cudaq-algorithms.block-encoding.zero-flagged.v1`; optionally requires `cudaq-algorithms.state-preparation.unitary.v1` | [Pauli LCU](pauli-lcu.md) |
