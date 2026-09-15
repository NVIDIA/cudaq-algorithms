# Chemistry bridges — family front door

Status: draft.

| Operation + object | Record |
| --- | --- |
| load / chemist-notation spatial integral triple | [chemistry-integral-loaders.md](chemistry-integral-loaders.md) |
| transform / spatial integrals to spin-orbital coefficient tensors | [chemistry-spin-orbital-tensors.md](chemistry-spin-orbital-tensors.md) |
| transform / spatial integrals to a qubit Hamiltonian | [chemistry-qubit-hamiltonian.md](chemistry-qubit-hamiltonian.md) |

## Shared representation — chemist integral triple

The exchanged object is `(one_body, eri, scalar_offset)`:

- `one_body[p,q]`: square spatial-orbital core Hamiltonian;
- `eri[p,q,r,s] = (pq|rs)`: dense chemist-notation spatial tensor;
- scalar core/nuclear-repulsion energy.

The real-orbital path expects the eightfold chemist permutation symmetry. The
three loaders produce this representation; spin expansion, qubit conversion,
and double factorization consume it.

## Capability record — chemistry integral source

- Stable ID: `cudaq-algorithms.chemistry-integrals.v1`.
- Status: provisional.
- Contract type: documentation-only.
- Owner: this family record.
- Boundary: the chemist integral triple above.
- Providers: `from_fcidump`, `from_pyscf`, `from_psi4`.
- Consumers: `spin_orbital_tensors`, `qubit_hamiltonian`, explicit and
  compressed double factorization.
- Invariants: spatial-orbital dimensions agree; ordering is chemist notation;
  scalar offset remains separate; restricted real-orbital assumptions are
  stated per provider.
- Unsupported/unverified: unrestricted loader inputs are unsupported; complex
  integral symmetry and external-package version ranges are unverified.

The identifier is final for this documentation version even though the
capability status remains provisional.

## Provenance

Source: `python/cudaq_algorithms/chemistry.py`. Tests include
`test_fcidump.py`, `test_psi4_conversion.py`, and `test_df_qsvt_bridge.py`.
Current public source and tests are authoritative and must be rechecked at use
time. [source-provenance.md](../source-provenance.md) records historical
last-review audit context. These records were not freshly executed.
