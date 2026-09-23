# Chemistry bridges — family front door

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| load / FCIDUMP text to chemist spatial integral triple | `chemistry.from_fcidump` | classical transformation; host parser; NumPy only | provides `cudaq-algorithms.chemistry-integrals.v1` | [FCIDUMP loader](chemistry-from-fcidump.md) |
| load / restricted PySCF mean field to chemist spatial integral triple | `chemistry.from_pyscf` | classical transformation; host/provider bridge; PySCF at call time | provides `cudaq-algorithms.chemistry-integrals.v1` | [PySCF loader](chemistry-from-pyscf.md) |
| load / restricted C1 Psi4 wavefunction to chemist spatial integral triple | `chemistry.from_psi4` | classical transformation; host/provider bridge; Psi4 at call time | provides `cudaq-algorithms.chemistry-integrals.v1` | [Psi4 loader](chemistry-from-psi4.md) |
| transform / spatial integrals to spin-orbital tensors | `chemistry.spin_orbital_tensors` | classical transformation; host | requires `cudaq-algorithms.chemistry-integrals.v1` | [spin expansion](chemistry-spin-orbital-tensors.md) |
| transform / spatial integrals to Jordan–Wigner qubit Hamiltonian | `chemistry.qubit_hamiltonian` | classical driver; host | requires `cudaq-algorithms.chemistry-integrals.v1` | [qubit Hamiltonian bridge](chemistry-qubit-hamiltonian.md) |

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

Select an integral loader by its concrete input object. All three providers
return this shared representation, but their inputs, dependencies, and
rejection behavior are not interchangeable: `from_fcidump` accepts FCIDUMP
text already read by the caller and needs only NumPy; `from_pyscf` accepts a
restricted PySCF mean-field object and needs PySCF at call time; `from_psi4`
accepts a restricted, C1 Psi4 wavefunction and needs Psi4 at call time. These
provider calls perform host preprocessing only. They do not spin-expand,
choose a fermion transform, build a `cudaq.SpinOperator`, run double
factorization, prepare a state, or submit quantum work.

## Provenance

Source: `python/cudaq_algorithms/chemistry.py`. Tests include
`test_fcidump.py`, `test_psi4_conversion.py`, and `test_df_qsvt_bridge.py`.
Current public source and tests are authoritative and must be rechecked at use
time. [Source lookup](../source-provenance.md) gives shared current-source paths.
