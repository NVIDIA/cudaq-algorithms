# Chemistry integral loaders — family front door

Status: draft. This file routes among independently selectable providers; it
does not define a combined loader contract.

| Input source | Provider record | Dependency at call time |
| --- | --- | --- |
| FCIDUMP text already read by the caller | [`chemistry.from_fcidump`](chemistry-from-fcidump.md) | NumPy only |
| Restricted PySCF mean-field object | [`chemistry.from_pyscf`](chemistry-from-pyscf.md) | PySCF |
| Restricted, C1 Psi4 wavefunction | [`chemistry.from_psi4`](chemistry-from-psi4.md) | Psi4 |

All three providers return the chemist-notation spatial integral triple owned
by [chemistry-bridges.md](chemistry-bridges.md) and provide
`cudaq-algorithms.chemistry-integrals.v1`. Select the record by the concrete
input object; the shared output does not make the provider inputs,
dependencies, or rejection behavior interchangeable.

Provider calls perform host preprocessing only. They do not spin-expand,
choose a fermion transform, build a `cudaq.SpinOperator`, run double
factorization, prepare a state, or submit quantum work.

Current public source and tests are authoritative and must be rechecked at use
time. [source-provenance.md](../source-provenance.md) records historical
last-review audit context.
