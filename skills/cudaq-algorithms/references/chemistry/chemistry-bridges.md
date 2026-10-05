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

## Workflow

Start with geometry and its units, basis/ECP, charge, electron count and spin
convention. For a provider calculation, converge the restricted Hartree–Fock reference
and retain its MO coefficients, occupations and orbital ordering before calling
the loader. For FCIDUMP, retain electron/spin and orbital metadata separately:
the returned triple alone does not identify the physical sector.

Choose full-space or frozen-core/active-space physics before spin expansion.
The loaders do not make this choice. In the real orthonormal spatial basis,
freeze only doubly occupied orbitals; every discarded noncore orbital is assumed
empty. The following caller-owned preprocessing preserves the order of `active`.
Inputs are the loaded `h`, `eri`, `offset` and disjoint zero-based spatial-index
lists `core`, `active`; full space uses an empty core and all spatial indices.

```python
import numpy as np
from cudaq_algorithms import chemistry

def frozen_core_integrals(h, eri, offset, core, active):
    h, eri = np.asarray(h), np.asarray(eri)
    core, active = list(core), list(active)
    indices = core + active
    assert len(set(indices)) == len(indices)
    assert all(0 <= i < len(h) for i in indices)
    effective_h = h.copy()
    for i in core:
        effective_h += 2 * eri[:, :, i, i] - eri[:, i, i, :]
    core_energy = offset + 2 * sum(h[i, i] for i in core)
    core_energy += sum(2 * eri[i, i, j, j] - eri[i, j, j, i]
                       for i in core for j in core)
    return (effective_h[np.ix_(active, active)],
            eri[np.ix_(active, active, active, active)], core_energy)

h_active, eri_active, active_offset = frozen_core_integrals(
    h, eri, offset, core, active)
# prune_tol is the caller's coefficient-pruning budget.
hamiltonian = chemistry.qubit_hamiltonian(
    h_active, eri_active, scalar_offset=active_offset, tolerance=prune_tol)
```

Subtract `len(core)` from each spin's electron count. Remap occupied orbitals
into the active list: spin orbitals are interleaved (`2*p` up, `2*p+1` down).
Retain the intended width `2*len(active)` even when operator pruning reduces its
support. Prepare that determinant with the
[state-preparation contracts](../state-preparation/state-preparation.md) when
quantum execution is needed; a Hamiltonian or classical energy request can stop
at host calculations. An energy estimator using a shifted/scaled Hamiltonian
must undo that transformation and count `active_offset` exactly once.

For geometry comparisons, recompute SCF, integrals and scalar contributions at
each geometry. Define a consistent active-space selection rule and track orbital
character across crossings; reusing an orbital index is not evidence that it
represents the same orbital. Report active-space energies as such, separately
from full-space FCI, and assess the reference determinant's quality along the
curve rather than assuming a single-reference description remains adequate.

## Verification

First check MO orthonormality in the AO overlap metric, SCF convergence, tensor
dimensions, Hermiticity and chemist symmetries. A useful independent checkpoint
is the determinant energy from spatial Slater–Condon terms versus the compiled
Pauli operator. Here `occupied` contains distinct active spin-orbital indices,
and `atol` is the chosen absolute energy-check tolerance, including pruning.

```python
spins = [divmod(i, 2) for i in occupied]
det_energy = active_offset + sum(h_active[p, p] for p, spin in spins)
det_energy += 0.5 * sum(
    eri_active[p, p, q, q] - (spin == other) * eri_active[p, q, q, p]
    for p, spin in spins for q, other in spins)
bits = sum(1 << i for i in occupied)
qubit_energy = 0j
for term in hamiltonian:
    word = term.get_pauli_word(2 * len(active))
    if "X" not in word and "Y" not in word:
        parity = sum((bits >> q) & 1 for q, pauli in enumerate(word)
                     if pauli == "Z")
        qubit_energy += complex(term.evaluate_coefficient()) * (-1)**parity
np.testing.assert_allclose(qubit_energy, det_energy, atol=atol, rtol=0)
```

For the HF determinant this should reproduce the provider's total HF energy
when frozen/empty selections preserve its occupations. For a small correlated
reference, use an independent FCI implementation with the same active integrals,
scalar and `(N_up, N_down)`, or occupation-space ladder algebra. Check the
sector before diagonalization: the unconstrained Fock-space minimum can belong
to a different molecule/charge. An `N_up,N_down` sector fixes spin projection,
not total spin; select/check total spin when the research target requires it.
Validate frozen-core reduction against projection of a small unreduced model,
not merely a second invocation of the reduction formula.

The molecule-to-energy example (`docs/sphinx/examples/python/03_chemistry_to_ground_state.py`)
shows provider-to-qubit composition and an FCI comparison;
FCIDUMP tests (`tests/python/test_fcidump.py`) pin integral ordering
and offsets. The [fermion checkpoint](../fermion-transforms/fermion-transforms.md#verification)
provides independent ladder algebra for small matrices.

## Provenance

Source: `python/cudaq_algorithms/chemistry.py`. Tests include
`test_fcidump.py`, `test_psi4_conversion.py`, and `test_df_qsvt_bridge.py`.
Current public source and tests are authoritative and must be rechecked at use
time. [Source lookup](../source-provenance.md) gives shared current-source paths.
