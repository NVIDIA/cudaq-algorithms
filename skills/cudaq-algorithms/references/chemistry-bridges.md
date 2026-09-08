# Chemistry input bridges and integral-tensor transformations

Status: draft. Three operation + object identities live in this one family file:

| Identity | Providers |
| --- | --- |
| **load** + **fermionic integral tensor** | `from_fcidump`, `from_pyscf`, `from_psi4` |
| **transform** + **fermionic integral tensor** | `spin_orbital_tensors` |
| **transform** + **Pauli operator** (from a fermionic integral tensor) | `qubit_hamiltonian` |

All five are host classical transformations in one module, exchanging one shared
object, so they are one family. Following
`assets/primitive-record-template.md`, every canonical Primitive-Record heading
below is instantiated **once at family level**, with per-provider material in
visibly separate `A`–`E` subsections beneath it. The two optional schemas — a
**Representation Record** for the chemist-notation integral triple and a
**Capability Record** for the integral-source boundary — are separate sections
at the end, so the three record types stay visibly distinct.

`domain: quantum-chemistry` is a **tag on these records, not a parallel
taxonomy** (`architecture.md`). The objects they emit — fermionic tensors and a
`cudaq.SpinOperator` — are consumed by domain-independent primitives.

Provider labels used throughout:

- **A** `from_fcidump` — FCIDUMP text -> chemist-notation triple
- **B** `from_pyscf` — converged restricted PySCF mean field -> triple
- **C** `from_psi4` — converged restricted Psi4 wavefunction -> triple
- **D** `spin_orbital_tensors` — spatial chemist tensors -> spin-orbital
  fermionic tensors
- **E** `qubit_hamiltonian` — spatial chemist tensors -> `cudaq.SpinOperator`

Cross-cutting layout, ordering, and validation-placement rules are
`conventions.md` and are not repeated here.

## Identity and provenance

- Owner: CUDA-Q Algorithms Team.
- Public symbols and import paths: `from_fcidump`, `spin_orbital_tensors`,
  `qubit_hamiltonian`, `from_pyscf`, `from_psi4` — the module `__all__`
  (`chemistry.py:25-28`). Reach them through the submodule, e.g.
  `from cudaq_algorithms import chemistry` then `chemistry.qubit_hamiltonian`,
  or `cudaq_algorithms.chemistry.from_pyscf`. The submodule is imported at
  package root (`python/cudaq_algorithms/__init__.py:33-35`), but **none of the
  five functions is exported at the package root**, so `from cudaq_algorithms
  import qubit_hamiltonian` is not a valid import.
- Source paths: `python/cudaq_algorithms/chemistry.py` (the whole module);
  `python/cudaq_algorithms/fermion/_compilers.py` for `jordan_wigner`, which
  **E** calls and whose contract is owned by
  [fermion-transforms.md](fermion-transforms.md).
- Authoritative tests: `tests/python/test_fcidump.py` (A, and A -> E);
  `tests/python/test_psi4_conversion.py` (B, C, and both -> E);
  `tests/python/test_df_qsvt_bridge.py` (D, E, and E -> `PauliLCU`/`Walk`).
- Authoritative documentation: `docs/sphinx/guide/preprocessing.rst`
  (integral sources; the chemistry bridge; feeding a compressed tensor to the
  transform); `docs/sphinx/conventions.rst` (qubit ordering, Pauli words,
  interleaved spin orbitals, fermionic integral tensors, Jordan-Wigner ladder
  operators, Hermiticity); `docs/sphinx/api/python_api.rst:50-51`
  (`automodule cudaq_algorithms.chemistry`). Runnable examples:
  `docs/sphinx/examples/python/03_chemistry_to_ground_state.py`,
  `04_double_factorization_and_the_protocol.py`,
  `08_quantum_phase_estimation.py`, `df_block_encoding.py`,
  `df_compression_to_qsvt.py`.
- Package/CUDA-Q versions verified: **unverified.** The declared environment is
  Python `>=3.11` with `cudaq >= 0.15.0, < 0.16` (`pyproject.toml`) and CUDA-Q
  pinned at `b28cf0f6f2d9387f12ca81b6824b8833a38530af` (`.cudaq_version`). No
  test was executed for this family, and **no version floor exists anywhere in
  the repository for PySCF or Psi4** — `pyproject.toml` declares no optional
  extras.
- Commit/date last verified: `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`
  (2026-09-03).
- Lifecycle: draft.
- Replacement and migration notes: nothing in `chemistry.py` is marked
  deprecated at the cited commit. One inherited migration note matters when a
  caller reaches past **E**: `fermion.bravyi_kitaev` no longer antisymmetrizes a
  two-body tensor internally, so a caller who feeds **D**'s output to it gets a
  literal, entry-by-entry compilation (`_compilers.py:339-347`), and invalid
  shapes raise `ValueError` where the retired compiled binding raised
  `RuntimeError` (`_compilers.py:46-52`).
- **Evidence rule for this family.** Every test named in this file is *cited
  repository evidence* — a committed assertion read at the cited commit, labeled
  `derived` — never a fresh measurement. Nothing here was executed.

## Classification

- Operation + mathematical object (primary identity): the three identities
  tabulated at the top of this file. Routing is on that pair; every dimension
  below is orthogonal metadata.
- Kind: **classical transformation / preprocessing** for all five. No provider
  emits a device kernel, allocates a register, or measures anything. **E**
  constructs a `cudaq.SpinOperator` on the host; that is an operator object, not
  a circuit.
- Routine role: **computational** for A–D — each performs one distinct,
  independently usable task, and running on the host does not make it auxiliary
  (`architecture.md` names chemistry input bridges explicitly). **E** is a
  **driver**: it solves the complete user problem "chemist integrals to qubit
  Hamiltonian" by sequencing **D** and `fermion.jordan_wigner`.
- Abstraction level: leaf operation for A–D; **composite protocol** for **E**
  (see "Composite protocol").
- Parameterization: A–C take a single positional input and no controls; **D**
  and **E** take keyword controls evaluated at call time (`validate_symmetry`;
  plus `scalar_offset`, `tolerance` for **E**). The template's
  `none | construction-time | runtime` vocabulary was written for kernel
  factories; recorded here as **construction-time**, meaning all controls are
  fixed when the host function is called, with **no runtime device parameter
  anywhere in this family**. Flagging the stretch rather than silently
  reinterpreting the vocabulary.
- Execution layers: **host preprocessing only**, for all five. No kernel
  factory, no device kernel, no observable, no simulation-only dependency.
- Input representations: FCIDUMP file text (A); a PySCF restricted mean-field
  object (B); a Psi4 restricted wavefunction object (C); the chemist-notation
  spatial integral triple of the Representation Record (D, E).
- Output representations: the chemist-notation spatial integral triple (A, B,
  C); spin-orbital fermionic tensors `(one_body_so, two_body_so)` in the
  library's `adag adag a a` coefficient convention (D); `cudaq.SpinOperator`
  (E).
- Domain: `quantum-chemistry` for all five — a tag. The emitted objects carry no
  chemistry-specific interface, which is why domain-independent consumers accept
  them unchanged.
- Required dependencies: `numpy` for all five. A–D **never import `cudaq`**;
  **E** needs `cudaq` transitively, because `fermion._compilers` imports it at
  module import (`_compilers.py:68`). `scipy` is a package-level dependency
  (`pyproject.toml`) but is not used by this module.
- Optional dependencies:
  - **B** requires `pyscf` (`from pyscf import ao2mo` at `chemistry.py:143`,
    inside the function body).
  - **C** requires `psi4` (`import psi4` at `chemistry.py:187`, inside the
    function body).
  - Both imports are function-local, so `import cudaq_algorithms.chemistry`
    never raises for a missing electronic-structure package; the `ImportError`
    surfaces on the call. The module docstring states this pattern explicitly
    for the `fermion` import of **E** (`chemistry.py:11-16`); for `pyscf` and
    `psi4` it is `derived` from the import placement rather than documented
    prose.
  - Neither is declared as an extra in `pyproject.toml`. CI installs `pyscf`
    (`.github/workflows/lib_algorithms.yaml:119`, `scripts/ci/test_wheels.sh:23`)
    and **installs no `psi4` anywhere**, so the only test that exercises **C**
    skips in CI; `test_psi4_conversion.py:19-21` records that Psi4 ships through
    conda-forge with no PyPI wheel.

Provisional metadata — record the value, do not route on it:

- Exactness: **exact** for A–D. A is exact with respect to the digits the
  FCIDUMP text actually carries; B and C are exact given the converged
  mean-field object the caller supplies (they perform no SCF of their own). **E**
  is exact apart from `tolerance`, which prunes negligible terms; that is a
  screening threshold, not a controlled approximation with a convergence
  parameter.
- Uncertainty: deterministic for all five. B and C inherit whatever convergence
  the caller's SCF reached, which is an input-quality question, not randomness.
- Method: direct.

## Scientific contract

Shared mathematical frame, from `docs/sphinx/conventions.rst:46-65`: the library's
fermionic tensors are coefficient arrays for normal-ordered ladder products with
**no implicit symmetry factors**,

```text
H = c * I + sum_pq h_pq adag_p a_q + sum_pqrs V_pqrs adag_p adag_q a_r a_s
```

with `c` supplied as `scalar_offset`. Any `1/2` from a chemist-notation source
must already be folded into `V` — **D** is the routine that folds it.

### A. `from_fcidump`

- Purpose: parse the *text* of an FCIDUMP file into the chemist-notation
  spatial-integral triple, with no file access of its own
  (`chemistry.py:225-252`).
- Mathematical definition: reconstruct `h_ij` and the dense
  `eri[p, q, r, s] == (pq|rs)` tensor from the symmetry-unique records the format
  stores, filling all eight index partners of the real-orbital chemist orbit
  `{(pq|rs), (qp|rs), (pq|sr), (qp|sr), (rs|pq), (sr|pq), (rs|qp), (sr|qp)}`
  (`chemistry.py:217-222`), plus the scalar core energy.
- Why and when to use: the interchange path — integrals produced by Molpro,
  PySCF, Psi4, or a DMRG code and handed over as a file, with no
  electronic-structure package installed (`preprocessing.rst:37-63`).
- When not to use: when a live mean-field object is in hand (use **B** or
  **C**, which need no text round-trip and no printed-precision loss); when the
  file is an unrestricted variant (rejected — see "Inputs").
- Approximation controls: none. The parse introduces no approximation; the
  values carry the precision the file printed.

### B. `from_pyscf`

- Purpose: extract chemist-notation molecular-orbital integrals plus the
  nuclear-repulsion energy from a converged **restricted** PySCF mean field
  (`chemistry.py:128-167`).
- Mathematical definition: with MO coefficients `C = mean_field.mo_coeff`,
  `one_body = C^T H_core C` where `H_core = mean_field.get_hcore()` is the AO
  core Hamiltonian the mean field actually used, and
  `eri = ao2mo.restore(1, ao2mo.full(mol, C), n)` is the dense chemist
  `(pq|rs)` MO tensor. The scalar is `mean_field.energy_nuc()`.
- Why and when to use: the standard live-object path; `get_hcore()` is read
  rather than rebuilt from `int1e_kin + int1e_nuc` precisely so that ECP, X2C,
  and QM/MM contributions are retained (`chemistry.py:154-159`).
- When not to use: for an unrestricted reference (rejected); when the target is
  an active space — there is no active-space parameter, so the caller folds the
  inactive orbitals themselves (see "Inputs").
- Approximation controls: none in this function. All approximation lives in the
  caller's basis set, mean-field method, and any active-space choice.

### C. `from_psi4`

- Purpose: the Psi4 counterpart of **B**, returning "identical in meaning and
  convention" tensors so either loader drives **E** unchanged
  (`chemistry.py:170-186`).
- Mathematical definition: with `C = wavefunction.Ca()`,
  `one_body = C^T H C` where `H = wavefunction.H()` is the AO core Hamiltonian
  (documented as kinetic + potential), and
  `eri = MintsHelper(wavefunction.basisset()).mo_eri(C, C, C, C)`, already in
  chemist `(pq|rs)` ordering. The scalar is
  `wavefunction.molecule().nuclear_repulsion_energy()`.
- Why and when to use: when the electronic structure was produced by Psi4.
- When not to use: for an unrestricted reference or a wavefunction computed in
  any symmetry other than C1 (both rejected); when Psi4 is unavailable — it has
  no PyPI wheel and no CI coverage in this repository.
- Approximation controls: none in this function.

### D. `spin_orbital_tensors`

- Purpose: spin-expand chemist-notation spatial integrals into the fermionic
  tensors the `fermion` transforms consume (`chemistry.py:31-52`).
- Mathematical definition. Reorder chemist `(pq|rs)` into `adag adag a a` index
  order, `reordered = eri.transpose(0, 2, 3, 1)` (`chemistry.py:74`), and set
  `coefficient = 0.5 * reordered[p, q, r, s]`. Over `2n` interleaved spin
  orbitals (`2p` alpha, `2p+1` beta):

  ```text
  one_body_so[2p,   2q  ] = h[p, q]
  one_body_so[2p+1, 2q+1] = h[p, q]

  two_body_so[2p,   2q,   2r,   2s  ] = coefficient
  two_body_so[2p+1, 2q+1, 2r+1, 2s+1] = coefficient
  two_body_so[2p,   2q+1, 2r+1, 2s  ] = coefficient
  two_body_so[2p+1, 2q,   2r,   2s+1] = coefficient
  ```

  Only these four of the sixteen spin patterns are populated — the
  spin-conserving ones (`chemistry.py:76-93`). The result is **not**
  antisymmetrized; the transforms compile tensors exactly as given
  (`conventions.rst:66-71`).
- Why and when to use: when the spin-orbital tensors themselves are wanted — to
  feed `fermion.bravyi_kitaev`, to inspect the layout, or to compare against
  external data — rather than the finished qubit operator. It is also the only
  part of this family that runs with no `cudaq` import.
- When not to use: when the qubit Hamiltonian is the goal; **E** is the
  single call. Also not for genuinely complex integrals unless
  `validate_symmetry=False` is passed deliberately, because the default symmetry
  check encodes the *real*-orbital symmetry (`chemistry.py:46-51`).
- Approximation controls: none. `validate_symmetry` is a validation switch, not
  an approximation knob.

### E. `qubit_hamiltonian`

- Purpose: the closing bridge — chemist spatial integrals to a
  `cudaq.SpinOperator` ready for `PauliLCU`, `Walk`, `QSVT`, or `Trotter`
  (`chemistry.py:96-126`; `preprocessing.rst:145-177`).
- Mathematical definition: **D**, then the Jordan-Wigner transform with
  `a_j^dag = (1/2) Z_0...Z_{j-1} (X_j - i Y_j)` and
  `a_j = (1/2) Z_0...Z_{j-1} (X_j + i Y_j)` (`conventions.rst:74-81`), with
  `scalar_offset` added as an identity term and `tolerance` pruning small
  coefficients. The transform is **hard-wired to Jordan-Wigner**
  (`chemistry.py:122`): there is no encoding selector, so Bravyi-Kitaev is not
  reachable through this function.
- Why and when to use: whenever a qubit Hamiltonian is what the downstream
  primitive needs, including after a double-factorization compression of `eri`.
- When not to use: when the spin-orbital tensors are the deliverable (**D**);
  when a non-Jordan-Wigner encoding is wanted (call **D** plus
  `fermion.bravyi_kitaev`); when the Hamiltonian must be assembled in the
  double-factorized *operator* form rather than as a flat Pauli sum — the
  existing DF encoding is example-only, as recorded in
  [double-factorization.md](double-factorization.md) and
  [block-encoding.md](block-encoding.md).
- Approximation controls: `tolerance` (default `1e-12`) only, and it is a
  magnitude screen applied both to input tensor entries and to compiled Pauli
  coefficients (`_compilers.py:270-275`, `_compilers.py:233-238`). Note the
  default differs from `fermion.jordan_wigner`'s own `1e-15`
  (`_compilers.py:313`): **E** prunes more aggressively than the transform it
  calls.

## Inputs

Shared units statement, because no function in this family converts or labels
units: the tensors are **dimensionless coefficient arrays whose energy scale is
whatever the source produced**. For all three loaders that scale is Hartree
(atomic units) — `derived` from the examples and tests, which print and compare
energies in Ha against FCI references (`03_chemistry_to_ground_state.py:115-118`;
`test_df_qsvt_bridge.py:44`). The Hartree convention itself belongs to PySCF,
Psi4, and the FCIDUMP format and is `unverified` here. Molecular geometry units
(angstrom in the tests) are set in the caller's mean-field construction and are
never seen by these functions. `scalar_offset`, `tolerance`, and the symmetry
check's tolerances are dimensionless with respect to the code and carry the same
energy scale as the integrals in physical terms.

Shared ordering and layout statement: spatial orbitals index the MO basis in the
source package's own order (no reordering happens here); spin orbitals are
interleaved `2p` alpha / `2p+1` beta; qubit 0 is the least significant bit of a
statevector index; Pauli-word position equals qubit index
(`conventions.md`, `conventions.rst:9-44`).

### A. `from_fcidump`

- Arguments: `from_fcidump(contents: str)` (`chemistry.py:225`). The **text**,
  not a path — the caller does the I/O
  (`Path("mol.fcidump").read_text()`).
- Shapes/ranks: not applicable; free-form FCIDUMP text. `NORB` in the header
  fixes `n`.
- Dtypes/domains: real values only. A Fortran `D` exponent is accepted and
  normalized (`chemistry.py:320-321`).
- Ordering/layout: FCIDUMP orbital indices are **Fortran 1-based** and are
  converted to 0-based on read. A record `value i j k l` is a two-electron
  integral when all four indices are nonzero, a one-electron integral `h_ij`
  when `k == l == 0`, and the core energy when all four are zero
  (`chemistry.py:244-247`).
- Normalization: none applied. Records are stored as printed.
- Required mathematical properties: the file must hold a real, restricted
  (RHF/ROHF-style) integral set. `h` is symmetrized on read by assigning both
  `[i, j]` and `[j, i]` (`chemistry.py:333-334`).
- **Header keys read, and ignored.** Only `NORB` and the unrestricted guard are
  consumed (`chemistry.py:292-308`). `NELEC`, `MS2`, `ORBSYM`, and `ISYM` are
  parsed into the header string and **discarded**, so the electron count and
  spin do **not** come out of this function — a downstream state preparation
  needs them supplied separately.
- Tolerated format variation, all asserted equal to the strict parse at
  `atol=1e-12` (`test_fcidump.py:110-117`, `:148-157`): lowercase `&fci` or
  `$fci`; `&END`, `$END`, or `/` terminators; missing `ORBSYM`/`ISYM`; blank
  lines; `#` and `!` comment lines in the body; irregular spacing; Fortran `D`
  and lowercase `e` exponents; and optional orbital-energy records
  `value i 0 0 0`, which are skipped rather than rejected
  (`chemistry.py:335-340`).
- Validation and rejection behavior — every failure is a `ValueError`:

| # | Rejected condition | Source | Cited test |
| --- | --- | --- | --- |
| 1 | contents do not begin with an `&FCI`/`$FCI` namelist | `:268`, `:284` | `test_fcidump.py:128` |
| 2 | header namelist never terminated | `:287` | `test_fcidump.py:196` |
| 3 | no integral records at all | `:290` | `test_fcidump.py:203` |
| 4 | unrestricted file (`IUHF=1` or `UHF=.TRUE.`) | `:298-302` | `test_fcidump.py:159`, `:165` |
| 5 | header does not specify `NORB` | `:305` | `test_fcidump.py:133` |
| 6 | `NORB` not a positive integer | `:308` | none |
| 7 | integral line without exactly 5 fields | `:316` | `test_fcidump.py:138` |
| 8 | malformed value or index token | `:324` | none |
| 9 | orbital index outside `[0, NORB]` | `:327` | `test_fcidump.py:143` |
| 10 | index pattern matching none of the four record kinds | `:344` | none |

  `IUHF=0` is restricted and must not trip guard 4 — asserted at
  `test_fcidump.py:171-176`. Rows 2 and 3 exist specifically so a truncated or
  header-only file cannot silently return `H = 0`.

### B. `from_pyscf`

- Arguments: `from_pyscf(mean_field)` (`chemistry.py:128`) — a converged
  restricted mean field, e.g. `pyscf.scf.RHF(mol).run()`.
- Shapes/ranks: `mean_field.mo_coeff` must be a single 2D matrix; `n` is
  `mo_coeff.shape[1]`.
- Dtypes/domains: whatever PySCF returns. The function applies
  `np.ascontiguousarray` with **no dtype cast** (`chemistry.py:166-167`), so a
  complex `mo_coeff` would propagate complex tensors; that path is `unverified`
  (no source or test exercises it).
- Ordering/layout: PySCF's MO ordering, unchanged. Spatial orbitals only — the
  spin expansion happens later, in **D**.
- Required mathematical properties: a restricted reference, so that one spatial
  orbital set is shared by both spins (`chemistry.py:138-139`).
- Validation and rejection behavior: exactly one check —
  `mo_coeff.ndim != 2` raises `ValueError` naming the `(2, nao, nmo)` UHF shape
  (`chemistry.py:147-152`). Not checked, and therefore caller obligations with
  no detection: SCF convergence, and **whether the reference is truly closed
  shell**. An ROHF object carries a single 2D `mo_coeff` and so passes the only
  guard, while the docstring restricts the function to restricted references —
  behavior for ROHF is `unverified`.
- No active-space parameter exists. The documented pattern is for the caller to
  fold the inactive orbitals first and pass the resulting effective integrals
  and CAS core energy in: `df_block_encoding.py:93-105` uses
  `mcscf.CASCI(...).get_h1eff()` and `ao2mo.restore(1, cas.get_h2eff(), ncas)`.
  Frozen-core selection, orbital localization, and symmetry-adapted input are
  likewise absent, not merely undocumented.
- No electron count is returned. `03_chemistry_to_ground_state.py:100-105`
  reads `mol.nelectron` separately to build its Hartree-Fock reference.

### C. `from_psi4`

- Arguments: `from_psi4(wavefunction)` (`chemistry.py:170`) — e.g. the second
  return value of `psi4.energy("scf", return_wfn=True)`.
- Shapes/ranks: `Ca()` is read as one dense block; `H()` is the AO core
  Hamiltonian.
- Dtypes/domains: real Psi4 matrices, converted with `np.asarray` and returned
  contiguous, again with no dtype cast.
- Ordering/layout: Psi4's C1 MO ordering, unchanged. `MintsHelper.mo_eri`
  already returns chemist `(pq|rs)`, matching the PySCF path
  (`chemistry.py:180-181`).
- Required mathematical properties: a restricted wavefunction computed in C1
  symmetry.
- Validation and rejection behavior — two checks, both `ValueError`:

| # | Rejected condition | Source |
| --- | --- | --- |
| 1 | `wavefunction.nirrep() != 1` (irrep-blocked, higher symmetry) | `:189-193` |
| 2 | `wavefunction.same_a_b_orbs()` false (unrestricted reference) | `:194-197` |

  Neither rejection has a dedicated test: `test_psi4_conversion.py` builds only
  valid C1 restricted wavefunctions (`:45-60`). Both messages tell the caller
  the fix (`symmetry c1` in the geometry; use RHF/RKS).

### D. `spin_orbital_tensors`

- Arguments: `spin_orbital_tensors(one_body, eri, *, validate_symmetry=True)`
  (`chemistry.py:31-35`).
- Shapes/ranks: `one_body` is a square rank-2 `(n, n)`; `eri` must be exactly
  `(n, n, n, n)` with the same `n`.
- Dtypes/domains: any `ArrayLike`; both inputs are coerced to `complex128`
  unconditionally (`chemistry.py:53-54`). A complex-dtype `eri` gives the same
  operator as its real twin — asserted at `test_df_qsvt_bridge.py:176-187`,
  `< 1e-12`.
- Units: as in the shared statement; the `0.5` factor is a convention factor,
  not a unit conversion.
- Ordering/layout: `eri[p, q, r, s] == (pq|rs)` in chemist notation over **real
  spatial** orbitals (`preprocessing.rst:205-207`); output rows/indices are
  interleaved spin orbitals.
- Normalization: the `1/2` two-body factor is applied here, once
  (`chemistry.py:85`). Do not pre-scale the input.
- Required mathematical properties: `eri` must obey the real-orbital chemist
  permutation symmetry `(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)` and their
  compositions. This is what makes the resulting qubit Hamiltonian Hermitian
  (`chemistry.py:46-49`).
- Validation and rejection behavior — three `ValueError` conditions:

| # | Rejected condition | Source | Cited test |
| --- | --- | --- | --- |
| 1 | `one_body` not a square rank-2 matrix (a scalar included) | `:55-56` | `test_df_qsvt_bridge.py:55-58`, `:141-143` |
| 2 | `eri.shape != (n, n, n, n)` | `:57-61` | `test_df_qsvt_bridge.py:59-60` |
| 3 | `eri` fails the three symmetry generators `(1,0,2,3)`, `(0,1,3,2)`, `(2,3,0,1)` | `:62-71` | `test_df_qsvt_bridge.py:134-138` |

  Check 3 is `np.allclose(eri, eri.transpose(axes), atol=1e-8)`
  (`chemistry.py:67`). Because `np.allclose` keeps its **default
  `rtol=1e-05`**, the effective admission test is
  `|a - b| <= 1e-8 + 1e-5 * |b|`, i.e. large entries get relative slack — not a
  pure `1e-8` absolute screen. `validate_symmetry=False` skips it entirely and
  is accepted deliberately (`test_df_qsvt_bridge.py:139-140`); see "Accuracy and
  limitations" for what that then permits downstream.

### E. `qubit_hamiltonian`

- Arguments: `qubit_hamiltonian(one_body, eri, *, scalar_offset=0.0,
  tolerance=1e-12, validate_symmetry=True)` (`chemistry.py:96-101`).
- Shapes/ranks, dtypes, ordering, normalization, required properties: exactly
  **D**'s, which it calls first and whose `validate_symmetry` flag it forwards.
- Units: `scalar_offset` shares the integrals' energy scale — the
  nuclear-repulsion or FCIDUMP core energy. It is coerced with `float()`
  (`chemistry.py:124`), so a complex offset raises `TypeError` from Python, not a
  library `ValueError` (`unverified`: no test).
- Validation and rejection behavior: **D**'s three `ValueError` conditions, plus
  a lazily raised `ImportError` when the `fermion` subpackage is unavailable
  (`chemistry.py:118`, documented at `chemistry.py:11-16`).
  `fermion._compilers._validate_tensors` has its own shape rejections
  (`_compilers.py:193-221`), but **D** has already guaranteed a square `(2n, 2n)`
  and matching `(2n, 2n, 2n, 2n)` pair, so those messages are unreachable
  through this path (`derived`).

## Outputs

### A, B, C — the three loaders

- Return type: a 3-tuple `(one_body, eri, scalar)`, contiguous NumPy arrays plus
  a Python `float`. The scalar is the FCIDUMP **core energy** for A and the
  **nuclear-repulsion energy** for B and C.
- Mathematical meaning: the chemist-notation spatial integral triple of the
  Representation Record — precisely the arguments **D**, **E**, and the
  double-factorization entry points expect (`preprocessing.rst:29-36`).
- Shape/register geometry: `(n, n)` and `(n, n, n, n)`; no register exists at
  this layer. `n` is the number of *spatial* orbitals, so the eventual qubit
  count is `2n`.
- Normalization, sign, and phase: **the MO phase and degenerate-orbital ordering
  are an arbitrary gauge.** Two packages may return canonical MOs differing by
  orbital phase or ordering, a unitary gauge that leaves the spectrum and every
  physical observable unchanged (`test_psi4_conversion.py:12-17`). Compare
  spectra, never tensor entries, across sources.
- Dtype: A builds `np.zeros((n, n))` and `np.zeros((n, n, n, n))`, so float64
  (`chemistry.py:310-311`). B and C pass through the source package's dtype.
- Observable or measurement interpretation: none.
- Error/status information: none. Failure is an exception; there is no status
  object, no warning channel, and no "records skipped" report — a skipped
  orbital-energy record in A leaves no trace.

### D. `spin_orbital_tensors`

- Return type: `(one_body_so, two_body_so)`, both `complex128`.
- Mathematical meaning: the coefficient arrays of `adag_p a_q` and
  `adag_p adag_q a_r a_s` over `2n` spin orbitals, in the library's
  no-implicit-symmetry-factor convention.
- Shape: `(2n, 2n)` and `(2n, 2n, 2n, 2n)`, dense, with only the four
  spin-conserving patterns nonzero.
- Normalization, sign, phase: the `1/2` factor is already applied; no phase
  convention of its own; not antisymmetrized.
- Observable interpretation: none. Error/status: none.

### E. `qubit_hamiltonian`

- Return type: `cudaq.SpinOperator`.
- Mathematical meaning: the Jordan-Wigner image of the spin-expanded
  Hamiltonian, plus `scalar_offset * I`.
- Register geometry: **the operator's qubit width tracks the qubits actually
  touched, not `2n`.** A Hamiltonian that never couples the highest spin
  orbital yields a narrower operator, and `to_matrix()` is then
  `2^(touched)` (`_compilers.py:51-57`). For H2/STO-3G the width is the expected
  4 (`test_df_qsvt_bridge.py:50`), but a caller needing a fixed `2^(2n)` must
  ensure the top orbital is touched or pad downstream.
- Normalization, sign, and phase: no normalization is applied — coefficients are
  the compiled integrals, and the one-norm `alpha` appears only when a consumer
  block-encodes the operator. Hermiticity holds when `eri` satisfies the
  chemist symmetry (`test_df_qsvt_bridge.py:147-155`, `:158-172`, `atol=1e-10`).
- Observable or measurement interpretation: none of its own. Expectation values
  and moments belong to the consumers (`Walk.moment`, `select_observable`).
- Error/status information: none beyond exceptions. `scalar_offset` becomes a
  **scalar identity term acting on no degrees**, which is a real edge case
  downstream: `test_pauli_lcu.py:317-329` exists specifically because CUDA-Q's
  `max_degree` raises on such a term, and it records that
  `PauliLCU`'s register-extent inference must not be constrained or crashed by
  it. That test names the chemistry bridge's `scalar_offset` as the producer of
  the form.

## Capabilities and composition

### Provided — the chemist-notation integral source

- Stable ID: `cudaq-algorithms.chemistry-integrals.v1`
- Direction: **provides** (A, B, C) and **requires** (D, E).
- Owning family record: this file; contract in the Capability Record below.
- Boundary representation: the Representation Record's triple.
- Semantic invariants: chemist `(pq|rs)` ordering over real spatial orbitals;
  8-fold permutation symmetry; MO basis in the source's own orbital order;
  the scalar energy kept **outside** the tensors.
- Shape/register geometry: `(n, n)` with `(n, n, n, n)`; no register.
- Normalization, sign, phase, and ordering: no `1/2` folded in yet (that is
  **D**'s job); MO phase and degenerate ordering are a free gauge.
- Convention requirements: real spatial orbitals; a restricted reference.
- Host/device/simulation boundary: host only; NumPy in, NumPy out.
- Unsupported conditions: unrestricted or spin-resolved integral sets;
  irrep-blocked Psi4 wavefunctions; complex integrals unless the consumer's
  symmetry check is explicitly disabled.

### Required — the fermion-to-qubit transform

- Stable ID: none. [fermion-transforms.md](fermion-transforms.md) owns the
  populated primitive contract, but no reusable capability was extracted
  because the architecture gate is not met.
- Direction: **E** requires it and satisfies it with `fermion.jordan_wigner`
  (`chemistry.py:122`).
- Boundary representation: `(one_body_so, two_body_so)` in the
  no-implicit-symmetry-factor convention, plus `scalar_offset` and `tolerance`.
- Unsupported conditions: **no transform selector exists.** `bravyi_kitaev` is
  not reachable through **E**; substituting it means calling **D** and then the
  transform directly, accepting that the two-body tensor is compiled literally
  (`_compilers.py:339-347`).

### Downstream — where the qubit Hamiltonian goes

- Stable ID: none declared. A Pauli operator is input data for the concrete
  `PauliLCU` and `Trotter` constructors, not itself the zero-flagged
  block-encoding capability owned by [block-encoding.md](block-encoding.md).
  Check each consumer contract; do not infer support by name.
- Source-grounded consumers of **E**'s output at the cited commit:
  `PauliLCU(hamiltonian, num_qubits=None, ...)`
  (`pauli_lcu.py:478-484`) and `Trotter(hamiltonian, ...)`
  (`trotter.py:339-347`); `Walk` and `QSVT` consume the *encoding*, not the
  operator, so they are reached through `PauliLCU`
  (`03_chemistry_to_ground_state.py:94-107`).
- Width behavior to check at that boundary: `PauliLCU` infers the register
  extent as *largest targeted qubit + 1*, **not** `qubit_count`, and accepts an
  explicit wider `num_qubits` as legal padding while rejecting a narrower one
  (`pauli_lcu.py:359-371`; `test_pauli_lcu.py:311-315`). `Trotter` infers width
  the same way from term extents (`trotter.py:156-167`).
- Coefficient reality: both consumers route coefficients through
  `_real_coefficient`, which raises
  `ValueError("complex Hamiltonian coefficients are not supported")` when
  `|imag| > 1e-10` (`common_kernels.py:60-70`). A non-Hermitian input admitted
  by `validate_symmetry=False` will *generally* produce complex coefficients and
  be rejected there; that every non-Hermitian input trips this guard is
  `unverified`.
- Offset placement is a real choice, and both are exercised in the repository:
  fold the scalar into the operator via `scalar_offset`
  (`test_df_qsvt_bridge.py:49`), or keep it classical and add it after the
  quantum step. `03_chemistry_to_ground_state.py:92` deliberately passes
  `scalar_offset=0.0` and adds the nuclear repulsion **after** rescaling the
  Krylov eigenvalue by `encoding.alpha` (`:109-111`) — because the block
  encoding normalizes by `alpha`, an offset folded into the operator is scaled
  with everything else and must be un-scaled consistently. Keeping it classical
  also keeps `alpha` smaller.

### Upstream — double-factorization compression

- `double_factorization.reconstruct_eri(factorization)` returns a tensor in the
  same chemist `(pq|rs)` convention, so a compressed factorization drops
  straight into **D** or **E** in place of the exact `eri`
  (`preprocessing.rst:359-374`; `test_df_qsvt_bridge.py:63-96`). At full rank the
  round-trip leaves the spectrum and `alpha` unchanged to `1e-8`
  (`:71-77`); truncation lowers `alpha` and shifts the spectrum by an amount the
  tests bound with the tensor reconstruction error (`:81-96`).
- **Do not substitute the DF-corrected one-body matrix.**
  `double_factorization.modified_one_body_integrals` returns
  `kappa_pq = h_pq - (1/2) sum_r (pr|qr)` (`_factorization.py:714-721`), which
  belongs to assembling the *double-factorized* Hamiltonian. Every packaged and
  documented path into **E** passes `one_body` **unmodified** alongside the
  reconstructed `eri` (`preprocessing.rst:369-374`;
  `test_df_qsvt_bridge.py:67-69`), because **D** applies its own `1/2` factor to
  the full tensor. Passing `kappa` to **E** is unsupported: no source or test
  does it, and its numerical consequence here is `unverified`.

## Composite protocol

Applies to **E** only (`Abstraction level: composite protocol`). A–D are leaf
operations and this section does not apply to them.

- Required lower-level capabilities: the spin expansion of **D**, and the
  fermion-to-qubit transform described above.
- Canonical reference composition (`chemistry.py:118-126`):

  ```python
  one_body_so, two_body_so = spin_orbital_tensors(
      one_body, eri, validate_symmetry=validate_symmetry)
  return fermion.jordan_wigner(one_body_so, two_body_so,
                               scalar_offset=float(scalar_offset),
                               tolerance=float(tolerance))
  ```

- Default recipe and its applicability conditions: Jordan-Wigner with symmetry
  validation on, applicable to real chemist-symmetric spatial integrals from a
  restricted reference. Nothing in the default is target- or resource-adaptive.
- Materially different alternatives: **D** plus `fermion.bravyi_kitaev` for
  `O(log n)`-weight Pauli words (`_compilers.py:333-337`); a DF-compressed `eri`
  in place of the exact one; keeping the scalar energy classical instead of
  folding it in.
- Propagated conventions: interleaved spin orbitals; qubit 0 least significant;
  Pauli-word position equals qubit index; no implicit symmetry factors; the
  `1/2` applied exactly once, in **D**.
- Propagated errors: **D**'s three `ValueError` conditions, forwarded unchanged;
  a lazy `ImportError` for a missing `fermion` subpackage; the transform's own
  shape errors, unreachable on this path.
- Propagated resources and how they compose: **D**'s dense `(2n)^4` tensor is
  materialized first and then consumed term by term, so peak host memory is
  **D**'s tensor plus the accumulating operator. See "Resources".
- Component substitution: **not parameterized.** There is no `transform=`
  argument, so substitution means writing the two calls yourself. A substitute
  must preserve the tensor layout, the interleaved spin convention, and the
  no-implicit-symmetry-factor meaning of the entries; a transform that
  antisymmetrizes internally would silently change the operator
  (`_compilers.py:339-347`).

## Accuracy and limitations

- Error behavior or bounds. A–D introduce no approximation. For **E**, the only
  loss is `tolerance` pruning; no source states a bound, and the elementary
  bound that dropping terms perturbs the operator by at most the sum of the
  dropped coefficient magnitudes follows from norm subadditivity — stated here
  as `derived` analysis, **not** as a repository claim, and never as a measured
  spectral error. Truncated double factorization upstream is a separate,
  larger error whose contract belongs to that module.
- Precision sensitivity:
  - the symmetry check's effective slack is `1e-8 + 1e-5 * |entry|`, per the
    `np.allclose` default `rtol` noted under "Inputs";
  - **A** recovers only the digits the file printed, so a writer round-trip is
    exact only to that precision — the PySCF-writer interop test asserts
    spectrum agreement at `atol=1e-10`, not tensor equality
    (`test_fcidump.py:179-193`);
  - cross-package agreement is limited by physical constants, not by this code:
    PySCF and Psi4 use marginally different Bohr-radius values, so the same
    geometry yields nuclear repulsion and integrals differing at `~1e-9`
    (`test_psi4_conversion.py:93-97`);
  - **E** builds a `complex128` operator; simulator precision is irrelevant
    here because nothing in this family runs on a device.
- Unsupported inputs: unrestricted references in all three loaders (A's header
  guard, B's `ndim` check, C's `same_a_b_orbs` check); non-C1 Psi4
  wavefunctions; non-square or mismatched tensors; asymmetric `eri` unless the
  check is explicitly disabled.
- Known implementation limitations:
  1. **Pure-Python quadruple loop.** **D** expands the tensor in nested Python
     `for` loops over `p, q, r, s` (`chemistry.py:78-92`), so cost grows as
     `n^4` with interpreter overhead — no vectorized path exists.
  2. **Dense two-body storage.** `(2n)^4` `complex128` entries, i.e.
     `256 * n^4` bytes, with no sparse or symmetry-packed alternative.
  3. **ROHF is unguarded in B.** A single 2D `mo_coeff` passes the only check
     while the docstring restricts the function to restricted references.
  4. **A discards `NELEC`, `MS2`, `ORBSYM`, `ISYM`.** Electron count and spin
     must be carried separately by the caller.
  5. **A has no duplicate-record detection.** Later records overwrite earlier
     ones in both the `eri` orbit fill and the symmetrized `h` assignment
     (`chemistry.py:329-334`), so a file with conflicting duplicates parses
     silently, last record winning; a genuinely non-symmetric `h_ij`/`h_ji` pair
     is silently symmetrized.
  6. **No active space, frozen core, or orbital localization** in B or C — the
     caller folds inactive orbitals and passes effective integrals plus a CAS
     core energy (`df_block_encoding.py:93-105`).
  7. **No encoding selector in E** (Jordan-Wigner only).
  8. **The output operator's width can be narrower than `2n`** when the top
     spin orbital is untouched (`_compilers.py:51-57`).
- Unsupported versus unverified — read the labels literally:

| Behavior | Label | Why the label |
| --- | --- | --- |
| Unrestricted / spin-resolved integral sets | unsupported | rejected by an explicit guard in each loader, with a message naming the reason |
| Irrep-blocked (non-C1) Psi4 wavefunction | unsupported | rejected at `chemistry.py:189-193` |
| Bravyi-Kitaev through `qubit_hamiltonian` | unsupported | no selector exists; the call site is hard-wired to `jordan_wigner` |
| Active space / frozen core inside B or C | absent | no parameter exists; the caller folds the integrals first |
| Electron count or spin from any loader | absent | not in any return signature |
| ROHF passed to `from_pyscf` | unverified | passes the only guard; the docstring restricts the function to restricted references; no test |
| Complex integrals with `validate_symmetry=False` | unverified | accepted by **D** (`test_df_qsvt_bridge.py:139-140`), but no source or test characterizes the resulting operator or its downstream behavior |
| Complex `mo_coeff` or complex `scalar_offset` | unverified | no dtype cast and a bare `float()` coercion; no test |
| `NORB <= 0`, malformed tokens, unexpected index patterns, duplicate records | unverified | guards 6, 8, 10 exist in source with no cited test; duplicates have no guard at all |
| Psi4 rejection paths | unverified | both raise in source; no test constructs an invalid wavefunction, and Psi4 is absent from CI |

## Resources

This family runs entirely on the host, so the relevant quantities are classical
memory and work plus the *size* of the object handed downstream. Nothing below
was executed; every figure is a structural count.

| Quantity | Metric and unit | Abstraction level | Assumptions | Status | Controlling parameter | Limitations / composition |
| --- | --- | --- | --- | --- | --- | --- |
| **D** two-body tensor | bytes: `(2n)^4 * 16 = 256 * n^4` | host memory | dense `complex128` array (`chemistry.py:53-77`) | exact structural count, `derived` | `n` spatial orbitals | e.g. ~41 MB at `n = 20`, ~207 MB at `n = 30`; peak also holds the input `eri` and, in **E**, the accumulating operator |
| **D** expansion work | iterations: `n^4` Python loop bodies | host operation count | nested `for` loops (`chemistry.py:78-92`) | exact structural count, `derived` | `n` | interpreter-bound, **not** a runtime figure; no timing evidence exists anywhere for this module |
| **E** term compilation | Pauli-word products: up to `2^4 = 16` words per nonzero two-body entry, `2^2 = 4` per one-body entry | host operation count | each ladder operator is a two-term word sum (`_compilers.py:159-167`), accumulated over `np.argwhere` nonzeros (`_compilers.py:287-300`) | `derived` from source structure | number of nonzero tensor entries; `tolerance` | words collide in the accumulator, so the final term count is far smaller than the product count; not a gate count |
| **E** output size | `hamiltonian.term_count`; qubit width | qubit-operator size | after `tolerance` pruning and `canonicalize()` | inspectable property, `derived` | `tolerance`, integral sparsity | the metric the examples report alongside `PauliLCU.alpha` (`df_compression_to_qsvt.py:48-70`) |

Explicitly **not** provided by this family: gate counts, circuit depth, T or
Toffoli counts, runtime, and device memory. `PauliLCU.alpha` and the SELECT cost
are consumer-side quantities; a resource proxy from one abstraction level must
never be compared with another (`conventions.md`).

`Deferred:` a per-quantity resource contract for the qubit-side metrics
(`alpha`, term count as a SELECT cost, QSVT degree `~ alpha * t`), until the
block-encoding family has a populated record that owns them. There is no
resource estimator in this family.

## Validation

- Independent oracles. Six, cited and not executed:
  1. **Literature H2/STO-3G integrals plus an FCI total energy** of
     `-1.137270 Ha`: the bridged operator's minimum eigenvalue must match to
     `abs=5e-5` (`test_df_qsvt_bridge.py:36-52`). The same integrals are the
     reference the FCIDUMP parse must reproduce at `atol=1e-12`
     (`test_fcidump.py:34-42`, `:89-95`).
  2. **PySCF's own FCI solver** — an independent classical algorithm — for
     H2 and H4 / STO-3G: the Jordan-Wigner ground state must match to
     `< 1e-8` (`test_psi4_conversion.py:70-81`).
  3. **A second electronic-structure package.** PySCF and Psi4 tensors for the
     same molecule and basis must give qubit Hamiltonians with the *same
     spectrum* to `atol=1e-6`, with nuclear repulsion agreeing to `1e-6`
     (`test_psi4_conversion.py:84-103`). Spectrum, not coefficient equality, is
     the correct invariant because the MO gauge is free (`:12-17`).
  4. **The canonical external writer.** `pyscf.tools.fcidump.from_integrals`
     writes a file; the parse of its text must give the same spectrum as the
     integrals fed directly, `atol=1e-10` (`test_fcidump.py:179-193`).
  5. **Hand-computed spin expansion.** Individual entries of **D**'s output
     checked against `0.5 * eri.transpose(0, 2, 3, 1)` and the interleaved
     one-body placement — no eigensolver involved, so index and transpose typos
     are caught directly (`test_df_qsvt_bridge.py:115-131`).
  6. **Dense-matrix Hermiticity**, including a generic `n = 3` case built from
     random 8-fold-symmetric integrals, where typos that cancel under H2's
     sparsity are exposed (`test_df_qsvt_bridge.py:147-172`, `atol=1e-10`).
- Invariants under test: Hermiticity of the bridged operator; spectrum
  invariance under the MO gauge; spectrum and `alpha` invariance under a
  full-rank DF round-trip (`:63-77`); strict-versus-tolerant FCIDUMP parse
  equality (`atol=1e-12`, `test_fcidump.py:110-117`); the eight-fold symmetry
  fill from one stored record, which **D**'s own symmetry check must then accept
  (`test_fcidump.py:98-107`); real-versus-complex dtype equivalence
  (`< 1e-12`); the downstream identity
  `Walk.moment(ket, 1) == <H>/alpha` and `moment(ket, 0) == 1`
  (`test_df_qsvt_bridge.py:100-112`, `abs=1e-8`).
- Representative cases: H2/STO-3G (literature integrals, PySCF, and FCIDUMP
  text), H4/STO-3G chain, a generic random `n = 3` symmetric integral set, and
  H2O CAS(4e,4o) in the example path.
- Predeclared tolerances: as tabulated in each item above. They are the
  committed values at the cited commit, chosen in the tests rather than by this
  record.
- Expected failure / adversarial cases: every row of the three validation
  tables that carries a cited test — asymmetric `eri` rejected and then
  deliberately admitted with `validate_symmetry=False`; a scalar `one_body`
  giving a clear `ValueError` rather than an `IndexError`; shape mismatches;
  and A's missing header, unterminated header, empty body, malformed line,
  out-of-range index, `IUHF=1`, and `UHF=.TRUE.`, with `IUHF=0` accepted. Rows 2
  and 3 of A's table are the "silent zero" adversarial pair: a truncated file
  must raise rather than return `H = 0`.
- Reference results: the H2/STO-3G integral values, `E_nuclear = 0.71375697`,
  and `FCI = -1.137270 Ha` are literals in `test_df_qsvt_bridge.py:36-44` and
  `test_fcidump.py:34-42`. No other reference data is stored in the repository.
- Evidence status per claim: `derived` from the cited source line or committed
  test assertion, except where labeled `assumed` or `unverified` in place.
  **Nothing in this record is `measured`.**
- Coverage gaps worth knowing before relying on a boundary: A's guards 6, 8, and
  10 have no test; duplicate FCIDUMP records have no guard; C's two rejection
  paths have no test and no CI environment; every B and C test skips without its
  optional package; and the consequences of `validate_symmetry=False` are tested
  only to the point of acceptance.

## Evaluation coverage

Declared coverage: `chemistry-bridge-dependency-boundary` in
`evals/evals.json` tests the supported chemistry-input bridge, integral
representation, restricted-reference boundary, hard-wired Jordan-Wigner path,
and optional dependency status. It has not been run with or without the skill,
so no uplift is claimed.

The following are additional coverage opportunities:

| Template slot | Additional case, and the sections that would answer it |
| --- | --- |
| Positive selection/application | choose the loader from the artifact in hand (file text vs PySCF mean field vs Psi4 wavefunction), then reach a `SpinOperator`: Scientific contract A–C and E; Inputs A–C; Outputs |
| Convention or misconception | chemist vs physicist ordering with the `transpose(0, 2, 3, 1)` and single `1/2`; interleaved spin orbitals; `modified_one_body_integrals` is **not** a drop-in `one_body`; Bravyi-Kitaev is unreachable through `qubit_hamiltonian`: Scientific contract D and E; Capabilities and composition (upstream DF); Composite protocol |
| Capability composition | DF `reconstruct_eri` -> `qubit_hamiltonian` -> `PauliLCU` -> `Walk`/`QSVT`, including width inference, the scalar identity term, and where the offset must be re-added relative to `alpha`: Capabilities and composition (downstream and upstream) |
| Invalid/unsupported boundary | an unrestricted FCIDUMP or UHF mean field, a symmetry-blocked Psi4 wavefunction, a request for an active space, or a request for the electron count: Inputs validation tables; Accuracy and limitations |
| Negative activation | choosing a basis set, converging an SCF, or optimizing a geometry belongs to PySCF or Psi4, not to this library; and no version, runtime, or performance figure for these bridges may be presented as verified: Identity and provenance (versions unverified); Resources |

Declared coverage is not validated coverage.

## External alignment

- Literature conventions: chemist `(pq|rs)` notation and the Jordan-Wigner
  ladder operators as stated in `docs/sphinx/conventions.rst:46-84`. This module
  cites no paper of its own; the double-factorization papers belong to that
  module's record.
- External package translations, all as used in source:
  - PySCF: `ao2mo.full` returns symmetry-packed chemist MO integrals and
    `ao2mo.restore(1, ...)` (equivalently `"s1"`) expands them to the dense
    `(n, n, n, n)` form (`chemistry.py:160-164`;
    `preprocessing.rst:206-208`); `get_hcore()` rather than
    `int1e_kin + int1e_nuc`, to retain ECP, X2C, and QM/MM contributions;
    `energy_nuc()` for the scalar; `mcscf.CASCI.get_h1eff()/get_h2eff()` as the
    caller-side active-space pattern.
  - Psi4: `Ca()`, `H()`, `MintsHelper.mo_eri` (already chemist-ordered),
    `molecule().nuclear_repulsion_energy()`; requires `symmetry c1`.
  - FCIDUMP: the Knowles-Handy interchange format, 1-based Fortran indices,
    with Psi4's optional `oe_ints=['EIGENVALUES']` orbital-energy records
    skipped (`chemistry.py:335-340`); interoperates with
    `pyscf.tools.fcidump.from_integrals` (`test_fcidump.py:179-193`).
- Known semantic differences:
  - **The two loaders' one-body matrices are not guaranteed to coincide.**
    PySCF's `get_hcore()` is documented here as including ECP, X2C, or QM/MM
    contributions, while Psi4's `H()` is described as kinetic + potential. For
    an ECP or relativistic setup the two definitions may differ; the tests only
    compare all-electron non-relativistic cases, so agreement beyond those is
    `unverified`.
  - Physical constants differ between the packages (Bohr radius), giving
    `~1e-9` differences in nuclear repulsion and integrals for identical
    geometries (`test_psi4_conversion.py:93-97`).
  - OpenFermion and literature tables: many inequivalent tensor layouts encode
    the same physical operator, and the transforms compile tensors exactly as
    given, so `conventions.rst:66-71` directs callers to normalize the layout
    first or compare spectra. **No translation code for an external quantum-chemistry
    package exists in this repository**, so any specific mapping is `unverified`
    until checked against that package's own documentation.

---

# Representation Record — chemist-notation spatial integral triple

Justified because **three independent producers and several independent
consumers** exchange this exact object.

- Object name and canonical symbol: the chemist-notation spatial integral
  triple, `(one_body, eri, scalar_energy)`; commonly `h_pq`, `(pq|rs)`, and the
  core or nuclear-repulsion energy.
- Public type or structural form, and source path: a plain 3-tuple of two NumPy
  arrays and a `float`. **There is no named public type** — the form is
  established by the three loaders' return statements
  (`chemistry.py:166-167`, `:213`, `:345-346`) and by the parameters of
  `spin_orbital_tensors` and `qubit_hamiltonian`.
- Mathematical meaning: `one_body[p, q] = h_pq`, the one-electron (core
  Hamiltonian) matrix element in the MO basis; `eri[p, q, r, s] = (pq|rs)`, the
  two-electron integral in chemist notation over real spatial orbitals; the
  scalar is the constant energy held **outside** the operator.
- Shape, layout, ordering, dtype, and units: `(n, n)` and `(n, n, n, n)` for `n`
  *spatial* orbitals, so `2n` spin orbitals and `2n` qubits downstream;
  contiguous float64 in practice; MO ordering as the source produced it; energy
  scale as the source produced it (Hartree for all three loaders — see the
  shared units statement under "Inputs").
- Normalization, sign, and phase convention: no normalization; **no `1/2`
  folded into `eri`**; the MO phase and degenerate-orbital ordering are a free
  unitary gauge, so cross-source comparison must use spectra.
- Required mathematical properties (the applicability preconditions): real
  spatial orbitals from a restricted reference, and the 8-fold chemist
  permutation symmetry `(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)` and compositions.
  The symmetry is what makes the eventual qubit Hamiltonian Hermitian.
- Producers (>=2 required to justify this record): `from_fcidump`, `from_pyscf`,
  `from_psi4` (`chemistry.py`); `double_factorization.reconstruct_eri`, which
  re-emits the `eri` slot in the same convention
  (`double_factorization/_factorization.py:700-703`); plus the caller-side CASCI
  folding pattern in `docs/sphinx/examples/python/df_block_encoding.py:93-105`.
- Consumers: `spin_orbital_tensors` and `qubit_hamiltonian` (this file);
  `double_factorization.explicit_double_factorization`,
  `compressed_double_factorization`, `factorization_error`, and
  `modified_one_body_integrals` (`eri`, and `one_body` for the last);
  the `DoubleFactorizedEncoding` worked example
  (`docs/sphinx/examples/python/df_encoding.py`).
- Invariants preserved across the boundary: index convention, symmetry, MO
  basis, and the separation of the scalar energy from the tensors. Compression
  changes the *values* of `eri` while preserving all four
  (`preprocessing.rst:359-364`).
- Observable symptom of a misinterpretation: a physicist-ordered tensor, or one
  with a `1/2` already folded in, still produces a plausible Hermitian operator
  with a **wrong spectrum** — which is why the tests pin an FCI energy and a
  cross-package spectrum rather than checking shapes. Feeding `kappa` from
  `modified_one_body_integrals` in the `one_body` slot double-counts the DF
  one-body correction.
- Unsupported or ambiguous forms: symmetry-packed (4-fold) storage — must be
  expanded with `ao2mo.restore(1, ...)` first; spin-resolved (unrestricted)
  integral sets; complex integrals, whose symmetry differs and which require the
  consumer's check to be disabled explicitly; an `eri` whose `1/2` or spin
  expansion has already been applied.
- Source paths, tests, docs, and last verification: "Identity and provenance"
  above.

---

# Capability Record — chemist-notation integral source

- Stable ID: `cudaq-algorithms.chemistry-integrals.v1`. The name and the `.v1`
  suffix are an **open owner decision**: the ID conforms to
  `architecture.md`'s `cudaq-algorithms.<capability-name>.v<major>` pattern, but
  no second version is in sight, and the same reservation the state-preparation
  record raises about premature versioning applies here.
- Status: **provisional.** The extraction gate is met — three independent
  producers and several independent consumers already exchange the boundary —
  but the contract has not been verified and is not public API. An alternative
  reading is defensible and should be recorded if the owner prefers it: the
  Representation Record above may be sufficient on its own, with no capability
  extracted at all.
- Contract type: **documentation / taxonomy only.** Not a Python `Protocol`, not
  an ABC, not a public symbol. There is no runtime check anywhere that an object
  "is" this triple; consumers validate shapes and symmetry themselves.
- Boundary representation and exact signature: the Representation Record's
  triple. Producers return
  `tuple[np.ndarray, np.ndarray, float]`; consumers take
  `(one_body: ArrayLike, eri: ArrayLike)` positionally, with the scalar passed
  separately (`scalar_offset` for **E**) rather than as part of the tensor
  contract.
- Semantic invariants: chemist `(pq|rs)` ordering; real spatial orbitals; MO
  basis in the producer's own orbital order; 8-fold permutation symmetry; no
  `1/2` folded in; the scalar energy carried outside the tensors; consistent `n`
  between the two arrays.
- Register or shape geometry and ownership: `(n, n)` with `(n, n, n, n)`. There
  is no register at this boundary and no ownership question — the producer
  allocates the arrays and hands them over; consumers copy on coercion
  (`np.asarray(..., dtype=np.complex128)` in **D**) and never mutate the input.
- Convention requirements: the shared ordering and layout statement under
  "Inputs", plus `conventions.md` for everything the spin expansion and
  transform then impose.
- Host/device/simulation boundary: host only, NumPy in and NumPy out. No
  provider or consumer of this boundary touches a device, a simulator, or
  `cudaq` — the `cudaq` dependency appears only *after* the transform, inside
  **E**.
- Providers, with source paths: `from_fcidump`, `from_pyscf`, `from_psi4`
  (`python/cudaq_algorithms/chemistry.py`);
  `double_factorization.reconstruct_eri`
  (`python/cudaq_algorithms/double_factorization/_factorization.py`) for the
  `eri` slot; caller-side folding such as the CASCI pattern in
  `docs/sphinx/examples/python/df_block_encoding.py`.
- Consumers, with source paths: `spin_orbital_tensors` and `qubit_hamiltonian`
  (`python/cudaq_algorithms/chemistry.py`); the double-factorization entry
  points (`python/cudaq_algorithms/double_factorization/`); the
  `DoubleFactorizedEncoding` worked example
  (`docs/sphinx/examples/python/df_encoding.py`).
- Unsupported and unverified conditions: as tabulated under "Accuracy and
  limitations". In capability terms, a provider that emits spin-resolved,
  physicist-ordered, pre-scaled, symmetry-packed, or complex integrals does
  **not** satisfy this boundary, and the DF-corrected `kappa` matrix is not a
  substitute for the `one_body` slot.
- Promotion criteria to a public protocol or compiler IR operation, and the
  explicit decision still required: (a) a producer outside `chemistry.py` and
  `double_factorization` that exercises the boundary without widening it;
  (b) a decision on whether the scalar energy belongs *inside* the exchanged
  object, since it is returned as part of the triple but passed separately to
  every consumer; (c) a decision on whether active-space and electron-count
  metadata belong to this boundary, since callers currently carry them
  out-of-band; and (d) an explicit team decision recorded with an owner. None
  holds at the cited commit, and **no promotion is proposed**.
