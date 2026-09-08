# Fermion-to-qubit transforms

Status: draft. Operation + object: **transform (encode)** a **fermionic
ladder-coefficient tensor** into a **Pauli operator**.

This file is one family reference covering the **two implemented public
transforms**, `jordan_wigner` and `bravyi_kitaev`. Each Primitive-Record
heading of `assets/primitive-record-template.md` is instantiated **once at
family level**; per-provider material appears in visibly separate `###`
subsections beneath the heading it belongs to. One optional schema follows the
primitive record: a **Representation Record** for the ladder-coefficient tensor
pair the two transforms consume. No Capability Record is created, and the
reason is recorded under "Capabilities and composition".

**Two records inside one file, deliberately.** The two transforms share their
argument list, coercion, shape validation, error messages, `tolerance`
semantics, output type, and Hamiltonian convention *exactly* — one shared
implementation path (`_validate_tensors`, `_compile_hamiltonian`,
`_to_spin_operator`) with one parameter changed. They differ in the qubit
encoding, the computational basis, Pauli-word weight and mode-count dependence,
the reachable call paths, and their migration hazards. Shared contract is
therefore stated once at family level; the differences are never merged. The
`## Scientific contract` → "Semantic differences" subsection is the index of
every place they diverge.

**Evidence rule for this family.** Nothing was executed in this session. Every
test named below is *cited repository evidence* — a committed assertion read at
the cited commit, labeled `derived` — and never a fresh measurement, runtime, or
performance claim. Algebraic statements labeled `derived` follow from a
definition in the cited source.

## Identity and provenance

- Owner: CUDA-Q Algorithms Team.
- Public symbols and import paths: `jordan_wigner`, `bravyi_kitaev` — both
  `__all__` of `cudaq_algorithms.fermion`; reachable as
  `from cudaq_algorithms.fermion import jordan_wigner, bravyi_kitaev` or
  `cudaq_algorithms.fermion.jordan_wigner`. The subpackage is imported by
  `cudaq_algorithms/__init__.py`, but **neither transform is exported from the
  package root**; there is no `cudaq_algorithms.jordan_wigner`.
- Source paths: `python/cudaq_algorithms/fermion/__init__.py`,
  `python/cudaq_algorithms/fermion/_compilers.py` (the whole implementation,
  351 lines). Private and **not** public contract: `_Encoding`,
  `_identity_matrix`, `_fenwick_matrix`, `_gf2_inverse`, `_word_product`,
  `_terms_product`, `_validate_tensors`, `_accumulate`, `_compile_hamiltonian`,
  `_to_spin_operator`.
- Authoritative tests: `tests/python/test_fermion_compilers.py` (640 lines, the
  primary suite), `tests/python/test_jordan_wigner.py` (live-PySCF FCI
  cross-check, `pytest.importorskip("pyscf")`), `tests/python/test_fermion.py`
  (two import/smoke tests asserting only `op is not None`). Downstream-path tests that exercise the
  Jordan-Wigner transform through `chemistry.qubit_hamiltonian`:
  `tests/python/test_df_qsvt_bridge.py`, `tests/python/test_fcidump.py` (both
  gate those cases on a successful `fermion` import).
- Authoritative documentation: `docs/sphinx/guide/preprocessing.rst`
  ("Fermion-to-qubit transforms (Jordan-Wigner and Bravyi-Kitaev)");
  `docs/sphinx/conventions.rst` ("Fermionic integral tensors",
  "Jordan-Wigner ladder operators", "Spin orbitals: interleaved", "Qubit
  ordering", "Pauli words", "Hermiticity");
  `docs/sphinx/api/python_api.rst` (`automodule cudaq_algorithms.fermion` with
  `:members:`, which publishes the subpackage docstring and the two function
  docstrings). **The most detailed statement of the construction and of the
  migration differences is the `_compilers.py` module docstring, which that
  directive does not publish**, so several contract facts below exist only in
  source and appear in no rendered document.
- Package/CUDA-Q versions verified: **unverified.** The declared environment is
  Python `>=3.11` with `cudaq >= 0.15.0, < 0.16` (`pyproject.toml`) and CUDA-Q
  pinned at `b28cf0f6f2d9387f12ca81b6824b8833a38530af` (`.cudaq_version`). No
  test was executed for this family and no version was confirmed by running
  anything.
- Commit/date last verified: `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`
  (2026-09-03).
- Lifecycle: draft (both transforms).
- Replacement and migration notes: both transforms **replace retired compiled
  C++ transforms**. The module docstring lists three behavior differences that
  are documented as intended changes rather than bugs; they are recorded under
  `## Accuracy and limitations` → "Migration hazards from the retired compiled
  extension" because they silently change results for existing callers. No
  deprecation is in effect for either public symbol.

## Classification

Family-level values, identical for both transforms unless a subsection says
otherwise.

- Operation + mathematical object (primary identity): **transform (encode)** a
  fermionic ladder-coefficient tensor pair into a Pauli operator
  (`cudaq.SpinOperator`). The input-object → output-object *pair* is the
  identity; `transform` alone does not distinguish this family from
  `chemistry.spin_orbital_tensors` (fermionic → fermionic) or
  `double_factorization.modified_one_body_integrals` (one-body → one-body).
- Kind: **classical transformation / preprocessing.** No quantum operation, no
  kernel factory, no measurement, no simulation-only analysis, no resource
  estimator. Nothing in this family emits or launches a device kernel.
- Routine role: **computational.** Each transform performs one distinct,
  well-defined, independently usable task. Per `references/architecture.md`, a
  host classical transform is computational, not auxiliary; running on the host
  is execution layer, not role.
- Abstraction level: **leaf operation.** The composite that sequences a spin
  expansion and a transform is `chemistry.qubit_hamiltonian`, which belongs to
  the chemistry-bridge family, not to this one.
- Parameterization: **construction-time** only — every argument is supplied at
  the single call and the result is a fully materialized operator. There is no
  runtime-parameterized object and no deferred evaluation.
- Execution layers: **host preprocessing** (pure Python plus NumPy plus
  `cudaq` operator construction). No device kernel, no simulator, no GPU path,
  no compiled extension, no network, no file or environment access.
- Input representations: the ladder-coefficient tensor pair plus a scalar
  offset — see the Representation Record below. Accepted as `ArrayLike`.
- Output representations: `cudaq.SpinOperator`, an **upstream CUDA-Q type this
  library does not own.** No representation record is created for it; only the
  conventions that cross the boundary are recorded under `## Outputs`.
- Domain: **domain-independent.** Neither transform contains any chemistry
  structure: arbitrary coefficient tensors, including non-Hermitian and purely
  random ones, are accepted and compiled faithfully
  (`test_jordan_wigner_nonhermitian_tensor_matches_dense`,
  `test_bravyi_kitaev_is_permuted_jordan_wigner_nonhermitian`). Chemistry enters
  only in the *producer* above them (`chemistry.py`) and in the interleaved
  spin-orbital convention that producer uses. This is direct evidence for
  `architecture.md`'s "domain is a tag, not a parallel taxonomy" invariant:
  domain varies per contract, and `fermion/` is the domain-independent core of
  the chemistry pipeline.
- Required dependencies: `numpy`; `cudaq` (for `cudaq.SpinOperator`,
  `cudaq.SpinOperator.empty`, and `cudaq.spin.{x,y,z}`); CPython `>= 3.11` as
  declared by `pyproject.toml` (`int.bit_count`, which `_word_product` uses for
  the symplectic phase and sign, itself requires only `>= 3.10`).
- Optional dependencies: **none.** `scipy` is a package dependency but is not
  imported by this family; it appears only in the test suite's sparse
  Fock-space oracle. `pyscf` is needed only by
  `tests/python/test_jordan_wigner.py`.

Provisional metadata — recorded, not routed on:

- Exactness: **exact up to the `tolerance` prune.** The algebraic map is exact
  (no series truncation, no discretization, no iterative solve); the only
  approximation is the magnitude cutoff, plus double-precision floating-point
  arithmetic. With `tolerance` small enough to retain every term, the compiled
  operator equals the operator denoted by the tensors exactly, up to rounding.
- Uncertainty: **deterministic.** No randomness anywhere; the accumulation
  order is fixed (`sorted(accumulator.items())` on the symplectic key).
- Method: **direct.**

### Jordan-Wigner

Same values as the family. Method sub-detail: the GF(2) encoding matrix is the
identity, so the transform is the textbook occupation-basis encoding.

### Bravyi-Kitaev

Same values as the family. Method sub-detail: the GF(2) encoding matrix is the
Fenwick (binary-indexed-tree) partial-sum matrix.

## Scientific contract

- Purpose: compile a fermionic operator, supplied as coefficient tensors over
  `n` fermionic modes, into an equivalent qubit operator expressed as a
  weighted sum of Pauli words, so it can be block-encoded, evolved, or
  measured by the quantum primitives.

- Mathematical definition. Both transforms compile the same operator
  expression, stated identically in the `_compilers.py` module docstring,
  `docs/sphinx/guide/preprocessing.rst`, and
  `docs/sphinx/conventions.rst` ("Fermionic integral tensors"):

  ```text
  H = scalar_offset * I
      + sum_{ij}   h[i, j]       adag_i a_j
      + sum_{ijkl} V[i, j, k, l] adag_i adag_j a_k a_l
  ```

  with **no implicit symmetry factors and no implicit normal ordering or
  antisymmetrization**: the sums run over all index tuples and every entry is
  taken literally in the operator order written above. The name "Hamiltonian
  convention" in the source does not add a Hermiticity requirement; see
  `## Inputs` → "Required mathematical properties".

- Shared construction (the unifying scientific content, `derived` from the
  `_compilers.py` module docstring and `_Encoding`). A fermion-to-qubit
  encoding here is a linear map over GF(2): an invertible binary matrix `A`
  stores the occupation vector `n` in the qubit bits as `x = A n (mod 2)`.
  Three masks per mode `j` follow from `A` and its GF(2) inverse:

  | Mask | Definition in source | Meaning |
  | --- | --- | --- |
  | update `U(j)` | column `j` of `A` (`_Encoding.update_masks`) | qubits that flip when occupation `j` flips — the X part |
  | parity `P(j)` | XOR of rows `0..j-1` of `A^-1` (`_Encoding.parity_masks`) | qubits whose parity is the fermionic parity below `j` — the Z part carrying the anticommutation sign |
  | flip `F(j)` | row `j` of `A^-1` (`_Encoding.flip_masks`) | qubits whose parity equals occupation `j` — the number-operator Z word |

  With the Majorana word `M_j = X_{U(j)} Z_{P(j)}` (parity read *before* the
  flip), the ladder operators are

  ```text
  adag_j = (M_j + M_j Z_{F(j)}) / 2
  a_j    = (M_j - M_j Z_{F(j)}) / 2
  ```

  and each tensor entry is compiled by multiplying out the corresponding
  product of these two-term sums (`_Encoding.ladder_terms`, `_accumulate`).
  Pauli words are carried symplectically as a bitmask pair `(x, z)` denoting
  `i^{popcount(x & z)} X^x Z^z` (so `Y` sits on every overlap bit), which makes
  a word product two XORs plus a phase (`_word_product`).

  **This unifying abstraction is implementation-internal, not contract.**
  `_Encoding` is private, and no public API accepts a user-supplied encoding
  matrix. Record it as scientific context; do not present it as an extension
  point, and do not propose a public encoding protocol on its strength.

- Why and when to use: whenever a fermionic operator given as coefficient
  tensors has to become a `cudaq.SpinOperator` for `PauliLCU`, `Walk`, `QSVT`,
  or `Trotter`. Choose by the properties in "Semantic differences" below,
  never by name familiarity.

- When not to use:
  - the input is chemist-notation spatial integrals rather than spin-orbital
    ladder coefficients — spin-expand first (`chemistry.spin_orbital_tensors`),
    or use the packaged composite `chemistry.qubit_hamiltonian` for the
    Jordan-Wigner path;
  - a *structured* (non-Pauli-sum) encoding is wanted, for example a
    double-factorized block encoding built from leaf rotations — that consumes
    the factorization directly and does not route through these transforms
    (`docs/sphinx/guide/preprocessing.rst`: "the classical factorization itself
    does not [use `fermion.jordan_wigner`]");
  - the operator is already a `cudaq.SpinOperator`;
  - only a mode-count-independent Pauli-weight guarantee will do — see the
    Bravyi-Kitaev limitation on mode-count dependence.

- Approximation controls: `tolerance` only, applied at two stages, described
  in full under `## Inputs` and `## Accuracy and limitations`.

### Semantic differences between the two transforms

Every point on which the two contracts diverge. Same-numbered material is not
merged anywhere else in this file.

1. **Encoding matrix and qubit meaning.** Jordan-Wigner uses `A = I`
   (`_identity_matrix`), so qubit `j` stores occupation `n_j` directly.
   Bravyi-Kitaev uses the Fenwick partial-sum matrix (`_fenwick_matrix`), so
   qubit `i` stores the GF(2) sum of the occupations in the binary-indexed-tree
   range that ends at mode `i` — in the source's one-based statement, node `i`
   covers `(i - lowbit(i), i]`, so zero-based qubit `i` covers modes
   `i + 1 - lowbit(i + 1) .. i`. `derived` from source.
2. **Computational basis.** The two operators live in **different** qubit
   bases, related by the exact basis permutation `P |n> = |A n mod 2>` in the
   little-endian bit order of `to_matrix()`. `bravyi_kitaev(...) =
   P jordan_wigner(...) P^T` is asserted as an exact matrix identity at
   `atol=1e-12` for Hermitian tensors (`m = 3..6`) and for a non-Hermitian
   tensor (`m = 4`) in `test_bravyi_kitaev_is_permuted_jordan_wigner*`. `P` is
   constructed **inside the test** from the private `_fenwick_matrix`; the
   package exports no public basis-mapping helper. Consequence: the spectra
   agree (`test_bravyi_kitaev_isospectral_to_jordan_wigner`, `atol=1e-10`) but
   the two operators are *not* interchangeable in any computation that
   interprets basis states, amplitudes, or per-qubit locality.
3. **Pauli-word weight.** Jordan-Wigner parity strings grow with the mode
   index: the compiled support of the entry `(i, j)` lies in
   `{0, ..., max(i, j)}` (`derived` from `U(j) = {j}`, `F(j) = {j}`,
   `P(j) = {0..j-1}`), and the maximal-range one-body hop reaches full weight —
   `max_weight(jordan_wigner(...)) == 16` for the `(0, 15)` hop at `m = 16` in
   `test_bravyi_kitaev_word_weight_advantage`. Bravyi-Kitaev's docstring claims
   `O(log n)`-weight words; the same test asserts only the weaker
   `max_weight(bravyi_kitaev(...)) < 8` for that case. Treat `O(log n)` as a
   **documentation claim** (`derived` from the docstring, no asymptotic study in
   the repository) and the `< n/2` figure as the pinned assertion at one size.
4. **Mode-count dependence of the compiled words.** Jordan-Wigner's masks are
   independent of `n`, so appending idle modes cannot change the Pauli words of
   an existing entry. Bravyi-Kitaev's update mask is a *column* of the Fenwick
   matrix, whose support grows with `n` (for `n = 20`, column 0 has support
   `{0, 1, 3, 7, 15}`), so the same tensor entry can compile to different words
   at different `n`. High-index bits shared by all update masks of a term
   cancel in the XOR — visibly so in the pinned known answers, where the
   `(1, 2)` and `(0, 7)` pairs at `m = 20` touch only low qubits — but whether
   they cancel is term-dependent. `derived` from `_fenwick_matrix` and
   `_Encoding`; no general per-term bound is established in source or tests.
5. **Two-body antisymmetrization.** Neither transform antisymmetrizes now. The
   `bravyi_kitaev` docstring records that the retired compiled binding *did*
   antisymmetrize the two-body tensor internally, that removing it repairs a
   prior JW/BK inconsistency, and that a caller who passed a raw,
   non-antisymmetrized (for example chemist-ordered) tensor and relied on that
   behavior "will silently get a different operator." Jordan-Wigner carries no
   such change.
6. **Reachable call paths.** `chemistry.qubit_hamiltonian` hard-wires
   `fermion.jordan_wigner`; **Bravyi-Kitaev is not reachable through the
   chemistry bridge.** A Bravyi-Kitaev chemistry Hamiltonian requires the
   explicit two-step path `chemistry.spin_orbital_tensors(...)` then
   `bravyi_kitaev(...)`, and that path also loses the bridge's `float()`
   coercion of `scalar_offset` and its `tolerance` default of `1e-12`. Whether
   the omission is deliberate is **not stated by any source read here**.
7. **Interoperability with the packaged state preparation.** The
   state-preparation family is explicitly a "Jordan-Wigner / little-endian
   qubit layout" family (`python/cudaq_algorithms/stateprep/_givens.py` module
   docstring; `docs/sphinx/guide/state_prep.rst`), and
   `hartree_fock_occupation` prepares a basis state indexed by *occupied
   spin orbitals*. Those states therefore match a Jordan-Wigner Hamiltonian
   directly. Pairing them with a Bravyi-Kitaev Hamiltonian requires mapping the
   occupation through `A`, and **no public helper does that** (point 2). No
   test in the repository combines `bravyi_kitaev` with any state-preparation
   provider. Label a Bravyi-Kitaev plus packaged-state-preparation composition
   **unverified and unsupported by evidence**, not merely untested-but-fine.
8. **Oracle sets.** Jordan-Wigner alone is pinned against dense NumPy ladder
   operators and against a **live** PySCF RHF+FCI cross-check over four
   molecules. Bravyi-Kitaev alone is pinned by a direct Fenwick-matrix check,
   five hand-computed single-pair known answers, and the exact
   permutation identity `BK = P·JW·P^T`. Only the frozen-PySCF H2 energy and
   the three-way spectral agreement cover both. The sets are different, and
   `## Validation` keeps them separate rather than reporting one confidence for
   the family.
9. **Width-tracking evidence.** The output-width behavior of
   `## Outputs` is asserted for `jordan_wigner` only
   (`test_operator_width_tracks_touched_qubits`). Because Bravyi-Kitaev's
   touched-qubit set is not `{0..max coupled mode}` in general (point 4), the
   analogous Bravyi-Kitaev bound is **not established here**.

Everything else — the argument list, coercion, accepted shapes, every error
message, both `tolerance` stages, the returned type, the identity-term
handling, the deterministic accumulation order, the `ValueError`-not-
`RuntimeError` migration change, and the width-tracking *mechanism* — is
literally shared code and is stated once at family level.

## Inputs

Shared by both transforms; the signatures are identical.

```python
jordan_wigner(one_body_or_two_body, two_body=None, scalar_offset=0.0, tolerance=1e-15)
bravyi_kitaev(one_body_or_two_body, two_body=None, scalar_offset=0.0, tolerance=1e-15)
```

- Arguments:
  - `one_body_or_two_body: ArrayLike` — a rank-2 one-body tensor **or** a
    rank-4 two-body tensor. The overload is resolved by rank alone.
  - `two_body: Optional[ArrayLike] = None` — the rank-4 tensor, legal only when
    the first argument is rank 2.
  - `scalar_offset: float = 0.0` — added as an identity term.
  - `tolerance: float = 1e-15` — magnitude cutoff, applied twice (below).
  All four are positional-or-keyword; none is keyword-only.
- Shapes/ranks. Three accepted forms (`_validate_tensors`):
  1. rank-2 `(n, n)` alone — the two-body tensor becomes `zeros((0,0,0,0))` and
     is skipped entirely (`if two_body.size:`);
  2. rank-2 `(n, n)` plus rank-4 `(n, n, n, n)` with `n` from the one-body
     tensor;
  3. rank-4 `(n, n, n, n)` alone — the one-body tensor becomes `zeros((n, n))`.
  `n` is the number of fermionic modes and equals `shape[0]` of the leading
  tensor. Every other rank or shape is rejected; see the error table.
- Dtypes/domains. Both tensors are coerced unconditionally with
  `np.asarray(..., dtype=np.complex128)`, so Python lists, real arrays,
  integer arrays, and `float32` arrays are all accepted and upcast, and a
  complex tensor is taken as is. Coefficients may be complex; there is no
  real-valued restriction and no Hermiticity restriction **at this boundary**
  (the downstream consumers do restrict — see `## Capabilities and
  composition`). Input that NumPy cannot cast to `complex128` raises NumPy's
  own error, whose type and message are **unverified** here.
- Units: none. The coefficients carry whatever energy unit the caller's
  integrals use (Hartree throughout the repository's chemistry paths) and the
  transform is unit-agnostic; `scalar_offset` must be in the same unit.
- Ordering/layout.
  - Mode indices are the fermionic mode (spin-orbital) indices, and mode `j`
    maps to **qubit** `j` for Jordan-Wigner. Qubit 0 is the least significant
    bit of the basis index (`conventions.md`, `docs/sphinx/conventions.rst`).
  - The two-body index order is fixed:
    `V[i, j, k, l]` multiplies `adag_i adag_j a_k a_l`, so `l` is the innermost
    (rightmost) annihilation index. The Hermitian conjugate of that product is
    `adag_l adag_k a_j a_i`, i.e. the Hermiticity condition on `V` is
    `V[i, j, k, l] == conj(V[l, k, j, i])` (stated in the
    `_random_generic_system` docstring of `tests/python/test_fermion_compilers.py`).
  - **Interleaved spin orbitals are a producer convention, not a transform
    requirement.** `2p` is alpha/up and `2p + 1` is beta/down throughout the
    repository (`conventions.md`), and every packaged producer emits that
    layout, but the transforms never inspect or assume it.
  - **Layout non-uniqueness is the highest-value convention here**
    (`docs/sphinx/conventions.rst`): many inequivalent tensor layouts encode
    the *same* physical operator, because indices can be traded against
    fermionic antisymmetry. Since the transforms compile literally, any valid
    layout works — but a cross-source comparison (OpenFermion, literature
    tables) must normalize the layout first, or compare spectra rather than
    tensors.
- Normalization. None is applied and none is assumed. Any `1/2` factor from a
  chemist-notation source must already be folded into `V` by the caller; the
  packaged producer does exactly that (`chemistry.spin_orbital_tensors`
  multiplies by `0.5`).
- Required mathematical properties. **None are required and none are checked
  beyond shape.** In particular:
  - Hermiticity is *not* required. Generic tensors produce a non-Hermitian
    operator, faithfully (`test_jordan_wigner_nonhermitian_tensor_matches_dense`),
    and `docs/sphinx/conventions.rst` warns that `eigvalsh`, `.real`, and
    spectrum sorting are then invalid.
  - Antisymmetry of `V` is not required, not checked, and not imposed.
  - No index-permutation symmetry, positivity, or sparsity is required.
  This permissiveness is the contract: validity of the *physics* is the
  caller's responsibility, and the enforcement point for the chemistry
  precondition is one level up (`chemistry.spin_orbital_tensors`'s
  `validate_symmetry`), not here.
- Validation and rejection behavior. All validation runs on the host at call
  time, before any compilation, and there is no device-kernel boundary in this
  family for a silent no-op to hide behind (contrast `conventions.md`,
  "Validation happens on the host, before device kernels", which exists for the
  kernel-factory families). Every rejection is a `ValueError`:

  | Condition | Message |
  | --- | --- |
  | rank-2 first argument that is not square | `one_body dimensions must match` |
  | second argument present but not rank 4 | `two_body has the wrong rank` |
  | rank-4 second argument whose shape is not `(n, n, n, n)` | `one_body and two_body dimensions differ` |
  | rank-4 first argument with a second argument supplied | `second tensor is invalid when first is rank 4` |
  | rank-4 first argument whose shape is not `(n, n, n, n)` | `two_body dimensions must match` |
  | any other rank (0, 1, 3, 5, ...) | `expected rank-2 one_body or rank-4 two_body` |

  Four of the six are pinned by `test_validation_errors` with
  `pytest.raises(ValueError, match=...)` on the substrings `rank`,
  `dimensions differ`, `rank 4`, and `wrong rank`. The two non-square/non-cubic
  messages are `derived` from source and are **not** covered by a test read
  here. `_gf2_inverse` can raise
  `ValueError("encoding matrix is singular over GF(2)")`, but both packaged
  matrices are invertible by construction, so that path is unreachable through
  the public API.
- `tolerance` semantics (both stages, `derived` from `_compile_hamiltonian` and
  `_to_spin_operator`):
  1. **input side** — entries are first restricted to the nonzeros
     (`np.argwhere`), then an entry with `abs(value) < tolerance` is skipped;
  2. **output side** — a compiled word whose accumulated coefficient satisfies
     `abs(coefficient) < tolerance` is dropped.
  The test is a **magnitude disk on the complex value, not a componentwise
  square** (stated in a source comment), and the comparison is strict `<`, so a
  value exactly equal to `tolerance` is kept. Because an entry expands to
  several smaller words, an entry that survives stage 1 can still lose terms in
  stage 2: `test_output_side_tolerance_trims_small_compiled_terms` pins this
  with `tolerance = 1e-6`, where a diagonal entry `3e-6` keeps its `1.5e-6`
  `Z` term while `1.5e-6` loses its `0.75e-6` one.
- `scalar_offset` semantics: it seeds the identity word,
  `accumulator = {(0, 0): complex(scalar_offset)}`, so it is **summed with the
  identity contributions of the number operators** rather than kept separate,
  and the combined identity coefficient is then subject to the stage-2 prune.
  `test_scalar_offset_and_tolerance` pins `II == 2.25` for `scalar_offset = 2.0`
  with `h[0,0] = 0.5` (whose `adag_0 a_0 = (I - Z_0)/2` contributes `0.25`).
  The parameter is *typed* `float` and reaches `complex(scalar_offset)`, so a
  complex value would not raise at this boundary; that is undocumented and
  **unverified**, and the composite `chemistry.qubit_hamiltonian` does apply
  `float(...)`, which would reject it.

### Jordan-Wigner

No additional input contract.

### Bravyi-Kitaev

No additional input contract. The only input-adjacent asymmetry is that `n`
also determines the encoding matrix here, which is why appending idle modes
changes the compiled words (Semantic differences, point 4).

## Outputs

- Return type: `cudaq.SpinOperator` — an **upstream CUDA-Q type**, returned by
  both transforms, built term by term from `cudaq.spin.{x,y,z}` and finished
  with `.canonicalize()`.
- Mathematical meaning: the qubit operator equal to the fermionic operator of
  `## Scientific contract` under the transform's encoding, minus every pruned
  contribution.
- Shape/register geometry — the family's single largest composition hazard.
  The module docstring states it as a documented migration difference: "the
  returned operator's qubit width tracks the qubits actually touched." So
  - a Hamiltonian that never couples the highest mode(s) yields a **narrower**
    operator than `n` qubits, and `to_matrix()` is then `2^(touched)`, not
    `2^n` — `test_operator_width_tracks_touched_qubits` pins a `(3, 3)`
    one-body tensor with only `h[0,0]` nonzero to a `(2, 2)` matrix;
  - a fully pruned or all-zero Hamiltonian yields
    `cudaq.SpinOperator.empty()`, with `term_count == 0` and a `(0, 0)`
    `to_matrix()` (same test, at the default `scalar_offset = 0.0`). The
    docstring states this as "a fully-pruned or all-zero Hamiltonian yields an
    empty operator"; precisely, it requires the *combined* identity coefficient
    to be pruned too, so an all-zero tensor with a surviving `scalar_offset`
    returns an identity-only operator of extent 0 instead — see
    `## Capabilities and composition`, consequence 3;
  - a caller needing a fixed `2^n` must ensure the top mode is touched, or pad
    the result — the docstring says so explicitly. `## Capabilities and
    composition` records which consumers offer a width override and which do
    not.
  Per Semantic differences point 9, the assertion covers Jordan-Wigner only.
- Normalization, sign, and phase.
  - Words are canonical Pauli words: `i^{popcount(x & z)} X^x Z^z`, with `Y` on
    every X/Z overlap bit (`_word_product`). Coefficients are complex and
    carry all of the sign and phase; no global phase or normalization is
    introduced anywhere.
  - The ladder-operator sign convention that fixes every relative sign is, for
    Jordan-Wigner, exactly the one published in
    `docs/sphinx/conventions.rst`:
    `adag_j = ½ Z_0 ... Z_{j-1} (X_j - i Y_j)` and
    `a_j = ½ Z_0 ... Z_{j-1} (X_j + i Y_j)`. `derived`: substituting
    `U(j) = {j}`, `P(j) = {0..j-1}`, `F(j) = {j}` into
    `_Encoding.ladder_terms` (whose `sign` is `+1` for `dagger` and `-1`
    otherwise) reproduces those two expressions term for term.
  - The anticommutation sign is carried by the parity string over the modes of
    **lower index** in ascending mode order. Operator order inside a tensor
    entry is therefore significant: permuting indices of `V` changes the
    operator unless the caller antisymmetrizes.
- Observable or measurement interpretation: the returned operator is a
  Hamiltonian/observable in the Pauli-word convention `word[0]` acts on qubit
  0 (`conventions.md`). Extract terms with
  `term.get_pauli_word(width)` and `term.evaluate_coefficient()`, always
  passing an explicit `width`, so identity padding is explicit — the repository
  does exactly that in its own comparison helper
  (`_terms` in `tests/python/test_fermion_compilers.py`).
  Hermiticity — and therefore the validity of `eigvalsh`, `.real`, and sorted
  real spectra — holds only if the caller's tensors satisfy `h = h†` and
  `V[i,j,k,l] = conj(V[l,k,j,i])` and `scalar_offset` is real; it is a property
  of the input, never of the transform.
- Error/status information: **none in the return value.** There is no status
  field, no warning, and no return code; the only channel is a host exception
  at validation time. Notably, a *silent* outcome is possible and legitimate:
  an entirely pruned Hamiltonian returns an empty operator rather than raising.
- Determinism of term order: contributions are aggregated in a dict and then
  iterated in `sorted(...)` order on the symplectic key `(x, z)`, and summed by
  a pairwise binary-tree reduction before `.canonicalize()`. The aggregation is
  deterministic; the **term order of the final `cudaq.SpinOperator` after
  `canonicalize()` is an upstream detail and is not specified here.**

### Jordan-Wigner

No additional output contract.

### Bravyi-Kitaev

Same type and conventions. The basis in which the operator's matrix is written
differs — Semantic differences, points 2 and 9.

## Capabilities and composition

**No capability ID is minted for this family, deliberately.**
`references/architecture.md` extracts a capability only when multiple
independent producers or consumers demonstrate a *reusable boundary*, and
capabilities are documentation contracts that must not drive speculative API
design. Neither candidate clears that gate at the cited commit:

- a `fermion-to-qubit-encoding` capability would have to be carried by
  `_Encoding`, which is private, with no public API accepting a user-supplied
  encoding matrix — the abstraction exists in the implementation, not in the
  contract;
- a "qubit Hamiltonian" capability would be a capability over
  `cudaq.SpinOperator`, an upstream type this library does not own, whose
  consumers (`PauliLCU`, `Trotter`) accept it as a positional argument and each
  re-derive width and coefficients independently — nothing dispatches on a
  declared interface. Minting the ID here would freeze positional function
  signatures rather than a semantic boundary, and would put the owning record
  in the wrong family: the requirement side belongs to the consumer families.

Promotion trigger, so the decision is not re-litigated from scratch: a public
entry point that accepts a caller-supplied encoding (for the first), or a
consumer-family record that states the requirement side of the operator
boundary and an owner willing to hold the ID there (for the second), **plus**
an explicit team decision recorded with that owner.

Until then, composition is checked concretely, against representations and
conventions rather than IDs.

- Boundary representation, upstream (input): the ladder-coefficient tensor pair
  of the Representation Record below.
- Boundary representation, downstream (output): `cudaq.SpinOperator`.
- Host/device/simulation boundary: entirely host. Nothing here allocates a
  register, emits a kernel, or touches a simulator, so no register geometry or
  simulation precision enters the transform's own contract.

**Producer above the transform.** `chemistry.spin_orbital_tensors(one_body,
eri, *, validate_symmetry=True)` is the packaged producer of the input pair. It
consumes chemist-notation `(pq|rs)` spatial integrals over real spatial
orbitals and applies the documented translation — `eri.transpose(0, 2, 3, 1)`
(chemist → coefficients of `adag adag a a`), scale by `1/2`, distribute over
the four spin patterns `(2p, 2q, 2r, 2s)`, `(2p+1, 2q+1, 2r+1, 2s+1)`,
`(2p, 2q+1, 2r+1, 2s)`, `(2p+1, 2q, 2r, 2s+1)` — returning `complex128` tensors
over `2n` interleaved spin orbitals. It validates the three generators of the
real-orbital 8-fold chemist symmetry, `(1,0,2,3)`, `(0,1,3,2)`, `(2,3,0,1)`,
with `np.allclose(..., atol=1e-8)`; its docstring states *why*: an asymmetric
`eri` yields a non-Hermitian operator that the downstream primitives would
otherwise consume as if Hermitian. That check is the enforcement point for a
precondition these transforms deliberately do not impose. The chemist triple
itself, the three integral loaders (`from_fcidump`, `from_pyscf`, `from_psi4`),
and `qubit_hamiltonian` belong to the **chemistry-bridge family, not to this
one**; see the cross-reference note at the end of this section.

**Composite above the transform.** `chemistry.qubit_hamiltonian(one_body, eri,
*, scalar_offset=0.0, tolerance=1e-12, validate_symmetry=True)` sequences the
spin expansion and `fermion.jordan_wigner`, importing `fermion` lazily so that
importing `chemistry` never raises and any `ImportError` surfaces at call time
(deliberate, per its module docstring). Two propagation details that change
results and are easy to miss:

- its default `tolerance` is `1e-12` while the transforms' own default is
  `1e-15`, so the bridge **prunes a thousand times more aggressively** than a
  direct call;
- it is Jordan-Wigner only (Semantic differences, point 6).

**Consumers below the transform** (`derived` from source; only the boundary
facts that constrain *this* family's output are recorded here — the consumers'
own contracts belong to their own records):

| Consumer | How it reads the operator | Constraint it imposes on this family's output |
| --- | --- | --- |
| `PauliLCU(hamiltonian, num_qubits=None, *, include_identity=True, coefficient_threshold=1e-12)` | iterates terms; width is the **register extent** = largest targeted qubit + 1, explicitly *not* `qubit_count`, which counts distinct targets and undercounts gapped operators | complex coefficients rejected; empty operator rejected; a narrower-than-`n` operator can be widened by passing `num_qubits` (and `num_qubits` below the extent raises) |
| `Trotter(hamiltonian, ordering=..., *, coefficient_tolerance=1e-12)` | same extraction via `make_trotter_terms`/`_word_pairs_from_input`; width is the widest term's extent | complex coefficients rejected; empty operator rejected (`hamiltonian has no terms`); an **identity-only** operator rejected separately (`hamiltonian must act on at least one qubit`, since an identity term has extent 0); **no `num_qubits` parameter exists, so the narrow-width hazard has no override here** |
| `Walk`, `QSVT` | consume a `BlockEncoding` (e.g. a `PauliLCU`), not the operator directly | inherit `PauliLCU`'s constraints |

Three consequences worth stating plainly, all `derived`:

1. **Non-Hermitian operators do not compose downstream.** These transforms
   accept and faithfully compile complex-coefficient tensors, but both
   consumers route every coefficient through
   `common_kernels._real_coefficient`, which raises
   `ValueError("complex Hamiltonian coefficients are not supported")` once
   `abs(imag) > 1e-10`. That threshold is fixed package-wide and is
   **independent of the transform's `tolerance`**, so tightening or loosening
   `tolerance` will not make a non-Hermitian operator acceptable.
2. **Pruning happens at least twice.** The transform's `tolerance` (two
   stages), then the consumer's own `coefficient_threshold` /
   `coefficient_tolerance` (default `1e-12` in both). Do not attribute a
   missing term to a single cutoff without checking which stage dropped it.
3. **The two degenerate outputs this family can produce are the two the
   consumers reject.** An all-zero or fully pruned Hamiltonian returns an empty
   operator; a Hamiltonian with only `scalar_offset` surviving returns an
   identity-only operator of extent 0. `Trotter` rejects both, with the two
   distinct messages above. `PauliLCU` rejects the empty case; its behavior on
   an identity-only operator is **not characterized here** — its identity test
   is `set(word) == {"I"}`, which does not match the zero-length word an
   extent-0 operator produces, and no test read here covers it.

**State preparation.** Composing a Jordan-Wigner Hamiltonian with the packaged
state-preparation providers is convention-aligned, because that family is
documented as Jordan-Wigner / little-endian. The Bravyi-Kitaev case is
unverified and unsupported by evidence — Semantic differences, point 7. Read
[state-preparation.md](state-preparation.md) for the preparation side of any
such composition; nothing in this file establishes an injection contract.

**Cross-reference note.** Every neighbouring boundary fact above remains
grounded in repository source at the cited commit. For the complete adjacent
contracts, route through [chemistry-bridges.md](chemistry-bridges.md),
[block-encoding.md](block-encoding.md), [qubitization.md](qubitization.md),
[qsvt.md](qsvt.md), and [trotter.md](trotter.md). Resolve any future
cross-record disagreement against current source and tests.

## Composite protocol

`Deferred:` not applicable — both transforms are leaf operations. The composite
that sequences a spin expansion with a transform is
`chemistry.qubit_hamiltonian`, which belongs to the chemistry-bridge family;
only the boundary facts that constrain *this* family appear above, under
"Composite above the transform". Follow-up trigger: the chemistry-bridge record
owns `qubit_hamiltonian`'s composite-protocol headings — required lower-level
operations, canonical reference composition, applicability conditions, the
materially different alternative it does not expose (`bravyi_kitaev`), and its
propagated conventions, errors, and resources. If that record disagrees with
the two propagation details recorded above, resolve it against
`python/cudaq_algorithms/chemistry.py`.

## Accuracy and limitations

- Error behavior or bounds. The only error source other than floating point is
  the `tolerance` prune, and its effect is bounded:
  - each ladder operator expands to exactly 2 symplectic words of coefficient
    magnitude `1/2`, so a one-body entry expands to 4 words of magnitude
    `|h|/4` and a two-body entry to 16 words of magnitude `|V|/16` — in both
    cases the expanded word magnitudes **sum to the entry magnitude**
    (`derived` from `_Encoding.ladder_terms` and `_terms_product`);
  - therefore dropping an input entry perturbs the operator by at most its own
    magnitude in spectral norm, dropping a compiled word perturbs it by at most
    that word's coefficient magnitude, and the total perturbation is at most the
    sum of all dropped magnitudes (triangle inequality over unit-norm Pauli
    words).
  This bound is `derived` and is a bound, not a measurement. **No test in the
  repository asserts it**, and no convergence study in `tolerance` exists.
- Precision sensitivity. All arithmetic is `complex128` on the host; the
  symplectic phase bookkeeping is exact (integers and fourth roots of unity),
  so rounding enters only through the coefficient products and the accumulator
  sums. The repository's own strongest term-level assertions sit at
  `atol=1e-12`, and its spectral ones at `atol=1e-10`. No device kernel is
  launched by this family, so the `qpp-cpu`/fp64 simulator pinning in
  `tests/python/conftest.py` is not part of these transforms' precision story;
  the oracles are host NumPy/SciPy dense references.
- Unsupported inputs: every rejected rank/shape in the error table; any input
  NumPy cannot cast to `complex128` (rejection type **unverified**).
- Known implementation limitations:
  1. **Cost scales with the number of nonzero entries, not with sparsity of the
     result.** The compile loops in Python over `np.argwhere` of each tensor, so
     a dense two-body tensor over `m` modes drives `m^4` iterations and `16 m^4`
     word products regardless of how many distinct words survive.
  2. **Output width tracks touched qubits** (`## Outputs`), which silently
     changes downstream register geometry.
  3. **No public encoding extension point** (`## Scientific contract`).
  4. **Bravyi-Kitaev is unreachable from the chemistry bridge** (Semantic
     differences, point 6).
  5. **No resource estimator** exists in this family (`## Resources`).
  6. **An entirely pruned Hamiltonian returns an empty operator silently**,
     which then raises only at the consumer.
- Migration hazards from the retired compiled extension (all three from the
  `_compilers.py` module docstring, all documented as intended behavior
  changes):
  1. invalid shapes/ranks now raise `ValueError`, where the compiled binding
     surfaced a C++ `throw` as `RuntimeError` — code with
     `except RuntimeError` around a transform must be updated;
  2. the returned operator's width now tracks the touched qubits (above);
  3. `bravyi_kitaev` no longer antisymmetrizes the two-body tensor internally,
     and a caller who relied on that will *silently* get a different operator
     (Semantic differences, point 5).
- Unsupported versus unverified — read each label literally:

  | Behavior | Label | Why |
  | --- | --- | --- |
  | Rejected ranks/shapes in the error table | **unsupported** | explicit `ValueError` in source, four of six pinned by `test_validation_errors` |
  | Non-Hermitian tensors at this boundary | **supported** | compiled faithfully, asserted against a dense oracle for both transforms |
  | Non-Hermitian operators at `PauliLCU`/`Trotter` | **unsupported** | `_real_coefficient` raises above `abs(imag) > 1e-10` |
  | Complex `scalar_offset` | **unverified** | typed `float`, reaches `complex(...)`, undocumented, untested; `chemistry.qubit_hamiltonian` would reject it via `float(...)` |
  | Caller-supplied GF(2) encoding matrix | **absent** | `_Encoding` is private; no public entry point exists |
  | Identity term pruned when its net coefficient falls below `tolerance` | **unverified** | follows from the stage-2 prune, but the pinned prune test asserts only about non-identity terms |
  | Bravyi-Kitaev output-width bound | **unverified** | asserted for `jordan_wigner` only; BK's touched set is not `{0..max coupled mode}` in general |
  | Bravyi-Kitaev with packaged state preparation | **unverified** | no basis-mapping helper, no test, no documentation |
  | Non-numeric or ragged input | **unverified** | NumPy's own coercion error surfaces; type and message uncharacterized |
  | Singular GF(2) encoding matrix | **unreachable** | guarded in `_gf2_inverse`, but both packaged matrices are invertible |
  | Asymptotic `O(log n)` Bravyi-Kitaev weight | **unverified** | a docstring claim; the one pinned case asserts `< n/2` at `n = 16` |

## Resources

These are host classical transforms, so the resource contract has two parts:
the classical cost of the transform itself, and the properties of the produced
operator that drive downstream quantum cost. **This family provides no
`estimate_*` helper** — the absence is a fact about the library, not a deferred
section. Every quantity below is `derived` from source or from a cited
assertion; **none was measured, and no runtime, memory, or gate-count figure
here may be presented as measured.**

| Quantity | Metric and unit | Abstraction level | Assumptions | Status | Controlling parameters | Limitations / composition |
| --- | --- | --- | --- | --- | --- | --- |
| Word products performed | count of symplectic word products | host classical work | pure-Python loops over `np.argwhere`; `nnz1`, `nnz2` are the nonzero counts | **exact** given the inputs | `nnz1`, `nnz2` | `4 * nnz1 + 16 * nnz2` (2 words per ladder operator, 2 factors for one-body and 4 for two-body); a dense two-body tensor makes this `16 m^4` |
| Distinct compiled words `T` | count of Pauli terms before pruning | operator representation size | words collapse in the accumulator dict | **bounded** | `nnz1`, `nnz2`, touched width `w` | `T <= min(1 + 4*nnz1 + 16*nnz2, 4^w)`; the actual value depends on cancellation and is not predicted in source |
| Term summation cost | term copies during reduction | host classical work | `cudaq` operator addition copies both sides | **bounded**, per a source comment | `T` | pairwise binary-tree reduction is `O(T log T)` copies, chosen because a left fold is `O(T^2)`; the comment's "seconds versus hours" phrasing is a source remark, **not a measurement made here** |
| Peak accumulator size | number of dict entries | host memory | one complex coefficient per distinct word | **bounded** | `T` | grows with `T`, not with `n` directly |
| Retained term count | Pauli terms in the returned operator | operator representation size | after both prune stages | **exact** given inputs and `tolerance` | `tolerance` | this is the quantity the guide names as **SELECT cost** downstream (`docs/sphinx/guide/preprocessing.rst`) |
| Maximum Pauli word weight | non-identity letters in a word | operator locality | per encoding | Jordan-Wigner: **exact** for the pinned case; Bravyi-Kitaev: **bounded** by the pinned case, `O(log n)` claim is documentation only | encoding, mode indices | JW support of entry `(i, j)` lies in `{0..max(i,j)}`, and `max_weight == 16` for the `(0, 15)` hop at `m = 16`; BK asserted `< 8` for the same case |
| Output register width | qubits | register geometry | touched-qubit tracking | **exact** given inputs; asserted for JW only | which modes are coupled | narrower than `n` when top modes are idle; `PauliLCU` can widen via `num_qubits`, `Trotter` cannot |
| One-norm `alpha` of the encoded operator | sum of retained coefficient magnitudes | algorithmic cost driver | computed by `PauliLCU`, **not** by this family | not produced here | retained terms | the guide states QSVT degree scales like `alpha * t` for time evolution; a **different abstraction level** from every count above and never comparable to a gate count |

`conventions.md` ("Resource abstraction levels must not be conflated") governs
this table: none of these quantities is a transpiled gate count, a circuit
depth after synthesis, a runtime, or a memory figure, and they must not be
compared with one another as though they were one metric.

## Validation

- Independent oracle. Four oracle families, all reachable in the repository,
  none of which reuses the transform's own code path:
  1. **dense ladder operators built directly in NumPy** — `_dense_ladders` /
     `_dense_hamiltonian` in `tests/python/test_fermion_compilers.py` construct
     `a_j = Z^{⊗j} ⊗ [[0,1],[0,0]] ⊗ I^{⊗(m-j-1)}` by explicit Kronecker
     products in the little-endian order and assemble `H` entry by entry
     (evidence level 2);
  2. **exact Fock-space diagonalization via sparse ladder matrices** —
     `_sparse_fermionic_hamiltonian`, a SciPy construction independent of both
     transforms, used up to 10 qubits (level 2/5);
  3. **other implementations after convention alignment** — live PySCF
     RHF + FCI in `tests/python/test_jordan_wigner.py` and a frozen
     PySCF 2.13.1 H2 RHF+FCI reference, both genuinely external (level 3);
     plus the known-answer values inherited from the **retired in-house C++
     implementation's** unit tests, which are a different implementation but
     not an independent third party, and whose H2 coefficients the test itself
     notes are rounded;
  4. **exact analytical identities and invariants** — the Fenwick-matrix
     cross-check, the encoding-permutation identity, additivity, and
     isospectrality (levels 1 and 4).
- Invariants asserted in the repository:
  - `bravyi_kitaev(...) == P jordan_wigner(...) P^T` exactly, with `P` the
    basis permutation `|n> -> |A n mod 2>` — the strongest invariant here,
    because it pins Bravyi-Kitaev *term content* rather than only its spectrum;
  - isospectrality of the two transforms;
  - additivity: `jordan_wigner(h, V) == jordan_wigner(h) + jordan_wigner(V)`;
  - the Fenwick matrix equals an independently formulated binary-indexed tree.
- Representative cases and predeclared tolerances. These are **the
  repository's asserted tolerances read at the cited commit**; no tolerance was
  predeclared by this record, because nothing was executed.

  | Case | Test | Domain | Asserted tolerance |
  | --- | --- | --- | --- |
  | JW vs dense ladders, Hermitian tensors, `scalar_offset=0.25` | `test_jordan_wigner_matches_dense_ladders` | `m = 4`, seeds 5 and 17 | `atol=1e-12` |
  | JW vs dense ladders, non-Hermitian tensor | `test_jordan_wigner_nonhermitian_tensor_matches_dense` | `m = 3` | `atol=1e-12` |
  | One-body/two-body additivity | `test_jordan_wigner_one_body_only_and_two_body_only` | `m = 3` | max term difference `< 1e-12` |
  | Fenwick matrix vs independent BIT + hand-written `n = 8` | `test_fenwick_matrix_matches_binary_indexed_tree` | `n ∈ {1,2,3,4,5,8,12,13,16,20}` | exact equality |
  | BK hand-computed single pairs: number operator `(2,2)`, neighbor `(1,2)`, cross-branch `(2,6)`, root `(18,19)`, long-range `(0,7)` | five `test_bravyi_kitaev_*` cases | `m = 20` | `abs=1e-12` |
  | BK H2 known answer from the retired C++ test | `test_bravyi_kitaev_h2_known_answer` | 4 qubits | `tol=1e-4` (test notes the reference coefficients are rounded) |
  | BK isospectral to JW | `test_bravyi_kitaev_isospectral_to_jordan_wigner` | `m = 3,4,5` | `atol=1e-10` |
  | BK Pauli-weight advantage | `test_bravyi_kitaev_word_weight_advantage` | `m = 16`, `(0,15)` hop | JW `== 16`, BK `< 8` |
  | Exact fermionic vs JW vs BK spectra | `test_three_way_spectrum_agreement` | `n_spatial = 2..5`, i.e. 4–10 qubits, physically symmetric integrals | `atol=1e-10`, all three pairings |
  | BK `== P·JW·P^T` | `test_bravyi_kitaev_is_permuted_jordan_wigner` and `..._nonhermitian` | `m = 3..6` Hermitian; `m = 4` non-Hermitian | `atol=1e-12` |
  | H2 ground state vs frozen PySCF FCI, **both transforms** | `test_h2_ground_state_matches_frozen_fci` | 4 qubits, `tolerance=1e-12` | `< 1e-10` of `-1.1371757102406845` |
  | JW ground state vs live PySCF FCI | `tests/python/test_jordan_wigner.py::test_ground_state` | H2, two H4 geometries, LiH | `atol=1e-4`; skipped without `pyscf` |
  | `scalar_offset` and input-side prune | `test_scalar_offset_and_tolerance` | `m = 2`, `tolerance=1e-6` | `pytest.approx` default relative tolerance |
  | Output-side prune | `test_output_side_tolerance_trims_small_compiled_terms` | `m = 2`, `tolerance=1e-6` | `abs=1e-12` on the retained term |
  | Width tracking and empty operator | `test_operator_width_tracks_touched_qubits` | `m = 3`, JW | exact shapes `(2,2)` and `(0,0)` |
  | Shape/rank rejection | `test_validation_errors` | four malformed inputs | `ValueError` message match |
  | Import/smoke, both transforms | `tests/python/test_fermion.py` | `m = 2` | `op is not None` only |
- Expected failure/adversarial cases actually exercised: non-Hermitian tensors
  (both transforms); a raw random tensor with no symmetry at all; sub-tolerance
  entries; an entry that survives the input prune but whose compiled terms do
  not; an all-zero Hamiltonian; an idle top orbital; four malformed shapes. The
  Bravyi-Kitaev suite additionally records *why* the Fenwick matrix is pinned
  directly: the permutation and spectral tests hold for **any** invertible
  GF(2) matrix, so a mid-range off-by-one would otherwise leave the suite
  green — a well-reasoned adversarial argument worth reusing.
- Reference results worth carrying: the frozen H2/STO-3G FCI total energy
  `-1.1371757102406845` Ha at `R = 0.7474 Å` with nuclear repulsion
  `0.7080240981000804` Ha, reproduced by both transforms to `< 1e-10`; and the
  15-term Bravyi-Kitaev H2 operator tabulated in
  `test_bravyi_kitaev_h2_known_answer`.
- Evidence status per claim:
  - the mathematical construction, mask definitions, ladder-operator signs,
    accepted shapes, error messages, both prune stages, width tracking,
    migration differences, and the classical cost formulas — **derived**;
  - every tolerance and reference number quoted above — **derived** (committed
    assertions read at commit `61ac072d7823481ad0d303dac84d8ee29a9a8cd0`, *not*
    executed in this session, therefore never `measured`);
  - Bravyi-Kitaev's `O(log n)` weight scaling, the output-perturbation bound,
    and the downstream `alpha * t` degree relation — **derived** from a
    docstring, an inequality, and the guide respectively, with no numerical
    study in the repository;
  - complex `scalar_offset`, identity-term pruning, non-numeric input,
    Bravyi-Kitaev width tracking, and Bravyi-Kitaev with packaged state
    preparation — **unverified**;
  - any hardware or runtime performance claim — **absent**; the repository
    offers no such evidence for this family.
- Validation gaps a reviewer should know about: no test drives
  `bravyi_kitaev` through any chemistry path (the bridge is Jordan-Wigner
  only); no test combines either transform with a state-preparation provider;
  no test asserts the `tolerance` perturbation bound or studies convergence in
  `tolerance`; `tests/python/test_fermion.py` asserts only `op is not None`
  and is not contract evidence.

## Evaluation coverage

Declared coverage: `fermion-transform-selection-boundary` in
`evals/evals.json` tests explicit provider selection, shared tensor and
tolerance contracts, Bravyi-Kitaev mode-count dependence, and the unsupported
state-preparation interoperability assumption. It has not been run with or
without the skill, so no uplift is claimed.

The table below lists additional coverage opportunities:

| Coverage opportunity | Sections that would support it |
| --- | --- |
| Positive selection: choose between the two transforms for a stated objective | Classification; Scientific contract → Semantic differences (points 1–4) |
| Convention/misconception: the two-body index order, the missing `1/2`, and layout non-uniqueness | Inputs → Ordering/layout and Normalization; Representation Record |
| Convention/misconception: output width tracks touched qubits, so `to_matrix()` is not `2^n` | Outputs → Shape/register geometry; Capabilities and composition (consumer table) |
| Composition, legal: a Jordan-Wigner operator into `PauliLCU`/`Trotter`, checked by representation and convention rather than by a capability ID | Capabilities and composition |
| Composition, illegal: a non-Hermitian operator downstream, or Bravyi-Kitaev with packaged state preparation | Capabilities and composition (consequence 1); Semantic differences (point 7) |
| Approximation/resource claim: what `tolerance` costs, and refusing to quote a gate count or runtime | Accuracy and limitations; Resources |
| Invalid/unsupported boundary: malformed ranks, and Bravyi-Kitaev through the chemistry bridge | Inputs → Validation and rejection behavior; Semantic differences (point 6) |
| Refusal to fabricate: no resource estimator, no public encoding extension point, no measured performance | Resources; Scientific contract; Accuracy and limitations → Unsupported versus unverified |
| Negative activation: a general "what is Jordan-Wigner" question with no CUDA-Q Algorithms context | `SKILL.md` ownership boundary, not this file |

## External alignment

- Literature conventions.
  - **Jordan-Wigner** matches the standard textbook form in this library's own
    published convention,
    `adag_j = ½ Z_0...Z_{j-1}(X_j - i Y_j)`,
    `a_j = ½ Z_0...Z_{j-1}(X_j + i Y_j)`
    (`docs/sphinx/conventions.rst`), with the parity string over lower-indexed
    modes.
  - **Bravyi-Kitaev** is implemented in the Fenwick / binary-indexed-tree
    partial-sum formulation. The repository's known-answer cases are described
    as "the exact Seeley-Richard-Love per-pair operators … inherited from the
    retired C++ implementation's unit tests"
    (`tests/python/test_fermion_compilers.py` module docstring), i.e. the
    Seeley-Richard-Love per-pair operator tables are the external anchor. The
    citation is inherited rather than re-derived in the repository; no paper is
    cited in the module source itself.
  - No qubit-count reduction, symmetry taper, parity or ternary-tree encoding,
    or Majorana-only encoding is implemented. Only these two encodings exist.
- Convention translations required before comparing with any external source:
  1. **Qubit ordering** — qubit 0 is least significant here; most papers put
     the leftmost tensor factor first. Build dense references with
     `functools.reduce(np.kron, ops[::-1])` (`conventions.md`).
  2. **Pauli-word strings** — `word[0]` acts on qubit 0, reversed relative to
     left-to-right tensor notation; always pass an explicit `width`.
  3. **Tensor layout** — chemist `(pq|rs)` integrals must be reordered with
     `transpose(0, 2, 3, 1)`, scaled by `1/2`, and spin-expanded over the four
     patterns before they mean anything to these transforms.
  4. **Spin-orbital ordering** — interleaved (`2p` alpha, `2p+1` beta) here;
     permute indices before comparing with blocked-ordering data
     (`conventions.md`).
  5. **Layout non-uniqueness** — because several inequivalent tensor layouts
     denote the same physical operator, prefer comparing **spectra** rather
     than tensors or term dictionaries across sources
     (`docs/sphinx/conventions.rst`). The repository follows its own advice:
     its external cross-checks compare ground-state energies and spectra.
- Known semantic differences from external packages and from the library's own
  history:
  - **versus the retired compiled C++ transforms**: `ValueError` instead of
    `RuntimeError`; touched-qubit output width; and no internal Bravyi-Kitaev
    antisymmetrization — the last silently changes results for a caller who
    passed a raw chemist-ordered two-body tensor. The module docstring also
    notes the retired C++ Bravyi-Kitaev "assumed additional tensor structure
    beyond hermiticity", whereas both transforms here compile the tensors
    exactly as given.
  - **versus OpenFermion and similar packages**: no comparison test against
    OpenFermion's fermion-to-qubit transforms was found in this repository, and
    OpenFermion's operator-ordering, index, and normalization conventions are
    **not** characterized anywhere in this source. Treat any OpenFermion
    equivalence as **unverified**, translate layout explicitly, and compare
    spectra. (The repository's OpenFermion cross-check that does exist targets
    the double-factorization module, not this family.)
  - **versus other "Bravyi-Kitaev" formulations**: the repository's encoding is
    pinned directly to the binary-indexed-tree range-marking matrix, with the
    `n = 8` matrix written out in
    `test_fenwick_matrix_matches_binary_indexed_tree`. Compare any external
    implementation against **that matrix**, not against its spectra: the test
    file's own comment records that the permutation and spectral invariants
    hold for *any* invertible GF(2) matrix, so they cannot discriminate one
    formulation from another. (That the literature contains inequivalent
    matrices called "Bravyi-Kitaev" is a general remark, not a claim this
    source establishes.)

---

# Optional schema: Representation Record — fermionic ladder-coefficient tensor pair

Justified because two public primitives interpret this object identically
(`jordan_wigner`, `bravyi_kitaev`), with a third public symbol producing it
(`chemistry.spin_orbital_tensors`) and a canonical construction recipe named by
the conventions page.

- Object name and canonical symbol: the ladder-coefficient tensor pair plus
  scalar offset — `(h, V, c)`, spelled `(one_body, two_body, scalar_offset)`
  in source.
- Public type or structural form, and source path: no named public type. The
  form is two `numpy.ndarray`s (accepted as any `ArrayLike`, coerced to
  `complex128`) plus a Python scalar, established by `_validate_tensors` in
  `python/cudaq_algorithms/fermion/_compilers.py` and by the return value of
  `chemistry.spin_orbital_tensors`.
- Mathematical meaning: coefficient arrays for normal-ordered ladder products
  over `n` fermionic modes, denoting
  `c·I + Σ_{ij} h[i,j] adag_i a_j + Σ_{ijkl} V[i,j,k,l] adag_i adag_j a_k a_l`,
  with **no implicit symmetry factors** (`docs/sphinx/conventions.rst`).
  "Normal-ordered" describes the *fixed product form* the indices bind to —
  creation operators to the left of annihilation operators, in the written
  order — and not an operation the consumers perform: neither transform
  reorders, normal-orders, or antisymmetrizes what it is given.
- Shape, layout, ordering, dtype, and units: `h` is `(n, n)`, `V` is
  `(n, n, n, n)`, both coerced to `complex128`; either may be omitted (see the
  three accepted forms in `## Inputs`). Index positions are fermionic mode
  indices; `V[i,j,k,l]` binds to `adag_i adag_j a_k a_l` in exactly that
  order. Mode indices are spin-orbital indices in the producers' interleaved
  layout (`2p` alpha, `2p+1` beta), a producer convention the consumers do not
  enforce. Units are the caller's energy units, shared with `c`.
- Normalization, sign, and phase convention: none applied. Chemist-source
  `1/2` factors must already be folded into `V`. Signs and phases are entirely
  in the coefficients; the operator ordering above fixes their meaning.
- Required mathematical properties (applicability preconditions): only the
  shapes. Hermiticity — `h = h†` together with
  `V[i,j,k,l] = conj(V[l,k,j,i])` and real `c` — is required for the
  *downstream* operator to be Hermitian, and hence for `eigvalsh`, `.real`,
  sorted real spectra, and `PauliLCU`/`Trotter` acceptance, but it is not a
  precondition of the representation itself. Antisymmetry of `V` is neither
  required nor imposed. **Layout is not unique**: distinct tensors can denote
  the same physical operator, so tensor-level equality is a stronger and often
  wrong comparison.
- Producers (≥2 required to justify this record):
  `chemistry.spin_orbital_tensors` (the packaged producer);
  `chemistry.qubit_hamiltonian` internally; the canonical hand-written spin
  expansion in `tests/python/test_jordan_wigner.py`, which
  `docs/sphinx/conventions.rst` names as the reference implementation; and the
  same loop reproduced in `tests/python/test_fermion_compilers.py`
  (`_physical_system`, `_h2_frozen_spin_orbital_tensors`).
- Consumers: `fermion.jordan_wigner` and `fermion.bravyi_kitaev` — and, at the
  cited commit, no others. The double-factorization module consumes the
  *chemist spatial* tensor instead, which is a **different object** owned by a
  different family.
- Invariants preserved across the boundary: the index-to-ladder-operator
  binding, the absence of implicit symmetry factors and of implicit normal
  ordering, the mode-index-to-qubit-index correspondence for Jordan-Wigner, and
  the fact that both consumers compile the tensors literally so no entry is
  reinterpreted.
- Observable symptom of a misinterpretation:
  - a forgotten `1/2` or a wrong `transpose` yields a Hamiltonian whose
    *minimum* eigenvalue can look right to several digits while the full
    spectrum, the LCU `alpha`, and Hermiticity are wrong — the failure mode
    `tests/python/test_df_qsvt_bridge.py` explicitly guards against with a
    Hermiticity assertion at `atol=1e-10`;
  - a chemist-ordered `V` passed straight through gives a plausible operator
    with the wrong physics and, for callers migrating from the retired
    Bravyi-Kitaev binding, silently different results;
  - blocked instead of interleaved spin ordering leaves norms intact while
    per-orbital locality and any excitation pool built at the same spin no
    longer match.
- Unsupported or ambiguous forms: every rejected rank/shape in `## Inputs`;
  the chemist spatial `(pq|rs)` triple, which is a **different representation**
  and not a variant of this one; complex `c`, which is undocumented at this
  boundary and rejected by `chemistry.qubit_hamiltonian`.
- Source paths, tests, docs, and last verification: as in
  `## Identity and provenance`, plus `python/cudaq_algorithms/chemistry.py`
  (`spin_orbital_tensors`) and `docs/sphinx/conventions.rst` ("Fermionic
  integral tensors"). Read at commit
  `61ac072d7823481ad0d303dac84d8ee29a9a8cd0` (2026-09-03); nothing executed.
