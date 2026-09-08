# Excitation enumeration and operator pools

Status: draft. Operation + object: **preprocess** a **fermionic-excitation
operator pool** (an ordered list of Hermitian Pauli generators).

This family file covers **one independently selectable scientific contract per
pool construction**, plus the excitation enumeration that fixes their order. It
instantiates each applicable canonical Primitive-Record heading of
`assets/primitive-record-template.md` **once at family level**, with the four
provider constructions as subsections beneath, and adds one **Representation
Record** for the pool object itself.

## Why this is a separate record, and where its boundary is

A pool is not a prepared state and not a kernel. It is a host-side classical
construction whose output — an ordered list of `cudaq.SpinOperator` objects — is
consumed by several independent callers, only one of which is the
Hartree-Fock-plus-UCC factory. Under the record-granularity rule of
`architecture.md`, each construction has its own mathematical semantics, its own
index and spin conventions, its own argument units, and its own rejection
behavior, so the enumeration/pool layer is recorded here rather than folded into
a preparation contract.

| Owned here | Owned elsewhere |
| --- | --- |
| what each pool operator *is* (generator definition, coefficients, signs), pool order, index and spin conventions, per-pool argument units, rejection behavior, the three pool-specific Pauli-list converters, and the pool object's boundary contract | the `(qubits: cudaq.qview)` injection seam and the unitary state-preparation capability — [state-preparation.md](state-preparation.md); the `hartree_fock_ucc_kernel` contract, `get_fixed_parameter_ucc_pauli_lists`, `validate_fixed_parameter_ucc`, the HF occupation builders, and the UCC amplitude-convention (`scale`) table — [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md); interleaved spin-orbital layout, `spin = 2 * S_z`, Pauli-word position, and resource-abstraction rules — `conventions.md` |

**Out of scope, stated once.** *Choosing* which pool operators to keep and
*optimizing* their amplitudes is the excluded consumer workflow (`SKILL.md`):
ADAPT-VQE-style selection and VQE-style optimization are named by
`docs/sphinx/guide/state_prep.rst` as motivations for exposing the pools, and
they are not part of this contract. This record says what the pool *is* and in
what order it comes out; it says nothing about how many amplitudes should be
nonzero or what values they should take. The amplitude-convention comparison
between the `uccsd` circuit and the pool generators is **not** repeated here —
it belongs to the consumer record cited above.

## Identity and provenance

| Field | Value |
| --- | --- |
| Owner | CUDA-Q Algorithms Team |
| Public symbols and import paths | from `cudaq_algorithms.stateprep`: `get_uccsd_excitations`, `get_num_uccsd_parameters`, `make_uccsd_operator_pool`, `make_uccgsd_operator_pool`, `make_upccgsd_operator_pool`, `make_ceo_operator_pool`, `get_uccgsd_pauli_lists`, `get_upccgsd_pauli_lists`, `get_ceo_pauli_lists`. Reachable as `cudaq_algorithms.stateprep.<name>`; the package root re-exports the `stateprep` module, not these names (`python/cudaq_algorithms/__init__.py:34`) |
| Source paths | `python/cudaq_algorithms/stateprep/_pools.py` (whole file); `python/cudaq_algorithms/stateprep/__init__.py:17-22, 43-47` for the error-type/CEO-unit notes and the exported set |
| Authoritative tests | `tests/python/test_operator_pools.py` (counts, absolute content pins, fermionic-generator bijections); `tests/python/test_stateprep.py:8-82` (enumeration values, converter shapes) and `:135-181` (kernel smoke tests); `tests/python/test_stateprep_kernels.py:211-255` (pool-versus-circuit dense agreement) and `:263-291` (rejection behavior) |
| Authoritative documentation | `docs/sphinx/guide/state_prep.rst` ("Ansatz kernels and operator pools", "Package conventions"); `docs/sphinx/conventions.rst`; example `docs/sphinx/examples/python/hartree_fock_ucc.py` |
| Package/CUDA-Q versions verified | **unverified.** Declared environment is Python `>=3.11` with `cudaq >= 0.15.0, < 0.16` (`pyproject.toml`) and CUDA-Q pinned at `b28cf0f6f2d9387f12ca81b6824b8833a38530af` (`.cudaq_version`); nothing was executed for this record |
| Commit/date last verified | `61ac072d7823481ad0d303dac84d8ee29a9a8cd0` (2026-09-03). `_pools.py` itself last changed at `1c737cf42d67234a0da6e230c19e8edf334fd901` (2026-08-11) and `test_operator_pools.py` at `0f223ed3149c6df5c6668f4b11d2666838b5da1a` (2026-08-12) |
| Lifecycle | draft |
| Replacement and migration notes | none; no symbol here is marked deprecated at the cited commit. The module docstring describes the file as a pure-Python port of `lib/stateprep/excitations.cpp` and `lib/stateprep/device/*.cpp`, which **do not exist in the repository at this commit**, so "orders, index conventions, and coefficient signs match the C++ implementation exactly" is `unverified` as a cross-implementation claim. Every order and sign stated below is `derived` from the Python source and, where noted, pinned by a committed test |

**Evidence rule.** Every test named below is *cited repository evidence* — a
committed assertion read at the cited commit, labeled `derived` — never a fresh
measurement in this session.

## Classification

| Dimension | Value |
| --- | --- |
| Operation + mathematical object (primary identity) | **preprocess** + **fermionic-excitation operator pool**. `synthesize` is a defensible alternative reading of the same functions; `preprocess` is chosen because the output is host data consumed by a later kernel or converter. The alternative is recorded, not routed on, and the final choice is an **open** owner decision |
| Kind | classical transformation (host). No kernel, no measurement, no simulator dependency |
| Routine role | computational — each pool builder solves one distinct, independently usable task. Role follows problem completeness, not execution location (`architecture.md`), so running on the host does not make these auxiliaries |
| Abstraction level | leaf operation |
| Parameterization | none. The pools carry no variational parameter; "one parameter per pool operator" is a property of the *consumer* kernels, and this record only fixes the order those parameters are matched against |
| Execution layers | host preprocessing only |
| Input representations | integer problem sizes (`num_qubits` in spin orbitals, or `num_orbitals` in **spatial** orbitals for CEO; `num_electrons`; `spin`) plus boolean subset switches |
| Output representations | an ordered `list` of `cudaq.SpinOperator`; for UCCSD also five lists of excitation index lists; from the three converters, grouped Pauli words and coefficients |
| Domain | `quantum-chemistry` — a tag, not a parallel taxonomy. The emitted objects are ordinary Pauli operators and carry no chemistry-specific type |
| Required dependencies | `cudaq` only (`cudaq.spin`, `cudaq.SpinOperator`, `cudaq.pauli_word`). No NumPy or SciPy on this path (`_pools.py:22-23`) |
| Optional dependencies | none |
| Exactness | exact. The enumerations are closed-form loops and every coefficient is a dyadic rational (`0.5`, `0.25`, `0.125`) that is exactly representable in binary floating point |
| Uncertainty | deterministic |
| Method | direct |

The last three rows are provisional metadata: record the value, never route on
it.

## Scientific contract

- Purpose: enumerate a fixed, ordered family of excitation operators for a given
  system size and, for UCCSD, a given electron count and spin, as Hermitian
  Pauli sums ready to be exponentiated by a consumer kernel.
- Approximation controls: none. Nothing is truncated, screened, or approximated;
  the only selection knobs are the boolean subset switches per pool.

### What a pool operator is

Notation, all `derived` from `_pools.py`:

| Symbol | Meaning |
| --- | --- |
| `M` | number of **spatial** orbitals |
| `n` | number of **spin orbitals** = qubits; `n = 2M` wherever a spin layout applies |
| `a_p`, `a_p^dagger` | Jordan-Wigner annihilation/creation operator on spin orbital `p` |
| `T` | a fermionic excitation operator, e.g. `a_p^dagger a_q` (single) or `a_p^dagger a_q^dagger a_r a_s` (double) |
| `G` | a pool operator, one element of the returned list |
| `Z(l, h)` | the parity string `prod_{l < i < h} Z_i`, empty when `h <= l + 1` (`_pools.py:45-51`) |

Every UCC-family pool operator is a **Hermitian** Pauli sum, not the
anti-Hermitian fermionic generator itself. The relationship established by the
repository's own independent oracle is

```text
G = (+/- i) * (T - T^dagger)
```

with the overall sign left to the construction. The bijection test asserts
exactly this and is explicitly written to be "agnostic to the arbitrary global
sign convention" (`tests/python/test_operator_pools.py:189-202`), so **the
per-operator overall sign is pinned absolutely only where an absolute
word/coefficient list is asserted** — UCCSD at `(4, 2, 0)`
(`:92-127`) and CEO at `num_orbitals = 2` (`:255-273`). For the other cited
cases the sign is `derived` from source arithmetic but not independently
oracle-checked.

Consumers exponentiate a pool operator as `exp(+i * theta * c * P)` per Pauli
term `c * P` of `G`, which is where the Hermitian convention pays off. That
exponent convention, and the fact that the `uccsd` device-kernel circuit does
**not** follow it, are the consumer record's contract, not this one.

### Provider A — UCCSD excitations and pool

`get_uccsd_excitations(num_qubits, num_electrons, spin=0)` partitions the
interleaved layout into occupied and virtual sets and returns the five groups
`(singles_alpha, singles_beta, doubles_mixed, doubles_alpha, doubles_beta)`
(`_pools.py:69-140`):

- `spin > 0`: `n_occ_beta = (num_electrons - spin) // 2`,
  `n_occ_alpha = num_electrons - n_occ_beta`; occupied alpha `2i`, virtual alpha
  `2j + 2*n_occ_alpha`, occupied beta `2i + 1`, virtual beta
  `2j + 2*n_occ_beta + 1` (`:89-101`).
- `spin == 0` with even `num_electrons`: `n_occ = num_electrons // 2`, and the
  virtual offsets reduce to `2j + num_electrons` and `2j + num_electrons + 1`
  (`:102-108`).
- Group order, which **is** the contract: singles alpha, singles beta, mixed
  doubles, alpha doubles, beta doubles (`:114-140`). Within a group, indices run
  in nested-loop order over occupied then virtual; same-spin doubles enumerate
  strictly ascending occupied pairs and strictly ascending virtual pairs.
- Mixed doubles are emitted as `[p_alpha_occ, q_beta_occ, r_beta_virt,
  s_alpha_virt]` — occupied alpha, occupied beta, **virtual beta**, virtual
  alpha (`:116-118`). The alpha/beta order is not the same on both sides of the
  excitation; reading it as `(occ, occ, virt_alpha, virt_beta)` silently permutes
  every mixed double.

`get_num_uccsd_parameters` is the sum of the five group lengths (`:143-147`).

`make_uccsd_operator_pool(num_qubits, num_electrons, spin=0)` emits one operator
per excitation in exactly that order (`:186-201`):

- single `(p, q)` -> `0.5 * Y_p Z(p,q) X_q - 0.5 * X_p Z(p,q) Y_q`
  (`:150-154`);
- double `(p, q, r, s)` -> `0.125 *` an eight-term `XXXY`-family sum, after the
  indices are canonicalized to `(i_occ < j_occ, a_virt, b_virt)` by the four
  index-pattern branches at `:159-166`, with parity strings `Z(i_occ, j_occ)`
  and `Z(a_virt, b_virt)` (`:157-183`). The four branches reorder indices; the
  **pool** applies no sign flip of its own in any branch.

Term coefficients are therefore `+/-0.5` for singles and `+/-0.125` for doubles,
asserted at `tests/python/test_operator_pools.py:28-39`.

### Provider B — UCCGSD (generalized) pool

`make_uccgsd_operator_pool(num_qubits, only_singles=False, only_doubles=False)`
(`_pools.py:253-265`) ignores electron count and occupancy entirely: it is
"generalized" in the sense that every qubit pair and quadruple is a candidate.

- Singles: all `(p, q)` with `p > q` over the full register, in ascending `p`
  then ascending `q` (`:209-210`), each mapped to
  `0.5 * Y_q Z(q,p) X_p - 0.5 * X_q Z(q,p) Y_p` (`:227-231`).
- Doubles: for every 4-combination `a < b < c < d`, the **three** pairings
  `((a,b),(c,d))`, `((a,c),(b,d))`, `((a,d),(b,c))`; each pair is normalized to
  `(high, low)` and the pair-of-pairs to `(min, max)`, then collected in a `set`
  and returned `sorted` (`:213-224`). **Pool order is the sorted order of that
  normalized nested tuple, not the `a<b<c<d` enumeration order.** Each entry
  becomes `0.125 *` an eight-term generalized double generator with parity
  strings `Z(q,p)` and `Z(s,r)` (`:234-250`), called with `p > q` and `r > s`
  (`:264`).
- The `set` removes nothing at the sizes covered by tests: a pair-of-pairs
  determines its 4-subset, and the three pairings of one subset are distinct, so
  the count is exactly `C(n,2) + 3*C(n,4)` — matching the asserted `9` at
  `n = 4` and `238` at `n = 8` (`tests/python/test_operator_pools.py:60-69`).
  The `set` therefore functions as an ordering device.
- Singles here are **not** spin-restricted: pairs with mixed parity, i.e.
  alpha-beta, are included. This pool does not preserve the alpha/beta electron
  counts.
- `num_qubits` need not be even; nothing in this provider imposes a spin layout.
  Interpreting the indices as interleaved spin orbitals is the caller's choice.
- Both switches true returns an **empty list** without raising (`:259-264`).

### Provider C — UpCCGSD (paired) pool

`make_upccgsd_operator_pool(num_qubits, only_doubles=False)`
(`_pools.py:273-288`) requires an even `num_qubits` and restricts Provider B's
building blocks:

- Singles: the same `p > q` enumeration, filtered to `p % 2 == q % 2` — i.e.
  spin-preserving, alpha-with-alpha and beta-with-beta (`:281-283`).
- Doubles: one **paired** operator per spatial-orbital pair `p < q`, built as
  `_uccgsd_double(2q+1, 2q, 2p+1, 2p)` — both electrons moved between complete
  spatial orbitals (`:284-287`). This is the `k`-UpCCGSD pairing, named as such
  by `tests/python/test_operator_pools.py:231-252`.
- Count `3 * C(num_qubits/2, 2)` overall and `C(num_qubits/2, 2)` for
  `only_doubles=True`, asserted as `135` and `45` at `num_qubits = 20`
  (`:72-79`) and `3` and `1` at `num_qubits = 4`
  (`tests/python/test_stateprep.py:60-61`).
- `only_doubles=True` suppresses only the singles; there is no `only_singles`
  switch on this provider, and the paired doubles are always emitted.

### Provider D — CEO (coupled exchange operator) pool

`make_ceo_operator_pool(num_orbitals)` (`_pools.py:342-357`) follows
arXiv:2407.08696 as cited in the module docstring (`:16-17`) and differs from
A-C in three ways that are easy to get wrong:

1. **Its argument is a count of spatial orbitals**, so the pool acts on
   `2 * num_orbitals` qubits. Everything else in the package counts spin
   orbitals (`stateprep/__init__.py:20-22`; `docs/sphinx/guide/state_prep.rst`,
   "CEO orbital counts"). `get_ceo_pauli_lists` builds words at
   `2 * num_orbitals` width for this reason (`_pools.py:391-395`).
2. **It carries no Jordan-Wigner parity string.** The single is
   `0.5 * Y_q X_p - 0.5 * X_q Y_p` with no `Z` interior (`:323-325`), and the
   doubles are plain four-Pauli products (`:328-339`). The repository calls this
   "a deliberately non-fermionic coupled-exchange construction ... so it has no
   simpler physical reference" (`tests/python/test_operator_pools.py:255-259`).
   Do not describe a CEO operator as a Jordan-Wigner fermionic excitation, and
   do not expect the fermionic-generator bijection that pins Providers A-C.
3. **Each double contributes two operators**, `0.25 * op_a` and `0.25 * op_b`,
   appended consecutively (`:328-339`, `:350-355`), so the pool length is not
   the number of index quadruples and each of the two receives its own consumer
   parameter.

Emission order is: alpha singles, beta singles, alpha same-spin doubles, beta
same-spin doubles, mixed doubles (`:346-355`), where

- singles are `(2i + offset, 2j + offset)` with `j < i`, offset `0` alpha and
  `1` beta (`:296-299`);
- same-spin doubles take each descending 4-combination `i > j > k > l` of
  spatial orbitals and emit the three pairings `(p,q,r,s)`, `(p,r,q,s)`,
  `(q,p,r,s)` (`:302-314`);
- mixed doubles are `(2i, 2j+1, 2k, 2l+1)` with `k < i` and `l < j`, i.e.
  alpha-and-beta pairs with `p > r` and `q > s` (`:317-321`).

Counts follow as `2*C(M,2) + 12*C(M,4) + 2*C(M,2)^2`, matching the asserted `4`
at `M = 2` and `96` at `M = 4` (`tests/python/test_operator_pools.py:82-89`).
Small sizes degrade quietly rather than raising: `M <= 1` yields an **empty
pool**, and same-spin doubles first appear at `M = 4`
(`tests/python/test_stateprep_kernels.py:241-243`).

### Why and when to use

- Use a pool when the task needs the *operator content and order* of a UCC-style
  excitation family: to feed one of the packaged consumers, to convert to
  grouped Pauli data, or to build an independent dense reference from the same
  generators.
- Use `get_uccsd_excitations` when the task needs the **index tuples** or the
  parameter order of the `uccsd` device kernel, whose parameter order is fixed
  by this enumeration (`docs/sphinx/guide/state_prep.rst`;
  `python/cudaq_algorithms/stateprep/_kernels.py:449-460`).
- Do not use a pool as a Hamiltonian, an observable, or a block-encoded
  operator: these are excitation generators intended for exponentiation, and no
  packaged encoding consumes them.
- Do not use Provider B or C to hold electron number or spin fixed by
  construction: only Provider A is built from an occupied/virtual partition, and
  only Provider C restricts singles to one spin.
- Do not use a pool to decide which excitations matter — that is the excluded
  selection/optimization workflow.

## Inputs

| Field | Contract |
| --- | --- |
| Arguments | `get_uccsd_excitations(num_qubits, num_electrons, spin=0)`; `get_num_uccsd_parameters(num_qubits, num_electrons, spin=0)`; `make_uccsd_operator_pool(num_qubits, num_electrons, spin=0)`; `make_uccgsd_operator_pool(num_qubits, only_singles=False, only_doubles=False)`; `make_upccgsd_operator_pool(num_qubits, only_doubles=False)`; `make_ceo_operator_pool(num_orbitals)`; `get_uccgsd_pauli_lists(num_qubits, only_singles=False, only_doubles=False)`; `get_upccgsd_pauli_lists(num_qubits, only_doubles=False)`; `get_ceo_pauli_lists(num_orbitals)` |
| Shapes/ranks | scalars only; the subset switches are positional-or-keyword booleans |
| Dtypes/domains | every count must be integral and non-negative. `_as_count` rejects `bool` explicitly (`True` is not a qubit count) and rejects any value where `int(value) != value`, so `2.5` raises instead of truncating to `2` (`_pools.py:30-42`) |
| Units | `num_qubits` and `num_electrons` count **spin orbitals** / electrons; `spin` is `2 * S_z` (`conventions.md`); `num_orbitals` in the CEO helpers counts **spatial** orbitals |
| Ordering/layout | interleaved spin orbitals, alpha even and beta odd (`conventions.md`). Providers B and C read raw qubit indices; Provider D derives indices from spatial orbitals |
| Normalization | not applicable; no input is normalized |
| Required mathematical properties | `num_qubits` even for Providers A and C; `(num_electrons, spin)` must fit the register for Provider A |

### Validation and rejection behavior

Validation is host-side and eager, in the order below. Two legacy cases raise
`RuntimeError`; every guard added in the pure-Python implementation raises
`ValueError` (`stateprep/__init__.py:17-20`;
`docs/sphinx/guide/state_prep.rst`, "Error types"). Do not write `except
ValueError` around Provider A and expect it to catch everything.

| # | Provider | Rejected condition | Type and message fragment |
| --- | --- | --- | --- |
| 1 | all | a count that is `bool`, fractional, or negative | `ValueError`, "must be a non-negative integer" (`_pools.py:38`, `:41`) |
| 2 | A | odd `num_qubits` | **`RuntimeError`**, "The total number of qubits should be even." (`:79-80`) |
| 3 | A | `num_electrons > num_qubits` | `ValueError`, "num_electrons cannot exceed num_qubits" (`:83-84`) |
| 4 | A | `spin > num_electrons` | `ValueError`, "spin cannot exceed num_electrons" (`:85-86`) |
| 5 | A | `spin > 0` and the resulting alpha count exceeds `num_qubits // 2` | `ValueError`, "does not fit in num_qubits spin orbitals" (`:91-95`) |
| 6 | A | odd `num_electrons` with `spin == 0` | **`RuntimeError`**, "Incorrect spin multiplicity ..." (`:109-112`) |
| 7 | C | odd `num_qubits` | `ValueError`, "expects an even number of spin orbitals." (`:276-278`) |

These are asserted at `tests/python/test_stateprep_kernels.py:263-291`, which
covers all seven rows including the `bool` and fractional cases.

**Silent-acceptance gap worth knowing (`derived`).** Provider A has **no parity
guard** on `(num_electrons - spin)`: the grep-verifiable set of `raise`
statements in `get_uccsd_excitations` is rows 2-6 above, and the `spin > 0`
branch takes the floor `(num_electrons - spin) // 2` unconditionally
(`_pools.py:89-101`). So an odd `(num_electrons - spin)` is **accepted** and
silently realizes `spin + 1`: at `(8, 4, 1)` the partition is
`n_occ_beta = 1, n_occ_alpha = 3`, identical to `(8, 4, 2)`, and since
everything downstream depends only on those two counts and `num_spatial`, the
excitation lists, parameter count, and pool are the same as at `spin = 2`. The
Hartree-Fock occupation builder **rejects** the same input with
`ValueError("(num_electrons - spin) must be even when spin > 0 (spin is
2*S_z)")` (`_hartree_fock.py:104-111`, whose comment says the guard "Matches
get_uccsd_excitations" — that describes the shared floor arithmetic, not a
shared guard). Consequence: `make_uccsd_operator_pool(8, 4, 1)` succeeds while
`hartree_fock_ucc_kernel(..., num_electrons=4, spin=1)` raises, so a caller who
builds the pool first sees the failure only at the later factory call. Report the
mismatch; do not describe `spin = 1` as supported by this pool.

## Outputs

| Field | Contract |
| --- | --- |
| Return type | `get_uccsd_excitations`: a 5-tuple of `list[list[int]]`. `get_num_uccsd_parameters`: `int`. `make_*_operator_pool`: `list` of `cudaq.SpinOperator`-compatible spin-operator expressions. `get_*_pauli_lists`: `(list[list[cudaq.pauli_word]], list[list[float]])` |
| Mathematical meaning | each list element is one Hermitian Pauli-sum generator `G` as defined above; the enclosing list order is the pool order that a consumer's parameter list is matched against, position by position |
| Shape/register geometry | operators span at most the full width; the shared helper asserting `op.qubit_count <= num_qubits` is applied to all four pool families (`tests/python/test_operator_pools.py:16-18`, used at `:21-89` and `:276-291`), and an operator acting on a strict subset of qubits is normal. Widths are made explicit only by the converters, which pad every word to full width |
| Normalization, sign, and phase | coefficients are real dyadic rationals; the overall sign of each `G` relative to `i * (T - T^dagger)` is a construction convention, absolutely pinned only for the two cases named above |
| Observable or measurement interpretation | none. A pool operator is a generator to exponentiate, not an observable the library measures |
| Error/status information | exceptions only; there is no status object and no partial-success mode |

### The three pool-specific Pauli-list converters

`get_uccgsd_pauli_lists`, `get_upccgsd_pauli_lists`, and `get_ceo_pauli_lists`
build their pool and flatten it through `_pauli_lists_from_pool`
(`_pools.py:364-395`): one group per pool operator, in pool order; words built at
full width via `term.get_pauli_word(num_qubits)` so identity padding is
explicit; group lengths equal between words and coefficients
(`tests/python/test_stateprep.py:41-82`).

They are **not** interchangeable with the generic
`get_fixed_parameter_ucc_pauli_lists`, which lives in the consumer record. Two
differences, both `derived`:

| Behavior | The three converters here | `get_fixed_parameter_ucc_pauli_lists` |
| --- | --- | --- |
| a term whose coefficient has an imaginary part | takes `float(coefficient.real)` and **silently discards the imaginary part** (`_pools.py:373`) | raises `ValueError("only real operator-pool coefficients are supported")` |
| a negligible-coefficient term | kept, so every group has exactly its operator's term count | dropped when `abs(real) <= coefficient_tolerance`, so a group may be shorter or empty |

Whether any packaged pool can actually produce an imaginary coefficient is
**unverified**: for every case with an independent oracle the operator is
Hermitian, which forces real Pauli coefficients, and no source or test
characterizes an overlapping-parity case that breaks that. Treat the `.real`
truncation as an unguarded path, not as evidence that imaginary coefficients
occur. If a caller needs the rejecting behavior, use the generic converter and
read its contract in [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md).

## Capabilities and composition

- Stable ID: **none minted.** No capability identifier is created for this
  family. What crosses the boundary is an *object*, so the composition contract
  is the Representation Record below; `architecture.md` reserves capability
  records for a reusable *semantic behavior* demonstrated by independent
  producers and consumers.
- Direction: this family provides no capability ID and requires none. It does
  not consume `cudaq-algorithms.state-preparation.unitary.v1`; it produces data
  that a provider of that capability may be built from.
- `Deferred:` minting a pool-exchange capability ID — trigger: a consumer
  outside the `stateprep` namespace, plus an explicit owner decision on the
  identifier, since the existing capability ID's own spelling is already an open
  question in [state-preparation.md](state-preparation.md).

Composition checks that actually apply, all `derived`:

1. **Width agreement.** A pool built at `num_qubits` and a converter or kernel
   run at a different width silently disagree; only the converters record a
   width, and they record the one they were given.
2. **CEO unit conversion.** A CEO pool built at `num_orbitals` composes with a
   register of `2 * num_orbitals` qubits. Passing a qubit count where a spatial
   count is expected doubles the intended system.
3. **Positional parameter matching.** Consumers pair `parameters[g]` with pool
   element `g`; there is no name, index, or excitation label carried by the pool,
   so a reordering is undetectable at the boundary.
4. **Reference determinant.** The pool-consuming device kernels are applied on
   top of an already-prepared determinant; on `|0...0>` the source calls the
   result "physically meaningless"
   (`python/cudaq_algorithms/stateprep/_kernels.py:571-579`). Pool order does
   not fix that — the consumer record does.

Known consumers at the cited commit, source-verified:
`get_fixed_parameter_ucc_pauli_lists` and `hartree_fock_ucc_kernel`
(`_hartree_fock.py`); the `uccgsd`, `upccgsd`, `ceo`, and `fixed_parameter_ucc`
device kernels (`_kernels.py:529-586`); the `uccsd` device kernel, which
re-derives the same enumeration inline rather than calling it
(`_kernels.py:446-526`). Anything else — including ADAPT-style selection loops
named in the guide — is a caller-side workflow this record does not cover.

## Composite protocol

Not applicable: `Abstraction level` is `leaf operation`; the template omits the
heading.

## Accuracy and limitations

| Field | Statement |
| --- | --- |
| Error behavior or bounds | no approximation and no floating-point error in the enumeration; all coefficients are exactly representable dyadic rationals |
| Precision sensitivity | none on this path — it is integer and exact-rational host code. Precision enters only when a consumer exponentiates the operators, or when a dense oracle is built for comparison |
| Unsupported inputs | the seven rejected conditions tabulated above |
| Known implementation limitation 1 | Provider A accepts an odd `(num_electrons - spin)` and silently realizes `spin + 1`, as documented under "Silent-acceptance gap" |
| Known implementation limitation 2 | empty results are returned without warning: `make_uccgsd_operator_pool(n, True, True)`, a CEO pool at `num_orbitals <= 1`, and any size whose excitation groups are all empty yield `[]`. A downstream kernel then applies nothing, and the degenerate `pass`-bodied preparation shape is a legal outcome (`_hartree_fock.py`, "hf_only"/empty shapes) |
| Known implementation limitation 3 | the three converters take the real part of every coefficient without checking the imaginary part (`_pools.py:373`), unlike the generic converter |
| Unsupported versus unverified | **unsupported/absent:** any notion of pool truncation, screening, ranking, deduplication across pools, or spin-symmetry adaptation beyond what each provider enumerates; no such symbol exists. **unverified:** equivalence to the named C++ implementation (absent from the repository); reachability of an imaginary Pauli coefficient; per-operator overall sign for every case outside the two absolute pins; behavior of Provider B when its indices are *interpreted* as an interleaved spin layout, which no source or test addresses |

## Resources

There is **no resource estimator for pools.** `estimate_fixed_parameter_ucc_resources`
counts groups and Pauli rotations of already-grouped data and belongs to
[state-preparation-hf-ucc.md](state-preparation-hf-ucc.md).

What this family does fix are the two counts that drive every downstream
estimate — the number of pool operators (hence consumer parameters) and the
number of Pauli terms per operator. Both are `derived` from the enumeration
loops, and each formula below is corroborated by a cited count assertion in
`tests/python/test_operator_pools.py`; they are combinatorial host quantities,
**not** logical gate counts, transpiled gate counts, depth, runtime, or memory
(`conventions.md`).

| Provider | Number of pool operators | Corroborating assertion |
| --- | --- | --- |
| A (UCCSD) | `n_occ_a*n_virt_a + n_occ_b*n_virt_b + n_occ_a*n_occ_b*n_virt_b*n_virt_a + C(n_occ_a,2)*C(n_virt_a,2) + C(n_occ_b,2)*C(n_virt_b,2)` | `3` at `(4,2,0)`, `8` at `(6,3,1)`, `875` at `(20,10,0)` (`:21-57`) |
| B (UCCGSD) | `C(n,2) + 3*C(n,4)`; `C(n,2)` singles-only; `3*C(n,4)` doubles-only | `9`/`6`/`3` at `n=4`, `238` at `n=8` (`:60-69`) |
| C (UpCCGSD) | `3*C(n/2,2)`; `C(n/2,2)` doubles-only | `135`/`45` at `n=20` (`:72-79`) |
| D (CEO) | `2*C(M,2) + 12*C(M,4) + 2*C(M,2)^2` | `4` at `M=2`, `96` at `M=4` (`:82-89`) |

Terms per operator: `2` for every single and `8` for every UCC-family double
(`_pools.py:150-154`, `:157-183`, `:227-250`); `4` for every CEO double
(`:328-339`). Asserted for `n = 4` as group sizes
`[2,2,2,2,2,2,8,8,8]` (UCCGSD), `[2,2,8]` (UpCCGSD), and `[2,2,4,4]` (CEO at
`M = 2`) in `tests/python/test_stateprep.py:41-69`.

`Deferred:` the per-quantity resource contract (metric, unit, abstraction level,
execution assumptions, exact/bounded/estimated status, confidence, composition
rule), until the resource-metric vocabulary is agreed — the same deferral the
consumer record carries.

## Validation

| Field | Evidence |
| --- | --- |
| Independent oracles | two kinds, both committed in the repository and cited rather than executed. (1) **Absolute content pins**: exact expected Pauli words and coefficients for UCCSD at `(4,2,0)` (`tests/python/test_operator_pools.py:92-127`) and CEO at `M=2` (`:255-273`); these pin words, coefficients, and signs with no shared machinery. (2) **An independent dense fermionic construction**: Jordan-Wigner ladder operators built from `Z`/lowering Kronecker products, forming `T - T^dagger`, compared as a bijection against the pool (`:141-202`) |
| Invariants | every pool operator matches exactly one independent fermionic generator and every generator is produced — a full bijection, not a one-sided containment (`:189-202`); `op.qubit_count <= num_qubits` for every pool (`:16-18`); converter word groups and coefficient groups have equal lengths (`tests/python/test_stateprep.py:41-69`) |
| Representative cases | UCCGSD and UpCCGSD content at `n = 4`, including the `only_singles` / `only_doubles` paths that are otherwise only count-checked (`:205-252`); UCCSD content at `(8,4,2)`, whose interleaved mixed doubles are the hardest index case (`:276-291`); CEO at `M = 2`; counts up to `n = 20` and `M = 4` |
| Convention translation | the dense oracle builds words with `word[0]` as qubit 0 and reverses the Kronecker order accordingly (`:153`, `:160-169`) — the translation `conventions.md` prescribes |
| Predeclared tolerances | `atol=1e-10` in the bijection comparison (`:196-199`); the absolute content pins use exact equality on words and coefficients. The pool-versus-circuit agreement tests in `tests/python/test_stateprep_kernels.py:127-131` instead select `1e-12` (fp64) or `5e-5` (fp32) from the active simulator precision |
| Expected failure/adversarial case | the seven rejection rows, asserted at `tests/python/test_stateprep_kernels.py:263-291`; the test file also records *why* the independent pins exist — the kernel tests derive their dense reference from the same `make_*_operator_pool` they exercise, so "a count-preserving error in a pool's Pauli words, signs, or index conventions would pass on both sides" (`tests/python/test_operator_pools.py:130-139`) |
| Known validation gap | the bijection is deliberately agnostic to the overall `+/- i` factor, so per-operator sign is oracle-pinned only by the two absolute-content tests. CEO is non-fermionic and has no generator oracle at all — only the `M = 2` absolute pin plus dense-exponential agreement with its own kernel. Provider B and C at sizes above `n = 4` are count-checked only |
| Evidence status per claim | `derived` from the cited source line or committed assertion, unless labeled `assumed` or `unverified` in place. Nothing here is `measured` |

## Evaluation coverage

Declared coverage: `operator-pool-selection-boundary` in `evals/evals.json`
tests UCCSD-versus-UCCGSD selection, ordered pool outputs, shared conventions,
and the excluded parameter-optimization workflow. It has not been run with or
without the skill, so no uplift is claimed.

Other existing cases use this record as supporting context:

| Existing case id | Sections that support it |
| --- | --- |
| `state-preparation-ucc-parameterization-boundary` | "What a pool operator is" (Hermitian generator, `+/- i` relation) and Capabilities → positional parameter matching, as the pool-side half of a question whose amplitude-convention half lives in [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md) |
| `state-preparation-provider-selection` | Provider A-D subsections, when the requested object is an excitation family rather than a prepared state |

Additional coverage opportunities include a focused CEO spatial-orbital
argument case and a boundary case on odd `(num_electrons - spin)` acceptance.

## External alignment

- Literature conventions: the CEO construction cites
  https://arxiv.org/abs/2407.08696 in the module docstring (`_pools.py:17-18`);
  the paper itself was not consulted for this record, so agreement with it is
  `unverified`. UCCSD, UCCGSD, and UpCCGSD carry no literature citation in
  source; the `k`-UpCCGSD naming of "spin-preserving singles plus paired
  doubles" comes from `tests/python/test_operator_pools.py:235-236`.
- Cross-implementation alignment: the claim that orders, index conventions, and
  coefficient signs match a C++ implementation is `unverified` here, because
  `lib/stateprep/` is absent from the repository at this commit.
- External package translations: `Deferred:` until a task needs one and it can be
  checked against that package's own documentation. Nothing in the repository
  translates these pools to an external chemistry package, and the overall
  `+/- i` sign convention plus the interleaved layout are exactly what such a
  translation would have to fix first.
- Known semantic differences: inside this repository, Provider A's pool applies
  no per-branch sign while the `uccsd` device-kernel circuit does; that
  comparison is contracted in
  [state-preparation-hf-ucc.md](state-preparation-hf-ucc.md) and is not restated
  here.

---

# Representation Record — the operator pool

- Object name and canonical symbol: an **operator pool** — the ordered `list`
  returned by any `make_*_operator_pool` function.
- Public type or structural form, and source path: a Python `list` of spin
  operators, each accepted by `cudaq.SpinOperator(op)` and iterable as terms
  exposing `get_pauli_word(width)` and `evaluate_coefficient()`
  (`python/cudaq_algorithms/stateprep/_pools.py:364-376`). There is no named
  public pool type, no wrapper class, and no carried width.
- Mathematical meaning: element `g` is a Hermitian Pauli sum `G_g` equal to
  `(+/- i) * (T_g - T_g^dagger)` for the excitation `T_g` of that provider's
  enumeration; CEO is the documented exception, being a non-fermionic
  coupled-exchange construction.
- Shape, layout, ordering, dtype, and units: list order **is** the contract, and
  it is the per-provider order documented above. Qubit indices follow the
  interleaved layout with qubit 0 least significant (`conventions.md`).
  Coefficients are real dyadic rationals. The object carries no width, no spin
  metadata, and no excitation labels.
- Normalization, sign, and phase convention: no normalization; per-operator
  overall sign is a construction convention, absolutely pinned only for UCCSD at
  `(4,2,0)` and CEO at `M = 2`.
- Required mathematical properties (applicability preconditions): the consumer
  must supply the width the pool was built for, one parameter per element in
  order, and — for the pool-consuming device kernels — a register already
  holding a reference determinant.
- Producers (>=2 required to justify this record): `make_uccsd_operator_pool`,
  `make_uccgsd_operator_pool`, `make_upccgsd_operator_pool`,
  `make_ceo_operator_pool`; plus any caller-built list of spin operators, which
  `get_fixed_parameter_ucc_pauli_lists` accepts by iterating it.
- Consumers: the three converters in `_pools.py:379-395`;
  `get_fixed_parameter_ucc_pauli_lists` and, through it,
  `hartree_fock_ucc_kernel` (`_hartree_fock.py:161-193`, `:240-320`); the dense
  reference constructions in `tests/python/test_operator_pools.py`,
  `tests/python/test_stateprep_kernels.py:98-120`, and
  `docs/sphinx/examples/python/hartree_fock_ucc.py:38-41`.
- Invariants preserved across the boundary: order; Hermiticity of each element
  (hence real Pauli coefficients) for every case with an independent oracle;
  index layout. Nothing else travels — the width the pool was built for is not
  recoverable from the object, since `qubit_count` reflects only the highest
  index actually touched.
- Observable symptom of a misinterpretation: a reordered or re-widthed pool
  produces a normalized, plausible state with wrong amplitudes and no error;
  treating a CEO `num_orbitals` as a qubit count doubles the system; treating a
  pool element as anti-Hermitian introduces a spurious factor of `i` in a dense
  reference and makes an otherwise-correct comparison fail.
- Unsupported or ambiguous forms: a pool mixed from two providers or two widths
  (nothing rejects it, and no source or test covers it); a pool carrying complex
  coefficients (see the converter table); an empty pool, which is legal and
  silently produces a no-op product.
- Source paths, tests, docs, and last verification: "Identity and provenance"
  above.
