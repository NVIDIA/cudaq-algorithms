# Scientific conventions

Cross-source comparisons are valid only after their conventions are aligned.
The repository's `docs/sphinx/conventions.rst`, public source, and tests remain
authoritative. The records below are populated from those sources at commit
`61ac072d7823481ad0d303dac84d8ee29a9a8cd0` (2026-09-03). Nothing here was
executed in this session; every claim is `derived` from the cited path unless
labeled otherwise.

## Convention record

Each populated convention must state:

- concept and canonical symbol;
- CUDA-Q Algorithms convention;
- common alternatives;
- exact translation at the boundary;
- invariants unaffected by representation;
- observable symptom of a mismatch;
- minimal independent verification;
- source paths, tests, version range, and last verification.

## Qubit ordering: qubit 0 is least significant

- **Convention.** In statevectors and `SpinOperator.to_matrix()`, qubit 0 is
  the least significant bit of the computational-basis index. Basis state
  `|q_{n-1} ... q_1 q_0>` has index `sum_k q_k 2^k`.
- **Alternative.** Most papers write the leftmost tensor factor as qubit 0.
- **Translation.** Build dense references by reversing the factor order:
  `functools.reduce(np.kron, ops[::-1])`.
- **Mismatch symptom.** Amplitudes appear bit-reversed; single-qubit operators
  act on the wrong wire while norms stay correct.
- **Verification.** `cudaq.spin.z(0).to_matrix()` on one qubit is
  `diag(1, -1)`; on two qubits `spin.z(0)` acts on index bit 0.
- **Source.** `docs/sphinx/conventions.rst` ("Qubit ordering").

## Pauli words: string position equals qubit index

- **Convention.** Pauli words are strings over `IXYZ` with `word[0]` acting on
  qubit 0, so `"ZI"` means `Z` on qubit 0.
- **Alternative.** Left-to-right tensor-product notation, i.e. the reverse.
- **Translation.** Reverse the string at the boundary. Extract canonical terms
  with `term.get_pauli_word(width)` and `term.evaluate_coefficient()`, passing
  the intended qubit count as `width` so identity padding is explicit.
- **Mismatch symptom.** Correct spectra with wrong per-qubit locality; gapped
  operators silently shift.
- **Source.** `docs/sphinx/conventions.rst` ("Pauli words").

## Spin orbitals: interleaved, and `spin` is `2 * S_z`

- **Convention.** Spatial orbital `p` maps to spin orbitals `2p` (alpha, even)
  and `2p + 1` (beta, odd). All fermionic modules, pools, excitation
  enumerations, and Hartree-Fock occupations share this layout.
- **`spin` semantics.** `spin` is the alpha-minus-beta electron-count
  difference, i.e. twice the total `S_z`, with
  `n_beta = (num_electrons - spin) // 2` and `n_alpha = num_electrons -
  n_beta`. `spin == 0` selects closed-shell forms; `spin > 0` requires an even
  qubit count and produces interleaved open-shell occupations.
- **Alternative.** Blocked ordering, all alpha orbitals before all beta.
- **Translation.** Permute rows/indices between blocked and interleaved before
  comparing with external data.
- **Mismatch symptom.** An open-shell reference occupies contiguous orbitals
  (for example `{0,1,2,3}` instead of the interleaved `{0,1,2,4}`), so the
  determinant no longer matches an excitation pool built at the same spin.
- **Source.** `docs/sphinx/conventions.rst` ("Spin orbitals: interleaved");
  `docs/sphinx/guide/state_prep.rst` ("Package conventions").

## Kernel boundary: data erasure and register-only signatures

- **Convention.** Kernel factories return `@cudaq.kernel` objects whose
  signatures carry registers only; all problem data is captured inside the
  kernel at factory time. The state-preparation seam is the one-register form
  `(qubits: cudaq.qview)`.
- **Consequence.** A multi-argument device kernel is not itself injectable at
  a register-only seam, even when it implements the same physics. Wrapping it
  in a one-argument kernel is the caller's explicit act, not an equivalence.
- **Mismatch symptom.** A kernel that will not launch, or a consumer factory
  that silently keeps its `cudaq.State` argument mode.
- **Source.** `python/cudaq_algorithms/block_encoding.py` module docstring
  ("data erasure at the kernel boundary"); `docs/sphinx/guide/state_prep.rst`
  ("State preparation injection").

## Register ownership at the current state-preparation seam

- **Source-derived behavior (current implementation only).** Every packaged
  consumer factory that accepts `state_prep` allocates the system register
  itself, so the register arrives fresh in `|0...0>`; the preparation kernel
  receives only that register, runs before the consumer operation, and the
  consumer allocates ancilla, signal, or control registers afterwards. The
  preparation width must equal the consumer's system width exactly; this is
  documented as not verifiable at factory time, so a mismatch fails at launch
  rather than raising from the factory.
- **Scope limit.** This is a `derived` description of the current injection
  seam at the cited commit. It is **not** a decided capability-level policy.
  Whether the caller or the primitive should own system and ancilla registers
  in general, and the exact input-state, width, ancilla, inverse, control, and
  failure semantics of state preparation, remain **open** questions in the
  taxonomy design record, which lives outside this skill package. Do not
  present the current behavior as a library-wide guarantee for future
  capabilities, and do not present the open questions as settled.
- **Unverified.** Dirty (non-zero) input registers, width-mismatch error type
  and message, and foreign-consumer injection are not characterized by source
  or tests.
- **Source.** `python/cudaq_algorithms/pauli_lcu.py`,
  `python/cudaq_algorithms/qubitization.py`, `python/cudaq_algorithms/qsvt.py`,
  `python/cudaq_algorithms/trotter.py` (the `state_prep` call sites);
  `docs/sphinx/guide/state_prep.rst`, at the quoted prose "Contract (shared
  with the other primitives)" rather than a section heading.

## Validation happens on the host, before device kernels

- **Convention.** Device kernels have no error channel: an invalid argument is
  a silent no-op rather than an exception, and a kernel `return` is ignored by
  the compiler. Input validation therefore runs on the host at factory time,
  and the library exposes callable `validate_*` helpers for hand-built inputs.
- **Rule.** An illegal argument must never silently produce a result. Where a
  device kernel is a silent no-op on invalid input, record it as a known
  implementation limitation, not as established semantics.
- **Mismatch symptom.** A circuit that runs and returns a plausible but wrong
  state instead of raising.
- **Source.** `python/cudaq_algorithms/stateprep/_givens.py` module docstring
  and kernel guards; `docs/sphinx/guide/state_prep.rst`, at the quoted prose
  "Validation runs at factory time" rather than a section heading; public
  `validate_givens_rotation_schedule`,
  `validate_hartree_fock_occupation`, `validate_fixed_parameter_ucc`.

## Resource abstraction levels must not be conflated

- **Convention.** The library's `estimate_*_resources` helpers return counts of
  logical operations before transpilation. `GivensResourceEstimate`'s docstring
  is the strictest statement in source: its proxies are
  decomposition-independent **upper bounds**, explicitly *not* transpiled gate
  counts. The Hartree-Fock and fixed-parameter UCC estimator docstrings carry
  no such wording, so for those the bound comes from the Sphinx guide.
- **Rule.** Never present a proxy as a transpiled gate count, circuit depth
  after synthesis, runtime, or memory figure, and never compare quantities
  from different abstraction levels as though they were the same metric.
- **Mismatch symptom.** A resource comparison that changes conclusion when the
  transpiler or target changes.
- **Source.** `python/cudaq_algorithms/stateprep/_givens.py`
  (`estimate_givens_resources` docstring); `docs/sphinx/guide/state_prep.rst`
  (resource-estimate paragraphs).

## Register width is extent, not population

- **Convention.** For a Pauli operator, register width is one plus the largest
  touched target index, with explicit handling for identity-only operators. It
  is not the number of distinct touched qubits: an operator on qubits 0 and 3
  needs width 4, not 2.
- **Mismatch symptom.** Gapped Pauli terms are truncated or applied to the
  wrong-size register.
- **Source.** `python/cudaq_algorithms/common_kernels.py`
  (`_term_qubit_extent`); the block-encoding and Trotter family records.

## Block normalization and the good subspace

- **Convention.** A zero-flagged block encoding satisfies
  `(<0|_anc x I) U_A (|0>_anc x I) = H / alpha`. Packaged circuits allocate
  the system register first and ancillas after it, so the all-zero-ancilla
  amplitude block is the leading `2**num_system` entries.
- **Normalization.** Simulation helpers return that block unnormalized; its
  squared norm is the postselection probability. They do not implement a
  measurement or hardware postselection protocol.
- **Mismatch symptom.** A result has the right norm but wrong amplitudes,
  or a postselection probability is mistaken for a normalization factor.
- **Source.** `python/cudaq_algorithms/block_encoding.py`,
  `python/cudaq_algorithms/sim_utils.py`; [block-encoding.md](block-encoding.md).

## Qubitization walk sign and moment convention

- **Convention.** One packaged walk step block-encodes `-H / alpha`. The
  qubitization implementation accounts for that sign so `Walk.moment(s)`
  reports the documented positive Chebyshev convention
  `<T_k(H / alpha)>`; callers must not add another negation.
- **Mismatch symptom.** Odd moments have the opposite sign while even moments
  still agree.
- **Source.** `python/cudaq_algorithms/pauli_lcu.py`,
  `python/cudaq_algorithms/qubitization.py`; [qubitization.md](qubitization.md).

## QSP and QSVT phases

- **Convention.** `PhaseSequence` distinguishes `qsvt` projector phases from
  `qsp` phases. At execution, `qsp` phases are doubled to projector phases.
  Relative to the QSP signal model, the circuit carries the global phase
  `exp(i * sum(phases))`; recovery helpers remove it explicitly.
- **Rule.** Never compare raw phase lists or output vectors before translating
  the declared convention and accounting for global phase.
- **Mismatch symptom.** A uniformly phase-rotated response, or a polynomial
  response produced with doubled/halved phase angles.
- **Source.** `python/cudaq_algorithms/qsvt.py`;
  [qsvt.md](qsvt.md).

## Reflection gate versus reflection observable

- **Convention.** A reflection used as a circuit operation and the
  `SpinOperator` used for an expectation value are different interface
  objects even when they represent related mathematics. `Walk` uses circuit
  reflections in its kernels and observables for moments.
- **Mismatch symptom.** Passing an observable where a kernel factory is
  required, or describing an expectation value as a circuit application.
- **Source.** `python/cudaq_algorithms/common_kernels.py`,
  `python/cudaq_algorithms/qubitization.py`; [qubitization.md](qubitization.md).

## Precision, Hermiticity, and tolerances

- **Convention.** Establish dtype, target precision, and tolerance before a
  numerical comparison. Coefficient-pruning tolerances change the represented
  operator; comparison tolerances only judge outputs. A routine accepting
  non-Hermitian data must not be described as enforcing Hermiticity.
- **Rule.** Report global-phase-insensitive comparisons explicitly, and use
  invariant checks such as reconstructed operators, spectra, or expectation
  values when raw representations differ.
- **Source.** `docs/sphinx/conventions.rst`; fermion-transform, chemistry,
  QSVT, and validation family evidence linked from the catalog.

## Comparison discipline

Before comparing two implementations or sources:

1. identify both convention sets;
2. write the explicit translation;
3. select an invariant comparison when possible;
4. fix precision and tolerance;
5. label unresolved differences as unknown.

Prefer spectra, expectation values, reconstructed operators, known identities,
or other invariants over raw arrays when equivalent representations exist.
