# Pauli LCU block encoding

Operation + object: **encode** a **Pauli-sum operator as the
zero-flagged block of a unitary**.

## Identity and provenance

- Public symbols: `cudaq_algorithms.PauliLCU`,
  `cudaq_algorithms.select_observable`; module-level kernels remain under
  `cudaq_algorithms.pauli_lcu`.
- Source: `python/cudaq_algorithms/pauli_lcu.py` and shared kernels in
  `common_kernels.py`.
- Tests: `test_pauli_lcu.py`, `test_block_encoding_protocol.py`,
  `test_qubitization.py`, `test_qsvt.py`.
- Examples: `01_quickstart_block_encoding.py`, `pauli_lcu_demo.py`.
- Source provenance: current public source and tests are authoritative and must
  be rechecked at use time. [Source lookup](../source-provenance.md) gives shared current-source paths.
## Classification

- Identity: encode / Pauli-sum operator.
- Kind/role: quantum operation, computational leaf.
- Parameterization: construction-time.
- Layers: host validation/decomposition, kernel factory, device kernels.
- Input: `cudaq.SpinOperator`, one term, `{word: coefficient}` mapping, or
  iterable of `(coefficient, word)` pairs.
- Output: immutable encoding data plus protocol and convenience kernel
  factories.
- Exactness: exact for retained terms; threshold pruning is caller-controlled.
- Uncertainty/method/domain: deterministic, direct, domain-independent.

## Scientific contract

For retained real terms `H = sum_j a_j P_j`, `alpha = sum_j |a_j|` and
PREPARE/SELECT/UNPREPARE construct a unitary whose all-zero-ancilla block is
`H/alpha`. For positive-width all-`I` words, `include_identity=False` excludes
identity terms from the encoded operator while `constant_term` still reports
their sum. The zero-width edge below is outside that contract.

Use this for a Pauli decomposition that needs the shared block-encoding
boundary. Do not use it to certify a foreign encoding, infer a postselection
probability without a state, or encode unsupported complex coefficients.

## Inputs and rejection behavior

```python
from cudaq_algorithms import PauliLCU

encoding = PauliLCU(
    {"ZI": 0.75, "IX": -0.25},
    include_identity=True,
    coefficient_threshold=1.0e-12,
)
```

- The supported mapping/pair contract requires one common **positive** width
  and only `I/X/Y/Z`; position equals qubit index. Common width and characters
  are checked, but the current parser does not enforce positive width.
- Spin-operator width is the largest targeted qubit plus one; an explicit
  `num_qubits` may pad but cannot be smaller.
- Coefficients must be real within the package's fixed complex-noise tolerance.
- Terms below `coefficient_threshold` are dropped; an empty retained set is a
  `ValueError`.
- A string is not an iterable Hamiltonian and is rejected with `TypeError`.
- The number of ancillas is `max(1, ceil(log2(num_terms)))`; padding states
  have zero probability.

Known zero-width gap: a mapping/pair word `""` passes the source's length and
character loops, is not recognized as identity, and therefore gives incorrect
`include_identity`/`constant_term` semantics; later kernel behavior is
unverified. A scalar-only `cudaq.SpinOperator` without explicit `num_qubits`
reaches the same empty-word representation. Do not use either form. For a
scalar-only spin operator, pass a positive explicit `num_qubits`; for mapping
or pair input, provide an all-`I` word of the intended positive width.

## Outputs and composition

Inspection attributes include `num_system`, `num_ancilla`, `num_terms`,
`alpha`, `constant_term`, `terms`, and defensive-copy `kernel_args`.

- `encode_kernel()` returns `(state: cudaq.State)`; with a one-argument
  `state_prep`, it returns a zero-argument kernel.
- `walk_kernel(power, ...)` has the same input modes and yields the
  all-zero-ancilla block `T_power(-H/alpha)`.
- Protocol factories have the signatures in
  [block-encoding.md](block-encoding.md).
- `select_observable(encoding)` provides the LCU-specific odd-moment
  observable.

Provides `cudaq-algorithms.block-encoding.zero-flagged.v1` (provisional) and
optionally consumes `cudaq-algorithms.state-preparation.unitary.v1` through the
two convenience factories. The preparation seam is consumer-specific and is
not part of `BlockEncoding`.

## Accuracy and limitations

- The block contract is exact for retained terms, subject to floating-point
  angle synthesis.
- `coefficient_threshold` changes the encoded operator; it is not merely a
  performance hint.
- `alpha` is a normalization and spectral upper bound, not a success
  probability or resource estimate.
- Hardware support, controlled use of an injected preparation, and performance
  are unverified unless executed in the current task.

## Resources

No packaged resource estimator exists. Exact structural facts are register
width `num_system + num_ancilla`, one PREPARE and UNPREPARE around SELECT for
`apply`, and repeated walk subcircuits according to `power`. These are logical
subcircuit counts, not decomposed gates, depth, runtime, memory, T count, or
Toffoli count.

## Workflow

For a **heralded Hamiltonian application**, specify the normalized input, the
full operator to apply, and the successful ancilla outcome. Keep the accepted
branch `b = (H/alpha) @ psi` unnormalized: its squared norm is the success
probability. Conditional observables use `b† O b / (b† b)` and are undefined
when success has zero probability. Removing a scalar from `H` changes this
filter; restoring an energy offset afterward does not restore the coherent
action. The unused ancilla probability is expected block-encoding leakage.

For **pruning and scalar-offset tradeoffs**, distinguish the physical model
`H = c*I + H0` from the retained encoded operator. Collect duplicate Pauli words
before applying a physical coefficient cutoff; iterable input is not collected
automatically, and `alpha` sums absolute retained decomposition coefficients.
Store `c` separately before pruning: a coefficient below the threshold is
dropped before `constant_term` is accumulated. For an energy calculation, keep
`c` classical and restore it once. Compare candidate energies in the requested
physical sector and in energy units, not as eigenvalues of differently scaled
`H/alpha` blocks. Report normalization, term count and register size as their
own quantities. Circuit-cost claims require a declared decomposition model.

Classical spectra and error bounds can answer a representation-choice question
without launching a circuit. If validating an emitted encoding, execute its
actual retained decomposition and distinguish that result from the prediction.

## Verification

Small independent checkpoint: `terms` is a nonempty list of real
`(coefficient, word)` pairs with common positive width and positive total
absolute coefficient sum; it already describes the retained operator. `ket`
is normalized in little-endian order and `observable` is its dense Hermitian
system observable. The default path is a classical prediction. Setting
`run_circuit=True` additionally executes the library on the active target.

```python
import numpy as np

def pauli_checkpoint(terms, ket, observable, *, run_circuit=False):
    terms = list(terms)
    paulis = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
              "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    width = len(terms[0][1])
    matrix = np.zeros((1 << width, 1 << width), dtype=complex)
    for coefficient, word in terms:
        product = np.ones((1, 1), dtype=complex)
        for label in reversed(word):
            product = np.kron(product, paulis[label])
        matrix += coefficient * product
    alpha = sum(abs(c) for c, _ in terms)
    branch = matrix @ ket / alpha
    probability = float(np.vdot(branch, branch).real)
    conditional = (np.vdot(branch, observable @ branch).real / probability
                   if probability > 0 else None)
    result = {"alpha": alpha, "probability": probability,
              "conditional": conditional}
    if run_circuit:
        from cudaq_algorithms import PauliLCU, sim_utils
        encoding = PauliLCU(terms, coefficient_threshold=0.0)
        actual = sim_utils.action(encoding, ket)
        result["block_error"] = float(np.linalg.norm(actual - branch))
        result["executed_probability"] = float(np.vdot(actual, actual).real)
    return result
```

For a pruning checkpoint, `H_full` and `H_candidate` below are independently
assembled physical matrices with all scalar offsets restored, restricted to
the same physical sector when required:

```python
import numpy as np

def pruning_checkpoint(H_full, H_candidate):
    bias = abs(np.linalg.eigvalsh(H_candidate)[0]
               - np.linalg.eigvalsh(H_full)[0])
    bound = np.linalg.norm(H_candidate - H_full, ord=2)
    return {"energy_bias": float(bias), "operator_bound": float(bound)}
```

The norm bound can certify an energy tolerance, but a bound above the budget
does not establish that the actual bias fails. Separate pruning error from
circuit numerical error. Conditional observables become sensitive to error
when success is small; check the raw probability before normalizing. Helpers
in `python/cudaq_algorithms/sim_utils.py` preserve the branch norm and do not
implement hardware postselection. The source parser and threshold ordering
are in `python/cudaq_algorithms/pauli_lcu.py`.

- Independent oracle: build the dense Pauli matrix without calling
  `PauliLCU`, apply the emitted unitary, and extract the zero-ancilla block.
- Invariants: action equals `(H/alpha)|psi>`, `unprepare` inverts `prepare`,
  single negative terms retain sign, controlled application is identity for
  control `|0>`.
- Tests: `test_pauli_lcu.py` and the generic protocol/consumer suites named
  above, with tolerances fixed in those tests.
- Runnable pointers: `tests/python/test_pauli_lcu.py` and
  `docs/sphinx/examples/python/01_quickstart_block_encoding.py`.
- Expected failures: invalid nonempty words, inconsistent widths, complex
  coefficients, no retained terms, and undersized explicit width. Empty-word
  input is a known unguarded source gap, not a promised rejection.

## External alignment

Translate LCU indexing, ancilla order, normalization, and walk sign before
comparing with external packages or literature. QSP/QSVT phase generation is a
consumer responsibility and is not provided here.
