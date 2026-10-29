# Spectral Processing

## Block normalization and the good subspace

- **Convention.** A zero-flagged block encoding satisfies
  `(<0|_anc x I) U_A (|0>_anc x I) = H / alpha`. Packaged circuits allocate
  the system register first and ancillas after it, so the all-zero-ancilla
  amplitude block is the leading `2**num_system` entries.
- **Normalization.** Simulation helpers return that block unnormalized. Its
  squared norm is the postselection probability only when the full input
  state is normalized; otherwise it is merely the block weight. The helpers
  do not implement a measurement or hardware postselection protocol.
- **Mismatch symptom.** A result has the right norm but wrong amplitudes,
  or a postselection probability is mistaken for a normalization factor.
- **Source.** `python/cudaq_algorithms/block_encoding.py`,
  `python/cudaq_algorithms/sim_utils.py`; [block-encoding.md](../block-encoding/block-encoding.md).

## Qubitization walk sign and moment convention

- **Convention.** One packaged walk step block-encodes `-H / alpha`. The
  qubitization implementation accounts for that sign so `Walk.moment(s)`
  reports the documented positive Chebyshev convention
  `<T_k(H / alpha)>`; callers must not add another negation.
- **Mismatch symptom.** Odd moments have the opposite sign while even moments
  still agree.
- **Source.** `python/cudaq_algorithms/pauli_lcu.py`,
  `python/cudaq_algorithms/qubitization.py`; [qubitization.md](../qubitization/qubitization.md).

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
  [qsvt.md](../qsvt/qsvt.md).

## Reflection gate versus reflection observable

- **Convention.** A reflection used as a circuit operation and the
  `SpinOperator` used for an expectation value are different interface
  objects even when they represent related mathematics. `Walk` uses circuit
  reflections in its kernels and observables for moments.
- **Mismatch symptom.** Passing an observable where a kernel factory is
  required, or describing an expectation value as a circuit application.
- **Source.** `python/cudaq_algorithms/common_kernels.py`,
  `python/cudaq_algorithms/qubitization.py`; [qubitization.md](../qubitization/qubitization.md).
