# Pauli LCU block encoding, qubitization, and QSVT

Pure-Python quantum primitives in `cudaq_algorithms`, implemented as CUDA-Q
Python kernels: Pauli LCU block encoding, qubitization walks with Chebyshev
moment measurement, and QSVT phase sequences, each with controlled variants.
They require only the `cudaq` Python package and numpy — the compiled
`_pycudaq_algorithms` extension is not involved.

```
python/cudaq_algorithms/
  block_encoding.py  the BlockEncoding protocol Walk and QSVT consume
  common_kernels.py  encoding-independent kernels (zero reflections,
                     signal phases)
  pauli_lcu.py       everything LCU: PauliLCU + prepare/select/apply,
                     walk-step and phase-sequence kernels, SELECT observable
  qubitization.py    Walk (moments) + reflection observable, encoding-generic
  qsvt.py            PhaseSequence, QSVT, encoding-generic
  sim_utils.py       simulation-only helpers (statevector access)
tests/python/
  test_pauli_lcu.py, test_qubitization.py, test_qsvt.py
examples/pauli_lcu_qsvt/
  pauli_lcu_demo.py                printed walkthrough of the LCU surface
  hamiltonian_simulation_qsvt.py   QSVT time evolution vs. exact
                                   diagonalization (needs qsppack, scipy)
```

## Using the primitives

```python
from cudaq_algorithms import PauliLCU, Walk, QSVT, PhaseSequence
```

### Block encoding

```python
# Flexible construction: dict, SpinOperator, or (coeff, word) pairs.
enc = PauliLCU({"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11})
enc = PauliLCU(spin_op, num_qubits=2)
enc = PauliLCU([(0.7, "ZI"), (-0.43, "IZ")])

enc.num_system, enc.num_ancilla, enc.num_terms
enc.alpha                  # LCU 1-norm
enc.terms                  # [(coeff, word), ...]
enc.constant_term          # sum of identity terms

kernel = enc.encode_kernel()          # @cudaq.kernel(state): full U_A
```

### Qubitization

```python
walk = Walk(enc)
walk.kernel(power=3)                  # PREPARE + W^3 + UNPREPARE
walk.kernel(power=3, uncompute=False) # ... without UNPREPARE
walk.adjoint_kernel(power=2)          # (W dagger)^2
walk.roundtrip_kernel(power=2)        # W^2 (W dagger)^2 == identity
walk.controlled_kernel(power=2)       # controlled walks (see below)

walk.moment(psi, k)                   # <T_k(H/alpha)> via cudaq.observe
walk.moments(psi, 8)                  # even/odd observable convention

from cudaq_algorithms import reflection_observable, select_observable
reflection_observable(enc)            # 2|0..0><0..0| - I  (SpinOperator)
select_observable(enc)                # sum_i sign_i |i><i| x P_i
```

### QSVT

```python
seq = PhaseSequence(phases)                      # projector convention
seq = PhaseSequence(phases, convention="qsp")    # QSPPACK convention,
                                                 # converted automatically
seq = PhaseSequence(phases, walk_directions=["forward", "adjoint"])

transformer = QSVT(enc)
kernel = transformer.kernel(seq)                 # @cudaq.kernel(state)
controlled = transformer.controlled_kernel(seq)  # controlled sequence

from cudaq_algorithms import recover_real_time_evolution
evolved = recover_real_time_evolution(cos_state, sin_state,
                                      cos_phases, sin_phases)
```

Sign convention: the walk block encodes `-H/alpha`, and the circuits fold
the sign in, so on an eigenstate with eigenvalue `lambda` the good-subspace
block implements `p(lambda / alpha)` — no caller-side negation. The 2x2
signal-model reference implementing this convention lives in
`tests/python/test_qsvt.py` (`reference_response`), where it serves as the
test oracle.

Phase *generation* stays external (e.g. QSPPACK): the primitives consume
whatever phase list they are given.

## Hardware-shaped vs. simulation-only

`cudaq.get_state` is a simulator-only API, so nothing in `pauli_lcu`,
`qubitization`, or `qsvt` calls it. Statevector conveniences are isolated
in `cudaq_algorithms.sim_utils`:

```python
from cudaq_algorithms import sim_utils as sim

good = sim.good_subspace(enc, state)  # postselect the |0..0>-ancilla block
hpsi = sim.action(enc, psi)           # (H/alpha)|psi>
out  = sim.transform(transformer, psi, seq)
psi0 = sim.state_from(ket)            # precision-aware cudaq.State
```

`Walk.moment`/`moments` stay in the library proper: they measure through
`cudaq.observe`, which is a hardware-legitimate path.

Escape hatch at every level: the module-level kernels compose inside user
kernels, with `enc.kernel_args` supplying the flattened arrays they take as
arguments. They live in their module namespaces — their names are too
generic for the package root — as `cudaq_algorithms.pauli_lcu.{prepare,
select, apply, walk, adjoint_walk, apply_phase_sequence, ...}` and
`cudaq_algorithms.common_kernels.{reflect_about_zero, signal_phase, ...}`.

## Simulator selection

Tests and examples default to `qpp-cpu` and honor CUDA-Q's standard
`CUDAQ_DEFAULT_SIMULATOR` variable (e.g. `nvidia-fp64` for the GPU
statevector simulator). Use an fp64 target for the test suite — the
`nvidia` target is fp32 and misses the 1e-8..1e-10 tolerances (precision,
not correctness). State construction goes through `state_from`, which
matches the input dtype to the active target's precision
(`cudaq.complex()`), since fp32 simulators reject complex128 initial-state
data.

## Implementation notes

1. Multi-controlled gates use CUDA-Q Python's variadic control support
   (`x.ctrl(ancilla, target)`, `z.ctrl(reg.front(n-1), reg[n-1])`), so
   there is no ancilla-count cap.

2. Kernels defined inside factory methods capture the flattened arrays and
   call module-level kernels; `kernel_args` remains available for composing
   inside user kernels.

3. Controlled kernels use a combined-register convention: a CUDA-Q Python
   control set cannot mix a bare qubit with a `qview`
   ([cuda-quantum#4848](https://github.com/NVIDIA/cuda-quantum/issues/4848)),
   and `cudaq.control(...)` of a kernel that calls other kernels fails. The
   controlled kernels therefore take a single register whose qubit 0 is the
   external control and whose remaining qubits are the ancilla/signal
   register — every control set is then a view of that register.
   Uncontrolled PREPARE pairs wrap the controlled SELECT, so everything
   collapses to the identity at control |0> (verified in tests for walks,
   roundtrips, and sequences, both control states).

4. Known CUDA-Q Python lowering limitations worked around in the modules:
   - Empty `list` kernel arguments fail with "Cannot infer runtime argument
     type" ([cuda-quantum#4847](https://github.com/NVIDIA/cuda-quantum/issues/4847))
     — single-term encodings are normalized to one (idle) ancilla,
     identity-only extractions pad `term_ops` with a never-read
     entry, and the QSVT factories pad degree-0 direction lists
     (callers composing the module-level sequence kernels must pass
     a non-empty `walk_directions` themselves).
   - A `@dataclass` kernel argument with a `list[int]` field containing a
     negative value fails with `std::bad_cast`
     ([cuda-quantum#4846](https://github.com/NVIDIA/cuda-quantum/issues/4846))
     — this blocks passing one aggregated kernel-args object instead of
     flat lists.
   - `return` inside a Python kernel body is silently ignored
     ([cuda-quantum#4845](https://github.com/NVIDIA/cuda-quantum/issues/4845))
     — kernel bodies use positively-guarded blocks instead of early-return
     guards.

5. Circuit-level optimizations deliberately deferred (documented, not
   implemented): cancelling the PREPARE / PREPARE-dagger identity pair at
   the final walk step's uncompute boundary, folding the QSVT zero
   reflection into the adjacent projector phase (both are diagonal on the
   signal register), and Gray-code ordering of SELECT's control-bit
   updates (~num_ancilla-fold fewer X gates). All are depth optimizations
   with no effect on simulator results.

6. Test methodology: dense Pauli-sum action match, a single-term
   negative-coefficient sign regression, Chebyshev moments on an asymmetric
   spectrum via both observables, adjoint walks inverting forward walks,
   controlled walks/sequences against their uncontrolled references at both
   control states, the QSVT good-subspace block matching the 2x2 signal
   model column by column (including mixed walk directions), qsp/qsvt
   convention equivalence, and a QSPPACK Hamiltonian-simulation run
   reaching ~1e-15 state error.
