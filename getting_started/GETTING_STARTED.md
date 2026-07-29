# Getting started with cudaq-algorithms

A lightweight, composable, pure-Python layer of **QPU primitives for
fault-tolerant quantum algorithms**, built on CUDA-Q. You compose the
primitives; applications (phase estimation, Hamiltonian simulation,
quantum exact Lanczos, ...) are a few lines on top.

This guide assumes you're fluent in the algorithms themselves
(qubitization, QSVT, Trotter, chemistry Hamiltonians) and just need to
learn *this library* — its API, its conventions, and its mental model.
For the theory, see the pedagogical notes (`NOTES_*.pdf`).

---

## 1. What it is (and isn't)

- **Is:** reusable FTQC building blocks — block encodings (`PauliLCU`,
  `DoubleFactorizedEncoding`), qubitization (`Walk`), `QSVT`, `Trotter`,
  state preparation, fermion transforms, double factorization, and a
  chemistry bridge — each with a documented, test-pinned contract.
- **Isn't:** NISQ application workflows (VQE / ADAPT / QAOA / GQE). Those
  live in `cudaqx/solvers`. This library is for *building* algorithms
  from primitives, not for running application loops.

## 2. Install

```bash
pip install cudaq_algorithms_cu12-*.whl      # Linux (CUDA 12) or macOS (Apple Silicon)
```

The wheel pulls in the CUDA-Q runtime (`cuda-quantum-cu12` on Linux,
`cuda-quantum-cu13` on macOS), NumPy/SciPy, and `qsppack` (QSVT phase
generation). Some examples also use PySCF for real molecules
(`pip install pyscf`). The default simulator is the CPU statevector
target `qpp-cpu`; override with `CUDAQ_DEFAULT_SIMULATOR`.

## 3. The mental model — four things to internalize

1. **Factories emit kernels.** `PauliLCU(H)`, `Walk(enc)`, `QSVT(enc)`,
   `Trotter(H)` are *compilers, not executors*: the object holds the
   classical data, and `.kernel(...)` emits a ready-to-run CUDA-Q kernel.
   What you do with that kernel — sample it, observe it, synthesize it,
   inject it into another factory — is up to you.
2. **The `BlockEncoding` protocol is the seam.** `Walk` and `QSVT` are
   written once, against the protocol; they never learn which encoding
   they got. Implement the protocol and you inherit the whole primitive
   stack (example 6).
3. **Hardware-shaped by default.** The real measurement path is
   observables (`walk.moment`) and sampling (`cudaq.sample`) — never
   `get_state`. Statevector conveniences for *validation* are quarantined
   in `sim_utils` (`transform`, `evolve`, `action`).
4. **Conventions are explicit and load-bearing.** Signs and orderings are
   documented in **`docs/conventions.md`** — read it first. (For example:
   the walk internally encodes `-H/alpha`, but `walk.moment` returns
   `+<T_k(H/alpha)>`.)

## 4. Five-minute first success

Block-encode a Hamiltonian and measure a Chebyshev moment — that's the
core loop everything else builds on:

```python
import numpy as np, cudaq
from cudaq_algorithms import PauliLCU, Walk
cudaq.set_target("qpp-cpu")

enc  = PauliLCU({"ZZ": 0.5, "XI": 0.3, "IX": 0.3})   # block-encode H/alpha
walk = Walk(enc)                                       # qubitization
state = np.array([1, 0, 0, 0], dtype=complex)
print(walk.moments(state, 4))   # <T_0>, <T_1>, <T_2>, <T_3> of H/alpha
```

Run `01_quickstart_block_encoding.py` for the version that checks these
against a dense matrix.

## 5. The examples — read them in order

| # | File | What it teaches |
|---|------|-----------------|
| 1 | `01_quickstart_block_encoding.py` | `PauliLCU` + `Walk`; moments as the observable path; verify-against-dense |
| 2 | `02_hamiltonian_simulation.py` | `exp(-iHt)` two ways — QSVT (phase sequences) and Trotter (product formulas) — cross-checked |
| 3 | `03_chemistry_to_ground_state.py` | molecule → `qubit_hamiltonian` → walk moments → quantum exact Lanczos vs FCI |
| 4 | `04_double_factorization_and_the_protocol.py` | DF compression dial (H2) and compact encoding at scale (N2, 20q); `PauliLCU` and `DoubleFactorizedEncoding` under one protocol |
| 5 | `05_state_prep_and_injection.py` | `state_prep=` injection → zero-argument synthesizable circuits; Givens Slater determinants |
| 6 | `06_bring_your_own_encoding.py` | implement `BlockEncoding` from scratch; watch `Walk` work on it unchanged |

Each script is self-contained and runnable (`python3 0N_*.py`), prints a
narrated walkthrough, and verifies its numbers against an independent
reference — the house style, and how you should write yours.

## 6. Going deeper

- **Conventions:** `docs/conventions.md` (start here before you trust a sign).
- **Per-family reference:** `docs/pauli_lcu_qsvt.md`, `docs/trotter.md`,
  `docs/double_factorization.md`, `docs/df_block_encoding.md`,
  `docs/stateprep.md`, `docs/fermion_transforms.md`.
- **Pedagogical notes (theory + circuits):** LCU / qubitization / QSVT
  from scratch, and double factorization and its applications.

## 7. The extension point

The `BlockEncoding` protocol is where you plug in. Bring an encoding
(three sizes + the kernel factories, example 6) and `Walk`/`QSVT` work on
it with zero changes. That's the design: the primitives are the product,
and your encoding inherits all of them.
