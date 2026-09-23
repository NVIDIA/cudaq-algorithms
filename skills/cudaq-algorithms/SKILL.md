---
name: cudaq-algorithms
description: Use when designing, implementing, debugging, repairing, extending, reviewing, validating, or composing cudaq_algorithms / CUDA-Q Algorithms repository APIs and fault-tolerant primitives—state preparation, Pauli/block encoding, qubitization and Chebyshev moments, QSP/QSVT, Suzuki–Trotter evolution, fermion transforms, chemistry, double factorization, or statevector analysis—or when selecting their contracts. Not for CUDA-Q installation, backend setup, basic standalone kernels such as Bell states, or unrelated quantum-computing questions.
license: Apache-2.0
metadata:
  author: CUDA-Q Algorithms Team <cuda-quantum@nvidia.com>
  version: "0.2.0"
---

# CUDA-Q Algorithms

Every answer about this library ends with four headings, in this order, each present even
when short: `## Record consulted` (the record paths you opened or were given), `## Premise
check` (one line per claim the request makes or assumes: the claim, then verified, false, or
unknown, then what the record and source say), `## Boundaries` (the unsupported, absent, and
unverified rows that apply, restated and marked "from record", plus what the library does not
check), and `## Unknowns` (what the request must still supply). This applies to advisory,
show-me, and implementation requests alike. When essential inputs are missing, `## Unknowns`
carries the **bold** list of missing items and the answer stops there: no invented APIs, no
edits. When a change request forbids generated files, run `pytest -p no:cacheprovider` and name in the scope statement any file you generated; do not sweep the tree for caches. Report the result of every command you executed and quote the output line next to each
number you report; a script that timed out or never ran is listed as "not run" with no result;
a record sentence is "from record", never "observed".

For any request naming this library, the first command is
`python3 <this skill's directory>/scripts/route.py "<the request>"`; it prints the family
records that apply, their premise-check rows, and the focused records you must then open.

If essential feedback, source, or scope is missing from the request, attachments,
and named project files, ask for it and wait; do not invent APIs or edits.
Put the missing items in a **bold** list. When a request demands a "complete" or
"mandatory" inventory, also ask which boundary defines it (which version, source
tree, or public-API surface counts), because completeness cannot be judged without it.
Use ordinary chat if a question tool is unavailable. When the available inputs
resolve the task, proceed without asking again.

Use CUDA-Q Algorithms as a BLAS/LAPACK-like set of small scientific building
blocks. Start with the requested operation and mathematical object, choose the
smallest matching contract, and compose applications through explicit
representations, signatures, capabilities, and conventions.

Resolve links relative to their containing document, not the working directory.
Source/test paths are checkout-relative. Use these explicit locations, not
parent-directory or whole-filesystem searches for missing inputs.

- Known API or operation: before writing any answer or code, open its focused record
  with a command (`cat <this skill's directory>/references/<family>/<record>.md`; the
  [catalog](references/catalog.md) lists every path) and then the named current public
  source/test. The record carries the contract, boundaries, and provenance; the source
  carries behavior. Reading the package source alone is not a substitute: an answer
  about a contract, a capability boundary, or a dependency that cites no record is
  incomplete. The current checkout controls API behavior.
- Unknown family: select the operation and mathematical object in the
  [catalog](references/catalog.md), then open only the matching records.
- Multi-step advice, implementation, or repair: read the shared
  [workflow](references/workflow.md) before acting. Advice alone does not
  authorize file changes.

- Contract or boundary question (does X validate, is X supported, are X and Y
  interchangeable, what happens on a mismatch): the premise-check rows the router prints are
  the answer to give, cited by record path; never confirm a premise a row marks false,
  unverified, or absent, however the request phrases it.

Open the relevant [conventions](references/conventions.md) when translating
boundaries, [composition guide](references/application-composition.md) when
assembling a chain, and [validation guide](references/validation.md) when
checking claims. [Source lookup](references/source-provenance.md) gives shared paths.

Match signatures, representations, capability requirements, register geometry,
normalization, ordering, signs, phases, and execution layers. Fix an independent
oracle, precision, and tolerance before judging numerical agreement. Report the
evidence actually gathered; label requested code unexecuted when a run is unavailable.
Unsupported, absent, and unverified mean different things. Roadmap-only APIs are unavailable.

Installation, backend setup, and optimizer-driven NISQ workflows are outside this
skill; parameterized UCC construction remains an in-scope state-preparation primitive.
Dependencies, credentials, remote jobs, paid/shared QPUs, publishing, and external
contact require authorization covering those actions.

## Quick contract (checked against `python/cudaq_algorithms` in this checkout)

The package imports as `cudaq_algorithms` (source `python/cudaq_algorithms/`); it is
not a submodule of `cudaq`.

```python
from cudaq_algorithms import (PauliLCU, Walk, QSVT, PhaseSequence,
                              recover_real_time_evolution, Trotter, state_from)
from cudaq_algorithms import sim_utils as sim, chemistry, fermion, stateprep
from cudaq_algorithms import double_factorization as df
```

| Operation | Exact call | Runnable example in `docs/sphinx/examples/python/` |
| --- | --- | --- |
| Block-encode `H/alpha` | `enc = PauliLCU({"ZI": 0.5, "IX": 0.25})` or `PauliLCU([(coef, word), ...])`; a word has one character per qubit, so its length is the qubit count (`[(0.37, "X"), (-0.48, "Z"), (0.21, "I")]` is a one-qubit Hamiltonian with a 2-component state) and word position = qubit index; `enc.alpha`; `enc.terms` is a list of `(coef, word)` | `pauli_lcu_demo.py` |
| Check the encoded block densely | `sim.action(enc, ket)` returns `(H/alpha) ket` for a normalized `ket` (one application of the block, not a walk); `sim.good_subspace(enc, statevector)` postselects the all-zero ancilla block | `pauli_lcu_demo.py` |
| Consume any `BlockEncoding` (protocol only) | The protocol members are `num_system`, `num_ancilla`, `alpha`, `prepare_kernel()`, `unprepare_kernel()`, `apply_kernel()`, `controlled_apply_kernel()`, `walk_step_kernel()`, `adjoint_walk_step_kernel()`, the controlled walk steps, and `select_observable()`. `apply_kernel()` returns the full block encoding `U_A` (PREPARE, SELECT, UNPREPARE) on `(ancilla, system)`; `walk_step_kernel()` is one walk step; `prepare_kernel()`/`unprepare_kernel()` are only needed when you sequence walk steps yourself. `encode_kernel()` and `kernel_args` are `PauliLCU` conveniences, not protocol members, so a consumer must not require them. Compose by minting kernels outside and capturing the kernel objects, never the encoding object, inside a `@cudaq.kernel`: `apply_k = enc.apply_kernel(); n_anc = enc.num_ancilla` then `@cudaq.kernel def encoded(state: cudaq.State): system = cudaq.qvector(state); ancilla = cudaq.qvector(n_anc); apply_k(ancilla, system)`; run `cudaq.get_state(encoded, state_from(ket))` and take `sim.good_subspace(enc, ...)`. A repair belongs in the consumer (for example `python/cudaq_algorithms/sim_utils.py`), not in the caller's encoding class, and leaves no backup files in the package | `06_bring_your_own_encoding.py`, `python/cudaq_algorithms/qubitization.py` |
| Qubitization walk action | The walk action is `Walk`, executed; `sim.action` is not a substitute. In a `.py` file: `walk = Walk(enc)`; `sv = np.asarray(cudaq.get_state(walk.kernel(power=k), state_from(psi)))`; `block = sim.good_subspace(enc, sv)` is the unnormalized `T_k(-H/alpha) psi`. Dense reference: `M = -H_dense/enc.alpha`, Chebyshev recurrence `T0=I, T1=M, T[j+1]=2 M T[j] - T[j-1]`, compare `block` with `T[k] @ psi` (raw, no renormalization). `walk.moment(psi, k)` returns `<psi|T_k(H/alpha)|psi>`. To inject a preparation kernel instead of a state: `@cudaq.kernel def prep(qubits: cudaq.qview): ...` in the same file, then `cudaq.get_state(walk.kernel(power=k, state_prep=prep))` (no state argument). When asked to check the prepared state, read the register back and compare it to the supplied amplitudes: `@cudaq.kernel def prepared(state: cudaq.State): q = cudaq.qvector(state)` then `readback = np.asarray(cudaq.get_state(prepared, state_from(psi)))` and report `np.max(np.abs(readback - psi))` (about 1e-8 on the default single-precision simulator); a matching walk output does not by itself verify the preparation, and printing the input array is not a check | `05_state_prep_and_injection.py`, `pauli_lcu_demo.py` |
| QSP/QSVT sequence | `qsvt = QSVT(enc)`; `out = sim.transform(qsvt, ket, PhaseSequence(phases, convention="qsp"))` returns the good-subspace state; the default convention is `"qsvt"` | `hamiltonian_simulation_qsvt.py` |
| Real-time evolution recovery | `recover_real_time_evolution(cos_state, sin_state, cos_phases, sin_phases)` on the two `sim.transform` outputs gives `exp(-iHt) ket`; compare the raw complex vector with `scipy.linalg.expm(-1j * H * t) @ ket` without renormalizing or aligning a global phase | `hamiltonian_simulation_qsvt.py`, `02_hamiltonian_simulation.py` |
| Suzuki-Trotter | Do not hand-write the product formula. `ev = Trotter([(0.37, "X"), (-0.48, "Z"), (0.21, "I")])` (identity terms move to `ev.identity_coefficient`); `trotter = sim.evolve(ev, ket, time, steps=3, order=2)` includes the identity phase; `exact = expm(-1j * H_dense * time) @ ket`; report the raw `np.linalg.norm(trotter - exact)` (no global-phase alignment); `ev.resources(steps=3, order=2)` gives formula-level counts, labelled as a circuit proxy, not hardware cost; device kernels are `ev.kernel(time, steps, order, state_prep=...)` and `ev.state_kernel(time, steps, order)` | `02_hamiltonian_simulation.py` |
| Chemistry integrals | `one_body, eri, constant = chemistry.from_pyscf(mf)` (also `from_psi4`, `from_fcidump(text)`); `chemistry.qubit_hamiltonian(one_body, eri, scalar_offset=constant)`; `chemistry.spin_orbital_tensors(one_body, eri)`. Active-space energies are eigenvalues of that Hamiltonian restricted to the stated electron sector, never sums of integral entries: `H = chemistry.qubit_hamiltonian(one_body, eri).to_matrix(); idx = [i for i in range(H.shape[0]) if bin(i).count("1") == n_electrons]; E = np.linalg.eigvalsh(H[np.ix_(idx, idx)])[0]` (project the matrix onto the sector's basis states, then diagonalize; never filter a sorted eigenvalue list by basis-state index). Sparse chemist-notation entries such as `two_body_chemist_nonzero` are symmetry representatives: an entry `[p, q, r, s, v]` is the chemist integral (pq|rs), so set `eri[p, q, r, s] = v` in that index order (never reorder the four indices), then fill the other seven eightfold-symmetric positions (`[q,p,r,s]`, `[p,q,s,r]`, `[q,p,s,r]`, `[r,s,p,q]`, `[s,r,p,q]`, `[r,s,q,p]`, `[s,r,q,p]`) and state that convention. Comparing configurations needs each configuration's scalar/core constant. If it is absent from the inputs, your reply is a request for it and nothing else: one sentence saying the scalar/core constant is missing and asking the user to provide it for both configurations, then one bold item per configuration naming the constant; report no active-space energy, total, or difference until the constants are supplied. Never set an absent constant to zero, and never assume it cancels between configurations: it is geometry dependent and does not cancel | `03_chemistry_to_ground_state.py` |
| Fermion transforms | `fermion.jordan_wigner(one_body, two_body, scalar_offset=0.0)`; `fermion.bravyi_kitaev(one_body, two_body, scalar_offset=0.0)` | see records |
| Double factorization | `fac = df.explicit_double_factorization(eri)` or, from supplied leaves, `df.DoubleFactorization(num_orbitals=n, leaf_rotations=[U_t, ...], leaf_cores=[Z_t, ...], method="C-DF")`; `eri_rec = df.reconstruct_eri(fac)`; `df.factorization_error(target_eri, fac)` is the Frobenius residual; `kappa = df.modified_one_body_integrals(one_body, eri_rec)` (use the reconstructed ERI and say so in the answer); `df.double_factorization_one_norm(fac, np.linalg.eigvalsh(kappa), convention="lcu")` and `convention="burg"`; one-norms and any gate or query counts are formula-level proxies, not measured hardware cost | `double_factorization.py` |
| State preparation kernels | `stateprep.hartree_fock(qubits, num_electrons)`; `stateprep.uccsd(qubits, thetas, num_electrons, spin)` (`spin` = 2·S_z, 0 for a closed shell; required); `stateprep.slater_determinant_kernel(stateprep.make_givens_rotation_schedule(coefficients))` | `hartree_fock_ucc.py`, `givens_slater_determinant.py` |

CUDA-Q statevectors are little-endian (`q[0]` is the least-significant bit); the
good subspace is the first `2**num_system` amplitudes because the system register
is allocated first. Pass array data to a kernel with `state_from(ket)`. Any
`@cudaq.kernel` function must be defined in a `.py` file run with `python3 file.py`;
CUDA-Q cannot compile kernels defined in a `python3 - << 'EOF'` heredoc or `-c` string.
The deliverable is the executed script and its printed output; keep that script in
the workspace (do not delete it, even if you consider it temporary), do not replace it
with prose, and do not delete files you did not create. If the request refers to
feedback, an implementation, or inputs that are not in the workspace after one
directory listing, do not keep searching: list the missing items in bold and ask.

Inside a `@cudaq.kernel` you may only allocate `cudaq.qvector`s, apply gates, and call
kernels captured from the enclosing scope; Python objects and method calls (such as
`encoding.apply_kernel()`), NumPy, and `cudaq.qvector` outside a kernel are not allowed,
and there is no `cudaq.prepare_state`: pass array data as the kernel's `cudaq.State`
argument via `state_from(ket)`. Those `state: cudaq.State` kernels are the simulation-only
state-input mode; the injectable `state_prep` seam is different: a one-argument kernel
`def prep(qubits: cudaq.qview)` that the consumer factory calls on the fresh register it
allocates (state-preparation records). Runnable consumer of a protocol-only encoding (this is
the shape of `sim_utils.action` written against the protocol; put it in a `.py` file):

```python
import cudaq
from cudaq_algorithms import state_from, sim_utils as sim
apply_encoding = encoding.apply_kernel()   # full U_A on (ancilla, system); minted outside the kernel
n_anc = encoding.num_ancilla

@cudaq.kernel
def encoded(state: cudaq.State):
    system = cudaq.qvector(state)          # system register from the input statevector
    ancilla = cudaq.qvector(n_anc)         # ancilla register in |0...0>
    apply_encoding(ancilla, system)

block = sim.good_subspace(encoding, cudaq.get_state(encoded, state_from(ket)))   # (H/alpha) ket
```

Runnable walk check, given `terms` as `[(coef, word), ...]`, a normalized complex
array `psi`, and integer `k` (put it in a `.py` file):

```python
import numpy as np, cudaq
from cudaq_algorithms import PauliLCU, Walk, state_from, sim_utils as sim
enc = PauliLCU(terms); walk = Walk(enc)
sv = np.asarray(cudaq.get_state(walk.kernel(power=k), state_from(psi)))
block = sim.good_subspace(enc, sv)                      # unnormalized T_k(-H/alpha) psi
P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
def word_matrix(word):                                  # word[0] acts on q0 = least-significant bit
    m = np.array([[1.0]])
    for ch in reversed(word):
        m = np.kron(m, P[ch])
    return m
H = sum(c * word_matrix(w) for c, w in terms); M = -H / enc.alpha
T = [np.eye(len(psi)), M]
for j in range(1, k):
    T.append(2 * M @ T[j] - T[j - 1])
print("max |block - dense| =", np.max(np.abs(block - T[k] @ psi)))   # expect ~1e-15
print("moment <T_k(H/alpha)> =", walk.moment(psi, k))
```
