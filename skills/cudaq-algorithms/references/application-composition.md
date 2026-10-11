# Application composition

Use this guide after routing each requested operation/object pair through
[the catalog](catalog.md). It maps application goals to primitive chains; it
does not define new primitives or promise pairwise compatibility.

## Composition procedure

1. Write the desired mathematical map from input to output.
2. Select one record for each independently selectable step.
3. Match capability IDs where present, then verify concrete signatures,
   representations, widths, ordering, normalization, signs, phases, execution
   layer, and unsupported conditions.
4. Inspect the cited source test or self-checking example for every selected
   seam before adapting code.
5. Keep simulation-only extraction and classical post-processing visible.
6. Choose an independent end-to-end oracle and tolerance before execution.
7. Report component and end-to-end evidence separately.

## Common application chains

| Goal | Primitive chain | Critical boundary |
| --- | --- | --- |
| Apply or inspect a Pauli Hamiltonian block | state preparation (optional) -> `PauliLCU` -> `action` or `good_subspace` for simulation analysis | The flagged block is `H/alpha`; the returned good block is unnormalized. |
| Chebyshev moments | state preparation or `cudaq.State` -> block encoding -> `Walk` -> `moment`/`moments` | Walk circuits encode `-H/alpha`, while the measurement API reports the documented positive `T_k(H/alpha)` convention. Odd moments require `select_observable`. |
| Polynomial spectral transformation | state preparation or `cudaq.State` -> block encoding -> `PhaseSequence` -> `QSVT` -> optional `transform` | QSP phases are doubled to projector phases; account for `exp(i sum(phi))`. |
| Real-time evolution by QSVT | real state -> block encoding -> cosine QSVT and sine QSVT -> two good-subspace vectors -> recovery | Both components are required; the packaged recovery helper is documented only for real Hamiltonians and real inputs. |
| Product-formula evolution | state preparation or statevector -> `Trotter` -> kernel or simulation `evolve` | Identity terms are global phase and absent from device circuits; the simulation helper can restore them. |
| Chemistry Hamiltonian for Pauli algorithms | integral loader -> either (A) `chemistry.qubit_hamiltonian` or (B) `chemistry.spin_orbital_tensors` -> `fermion.jordan_wigner` / `fermion.bravyi_kitaev` -> `PauliLCU`, `Walk`, QSVT, or `Trotter` | `qubit_hamiltonian` consumes spatial tensors and performs spin expansion plus Jordan-Wigner internally. Do not feed it spin-expanded tensors. Pass the loader's scalar offset to `qubit_hamiltonian` or the selected explicit transform. |
| Compressed chemistry approximation | integral loader -> compressed double factorization -> reconstruction/error analysis -> qubit Hamiltonian | `DoubleFactorization` is host data, not a packaged quantum kernel; approximation and optimizer status remain visible. |
| Quantum phase estimation | Hartree-Fock preparation -> counting register -> controlled oracle (`Trotter` product-formula evolution, or the block encoding's `controlled_walk_step_kernel`/`controlled_adjoint_walk_step_kernel` with `prepare_kernel` on the signal register) -> inverse QFT -> classical decoding | Not a packaged primitive; `docs/sphinx/examples/python/08_quantum_phase_estimation.py` is the self-verifying shell. Time-evolution oracles decode linearly in `t`; the walk decodes as `E = -alpha*cos(theta)`. QPE samples eigenvalues with the input state's spectral weights; it does not choose the ground state. |
| Lookup-based SELECT/PREPARE components | `cudaq_algorithms.primitives`: `unary_iteration_kernels` or `QROM` for coherent table lookup; `AliasSamplingPrepare` for a weighted index marginal; reversible adders/comparators (see [circuit primitives](primitives/primitives.md)) | Strictly unitary kernels with exact logical Toffoli counts; alias-sampling garbage registers are not orthogonal, so only index-register amplitude magnitudes are guaranteed. |

## Example adaptation rules

- Start from a repository example or test named in
  [source-provenance.md](source-provenance.md); confirm its selected symbols
  against the current checkout, and do not invent an API from a roadmap
  concept or from a similarly named external package.
- Replace only the scientific inputs needed by the task. Preserve convention
  conversions, validation, target/precision setup, oracle, and tolerance until
  each change has been justified.
- Do not copy example-only classes into the answer as if they were installed
  symbols. If adapting one is requested, label it application code.
- Keep optional dependencies optional. PySCF, Psi4, QSPPACK, and CuPy are not
  unconditional package requirements merely because an example or provider can
  use them.
- A custom object that structurally satisfies `BlockEncoding` still requires
  scientific validation of its `H/alpha` block and each member a consumer uses.
- A one-argument preparation kernel must be checked against each concrete
  consumer. `BlockEncoding` conformance alone does not promise injection.

For implementation, use the shared [workflow](workflow.md#advice-or-implementation).
State input shapes, dtypes, units, ordering, output signatures, and register
allocation explicitly; report component and end-to-end evidence with the target,
precision, version, and tolerance actually used.

## Workflow

The three paths below start from scientific inputs and terminate at a requested
quantity. Each arrow is a checked handoff, not a requirement to use every
primitive. Read the relevant topic's Workflow and Verification as a stage is
selected; API details remain in its focused contracts.

### Molecular structure to an energy estimate

Inputs: geometries/units or spatial integrals, basis and orbital convention,
charge/spin sector, active-space rule if any, and the requested energies, gaps,
or reference-state overlaps with their tolerances.

| Stage | Resource and action | Check before continuing |
| --- | --- | --- |
| Physical model | [Chemistry](chemistry/chemistry-bridges.md#workflow): converge the specified reference, load tensors, optionally freeze core/select active orbitals | Orthogonality, integral ordering, scalar/core contribution, electron counts, and consistent orbital character across geometries |
| Representation | Chemistry's direct JW Hamiltonian, or [fermion transforms](fermion-transforms/fermion-transforms.md#workflow) for explicit JW/BK comparison | Independent determinant/sector energy; map the state and observables with the Hamiltonian |
| Reference | [State preparation](state-preparation/state-preparation.md#workflow): determinant or supplied ordered correlated preparation | Occupations, sector, and state fidelity; quantify loss of reference overlap near crossings |
| Estimator | [Pauli LCU](block-encoding/pauli-lcu.md#workflow) and [moments/Krylov](qubitization/qubitization.md#workflow) when that quantum path is requested; quantum phase estimation via the QPE chain above (`08_quantum_phase_estimation.py`), whose counting-register width sets the energy resolution | Raw block, moment sign/normalization, Gram rank, and energy versus an independent sector reference; for QPE, the modal bin within one phase cell of the reference and the reference-state overlap |
| Optional approximation | [Double factorization](double-factorization/double-factorization.md#workflow) only when compression/cost is part of the task | Reconstructed physical spectrum and requested accuracy, separately from tensor residual and norm proxy |

Adapt `docs/sphinx/examples/python/03_chemistry_to_ground_state.py` for the
provider-to-qubit-to-moments chain. Its Krylov helpers are application code, not
installed APIs. PySCF is optional; when starting with integrals, bypass SCF and
retain supplied electron/orbital metadata. Dense FCI is an oracle on tractable
cases, not a scalable guarantee. HF-supported Krylov convergence need not reach
the lowest state of a different symmetry sector.

The following parameterized checkpoint connects a **converged closed-shell RHF**
object `mf` to executed moments and an independent PySCF FCI reference. The
chain assumes a restricted closed-shell singlet reference: `chemistry.from_pyscf`
and `chemistry.from_psi4` accept restricted references only and reject
unrestricted (UHF) ones, and `from_fcidump` reads real RHF/ROHF-style files, so
open-shell or unrestricted references are outside this checkpoint. Define
`krylov_checkpoint` from the [qubitization Verification](qubitization/qubitization.md#verification)
in the same `.py` script first. Choose positive `dimension`/`cutoff` and the
energy/moment tolerances before running. For active-space tasks, replace the
integrals, occupations, and FCI problem together using the chemistry workflow.

```python
# requires krylov_checkpoint from qubitization/qubitization.md#verification,
# defined in this same .py file before use
import cudaq
import numpy as np
from pyscf import fci
from cudaq_algorithms import chemistry, PauliLCU, Walk

cudaq.set_target("qpp-cpu", precision="fp64")

def molecular_energy_checkpoint(mf, dimension, cutoff, coefficient_tol=1e-10):
    assert mf.converged and mf.mol.spin == 0   # restricted closed-shell singlet only
    h, eri, nuclear = chemistry.from_pyscf(mf)
    width = 2 * len(h)
    electronic = chemistry.qubit_hamiltonian(h, eri, scalar_offset=0.0)
    # Explicit word width retains idle orbitals; no scalar is counted twice.
    coefficients = [complex(t.evaluate_coefficient()) for t in electronic]
    # A Hermitian chemistry Hamiltonian has real coefficients; an imaginary
    # part signals a tensor-ordering or convention error, so fail loudly.
    assert all(abs(c.imag) < coefficient_tol for c in coefficients)
    terms = [(c.real, t.get_pauli_word(width))
             for c, t in zip(coefficients, electronic)]
    encoding = PauliLCU(terms)
    psi = np.zeros(1 << width, complex)
    occupied = np.flatnonzero(np.asarray(mf.mo_occ) > 1.0)
    assert np.allclose(np.asarray(mf.mo_occ)[occupied], 2.0)
    assert 2 * len(occupied) == mf.mol.nelectron
    bits = sum((1 << (2*int(p))) | (1 << (2*int(p)+1)) for p in occupied)
    psi[bits] = 1.0
    mu = np.asarray(Walk(encoding).moments(psi, 2 * dimension))
    estimate, gram_eigenvalues, rank = krylov_checkpoint(
        mu, encoding.alpha, dimension, cutoff, offset=nuclear)
    reference = float(fci.FCI(mf).kernel()[0])
    return {
        "energy": estimate,
        "fci": reference,
        "energy_error": abs(estimate-reference),
        "rank": rank,
        "gram_eigenvalues": gram_eigenvalues,
        "moments": mu,
    }

# result = molecular_energy_checkpoint(mf, dimension, cutoff)
```

### Lattice Hamiltonian to a ground or sector energy

Inputs: lattice/boundaries and couplings (Hubbard `t`/`U`, Heisenberg/Ising
couplings and fields) or an explicit Hamiltonian, the particle-number/spin
sector (for example half filling, `S_z`), and the requested energy, gap, or
spectrum with its tolerance.

| Stage | Resource and action | Check before continuing |
| --- | --- | --- |
| Model | [Fermion transforms](fermion-transforms/fermion-transforms.md#workflow) for fermionic lattices (hopping and on-site tensors -> Pauli operator); spin models are already Pauli sums | Boundary signs, site/spin-orbital order, Hermiticity, and that the dense operator on a small instance reproduces the intended couplings |
| Reference | [State preparation](state-preparation/state-preparation.md#workflow): a determinant or occupation state in the requested sector | Occupations, particle number, and `S_z`; the reference must overlap the sector ground state |
| Estimator | [Pauli LCU](block-encoding/pauli-lcu.md#workflow) -> [moments/Krylov](qubitization/qubitization.md#workflow), or the QPE chain above with the same encoding | Moment sign/normalization and Gram rank, or QPE bin resolution; energy versus an exact diagonalization of the same sector on a tractable instance |
| Decision | Sector/parity check as the handoff: project the dense Hamiltonian onto the requested sector before diagonalizing | A sector ground state need not be the global ground state; label the sector explicitly |

The qubitization Verification `krylov_checkpoint` applies unchanged with
`offset` equal to any separated scalar; `Walk.moments` on the sector reference
supplies `mu`. No packaged lattice-model builder exists: the Hamiltonian is
caller-supplied `(coefficient, word)` terms or fermion tensors.

### Lattice Hamiltonian to observable dynamics

Inputs: lattice/boundaries and couplings or an explicit Hamiltonian, state,
observable, times, and accuracy/cost constraints. A named method or supplied
phase sequence remains part of the problem; otherwise choose a path suited to
the desired quantity and distinguish a classical prediction from a circuit run.

| Stage | Resource and action | Check before continuing |
| --- | --- | --- |
| Model and state | [Fermion transforms](fermion-transforms/fermion-transforms.md#workflow) for fermionic input; [state preparation](state-preparation/state-preparation.md#workflow) if a preparation is needed | Boundary signs, basis order, complex hopping/current terms, and physical sector |
| Dynamics | [Trotter](trotter/trotter.md#workflow) for product-formula circuits; [QSVT](qsvt/qsvt.md#workflow) for supplied polynomial responses; [moments](qubitization/qubitization.md#workflow) for spectral/return-amplitude predictions | Match the requested approximation, phase convention, scalar and normalization; validate supplied phases rather than assuming they realize the desired function |
| Readout | [Simulation analysis](simulation/simulation-analysis.md#workflow) | Observable normalization, success probability before conditioning, or the actual reduced state when ancillas are discarded |
| Decision | Compare the declared candidate grid at fixed physical model and metric | Accuracy at every requested time and consistent cost; symmetry conservation is a separate diagnostic |

Reuse the Trotter section of `docs/sphinx/examples/python/02_hamiltonian_simulation.py`
and the topic checkpoint. This multi-time variant accepts normalized `psi`,
Hermitian observable `O`, ordered `(coefficient, word)` terms, an independently
built dense `H` in the same little-endian basis, nonempty `times`, and a declared
list of `(order, steps)` candidates. Save it in a `.py` file. Other methods use
their own topic checkpoints rather than silently replacing a requested method.

```python
import numpy as np
from scipy.linalg import expm
from cudaq_algorithms import Trotter, sim_utils as sim

def observable_grid(terms, H, psi, O, times, candidates, tolerance):
    expectation = lambda v: float(np.vdot(v, O @ v).real)
    reference = [expectation(expm(-1j*t*H) @ psi) for t in times]
    plan = Trotter(terms)
    rows = []
    for order, steps in candidates:
        values = [expectation(sim.evolve(plan, psi, t, steps=steps, order=order))
                  for t in times]
        error = float(np.max(np.abs(np.asarray(values)-reference)))
        cost = plan.resources(steps=steps, order=order)
        rows.append({"order": order, "steps": steps, "values": values,
                     "max_error": error, "cx_per_time": cost.estimated_cx_count})
    feasible = [r for r in rows if r["max_error"] <= tolerance]
    best = min((r["cx_per_time"] for r in feasible), default=None)
    return rows, [r for r in feasible if r["cx_per_time"] == best]
```

`cx_per_time` is a logical decomposition proxy for one circuit evaluation, not
the cost of all measurement shots or a compiled device run. A finite time grid
does not certify a continuous-time maximum or event time: refine/interpolate
under a stated error criterion if the researcher asks when an event occurs.

## Verification

Choose the requested physical metric from [validation](validation.md#match-the-check-to-the-scientific-claim).
Run each selected stage checkpoint before judging the final result. For the
molecular chain, compare the restored energy to the same model/sector FCI
reference and inspect rank/overlap sensitivity; for lattice dynamics, compare
the requested observable/time series to an independent propagator and assess
sector leakage separately. Keep intermediate tolerances distinct from the
end-to-end target. Report an inaccurate or infeasible candidate as such; do not
replace its inputs until it passes. State which stages were executed and which
were derived or source-checked, including optional dependencies and version.

## Source-grounded example pointers

- State preparation and injection:
  `docs/sphinx/examples/python/05_state_prep_and_injection.py`.
- Pauli LCU quick start:
  `docs/sphinx/examples/python/01_quickstart_block_encoding.py`.
- Custom block encoding:
  `docs/sphinx/examples/python/06_bring_your_own_encoding.py`.
- QSVT Hamiltonian simulation and matrix inversion:
  `02_hamiltonian_simulation.py`, `07_matrix_inversion_qsvt.py`, and
  `hamiltonian_simulation_qsvt.py` under `docs/sphinx/examples/python/`.
- Double-factorized encoding:
  `04_double_factorization_and_the_protocol.py`, `df_compression_to_qsvt.py`,
  `df_encoding.py`, and `df_block_encoding.py` under that directory; the
  encoding classes are application examples, not packaged providers.
- Trotter end-to-end behavior: the executable reference cases in
  `tests/python/test_trotter.py`, plus `03_chemistry_to_ground_state.py` and
  `trotter_chemistry.py`.
- Quantum phase estimation: `docs/sphinx/examples/python/08_quantum_phase_estimation.py`
  (hand-written QPE shell and inverse QFT; the oracle kernels come from
  `trotter.apply_trotter` and `PauliLCU`).
- Circuit primitives: `tests/python/test_primitives_*.py` (QROM, unary
  iteration, arithmetic, alias sampling, resource pins); no docs example.
