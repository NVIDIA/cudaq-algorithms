# Quantum benchmark implementation and preflight report

Implemented the ten Task 1 cases in `quantum_cases.py` and numerical contract tests in `tests/test_quantum_cases.py`. All gold programs execute actual packaged primitives in separate Python processes. No benchmark model calls, package mocks, dependency installations, commits, or changes to another task's maintained files were made.

## Numerical contracts

| Case | Independent reference | Sensitive behavior |
| --- | --- | --- |
| `slater_energy` | Determinant minor expansion, dense Pauli energy | Complex relative phase, fermion determinant signs, occupation, identity energy |
| `pool_uccsd` | Fock ladder-operator generators and SciPy matrix exponentials | Full fixed-pool angle, single/double sign, ordered product |
| `pool_uccgsd` | Fock ladder algebra for six singles and three distinct doubles | Generalized pair order and crossing parity |
| `pool_upccgsd` | Fock ladder algebra for spin-preserving singles and paired double | Reversed paired-double orientation |
| `pool_ceo` | Qubit ladder algebra for two singles and two coupled mixed-spin exchanges | No Jordan–Wigner parity, distinct coupled-exchange signs |
| `direct_uccsd` | Separate half-angle/sign factors applied to independently built generators | Direct circuit differs from fixed-pool UCC; mixed double has opposite circuit sign |
| `pauli_action` | Explicit little-endian Pauli Kronecker matrices | Negative coefficients, Y phase, nonzero identity, alpha, unnormalized success block |
| `walk_spectrum` | Matrix Chebyshev recurrence | Odd/even moments, asymmetric spectrum, walk block `T_k(-H/alpha)` |
| `qsvt_filter` | Independent two-dimensional invariant-subspace product and Hermitian eigendecomposition | Directed walks, projector/QSP conventions, complex phase, success norm |
| `qsvt_recovery` | Dense `expm(-i H t)`; separate analytic phase-sequence blocks | QSP global-phase removal and factor two, real input/Hamiltonian domain |

The evaluator oracle imports only NumPy/SciPy and standard-library modules. It neither obtains generators from package pools nor calls package phase-response helpers. The fixed four-mode pool references use second-quantized or qubit ladder algebra rather than copied Pauli expansions. Each case has deterministic public and held-out scientific inputs. Seeds are `4171 + variant`, and numerical grading tolerances are fixed at `atol=rtol=2e-9`. Only Slater `state` permits global-phase equivalence; all other state and block fields preserve absolute phase.

The recovery construction is deliberately explicit: for `H=aX+bZ`, its eigenvalues are `±r`, `r=sqrt(a²+b²)`. A degree-zero QSP cosine sequence satisfies `cos(phi_c)=cos(r*t)/2`; a degree-one sine sequence satisfies `sin(sum(phi_s))=alpha*sin(r*t)/(2*r)`. Their actual circuit blocks recover `exp(-i H t)` on both eigenvalues. This tests recovery with genuine circuits and non-eigenstate input, without claiming phase synthesis or a uniform approximation over all `[-1,1]`.

## Red/green evidence and exact invocation

Tests were added before `quantum_cases.py`. The first invocation failed in `setUpClass` with the intended assertion `quantum_cases.py must implement the numerical contracts` (unittest reports this missing-implementation setup assertion as one error and zero completed tests). After implementation, the same host-only invocation passed nine numerical/contract checks and explicitly skipped the real-runtime test. A later literal LCU normalization check strengthened coverage.

Literal tests pin `Y|0>=i|1>`, q0 as the low state-index bit, a complex two-electron determinant's signed minors, pool generator transition matrix entries, direct double `cos(theta/2)|0011>-sin(theta/2)|1100>`, exact Chebyshev values, directed-walk reversal, doubled QSP phases, signed LCU action, and identity energy. Negative controls explicitly differ for conjugated Slater phases, the fixed/full versus direct/half angle, normalized versus unnormalized postselection, and recovery with missing phase removal/factor two.

The first supported-runtime invocation completed all twenty gold applications (ten IDs times two input variants) with no numerical failures, in 32.996 seconds. Each gold application accepted `--input input.json --output result.npz`, executed in its own temporary directory, and returned a non-object NPZ; tests loaded with `allow_pickle=False` and compared every required output against the independent reference. Temporary test artifacts are deleted by the test fixture; campaign artifacts belong to the controller.

Final verification command from `/workspaces/cudaq-algorithms`:

```bash
CUDAQ_E2E_PYTHON=/tmp/cudaq-e2e-runtime.zY0RJF/venv/bin/python \
PYTHONPATH=/workspaces/cudaq-algorithms/python \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/tmp/cudaq-e2e-runtime.zY0RJF/venv/bin/python -m unittest discover \
  -s skills/cudaq-algorithms/evals/e2e/tests -p test_quantum_cases.py -v
```

Final result: **11 tests passed, zero failures or skips, in 33.862 seconds**, including all twenty actual gold application executions.

Runtime: Python 3.12.3, `cudaq` 0.15.1 / `cuda-quantum-cu12` 0.15.1, NumPy 2.5.3, SciPy 1.18.1; every gold program selects `qpp-cpu` (fp64). The supported interpreter was provisioned by the controller, not this task.

## Execution instrumentation interface

Family IDs use the catalog's `simulation` name. Host `required_symbols` use defining module plus qualified function names, including class `__init__` where appropriate. `direct_uccsd` instead declares `required_kernels=["cudaq_algorithms.stateprep._kernels.uccsd"]`: device kernels are JIT compiled and cannot be observed as ordinary Python calls.

A real direct-UCCSD probe showed that CUDA-Q 0.15.1 inlines the device kernel, leaving only the root function in final launched IR. A safe diagnostic route was verified: a `sys.setprofile` call event for `cudaq.kernel.kernel_decorator.PyKernelDecorator.compile` exposes `self.kernelModuleName` and `self.name`. At the `uccsd` compile event these were `cudaq_algorithms.stateprep._kernels` and `uccsd`, and the ancestor `cudaq.runtime.state.get_state` frame identified the executing root `probe`. That `get_state` subsequently returned a state of norm `0.9999999999999977`. The controller can associate these compile events with successful root execution. Compile alone or mere source-name presence must not count as execution.

## Self-review and limits

- Both variants executed; every application returns substantive numerical output, including complex amplitudes where relevant.
- The Pauli/action/filter inputs include signed terms and identity shifts; walk inputs have unequal positive/negative spectral weights and non-eigenstate complex inputs. Recovery inputs and Hamiltonians stay in the documented real domain.
- Pool and direct-UCCSD tasks use four spin orbitals and two electrons. This covers the complete pools at that size, including generalized and CEO mixed-spin doubles. It does not claim open-shell UCCSD, same-spin UCCSD doubles, or CEO same-spin doubles at larger orbital counts.
- Supplied QSVT phases are scientific inputs. Phase synthesis and uniform Hamiltonian-simulation approximation are not evaluated.
- All host oracles and gold source strings are controller-only. Fair staging, symbol/compile trace enforcement, campaign execution, persistent campaign evidence, and independent review remain the controller's responsibilities.
- The parent performs independent review before model runs; this report is self-review evidence, not independent approval.

## Independent numerical and contract review, 2026-09-11

The independent reviewer checked the case contracts, literal tests, and actual
package implementations. No incorrect Pauli/Fock basis, generator sign,
normalization, phase convention, or energy oracle was found. The ten quantum
IDs and five classical IDs cover nine declared families; this is small-system
workflow coverage with the limits listed above.

Review reproduced a grading-contract defect with real held-out executions:
direct `encode_kernel`/`get_state`/`good_subspace` and
`QSVT.kernel`/`get_state`/`good_subspace` applications matched every numeric
field within `1.4e-16` but failed undisclosed `sim_utils.action` and
`sim_utils.transform` requirements. A `Walk.moment` loop matched moments
within `7.8e-16` but failed the undisclosed `Walk.moments` requirement.
The simulation-helper prompts now explicitly request their helpers, and
every spec supplies `required_public_apis` for the controller to disclose
all required workflow entry points equally to both arms. Contract tests
check that every traced requirement has its corresponding public API name.
The benchmark consequently measures implementation with specified API entry
points, including numerical conventions and composition; it does not measure
unaided discovery of which API to use.

The added disclosure regressions were observed failing before the fixes.
The final combined host verification passed all 24 executed checks, with
one explicit quantum-runtime test skip. That invocation excluded the
provider anchor because the outer command sandbox lacks matching
`/proc/<pid>/statm`; it does not replace the controller's supported-runtime
and final sandbox preflights. No numerical target, tolerance, or gold circuit
changed during this review.
