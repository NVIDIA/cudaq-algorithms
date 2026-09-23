# Classical, chemistry and evolution case implementation

Implemented all five Task 2 contracts in `classical_cases.py`, with public and
held-out inputs, scientific prompts, independent expected outputs and private
real-package gold applications. No package source or operational skill was
changed. All kernels execute through the real CUDA-Q package; no provider or
evaluated package is mocked.

| Case | Numerical contract and independent oracle |
| --- | --- |
| `trotter_dynamics` | Non-eigenstate two-qubit state, four noncommuting terms including Y, nonzero identity shift, second/fourth order on public/held-out input. Explicit SciPy product exponentials and exact exponential provide phase-sensitive state, probabilities, error and literal unmerged resource counts. |
| `fermion_transport` | Four modes with complex hopping, density interactions and complex quartic pair transfer. Independent signed occupation-basis ladder operators produce JW matrix; a binary-indexed-tree update traversal gives BK permutation. Both full matrices, both evolved states and physical occupations are graded. |
| `molecular_compression` | Synthetic two-orbital chemist ERI with a positive rank-two density core in a nontrivially rotated orbital frame. Complete FCIDUMP includes off-diagonal one-body, three-equal-index ERI records and a core energy. One-leaf C-DF reduces two X-DF leaves to one full-core leaf. Independent FCIDUMP expansion and direct second quantization give ERI, modified-one-body and fixed-electron energy targets. |
| `pyscf_energy` | Real PySCF RHF H2 at 1.4/1.9 Bohr, STO-3G; package extraction and qubit Hamiltonian provide fixed-two-electron ground energy and HF determinant expectation. Independent PySCF FCI and RHF energies plus analytic nuclear repulsion are the reference; no evaluated chemistry bridge participates in the oracle. |
| `psi4_energy` | Separate real Psi4 RHF job, C1, same two geometries. Package Psi4 extraction and qubit matrix provide energies; independent PySCF FCI/RHF supplies gauge-invariant energy references. Both `psi4` and `pyscf` dependencies are explicit. No substitute provider is used for this application. |

The compression application executes actual `compressed_double_factorization`;
it does not claim QSVT coverage from an example filename. No example-only DF
encoding class is assumed public.

## Verification evidence

- TDD red: eight independent contract tests failed on the deliberately absent
  case module, each with an explicit missing-implementation assertion.
- TDD green: all eight host tests passed after implementation (0.091 s).
- Added an independent physical H2 energy anchor and real subprocess gold
  execution on both variants, with non-object NPZ and all-field comparisons.
- Supported runtime: `/tmp/cudaq-e2e-runtime.zY0RJF/venv/bin/python`, real CUDA-Q
  0.15.1, source at `/workspaces/cudaq-algorithms/python`, CPU `qpp-cpu` fp64,
  `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.
- First full available-runtime green: ten tests passed in 10.914 s. This
  executes eight applications: each variant of Trotter, fermion transport,
  molecular compression and PySCF. Psi4 was absent and was dependency-blocked;
  this is not a passing Psi4 application.
- Added real `sys.setprofile` diagnostics to verify required primitives were
  actually called. Red caught four public re-export versus defining-module
  mismatches in JW/BK and DF cases; specifications now name the actual defining
  modules. Trotter requires `__init__` and `resources`, not a class body event.
- Final profiling-plus-numerical green: all ten tests passed in 36.469 s,
  including required-call assertions for all eight available applications.

## Infrastructure and limitations

The outer command sandbox presents PID values without corresponding
`/proc/<pid>/statm`. Real PySCF then raises `FileNotFoundError` while checking
memory before RHF. Authorized execution outside that outer sandbox restored
normal process metadata and all PySCF calculations passed without code
changes or monkeypatching. The controller must verify that its final worker
sandbox exposes working process metadata too.

Psi4 was unavailable in the supplied runtime during the first preflight.
A second preflight used the real conda runtime at
`/tmp/cudaq-e2e-runtime.zY0RJF/psi4/bin/python`. Nine host tests and all eight
non-Psi4 applications passed; both Psi4 variants failed at `import psi4`
(suite duration 42.253 s). This is an infrastructure failure, not an
application numerical failure:

- Psi4 1.10's `driver/procrouting/dft/libxc_functionals.py:64` requests
  `GGA_XC_TH_FL` during import-time functional registration.
- Installed LibXC 7.1.2's `include/xc_funcs.h:196` defines that functional as
  `XC_LDA_XC_TH_FL` instead. Psi4 reports “Could not find required LibXC
  functional”. Direct provider import reproduces the failure.
- `ldd` confirms the extension resolves the task-owned conda `libxc.so.15`,
  rather than an accidental system library. Psi4's conda dependency range
  `libxc-c >=7.0.0,<8.0a0` admitted this incompatible newer dependency.
- The parent was asked to pin a compatible LibXC 7.0 release in the private
  runtime; no application, provider or evaluated package was monkeypatched.

The controller subsequently pinned LibXC 7.0 and reported a successful
supported-runtime suite: ten tests, including all ten real classical gold
applications (both variants of all five cases), passed in 45.86 seconds.
This includes both genuine Psi4 executions. The independent numerical
reviewer received this result from the controller; final campaign-sandbox
preflight remains the controller's responsibility.

All statevector fields in these cases are phase-sensitive; none use phase
alignment. Tolerances are fixed at `atol=2e-8, rtol=2e-8` except compression
(`atol=2e-7`) to allow the actual nonlinear optimization. These are small
deterministic applications, not large-system scaling claims.

Run the full supported-runtime verification using the supplied interpreter,
`E2E_RUNTIME_PYTHON` set to that same interpreter and `python -m unittest
discover -s skills/cudaq-algorithms/evals/e2e/tests -p test_classical_cases.py`.
The runtime test intentionally does not score dependency-blocked apps. The
controller is responsible for recording those blockers in campaign reports.

## Independent numerical and contract review, 2026-09-11

Review found that compression input exposed its generating `orbital_angle`,
`density_core`, `one_body`, and `offset`, giving the worker analytic
factorization and parsing shortcuts. The payload now contains only FCIDUMP,
electron count, and requested leaf count. A separate evaluator-only FCIDUMP
expansion supplies the reference tensors; it imports no evaluated package.
A literal two-orbital test checks tensor values, the modified one-body term,
fixed-electron energy, and the energy response to changing the FCIDUMP core
shift. Both compression gold applications passed again with actual package
call checks. Revised reference ERIs differ from the original analytic
construction by at most `5.6e-17`, and energies by at most `4.5e-16`.

Both real compressed factorizations report `optimizer_success=True` and
`optimizer_nit=0`: the default initial rotation already admits exact
compression. This case exercises the real compressed-factorization API,
full-core solve, reconstruction, and energy pipeline; it does not exercise
iterative rotation optimization. The reconstruction errors observed in the
review probes were below `8e-16`. No iterative-optimizer coverage claim
should be inferred from this case.

Review also found the Trotter prompt did not disclose the grader's required
`sim_utils.evolve` helper. That requirement is now explicit. Every case
additionally exposes `required_public_apis`, with public reexports instead
of private tracing-module paths, for equal disclosure in both arms.
Regressions cover this mapping and the absence of compression answer fields.
All 24 executed combined quantum/classical host checks passed; the separate
compression gold check passed on both variants. Numerical tolerances and
the provider reference construction were unchanged.

## Restrictive sandbox provider diagnosis, 2026-09-11

Final campaign sandbox preflight exposed process metadata failures despite
both providers passing outside that sandbox. The profile remains unchanged.

- PySCF's default automatic memory check attempts `/proc/2/statm`, which is
  absent in the sandbox. Setting its supported `mol.incore_anyway=True`
  before RHF avoids that optional check for this tiny two-orbital molecule.
  The gold application now sets it, and the same requirement is disclosed
  in the scientific task prompt for both arms. No provider method is patched.
- New `ProviderSandboxContracts` regression failed on both input variants
  before this change, reproducing the actual `FileNotFoundError`. It then
  passed both real provider applications in 3.654 seconds, including energy
  comparisons and `runtime.isolation_probe` checks. The separate first
  diagnostic had maximum energy error `1.1e-15`.
- Isolation checks verify the original repository, skill, and evaluator
  are unreadable; input and staged package files are read-only; scratch
  writes work; and connection to a live parent localhost listener is blocked.
- The invocation-only `features.use_legacy_landlock=true` alternative exits
  101: “permission profiles requiring direct runtime enforcement are
  incompatible with --use-legacy-landlock”. It cannot replace the current
  read-isolating profile.
- Psi4's MKL runtime still exits 2 with “Cannot load <mkl-loader>”. Independent
  tests of explicit private `LD_LIBRARY_PATH`, sequential threading,
  preloading the actual MKL LP64/sequential/core libraries, and `MKLROOT`
  all reproduce the same error. The binary references `/proc/self/exe`;
  a read-only child diagnostic confirms that `/proc/self/exe`,
  `/proc/self/maps`, `/proc/self/statm`, and `/proc/2/statm` are all absent.
  This supports a loader/process-metadata incompatibility; no provider
  or sandbox bypass was introduced. A compatible real runtime build
  (potentially using OpenBLAS rather than the MKL loader) requires further
  controller investigation before this family can pass restrictive preflight.

The bounded diagnostic artifacts are under
`/tmp/provider-sandbox-diagnosis-Lc91k0/`. Run the maintained PySCF regression
with `E2E_SANDBOX_PYTHON` set to the supported interpreter and
`test_classical_cases.ProviderSandboxContracts` selected in unittest.
