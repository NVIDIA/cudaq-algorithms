# State preparation — family selector

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| preprocess / orbital-coefficient matrix into a Givens schedule | `cudaq_algorithms.stateprep.make_givens_rotation_schedule` | classical transformation; host | none | [Givens schedule](state-preparation-givens-schedule.md) |
| prepare / Slater-determinant state from a Givens schedule | `cudaq_algorithms.stateprep.slater_determinant_kernel` | quantum operation; host validation + kernel factory/device kernel | provides `cudaq-algorithms.state-preparation.unitary.v1` | [injectable Slater determinant](state-preparation-slater-determinant-kernel.md) |
| prepare / Hartree–Fock state with optional fixed-parameter UCC product | `cudaq_algorithms.stateprep.hartree_fock_ucc_kernel` | quantum operation; host validation + kernel factory/device kernel | provides `cudaq-algorithms.state-preparation.unitary.v1` | [HF + fixed UCC](state-preparation-hf-ucc.md) |
| prepare / contiguous Hartree–Fock occupation on a live register | `cudaq_algorithms.stateprep.hartree_fock` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [canonical HF device kernel](state-preparation-kernel-hartree-fock.md) |
| prepare / explicit occupation on a live register | `cudaq_algorithms.stateprep.hartree_fock_occupation` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [occupation HF device kernel](state-preparation-kernel-hartree-fock-occupation.md) |
| apply / one UCCSD single excitation to a live register | `cudaq_algorithms.stateprep.single_excitation` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [single-excitation device kernel](state-preparation-kernel-single-excitation.md) |
| apply / one UCCSD double excitation to a live register | `cudaq_algorithms.stateprep.double_excitation` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [double-excitation device kernel](state-preparation-kernel-double-excitation.md) |
| apply / occupied-to-virtual UCCSD product to a live register | `cudaq_algorithms.stateprep.uccsd` | quantum operation; device kernel | runtime amplitudes and concrete caller-owned `cudaq.qview`; no capability ID | [UCCSD device kernel](state-preparation-kernel-uccsd.md) |
| apply / generalized UCCGSD product to a live register | `cudaq_algorithms.stateprep.uccgsd` | quantum operation; device kernel | runtime amplitudes/grouped Pauli data; no capability ID | [UCCGSD device kernel](state-preparation-kernel-uccgsd.md) |
| apply / paired UpCCGSD product to a live register | `cudaq_algorithms.stateprep.upccgsd` | quantum operation; device kernel | runtime amplitudes/grouped Pauli data; no capability ID | [UpCCGSD device kernel](state-preparation-kernel-upccgsd.md) |
| apply / coupled-exchange product to a live register | `cudaq_algorithms.stateprep.ceo` | quantum operation; device kernel | runtime amplitudes/grouped Pauli data; no capability ID | [CEO device kernel](state-preparation-kernel-ceo.md) |
| apply / arbitrary fixed-parameter UCC product to a live register | `cudaq_algorithms.stateprep.fixed_parameter_ucc` | quantum operation; device kernel | runtime amplitudes/grouped Pauli data; no capability ID | [fixed-UCC device kernel](state-preparation-kernel-fixed-parameter-ucc.md) |
| apply / adjacent real fermionic Givens rotation | `cudaq_algorithms.stateprep.givens_rotation` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [Givens-rotation device kernel](state-preparation-kernel-givens-rotation.md) |
| apply / adjacent phase-aware fermionic Givens rotation | `cudaq_algorithms.stateprep.phase_givens_rotation` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [phase-Givens device kernel](state-preparation-kernel-phase-givens-rotation.md) |
| prepare / real Slater determinant from flattened arrays | `cudaq_algorithms.stateprep.slater_determinant` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [real Slater device kernel](state-preparation-kernel-slater-determinant.md) |
| prepare / complex Slater determinant from flattened arrays | `cudaq_algorithms.stateprep.complex_slater_determinant` | quantum operation; device kernel | concrete caller-owned `cudaq.qview`; no capability ID | [complex Slater device kernel](state-preparation-kernel-complex-slater-determinant.md) |
| preprocess / occupied-to-virtual UCCSD excitation pool | `cudaq_algorithms.stateprep.make_uccsd_operator_pool` | classical transformation; host | none | [UCCSD operator pool](operator-pool-uccsd.md) |
| preprocess / generalized UCCGSD excitation pool | `cudaq_algorithms.stateprep.make_uccgsd_operator_pool` | classical transformation; host | none | [UCCGSD operator pool](operator-pool-uccgsd.md) |
| preprocess / paired UpCCGSD excitation pool | `cudaq_algorithms.stateprep.make_upccgsd_operator_pool` | classical transformation; host | none | [UpCCGSD operator pool](operator-pool-upccgsd.md) |
| preprocess / coupled-exchange operator pool | `cudaq_algorithms.stateprep.make_ceo_operator_pool` | classical transformation; host | none | [CEO operator pool](operator-pool-ceo.md) |
| estimate / Givens determinant logical operations | `cudaq_algorithms.stateprep.estimate_givens_resources` | formula-level resource estimator; host | consumes a Givens schedule | [Givens resources](state-preparation-resources-givens.md) |
| estimate / canonical Hartree–Fock logical operations | `cudaq_algorithms.stateprep.estimate_hartree_fock_resources` | formula-level resource estimator; host | consumes qubit/electron counts | [canonical HF resources](state-preparation-resources-hartree-fock.md) |
| estimate / occupation-list Hartree–Fock logical operations | `cudaq_algorithms.stateprep.estimate_hartree_fock_occupation_resources` | formula-level resource estimator; host | consumes an occupation list | [occupation HF resources](state-preparation-resources-hartree-fock-occupation.md) |
| estimate / fixed-parameter UCC logical operations | `cudaq_algorithms.stateprep.estimate_fixed_parameter_ucc_resources` | formula-level resource estimator; host | consumes grouped Pauli words (structural counts only) | [fixed-UCC resources](state-preparation-resources-fixed-parameter-ucc.md) |

The schedule planner returns host data; the Slater-determinant and HF/UCC
factories return one-register kernels. Inputs, outputs, validation, and
provenance live in their focused records. A Givens schedule is host data, not a
kernel, and is independently consumed by `estimate_givens_resources`.

The raw Hartree–Fock, excitation, UCC, Givens, and flattened
Slater-determinant device kernels are independently callable and
runtime-parameterized. They mutate a caller-owned `cudaq.qview`, return no host
value, and are a different representation from the injectable one-register
seam below: they take runtime scientific data alongside a live register and
are not directly accepted as `state_prep` unless their full signature already
has that shape or a caller supplies a wrapper. Device bodies provide no general
host validation or status channel; use the host planners or validators named
by the selected focused record before launch.

Eleven device-kernel entries in the operation table are public exports in
`stateprep.__all__`. `single_excitation` and `double_excitation` are explicitly
imported into the public module and are addressable there, although omitted
from `__all__`; their focused records preserve that surface qualification.

## Resource estimator routing

Resource estimation is independently selectable because its inputs, results,
validation, and interpretation differ from state preparation. Route an exact
`cudaq_algorithms.stateprep` estimator symbol through the operation table, then
read only the focused estimator record it selects. Do not infer one estimator's
validation or bound status from either preparation record.

All four estimators and their frozen result dataclasses live in the
`cudaq_algorithms.stateprep` namespace; none is exported from the package root.
They execute on the host and do not build, compile, transpile, or run a quantum
circuit. Their quantities are logical operations before transpilation, not
hardware gate counts. The Givens result is additionally documented by its
source as decomposition-independent upper-bound proxies; the three
Hartree–Fock/UCC estimator docstrings do not make that bound claim. None reports
target-aware decomposition, routed depth, runtime, memory, T/Toffoli count,
measurement, or scientific approximation error, and no cross-estimator
composition rule is documented.

## Shared contracts

Read the [injection contract](injection-contract.md) for the one-register representation,
consumer methods, exact-width requirement, and unsupported/unverified boundaries.
Read [UCC parameterization](ucc-parameterization.md) when choosing among UCC kernels,
and [HF/UCC validation](hf-ucc-validation.md) for host rejection checks and oracles.
The distinct UCCSD, UCCGSD, UpCCGSD, and CEO operator-pool provider
representation remains indexed by [operator-pools.md](operator-pools.md).

Shared source for Givens scheduling, kernels, and resources is
`python/cudaq_algorithms/stateprep/_givens.py`; Hartree–Fock and UCC resource
estimation uses `python/cudaq_algorithms/stateprep/_hartree_fock.py`; remaining
device kernels use `python/cudaq_algorithms/stateprep/_kernels.py`; exports are
in `python/cudaq_algorithms/stateprep/__init__.py`. Authoritative tests include
`tests/python/test_stateprep_givens.py`,
`tests/python/test_state_prep_injection.py`, and
`tests/python/test_stateprep_hf_ucc.py`. Current public source/tests are
authoritative and must be checked at use time. Cited assertions are derived
evidence until checked or executed in the current task; compatibility and
hardware behavior require their own scoped execution evidence.

## Premise check (claim → verdict)

Requests about this family often carry one of these premises, in the user's own words. Test
every claim the request makes or assumes against this table before answering; a matching row
is the answer to give, cited by the record path in its last cell, and the records named there
are the ones you open next. Never confirm a premise marked false, unverified, or absent.

| If the request claims or assumes (also phrased as) | Verdict | Say first, then the rest |
| --- | --- | --- |
| A width mismatch between preparation and consumer always throws, with a fixed exception text (what exception do I get if widths differ; confirm it throws) | unverified | **Unverified: exact width is required, no factory-time check exists, and no uniform outcome is established.** The HF-only shape (fixed `X` indices) can silently prepare a wrong state on a wider register. Any simulator run is target-specific evidence, not a library guarantee. Records: `cat <this skill's directory>/references/state-preparation/injection-contract.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-hf-ucc.md` |
| Raw Givens or Slater device kernels raise on bad input (confirm every raw kernel raises before changing the register; mismatched index, angle, phase, or final-phase lists) | false | **None raises: `givens_rotation` on a non-adjacent pair is a silent no-op while `phase_givens_rotation` still applies `rz(phase)`; `slater_determinant` no-ops unless `len(orbital_indices) == 2*len(angles)`; `complex_slater_determinant` no-ops unless its three length predicates hold and silently ignores extra final phases.** Validated injectable route: `make_givens_rotation_schedule` then `slater_determinant_kernel`. Records: `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-givens-rotation.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-phase-givens-rotation.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-slater-determinant.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-complex-slater-determinant.md` |
| `uccgsd`, `upccgsd`, `ceo`, and `fixed_parameter_ucc` validate widths or parallel list lengths, their grouped data are interchangeable, or the reference determinant can be skipped (similar signatures; interchange their grouped data) | false | **`len(pauli_words_list)` drives the loop, so extra theta or coefficient entries are ignored and missing ones fail; there is no single rejection mode; the host check is `validate_fixed_parameter_ucc`.** Same grouped shape (one theta per ordered group, `exp(+i·theta·coefficient·P)` in group and term order) but distinct provenance and content; CEO converts spatial orbitals to twice as many qubits; all act on a caller-prepared reference register and prepare no determinant. Records: `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-uccgsd.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-upccgsd.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-ceo.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-fixed-parameter-ucc.md` |
| `hartree_fock_occupation`, `single_excitation`, `double_excitation`, or `uccsd` reject duplicate, reversed, or out-of-range indices (rely on the device kernels to reject bad indices) | false | **Device bodies perform no bounds, distinctness, or finiteness validation and raise no host exception; validate on the host with `validate_hartree_fock_occupation` and take endpoints from `get_uccsd_excitations`. Reversed or equal `single_excitation` endpoints are outside the demonstrated contract: the device body applies no canonicalization and produces a silently wrong unitary; `double_excitation` needs four distinct in-range indices, canonicalizes each pair, and flips the effective theta when exactly one pair is descending; all of these kernels mutate the caller-owned live register in place and return no host value, and none is the injectable one-register seam.** Records: `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-hartree-fock-occupation.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-single-excitation.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-double-excitation.md` |
| `hartree_fock` accepts a `spin` argument or builds an open-shell layout (call hartree_fock with a spin argument) | false | **The signature is `hartree_fock(qubits: cudaq.qview, num_electrons: int)`; it has no `spin` and fills a contiguous prefix.** Open-shell interleaved references go through `make_hartree_fock_occupation` plus `hartree_fock_occupation`. Record: `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-hartree-fock.md` |
| `get_uccsd_excitations(n, n_e, spin)` returning data means `spin` names a valid physical sector (confirm spin=1 represents the requested sector) | false | **`spin` is 2·S_z and `(num_electrons - spin)` must be even; `get_uccsd_excitations` has no parity guard: its floor division `(num_electrons - spin)//2` silently realizes the spin=2 partition for (8, 4, 1), three alpha and one beta.** `make_hartree_fock_occupation` rejects the mismatch, so run that host check first. Record: `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-uccsd.md` |
| The `uccsd` device kernel equals `make_uccsd_operator_pool` plus `fixed_parameter_ucc`, so they can be swapped or `uccsd` injected as `state_prep` (same ansatz; inject uccsd directly) | false | **Not equivalent: the fixed-parameter path applies `exp(+i·theta·coefficient·P)` per term with no factor of one half, one theta per group and one group per pool operator in pool order (so `len(thetas)` is the same on both paths); the `uccsd` ladder realizes `exp(-i·(theta/2)·c·P)` and negates theta for some double-excitation index patterns, shown for the tested case only, not a general conversion rule.** `uccsd` takes runtime arguments beyond the register, so it is not a one-argument `(qubits: cudaq.qview)` injectable; injectables are `hartree_fock_ucc_kernel`, `slater_determinant_kernel`, or an explicitly written one-argument wrapper. Records: `cat <this skill's directory>/references/state-preparation/ucc-parameterization.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-uccsd.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-kernel-fixed-parameter-ucc.md` |
| The Hartree-Fock reference plus fixed UCC amplitudes case is served by the `fixed_parameter_ucc` or `hartree_fock` device kernels (pick the packaged provider; how does the prepared register reach the consumer) | false | **The packaged injectable providers are `hartree_fock_ucc_kernel` for a Hartree-Fock reference with fixed UCC amplitudes (host validation: exactly one of `num_electrons` or `occupied_orbitals`) and `make_givens_rotation_schedule` then `slater_determinant_kernel` for an orthonormal coefficient matrix (host validation: shape and column normalization); both return a one-argument `(qubits: cudaq.qview)` kernel that the consumer factory calls on the register it allocates. Ordering for both is interleaved spin orbitals under Jordan–Wigner little-endian: qubit i = spin-orbital i, alpha even and beta odd; the coefficient matrix has shape `num_spin_orbitals x num_electrons` with rows in that interleaved order, and `make_givens_rotation_schedule` raises `ValueError` for an empty matrix, ragged rows, more electrons than spin orbitals, a column that is not normalized, or columns that are not orthogonal; `hartree_fock_ucc_kernel` takes `spin = 2 * S_z` and rejects an odd `num_electrons - spin`.** Resource estimates for each provider are in the focused estimator records named below; read them before quoting counts. Records: `cat <this skill's directory>/references/state-preparation/state-preparation-hf-ucc.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-givens-schedule.md`, `cat <this skill's directory>/references/state-preparation/state-preparation-slater-determinant-kernel.md` |
| `make_ceo_operator_pool(n)` takes a qubit count and yields Jordan–Wigner excitations with parity strings | false | **`num_orbitals` counts spatial orbitals, the pool acts on `2n` qubits, and CEO is a coupled-exchange construction with no Jordan–Wigner `Z` parity strings.** Record: `cat <this skill's directory>/references/state-preparation/operator-pool-ceo.md` |
| A packaged preparation wrapped in `cudaq.control` or `cudaq.adjoint` is a supported operation with known phase behavior (confirm this is supported) | unverified | **Unverified: controlled preparation, inverse/adjoint preparation, and global phase under control are documented and tested by no provider.** Label any general CUDA-Q attempt unverified and outside the library contract. Record: `cat <this skill's directory>/references/state-preparation/injection-contract.md` |
| The library offers measurement-assisted, repeat-until-success, reset, or feed-forward preparation with a reported success probability | absent | **Absent: providers are unitary, allocate no ancilla, perform no measurement, and report no success probability.** Do not hand-roll one and present it as a library API. Record: `cat <this skill's directory>/references/state-preparation/injection-contract.md` |
| `BlockEncoding` conformance means a consumer, or the caller's own encoding, accepts `state_prep` (my consumer already allocates the system register; confirm all of these consumers, including mine, accept the preparation kernel) | false | **No protocol member mentions `state_prep`; support is per consumer: `PauliLCU.encode_kernel`, `Walk.kernel`, `QSVT.kernel`, and `Trotter.kernel` each allocate the fresh all-zero system register, call the one-argument `(qubits: cudaq.qview)` preparation on it, then apply the operation; a caller's own encoding or consumer is unverified until its source is checked, so do not confirm it.** Exact width is required and unchecked at factory time. The `state: cudaq.State` input is a different, simulation-only path. Record: `cat <this skill's directory>/references/state-preparation/injection-contract.md` |
| The skill should choose or optimize the amplitudes | out of scope | **Pools return ordered `cudaq.SpinOperator` generators; amplitude selection or optimization is outside the primitive contract; do not invent an optimizer API.** Record: `cat <this skill's directory>/references/state-preparation/operator-pools.md` |
| One provider is faster, or the records verify a CUDA-Q version (which is faster on the CUDA-Q version you verified) | unverified | **Records declare only the dependency range `cudaq >=0.15.0,<0.16` and carry no verified-version field; the runtime version you observed in the session is a measurement, not a record fact.** Resource estimators return formula-level logical counts, not runtime; rank nothing you did not execute in the session. |

Boundaries to restate whenever they apply, marked "from record": exact width, unchecked at
factory time; the `(qubits: cudaq.qview)` one-register seam and which symbols are not it;
Jordan–Wigner little-endian layout with qubit i = spin-orbital i and alpha even / beta odd;
host validation lives in the helpers, never in the device kernels.
