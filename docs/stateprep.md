# State preparation

Chemistry-style state-preparation kernels and their host-side helpers,
implemented as a pure-Python peer of the LCU/QSVT and Trotter primitives:
UCC-family ansatz kernels, operator pools and Pauli-list converters,
Hartree-Fock reference determinants, fixed-parameter UCC products, and
Givens-rotation Slater determinants. Requires only the `cudaq` Python
package. Every preparation is (or can be packaged as) a
`(qubits: cudaq.qview)` kernel, which makes it directly injectable into
the other primitives via their `state_prep` parameter (see
[State preparation injection](#state-preparation-injection)).

```
python/cudaq_algorithms/stateprep/
  __init__.py       the public surface (everything re-exported here)
  _kernels.py       device kernels: uccsd, uccgsd, upccgsd, ceo,
                    single/double excitations, hartree_fock,
                    hartree_fock_occupation, fixed_parameter_ucc
  _pools.py         host side: excitation enumeration, operator pools,
                    grouped Pauli-list converters
  _hartree_fock.py  host side: occupation builders, pool-to-Pauli-list
                    conversion, hartree_fock_ucc_kernel factory,
                    resource estimators
  _givens.py        Givens-rotation Slater determinants
                    (lands with the Givens branch, see below)
tests/python/
  test_stateprep.py, test_stateprep_kernels.py, test_operator_pools.py,
  test_stateprep_hf_ucc.py, test_stateprep_givens.py (Givens branch)
examples/stateprep/
  hartree_fock_ucc.py                fixed-parameter UCCSD vs. dense
                                     matrix exponentials
  givens_slater_determinant.py       Slater determinants (Givens branch)
```

## Ansatz kernels and operator pools

The `uccsd`, `uccgsd`, `upccgsd`, and `ceo` device kernels are
`@cudaq.kernel` functions with the same API as the former compiled
bindings, composable from user kernels:

```python
import cudaq
from cudaq_algorithms import stateprep

@cudaq.kernel
def ansatz(thetas: list[float], num_electrons: int, spin: int):
    qubits = cudaq.qvector(num_qubits)
    for i in range(num_electrons):
        x(qubits[i])                    # Hartree-Fock reference
    stateprep.uccsd(qubits, thetas, num_electrons, spin)
```

`uccsd` computes its excitation indices inline; its parameter order is
fixed by `get_uccsd_excitations` (singles alpha, singles beta, mixed
doubles, alpha doubles, beta doubles), and `thetas` must hold
`get_num_uccsd_parameters(num_qubits, num_electrons, spin)` entries.

`uccgsd`, `upccgsd`, and `ceo` instead consume grouped Pauli data — one
variational parameter per group — produced on the host:

```python
words, coeffs = stateprep.get_uccgsd_pauli_lists(num_qubits)
stateprep.uccgsd(qubits, thetas, words, coeffs)      # inside a kernel

stateprep.get_upccgsd_pauli_lists(num_qubits, only_doubles=False)
stateprep.get_ceo_pauli_lists(num_orbitals)          # spatial orbitals!
```

The pools themselves are available as `cudaq.SpinOperator` lists for
ADAPT-style workflows or custom conversion:

```python
stateprep.make_uccsd_operator_pool(num_qubits, num_electrons, spin=0)
stateprep.make_uccgsd_operator_pool(num_qubits, only_singles=False,
                                    only_doubles=False)
stateprep.make_upccgsd_operator_pool(num_qubits, only_doubles=False)
stateprep.make_ceo_operator_pool(num_orbitals)       # spatial orbitals!
```

Device kernels have no error channel, so they perform no input
validation — the host helpers are where the contracts are enforced.

## Hartree-Fock references and fixed-parameter UCC

The reference determinant those ansatz kernels are applied to, plus a
non-variational UCC product on top of it (amplitudes fixed by, e.g., a
classical pre-optimization — no optimizer in the loop):

```python
# Device kernels, composable from user kernels:
stateprep.hartree_fock(qubits, num_electrons)             # closed shell
stateprep.hartree_fock_occupation(qubits, occupied)       # explicit/open shell
stateprep.fixed_parameter_ucc(qubits, thetas, words, coeffs)

# Host helpers:
occ = stateprep.make_hartree_fock_occupation(num_qubits, num_electrons,
                                             spin=2)
stateprep.validate_hartree_fock_occupation(num_qubits, occ)

pool = stateprep.make_uccsd_operator_pool(num_qubits, num_electrons)
words, coeffs = stateprep.get_fixed_parameter_ucc_pauli_lists(
    pool, num_qubits)                  # works for ANY operator pool
stateprep.validate_fixed_parameter_ucc(num_qubits, thetas, words, coeffs)
```

`make_hartree_fock_occupation` builds the occupied spin-orbital list:
contiguous `{0, ..., num_electrons - 1}` for closed shell, and for
`spin > 0` the interleaved alpha (even) / beta (odd) layout of
`get_uccsd_excitations` — e.g. 4 electrons at spin 2 occupy
`{0, 1, 2, 4}`, not `{0, 1, 2, 3}` — so the reference lines up with a
UCCSD pool built at the same spin.

`get_fixed_parameter_ucc_pauli_lists` converts any operator pool to the
grouped form the kernel takes, dropping terms with
`|coefficient| <= coefficient_tolerance` and rejecting complex
coefficients (`exp_pauli` angles are real).

### The `hartree_fock_ucc_kernel` factory

The packaged form: a `(qubits: cudaq.qview)` kernel preparing the
reference determinant and applying the fixed-amplitude UCC product —
directly injectable as a `state_prep` kernel.

```python
prep = stateprep.hartree_fock_ucc_kernel(
    num_qubits, parameters, words, coeffs,
    num_electrons=num_electrons, spin=0)      # canonical reference
prep = stateprep.hartree_fock_ucc_kernel(
    num_qubits, parameters, words, coeffs,
    occupied_orbitals=[0, 1, 2, 4])           # explicit reference
```

Provide exactly one of `num_electrons` (with optional `spin`) or
`occupied_orbitals`. The returned kernel expects a `num_qubits`-wide
register in |0...0>. Validation runs at factory time; the grouped data
is flattened into per-rotation angles before capture (nested lists
marshal as kernel *arguments* but cannot be closure-*captured*), and
empty occupation/rotation lists select a kernel shape with no empty
captures (empty list captures fail to launch,
[cuda-quantum#4847](https://github.com/NVIDIA/cuda-quantum/issues/4847)).

### Resource estimation

```python
stateprep.estimate_hartree_fock_resources(num_qubits, num_electrons,
                                          spin=0)
stateprep.estimate_hartree_fock_occupation_resources(num_qubits, occ)
# -> HartreeFockResourceEstimate(num_qubits, num_electrons, num_x_gates)

stateprep.estimate_fixed_parameter_ucc_resources(num_qubits, words)
# -> FixedParameterUccResourceEstimate(num_qubits, num_excitations,
#        num_pauli_rotations, max_pauli_rotations_per_excitation)
```

Like `TrotterResourceEstimate`, these are frozen dataclasses counting
logical operations before transpilation, not hardware gate counts.

## Givens-rotation Slater determinants (on the Givens branch)

> This section documents the API landing with the sibling branch
> `features/add_givens_rotation_state_prep_python`; none of it is
> available until that branch merges.

Prepares the Slater determinant of an orthonormal occupied-orbital
matrix `Q` (`num_spin_orbitals x num_electrons`) on the Jordan-Wigner /
little-endian layout: the amplitude of basis state `|S>` with occupied
set `S` is `det(Q[S, :])`, up to a global phase.

```python
schedule = stateprep.make_givens_rotation_schedule(orbital_coefficients)
schedule.num_spin_orbitals, schedule.num_electrons, schedule.is_complex
schedule.rotations                    # adjacent rotations, application order
schedule.final_phases                 # one per electron (complex only)

prep = stateprep.slater_determinant_kernel(schedule)   # (qubits: qview)

stateprep.estimate_givens_resources(schedule)  # GivensResourceEstimate
```

`make_givens_rotation_schedule` reduces `Q` to the computational-basis
determinant with adjacent (nearest-neighbor) Givens row rotations on the
host; the kernels apply the inverse rotations in reverse order to the
determinant `|1...10...0>`. Real and complex matrices dispatch
automatically (a complex dtype routes complex even when every value is
real); complex schedules carry a relative phase per rotation plus one
final phase per electron. `validate_givens_rotation_schedule` guards
hand-built schedules; built schedules always pass.

`slater_determinant_kernel(schedule)` is the packaged, injectable form.
The module-level device kernels remain composable from user kernels,
with `get_givens_rotation_indices` / `..._angles` / `..._phases`
supplying the flattened arrays that cross the kernel boundary:

```python
stateprep.slater_determinant(qubits, indices, angles, num_electrons)
stateprep.complex_slater_determinant(qubits, indices, angles, phases,
                                     final_phases, num_electrons)
stateprep.givens_rotation(qubits, theta, first, second)         # adjacent
stateprep.phase_givens_rotation(qubits, theta, phase, first, second)
```

`estimate_givens_resources` returns a `GivensResourceEstimate`
(`num_givens_rotations`, `num_exp_pauli_calls`, `num_phase_rotations`,
`two_qubit_gate_count_proxy`, `depth_proxy`) — decomposition-independent
proxies, not transpiled gate counts.

## State preparation injection

Every kernel factory in the package family — `PauliLCU.encode_kernel`,
`Walk.kernel` (and its adjoint/controlled variants), `QSVT.kernel`, and
`Trotter.kernel` — takes an optional `state_prep` kernel with signature
`(qubits: cudaq.qview)`. The returned circuit is then **zero-argument**:
the system register is allocated in |0...0>, `state_prep` runs on it,
and the operation follows — directly sampleable and fully synthesizable,
with no statevector anywhere.

Any `(qubits: cudaq.qview)` preparation qualifies, including the
factory outputs of this package — `hartree_fock_ucc_kernel(...)` and
`slater_determinant_kernel(...)` — and the raw kernels wrapped by the
caller:

```python
import cudaq
from cudaq_algorithms import PauliLCU, Trotter, stateprep

pool = stateprep.make_uccsd_operator_pool(4, 2)
words, coeffs = stateprep.get_fixed_parameter_ucc_pauli_lists(pool, 4)
prep = stateprep.hartree_fock_ucc_kernel(4, [0.11, -0.04, 0.28], words,
                                         coeffs, num_electrons=2)

hamiltonian = {"ZIII": 0.5, "IZII": 0.5, "XXII": 0.25}

# Trotter time evolution from the UCC state:
kernel = Trotter(hamiltonian).kernel(time=0.8, steps=4, state_prep=prep)
counts = cudaq.sample(kernel)

# The same prep drives the block-encoding/walk/QSVT factories:
enc = PauliLCU(hamiltonian)
kernel = enc.encode_kernel(state_prep=prep)
```

Contract (shared with the other primitives): `state_prep` acts only on
the system register it is handed, which arrives in |0...0> with the
factory's system width (`num_qubits` for `Trotter`, `num_system` for the
encodings) — not verifiable at factory time, so a mismatched prep fails
at launch. The `stateprep` factories bake their width in at construction
(`num_qubits` for `hartree_fock_ucc_kernel`, `num_spin_orbitals` for
`slater_determinant_kernel`), so build them to match.
`tests/python/test_state_prep_injection.py` exercises the contract
end-to-end.

## Package conventions

- **Qubit layout**: interleaved spin orbitals — alpha spin orbitals on
  even qubits, beta on odd (spatial orbital `i` maps to qubits `2i` /
  `2i + 1`). All pools, excitation enumerations, and Hartree-Fock
  occupations share it.
- **`spin` semantics**: `spin` is the difference between alpha and beta
  electron counts (twice the total S_z), so
  `n_beta = (num_electrons - spin) // 2` and
  `n_alpha = num_electrons - n_beta`. `spin == 0` selects the
  closed-shell forms; `spin > 0` requires an even `num_qubits` and
  produces the interleaved open-shell occupations and excitations.
- **Error types**: the two error cases the compiled bindings defined keep
  their historical `RuntimeError` (odd qubit count and
  odd-electrons-at-spin-0 in `get_uccsd_excitations`); every guard added
  in the pure implementation raises `ValueError`.
- **CEO orbital counts**: the CEO helpers (`make_ceo_operator_pool`,
  `get_ceo_pauli_lists`) take `num_orbitals` in *spatial* orbitals — the
  pool acts on `2 * num_orbitals` qubits — matching the compiled API.
  Everything else in the package counts qubits (spin orbitals).

## Testing

`tests/python/test_stateprep.py`, `test_stateprep_kernels.py`, and
`test_operator_pools.py` pin the ansatz kernels and pools against the
former compiled implementation (bit-identical circuits and operator
coefficients). `test_stateprep_hf_ucc.py` validates the Hartree-Fock and
fixed-parameter UCC surface against dense matrix exponentials of the
operator pools, including open-shell occupations and input guards
(injection through the primitives is covered by
`test_state_prep_injection.py`); `test_stateprep_givens.py` (on
the Givens branch) does the same for Slater determinants against dense
`det(Q[S, :])` amplitudes.
