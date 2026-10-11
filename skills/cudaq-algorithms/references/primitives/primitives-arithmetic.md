# Reversible integer arithmetic kernels

Operation + object: **add, subtract, or compare** **unsigned integers held
in little-endian quantum registers, in place**.

## Identity and classification

- Public entry points (all `@cudaq.kernel` device kernels in
  `cudaq_algorithms.primitives`, called inside a caller kernel):
  - CDKM/Cuccaro ripple-carry: `add_register(a, b, carry)`,
    `subtract_register(a, b, carry)`, `add_constant(target, constant_bits,
    work, carry)`, `subtract_constant(target, constant_bits, work, carry)`,
    `cmp_ge_constant(x_reg, complement_bits, k_is_zero, work, carry, out)`,
    `cmp_ge_register(a, b, carry, out)`, `cmp_gt_register(a, b, carry, out)`.
  - Draper QFT (no work qubits): `qft(reg)`, `iqft(reg)`,
    `phase_add_constant(reg, constant)`, `add_constant_qft(reg, constant)`,
    `subtract_constant_qft(reg, constant)`,
    `cmp_ge_constant_qft_shift(x_reg, constant, invert, out)`,
    `cmp_ge_constant_qft_shift_adj(x_reg, constant, invert, out)`.
- Source/tests: `python/cudaq_algorithms/primitives/_arithmetic.py` and
  `tests/python/test_primitives_arithmetic.py`.
- Kind/role/layers: quantum operation, computational leaf; device kernels
  only (no host factory, no host validation).
- Input/output: caller-owned `cudaq.qview`s plus constants -> in-place
  register update; no return value.
- Source provenance: [Source lookup](../source-provenance.md) gives shared current-source paths.

## Scientific, input, and output contract

All registers are little-endian: `register[0]` is the least-significant bit,
`constant_bits[k]` is bit `k` of `K`. The width `n` is read from the
**first** operand (`a.size()`, `target.size()`, `x_reg.size()`, `reg.size()`);
a longer second register is silently reduced to its low `n` bits. Views
passed to one kernel must be pairwise disjoint: the kernels never
alias-check, and overlapping views (for example `add_register(a, a, carry)`
or a `work` view sharing qubits with `target`) are undefined. `carry`,
`work`, and (where present) the comparator internals must be `|0>` on entry
and return to `|0>`; `out` is a one-qubit flag that is XOR-loaded from any
input state.

CDKM family semantics:

- `add_register`: `b <- (a + b) mod 2^n`; `a` and `carry` restored; `a`, `b`
  equal size `n`, `carry` one qubit.
- `subtract_register`: `b <- (b - a) mod 2^n`; the gate-reversal of
  `add_register`.
- `add_constant` / `subtract_constant`: `target <- (target +/- K) mod 2^n`;
  `work` is `n` qubits (the constant is X-loaded into it and unloaded). Only
  bits `0 .. n-1` of `constant_bits` are read: a longer list acts as
  `K mod 2^n`; a **shorter** list reads past the end with no bounds check
  (undefined), so pass at least `n` bits.
- `cmp_ge_constant`: `out[0] ^= (x >= K)` with `x_reg` restored.
  `complement_bits` are the `n` little-endian bits of `2^n - K`;
  `k_is_zero = 1` with all-zero bits encodes `K = 0` (always true, a bare X
  on `out`); all-zero bits with `k_is_zero = 0` encodes `K = 2^n` (always
  false). `work` is `n` qubits, `carry` one qubit.
- `cmp_ge_register` / `cmp_gt_register`: `out[0] ^= (a >= b)` /
  `out[0] ^= (a > b)` (unsigned) with `a`, `b`, `carry` restored; no `work`
  register (`b` is complemented in place and restored). Applying either
  twice with the same arguments is the identity.

Draper family semantics:

- `qft` omits the final bit-reversal swaps: after it, qubit `t` holds
  `(|0> + exp(2 pi i x / 2^(t+1)) |1>) / sqrt(2)`; `iqft` is its hand-written
  reversal.
- `phase_add_constant(reg, constant)` adds `K mod 2^n` in that Fourier basis
  via `r1(2 pi (K mod 2^(t+1)) / 2^(t+1))` on qubit `t`, reducing `K` as an
  integer first so large `K` (including `>= 2^53`) stays exact; valid for
  `n <= 62`.
- `add_constant_qft` / `subtract_constant_qft`: `reg <- (reg +/- K) mod 2^n`
  as `qft`, phases, `iqft`; `K` is a Python `int` of any sign.
- `cmp_ge_constant_qft_shift`: `out[0] ^= (x >= K)` (`invert = 1` flags
  `x < K`) and **leaves `x_reg` shifted to `(x - K) mod 2^n`** for
  `K >= 1`; pair with `cmp_ge_constant_qft_shift_adj` on the same arguments
  to restore, or follow with `add_constant_qft(x_reg, K)` to keep the flag
  and restore `x_reg` more cheaply. Preconditions `n <= 61` and
  `0 <= K <= 2^n`; `K > 2^n` is silently wrong; `K <= 0` skips the
  subtraction (`x_reg` left unshifted, `K = 0` is the constant-true flag).

Hand-written inverses are the supported route throughout. Source states
`cudaq.adjoint` fails on the CDKM kernels (cuda-quantum#4897/#4898) and does
work on the Draper kernels, whose inverses are hand-written anyway for a
uniform surface. Every `__all__` name of `_arithmetic` is re-exported from
`cudaq_algorithms.primitives` as the same object; none is on the package
root.

## Boundaries and resources

Documented per-operation costs on `n`-bit operands, pinned by the resource
tests against `cudaq.estimate_resources`:

```text
add_register, subtract_register, add_constant, subtract_constant : 2n - 2 Toffolis (0 at n = 1)
cmp_ge_constant (K >= 1), cmp_ge_register, cmp_gt_register        : 2n - 1 Toffolis; cmp_ge_constant at K = 0: 0
add_constant_qft, subtract_constant_qft : 0 Toffolis; n(n-1) controlled r1, n single r1, 2n H
cmp_ge_constant_qft_shift(_adj), K >= 1 : 0 Toffolis; n(n+1) controlled r1, n+1 single r1, 2(n+1) H; K = 0 emits no rotations
```

The mod-`2^n` adder never forms the top carry-out, so its MAJ/UMA Toffoli
pair cancels (`2n - 2` rather than `2n`); Cuccaro et al.'s `2n - 3` variant
is not adopted. These are logical gate counts of the kernels as written, not
T counts, transpiled depth, runtime, memory, or hardware cost. Not provided:
multiplication, division, signed arithmetic, modular arithmetic beyond
`mod 2^n`, an ancilla-light constant adder, or host-side validation of
widths and preconditions (misuse is undefined behavior, not an exception).
`cudaq.control` over a kernel that calls a sub-kernel (for example
`add_constant_qft`) is tested only when the installed CUDA-Q is `>= 0.16.0`
and skipped otherwise.

## Oracle and runnable evidence

Compare the statevector with classical integer arithmetic over all inputs:
superpose one operand and fix the other for the register adders (a uniform
superposition of both proves nothing), use basis-input truth tables for the
constant ops, superpose the input and read the correlated flag for the
comparators, and pin each inverse by op-then-inverse identity. Count
Toffolis with `cudaq.estimate_resources(kernel, *args).count_controls("x",
2)` on a harness that passes widths and constants as runtime arguments.

Run the authoritative spot checks, inverse, and cost cases from the
repository root:

```bash
PYTHONPATH=python pytest -q tests/python/test_primitives_arithmetic.py \
  -k "spot_checks or cmp_register_twice or cdkm_register_ops_cost or qft_add_constant_costs"
```

`test_register_add_subtract_all_inputs`, `test_cmp_ge_constant_all_inputs`,
`test_cmp_ge_constant_qft_shift_all_inputs`, and
`test_cmp_register_truth_table_exhaustive` cover the exhaustive contracts.
Current public source and tests are authoritative and must be rechecked at
use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
