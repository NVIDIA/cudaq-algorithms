# Circuit primitives - family front door

Family operation + object: **build** a **strictly unitary fault-tolerant
circuit component (coherent table lookup, indexed SELECT walk, reversible
integer arithmetic, weighted-index PREPARE) on caller-owned little-endian
registers**.

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| apply / a per-address gate list `U_k` exactly when an address register holds `k` (`sum_k |k><k| (x) U_k`) | `cudaq_algorithms.primitives.unary_iteration_kernels`, result type `UnaryIterationKernels` | quantum operation; host factory + flat device kernel | none; caller-owned `address`, `ladder`, `target`[, `work`] views | [unary iteration](primitives-unary-iteration.md) |
| look up / a classical non-negative integer table into an output register, `|k>|y> -> |k>|y XOR data[k]>` | `cudaq_algorithms.primitives.QROM` | quantum operation; host construction + flat device kernel | none; caller-owned `address`, `ladder`, `output` views | [QROM](primitives-qrom.md) |
| add, subtract, or compare / unsigned integers held in little-endian registers, in place | `add_register`, `subtract_register`, `add_constant`, `subtract_constant`, `cmp_ge_constant`, `cmp_ge_register`, `cmp_gt_register`, `qft`, `iqft`, `phase_add_constant`, `add_constant_qft`, `subtract_constant_qft`, `cmp_ge_constant_qft_shift`, `cmp_ge_constant_qft_shift_adj` | quantum operation; `@cudaq.kernel` device kernels called inside a caller kernel | none; caller-owned pairwise-disjoint views | [integer arithmetic](primitives-arithmetic.md) |
| prepare / an index register whose amplitude magnitudes squared equal `w / lambda` (coherent alias sampling, with entangled garbage) | `cudaq_algorithms.primitives.AliasSamplingPrepare` | quantum operation; host preprocessing + composed device kernels | none; caller-owned `index`, `garbage` views; not a `BlockEncoding` member | [alias-sampling PREPARE](primitives-alias-sampling.md) |

Every primitive is strictly unitary: no mid-circuit measurement, no
measurement-based ancilla uncomputation, and a hand-written inverse wherever
one is needed (`kernel_adj`, `subtract_*`, `iqft`, `*_adj`,
`adjoint_kernel()`); `cudaq.adjoint` is not used by the package on any of
them. Clean ancillas (`ladder`, `work`, `carry`) are `|0>` on entry and are
returned to `|0>`. All registers are little-endian: qubit 0 of a view is the
least-significant bit, matching the package-wide
[ordering convention](../conventions/ordering.md).

Resource facts are **exact logical Toffoli counts** of the emitted circuit:
`toffoli_count` on `UnaryIterationKernels`, `QROM`, and
`AliasSamplingPrepare.qrom` counts the two-controlled `x` gates in the
instruction tape, and the arithmetic kernels document closed forms
(`2n - 2`, `2n - 1`, zero for the Draper family). The tests pin each number
against `cudaq.estimate_resources(...).count_controls("x", 2)`. None of these
is a T count, a transpiled gate count, circuit depth, runtime, memory, or a
measured hardware cost; see [resource levels](../conventions/resource-levels.md).
The papers' headline counts (`N - 1` for unary iteration,
`ceil(N/B) + b(B - 1)` for QROAM) assume measurement-and-fixup uncomputation
and are **not** what these kernels cost; the coherent counts are
`3N/2 - 5` for a full uncontrolled walk (`N >= 8`) and the formulas on the
QROM record.

Import the subpackage directly: `from cudaq_algorithms.primitives import
QROM, AliasSamplingPrepare, unary_iteration_kernels, add_register, ...`.
Nothing in this family is re-exported from the `cudaq_algorithms` package
root. No packaged algorithm family (`PauliLCU`, `Walk`, `QSVT`, `Trotter`,
state preparation) imports this subpackage in current source; composing a
primitive into a SELECT, PREPARE, or block encoding is caller work, and
`AliasSamplingPrepare` is not a `BlockEncoding` and has no
`prepare_kernel()`/`select_observable()` members.

Shared source is `python/cudaq_algorithms/primitives/` (`_unary_iteration.py`,
`_qrom.py`, `_arithmetic.py`, `_alias_sampling.py`, `__init__.py`);
authoritative behavior is in `tests/python/test_primitives_*.py`. The Sphinx
API page renders the subpackage docstring via `automodule`; `@cudaq.kernel`
objects do not autodoc, so each kernel's own docstring is its rendered
contract. Current public source and tests are authoritative and must be
rechecked at use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
Resource claims also require rechecking at use time; runtime execution and
hardware measurement remain unverified.

## Premise check (claim -> verdict)

Test every claim the request makes or assumes against this table before answering; a matching row is
the answer to give, cited by the record path in its last cell, and the records named there are read
with the `cat` command shown.

| If the request claims or assumes | Verdict | Say (demanded words first) |
| --- | --- | --- |
| `QROM` or unary iteration costs `N - 1` Toffolis, or uses measurement-based uncomputation | false | **These primitives are strictly unitary; the fused walk costs exactly `3N/2 - 5` Toffolis on a full uncontrolled tree (`N >= 8`; 2 at `N = 4`, 0 at `N = 2`), `3N/2 - 1` controlled, fewer on partial trees, and `QROM.toffoli_count` reports the exact minted count (`W(address_bits, len(data))` for `"select"`, `2 W(high, ceil(N/B)) + 2 output_bits (B - 1)` for `"select_swap"`); the papers' `N - 1` and `ceil(N/B) + b(B-1)` price measurement-and-fixup uncomputation that is not provided.** Records: `cat <this skill's directory>/references/primitives/primitives-qrom.md`, `cat <this skill's directory>/references/primitives/primitives-unary-iteration.md` |
| `toffoli_count` is a T count, transpiled gate count, or hardware cost | false | **`toffoli_count` is the exact number of logical two-controlled `x` gates in the emitted instruction tape, pinned by the tests against `cudaq.estimate_resources(...).count_controls("x", 2)`; it is not a T count, depth, runtime, memory, or measured cost.** Records: `cat <this skill's directory>/references/primitives/primitives.md`, `cat <this skill's directory>/references/conventions/resource-levels.md` |
| `AliasSamplingPrepare` yields the clean state `sum_k sqrt(w_k/lambda) |k>|0...0>` | false | **The PREPARE leaves alias, keep, reference, and flag registers entangled with the index; only the index amplitude magnitudes squared equal the realized table distribution `W_k/(K 2^mu)` (within `discretization_bound = 1.5/(K 2^mu)` of `w/lambda`), and the garbage vectors are not orthogonal, so the reduced index state is not `diag(p_k)`; it is sound only inside a `PREPARE ... PREPARE^dagger` sandwich over the same registers using `adjoint_kernel()`.** Record: `cat <this skill's directory>/references/primitives/primitives-alias-sampling.md` |
| The primitives are reachable from the package root, or `PauliLCU`/`Walk` use them internally | false | **Import `cudaq_algorithms.primitives` directly; nothing is re-exported from the package root, and no packaged algorithm family imports the subpackage in current source.** Record: `cat <this skill's directory>/references/primitives/primitives.md` |
| `cudaq.adjoint` can invert any primitive kernel, or `add_register(a, a, carry)` is defined | false | **Inverses are hand-written (`kernel_adj`, `subtract_register`, `subtract_constant`, `iqft`, `subtract_constant_qft`, `cmp_ge_constant_qft_shift_adj`, `adjoint_kernel()`); source states `cudaq.adjoint` is off-limits for the CDKM family and the walk kernels (cuda-quantum#4897/#4898), and the arithmetic kernels never alias-check, so overlapping views are undefined.** Records: `cat <this skill's directory>/references/primitives/primitives-arithmetic.md`, `cat <this skill's directory>/references/primitives/primitives-unary-iteration.md` |
| Registers are big-endian, or `constant_bits[0]` is the most-significant bit | false | **Every register is little-endian: qubit 0 is the least-significant bit, `constant_bits[k]` is bit `k` of `K`, and `qft` omits the bit-reversal swaps so qubit `t` carries phase index `2^(t+1)`.** Records: `cat <this skill's directory>/references/primitives/primitives-arithmetic.md`, `cat <this skill's directory>/references/conventions/ordering.md` |
| `cmp_ge_constant_qft_shift` leaves `x_reg` unchanged like `cmp_ge_constant` | false | **The Draper comparator XORs `out[0] ^= (x >= K)` and leaves `x_reg` shifted to `(x - K) mod 2^n` for `K >= 1`; restore with `cmp_ge_constant_qft_shift_adj` or `add_constant_qft(x_reg, K)`; the CDKM `cmp_ge_constant` restores `x_reg`, `work`, and `carry`.** Record: `cat <this skill's directory>/references/primitives/primitives-arithmetic.md` |
| A controlled QROM or controlled PREPARE is a packaged contract | absent | **Only `unary_iteration_kernels(..., controlled=True)` mints a controlled walk (leading one-qubit `control` view); `QROM` and `AliasSamplingPrepare` have no `controlled` option, and `AliasSamplingPrepare` kernels call sub-kernels so source directs that they not be placed under `cudaq.control`; an external `cudaq.control` over a plain unary-iteration kernel is tested, over a QROM kernel it is unverified.** Records: `cat <this skill's directory>/references/primitives/primitives-unary-iteration.md`, `cat <this skill's directory>/references/primitives/primitives-alias-sampling.md` |

## Workflow

1. Fix the register layout first: widths, which views are clean ancillas,
   and the little-endian bit order of every integer. Views need not be
   contiguous for the minted lookups, but every view passed to one
   arithmetic kernel must be pairwise disjoint.
2. Do all data-dependent construction on the host: `unary_iteration_kernels`
   bakes the `body(k)` gate lists, `QROM` bakes the table and picks a variant,
   `AliasSamplingPrepare` runs integer Vose preprocessing. The returned
   kernels carry registers only ([kernel boundary](../conventions/kernel-boundaries.md)).
3. Read the price from the object (`toffoli_count`, `num_ladder`, `num_work`,
   `num_garbage`, `qrom.variant`) before allocating; compare variants by that
   exact logical count, not by paper headlines.
4. Allocate registers inside a caller `@cudaq.kernel` and pass views; the
   arithmetic kernels are called the same way, with constants either as
   little-endian bit lists (`add_constant`, `cmp_ge_constant`) or integers
   (`add_constant_qft`, `phase_add_constant`, `cmp_ge_constant_qft_shift`).
5. Uncompute with the paired inverse on the same views, in the reverse order
   of application; never discard or measure a dirty ancilla.

## Verification

Compare the full statevector with an independent classical model: a
basis-state readout per address or operand, a dense `sum_k |k><k| (x) U_k`
reference, or a brute-force enumeration of the alias branches. Check that
clean ancillas are exactly `|0>` by slicing the statevector tensor. Pin
costs with `cudaq.estimate_resources` on a harness kernel and compare to the
object's `toffoli_count` or the documented closed form; both are logical
counts, not hardware figures.

Save this checkpoint as a `.py` file with the package importable. It builds
a QROM, reads out every address (out-of-range addresses read zero), checks
the Toffoli formula, and checks an alias-sampling marginal against the
object's own exact table.

```python
import cudaq
import numpy as np
from cudaq_algorithms.primitives import QROM, AliasSamplingPrepare
from cudaq_algorithms.primitives._unary_iteration import _walk_toffoli_count

cudaq.set_target("qpp-cpu")
data = [5, 3, 0, 6, 7]
qrom = QROM(data, address_bits=3, output_bits=3)   # variant="auto"
assert qrom.variant == "select" and qrom.num_ladder == 3
assert qrom.toffoli_count == _walk_toffoli_count(3, len(data)) == 6
lookup, num_ladder = qrom.kernel, qrom.num_ladder
for address in range(8):

    @cudaq.kernel
    def run():
        address_reg = cudaq.qvector(3)
        ladder = cudaq.qvector(num_ladder)
        output = cudaq.qvector(3)
        for b in range(3):
            if ((address >> b) & 1) == 1:
                x(address_reg[b])
        lookup(address_reg, ladder, output)

    state = np.array(cudaq.get_state(run))
    word = data[address] if address < len(data) else 0
    expected = np.zeros(1 << 9, dtype=complex)
    expected[address + (word << 6)] = 1.0   # ladder back to |0>
    np.testing.assert_allclose(state, expected, atol=1e-12)

prep = AliasSamplingPrepare([0.7, 0.2, 1.4], mu=2)
assert prep.num_index == 2 and prep.num_garbage == 2 * 2 + 2 * 2 + 2
prepare, m, g = prep.kernel(), prep.num_index, prep.num_garbage

@cudaq.kernel
def run_prep():
    index = cudaq.qvector(m)
    garbage = cudaq.qvector(g)
    prepare(index, garbage)

state = np.array(cudaq.get_state(run_prep))
marginal = np.sum(np.abs(state.reshape(-1, 1 << m))**2, axis=0)
np.testing.assert_allclose(marginal, prep.table_probabilities, atol=1e-10)
assert np.max(np.abs(marginal - prep.probabilities)) <= prep.discretization_bound
print("QROM readout, Toffoli formula, and alias marginal verified")
```

Source and further checkpoints: `python/cudaq_algorithms/primitives/`,
`tests/python/test_primitives_qrom.py`, `test_primitives_unary_iteration.py`,
`test_primitives_body_gates.py`, `test_primitives_payload_sum.py`,
`test_primitives_interpreter.py`, `test_primitives_resources.py`,
`test_primitives_arithmetic.py`, and `test_primitives_alias_sampling.py`
(exhaustive readouts, dense SELECT references, compiler-pinned Toffoli
counts). No runnable example under `docs/sphinx/examples/python/` uses this
family.
