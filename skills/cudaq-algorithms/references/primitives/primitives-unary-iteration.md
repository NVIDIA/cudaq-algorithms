# Unary iteration walk factory

Operation + object: **apply** a **per-address gate list `U_k` exactly when
the address register holds `k`, i.e. the indexed SELECT
`sum_k |k><k| (x) U_k`**.

## Identity and classification

- Public entry point: `cudaq_algorithms.primitives.unary_iteration_kernels(
  num_address_bits, num_items, body, *, controlled=False,
  include_adjoint=True, num_work=None)`; result type
  `cudaq_algorithms.primitives.UnaryIterationKernels` (frozen dataclass).
- Source/tests: `python/cudaq_algorithms/primitives/_unary_iteration.py` and
  `tests/python/test_primitives_unary_iteration.py`,
  `test_primitives_body_gates.py`, `test_primitives_payload_sum.py`,
  `test_primitives_interpreter.py`, `test_primitives_resources.py`.
- Kind/role/layers: quantum operation, computational leaf; host factory that
  flattens the walk into an instruction tape, flat device-kernel interpreter.
- Input/output: widths plus a factory-time `body(k)` callback -> minted
  `kernel`, optional `kernel_adj`, register widths, and `toffoli_count`.
- Source provenance: [Source lookup](../source-provenance.md) gives shared current-source paths.

## Scientific, input, and output contract

The factory validates `num_address_bits` as a positive integer and
`num_items` as an integer in `[1, 2^num_address_bits]`; addresses
`k >= num_items` are never entered and the walk acts as the identity on
them. `body` is called exactly once per address, in address order, **on the
host at factory time**; it returns the gate list for `U_k` as data, which is
baked into the kernel (an empty list means `U_k = I`). The body never runs at
circuit time.

Body instruction set (operands index the `target`/`work` views; the three
basic gates are implicitly controlled on the address-`k` leaf line):

- `("x", t)`, `("y", t)`, `("z", t)`: leaf-controlled Pauli on `target[t]`.
- Free gates with no leaf control, executed at every address and sound only
  as conjugation pairs closed inside one body: `("free_x", t)`,
  `("free_cx", a, b)`, `("and_tt", a, b, w)` (Toffoli `target[a], target[b]
  -> work[w]`), `("and_wt", v, t, w)`, `("copy_tw", t, w)`.
- Leaf-referencing core gates: `("x_w", w, t)` (`x.ctrl(leaf, work[w],
  target[t])`), `("z_w", w)`, `("sign",)` (a `-1` phase on the address-`k`
  branch).

An unsupported gate name, wrong operand count, or negative index raises
`ValueError` at factory time. Any body using `work` operands adds a trailing
clean `work` view; `num_work=None` infers the width from the body, an
explicit value must cover that usage (else `ValueError`) and forces the view
into the signature even when unused.

Kernel signatures, all little-endian (qubit 0 = LSB):

- `kernel(address, ladder, target[, work])`; with `controlled=True` a
  leading one-qubit `control` view roots the tree and the walk is the
  identity when the control is `|0>`.
- `kernel_adj` is the hand-written inverse (the same interpreter over the
  reversed tape; every emitted gate is self-inverse); it is `None` when
  `include_adjoint=False`, which is appropriate only for an involutive walk
  such as the X-only QROM write.

`ladder` is `num_ladder == num_address_bits` clean ancillas, `|0>` in and
out; the leaf line is `ladder[num_address_bits - 1]`. Level 1 examines the
most-significant address bit. The Y body follows `|k>|0> -> i|k>|1>`.
Result fields: `kernel`, `kernel_adj`, `num_address`, `num_ladder`,
`num_items`, `controlled`, `toffoli_count`, `num_work`, `ops`;
`describe()` decodes the tape to one gate per line with `# Toffoli` tags
whose count equals `toffoli_count`.

## Boundaries and resources

`toffoli_count` is the exact number of two-controlled `x` gates in the tape
(walk Toffolis plus any `and_tt`, `and_wt`, `x_w` body gates). On a full tree
of `N = 2^num_address_bits` addresses the uncontrolled walk costs exactly
`3N/2 - 5` (`N >= 8`; 2 at `N = 4`, 0 at `N = 2`) and the controlled walk
`3N/2 - 1`; partial trees cost less. Pauli payload size does not change the
count. The naive compute/uncompute unitary walk costs `2N - 4`; the
`N - 1` of Babbush et al. prices measurement-and-fixup uncomputation that is
not provided. Source states the T-count comparison is narrower (about
`7.5N` versus `8N` T gates, since fused Toffolis target dirty lines); that is
a derived statement, not a measured one. All counts are logical, pinned by
the tests against `cudaq.estimate_resources(...).count_controls("x", 2)`,
and are not transpiled depth, runtime, memory, or hardware cost.

The minted kernels are flat (no nested kernel calls), which is what makes an
external `cudaq.control` over the plain kernel work; the test pins it equal
to the `controlled=True` route and to an analytic reference. `cudaq.adjoint`
is not used (`kernel_adj` is the supported inverse). The factory retains
every minted kernel for the process lifetime (a keep-alive registry against
CUDA-Q kernel-id recycling); the cost is a few KB per call.

## Oracle and runnable evidence

Build the dense `sum_k |k><k| (x) U_k` (identity on `k >= num_items`) from
NumPy Paulis in little-endian Kronecker order and compare column by column
through basis-state readout, or check a uniform address superposition picks
up the per-address phase kick; verify `kernel` then `kernel_adj` restores a
non-involutive body; check `controlled=True` at control `|0>`, `|1>`, and a
superposed control; confirm `toffoli_count` equals the tape's `# Toffoli`
lines and the compiled `ccx` count.

Run the authoritative statevector cases from the repository root:

```bash
PYTHONPATH=python pytest -q tests/python/test_primitives_unary_iteration.py \
  -k "phase_kick or adjoint_composes or controlled_variant or validation_raises"
```

`test_tape_is_select_for_every_tree_shape` verifies the tape classically for
every width up to 6 and every partial size;
`test_tape_toffoli_counts_match_documented_formulas` pins the closed forms;
`test_primitives_body_gates.py` pins each extended gate and the external
`cudaq.control` route. Current public source and tests are authoritative and
must be rechecked at use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
