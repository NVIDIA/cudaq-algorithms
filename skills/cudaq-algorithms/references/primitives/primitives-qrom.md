# QROM coherent table lookup

Operation + object: **look up** a **classical non-negative integer table into
an output register, `|k>|y> -> |k>|y XOR data[k]>`**.

## Identity and classification

- Public entry point: `cudaq_algorithms.primitives.QROM(data, address_bits,
  output_bits, *, variant="auto", block_size=None)`.
- Source/tests: `python/cudaq_algorithms/primitives/_qrom.py` and
  `tests/python/test_primitives_qrom.py`, `test_primitives_resources.py`.
- Kind/role/layers: quantum operation, computational leaf; host construction
  and variant pricing, one flat device kernel.
- Input/output: integer table plus register widths -> object exposing
  `kernel` with signature `(address: qview, ladder: qview, output: qview)`.
- Source provenance: [Source lookup](../source-provenance.md) gives shared current-source paths.

## Scientific, input, and output contract

`QROM` materializes `data` to a list and validates, in order: `address_bits`
and `output_bits` positive integers; `data` non-empty, all entries integral
(a float such as `1.5` raises, including from a single-pass iterable), at
most `2^address_bits` entries, all non-negative, and `max(data) <
2^output_bits`; `variant` in `("auto", "select", "select_swap")`;
`variant="select_swap"` requires `address_bits >= 2`; `block_size` is only
valid with `variant="select_swap"` and must be a power of two in
`[2, 2^(address_bits - 1)]`. Every violation raises `ValueError`.

The minted kernel implements the XOR lookup on little-endian registers
(qubit 0 = LSB): `address` has `address_bits` qubits, `output` has
`output_bits`, and `ladder` is `num_ladder` clean ancillas (`|0>` in, `|0>`
out). Addresses `k >= len(data)` read as zero. The three views may live
anywhere; no contiguity is required. Whatever the variant, the kernel is
exactly its own inverse (the write phase is X-only), so applying `kernel`
again uncomputes the lookup.

Two constructions sit behind `variant`:

- `"select"`: the plain unary-iteration walk whose per-address body CNOTs
  the set bits of `data[k]` into `output`; `num_ladder = address_bits`;
  `block_size` is `None`.
- `"select_swap"`: SELECT-SWAP/QROAM with `B = block_size` entries per
  block; the walk runs over the high `h = address_bits - log2(B)` address
  bits writing each visited block into `B` clean `output_bits`-wide block
  registers, a Fredkin network controlled on the low `log2(B)` address bits
  routes the selected entry into slot 0, CNOTs copy slot 0 into `output`, and
  the routing and writes are undone unitarily. `num_ladder = h +
  B * output_bits`, laid out as walk lines `ladder[0..h)` followed by block
  register `i` at `ladder[h + i*b .. h + (i+1)*b)`. With `block_size=None`
  the power of two minimizing the documented Toffoli count is chosen.
- `"auto"` (default): prices `"select"` and every power-of-two block size
  with the exact model below and keeps the cheapest, preferring `"select"`
  on ties. The chosen construction is reported by `variant` and
  `block_size`.

Readable attributes are `data` (tuple), `num_address`, `num_ladder`,
`num_output`, `variant`, `block_size`, `toffoli_count`, and `kernel`;
`repr` summarizes them.

## Boundaries and resources

`toffoli_count` is the exact number of two-controlled `x` gates in the minted
instruction tape. With `W(n, m)` the fused-walk count from
[unary iteration](primitives-unary-iteration.md) (`3 * 2^n / 2 - 5` on a
full uncontrolled tree, less on partial trees):

```text
select       : W(address_bits, len(data))
select_swap  : 2 * W(address_bits - log2(B), ceil(len(data) / B))
               + 2 * output_bits * (B - 1)
```

Each controlled register swap is `output_bits` Fredkins, each one Toffoli
plus two CNOTs; routing and block writes run twice (compute plus unitary
uncompute). These are logical counts pinned by the resource tests against
`cudaq.estimate_resources(...).count_controls("x", 2)`; they are not T
counts, transpiled depth, runtime, memory, or measured cost. The papers'
headline prices (`N - 1` lookup Toffolis, `ceil(N/B) + output_bits (B - 1)`)
assume measurement-based ancilla uncomputation, which is not provided; the
default `block_size` optimizes the coherent model (about
`sqrt(3N / (2 output_bits))`), not the papers' `sqrt(N / output_bits)`.

Not provided: a `controlled` option (only `unary_iteration_kernels` mints a
controlled walk), dirty-ancilla QROAM, measurement-based uncomputation, a
`kernel_adj` (the lookup is self-inverse), and any `BlockEncoding` member. An
external `cudaq.control` over a QROM kernel is not tested and is unverified.

## Oracle and runnable evidence

Read out every address by basis-state extraction with `cudaq.get_state` and
compare to `address + (data[k] << (address_bits + num_ladder))` with the
registers allocated in the order address, ladder, output; out-of-range
addresses must read zero and the ladder slice must be exactly `|0>`. Check
self-inversion on a superposed address with a non-zero initial output word,
and compare `"select"` against `"select_swap"` joint `(address, output)`
amplitudes on the same table. Verify `toffoli_count` against the formulas
and against `cudaq.estimate_resources`.

Run the authoritative self-inverse, auto-dispatch, and validation cases from
the repository root:

```bash
PYTHONPATH=python pytest -q tests/python/test_primitives_qrom.py \
  -k "qrom_is_self_inverse or qrom_auto_dispatch or qrom_validation_raises"
```

`test_qrom_select_exhaustive_readout` and
`test_qrom_select_swap_exhaustive_readout` cover the full readout contract;
`test_qrom_select_swap_cost_matches_documented_formula` and
`test_qrom_compiled_toffolis_match_emitter_count` in
`test_primitives_resources.py` pin the prices. Current public source and
tests are authoritative and must be rechecked at use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
