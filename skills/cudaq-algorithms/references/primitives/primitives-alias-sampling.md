# Alias-sampling PREPARE with garbage

Operation + object: **prepare** an **index register whose amplitude
magnitudes squared realize a weight distribution `w / lambda`, entangled
with alias-sampling garbage registers**.

## Identity and classification

- Public entry point: `cudaq_algorithms.primitives.AliasSamplingPrepare(
  weights, mu, *, variant="auto", block_size=None)`.
- Source/tests: `python/cudaq_algorithms/primitives/_alias_sampling.py` and
  `tests/python/test_primitives_alias_sampling.py`.
- Kind/role/layers: quantum operation, computational composite (uniform
  superposition, [QROM](primitives-qrom.md) lookup, `cmp_ge_register`
  comparator, controlled swaps); host integer Vose preprocessing, composed
  device kernels.
- Input/output: non-negative weights and a precision `mu` -> object with
  `kernel()` and `adjoint_kernel()`, both `(index: qview, garbage: qview)`.
- Source provenance: [Source lookup](../source-provenance.md) gives shared current-source paths.

## Scientific, input, and output contract

Construction validates `weights` as a non-empty sequence of finite,
non-negative floats with a positive sum (zero entries allowed) and `mu` as an
integer `>= 1` (`bool` rejected, integral NumPy ints accepted); each
violation raises `ValueError`. Weights are zero-padded to `K = 2^num_index`
bins with `num_index = max(1, ceil(log2(len(weights))))`; padded bins carry
exactly zero probability. Integer Vose preprocessing produces per-bin
`keep` thresholds in `[0, 2^mu - 1]` and `alias` indices (full bins are
self-aliased), packed into one QROM word per bin as
`alias[k] | (keep[k] << num_index)` with `output_bits = num_index + mu`.
`variant` and `block_size` are forwarded verbatim to that `QROM`; its
validation errors propagate unchanged.

`kernel()` applies: Hadamards on `index` and on the `mu`-bit reference;
the QROM lookup of `(alias, keep)`; `cmp_ge_register(ref, keep, carry,
flag)`; and `num_index` controlled swaps of `index` with the alias register
on the flag. `adjoint_kernel()` is the literal gate-reversal and returns both
registers to `|0...0>`; both registers must be `|0...0>` on entry to
`kernel()`. Registers are little-endian with `garbage` laid out as

```text
[alias(num_index) | keep(mu) | ref(mu) | flag(1) | ladder(qrom.num_ladder) | carry(1)]
num_garbage   = num_index + 2 mu + 2 + qrom.num_ladder
              = 2 num_index + 2 mu + 2  when qrom.variant == "select"
ladder_offset = num_index + 2 mu + 1
```

The realized index marginal is exactly `table_probabilities[k] =
W_k / (K 2^mu)`; its per-bin distance from the ideal `probabilities =
w / lambda` is bounded by `discretization_bound = 1.5 / (K 2^mu)`. Dyadic
and uniform weights are realized exactly.

**The output is not `sum_k sqrt(w_k / lambda) |k>|0...0>`.** The alias,
keep, reference, and flag registers remain entangled with the index, and the
per-bin garbage vectors are not mutually orthogonal, so the reduced index
state is not `diag(p_k)`; only the amplitude magnitudes marginalize to the
table distribution. The contract is sound only inside the symmetric sandwich
`PREPARE ... PREPARE^dagger` on the same registers with `adjoint_kernel()`,
with no reflection over or consumption of the dirty garbage in between.
The ladder slice `garbage[ladder_offset : ladder_offset + qrom.num_ladder]`
is clean (`|0>`) between the two kernels and may be reused as a SELECT's
ladder.

Readable attributes: `lam`, `mu`, `num_index`, `num_bins`, `num_garbage`,
`qrom`, `ladder_offset`, `keep`, `alias`, `probabilities`,
`table_probabilities`, `discretization_bound`.

## Boundaries and resources

Each of `kernel()` and `adjoint_kernel()` costs exactly
`qrom.toffoli_count + 2 mu - 1` Toffolis (the lookup at the QROM's own
reported price plus one `mu`-bit `cmp_ge_register`) plus `num_index`
controlled swaps and `num_index + mu` Hadamards; the adjoint has the same
gate multiset gate for gate. The tests pin these against
`cudaq.estimate_resources(...).count_controls("x", 2)` and
`count_controls("swap", 1)`. The lookup is uncomputed by re-applying the
forward QROM (self-inverse), so the adjoint is not cheaper than the forward
pass. All counts are logical, not T counts, depth, runtime, memory, or
hardware cost.

These kernels call the QROM and comparator sub-kernels and are therefore
**not flat**: source directs that they not be placed under `cudaq.control`
(control-variant generation rejects nested kernel calls through CUDA-Q
0.15); consumers control SELECT, not PREPARE. Not provided: a clean
(garbage-free) `sum sqrt(p_k)|k>` preparation, a `controlled` option, any
`BlockEncoding` member (`prepare_kernel`, `unprepare_kernel`,
`select_observable`, `alpha`), a SELECT, a block-encoding or `Walk`
integration, and amplitude signs (weights are non-negative magnitudes).
No packaged algorithm family imports this class in current source.

## Oracle and runnable evidence

Enumerate the classical branches independently: for bin `k` and reference
`sigma in [0, 2^mu)`, the branch lands on `k` if `sigma < keep[k]` else on
`alias[k]`, each with probability `1 / (K 2^mu)`; compare that marginal with
`sum` over garbage of `|amplitude|^2` from `cudaq.get_state` (index at qubits
`[0, num_index)`, so reshape to `(-1, 2^num_index)` and sum axis 0) and with
`table_probabilities`, then check the ideal within `discretization_bound`.
Verify `kernel()` then `adjoint_kernel()` returns the full register to
`|0...0>` and that the Toffoli count equals `qrom.toffoli_count + 2 mu - 1`.

Run the authoritative accounting, exact-marginal, round-trip, and validation
cases from the repository root:

```bash
PYTHONPATH=python pytest -q tests/python/test_primitives_alias_sampling.py \
  -k "register_accounting or exact_dyadic or prepare_then_adjoint or validation_raises"
```

`test_alias_sampling_index_marginal` covers the parametrized marginal
contract, `test_alias_sampling_cost_is_qrom_price_plus_comparator` and
`test_alias_sampling_forced_select_swap_variant` pin the price and the
variant pass-through. Current public source and tests are authoritative and
must be rechecked at use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.
