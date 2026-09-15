# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiler-pinned Toffoli costs of the primitives.

The fused unary-iteration walk documents an exact unitary Toffoli count
(``3 N / 2 - 5`` on full uncontrolled trees of ``N >= 8`` addresses,
``3 N / 2 - 1`` controlled) and QROM documents an exact price for each
variant. These tests hold those claims against the compiler rather than
the emitter: ``cudaq.estimate_resources`` counts the ``ccx`` operations
actually synthesized, which (a) must equal the emitter's bookkeeping —
a multi-control expansion or a decomposition surprise would break the
equality — and (b) must equal the documented closed forms, with the
per-doubling *increment* (not the ratio: affine costs approach their
asymptote from above) making the O(N) scaling directly visible.

The papers' headline counts (``N - 1`` for unary iteration in
arXiv:1805.03662, ``ceil(N/B) + b (B - 1)`` for QROAM in
arXiv:1812.00954) assume measurement-based ancilla uncomputation; the
strictly unitary counts pinned here are the honest coherent prices.

``estimate_resources`` traces the kernel without executing it, so no
simulator target or tolerance is involved; every assertion is an exact
integer comparison.
"""

import cudaq
import pytest

from cudaq_algorithms.primitives import QROM, unary_iteration_kernels
from cudaq_algorithms.primitives._unary_iteration import _walk_toffoli_count

pytestmark = pytest.mark.skipif(
    not hasattr(cudaq, "estimate_resources"),
    reason="cudaq.estimate_resources is not available in this CUDA-Q")


def _compiled_toffolis_of(kernel, *widths) -> int:
    """Compile a harness allocating the given register widths and count
    the synthesized Toffolis."""
    num_registers = len(widths)
    padded = list(widths) + [1] * (4 - num_registers)
    # Unpack to scalars before defining the kernel (no container capture).
    w0, w1, w2, w3 = padded

    if num_registers == 3:

        @cudaq.kernel
        def harness():
            r0 = cudaq.qvector(w0)
            r1 = cudaq.qvector(w1)
            r2 = cudaq.qvector(w2)
            kernel(r0, r1, r2)
    else:

        @cudaq.kernel
        def harness():
            r0 = cudaq.qvector(w0)
            r1 = cudaq.qvector(w1)
            r2 = cudaq.qvector(w2)
            r3 = cudaq.qvector(w3)
            kernel(r0, r1, r2, r3)

    resources = cudaq.estimate_resources(harness)
    # A Toffoli is an x with two controls; ``count_controls`` is the
    # arity-aware accessor (``count("ccx")`` matches nothing — the
    # display name is not the lookup key and returns 0).
    return resources.count_controls("x", 2)


def _walk_toffolis(num_address_bits: int, num_items: int,
                   controlled: bool) -> tuple[int, int]:
    """Return (emitter toffoli_count, compiler ccx count) for one walk."""
    walk = unary_iteration_kernels(num_address_bits,
                                   num_items,
                                   lambda k: [("x", 0)],
                                   controlled=controlled,
                                   include_adjoint=False)
    if controlled:
        compiled = _compiled_toffolis_of(walk.kernel, 1, num_address_bits,
                                         num_address_bits, 1)
    else:
        compiled = _compiled_toffolis_of(walk.kernel, num_address_bits,
                                         num_address_bits, 1)
    return walk.toffoli_count, compiled


@pytest.mark.parametrize("num_address_bits,num_items", [(2, 4), (3, 8), (3, 5),
                                                        (4, 16), (4, 11),
                                                        (2, 1)])
@pytest.mark.parametrize("controlled", [False, True])
def test_compiled_toffolis_match_emitter_count(num_address_bits, num_items,
                                               controlled):
    # The emitter's cost accounting is only trustworthy if the compiler
    # synthesizes exactly the Toffolis it claims to emit.
    claimed, compiled = _walk_toffolis(num_address_bits, num_items, controlled)
    assert compiled == claimed
    assert claimed == _walk_toffoli_count(num_address_bits, num_items,
                                          controlled)


def test_full_tree_walk_cost_is_the_documented_closed_form():
    # Headline: the fused walk costs exactly 3N/2 - 5 Toffolis on a full
    # uncontrolled tree of N >= 8 addresses (2 at N = 4, 0 at N = 2) and
    # 3N/2 - 1 controlled — compiler-counted, not emitter-claimed. The
    # paper's N - 1 (arXiv:1805.03662 Fig. 7) prices AND uncomputation
    # via measurement; unitarily, exhaustive circuit search shows these
    # counts are optimal at the small sizes it can cover.
    assert _walk_toffolis(1, 2, controlled=False)[1] == 0
    assert _walk_toffolis(2, 4, controlled=False)[1] == 2
    for num_address_bits in (3, 4, 5, 6):
        capacity = 1 << num_address_bits
        assert (_walk_toffolis(num_address_bits, capacity,
                               controlled=False)[1] == 3 * capacity // 2 - 5)
    for num_address_bits in (1, 2, 3, 4, 5):
        capacity = 1 << num_address_bits
        assert (_walk_toffolis(num_address_bits, capacity,
                               controlled=True)[1] == 3 * capacity // 2 - 1)


def test_walk_cost_grows_linearly_per_doubling():
    # O(N) via increments: the cost is affine (3N/2 - 5), so the
    # doubling *ratio* approaches 3/2 from above — the increment is the
    # robust check and is exactly 3N/2 additional Toffolis per doubling.
    compiled = {}
    for num_address_bits in (3, 4, 5, 6):
        num_items = 1 << num_address_bits
        compiled[num_items] = _walk_toffolis(num_address_bits,
                                             num_items,
                                             controlled=False)[1]
    for num_items in (8, 16, 32):
        increment = compiled[2 * num_items] - compiled[num_items]
        assert increment == 3 * num_items // 2, (
            f"doubling N from {num_items} added {increment} Toffolis "
            f"(expected exactly {3 * num_items // 2}): {compiled}")


def test_partial_tree_costs_no_more_than_full():
    # Skipped addresses are never entered: a partial tree cannot cost
    # more Toffolis than the full tree.
    full = _walk_toffolis(4, 16, controlled=False)[1]
    partial = _walk_toffolis(4, 11, controlled=False)[1]
    assert partial < full


@pytest.mark.parametrize("variant,block_size", [("select", None),
                                                ("select_swap", 2),
                                                ("select_swap", 4)])
def test_qrom_compiled_toffolis_match_emitter_count(variant, block_size):
    data = [3, 0, 5, 6, 1, 7, 2, 2, 4, 1, 6]
    qrom = QROM(data,
                address_bits=4,
                output_bits=3,
                variant=variant,
                block_size=block_size)
    compiled = _compiled_toffolis_of(qrom.kernel, qrom.num_address,
                                     qrom.num_ladder, qrom.num_output)
    assert compiled == qrom.toffoli_count


def test_qrom_select_swap_cost_matches_documented_formula():
    # Documented contract: 2 W(high_bits, num_blocks) + 2 b (B - 1),
    # every controlled register swap being b Fredkins (one Toffoli each)
    # and both the block writes and the routing running twice (compute +
    # unitary uncompute).
    data = list(range(2)) * 16
    for block_size in (2, 4, 8):
        qrom = QROM(data,
                    address_bits=5,
                    output_bits=1,
                    variant="select_swap",
                    block_size=block_size)
        low_bits = block_size.bit_length() - 1
        num_blocks = -(-len(data) // block_size)
        expected = (2 * _walk_toffoli_count(5 - low_bits, num_blocks) + 2 * 1 *
                    (block_size - 1))
        assert qrom.toffoli_count == expected
        compiled = _compiled_toffolis_of(qrom.kernel, qrom.num_address,
                                         qrom.num_ladder, qrom.num_output)
        assert compiled == expected


def test_qrom_auto_picks_the_cheaper_side_of_the_crossover():
    # Above the crossover (large table, narrow output) select_swap must
    # win and beat the plain walk...
    data = [int(v) % 2 for v in range(64)]
    auto = QROM(data, 6, 1)
    select = QROM(data, 6, 1, variant="select")
    assert auto.variant == "select_swap"
    assert auto.toffoli_count < select.toffoli_count
    # ...and below it (small table, wide output) the plain walk wins.
    data = [200, 3, 118, 25]
    auto = QROM(data, 2, 8)
    swapped = QROM(data, 2, 8, variant="select_swap")
    assert auto.variant == "select"
    assert auto.toffoli_count < swapped.toffoli_count
