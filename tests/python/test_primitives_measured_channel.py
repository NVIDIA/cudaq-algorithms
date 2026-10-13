# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""All Kraus histories equal the same independent clean-input SELECT map.

This small NumPy interpreter executes the emitted gate tape. Its expected
isometry is independently defined by address projectors and Pauli gates,
without a tree, ladder, or measurement. Each *unnormalized* history must
be exactly 2**(-measurements/2) times that isometry; no global phase is
removed. Thus relative phases, reset, and outcome probabilities are tested.
"""

import pytest
import cudaq
import numpy as np

from cudaq_algorithms import state_from
from cudaq_algorithms.primitives import MeasuredQROM
from cudaq_algorithms.primitives._unary_iteration import (
    _OP_AND_TT, _OP_AND_WT, _OP_BODY_X, _OP_BODY_X_W, _OP_BODY_Y, _OP_BODY_Z,
    _OP_BODY_Z_W, _OP_CCX, _OP_CCX_CTRL, _OP_COPY_TW, _OP_CX_ADDR_LADDER,
    _OP_CX_CTRL_LADDER, _OP_CX_LADDER_LADDER, _OP_CX_LADDER_TARGET,
    _OP_FREE_CX, _OP_FREE_X, _OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL,
    _OP_X_ADDR, _OP_X_LADDER, _OP_Z_LADDER)
from cudaq_algorithms.primitives._unary_iteration_measured import _emit_measured_walk
from dense_references import random_ket


def body(k):
    return [[('x', 0), ('y', 1)], [('z', 0), ('x', 0)],
            [('sign', ), ('y', 0), ('z', 1)]][k % 3]


def extended_body(k):
    pre = [('free_x', 0), ('free_cx', 0, 1), ('and_tt', 0, 1, 0),
           ('and_wt', 0, 1, 1), ('copy_tw', 0, 2)]
    core = [('x_w', 1, 2), ('z_w', 2), ('sign', ), ('y', 2), ('z', 2)]
    return pre + core + list(reversed(pre))


def verify(n, m, controlled, reverse, extended=False, oracle_only=False):
    nt, nw = (3, 3) if extended else (2, 0)
    nc = int(controlled)
    # Logical inputs contiguous, then clean ladder and work.
    na = nc
    tt = nc + n
    ll = tt + nt
    ww = ll + n
    dim = 1 << (ww + nw)
    logical = 1 << ll
    ids = np.arange(dim)
    init = np.zeros((dim, logical), complex)
    init[np.arange(logical), np.arange(logical)] = 1
    bb = [list((extended_body if extended else body)(k)) for k in range(m)]
    ops = _emit_measured_walk(n, m, controlled, bb, reverse)
    measurements = sum(o[0] in (_OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL)
                       for o in ops)

    def gate(v, g, target, controls=(), addr=None):
        active = np.ones(dim, bool)
        for c in controls:
            active &= ((ids >> c) & 1) != 0
        if addr is not None:
            active &= ((ids >> na) & ((1 << n) - 1)) == addr
            if controlled:
                active &= (ids & 1) == 1
        lo = ids[active & (((ids >> target) & 1) == 0)]
        hi = lo ^ (1 << target)
        if g == 'x':
            a = v[lo].copy()
            v[lo] = v[hi]
            v[hi] = a
        elif g == 'y':
            a = v[lo].copy()
            v[lo] = -1j * v[hi]
            v[hi] = 1j * a
        elif g == 'z':
            v[hi] *= -1
        elif g == 'h':
            a = v[lo].copy()
            b = v[hi].copy()
            v[lo] = (a + b) / np.sqrt(2)
            v[hi] = (a - b) / np.sqrt(2)

    expected = init.copy()
    # Independent direct-address SELECT; no tree or temporary leaf controls.
    for k in (range(m - 1, -1, -1) if reverse else range(m)):
        for ins in (reversed(bb[k]) if reverse else bb[k]):
            name, *a = ins
            if name in ('x', 'y', 'z'):
                gate(expected, name, tt + a[0], addr=k)
            elif name == 'sign':
                sel = ((ids >> na) & ((1 << n) - 1)) == k
                if controlled: sel &= (ids & 1) == 1
                expected[sel] *= -1
            elif name == 'free_x':
                gate(expected, 'x', tt + a[0])
            elif name == 'free_cx':
                gate(expected, 'x', tt + a[1], [tt + a[0]])
            elif name == 'and_tt':
                gate(expected, 'x', ww + a[2], [tt + a[0], tt + a[1]])
            elif name == 'and_wt':
                gate(expected, 'x', ww + a[2], [ww + a[0], tt + a[1]])
            elif name == 'copy_tw':
                gate(expected, 'x', ww + a[1], [tt + a[0]])
            elif name == 'x_w':
                gate(expected, 'x', tt + a[1], [ww + a[0]], addr=k)
            elif name == 'z_w':
                gate(expected, 'z', ww + a[0], addr=k)
            else:
                raise ValueError(ins)

    if oracle_only:
        return expected
    leaves, worst = 0, 0.0

    def visit(v, idx):
        nonlocal leaves, worst
        for j in range(idx, len(ops)):
            op, a, b, c = ops[j]
            if op in (_OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL):
                gate(v, 'h', ll + a)
                for outcome in (0, 1):
                    branch = v.copy()
                    branch[((ids >> (ll + a)) & 1) != outcome] = 0
                    if outcome:
                        gate(branch, 'x', ll + a)
                        gate(branch, 'z', na + c,
                             [ll + b] if op == _OP_MEAS_UNCOMPUTE else [0])
                    visit(branch, j + 1)
                return
            if op == _OP_X_ADDR:
                gate(v, 'x', na + a)
            elif op == _OP_X_LADDER:
                gate(v, 'x', ll + a)
            elif op == _OP_CX_ADDR_LADDER:
                gate(v, 'x', ll + b, [na + a])
            elif op == _OP_CX_LADDER_LADDER:
                gate(v, 'x', ll + b, [ll + a])
            elif op == _OP_CCX:
                gate(v, 'x', ll + c, [ll + a, na + b])
            elif op in (_OP_BODY_X, _OP_BODY_Y, _OP_BODY_Z):
                gate(v, {
                    _OP_BODY_X: 'x',
                    _OP_BODY_Y: 'y',
                    _OP_BODY_Z: 'z'
                }[op], tt + b, [ll + a])
            elif op == _OP_CX_CTRL_LADDER:
                gate(v, 'x', ll + b, [0])
            elif op == _OP_CCX_CTRL:
                gate(v, 'x', ll + c, [0, na + b])
            elif op == _OP_FREE_X:
                gate(v, 'x', tt + a)
            elif op == _OP_FREE_CX:
                gate(v, 'x', tt + b, [tt + a])
            elif op == _OP_AND_TT:
                gate(v, 'x', ww + c, [tt + a, tt + b])
            elif op == _OP_AND_WT:
                gate(v, 'x', ww + c, [ww + a, tt + b])
            elif op == _OP_COPY_TW:
                gate(v, 'x', ww + b, [tt + a])
            elif op == _OP_BODY_X_W:
                gate(v, 'x', tt + c, [ll + a, ww + b])
            elif op == _OP_BODY_Z_W:
                gate(v, 'z', ww + b, [ll + a])
            elif op == _OP_Z_LADDER:
                gate(v, 'z', ll + a)
            elif op == _OP_CX_LADDER_TARGET:
                gate(v, 'x', tt + b, [ll + a])
            else:
                raise ValueError(op)
        leaves += 1
        err = np.max(np.abs(v * 2**(measurements / 2) - expected))
        worst = max(worst, float(err))
        assert err < 2e-14, (n, m, controlled, reverse, extended, err)

    visit(init, 0)
    assert leaves == 2**measurements
    return leaves, worst


@pytest.mark.parametrize("n,m", [(n, m) for n in range(1, 4)
                                 for m in range(1, (1 << n) + 1)])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_all_measured_histories_match_direct_select(n, m, controlled, reverse):
    verify(n, m, controlled, reverse)


@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_extended_body_all_histories_and_native(controlled, reverse):
    # Exercises work+control together, free-gate compute/action/uncompute,
    # noncommuting Pauli bodies, and inverse leaf/body order.
    from cudaq_algorithms.primitives import measured_unary_iteration_kernels
    cudaq.set_random_seed(442)
    verify(2, 3, controlled, reverse, extended=True)
    expected_map = verify(2,
                          3,
                          controlled,
                          reverse,
                          extended=True,
                          oracle_only=True)
    logical = random_ket(int(np.log2(expected_map.shape[1])), 442)
    expected = expected_map @ logical
    walk = measured_unary_iteration_kernels(2,
                                            3,
                                            extended_body,
                                            controlled=controlled)
    kernel = walk.kernel_adj if reverse else walk.kernel
    state = state_from(logical)
    if controlled:

        @cudaq.kernel
        def run(initial: cudaq.State):
            q = cudaq.qvector(initial)
            ladder = cudaq.qvector(2)
            work = cudaq.qvector(3)
            kernel(q[0:1], q[1:3], ladder, q[3:6], work)
    else:

        @cudaq.kernel
        def run(initial: cudaq.State):
            q = cudaq.qvector(initial)
            ladder = cudaq.qvector(2)
            work = cudaq.qvector(3)
            kernel(q[0:2], ladder, q[2:5], work)

    for _ in range(3):
        np.testing.assert_allclose(np.asarray(cudaq.get_state(run, state)),
                                   expected,
                                   atol=1e-12,
                                   rtol=0)


def _qrom_histories(qrom, controlled):
    """Enumerate every measurement history of a MeasuredQROM tape and check
    each unnormalized branch against the XOR permutation on the logical
    (control, address, output) sector, with the clean ladder prefix at zero
    and the bank arbitrary when borrowed. Layout: control | address |
    ladder (clean prefix, then bank) | output."""
    n, b = qrom.num_address, qrom.num_output
    nc = int(controlled)
    clean = qrom.num_clean
    na = nc
    ll = nc + n
    tt = ll + qrom.num_ladder
    dim = 1 << (tt + b)
    ids = np.arange(dim)
    inputs = ids[((ids >> ll) & ((1 << clean) - 1)) == 0]
    init = np.zeros((dim, len(inputs)), complex)
    init[inputs, np.arange(len(inputs))] = 1
    expected = np.zeros_like(init)
    for col, src in enumerate(inputs):
        address = (src >> na) & ((1 << n) - 1)
        enabled = (not controlled) or (src & 1)
        word = qrom.data[address] if enabled and address < len(
            qrom.data) else 0
        expected[src ^ (word << tt), col] = 1
    ops = list(qrom._ops)
    measurements = sum(op[0] in (_OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL)
                       for op in ops)
    assert measurements == qrom.num_measurements

    def gate(v, g, target, controls=()):
        active = np.ones(dim, bool)
        for c in controls:
            active &= ((ids >> c) & 1) != 0
        lo = ids[active & (((ids >> target) & 1) == 0)]
        hi = lo ^ (1 << target)
        if g == 'x':
            a = v[lo].copy()
            v[lo] = v[hi]
            v[hi] = a
        elif g == 'z':
            v[hi] *= -1
        elif g == 'h':
            a = v[lo].copy()
            c = v[hi].copy()
            v[lo] = (a + c) / np.sqrt(2)
            v[hi] = (a - c) / np.sqrt(2)

    leaves = 0

    def visit(v, idx):
        nonlocal leaves
        for j in range(idx, len(ops)):
            op, a, bb, c = ops[j]
            if op in (_OP_MEAS_UNCOMPUTE, _OP_MEAS_UNCOMPUTE_CTRL):
                gate(v, 'h', ll + a)
                for outcome in (0, 1):
                    branch = v.copy()
                    branch[((ids >> (ll + a)) & 1) != outcome] = 0
                    if outcome:
                        gate(branch, 'x', ll + a)
                        gate(branch, 'z', na + c,
                             [ll + bb] if op == _OP_MEAS_UNCOMPUTE else [0])
                    visit(branch, j + 1)
                return
            if op == _OP_X_ADDR:
                gate(v, 'x', na + a)
            elif op == _OP_X_LADDER:
                gate(v, 'x', ll + a)
            elif op == _OP_CX_ADDR_LADDER:
                gate(v, 'x', ll + bb, [na + a])
            elif op == _OP_CX_LADDER_LADDER:
                gate(v, 'x', ll + bb, [ll + a])
            elif op == _OP_CCX:
                gate(v, 'x', ll + c, [ll + a, na + bb])
            elif op == _OP_BODY_X:
                gate(v, 'x', tt + bb, [ll + a])
            elif op == _OP_CX_CTRL_LADDER:
                gate(v, 'x', ll + bb, [0])
            elif op == _OP_CCX_CTRL:
                gate(v, 'x', ll + c, [0, na + bb])
            elif op == _OP_CX_LADDER_TARGET:
                gate(v, 'x', tt + bb, [ll + a])
            else:
                raise ValueError(op)
        leaves += 1
        err = np.max(np.abs(v * 2**(measurements / 2) - expected))
        assert err < 2e-14, (qrom, controlled, err)

    visit(init, 0)
    assert leaves == 2**measurements
    return leaves


@pytest.mark.parametrize("variant,block_size,n,m,b,dirty",
                         [("select", None, 3, 5, 2, False),
                          ("select_swap", 2, 3, 5, 2, False),
                          ("select_swap", 4, 4, 9, 1, False),
                          ("select_swap", 2, 3, 5, 1, True),
                          ("select_swap", 4, 4, 9, 1, True)])
@pytest.mark.parametrize("controlled", [False, True])
def test_all_measured_qrom_histories_match_the_xor_map(variant, block_size, n,
                                                       m, b, dirty,
                                                       controlled):
    # The lookup-level twin of the walk-tape oracle above: it covers the
    # SELECT-SWAP rebase of the fix-up operands, the routing network and
    # the dirty-bank echo, which the native trajectory tests reach only
    # with probability 1/2 per gadget.
    data = [(3 * i + 1) % (1 << b) for i in range(m)]
    qrom = MeasuredQROM(data,
                        n,
                        b,
                        variant=variant,
                        block_size=block_size,
                        controlled=controlled,
                        dirty_banks=dirty)
    assert qrom.num_measurements > 0
    leaves = _qrom_histories(qrom, controlled)
    assert leaves == 2**qrom.num_measurements
