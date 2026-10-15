"""Independent little-endian NumPy gates and unnormalized measurement channels.

Quantum bit zero is the least significant basis bit. Matrices have output basis
rows and input basis columns. apply_gate and measure_reset accept either vectors
or matrices, so a whole logical isometry can be checked without sampling.
No production helpers (including masks, packing and routing) are imported.
"""
import itertools
import sys
import numpy as np

ATOL = 2e-14


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)


def apply_gate(state, gate, *bits):
    """Apply X/H/Z/CX/CZ/CCX/SWAP; controls precede target."""
    result = np.array(state, dtype=complex, copy=True)
    indices = np.arange(len(result))
    target = bits[-1]
    selected = (indices & (1 << target)) == 0
    for control in bits[:-1]:
        selected &= (indices & (1 << control)) != 0
    lo = indices[selected]
    hi = lo | (1 << target)
    if gate in ('x', 'cx', 'ccx'):
        result[lo], result[hi] = state[hi], state[lo]
    elif gate == 'h':
        result[lo] = (state[lo] + state[hi]) / np.sqrt(2)
        result[hi] = (state[lo] - state[hi]) / np.sqrt(2)
    elif gate in ('z', 'cz'):
        result[hi] *= -1
    elif gate == 'swap':
        a, b = bits
        lo = indices[((indices >> a) & 1 == 0) & ((indices >> b) & 1 == 1)]
        hi = lo ^ (1 << a) ^ (1 << b)
        result[lo], result[hi] = state[hi], state[lo]
    else:
        raise ValueError(gate)
    return result


def measure_reset(state, bit, outcome, *, reset=True):
    """Z projection followed by optional X reset; NEVER normalize."""
    result = np.array(state, copy=True)
    indices = np.arange(len(result))
    result[((indices >> bit) & 1) != outcome] = 0
    return apply_gate(result, 'x', bit) if outcome and reset else result


def direct_xor_isometry(data, n, b, controlled, clean_width, dirty_width=0):
    """Layout [control?], address, output, dirty, clean (LSB first).

    Dirty is arbitrary input, clean starts at zero. References may be supplied
    separately by tensoring this map with their identity.
    """
    cwidth = int(controlled)
    logical = cwidth + n + b + dirty_width
    result = np.zeros((1 << (logical + clean_width), 1 << logical), complex)
    for source in range(1 << logical):
        control = source & 1 if controlled else 1
        address = (source >> cwidth) & ((1 << n) - 1)
        value = data[address] if address < len(data) else 0
        result[source ^ ((control * value) << (cwidth + n)), source] = 1
    return result


def direct_phase_diagonal(cache_words, h, cache_bits, outcome, controlled):
    """Diagonal entries on [control?], high address; absent rows are zero."""
    result = np.ones(1 << (h + int(controlled)), complex)
    for index in range(len(result)):
        c = index & 1 if controlled else 1
        j = index >> int(controlled)
        word = cache_words[j] if j < len(cache_words) else 0
        parity = sum(((word >> t) & 1) * ((outcome >> t) & 1)
                     for t in range(cache_bits)) % 2
        result[index] = (-1)**(c * parity)
    return result


def cache_branch(data,
                 n,
                 b,
                 outcome,
                 controlled=False,
                 *,
                 block_size=1,
                 mutation=None):
    """Explicit load, route, XOR, unroute, X-measure/reset, phase correction."""
    cwidth = int(controlled)
    logical = cwidth + n + b
    w = block_size * b
    dim = 1 << logical
    state = np.zeros((dim << w, dim), complex)
    for source in range(dim):
        c = source & 1 if controlled else 1
        address = (source >> cwidth) & ((1 << n) - 1)
        j, r = divmod(address, block_size)
        slots = [
            data[j * block_size + q] if j * block_size + q < len(data) else 0
            for q in range(block_size)
        ]
        # Follow slot contents rather than calling a production packing helper.
        bank = sum(c * value * (2**(b * q)) for q, value in enumerate(slots))
        routed = slots[r] * c
        state[(source ^ (routed << (cwidth + n))) | (bank << logical),
              source] = 1
    for bit in range(w):
        state = apply_gate(state, 'h', logical + bit)
        state = measure_reset(state, logical + bit, (outcome >> bit) & 1)
    for row in range(dim):
        c = row & 1 if controlled else 1
        j = ((row >> cwidth) & ((1 << n) - 1)) // block_size
        parity = 0
        for q in range(block_size):
            address = j * block_size + q
            value = data[address] if address < len(data) else 0
            for t in range(b):
                bit = q * b + t
                if mutation == 'packed_order':
                    bit = w - 1 - bit
                parity ^= ((value >> t) & 1) & ((outcome >> bit) & 1)
        if mutation != 'missing_fixup':
            state[row] *= (-1)**((1 if mutation == 'missing_control' else c) *
                                 parity)
    return state


def onehot_phase_branch(words,
                        h,
                        width,
                        outcome,
                        controlled,
                        k,
                        history=None,
                        *,
                        mutation=None):
    """Conservative coherent inverse if history=None; measured inverse otherwise."""
    logical = h + int(controlled)
    dim = 1 << logical
    state = np.zeros((dim << k, dim), complex)
    state[:dim] = np.eye(dim)
    splits = []
    if k > 1:
        state = apply_gate(state, 'x', logical)
        active = [0]
        for bit in reversed(range(k.bit_length() - 1)):
            new = []
            for parent in active:
                child = parent + (1 << bit)
                a = int(controlled) + bit
                state = apply_gate(state, 'ccx', logical + parent, a,
                                   logical + child)
                state = apply_gate(state, 'cx', logical + child,
                                   logical + parent)
                splits.append((logical + parent, a, logical + child))
                new.extend([parent, child])
            active = new
    # Phase gates controlled on high row and one-hot line. Compute each parity
    # directly from word bits, independently of direct_phase_diagonal.
    for index in range(len(state)):
        c = index & 1 if controlled else 1
        j = (index >> int(controlled)) & ((1 << h) - 1)
        u = j // k
        for q in range(k):
            row = u * k + q
            if row >= len(words):
                continue
            parity = (words[row] & outcome).bit_count() % 2
            enabled = (index >> (logical + q)) & 1 if k > 1 else 1
            if c and enabled and parity:
                state[index] *= -1
    for step, (parent, a, child) in enumerate(reversed(splits)):
        state = apply_gate(state, 'cx', child, parent)
        if history is None:
            state = apply_gate(state, 'ccx', parent, a, child)
        else:
            state = apply_gate(state, 'h', child)
            z = history[step]
            state = measure_reset(state,
                                  child,
                                  z,
                                  reset=mutation != 'missing_reset')
            if z:
                state = apply_gate(state, 'cz', parent, a)
    if k > 1:
        state = apply_gate(state, 'x', logical)
    return state


def check_cache(n, w, controlled=False, reference=False):
    data = tuple((3 * j + 1) % (1 << w) for j in range((1 << n) - 1))
    expected = direct_xor_isometry(data, n, w, controlled, w)
    completeness = np.zeros((expected.shape[1], ) * 2, complex)
    residual = 0.0
    rng = np.random.default_rng(417)
    psi = rng.normal(size=(expected.shape[1],
                           2)) + 1j * rng.normal(size=(expected.shape[1], 2))
    psi /= np.linalg.norm(psi)
    for z in range(1 << w):
        branch = cache_branch(data, n, w, z, controlled)
        target = expected * 2**(-w / 2)
        assert_close(branch, target)
        completeness += branch.conj().T @ branch
        assert_close(np.sum(abs(branch)**2, axis=0),
                     np.full(expected.shape[1], 2.**-w))
        if reference:
            assert_close(branch @ psi, target @ psi)
            assert_close(np.linalg.norm(branch @ psi)**2, 2.**-w)
        residual = max(residual, float(np.max(abs(branch - target))))
    assert_close(completeness, np.eye(expected.shape[1]))
    return residual


def check_phase(k):
    h, width = 3, 4
    words = (1, 6, 9, 3, 12)  # partial final block and wholly absent rows
    maximum = 0.
    for controlled in (False, True):
        dim = 1 << (h + int(controlled))
        for z in range(1 << width):
            target = np.zeros((dim << k, dim), complex)
            target[:dim] = np.diag(
                direct_phase_diagonal(words, h, width, z, controlled))
            assert_close(
                onehot_phase_branch(words, h, width, z, controlled, k), target)
            completeness = np.zeros((dim, dim), complex)
            for history in itertools.product((0, 1), repeat=k - 1):
                branch = onehot_phase_branch(words, h, width, z, controlled, k,
                                             history)
                expected = target * 2**(-(k - 1) / 2)
                assert_close(branch, expected)
                completeness += branch.conj().T @ branch
                maximum = max(maximum, float(np.max(abs(branch - expected))))
            assert_close(completeness, np.eye(dim))
    return maximum


def check_mutation(mutation):
    if mutation == 'missing_reset':
        actual = onehot_phase_branch((1, 2, 3),
                                     2,
                                     2,
                                     1,
                                     True,
                                     2, (1, ),
                                     mutation=mutation)
        expected = onehot_phase_branch((1, 2, 3), 2, 2, 1, True, 2, (1, ))
    else:
        actual = cache_branch((1, 2, 3),
                              2,
                              2,
                              1,
                              True,
                              block_size=2,
                              mutation=mutation)
        expected = direct_xor_isometry((1, 2, 3), 2, 2, True, 4) / 4
    rng = np.random.default_rng(987)
    psi = rng.normal(
        size=actual.shape[1]) + 1j * rng.normal(size=actual.shape[1])
    psi /= np.linalg.norm(psi)
    try:
        assert_close(actual @ psi, expected @ psi)
    except AssertionError:
        return
    raise AssertionError(f'mutation escaped detection: {mutation}')


if __name__ == '__main__':
    if sys.argv[1:] != ['--self-test']:
        raise SystemExit('usage: qrom_channel_reference.py --self-test')
    cache = max(check_cache(n, w) for n in range(1, 4) for w in range(1, 5))
    check_cache(2, 2, True, True)
    phase = max(check_phase(k) for k in (1, 2, 4))
    for mutation in ('missing_fixup', 'missing_control', 'packed_order',
                     'missing_reset'):
        check_mutation(mutation)
    assert 'cudaq' not in sys.modules
    print(
        f'cache n=1..3,w=1..4: max residual {cache:.3g}; controlled/reference passed'
    )
    print(
        f'phase K=1,2,4, all outcomes and inverse histories: max residual {phase:.3g}'
    )
    print('four mutation sentinels passed; cudaq absent')


def execute_plan(plan,
                 state,
                 history,
                 *,
                 address,
                 ladder,
                 output,
                 control=None,
                 work=()):
    """Interpret an actual tape on vectors/isometries with explicit wire maps.

    Numeric opcodes are decoded here independently of production dispatch or
    operand schemas. Persistent slots are fresh for EACH invocation; temporary
    AND outcomes never overwrite them. Histories include every measurement in
    tape order. No normalization or phase alignment is performed.
    """
    outcomes = iter(history)
    slots = [None] * plan.num_results
    state = np.asarray(state, dtype=complex)
    for op, a, b, c in plan.ops:
        if op == 0: gate = ('x', address[a])
        elif op == 1: gate = ('x', ladder[a])
        elif op == 2: gate = ('cx', address[a], ladder[b])
        elif op == 3: gate = ('cx', ladder[a], ladder[b])
        elif op == 4: gate = ('ccx', ladder[a], address[b], ladder[c])
        elif op == 5: gate = ('cx', ladder[a], output[b])
        elif op == 7: gate = ('cz', ladder[a], output[b])
        elif op == 8: gate = ('cx', control, ladder[b])
        elif op == 9: gate = ('ccx', control, address[b], ladder[c])
        elif op == 10: gate = ('x', output[a])
        elif op == 11: gate = ('cx', output[a], output[b])
        elif op == 12: gate = ('ccx', output[a], output[b], work[c])
        elif op == 13: gate = ('ccx', work[a], output[b], work[c])
        elif op == 14: gate = ('cx', output[a], work[b])
        elif op == 15: gate = ('ccx', ladder[a], work[b], output[c])
        elif op == 16: gate = ('cz', ladder[a], work[b])
        elif op == 17: gate = ('z', ladder[a])
        elif op == 18: gate = ('cx', address[a], address[b])
        elif op == 19: gate = ('ccx', address[a], address[b], ladder[c])
        elif op == 20: gate = ('cx', ladder[a], output[b])
        elif op == 21: gate = ('ccx', ladder[a], ladder[b], output[c])
        elif op in (22, 23, 24, 25):
            result = next(outcomes)
            assert result in (0, 1)
            bit = output[a] if op == 25 else ladder[a]
            state = measure_reset(apply_gate(state, 'h', bit), bit, result)
            if op in (24, 25):
                slots[b] = result
            elif result:
                parent = control if op == 23 else ladder[b]
                state = apply_gate(state, 'cz', parent, address[c])
            continue
        elif op in (26, 27):
            mask = plan.parity_masks[b if op == 26 else c]
            assert all(slots[t] is not None for t in mask)
            if sum(slots[t] for t in mask) % 2:
                state = apply_gate(state, 'z',
                                   ladder[a]) if op == 26 else apply_gate(
                                       state, 'cz', ladder[a], ladder[b])
            continue
        else:
            raise ValueError(f'unsupported reference opcode {op}')
        state = apply_gate(state, *gate)
    assert next(outcomes, None) is None, 'unused history bits'
    return state
