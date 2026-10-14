"""Host-only packet geometry and independent SELECT-COPY algebra proofs."""
import itertools
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from qrom_host_loader import load_qrom_host_module


def test_packet_layout_balanced_and_fixed_remainder():
    module = load_qrom_host_module('_select_copy_plan')
    for kwargs, widths, offsets in [
        ({
            'alpha': 3
        }, (4, 3, 3), (0, 4, 7)),
        ({
            'packet_width': 4
        }, (4, 4, 2), (0, 4, 8)),
        ({}, (10, ), (0, )),
    ]:
        layout = module.packet_layout(10, **kwargs)
        assert layout.widths == widths
        assert layout.offsets == offsets
        assert layout.stride == max(widths)
        assert sorted(itertools.chain.from_iterable(layout.fanout)) == list(
            range(10))
        for bit, targets in enumerate(layout.fanout):
            assert targets == tuple(offset + bit
                                    for width, offset in zip(widths, offsets)
                                    if bit < width)
    with pytest.raises(FrozenInstanceError):
        layout.stride = 1
    for bits in range(1, 17):
        for option in ('alpha', 'packet_width'):
            for value in range(1, bits + 1):
                layout = module.packet_layout(bits, **{option: value})
                assert sum(layout.widths) == bits
                assert layout.offsets == tuple(
                    sum(layout.widths[:j]) for j in range(len(layout.widths)))
                assert sorted(itertools.chain.from_iterable(
                    layout.fanout)) == list(range(bits))
                if option == 'alpha':
                    assert len(layout.widths) == value
                    assert max(layout.widths) - min(layout.widths) <= 1
                    assert tuple(sorted(layout.widths,
                                        reverse=True)) == layout.widths
                else:
                    assert layout.widths[:-1] == (value, ) * (
                        len(layout.widths) - 1)
                    assert 1 <= layout.widths[-1] <= value
    for bits, kwargs in [(0, {}), (True, {}), (2.5, {}), (10, {
            'alpha': 0
    }), (10, {
            'alpha': True
    }), (10, {
            'alpha': 11
    }), (10, {
            'packet_width': 0
    }), (10, {
            'packet_width': True
    }), (10, {
            'packet_width': 11
    }), (10, {
            'packet_width': 2.5
    }), (10, {
            'alpha': 2,
            'packet_width': 3
    })]:
        with pytest.raises((ValueError, TypeError)):
            module.packet_layout(bits, **kwargs)


def _check_boolean(data, address_bits, output_bits, block, widths, exhaustive):
    """Integer-only oracle: no production layout or scheduling helpers."""
    stride = max(widths)
    mask = (1 << stride) - 1
    dirty_cases = range(1 << (
        (block - 1) * stride)) if exhaustive else [0, 1, 0x15555, 0x2aaaa]
    outputs = range(1 << output_bits) if exhaustive else [
        0, 1, (1 << output_bits) - 1
    ]
    for address, control, dirty, original_y in itertools.product(
            range(1 << address_bits), (0, 1), dirty_cases, outputs):
        q, r = divmod(address, block)
        bank0 = [0] + [(dirty >> (i * stride)) & mask
                       for i in range(block - 1)]
        bank = bank0.copy()
        previous = [0] * block
        y = original_y
        offset = 0
        selected_dirty = 0
        for width in widths:
            packet_mask = (1 << width) - 1
            values = [
                ((data[q * block + i] if q * block + i < len(data) else 0) >>
                 offset) & packet_mask for i in range(block)
            ]
            differences = [v ^ values[0] for v in values]
            for i in range(1, block):
                bank[i] ^= control * (previous[i] ^ differences[i])
                assert bank[i] == bank0[i] ^ (control * differences[i])
            y ^= control * (values[0] << offset)
            y ^= control * ((bank[r] & packet_mask) << offset)
            selected_dirty ^= control * ((bank0[r] & packet_mask) << offset)
            expected_prefix = ((data[address] if address < len(data) else 0) &
                               ((1 << (offset + width)) - 1))
            assert y == original_y ^ (control *
                                      expected_prefix) ^ selected_dirty
            previous = differences
            offset += width
        for i in range(1, block):
            bank[i] ^= control * previous[i]
        assert bank == bank0
        # Restore only after unloading, using the original selected dirty word.
        offset = 0
        for width in widths:
            y ^= control * ((bank[r] & ((1 << width) - 1)) << offset)
            offset += width
        assert y == original_y ^ (
            control * (data[address] if address < len(data) else 0))


def test_select_copy_boolean_stage_invariants():
    for data in itertools.product(range(2), repeat=4):
        for block in (1, 2):
            _check_boolean(data, 2, 1, block, (1, ), True)
    _check_boolean((3, 17, 29, 4, 22), 3, 5, 4, (2, 2, 1), False)
    _check_boolean((7, 2, 5, 0, 3, 6, 1), 3, 3, 2, (2, 1), False)


def _restore_branch(outputs, bank_bits, history):
    from qrom_channel_reference import apply_gate, measure_reset

    # Logical layout: control, low-address predicate, borrowed bank, output.
    logical = 2 + bank_bits + outputs
    leaf, temporary = logical, logical + 1
    state = np.zeros((1 << (logical + 2), 1 << logical), complex)
    state[:1 << logical] = np.eye(1 << logical)
    state = apply_gate(state, 'ccx', 0, 1, leaf)
    for bank, outcome in enumerate(history):
        state = apply_gate(state, 'ccx', leaf, 2 + bank, temporary)
        for target in range(outputs):
            state = apply_gate(state, 'cx', temporary, 2 + bank_bits + target)
        state = apply_gate(state, 'h', temporary)
        state = measure_reset(state, temporary, outcome)
        if outcome:
            state = apply_gate(state, 'cz', leaf, 2 + bank)
        # One clean scratch is immediately available to the next bank bit.
        assert np.max(np.abs(state[1 << temporary:])) == 0
    return apply_gate(state, 'ccx', 0, 1, leaf)


@pytest.mark.parametrize('outputs,bank_bits', [(1, 1), (2, 1), (2, 2)])
def test_restore_kraus_channel_entangled_controls(outputs, bank_bits):
    logical = 2 + bank_bits + outputs
    dim = 1 << logical
    expected = np.zeros((4 * dim, dim), complex)
    for source in range(dim):
        enabled = (source & 1) & ((source >> 1) & 1)
        parity = sum(
            (source >> (2 + bank)) & 1 for bank in range(bank_bits)) % 2
        output_mask = ((1 << outputs) - 1) << (2 + bank_bits)
        expected[source ^ (output_mask if enabled and parity else 0),
                 source] = 1
    completeness = np.zeros((dim, dim), complex)
    # The first bank bit is maximally entangled with an explicit reference.
    # Control and address both have coherent active/inactive amplitudes.
    rng = np.random.default_rng(1941)
    output_state = rng.normal(
        size=1 << outputs) + 1j * rng.normal(size=1 << outputs)
    output_state /= np.linalg.norm(output_state)
    psi = np.zeros((dim, 2), complex)
    for c, a, bank, output in itertools.product((0, 1), (0, 1), (0, 1),
                                                range(1 << outputs)):
        source = c | (a << 1) | (bank << 2) | (output << (2 + bank_bits))
        psi[source, bank] = output_state[output] / np.sqrt(8)
    ideal = (expected @ psi).reshape(-1)
    density = np.zeros((len(ideal), len(ideal)), complex)
    branch_error = 0.
    for history in itertools.product((0, 1), repeat=bank_bits):
        branch = _restore_branch(outputs, bank_bits, history)
        gamma = 2**(-bank_bits / 2)
        branch_error = max(branch_error,
                           float(np.max(np.abs(branch - gamma * expected))))
        np.testing.assert_allclose(branch,
                                   gamma * expected,
                                   atol=1e-12,
                                   rtol=0)
        assert np.max(np.abs(branch[dim:])) == 0  # leaf AND temporary restored
        completeness += branch.conj().T @ branch
        vector = (branch @ psi).reshape(-1)
        density += np.outer(vector, vector.conj())
    channel_error = float(
        np.max(np.abs(density - np.outer(ideal, ideal.conj()))))
    np.testing.assert_allclose(completeness, np.eye(dim), atol=1e-12, rtol=0)
    assert channel_error <= 1e-12
    print(
        f'Restore outputs={outputs}, banks={bank_bits}: branch={branch_error:.3g}, channel={channel_error:.3g}'
    )
