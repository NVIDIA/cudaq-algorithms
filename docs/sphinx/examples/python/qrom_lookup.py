#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""Coherent QROM: lookup, use, unlookup, and external control.

A quantum algorithm may need different classical data for different branches
of a quantum state. This example shows how a quantum address can look up a
table entry while preserving the superposition and its relative phases. It
then uses the loaded data to change the state and reverses the lookup,
demonstrating how table data can influence a calculation while its temporary
storage is returned to zero.

You should come away able to identify the address, output, and scratch
registers and understand why their initial values and cleanup matter. The same
lookup is built with SELECT and SELECT-SWAP strategies and with an external
enable, making their qubit and gate tradeoffs visible. Full-state comparisons
with an independently constructed reference check that the chosen entry, the
phase change, and workspace restoration all agree.

Run with: python3 qrom_lookup.py
QROM XORs classical words into a quantum output without measuring the address.
All checks compare complex amplitudes, including restored ladder workspace.
"""

import os
import time

import numpy as np

import cudaq
from cudaq_algorithms import state_from
from cudaq_algorithms.primitives import QROM

DATA = [3, 0, 5, 6, 1]
ADDRESS_BITS = 3
OUTPUT_BITS = 3
ATOL = 1e-10


def check(label, actual, expected):
    error = float(np.max(np.abs(actual - expected)))
    print(f"CHECK {label}: {error:.3e} <= {ATOL:.1e} -> "
          f"{'PASS' if error <= ATOL else 'FAIL'}")
    if not np.isfinite(error) or error > ATOL:
        raise AssertionError(label)


def complex_ket(size, rng):
    ket = rng.normal(size=size) + 1j * rng.normal(size=size)
    return ket / np.linalg.norm(ket)


def reference(initial, *, controlled=False, phase=False):
    """Logical registers are [control? | address | output], low bits first."""
    result = np.zeros_like(initial)
    offset = int(controlled)
    for index, amplitude in enumerate(initial):
        address = (index >> offset) & ((1 << ADDRESS_BITS) - 1)
        word = DATA[address] if address < len(DATA) else 0
        enabled = not controlled or bool(index & 1)
        if phase:
            # lookup, Z on output bit 0, lookup: the phase of a zero output
            # word is (-1)**(word & 1); a nonzero y would add (-1)**(y & 1).
            result[index] = amplitude * (-1 if word & 1 else 1)
        else:
            target = index ^ (word <<
                              (ADDRESS_BITS + offset)) if enabled else index
            result[target] += amplitude
    return result


def lookup_circuit(table, *, wrapped=False, phase=False):
    # QROM.kernel is a callable PROPERTY. It is not a zero-argument factory.
    lookup = table.kernel
    m, width, ladder_width = table.num_address, table.num_output, table.num_ladder

    if table.controlled:

        @cudaq.kernel
        def native(initial: cudaq.State):
            logical = cudaq.qvector(initial)
            ladder = cudaq.qvector(ladder_width)
            lookup(logical[0:1], logical[1:1 + m], ladder,
                   logical[1 + m:1 + m + width])

        return native

    if wrapped:

        @cudaq.kernel
        def controlled_parent(initial: cudaq.State):
            logical = cudaq.qvector(initial)
            ladder = cudaq.qvector(ladder_width)
            cudaq.control(lookup, logical[0:1], logical[1:1 + m], ladder,
                          logical[1 + m:1 + m + width])

        return controlled_parent

    @cudaq.kernel
    def plain(initial: cudaq.State):
        logical = cudaq.qvector(initial)
        ladder = cudaq.qvector(ladder_width)
        lookup(logical[0:m], ladder, logical[m:m + width])
        if phase:
            # Use a loaded bit, then erase the word with the same XOR lookup.
            z(logical[m])
            lookup(logical[0:m], ladder, logical[m:m + width])

    return plain


def run_check(label, table, initial, *, wrapped=False, phase=False):
    controlled = table.controlled or wrapped
    expected_logical = reference(initial, controlled=controlled, phase=phase)
    expected = np.zeros(len(initial) << table.num_ladder, dtype=complex)
    expected[:len(initial)] = expected_logical
    circuit = lookup_circuit(table, wrapped=wrapped, phase=phase)
    actual = np.asarray(cudaq.get_state(circuit, state_from(initial)))
    check(label, actual, expected)
    return actual[:len(initial)]


def resource_kernel(table, *, wrapped=False):
    """Allocate clean registers; count only lookup, with no state preparation."""
    lookup = table.kernel
    m, width, ladder_width = table.num_address, table.num_output, table.num_ladder
    if table.controlled:

        @cudaq.kernel
        def native():
            control = cudaq.qvector(1)
            address = cudaq.qvector(m)
            ladder = cudaq.qvector(ladder_width)
            output = cudaq.qvector(width)
            lookup(control, address, ladder, output)

        return native

    if wrapped:

        @cudaq.kernel
        def wrapped_lookup():
            control = cudaq.qvector(1)
            address = cudaq.qvector(m)
            ladder = cudaq.qvector(ladder_width)
            output = cudaq.qvector(width)
            cudaq.control(lookup, control, address, ladder, output)

        return wrapped_lookup

    @cudaq.kernel
    def plain():
        address = cudaq.qvector(m)
        ladder = cudaq.qvector(ladder_width)
        output = cudaq.qvector(width)
        lookup(address, ladder, output)

    return plain


def report_resources(label, table, *, wrapped=False):
    resources = cudaq.estimate_resources(
        resource_kernel(table, wrapped=wrapped))
    # "ccx" is a display name, not the count() lookup key.
    toffolis = resources.count_controls("x", 2)
    higher_x = sum(
        resources.count_controls("x", controls)
        for controls in range(3, table.num_address + 3))
    fredkins = resources.count_controls("swap", 1)
    total = (table.num_address + table.num_output + table.num_ladder +
             int(table.controlled or wrapped))
    print(f"{label:24s} qubits={total:2d} ladder={table.num_ladder:2d} "
          f"CCX={toffolis:3d} higher-controlled-X={higher_x:3d} "
          f"controlled-SWAP={fredkins:2d}")
    if not wrapped and toffolis != table.toffoli_count:
        raise AssertionError(f"{label}: compiler/QROM Toffoli counts differ")
    return total


def main():
    start = time.perf_counter()
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))
    rng = np.random.default_rng(7)
    print("Table:", DATA, "; addresses 5, 6, 7 read zero.")
    print("Address 1 is a valid zero-valued entry; padding is also zero.")
    print("All ladder registers start clean and must be restored.")

    # Nonzero output and complex amplitudes test XOR semantics and coherence.
    initial = complex_ket(1 << (ADDRESS_BITS + OUTPUT_BITS), rng)
    marked_input = np.zeros_like(initial)
    marked_input[:1 << ADDRESS_BITS] = complex_ket(1 << ADDRESS_BITS, rng)
    variants = (("select", None), ("select_swap", 2), ("auto", None))
    plain_results = []
    maximum_qubits = 0
    for variant, block_size in variants:
        table = QROM(DATA,
                     ADDRESS_BITS,
                     OUTPUT_BITS,
                     variant=variant,
                     block_size=block_size)
        print(f"\nRequested {variant}, selected {table.variant}, "
              f"block_size={table.block_size}")
        plain_results.append(run_check(f"{variant} XOR", table, initial))
        run_check(f"{variant} lookup/use/unlookup",
                  table,
                  marked_input,
                  phase=True)
        maximum_qubits = max(maximum_qubits,
                             report_resources(f"{variant} plain", table))

        # Native control can change auto's construction and workspace geometry.
        native = QROM(DATA,
                      ADDRESS_BITS,
                      OUTPUT_BITS,
                      variant=variant,
                      block_size=block_size,
                      controlled=True)
        for control_value in (0, 1):
            controlled_input = np.zeros(2 * len(initial), dtype=complex)
            controlled_input[control_value::2] = initial
            run_check(f"{variant} enable={control_value}", native,
                      controlled_input)

        coherent_input = np.zeros(2 * len(initial), dtype=complex)
        coherent_input[0::2] = initial / np.sqrt(2)
        coherent_input[1::2] = 1j * initial / np.sqrt(2)
        run_check(f"{variant} coherent native control", native, coherent_input)
        run_check(f"{variant} coherent wrapped control",
                  table,
                  coherent_input,
                  wrapped=True)
        maximum_qubits = max(
            maximum_qubits,
            report_resources(f"{variant} native control", native),
            report_resources(f"{variant} wrapped control", table,
                             wrapped=True))

    for result in plain_results[1:]:
        # Complete states above check each geometry; these are logical slices.
        check("strategies agree on logical XOR", result, plain_results[0])

    print("\nResource-only pricing (no large statevector is allocated):")
    print(
        "entries width  select-CCX  swap-CCX  auto variant/block/ladder; native auto"
    )
    for entries, width in ((4, 8), (8, 1), (16, 1), (64, 1), (64, 8)):
        data = [(7 * k + 3) % (1 << width) for k in range(entries)]
        bits = (entries - 1).bit_length()
        select = QROM(data, bits, width, variant="select")
        swap_table = QROM(data, bits, width, variant="select_swap")
        auto = QROM(data, bits, width)
        native_auto = QROM(data, bits, width, controlled=True)
        if auto.toffoli_count != min(select.toffoli_count,
                                     swap_table.toffoli_count):
            raise AssertionError(
                "auto did not choose the cheapest coherent price")
        print(
            f"{entries:7d} {width:5d} {select.toffoli_count:11d} "
            f"{swap_table.toffoli_count:9d} "
            f"{auto.variant}/{auto.block_size}/{auto.num_ladder}; "
            f"{native_auto.variant}/{native_auto.block_size}/{native_auto.num_ladder}"
        )
    print(
        "auto minimizes the coherent Toffoli model, preferring SELECT on ties."
    )
    print(
        "It does not enforce a qubit budget. Logical counts are not T counts.")
    print(
        "QROM routing swaps are decomposed; their CCX gates are already included."
    )
    print(f"Maximum executed qubits: {maximum_qubits}; "
          f"elapsed: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
