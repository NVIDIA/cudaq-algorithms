Measured QROM
=============

``MeasuredQROM`` explicitly opts into measurement-based tree uncomputation.
It implements the same logical XOR lookup as ``QROM`` on clean workspace:

.. math::

   |k\rangle |0\ldots0\rangle |y\rangle \longmapsto
   |k\rangle |0\ldots0\rangle |y \mathbin{\mathrm{XOR}} d_k\rangle.

Addresses beyond the table read as zero. The address and output may have
arbitrary complex amplitudes and may be entangled. Registers are disjoint
little-endian views. The ``ladder`` view contains ``num_clean`` clean qubits followed by
``num_dirty`` borrowed qubits, with ``num_ladder`` their sum. The default
requires all workspace clean; blocked lookups also include cache registers.
Existing ``QROM`` consumers continue to use unitary kernels.

Construct and execute
---------------------

Mint the lookup outside the kernel and pass its kernel in. The complete
example below checks the raw complex output vector, including the clean
workspace, against an independent classical XOR permutation and then reads
one classical bit back with ``cudaq.run``:

.. literalinclude:: ../examples/python/measured_qrom.py
   :language: python
   :start-after: [Begin Documentation]

``cudaq.get_state`` and ``cudaq.observe`` are tested on CUDA-Q 0.16's
``qpp-cpu`` target. ``cudaq.run`` requires an entry kernel with a classical
return value, as above. ``cudaq.sample`` rejects measurement-feedback
kernels, direct or nested, on CUDA-Q 0.16; that rejection is CUDA-Q's
compiler check, not a check this package performs, and older CUDA-Q
releases accepted nested feedback while returning samples that are not the
logical lookup. A backend must support mid-circuit measurement and classical
feedback; these tests do not establish support on other backends or
hardware. Do not name your own kernels ``h`` or ``mz``: CUDA-Q 0.16 resolves
bare gate names against kernel decorators visible from the defining frame,
so a user kernel with one of those names breaks the library kernels'
compilation.

For a coherent external control, construct with ``controlled=True`` and call
``lookup(control, address, ladder, output)``, where ``control`` is a one-qubit
view; the control is folded into the tree root. Do not apply ``cudaq.control``
to a measured kernel: on CUDA-Q 0.16 it compiles and runs without any error,
but the mid-circuit measurements cannot be controlled, so each run measures
the caller's control register in the Z basis, and on a superposed control the
resulting state differs from the controlled lookup and from run to run.
``cudaq.adjoint`` of a measured kernel raises. ``controlled=True`` and the
walker's ``kernel_adj`` are the only supported forms.

Applying the lookup twice restores the logical state when the workspace
satisfies the declared clean-prefix and borrowed-suffix contract.
``measured_unary_iteration_kernels`` also exposes ``kernel_adj`` for
noncommuting bodies. That kernel reverses the logical SELECT map on clean
workspace. It does not invert a quantum channel on arbitrary dirty inputs.

Constructions and resource counts
---------------------------------

``variant="select"`` directly walks the table and XORs each word into the
output. ``variant="select_swap"`` walks the high address bits, loads a clean
bank of ``B`` words, routes one word to slot zero using the low address bits,
copies it into the output, reverses routing, and runs a second measured walk
to restore the bank. ``variant="auto"`` minimizes the emitted Toffoli count,
preferring SELECT on ties and otherwise the smaller block. An explicit block
size must be a power of two between 2 and ``2**(address_bits-1)``.

For ``n`` address bits, ``m`` entries, ``b`` output bits, and ``c`` equal to
1 for an external control and 0 otherwise, define

.. math::

   C(n,m,c) = m + n - 2 + c - \operatorname{popcount}(m-1).

With ``h = n-log2(B)`` and ``M = ceil(m/B)``, the exact logical counts are:

.. list-table::
   :header-rows: 1

   * - Construction
     - Toffolis
     - Measurements / tree ANDs
     - Clean / borrowed workspace
   * - SELECT
     - ``C(n,m,c)``
     - ``C(n,m,c)``
     - ``n / 0``
   * - SELECT-SWAP
     - ``2*C(h,M,c) + 2*b*(B-1)``
     - ``2*C(h,M,c)``
     - ``(h + B*b) / 0``
   * - SELECT-SWAP, dirty bank
     - ``2*C(h,M,c) + 4*b*(B-1)``
     - ``2*C(h,M,c)``
     - ``h / (B*b)``

Set ``dirty_banks=True`` only with explicit ``variant="select_swap"`` to
borrow its bank. The first ``h`` workspace lines still start and end at zero;
the bank suffix may be in any state, including entangled with the address,
output, control, or an external reference. Both portions are restored. An
additional routing/copy/unrouting echo removes the borrowed selected word
that the first copy included. In circuit order the sequence is
``W S C S^-1 W S C S^-1``. Writing ``D_r`` for the selected initial bank
word, the output receives ``D_r XOR data[k]`` and then ``D_r``, leaving only
the desired data. This basis-permutation identity also preserves arbitrary
entanglement. With the external control off, the two copies of ``D_r``
cancel. The echo doubles routing cost but leaves tree measurements unchanged.

``toffoli_count`` reports all emitted Toffolis. ``tree_and_count`` counts
only temporary tree ANDs; ``num_measurements`` counts their X-basis erasures.
The general walker additionally exposes ``body_toffoli_count``. Its body
instructions can target data rather than clean ancillas.

The four-T temporary-AND estimate applies only to the tree count, and only
under a compatible synthesis. These kernels emit ordinary Toffolis for both
the tree ANDs and the routing network; a synthesis may realise a routing
Toffoli as a temporary AND as well, at the price of an extra ancilla, so
neither compiled T counts nor physical hardware costs are claimed.
Conditional CZ fixup counts depend on measurement outcomes.
``cudaq.estimate_resources`` reports no T gates for these kernels and follows
its own branch choice for the conditional fixups, so only the unconditional
Toffoli and measurement counts are checked against it.

On a full tree, ``C = 2**n - 2 + c``. For a partial table, this implementation
preserves the full address space, including a no-op outside the table. The
partial-tree formula is therefore distinct from the ``L-1`` accounting under
an input promise restricting addresses to the visited range. A single entry
can still require ``n-1+c`` tree ANDs to test its full address.

Scope and evidence
------------------

Both clean and borrowed SELECT-SWAP banks are restored by a second measured
tree traversal, with the echo above needed for borrowed banks. A different
architecture that erases the entire cache by measurement needs additional
lookup-dependent phase corrections; it does not follow merely by measuring
the existing cache and is not provided here.

Tests cover arbitrary complex XOR inputs, padded tables, explicit coherent
controls, borrowed-bank restoration with spectator entanglement,
noncommuting walker bodies and their logical inverse, controlled
work registers, and a nonzero address-coherence-sensitive observable. Small
walker instances additionally enumerate every measurement history and check
each unnormalized Kraus isometry against an independent direct SELECT map.
Repeated native trajectories test the CUDA-Q implementation as a separate
layer. The runnable example is a statevector check, not a hardware benchmark.

The tree follows `Babbush et al., Section III.A and Figures 4–7
<https://arxiv.org/abs/1805.03662>`_; the temporary-AND erasure follows
`Gidney, Figure 3 <https://arxiv.org/abs/1709.06648>`_. The clean blocked lookup
composes the SELECT-SWAP construction of `Low, Kliuchnikov and Schaeffer
<https://arxiv.org/abs/1812.00954>`_ with those measured tree traversals.
