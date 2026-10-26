Measured QROM
=============

``MeasuredQROM`` opts into measurement-based tree uncomputation and optionally
measurement-based cleanup of a known clean SELECT-SWAP cache.
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
hardware. The shared interpreter explicitly binds ``h``, ``mz``, ``x``,
``y``, ``z``, and ``cx`` so caller kernels with those names do not replace
its gates. Gate-name shadowing has dedicated regression coverage.

For a coherent external control, construct with ``controlled=True`` and call
``lookup(control, address, ladder, output)``, where ``control`` is a one-qubit
view; the control is folded into the tree root. Do not apply ``cudaq.control``
to a measured kernel: on CUDA-Q 0.16 it compiles and runs without any error,
but the mid-circuit measurements cannot be controlled, so they act on the
caller's control register: a fix-up that fires collapses a superposed control
to one, and one that does not fire attenuates the control-on branch, so the
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
copies it into the output, and reverses routing. The default
``cache_uncompute="traverse"`` runs a second measured walk to restore the bank. ``variant="auto"`` minimizes the emitted Toffoli count,
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

Clean-cache measurement
-----------------------

For explicit ``variant="select_swap"``, ``cache_uncompute="measure"``
replaces the final bank-writing traversal by X-basis measurement, conditional
X reset, and a data-dependent phase correction. It requires a known clean
initial bank; ``dirty_banks=True`` rejects this mode. Measuring an unknown
borrowed bank would destroy its state and its entanglement. The dirty echo
above remains the implementation for borrowed banks.

``cache_uncompute="auto"`` considers measurement for clean blocked candidates
and chooses it only for a strict Toffoli saving; traversal wins cleanup ties.
The default ``"traverse"`` preserves previous choices. SELECT has no cache
and reports ``cache_uncompute == "none"``. Automatic variant selection still
prefers SELECT on ties and otherwise the smaller block.

Write ``w=B*b`` and choose ``K=2**s`` with
``1 <= K <= min(w, 2**(h-1))``. Explicit ``phase_block_size=K`` is accepted
only with explicit SELECT-SWAP and measurement cleanup. Otherwise measurement
chooses the K minimizing correction Toffolis, preferring smaller K on ties.
K=1 uses a direct phase walk; K>1 reuses K erased cache lines as a one-hot
register. No additional quantum lines are needed, but ``w`` classical
outcome slots persist through the correction, freshly initialized on every
invocation. This is retained buffer capacity, not total classical memory.

After unrouting, cache slot i contains the controlled table word at
``B*j+i``. Let F(j) pack these words little-endian, zero-padding missing
entries and blocks. For cache outcomes z, X-measurement contributes

.. math::

   2^{-w/2}(-1)^{c\langle z,F(j)\rangle},

where here c is the quantum control bit (or one for an uncontrolled lookup).
The phase oracle applies the same sign to cancel it. Outcomes therefore reveal
no address or control information. It depends on the high address j, because
routing has already been undone. The output is never measured. One-hot
cleanup restores each split's parent with CX, then erases its temporary AND
with H, measurement, CZ feedback, and X reset. This remains valid after phase
correction because the one-hot support relations are unchanged.

For the resource formulas, c again denotes whether the construction has an
external control. Set ``L=C(h,M,c)``, ``P=C(h-s,ceil(M/K),c)``, and
``R=2*b*(B-1)``. Measurement cleanup emits:

* Toffolis: ``L+R+P+(K-1)``.
* Measurements: ``L+w+P+(K-1)``.
* Tree ANDs: ``L+P``; all temporary ANDs: ``L+P+(K-1)``.
* Clean workspace: ``h+w``; persistent classical bits: ``w``.

For ``n=8, m=256, b=2, B=4, K=8`` without external control,
``L=62, P=6, R=12``. The stages are load (62 Toffolis, 62 measurements),
route/copy/unroute (12, 0), cache erasure (0, 8), one-hot preparation
(7, 0), phase walk (6, 6), and one-hot cleanup (0, 7).
The total is **87 Toffolis, 83 measurements, and 14 clean qubits**, versus
**136 Toffolis, 124 measurements, and 14 clean qubits** for traversal.
These are exact emitted logical counts, not a hardware benchmark.

``resources`` exposes immutable totals and stage records, including
``classical_bits`` and ``temporary_and_count``. ``toffoli_count`` counts
all emitted Toffolis; ``tree_and_count`` excludes routing and one-hot
preparation. ``num_measurements`` includes cache and one-hot erasures.
The four-T estimate applies to proven temporary ANDs under compatible
synthesis; emitted ordinary Toffolis do not establish compiled T counts.
Conditional Z/CZ counts depend on outcomes. Compiler resource reports follow
a branch choice, so unconditional Toffoli and measurement totals are the
appropriate comparison. Automatic selection optimizes Toffolis only, not
measurement latency, classical parity work, Clifford counts, or hardware cost.
The K bound excludes zero-width correction trees and additional workspace;
no optimum over other architectures is claimed.

On a full tree, ``C = 2**n - 2 + c``. For a partial table, this implementation
preserves the full address space, including a no-op outside the table. The
partial-tree formula is therefore distinct from the ``L-1`` accounting under
an input promise restricting addresses to the visited range. A single entry
can still require ``n-1+c`` tree ANDs to test its full address.

Scope and evidence
------------------

Tests cover arbitrary complex XOR inputs, padded tables, explicit coherent
controls, borrowed-bank restoration with spectator entanglement,
noncommuting walker bodies and their logical inverse, controlled
work registers, and a nonzero address-coherence-sensitive observable. Small
walker and complete cache-erasure plans additionally enumerate every measurement
history and check each unnormalized Kraus isometry against independent direct
SELECT or XOR maps. Cache fixtures cover both coherent control modes, K=1 and
K=2, partial/full tables, and repeated application. This host all-history
evidence is distinct from native sampled trajectories.
Repeated native trajectories test the CUDA-Q implementation as a separate
layer. The runnable example is a statevector check, not a hardware benchmark.

The tree follows `Babbush et al., Section III.A and Figures 4–7
<https://arxiv.org/abs/1805.03662>`_; the temporary-AND erasure follows
`Gidney, Figure 3 <https://arxiv.org/abs/1709.06648>`_. The clean blocked lookup
composes the SELECT-SWAP construction of `Low, Kliuchnikov and Schaeffer
<https://arxiv.org/abs/1812.00954>`_ with those measured tree traversals.

The cache X-erasure, phase table, and one-hot cleanup follow the construction
in `Berry et al., Appendix C, Theorem 3 and Figures 5–10
<https://arxiv.org/abs/1902.02134>`_. The formulas above account separately
for this API's padding and coherent-control contract; they do not substitute
the paper's auxiliary-register bounds for this implementation's counts.
