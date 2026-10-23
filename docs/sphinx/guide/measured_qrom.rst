Measured QROM
=============

``MeasuredQROM`` opts into measurement-based tree uncomputation and optionally
measurement-based cleanup of a known clean SELECT-SWAP cache. It also provides
SELECT-COPY packets with temporary-AND Restore. ``RetainedQROMChain`` shares a
full-word bank and offset cache across an explicitly delimited epoch.
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
``cache_uncompute="traverse"`` runs a second measured walk to restore the bank.
``variant="auto"`` compares SELECT, SELECT-SWAP and SELECT-COPY by emitted
logical Toffolis; the packet section below describes its tie-breaking rules. An explicit block
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

Set ``dirty_banks=True`` with ``variant="select_swap"`` to borrow its bank.
Borrowing is also available for SELECT-COPY and automatic selection; an
automatically selected SELECT needs no borrowed bank. The first ``h`` workspace lines still start and end at zero;
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
and reports ``cache_uncompute == "none"``. Among SELECT and SELECT-SWAP candidates, ties prefer SELECT, then traversal
cleanup, then smaller B, then smaller K. SELECT-COPY must improve the logical
Toffoli count to replace that winner.

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

Measured SELECT-COPY packets
----------------------------

``MeasuredQROM(..., variant="select_copy", block_size=B)`` divides an output
word into packets sharing a bank of ``B-1`` words. For each packet it loads
high-address differences relative to the first word in each block and copies
the selected bank using the low address. Adjacent packet transitions fuse
unload/load work; a final high transition restores the bank. All missing
entries, partial final blocks and addresses above the table are zero padded.
The logical action remains XOR on arbitrary output, including complex
entangled states.

Use one of these explicit packet choices, never both:

* ``alpha=p``: p balanced packets, widest first; for seven bits and p=3 the
  widths are ``(3, 2, 2)``.
* ``packet_width=mu``: full mu-bit packets followed by a nonempty remainder;
  for seven bits and mu=3 the widths are ``(3, 3, 1)``.

Both parameters require positive integers no larger than the output width;
booleans are rejected. With neither parameter the measured selector searches
both balanced and fixed-width layouts. ``lookup.packets`` exposes the resolved
immutable widths, offsets, bank stride and output fanout. These packet options
require explicit SELECT-COPY. A power-of-two B leaves at least one high
address bit. ``cache_uncompute="traverse"`` and ``"auto"`` both use the same
proved SELECT-COPY unload/Restore schedule. Explicit ``"measure"`` and
``phase_block_size`` remain SELECT-SWAP cache-cleanup options and are rejected
for SELECT-COPY; image erasure below is a distinct operation.

With ``dirty_banks=True``, the final high transition restores every borrowed
bank qubit before Restore removes the repeated initial bank offsets from
output. For each bank bit, Restore computes a fresh clean temporary AND,
fans it out to the packet positions that use that bit, X-measures only the
temporary, applies CZ feedback, and resets it. The bank is never measured or
reset. One separate clean scratch is reused serially for these ANDs; it is
included in ``num_clean`` and ``max_ancillas``. Clean-bank mode needs no Restore.

Write ``p=len(packets.widths)``, ``w=max(packets.widths)``,
``l=log2(B)``, ``h=n-l``, ``M=ceil(m/B)``, ``H=C(h,M,c)`` and
``L=C(l,B,c)`` using the C defined above. Set d=1 for dirty-bank mode and
0 otherwise. The emitted schedule has:

* Tree ANDs: ``T_tree=(p+1)*H + (p+d)*L``.
* All temporary ANDs: ``T_temp=T_tree + d*(B-1)*w``.
* Generic copy Toffolis: ``T_generic=(B-1)*b``.
* Total Toffolis: ``T_temp + T_generic``; measurements: ``T_temp``.
* Dirty-bank workspace: ``max(h,l)+1`` clean plus ``(B-1)*w`` borrowed.
* Clean-bank workspace: ``max(h,l)+(B-1)*w`` clean, no borrowed qubits.

Walk and Restore results are consumed immediately, so this load has zero
persistent outcome slots. For a full table without external control, the
formula is ``(p+1)*(M-2)+(p+d)*(B-2)+(B-1)*b+d*(B-1)*w``.
Do not substitute a paper's leading-order coefficient for these finite-tree
counts, control costs, or partial-table counts. The fixed-width option follows
Appendix A's packet geometry, while balanced packets keep their distinct
geometry. The dirty Restore implementation also budgets an extra clean scratch
instead of assuming a walk line is free. No equality with a paper workspace
constant is claimed without a liveness proof.

``max_ancillas`` bounds the complete load ladder, clean plus borrowed, excluding
address and output. Infeasible candidates are removed before ranking; an
explicit geometry over budget raises. Automatic selection prices the emitted
plans, leaving ``QROM`` auto-selection unchanged. Among SELECT-COPY candidates,
ties prefer smaller total workspace, then smaller B, then lexicographic packet
widths. A tie with a SELECT/SELECT-SWAP candidate keeps that candidate. Choosing
Toffolis alone does not minimize measurements, latency, or total hardware cost.

Erasing a known table image
---------------------------

``lookup.erase_kernel`` consumes the clean-copy image
``sum |c,x>|c*data[x]>|eta(c,x)>`` and returns its output register to zero,
preserving address, control, and external correlations. The inactive controlled
branch must have zero output. An uncontrolled image uses c=1. This precondition
also includes zero output at padded addresses. The eraser is not an inverse on
arbitrary output; use ``lookup.kernel`` again for an arbitrary-output XOR inverse.

X-measuring the output yields a parity phase depending on its table word.
A runtime parity correction removes that phase without revealing logical input
information. SELECT-COPY uses its measured phase-copy provider and a restored
borrowed phase bank; SELECT and SELECT-SWAP use a direct one-hot phase provider.
The phase bank is borrowed even if the load used a clean bank. Persistent
classical outcome capacity is the output width and is fresh per invocation.

The eraser has the same argument order as the load, but its independent
``erase_resources``, ``num_erase_clean``, ``num_erase_dirty`` and
``num_erase_ladder`` determine its allocation. ``max_ancillas`` filters the load,
not this separately resolved erase workspace. Never pass an unsliced mixed
ladder to both operations when their clean prefixes differ: a line borrowed by
one stage might be required clean by the other. Allocate separate ladders or
prove disjoint stage views with compatible states, and report their peak live
allocation. Each ladder is restored according to its own contract.

The runnable example uses fixed packets, a coherent external control, arbitrary
complex entangled bank/reference inputs, and separate load/erase ladders. It
compares the full complex vector with an independent XOR permutation at fp64
tolerance ``1e-12``, without phase alignment or renormalization:

.. literalinclude:: ../examples/python/qrom_select_copy_measured.py
   :language: python
   :start-after: [Begin Documentation]

Retaining a bank across operations
----------------------------------

``RetainedQROMChain(tables, address_bits, output_bits, ...)`` keeps a full-word
bank and offset cache live across caller operations. All original tables must
be nonempty and valid; unequal lengths are zero padded before differences.
For m original tables, the required order is:

1. ``begin = chain.begin_kernel`` initializes the offset cache.
2. Invoke all m+1 members of ``chain.kernels()`` in order: the first loads
   table zero, interior steps XOR consecutive table differences, and the final
   step unloads the last word. Place caller operations after each of the first
   m steps.
3. ``end = chain.end_kernel`` restores the original bank and clears the cache.

Mint all kernels outside the enclosing CUDA-Q kernel. Every call uses
``(address, ladder, output)``, with a leading one-qubit control view when
``controlled=True``. All calls share the same ladder geometry: ``num_clean``
zero-in/zero-out lines, including ``num_cache == output_bits`` cache lines,
followed by ``num_dirty`` borrowed lines. Cleanliness and bank restoration apply
at the epoch endpoints; intermediate bank/cache contents remain live.
``dirty_banks=True`` is the default. ``False`` puts the bank in the clean prefix.
``block_size`` may be explicit or selected by logical Toffolis, then workspace,
then smaller B. ``max_ancillas`` includes the entire ladder. This full-word API
has no ``alpha`` or ``packet_width`` option.

During the epoch, leave the ladder untouched and preserve address and control
in their computational basis. Caller operations may act on output and other
data and may be conditioned on address/control: they must commute with Z on
every address/control bit. They may create new entanglement. Address Hadamards,
address permutations and control flips require ending the epoch first, then
beginning a new epoch after the change. Recomputing a cache against an already
changed address cannot repair an open epoch. These are quantum register-use
preconditions; the factory does not inspect arbitrary intervening operations
or promise a runtime lifecycle exception.

The raw difference-step sequence agrees with independent XOR lookups even for
arbitrary output changes. The stronger assertion that output holds table j
between steps and ends zero requires initially zero output and operations that
preserve the encoded word. The following example uses each word only as controls
for noncommuting rotations on a separate data qubit. It compares against original
table-conditioned dense rotation matrices, including inactive-control coherence,
borrowed-bank/reference correlations, and zero final output and clean work:

.. literalinclude:: ../examples/python/qrom_retained_rotations.py
   :language: python
   :start-after: [Begin Documentation]

For s=m+1 difference steps and H,L as above, the complete epoch, including begin
and end, costs ``(s+1)*H + (s+2)*L`` tree ANDs and measurements, plus
``(s+2)*(B-1)*b`` generic copy Toffolis. ``temporary_and_count`` equals the tree
count here: copying into a live cache is a generic Toffoli, not a proven fresh
temporary AND. Dirty mode uses ``max(h,l)+b`` clean and ``(B-1)*b`` borrowed
qubits; clean mode uses their sum as clean workspace. Resources include the
full-word cache, both endpoints and every step; no persistent outcome buffer is
needed. This schedule has distinct costs from independent packet loads and
from coherent ``QROMChain``, which clears its bank after each step.

Relationship to coherent and paper constructions
------------------------------------------------

``QROM`` and ``QROMChain`` remain unitary, with their existing coherent packet
selection and defaults. ``AliasSamplingPrepare`` still uses that unitary path;
its PREPARE/adjoint pair never acquires measurement merely because these adaptive
classes exist. Adaptive alias preparation would need a separate contract.

The multiplexed-copy, cached-offset, fixed-packet and temporary-AND Restore
ideas follow `Motlagh and Pocrnic, Halving the cost of QROM, Sections II.B-D
and Appendices A/B <https://arxiv.org/abs/2605.20334>`_. Here the complete XOR,
image-erasure and retained-epoch contracts are distinct. Their finite-size
counts, extra Restore scratch, arbitrary-output behavior, coherent controls,
padding, and endpoint work are explicit. ``resources`` and ``erase_resources``
are logical ledgers, not physical resource estimates. A compatible synthesis
may price eligible temporary ANDs at four T gates each; generic Toffolis must
be priced separately. Emitting ordinary CCX gates does not establish a compiled
four-T implementation or a paper constant.

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
SELECT-COPY host fixtures additionally check packet Restore, runtime parity
correction, retained epochs, arbitrary bank/reference correlations, and failure
of schedules that change a live address. Independent resource tests count
emitted instructions and all live register widths. Repeated native trajectories
test CUDA-Q bindings as a separate layer. These runnable examples are
statevector checks, not hardware benchmarks.

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
