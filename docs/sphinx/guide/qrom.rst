Coherent QROM and rotation chains
========================================

``QROM`` loads a classical integer table by XOR into a quantum output
register. Each register is little-endian. The lookup restores its clean
workspace and leaves the address unchanged, including for superpositions
and externally controlled calls. Addresses beyond the table length load
zero. Mint the kernel outside the enclosing CUDA-Q kernel:

.. code-block:: python

   import cudaq
   from cudaq_algorithms.primitives import QROM

   lookup = QROM([3, 0, 1, 2, 3], 3, 2,
                 variant="select_copy", block_size=2)
   apply_lookup = lookup.kernel
   workspace = lookup.num_ladder

   @cudaq.kernel
   def circuit():
       address = cudaq.qvector(3)
       ladder = cudaq.qvector(workspace)
       output = cudaq.qvector(2)
       h(address)
       apply_lookup(address, ladder, output)
       # Use output bits as controls on other registers here.
       apply_lookup(address, ladder, output)

``kernel`` is a property. The three register views must be disjoint and
have the requested widths. The ladder, including any block bank, must
start in zero; repeating a lookup then uncomputes it. This promise does
not extend to dirty walk ancillas or to caller operations that change
the output word between applications.

Choosing a construction
-----------------------

``variant="auto"`` compares SELECT, SELECT-SWAP and SELECT-COPY using
their exact logical Toffoli counts. Let ``N`` be the table length,
``A`` the address width, ``b`` the output width, ``B`` a power-of-two
block size, ``l=log2(B)``, ``h=A-l``, and ``W(n,m)`` the coherent unary
walk count, including partial trees:

.. list-table:: Logical costs before external controls
   :header-rows: 1
   :widths: 18 48 34

   * - Construction
     - Toffolis
     - Clean workspace qubits
   * - SELECT
     - ``W(A,N)``
     - ``A``
   * - SELECT-SWAP
     - ``2 W(h,ceil(N/B)) + 2 b (B-1)``
     - ``h + B b``
   * - SELECT-COPY
     - ``(alpha+1) W(h,ceil(N/B)) + alpha W(l,B) + b (B-1)``
     - ``max(h,l) + (B-1) ceil(b/alpha)``

``max_ancillas`` bounds the whole clean workspace, excluding address
and output. For SELECT-COPY, ``alpha`` divides the output into that many
balanced packets, widest first; divisibility is unnecessary. Adjacent
packet bank unload/load walks are already fused. Auto searches all valid
block sizes and packet counts, and rejects an impossible budget. Ties
prefer SELECT, then SELECT-COPY, then SELECT-SWAP, then smaller packet
count and smaller block size. Explicit block sizes require a blocked
variant and at least two address bits.

For 1,024 entries of four bits, the unconstrained choice costs 253
Toffolis (``B=32, alpha=1``). With a budget of 24 clean workspace qubits,
auto chooses ``B=16, alpha=4`` at 591 Toffolis. These are logical circuit
counts, not measured hardware cost. ``describe()`` lists the emitted
gates and identifies counted Toffolis.

``AliasSamplingPrepare`` accepts the same ``variant``, ``block_size``,
``alpha`` and ``max_ancillas`` options. Its budget applies only to the
QROM workspace; its other garbage registers remain additional.

Borrowed bank qubits
------------------------------

``QROM(..., variant="select_copy", dirty_banks=True)`` and the analogous
``select_swap`` option accept an arbitrary bank state, including
entanglement with registers outside the lookup. Both restore that bank
exactly. The ladder view is divided into ``num_clean`` zero-in/zero-out
walk qubits followed by ``num_dirty`` borrowed bank qubits;
``num_ladder = num_clean + num_dirty``. This explicit mode requires a
blocked variant. Walk qubits must still start in zero.

After restoring the bank, an extra coherent copy cancels its original
value from the output. SELECT-SWAP uses a route/copy/unroute echo, costing
``2 W(h,ceil(N/B)) + 4 b (B-1)`` total Toffolis. SELECT-COPY shares one
additional low-address walk across all packet outputs, costing
``(alpha+1) W(h,ceil(N/B)) + (alpha+1) W(l,B) + 2 b (B-1)`` total.
The echoes need no extra workspace. The budget continues to bound the
combined ladder size, and block/packet optimization uses these echo
costs. ``QROMChain`` and ``AliasSamplingPrepare`` use clean banks; a
chain's independently selected layouts can have different clean-prefix
widths and therefore cannot directly share arbitrary borrowed suffixes.

Sequential lookups
------------------

``QROMChain(tables, A, b)`` provides ``m+1`` step kernels for ``m``
tables through ``kernels()``. The first loads table zero, interior
steps load consecutive table XOR differences, and the last unloads the
final table. All steps share one clean ladder of ``num_ladder`` qubits.
Every original table must be nonempty and valid before XOR preprocessing;
different lengths are supported by zero padding.

Interleave operations after each load. When the address and ladder are
unchanged, the fused circuit is equivalent to separate
load/operate/unload circuits, including arbitrary operations on output
and other data registers. The output holds each table value and finishes
at zero only when callers preserve the encoded word, for example by
using it as controls for rotations on a separate register. An operation
such as ``h(output[0])`` changes that word and removes this stronger
guarantee.

``fused=False`` provides the separate-lookup reference. With equal-length
tables and per-lookup cost ``C``, fused cost is ``(m+1) C`` versus
``2 m C``. The reduction is strict only for ``m>1`` and ``C>0``.
Unequal-length chains are priced per step. Each step clears its bank;
the chain does not keep a bank live across the caller's operations.

The following executable example applies three noncommuting rotations
using loaded integer words as binary controls. Its independent dense
oracle checks the complete complex state, including clean output and
ladder registers:

.. literalinclude:: ../examples/python/qrom_controlled_rotations.py
   :language: python

Relationship to the original construction
-----------------------------------------

The multiplexed-copy, packet and XOR-transition ideas follow
`Motlagh and Pocrnic, Halving the cost of QROM, Sections II.B-D
<https://arxiv.org/abs/2605.20334>`_. This API implements coherent
adaptations with clean or explicitly borrowed banks. The paper's headline costs use a
different register and uncomputation model. In Appendices A/B,
``Sel`` adjoint restores the borrowed bank; Restore then removes the
repeated dirty offsets in the outputs. Measurement disposes temporary
ANDs, rather than erasing the borrowed bank.

Matching those costs would additionally require the retained-bank/cache
schedule across caller operations and temporary-AND fanout Restore with
measurement/feed-forward accounting. This implementation instead uses
the explicit coherent echoes above and checks arbitrary entangled bank
restoration. The optimized cache and adaptive Restore form a separate
construction; they are not obtained by relaxing the clean walk-prefix
requirement. Balanced packets here also differ from Appendix A's fixed
packet width followed by a shorter remainder.
