Coherent circuit primitives
===========================

These building blocks compose inside CUDA-Q kernels. The runnable examples
compare their complete outputs with independent classical references.

Modular arithmetic and comparison flags
---------------------------------------

All integer registers are unsigned and little-endian: bit zero has weight
one. Addition wraps modulo :math:`2^n`. Register views passed to an operation
must be pairwise disjoint and have the documented widths; the device kernels
do not check aliasing or validate inputs at runtime.

Import these kernels from `cudaq_algorithms.primitives`. Their signatures
are rendered explicitly here because automatic API documentation does not
render decorated CUDA-Q kernels.

Register and constant arithmetic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ripple-carry CDKM family restores its work and carry registers:

.. code-block:: text

   add_register(a: qview, b: qview, carry: qview)
   subtract_register(a: qview, b: qview, carry: qview)
   add_constant(target: qview, constant_bits: list[int],
                work: qview, carry: qview)
   subtract_constant(target: qview, constant_bits: list[int],
                     work: qview, carry: qview)

`a` and `b` have width n. Addition preserves a and maps b to
`(a+b) mod 2**n`; subtraction maps b to `(b-a) mod 2**n`.
The carry is a one-qubit view initially in zero. Constant arithmetic takes
exactly n little-endian bits for K, an n-qubit clean work register, and a
clean one-qubit carry. For example, adding three to a three-bit seven gives
two, with all workspace restored.

The QFT family takes an integer constant and needs no additional workspace:

.. code-block:: text

   add_constant_qft(target: qview, constant: int)
   subtract_constant_qft(target: qview, constant: int)

Both families implement the same modular permutation. Their supplied inverse
kernels should be used for round trips. The example also tests each forward
action independently: two incorrect operations can cancel in a round trip.

Comparison flags are XOR outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   cmp_ge_register(a: qview, b: qview, carry: qview, out: qview)
   cmp_ge_constant(x_reg: qview, complement_bits: list[int],
                   k_is_zero: int, work: qview, carry: qview, out: qview)

`out` is a one-qubit view and obeys `out ^= predicate`. It is not overwritten
with a classical Boolean. Preloaded-one and coherent flags are valid. The
register comparator preserves both n-bit operands and restores its clean
carry.

For the constant comparator, use the n low bits of `2**n-K`. The allowed
threshold range is `0 <= K <= 2**n`. At K=0 pass all-zero bits and
`k_is_zero=1`, producing an always-true predicate. At K=2**n pass all-zero
bits and `k_is_zero=0`, producing always false. An n-qubit work register and
one-qubit carry start and finish at zero.

A useful coherent pattern is compute/use/uncompute: start a flag at zero,
compute `x >= K` into it, apply Z to the flag, and repeat the comparator.
The operand acquires :math:`(-1)^{[x\ge K]}`, while the flag and workspace
return to zero. This is a building block for threshold selection and
boundary masking.

The QFT comparator temporarily shifts its operand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   cmp_ge_constant_qft_shift(x_reg: qview, constant: int,
                            invert: int, out: qview)
   cmp_ge_constant_qft_shift_adj(x_reg: qview, constant: int,
                                invert: int, out: qview)

At `invert=0` it XORs `x >= K` into out and leaves x at
`(x-K) mod 2**n` for positive K. At `invert=1` it flags `x < K`.
The allowed threshold range is again `0 <= K <= 2**n`; K=0 leaves x
unshifted. The implementation supports n <= 61.

Use only the flag between compute and its supplied adjoint. The adjoint
restores both operand and flag after a phase mark. Treating this compute
half as an operand-preserving comparator would consume the wrong integer.

Resources and correctness
~~~~~~~~~~~~~~~~~~~~~~~~~

At the current implementation, CDKM modular add/subtract uses `2*n-2`
Toffolis, while register comparison uses `2*n-1`. QFT constant addition
uses controlled and single-qubit rotations instead. Zero Toffolis does not
mean zero fault-tolerant synthesis cost.

The :doc:`../examples_rst/primitives` example uses nonuniform complex
superpositions, explicit overflow inputs, logical permutation references,
and complete-state cleanup checks. Its resource-only width sweep excludes
input-state preparation and reports rotations separately from Toffolis.
