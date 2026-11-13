Reversible arithmetic
=====================

These device kernels are imported from ``cudaq_algorithms.primitives``
and called inside a ``@cudaq.kernel``. They mutate the supplied registers
and return no classical result. The signatures below are registered
explicitly with Sphinx because autodoc does not render decorated CUDA-Q
kernels.

.. currentmodule:: cudaq_algorithms.primitives

Register conventions
--------------------

Integer registers are nonempty, unsigned, and little-endian: qubit zero
has weight one. An n-qubit binary register represents integers modulo
``2**n``. Every quantum view passed to an operation must be pairwise
disjoint from the others and have the stated width. Carry/work views
start in zero and are restored to zero unless stated otherwise.

These are caller preconditions, not runtime validation. The first operand
determines n; extra bits in a longer second operand are ignored, while
short operands or short constant-bit lists can cause out-of-bounds access.
Pass matching operand widths and exactly n constant bits.

Flags are XOR outputs: a comparison toggles ``out[0]`` by its predicate.
The one-qubit flag can start in zero, one, or a coherent state. It is
distinct from clean workspace. All maps extend coherently to superposed
inputs, preserving the phases needed by compute/use/uncompute.

Use the supplied inverse kernels. The CDKM family requires its explicit
inverses; ``cudaq.adjoint`` is not supported for those kernels. The QFT
family also supplies explicit inverses.

See :doc:`../guide/primitives` for the CDKM/QFT comparison and
:doc:`../examples_rst/primitives` for the executable arithmetic tutorial.

CDKM addition and subtraction
-----------------------------

.. py:function:: add_register(a: cudaq.qview, b: cudaq.qview, carry: cudaq.qview)

   Add a into b modulo ``2**n``, preserving a:
   ``|a,b,0> -> |a,(b+a) mod 2**n,0>``.

   :param a: n-qubit addend, restored after use.
   :param b: n-qubit target.
   :param carry: One clean qubit, restored to zero.

   The inverse is :func:`subtract_register`. Cost: ``2*n-2`` Toffolis
   (zero for n=1). Overflow is discarded; there is no carry-out result.

.. py:function:: subtract_register(a: cudaq.qview, b: cudaq.qview, carry: cudaq.qview)

   Subtract a from b modulo ``2**n``, preserving a:
   ``|a,b,0> -> |a,(b-a) mod 2**n,0>``.

   :param a: n-qubit subtrahend, restored after use.
   :param b: n-qubit target.
   :param carry: One clean qubit, restored to zero.

   This is the hand-written inverse of :func:`add_register`, with the
   same workspace contract and ``2*n-2`` Toffoli cost.

.. py:function:: add_constant(target: cudaq.qview, constant_bits: list[int], work: cudaq.qview, carry: cudaq.qview)

   Replace target by ``(target+K) mod 2**n``.

   :param target: n-qubit binary target.
   :param constant_bits: Exactly n bits, with bit k equal to ``(K >> k) & 1``
                         for ``0 <= K < 2**n``.
   :param work: n clean qubits, restored to zero.
   :param carry: One clean qubit, restored to zero.

   The constant is loaded into work, added, and unloaded. The inverse
   is :func:`subtract_constant`; each direction uses ``2*n-2`` Toffolis.

.. py:function:: subtract_constant(target: cudaq.qview, constant_bits: list[int], work: cudaq.qview, carry: cudaq.qview)

   Replace target by ``(target-K) mod 2**n``.

   :param target: n-qubit binary target.
   :param constant_bits: Exactly n little-endian bits of K,
                         with ``0 <= K < 2**n``.
   :param work: n clean qubits, restored to zero.
   :param carry: One clean qubit, restored to zero.

   This is the hand-written inverse of :func:`add_constant`, with
   the same ``2*n-2`` Toffoli cost.

CDKM comparisons
----------------

.. py:function:: cmp_ge_register(a: cudaq.qview, b: cudaq.qview, carry: cudaq.qview, out: cudaq.qview)

   Apply ``out[0] ^= (a >= b)`` using unsigned comparison.

   :param a: n-qubit operand, preserved.
   :param b: n-qubit operand, preserved.
   :param carry: One clean qubit, restored to zero.
   :param out: One-qubit XOR flag; any input state is allowed.

   No n-qubit work register is needed. Applying this kernel twice with
   the same operands uncomputes the flag. Cost: ``2*n-1`` Toffolis.

.. py:function:: cmp_gt_register(a: cudaq.qview, b: cudaq.qview, carry: cudaq.qview, out: cudaq.qview)

   Apply the strict unsigned comparison ``out[0] ^= (a > b)``.

   :param a: n-qubit operand, preserved.
   :param b: n-qubit operand, preserved.
   :param carry: One clean qubit, restored to zero.
   :param out: One-qubit XOR flag; any input state is allowed.

   This kernel is self-inverse under its workspace preconditions and
   has the same ``2*n-1`` Toffoli cost as :func:`cmp_ge_register`.

.. py:function:: cmp_ge_constant(x_reg: cudaq.qview, complement_bits: list[int], k_is_zero: int, work: cudaq.qview, carry: cudaq.qview, out: cudaq.qview)

   Apply ``out[0] ^= (x >= K)``, preserving x.

   :param x_reg: n-qubit binary operand, preserved.
   :param complement_bits: Exactly n little-endian bits of
                           ``(2**n-K) mod 2**n``.
   :param k_is_zero: 1 exactly when K=0; otherwise 0.
   :param work: n clean qubits, restored to zero.
   :param carry: One clean qubit, restored to zero.
   :param out: One-qubit XOR flag; any input state is allowed.

   The threshold satisfies ``0 <= K <= 2**n``. Both endpoints use
   all-zero complement bits: K=0 and ``k_is_zero=1`` always toggle out;
   K=``2**n`` and ``k_is_zero=0`` never toggle it. Applying the kernel
   twice uncomputes its flag. Cost: ``2*n-1`` Toffolis for positive K,
   and no Toffolis for K=0.

Fourier transforms and phase addition
-------------------------------------

.. py:function:: qft(reg: cudaq.qview)

   Apply the QFT without final bit-reversal swaps.

   :param reg: n-qubit register, initially in the binary basis.

   On a basis input x, qubit t has the factor
   ``(|0> + exp(2*pi*i*x/2**(t+1))*|1>)/sqrt(2)``. This ordering is
   the convention consumed by :func:`phase_add_constant` and undone by
   :func:`iqft`. No additional workspace is required.

.. py:function:: iqft(reg: cudaq.qview)

   Apply the hand-written inverse of the swap-free :func:`qft`.

   :param reg: n-qubit register in that Fourier convention.

   The output is back in the binary basis, with no additional workspace.
   Do not insert bit-reversal swaps between these transforms and
   :func:`phase_add_constant`.

.. py:function:: phase_add_constant(reg: cudaq.qview, constant: int)

   Add K modulo ``2**n`` to a register already in the swap-free Fourier
   basis. No forward or inverse QFT is included.

   :param reg: n-qubit Fourier-basis register, with ``1 <= n <= 62``.
   :param constant: Integer K, interpreted modulo ``2**n`` and
                    representable by the kernel's signed 64-bit integer.

   Qubit t receives phase ``2*pi*(K mod 2**(t+1))/2**(t+1)``.
   The register stays in the Fourier basis. Use ``-K`` to invert the
   operation, provided the negated constant is representable.
   For binary inputs use :func:`add_constant_qft` instead.

.. py:function:: add_constant_qft(reg: cudaq.qview, constant: int)

   Replace binary reg by ``(reg+K) mod 2**n`` using QFT,
   :func:`phase_add_constant`, and inverse QFT.

   :param reg: n-qubit binary target, with ``1 <= n <= 62``.
   :param constant: Integer K, interpreted modulo ``2**n`` and
                    representable by the kernel's signed 64-bit integer.

   No workspace is required. The inverse is :func:`subtract_constant_qft`.
   The implementation uses ``n*(n-1)`` controlled phase rotations and
   n single-qubit phase rotations, with no Toffolis. Rotation synthesis
   costs are not included in these logical counts.

.. py:function:: subtract_constant_qft(reg: cudaq.qview, constant: int)

   Replace binary reg by ``(reg-K) mod 2**n``.

   :param reg: n-qubit binary target, with ``1 <= n <= 62``.
   :param constant: Integer K; both K and its negation must fit the
                    kernel's signed 64-bit integer.

   This is the inverse of :func:`add_constant_qft`, implemented using
   phase addition by ``-K`` between QFT and inverse QFT. It needs no
   workspace and has the same logical rotation count.

QFT comparisons with a shifted operand
--------------------------------------

.. py:function:: cmp_ge_constant_qft_shift(x_reg: cudaq.qview, constant: int, invert: int, out: cudaq.qview)

   Toggle a comparison flag and leave the operand shifted:
   ``|x,f> -> |(x-K) mod 2**n, f XOR predicate(x)>``.

   :param x_reg: n-qubit binary operand, with ``1 <= n <= 61``.
   :param constant: Threshold K, with ``0 <= K <= 2**n``.
   :param invert: 0 selects ``x >= K``; 1 selects ``x < K``.
   :param out: One-qubit XOR flag; any input state is allowed.

   The predicate uses the original x. At K=0 the operand is unshifted;
   the predicate is always true for ``invert=0`` and always false for
   ``invert=1``. At K=``2**n`` the modular shift is also zero, and these
   constant predicates are reversed.

   No additional workspace is required. Between this kernel and
   :func:`cmp_ge_constant_qft_shift_adj`, use the flag while leaving
   x_reg untouched. To restore x while retaining the computed flag,
   follow the compute half with :func:`add_constant_qft` using K.
   Use :func:`cmp_ge_constant` for an operand-preserving comparator.

.. py:function:: cmp_ge_constant_qft_shift_adj(x_reg: cudaq.qview, constant: int, invert: int, out: cudaq.qview)

   Apply the hand-written inverse of
   :func:`cmp_ge_constant_qft_shift` with the same K and invert value.

   :param x_reg: n-qubit shifted operand, with ``1 <= n <= 61``.
   :param constant: The compute half's threshold, ``0 <= K <= 2**n``.
   :param invert: The compute half's predicate selector, 0 or 1.
   :param out: The one-qubit XOR flag from the compute half.

   On the compute half's output, this restores the original operand and
   flag without extra workspace. In a compute/Z/uncompute phase oracle,
   the flag returns to its input state and the selected phase remains
   on the operand.
