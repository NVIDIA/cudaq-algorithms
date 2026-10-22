Block encodings
===============

A *block encoding* embeds a (generally non-unitary) operator as a sub-block
of a larger unitary. Concretely, an encoding provides a unitary ``U_A``
acting on a system register plus one or more ancilla (signal) qubits such
that projecting the ancillas onto their zero state recovers the target
operator up to a scalar:

.. math::

   \langle 0 |_{\mathrm{anc}} \, U_A \, | 0 \rangle_{\mathrm{anc}}
   \;=\; H / \alpha

Here ``alpha`` is the *subnormalization* (the encoding normalization): the
flagged block implements ``H / alpha``, so ``alpha`` must be at least the
spectral norm of ``H``. Because the ancilla register flags the good block,
the same ``U_A`` drives every downstream algorithm — the qubitization walk
and QSVT (see :doc:`qubitization_qsvt`) are generic over *which* encoding
produced it. Everything below is pure-Python CUDA-Q kernels requiring only
`cudaq` and numpy; qubit ordering, integral-tensor, and normalization
conventions are collected in :doc:`../conventions`.

Two encodings ship in the package, and they play different roles.
`PauliLCU` is the front door for a Hamiltonian given as a Pauli sum: it
reads the terms and builds the encoding with an exact multiplexed-rotation
PREPARE by default, or with the alias-sampled PREPARE and unary-iteration
SELECT under ``backend="alias"``. `LCUBlockEncoding` is the body-programmed
primitive underneath such constructions (the alias backend delegates to
it): it takes term weights and, per term, a SELECT *body* in the
unary-iteration gate vocabulary — controlled Paulis, but also the extended
gates that use work qubits — so its terms need not be Pauli words; it
realizes an alias-sampled PREPARE and a unary-iteration SELECT. Encodings
built from structured data (the double-factorized example below, the
sparse-access encodings) program it directly. Any object satisfying the
structural :class:`cudaq_algorithms.block_encoding.BlockEncoding` protocol
plugs into the same consumers. Two *combinators*, `SumEncoding` and
`RotatedEncoding`, build new protocol-conforming encodings out of existing
ones (see `Combining encodings`_), and the worked double-factorized
encoding demonstrates the protocol at full scale.

The block-encoding idea and ``PauliLCU``
----------------------------------------

`PauliLCU` block-encodes a linear combination of unitaries (LCU): a
Hamiltonian written as a weighted sum of Pauli words ``H = sum_i c_i P_i``.
The subnormalization is the LCU one-norm ``alpha = sum_i |c_i|``.

Construction is flexible — a coefficient dict, a `cudaq.SpinOperator`, or a
list of ``(coeff, word)`` pairs:

.. code-block:: python

   from cudaq_algorithms import PauliLCU

   # Flexible construction: dict, SpinOperator, or (coeff, word) pairs.
   enc = PauliLCU({"ZI": 0.70, "IZ": -0.43, "XX": 0.19, "YZ": 0.11})
   enc = PauliLCU(spin_op, num_qubits=2)
   enc = PauliLCU([(0.7, "ZI"), (-0.43, "IZ")])
   enc = PauliLCU(spin_op, num_qubits=2, backend="alias", mu=4)  # see below

The encoding exposes its geometry and coefficients for inspection:

.. code-block:: python

   enc.num_system, enc.num_ancilla, enc.num_terms
   enc.alpha                  # LCU 1-norm, sum_i |c_i|
   enc.terms                  # [(coeff, word), ...]
   enc.constant_term          # sum of identity terms

   kernel = enc.encode_kernel()          # @cudaq.kernel(state): full U_A

PREPARE, SELECT, UNPREPARE
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LCU unitary factors into three stages on the ancilla superposition:

- **PREPARE** loads the coefficient amplitudes onto the ancilla register,
  producing a superposition weighted by ``sqrt(|c_i| / alpha)``.
- **SELECT** applies each Pauli word ``P_i`` to the system, controlled on
  the corresponding ancilla basis state (with the sign of ``c_i`` folded
  in).
- **UNPREPARE** is PREPARE-dagger, un-computing the ancilla so the flagged
  ``|0...0>`` block holds ``H / alpha``.

These stages are available as composable, module-level device kernels
(`cudaq_algorithms.pauli_lcu.prepare`, ``select``, ``apply``) that compose
inside user kernels, with ``enc.kernel_args`` supplying the flattened arrays
they take as arguments. They live in the module namespace rather than the
package root because their names are too generic to re-export.

The ``"alias"`` backend
~~~~~~~~~~~~~~~~~~~~~~~

``PauliLCU(..., backend="alias", mu=...)`` builds the same encoding from the
alias-sampled PREPARE and unary-iteration SELECT primitives by delegating to
`LCUBlockEncoding` (next section), with each Pauli word as a SELECT body.
The rotations of the exact PREPARE become Toffoli-style gates, at the cost
of extra ancillas — ``3 m + 2 mu + 3`` qubits for ``m`` index bits, so 20
rather than 3 for a five-term Hamiltonian at ``mu = 4`` — and of rounding
each term weight to ``mu`` bits (``alpha`` stays the exact one-norm). The
API is otherwise the same: every kernel factory, `Walk`, `QSVT` and
`select_observable` work unchanged, and `kernel_args` (the exact backend's
flattened arrays) is the one member that raises.

``LCUBlockEncoding``: alias-sampled PREPARE, unary-iteration SELECT
--------------------------------------------------------------------

`LCUBlockEncoding` builds the same ``PREPARE-dagger . SELECT . PREPARE``
structure from the circuit primitives in :mod:`cudaq_algorithms.primitives`:
PREPARE is coherent alias sampling over the term weights and SELECT is a
unary-iteration walk over the term index. It takes the Hamiltonian as term
data rather than Pauli strings — the weights ``|c_k|`` and, per term, a
SELECT *body* describing the Hermitian involution ``U_k`` in the
unary-iteration gate vocabulary — so the term unitaries need not be Pauli
words:

.. code-block:: python

   from cudaq_algorithms import LCUBlockEncoding

   # H = 0.6 Z0 - 0.4 Y1 + 0.5 X0 X1: the weights |c_k| and one SELECT body
   # per term; a negative coefficient is a ("sign",) op on its branch.
   enc = LCUBlockEncoding(
       num_system=2,
       weights=[0.6, 0.4, 0.5],
       bodies=[[("z", 0)], [("y", 1), ("sign",)], [("x", 0), ("x", 1)]],
       mu=4)

   enc.alpha                      # sum_k |c_k|
   enc.num_ancilla                # index + alias garbage + SELECT work
   enc.discretized_weights        # the weights the alias tables realize
   enc.discretization_bound       # per-bin rounding bound, set by mu
   kernel = enc.encode_kernel()   # @cudaq.kernel(state): full U_A

Every body op is validated at construction (known op, operand count,
system targets within ``num_system``, and a Hermitian product for
Pauli-word bodies) with a ``ValueError`` naming the term and op. The alias
tables hold the weights to ``mu`` bits, so the encoded operator is ``H``
with each weight ``|c_k|`` realized within ``alpha * discretization_bound``
while ``alpha`` stays the exact one-norm. The ancilla register is
``index(m) + garbage(2m + 2mu + 2) + work`` qubits with
``m = max(1, ceil(log2(num_terms)))``, so ``mu`` drives the width: a
five-term, two-qubit Hamiltonian is 22 qubits in total at ``mu = 4`` and
30 at the default ``mu = 8``, sized for resource estimation rather than
for the CI simulators. The walk and QSVT consumers work unchanged. The
walk runs in the PREPARE frame like `PauliLCU`'s, so `Walk.moment` serves
odd orders through the derived `select_observable` whenever every body is
a Pauli word (a body in the extended vocabulary raises
`NotImplementedError` there); even orders use the geometry-only
reflection observable, whose ``2^num_ancilla`` terms are practical only
at small ``mu``.

The ``BlockEncoding`` protocol
------------------------------

`Walk` and `QSVT` are generic over the encoding: they receive an encoding
object and delegate every encoding-specific circuit to it, keeping only
sequencing, control conventions, and measurement for themselves. Any object
satisfying the :class:`cudaq_algorithms.block_encoding.BlockEncoding`
protocol works — conformance is *structural* (`typing.Protocol`,
runtime-checkable), so implementations do not inherit from anything.

The contract that makes the composition work is *data erasure at the kernel
boundary*: every factory returns a `@cudaq.kernel` whose signature is fixed
by the protocol (registers only), with all encoding-specific data already
captured inside the kernel at factory time. The consumers can then call the
injected kernels without knowing anything about the encoding's internals.

An implementation exposes three properties and a family of kernel
factories:

- `num_system` — number of system qubits the encoded operator acts on.
- `num_ancilla` — number of ancilla (signal) qubits flagging the encoded
  block. Consumers require ``num_ancilla >= 1``: the walk's ``-H/alpha``
  sign comes from a reflection about the ancilla zero state, which is a
  no-op on an empty register. `PauliLCU` normalizes single-term inputs to
  one idle ancilla to satisfy this uniformly.
- `alpha` — the block-encoding normalization; the encoded block is
  ``H / alpha``.

Kernel factories (the returned kernels' signatures are fixed; all encoding
data is captured inside at factory time):

- `prepare_kernel` — ``(ancilla: qview)``: PREPARE the ancilla
  superposition.
- `unprepare_kernel` — ``(ancilla: qview)``: PREPARE-dagger.
- `apply_kernel` — ``(ancilla: qview, system: qview)``: the full block
  encoding ``U_A``.
- `controlled_apply_kernel` — ``(control_and_ancilla: qview, system:
  qview)``: ``U_A`` controlled by qubit 0 of the combined register.
- `walk_step_kernel` — ``(ancilla: qview, system: qview)``: one qubitization
  walk step ``W`` (block-encodes ``-H/alpha``).
- `adjoint_walk_step_kernel` — ``(ancilla: qview, system: qview)``: one
  adjoint walk step ``W dagger``.
- `controlled_walk_step_kernel` /
  `controlled_adjoint_walk_step_kernel` — the controlled variants, each
  taking a combined ``[control, ancilla...]`` register.

There is also one observable hook:

- `select_observable` — the odd-moment observable as a `cudaq.SpinOperator`.
  Measured after PREPARE and ``p`` walk steps (no UNPREPARE), its
  expectation is the odd Chebyshev moment ``<T_{2p+1}(H/alpha)>``. The
  construction is encoding-specific (for an LCU it is
  ``sum_i sign_i |i><i|_anc x P_i``); the even-moment reflection observable
  ``2|0..0><0..0| - I`` needs only the register geometry, so `Walk` derives
  it without an encoding hook.

Register conventions shared by all implementations:

- `encode_kernel`-produced kernels allocate the system register from a
  `cudaq.State` first, ancillas after it (so the good subspace is the first
  ``2**num_system`` amplitudes; see ``sim_utils.good_subspace``).
- Controlled variants take a combined ``[control, ancilla...]`` register
  whose qubit 0 is the external control (a CUDA-Q Python control set cannot
  mix a bare qubit with a separate register); with control ``|0>`` they must
  reduce to the identity.
- The flagged block of the *walk step* is ``-H/alpha`` (the sign folded into
  the walk construction); `Walk.moment` and the QSVT response conventions
  rely on this.

Implementing the protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because conformance is structural, a sparse-oracle, block-diagonal, or any
custom encoding plugs into `Walk` and `QSVT` simply by implementing the
surface above — no base class, no registration. The worked example
``06_bring_your_own_encoding.py`` (:doc:`../examples_rst/getting_started`)
implements a minimal `BlockEncoding` from scratch and runs it through the
shared consumers, validating it against a dense reference.

Bring your own encoding: the double-factorized example
------------------------------------------------------

The protocol's worked, full-scale demonstration is a *double-factorized*
encoding of the electronic-structure Hamiltonian (von Burg et al., *PRX
Quantum* **2**, 030305 (2021);
`arXiv:2007.14460 <https://arxiv.org/abs/2007.14460>`_), implemented
entirely against the public API as a runnable example — ``df_encoding.py``
on the :doc:`block-encodings examples page <../examples_rst/block_encodings>`. Its dense-reference test
suite (``tests/python/test_df_encoding.py``) runs in CI, so the example is
held to library-grade correctness, and `Walk`/`QSVT` consume it unchanged:

.. code-block:: python

   import sys
   sys.path.insert(0, "docs/sphinx/examples/python")  # or copy the file
   from df_encoding import DoubleFactorizedEncoding   # the example module

   from cudaq_algorithms import Walk, QSVT
   from cudaq_algorithms import double_factorization as df

   factorization = df.compressed_double_factorization(eri, num_leaves=T)
   encoding = DoubleFactorizedEncoding(one_body, factorization,
                                       scalar_offset=nuclear_repulsion)

   walk = Walk(encoding)                       # same consumers as PauliLCU
   kernel = QSVT(encoding).kernel(sequence)

Use the example file as the template for writing your own encoding; the
construction, its normalization, and a six-molecule benchmark against
`PauliLCU` are documented with the example itself (see
:doc:`../examples_rst/block_encodings`).

Combining encodings
-------------------

Two combinators in :mod:`cudaq_algorithms.combinators` take
`BlockEncoding` objects and return `BlockEncoding` objects, so `Walk` and
`QSVT` consume the result unchanged and the constructions nest:

- `SumEncoding([(c_1, enc_1), ..., (c_K, enc_K)])` block-encodes
  ``sum_i c_i H_i`` from encodings of the ``H_i`` with real, nonzero
  coefficients (``K >= 1``; a single term is "scale by a constant").
- `RotatedEncoding(inner, enter, exit)` block-encodes ``V H V^dagger``
  from an encoding of ``H`` and two ``(system: qview)`` kernels
  implementing ``V^dagger`` (``enter``) and ``V`` (``exit``).

.. code-block:: python

   from cudaq_algorithms import PauliLCU, RotatedEncoding, SumEncoding, Walk

   hopping = RotatedEncoding.from_orbital_rotation(
       PauliLCU(momentum_diagonal_words), modes)   # T, diagonal in momentum space
   interaction = PauliLCU(interaction_words)       # U, diagonal in real space
   combined = SumEncoding([(1.0, hopping), (1.0, interaction)])

   combined.alpha            # == hopping.alpha + interaction.alpha
   walk = Walk(combined)     # same consumers as PauliLCU
   shifted = SumEncoding([(1.0, combined), (-e_ref, PauliLCU({"I" * n: 1.0}))])

**Normalization.** The sum's subnormalization is the LCU triangle bound,
``alpha = sum_i |c_i| alpha_i``; a rotation leaves ``alpha`` unchanged. The
combinator itself therefore never lowers ``alpha`` — the saving comes from
choosing *where* each piece is encoded. A hopping term encoded in momentum
space costs ``sum_k |eps(k)|``, less than the one-norm of its Jordan-Wigner
``XX + YY`` strings, so the sum of a rotated momentum-space ``T`` and a
real-space ``U`` has a smaller ``alpha`` than one flat `PauliLCU` of the
whole Hamiltonian. (The identity part of ``U`` counts in both numbers.)

**Ancilla layout.** `SumEncoding` adds a *slot* register of
``max(1, ceil(log2 K))`` qubits selecting the summand and one *flag* qubit,
and the summands share one ancilla pool as wide as the widest of them:
``num_ancilla = num_slots + 1 + max_i num_ancilla_i``, in the order
``[slot][flag][shared pool]``. PREPARE puts amplitude
``sqrt(|c_i| alpha_i / alpha)`` on slot ``|i>``. For each summand, SELECT
uses the flag to control the signed operand on the contiguous
``[flag | pool]`` view: toggle ``flag ^= (slot == i)``, apply the signed
operand, restore the flag, then repeat the signed operand. The two calls
cancel on inactive slots and apply the signed operand once on the selected
slot, regardless of the incoming flag state, because each operand is an
involution. The flag is restored even when it is entangled with the pool
or system. Pool sharing is sound because SELECT is block-diagonal over the
slot register, each block a unitary on the whole pool, and the encoded block is the
all-zero subspace of the entire ``[slot][flag][pool]`` register, which the
walk's reflection covers; a summand's ancillas need not return to ``|0>``
(an alias-sampled summand's garbage does not). `RotatedEncoding` adds no
ancillas.

**Costs.** Each slot PREPARE or unprepare uses ``2^num_slots - 1`` rotations;
a full block-encoding application uses both. SELECT adds two
``num_slots``-controlled flag flips and two calls to each summand's
`controlled_apply_kernel`, including its PREPARE and unprepare costs. A
negative coefficient also adds two flag Z gates. For controlled SELECT,
the flag flips have one additional external control. The two operand calls
are needed even when the incoming flag happens to be zero, and the same
factor applies at each level of nested sums: a leaf below ``d`` sum levels
is called ``2^d`` times per path. An orbital-rotation
network applies ``num_rotations`` adjacent Givens rotations per spin sector
on entry and again on exit. The ``alpha`` saving is bought with these gates
and the extra ``num_slots + 1`` ancillas.

**Negative coefficients.** A term with ``c_i < 0`` is phased by ``-1`` on
its slot branch, so ``SumEncoding([(1.0, encoding), (-e_ref, identity)])``
with ``identity = PauliLCU({"I" * n: 1.0})`` block-encodes
``H - e_ref I`` at ``alpha + |e_ref|`` — the spectral shift ground-state
filters and QPE decoders need.

**Nesting.** Operands may themselves be sums or rotations: a
`SumEncoding` of a `RotatedEncoding` of a `PauliLCU` is one
`BlockEncoding`, and the walk's reflection acts on the outermost ancilla
register as a whole.

**Requirement on operands.** Every operand's `apply_kernel` must be
Hermitian, hence an involution (``U_A^2 = I``), on its entire ancilla and
system register. Its `controlled_apply_kernel` must implement exact
controlled-``U_A`` for arbitrary incoming ancilla states. These are the
protocol's requirements, which every encoding in this package satisfies;
matching its API members alone does not establish them. Verify both
properties for a foreign encoding before summing it, since the signed
operand cancellation and the walk and QSVT constructions rely on them.

**The orbital-rotation constructor.** `RotatedEncoding.from_orbital_rotation(inner, orthogonal)`
takes a real orthogonal ``n x n`` matrix whose columns are the target
modes, with ``2 n == inner.num_system`` in the interleaved spin-orbital
layout (``2p`` up, ``2p + 1`` down), and compiles it to a network of
adjacent Givens rotations applied to both spin sectors. The network
reproduces ``V`` *up to a sign per mode*. That is immaterial when the inner
encoding is diagonal in the rotated modes (number operators, the intended
use) and must be accounted for otherwise; for an arbitrary ``V``, supply
your own ``enter``/``exit`` kernels.

**Not available: the odd-moment observable.** Both combinators raise
``NotImplementedError`` from `select_observable`. The sum's SELECT drives
each summand's full block encoding — PREPARE trees included — so it has no
compact `cudaq.SpinOperator` form; a rotated encoding's terms are not
computational-frame Pauli words. Even moments (`Walk.moment` with even
order), `Walk.kernel` statevectors at any power, and QSVT sequences are all
available; `Walk.moments`, which alternates even and odd orders, is not.
For an odd order on a statevector simulator, take the good-subspace block
of ``Walk(encoding).kernel(power=k)``, which applies ``T_k(-H/alpha)``.

The worked example ``09_hubbard_combined_encoding.py``
(:doc:`../examples_rst/getting_started`) builds the Fermi-Hubbard
Hamiltonian this way — ``T`` in the analytic momentum basis, ``U`` in real
space — and verifies every block, the controlled variants, the walk
moments, a negative-coefficient shift, and a QSVT sequence against an
independent sparse Jordan-Wigner reference.
