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

One encoding ships in the package — `PauliLCU` (from a Pauli sum) — and
any object satisfying the structural
:class:`cudaq_algorithms.block_encoding.BlockEncoding` protocol plugs into
the same consumers. A complete worked encoding (the double-factorized
example, below) demonstrates the protocol at full scale.

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

Because conformance is structural, a block-diagonal or any other custom
encoding plugs into `Walk` and `QSVT` simply by implementing the surface
above — no base class, no registration. The worked example
``06_bring_your_own_encoding.py`` (:doc:`../examples_rst/getting_started`)
implements a minimal `BlockEncoding` from scratch and runs it through the
shared consumers, validating it against a dense reference.

Sparse-access encodings
-----------------------

:mod:`cudaq_algorithms.sparse` block-encodes a real sparse matrix given as
data (a dense array, a ``scipy.sparse`` matrix, or integer-indexed
``(rows, cols, vals)`` triples) or as structure oracles, with two
constructions whose normalizations favour different matrices:

- `SparseOracleEncoding` is the sparse-access construction of Camps,
  Lin, Van Beeumen and Yang (`arXiv:2203.10236
  <https://arxiv.org/abs/2203.10236>`_): ``U_A = T† S T`` from a
  *location* oracle (the column structure) and a *value* oracle
  (fixed-point matrix-element angles), bundled as `OracleKernels`.
  ``alpha = d_padded * h`` pays the largest element ``h`` at the padded
  slot count, so it wins on uniform structured matrices with few nonzeros
  per row. `banded_oracles` builds the bundle for constant-valued bands,
  `qrom_oracles` for arbitrary entries; the value oracle writes
  `quantized_angle` fixed-point angles with ``value_bits`` of precision.
  Negative off-diagonal entries are fine; a negative diagonal is not
  representable here (use the LCU construction or a diagonal shift).
- `SparseLCUEncoding` is a linear combination of unitaries over the
  entries themselves — transposition and reflection unitaries, one pair
  per symmetric off-diagonal entry and one per diagonal entry — with the
  alias-sampled PREPARE and unary-iteration SELECT primitives.
  ``alpha = sum_{i<j} |H_ij| + sum_i |H_ii|`` is the term one-norm, so it
  wins whenever a few entries dominate, and it supports negative
  diagonals. ``mu`` sets the alias-table precision (``discretization_bound``
  per term weight; ``alpha`` stays the exact one-norm).

.. code-block:: python

   from cudaq_algorithms import encode_sparse, SparseLCUEncoding, Walk

   enc = encode_sparse(matrix, mu=4, value_bits=6)   # cheaper alpha wins
   enc.report["path"], enc.report["alpha_alternatives"], enc.report["qubits"]
   enc = encode_sparse(matrix, prefer="oracle")      # override the pricing
   enc = encode_sparse(a, dilate=True)               # [[0, A], [A^T, 0]] of a general A
   walk = Walk(enc)                                  # same consumers as PauliLCU

`encode_sparse` prices both constructions classically, builds only the
winner (ties go to fewer qubits), and attaches the comparison as
``encoding.report`` (``path``, ``alpha``, ``alpha_alternatives``,
``qubits``, ``nnz``, ``max_row_nnz``, ``slots``, ``dim``, ``dilated``,
``ineligible``). Both constructions encode a *symmetric* matrix: a
non-symmetric input raises unless ``dilate=True`` asks for the Hermitian
dilation ``[[0, A], [A^T, 0]]`` on one extra system qubit (the halves bit
is the most significant), as `SparseLCUEncoding.from_general` and
`SparseOracleEncoding.from_general_oracles` do directly.

**What the oracle contract requires.** An `OracleKernels` bundle supplies
``o_loc`` / ``o_val`` and their explicit inverses, the slot count ``d``,
the value bound ``h``, ``value_bits``, the work-qubit count and the
reverse-slot involution; every inverse must hold on states whose work
register enters ``|0>`` (flat, work-free oracles such as `banded_oracles`
invert globally, table oracles built on unary-iteration walks only with
clean scratch). The bundle records the system width it was built for and
the encoding rejects a mismatch.

**Register widths and moments.** ``num_ancilla`` counts every ancilla,
scratch included; `SparseOracleEncoding` exposes ``num_block_ancilla`` and
``num_scratch``, `SparseLCUEncoding` ``num_index``, ``num_garbage``
(``2 num_index + 2 mu + 2``) and ``num_select_work``. Neither encoding
provides `select_observable`, so `Walk.moment` serves even orders only, and
at these widths the even-order reflection observable would expand to
``2^num_ancilla`` Pauli terms: `Walk` refuses above 20 ancillas. Take
Chebyshev data from ``walk_kernel`` powers on a statevector simulator.

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
