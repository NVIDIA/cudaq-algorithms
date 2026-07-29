Block encodings
===============

A *block encoding* embeds a (generally non-unitary) operator as a sub-block
of a larger unitary. Concretely, an encoding provides a unitary ``U_A``
acting on a system register plus one or more ancilla (signal) qubits such
that projecting the ancillas onto their zero state recovers the target
operator up to a scalar:

.. code-block:: text

   <0|_anc U_A |0>_anc = H / alpha

Here ``alpha`` is the *subnormalization* (the encoding normalization): the
flagged block implements ``H / alpha``, so ``alpha`` must be at least the
spectral norm of ``H``. Because the ancilla register flags the good block,
the same ``U_A`` drives every downstream algorithm — the qubitization walk
and QSVT (see :doc:`qubitization_qsvt`) are generic over *which* encoding
produced it. Everything below is pure-Python CUDA-Q kernels requiring only
`cudaq` and numpy; qubit ordering, integral-tensor, and normalization
conventions are collected in :doc:`../conventions`.

Two encodings ship in the package — `PauliLCU` (from a Pauli sum) and
`DoubleFactorizedEncoding` (from electronic-structure integrals) — and any
object satisfying the structural :class:`cudaq_algorithms.block_encoding.BlockEncoding`
protocol plugs into the same consumers.

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

``DoubleFactorizedEncoding``
----------------------------

`DoubleFactorizedEncoding` block-encodes the electronic-structure
Hamiltonian directly from its double-factorized integrals (von Burg et al.,
*PRX Quantum* **2**, 030305 (2021);
`arXiv:2007.14460 <https://arxiv.org/abs/2007.14460>`_), instead of first
expanding it into Pauli words. It satisfies the same
:class:`cudaq_algorithms.block_encoding.BlockEncoding` protocol as
`PauliLCU`, so `Walk` and `QSVT` consume it unchanged:

.. code-block:: python

   from cudaq_algorithms import DoubleFactorizedEncoding, Walk, QSVT
   from cudaq_algorithms import double_factorization as df

   factorization = df.compressed_double_factorization(eri, num_leaves=T)
   encoding = DoubleFactorizedEncoding(one_body, factorization,
                                       scalar_offset=nuclear_repulsion)

   walk = Walk(encoding)                       # same consumers as PauliLCU
   kernel = QSVT(encoding).kernel(sequence)

``one_body`` is the ``(n, n)`` symmetric core-Hamiltonian matrix and the
second argument is either a ``DoubleFactorization`` (truncation happens
there — ``explicit_double_factorization`` /
``compressed_double_factorization``) or a raw chemist-notation ``(pq|rs)``
tensor, which is factorized exactly. Conventions (spatial orbitals,
interleaved spins ``2p`` up / ``2p + 1`` down, Jordan-Wigner) match
`cudaq_algorithms.chemistry`.

Construction
~~~~~~~~~~~~

The factorized Hamiltonian is regrouped so that every term is *diagonal in
some rotated orbital basis*:

.. code-block:: text

   H = const + sum_k F_k N_k  +  1/2 sum_t sum_kl Z^t_kl (N^t_k - 1)(N^t_l - 1)

- **Frame 0** — the eigenbasis of the corrected one-body matrix ``kappa``
  (raw integrals + the exchange correction ``-1/2 sum_r (pr|rq)`` + the
  one-body remainder from centering the leaf number operators, all
  evaluated on the *factorized* tensor, so a truncated factorization
  encodes exactly its truncated Hamiltonian). Terms: one Z per spin
  orbital, coefficient ``-F_k / 2``.
- **Frames 1..T** — one per factorization leaf, in the leaf's eigenbasis
  ``U^t``. Centering makes each leaf *pure ZZ*: coefficient ``Z_kl / 4`` per
  spin pair for ``k < l``, plus one cross-spin ZZ of ``Z_kk / 4`` per
  diagonal.

SELECT walks through the frames: an **uncontrolled** Givens network rotates
the system into the frame's basis, the frame's Z words execute
**ancilla-controlled**, and the next segment rotates onward — with no
control active the segments telescope to the identity, which is what makes
the controlled variants cheap (only Z words and sign phases carry the
control). Each spatial Givens rotation lifts to two three-qubit `exp_pauli`
pairs (``XZY`` / ``YZX`` on contiguous slices), one per spin. PREPARE and
the walk/QSVT composites are the same machinery `PauliLCU` uses.

alpha and the published one-norm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The subnormalization is the 1-norm of the encoded coefficients, and by
construction it reproduces the LCU one-norm of
``double_factorization.double_factorization_one_norm(..., "lcu")``
(arXiv:2212.07957, Eq. 13) exactly, up to the identity term:

.. code-block:: text

   alpha = |const| + sum_k |F_k| + sum_t ( sum_{k<l} |Z^t_kl| + 1/4 sum_k |Z^t_kk| )

Compressing the factorization (fewer leaves) lowers ``alpha`` and the term
count together — the knob a flat Pauli expansion does not have. Since QSVT
circuit depth for time evolution scales like ``alpha * t``, the compression
translates directly into shallower circuits, at the price of a spectrum
shift bounded by the tensor reconstruction error.

Inspection
~~~~~~~~~~

`num_frames`, `num_givens_rotations`, `num_terms`, `constant_term`,
`factorization`, and `terms` (as ``(coefficient, z_qubits, frame_index)``,
where the qubits are Z positions *in that frame's rotated basis*).

Limitations
~~~~~~~~~~~

- `select_observable` raises `NotImplementedError`: the odd-Chebyshev-moment
  observable is LCU-specific (it needs computational-frame Pauli words).
  Even moments (`Walk.moment` with even order) and every kernel factory work
  unchanged. See :doc:`qubitization_qsvt` for the moment conventions.
- The Givens networks are emitted sequentially (one rotation at a time);
  merging adjacent exit/entry networks into a single relative rotation, and
  parallel-scheduling commuting rotations, are documented future
  circuit-level optimizations.

Example
~~~~~~~

`docs/sphinx/examples/python/df_block_encoding.py` builds a small system,
compares `DoubleFactorizedEncoding` with a `PauliLCU` of the same
Hamiltonian (alpha, term count, structure), sweeps factorization
truncation, and measures Chebyshev moments through the shared `Walk`
consumer.

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

Bring your own encoding
~~~~~~~~~~~~~~~~~~~~~~~~~

Because conformance is structural, a sparse-oracle, block-diagonal, or any
custom encoding plugs into `Walk` and `QSVT` simply by implementing the
surface above — no base class, no registration. The worked example
`docs/sphinx/examples/python/06_bring_your_own_encoding.py` implements a
minimal `BlockEncoding` from scratch and runs it through the shared
consumers, validating it against a dense reference.
