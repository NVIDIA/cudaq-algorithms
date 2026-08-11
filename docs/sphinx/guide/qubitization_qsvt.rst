Qubitization and QSVT
=====================

Given a block encoding (see :doc:`block_encodings`), *qubitization* builds a
walk operator ``W`` whose eigenphases encode the spectrum of ``H / alpha``,
and *quantum singular value transformation* (QSVT) applies a polynomial to
that spectrum by interleaving walk steps with signal-processing phases.
Both `Walk` and `QSVT` are generic over the encoding — they consume any
object satisfying the `BlockEncoding` protocol — and both have controlled
variants. QSVT-based Hamiltonian simulation is one of the two
time-evolution routes in the library; :doc:`trotter` is the other
(product-formula) route.

Everything below is pure-Python CUDA-Q kernels needing only `cudaq` and
numpy. Qubit-ordering and normalization conventions live in
:doc:`../conventions`. Import the object surface from the package root:

.. code-block:: python

   from cudaq_algorithms import Walk, QSVT, PhaseSequence

The qubitization walk
---------------------

`Walk` wraps an encoding and emits the qubitization walk operator ``W`` and
its powers. The constructor holds the problem; the method calls hold the
choices:

.. code-block:: python

   walk = Walk(enc)
   walk.kernel(power=3)                  # PREPARE + W^3 + UNPREPARE
   walk.kernel(power=3, uncompute=False) # ... without UNPREPARE
   walk.adjoint_kernel(power=2)          # (W dagger)^2
   walk.roundtrip_kernel(power=2)        # W^2 (W dagger)^2 == identity
   walk.controlled_kernel(power=2)       # controlled walks

The flagged block of a single walk step is ``-H/alpha`` (the sign is folded
into the walk construction), so on an eigenstate with eigenvalue ``lambda``
the good subspace realizes the Chebyshev response at ``lambda / alpha`` with
no caller-side negation. `roundtrip_kernel` composes a walk with its adjoint
and must collapse to the identity — the property tests pin this.

Chebyshev moments
~~~~~~~~~~~~~~~~~

`Walk` measures Chebyshev moments of the (normalized) Hamiltonian through
`cudaq.observe`, which is a hardware-legitimate measurement path:

.. code-block:: python

   walk.moment(psi, k)                   # <T_k(H/alpha)> via cudaq.observe
   walk.moments(psi, 8)                  # even/odd observable convention

   from cudaq_algorithms import reflection_observable, select_observable
   reflection_observable(enc)            # 2|0..0><0..0| - I  (SpinOperator)
   select_observable(enc)                # sum_i sign_i |i><i| x P_i

There are two moment observables. The **even** moments use the reflection
observable ``2|0..0><0..0| - I``, which needs only the register geometry, so
`Walk` derives it without an encoding hook. The **odd** moments use the
encoding-specific `select_observable`, measured after PREPARE and ``p`` walk
steps (no UNPREPARE), whose expectation is ``<T_{2p+1}(H/alpha)>``. For an
LCU this is ``sum_i sign_i |i><i|_anc x P_i``; encodings that cannot supply
it (for example `DoubleFactorizedEncoding`) raise `NotImplementedError` on
`select_observable`, and only even moments are available there.

QSVT and phase sequences
------------------------

A `PhaseSequence` is the list of signal-processing angles that define the
polynomial to apply; `QSVT` interleaves them with walk steps:

.. code-block:: python

   seq = PhaseSequence(phases)                      # projector convention
   seq = PhaseSequence(phases, convention="qsp")    # QSPPACK convention,
                                                    # converted automatically
   seq = PhaseSequence(phases, walk_directions=["forward", "adjoint"])

   transformer = QSVT(enc)
   kernel = transformer.kernel(seq)                 # @cudaq.kernel(state)
   controlled = transformer.controlled_kernel(seq)  # controlled sequence

`PhaseSequence` accepts phases in either the projector convention (default)
or the QSPPACK ``"qsp"`` convention, which it converts automatically;
``walk_directions`` lets a sequence mix forward and adjoint walk steps.

Sign convention: the walk block-encodes ``-H/alpha`` and the circuits fold
the sign in, so on an eigenstate with eigenvalue ``lambda`` the good-subspace
block implements ``p(lambda / alpha)`` — no caller-side negation. The 2x2
signal-model reference implementing this convention lives in
`tests/python/test_qsvt.py` (``reference_response``), where it serves as the
test oracle. Phase *generation* stays external (see below): the primitives
consume whatever phase list they are given.

Hamiltonian simulation
----------------------

Real-time evolution ``exp(-iHt)`` is not a single polynomial (it is
complex), so QSVT realizes its real and imaginary parts separately — a
cosine and a sine polynomial in ``H/alpha`` — and they are recombined:

.. code-block:: python

   from cudaq_algorithms import recover_real_time_evolution

   evolved = recover_real_time_evolution(cos_state, sin_state,
                                         cos_phases, sin_phases)

Phase-angle **generation** is external to the library. The primitives
consume whatever `PhaseSequence` they are handed, so any source of angles
works — you can supply your own. `QSPPACK <https://github.com/qsppack/QSPPACK>`_
is *one* optional, example-only way to compute angles for a target function;
it is **not** a runtime dependency of `cudaq_algorithms`. The worked example
`docs/sphinx/examples/python/hamiltonian_simulation_qsvt.py` uses QSPPACK
(and scipy) to generate the cosine/sine phases, runs QSVT time evolution,
and compares against exact diagonalization, reaching ~1e-15 state error.

State preparation injection
---------------------------

Every kernel factory (`Walk`, `QSVT`, and `Trotter`) takes an optional
``state_prep`` kernel with signature ``(qubits: cudaq.qview)``. Without it,
factories return `@cudaq.kernel(state)` circuits that load the input state
as data (the simulation-friendly form). With it, they return
**zero-argument** circuits — the system register is allocated in ``|0...0>``,
``state_prep`` runs on it, and the operation follows — directly sampleable
and fully synthesizable, with no statevector anywhere:

.. code-block:: python

   @cudaq.kernel
   def my_prep(qubits: cudaq.qview):
       rx(0.37, qubits[0])
       ry(-0.52, qubits[1])

   kernel = walk.kernel(power=3, state_prep=my_prep)   # zero-arg circuit
   counts = cudaq.sample(kernel)
   moment = walk.moment(None, 3, state_prep=my_prep)   # hardware-path moment

Contract: ``state_prep`` acts only on the system register it is handed,
which arrives in ``|0...0>`` with width ``num_system`` — not verifiable at
factory time, so a mismatched prep fails at launch.

Hardware-shaped vs. simulation-only
-----------------------------------

`cudaq.get_state` is a simulator-only API, so nothing in `pauli_lcu`,
`qubitization`, or `qsvt` calls it. Statevector conveniences are isolated in
`cudaq_algorithms.sim_utils`:

.. code-block:: python

   from cudaq_algorithms import sim_utils as sim

   good = sim.good_subspace(enc, state)  # postselect the |0..0>-ancilla block
   hpsi = sim.action(enc, psi)           # (H/alpha)|psi>
   out  = sim.transform(transformer, psi, seq)
   psi0 = sim.state_from(ket)            # precision-aware cudaq.State

`Walk.moment` / `moments` stay in the library proper: they measure through
`cudaq.observe`, which is a hardware-legitimate path.

Escape hatch at every level: the module-level kernels compose inside user
kernels, with ``enc.kernel_args`` supplying the flattened arrays they take
as arguments. They live in their module namespaces — their names are too
generic for the package root — as
`cudaq_algorithms.pauli_lcu.{prepare, select, apply, walk, adjoint_walk, apply_phase_sequence, ...}`
and `cudaq_algorithms.common_kernels.{reflect_about_zero, signal_phase, ...}`.

Simulator selection
-------------------

Tests and examples default to ``qpp-cpu`` and honor CUDA-Q's standard
``CUDAQ_DEFAULT_SIMULATOR`` variable (e.g. ``nvidia-fp64`` for the GPU
statevector simulator). Use an fp64 target for the test suite — the
``nvidia`` target is fp32 and misses the 1e-8..1e-10 tolerances (precision,
not correctness). State construction goes through ``state_from``, which
matches the input dtype to the active target's precision (`cudaq.complex()`),
since fp32 simulators reject complex128 initial-state data.
