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
steps (no UNPREPARE), whose expectation is ``<T_{2p+1}(H/alpha)>``. This
presumes a walk step defined in the PREPARE frame (``W = PREPARE R
PREPARE-dagger SELECT``), as both `PauliLCU` and `LCUBlockEncoding` do. For
an LCU this is ``sum_i sign_i |i><i|_anc x P_i`` (`select_observable(enc)`
serves both `PauliLCU` backends; `LCUBlockEncoding` derives it when every
SELECT body is a Pauli word); encodings that cannot supply it
(for example `DoubleFactorizedEncoding`, or `LCUBlockEncoding` with a body
in the extended vocabulary) raise `NotImplementedError` on
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
``hamiltonian_simulation_qsvt.py`` on the :doc:`Hamiltonian-simulation
examples page <../examples_rst/hamiltonian_simulation>` uses QSPPACK
(and scipy) to generate the cosine/sine phases, runs QSVT time evolution,
and compares against exact diagonalization, reaching ~1e-15 state error.
The getting-started series applies the same machinery to a different
function: ``07_matrix_inversion_qsvt.py`` (:doc:`../examples_rst/getting_started`)
inverts a linear system through a polynomial approximation of
:math:`1/x` — the QSVT reading of HHL — changing nothing but the phase
sequence.

Quantum phase estimation
------------------------

`PhaseEstimation` estimates eigenvalues of ``H`` from any `BlockEncoding` by
phase-estimating its qubitization walk: ``W`` has eigenvalues
``e^{±iθ}`` with ``cos θ = -E/alpha`` on each eigenspace of ``H``, so an
``m``-bit readout ``j`` decodes as ``E = -alpha cos(2π j / 2^m)``. No time
evolution and no phase-angle synthesis are needed — the controlled walk,
the inverse QFT and state preparation are already in the library.

.. code-block:: python

   from cudaq_algorithms import PauliLCU, PhaseEstimation, sim_utils

   qpe = PhaseEstimation(PauliLCU(hamiltonian))
   circuit = qpe.kernel(6, state_prep=prep)                # measurement-free
   measured = qpe.kernel(6, state_prep=prep, measure=True) # measures the phase register
   counts = cudaq.sample(measured, shots_count=2000)       # any target
   ranked = qpe.energies_from_counts(counts, 6, top=3)     # [(energy, probability), ...]
   exact = sim_utils.qpe.energies(qpe, 6, state_prep=prep, top=3)  # statevector, simulators

The class builds circuits and decodes readouts; running a circuit is your
executor on your target. ``measure=True`` measures the phase register
alone, so the sampled bitstrings are that register (qubit 0 first);
`PhaseEstimation.energies_from_counts` also accepts counts over every
qubit, as sampling the measurement-free circuit produces, and plain
mappings. The statevector shortcut, `sim_utils.qpe.phase_probabilities`
and `sim_utils.qpe.energies`, gives the exact marginal on a simulator.

The circuit is the textbook ladder — ``2^m - 1`` controlled walk steps and
an inverse QFT — so ``num_bits`` is the cost knob; the register layout is
system, then the ``m`` phase qubits (little-endian: phase qubit 0 is the
least significant readout bit), then the encoding's ancillas. Each
eigenvalue appears at the mirror readouts ``j`` and ``2^m - j``, and
`energies` folds the pair into one entry, so a pure eigenstate reads as a
single dominant energy. The energy resolution is about
``alpha π / 2^m``; a general input spreads its mass over the spectrum in
proportion to its overlaps.

Two simulation notes for long ladders. The compiler re-materializes the
walk step's captured tables on the host stack on every loop iteration
(about 1 kB per step for a `PauliLCU`), so from ``num_bits = 13`` the
circuit outgrows the default 8 MiB thread stack and the process dies with
a segmentation fault; the `sim_utils.qpe` helpers run the simulation on a
worker thread with a large stack, and a circuit you sample on a simulator
needs the same
(``sim_utils.run_on_large_stack(cudaq.sample, measured, shots_count=...)``).
On the ``nvidia`` target, gate fusion's host-side cost grows with the
circuit length; with fusion off (``CUDAQ_FUSION_MAX_QUBITS=0``) a 13-bit
H2 ladder runs in about half a minute on one GPU. The worked example
``08_quantum_phase_estimation.py`` (:doc:`../examples_rst/getting_started`)
puts a hand-written QPE shell around three oracles — exact time evolution,
a product formula and this walk — to compare their decoders; `PhaseEstimation`
packages the walk case.

State preparation injection
---------------------------

Every kernel factory (`Walk`, `QSVT`, and `Trotter`) takes an optional
``state_prep`` kernel with signature ``(qubits: cudaq.qview)``. Without it,
factories return `@cudaq.kernel(state)` circuits that load the input state
as data (the simulation-friendly form). With it, they return
**zero-argument** circuits — the system register is allocated in ``|0...0>``,
``state_prep`` runs on it, and the operation follows — directly sampleable
and fully synthesizable, with the statevector abstracted away from the caller:

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
