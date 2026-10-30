Fermionic basis changes
=======================

``fermion_basis_change`` constructs a number-preserving transformation of
fermionic modes and applies it to an existing quantum register. A square
unitary specifies the full transformation; a normalized vector specifies
one column, including its phase. It lives in
``cudaq_algorithms.basis_change``.

Conventions
-----------

The input is a matrix over *spin-orbital* modes, one mode per qubit: ``m``
modes on ``m`` qubits placed by ``mode_map``. It does not lift a spatial
matrix to both spin sectors. For a real orthogonal *spatial* rotation
applied to both spins of the interleaved layout, use the orbital-rotation
network in :doc:`state_prep` (``stateprep.make_orbital_rotation_schedule``
and ``stateprep.spatial_givens_network``), or pass ``np.kron(V,
np.eye(2))`` here. The network realizes ``V D`` with the schedule's sign
diagonal ``D``; ``np.kron(V @ np.diag(schedule.signs), np.eye(2))``
reproduces it exactly.

Each mixing gate of a schedule is the one-particle rotation
``[[cos θ, -sin θ], [sin θ, cos θ]]`` on logical modes ``(first,
second)``, that is ``exp(θ (a†_second a_first - a†_first a_second))``. It
is the same rotation as ``stateprep.givens_rotation(qubits, θ, first,
second)``. On spatial modes ``(mode, mode + 1)`` of the interleaved layout
it equals ``stateprep.spatial_givens_rotation(system, -θ, mode)``, which
rotates both spin sectors at once.

For an ``m`` by ``m`` unitary ``U``, the circuit implements the
vacuum-preserving second quantization

.. math::

   \Gamma(U)a_j^\dagger\Gamma(U)^\dagger
      = \sum_p U_{pj}a_p^\dagger,
   \qquad \Gamma(U)|\mathrm{vac}\rangle=|\mathrm{vac}\rangle.

``a_j`` acts on physical qubit ``mode_map[j]`` (qubit ``j`` by default),
with its Jordan-Wigner parity string over every lower-numbered physical
qubit, mapped or spectator. On equally sized occupied sets ``S`` and
``T``, each listed in increasing physical-qubit order, the Fock-space
matrix element is ``det(U[S, T])``; a spectator qubit acts as a mode that
``U`` leaves unchanged.

Fixed transformations
---------------------

.. code-block:: python

   import numpy as np
   import cudaq
   from cudaq_algorithms import fermion_basis_change

   U = np.array([[1., -1.], [1j, 1j]]) / np.sqrt(2.)
   plan = fermion_basis_change(U)
   apply_basis = plan.kernel()
   undo_basis = plan.adjoint_kernel()

   @cudaq.kernel
   def fixed_change(enable: cudaq.qubit, system: cudaq.qview):
       cudaq.control(apply_basis, enable, system)

The factory accepts shapes ``(m, m)``, ``(m,)``, and ``(m, 1)`` with
``m >= 2``; for a single mode apply ``r1`` directly. The input is a
one-particle matrix or vector, not a qubit unitary or a many-body state.
Input that is nonunitary or unnormalized beyond ``tolerance`` (default
``1e-12``) is rejected; ``schedule.reconstruction_error`` reports the
residual of an accepted inexact input.

A vector ``v`` gives a deterministic unitary completion ``W_v`` with
reference column ``v``. The caller first prepares the reference
occupation:

.. code-block:: python

   vector = np.array([1., 1j]) / np.sqrt(2.)
   partial = fermion_basis_change(vector)
   prepare_column = partial.kernel()

   @cudaq.kernel
   def prepare_vector():
       system = cudaq.qvector(2)
       x(system[0])
       prepare_column(system)

``GivensStructure.partial(m, reference_mode=r)`` chooses another reference
mode, and ``GivensStructure.full(m)`` a full transformation. Either
accepts a ``mode_map`` of distinct physical qubits in logical mode order;
a mapped partial plan expects its reference occupation on qubit
``mode_map[reference_mode]``.

``controlled_kernel()`` takes one register with the enable at index zero
and the base layout shifted by one (``plan.num_qubits + 1`` qubits). The
Clifford scaffold around each rotation cancels when the enable is off, so
only the Rz and R1 gates are controlled. ``cudaq.control`` of ``kernel()``,
as above, gives the same operator but controls every gate.

.. warning::

   A view narrower than ``num_qubits`` (``num_qubits + 1`` for
   ``controlled_kernel``) is not rejected. CUDA-Q does not bounds-check
   views, so the kernel silently addresses the parent-register qubits that
   follow the view, such as a QROM address or scratch qubit.

Programmable plans
------------------

A fixed plan with an external control covers one transformation. To
select among transformations with a quantum address, use a programmable
plan: one circuit whose angles are read from quantum words, for example
loaded by QROM.

.. code-block:: python

   from cudaq_algorithms.basis_change import GivensStructure

   program = fermion_basis_change(U, programmable=True, angle_bits=8)
   consume_words = program.kernel()
   codes = program.encoded_words

   structure = GivensStructure.partial(3, mode_map=(0, 2, 4))
   consumer = fermion_basis_change(structure=structure,
                                  programmable=True, angle_bits=8)

Each word is an unsigned little-endian full-turn fixed-point angle of
``b = angle_bits`` bits, from 1 through 52:

.. math::

   \Delta = 2\pi / 2^b, \qquad
   x = \operatorname{round}_{\mathrm{nearest\ even}}(\theta/\Delta)
        \bmod 2^b, \qquad \theta(x)=x\Delta.

Full schedules have ``m*m`` slots and partial schedules ``2*m-1``; every
slot is read from its word, including slots whose host value is zero.
The kernel takes one combined register. The system span is
``max(mode_map) + 1``, and by default the words follow it contiguously in
``schedule.parameter_names`` order. Explicit ``parameter_registers`` give
one list of ``angle_bits`` distinct qubit indices per slot,
least-significant bit first, outside the system span; they need not be
contiguous. ``num_qubits`` is the combined span; QROM addresses, scratch
and controls are additional.

Accuracy
--------

Three errors are kept separate: the input residual
(``schedule.reconstruction_error``), angle quantization
(``quantization_error_bound``), and the synthesis budget you supply.
``estimate_fermion_basis_change_error(schedule, represented_parameters,
*, synthesis_error_bound=0.)`` adds the parameter deltas and the
synthesis budget into a basis-change bound clipped at two, and gives a
reflection bound of twice that, also clipped at two. When each Rz and R1
is synthesized to spectral error ``eps``, including global phase, pass
``num_elementary_rotations * eps`` from
``estimate_fermion_basis_change_resources``
(``num_bit_controlled_rotations * eps`` for a programmable plan).

Nearest rounding gives the conservative a priori quantization bound

.. math::

   \epsilon_{\mathrm{quant}} \le \min(2, C\pi/2^b),\qquad
   C = \begin{cases}
     m+3m(m-1)/2 & \text{full},\\
     2m-1 & \text{partial}.
   \end{cases}

Two is the largest distance between unitaries, so this bound says nothing
once ``C*pi/2**b >= 2``: every full plan with ``m >= 3`` at
``angle_bits=4``, for example. A plan's ``quantization_error_bound`` sums
its actual rounding deltas instead, so it is never larger and usually
smaller: for random 3-mode unitaries at four bits it is typically 0.8 to
1.6, while the actual Fock-space error is typically 0.2 to 0.6.

Selecting with QROM
-------------------

The plan owns no QROM object. The caller packs each plan's
``encoded_words`` into table rows, constructs ``QROM``, and composes
lookup, consumer, and inverse lookup; the inverse lookup restores the
words and scratch to zero. Rows sharing a consumer must share its
structure and encoding. With the default layout, a row is
``sum(code << (angle_bits * s) for s, code in enumerate(codes))``,
written to the output view ``q[base:consumer.num_qubits]`` with
``base = consumer.structure.num_system_qubits`` and ``output_bits`` equal
to ``consumer.num_qubits - base``; for ``controlled_kernel`` shift the view
by one. The runnable example on the
:doc:`chemistry-primitive examples page
<../examples_rst/chemistry_primitives>` does this for a fixed, a
controlled and a selected transformation.

For explicit ``parameter_registers``, choose ``base`` from the system span
through the lowest word qubit and pack each row against the consumer's
layout:

.. code-block:: python

   def pack_row(codes, consumer, base):
       return sum(((code >> i) & 1) << (index - base)
                  for code, word in zip(codes, consumer.parameter_registers)
                  for i, index in enumerate(word))

The selected occupation reflection :math:`Q_j = W_j Z_r W_j^\dagger` is
lookup, ``plan.adjoint_kernel()``, ``Z`` on qubit ``mode_map[r]``,
``plan.kernel()``, and inverse lookup, where ``r`` is the reference mode of
a partial plan or a chosen mode of a full plan. Padded addresses with an
all-zero row still apply ``Z``; control it on address validity if they
must act as identity.

Resources
---------

``estimate_fermion_basis_change_resources(plan)`` counts the static tape
of a plan, including zero-valued slots; it also accepts a schedule with
``angle_bits`` and ``num_program_qubits``, where an explicit layout needs
``num_program_qubits=plan.num_qubits - plan.structure.num_system_qubits``.
Each Givens uses two Pauli rotations; at physical endpoint separation
``d`` its Jordan-Wigner scaffold uses ``4*d`` CNOTs, eight Hadamards, and
four S/S-dagger gates. Each number phase is one rotation. Programmable
plans use bit-controlled Rz and R1 gates. The counts exclude QROM and
external controls.

``controlled_kernel`` adds the enable to each rotation only: each Rz and
R1 becomes controlled, and a programmable plan's bit-controlled rotations
become doubly controlled. ``cudaq.control`` of ``kernel()`` also controls
every Hadamard and S gate and turns each parity CNOT into a Toffoli.
Compare QROM selection against the controlled counts of the path you use.
