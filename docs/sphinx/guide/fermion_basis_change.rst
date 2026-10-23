Fermionic basis changes
=======================

``fermion_basis_change`` constructs a number-preserving transformation of
fermionic modes and applies it to an existing quantum register. A square
unitary specifies the full transformation; a normalized vector specifies
one column, including its phase. The implementation uses released CUDA-Q
gates and lives in ``cudaq_algorithms.basis_change``. Existing
:doc:`state-preparation APIs <state_prep>` keep their own contracts.

Choosing how to supply angles
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 36 38 26

   * - Intended use
     - Parameters
     - QROM
   * - One fixed full transformation
     - Classical angles
     - Unnecessary
   * - Enable a fixed transformation coherently
     - Classical angles and an external control
     - Unnecessary
   * - Prepare one fixed one-particle vector
     - Classical angles and a reference occupation
     - Unnecessary
   * - Select transformations or vectors with a quantum address
     - Address-controlled fixed circuits, or a common circuit with quantum words
     - Optional source of words
   * - Reflect about an address-selected occupied mode
     - Selected inverse basis change, reference Z, then basis change
       (:math:`Q_j = W_j Z_r W_j^\dagger`)
     - Optional source of words
   * - Compute parameters reversibly
     - Words supplied by arithmetic
     - Unnecessary for the angle source

An entangled control can enable a fixed circuit without turning its angles
into quantum data. Coherent selection among different transformations is a
separate use case. QROM is useful when it supplies the common circuit's
words from a classical table; compare its lookup, unlookup, and workspace
cost with directly controlling fixed circuits for each address. Under
``cudaq.control`` or ``controlled_kernel``, control reaches every emitted
gate, not only the rotations, so that comparison must use the controlled
counts in `Resources and scope`_.

Full and partial transformations
--------------------------------

For an ``m`` by ``m`` unitary ``U``, the circuit implements the
vacuum-preserving second quantization

.. math::

   \Gamma(U)a_j^\dagger\Gamma(U)^\dagger
      = \sum_p U_{pj}a_p^\dagger,
   \qquad \Gamma(U)|\mathrm{vac}\rangle=|\mathrm{vac}\rangle.

There is one occupation qubit per mode: ``a_j`` acts on physical qubit
``mode_map[j]`` (qubit ``j`` by default), with its Jordan-Wigner parity
string over every lower-numbered physical qubit, mapped or spectator. On
equally sized occupied sets ``S`` and ``T``, each listed in increasing
physical-qubit order, the Fock-space matrix element is ``det(U[S, T])``;
a spectator qubit acts as a mode that ``U`` leaves unchanged. This retains
diagonal signs, determinant phases, and the phase relative to the vacuum,
including under external control.

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
``m >= 2``. A single mode has no mixing and admits only a number phase;
apply ``r1`` directly instead. For example, ``U = [[exp(1j * alpha)]]``
is ``r1(alpha, q[mode_map[0]])``, or
``r1.ctrl(alpha, enable, q[mode_map[0]])`` under an enable. The input is
a one-particle (orbital) matrix or vector, not a qubit operator or a
many-body state: a ``2**n`` by ``2**n`` qubit unitary is interpreted as
``2**n`` modes on ``2**n`` qubits, and a length-``2**n`` vector as one
orbital of ``2**n`` modes. This is not arbitrary qubit-unitary synthesis
or many-body state preparation.

Nonfinite data is rejected, as is nonunitary or unnormalized data outside
``tolerance`` (default ``1e-12``). The tolerance must be a real number with
``0 < tolerance < 1``; booleans, strings, bytes, and complex values are
rejected. An accepted inexact input is compiled from its elimination
angles, so the plan implements the unitary those angles define.
``schedule.reconstruction_error`` reports the maximum entry discrepancy
from the input; check it when loosening ``tolerance``.

A vector ``v`` gives a deterministic unitary completion ``W_v`` with
reference column ``v``. The caller first prepares the reference occupation;
the basis-change kernel itself does not initialize occupations.

.. code-block:: python

   vector = np.array([1., 1j]) / np.sqrt(2.)
   partial = fermion_basis_change(vector)
   prepare_column = partial.kernel()

   @cudaq.kernel
   def prepare_vector():
       system = cudaq.qvector(2)
       x(system[0])
       prepare_column(system)

``GivensStructure.partial(m, reference_mode=r)`` chooses another logical
reference mode. A full structure uses ``GivensStructure.full(m)``. Either
accepts a ``mode_map`` listing distinct physical qubits, from 0 through
``2**31 - 1``, in logical mode order. It must be an ordered sequence
(list, tuple, range or one-dimensional array); sets, mappings, iterators
and generators are rejected. A mapped partial plan expects its reference
occupation on physical qubit ``mode_map[reference_mode]``, not on qubit
``reference_mode``. Mixing between nonadjacent physical qubits includes
the Jordan-Wigner parity of every intervening physical qubit, whether it
holds another mapped mode or a spectator. A spatial matrix is not
automatically copied into both spin sectors.

Quantum words and register layout
---------------------------------

.. code-block:: python

   from cudaq_algorithms.basis_change import GivensStructure

   program = fermion_basis_change(U, programmable=True, angle_bits=8)
   consume_words = program.kernel()
   codes = program.encoded_words
   words = program.parameter_registers
   register_span = program.num_qubits

   structure = GivensStructure.partial(3, mode_map=(0, 2, 4))
   consumer = fermion_basis_change(structure=structure,
                                  programmable=True, angle_bits=8)

Programmable mode requires an integer width from 1 through 52; booleans
are rejected. ``programmable``, like the ``adjoint`` flag of
``controlled_kernel``, must be a Python ``bool``; ``numpy.bool_`` is
rejected. ``basis`` may also be a compiled ``GivensBasisChangeSchedule``.
Structure-only plans require programmable mode and an explicit width.
Classical mode rejects ``angle_bits`` and ``parameter_registers``.

The kernel accepts a single combined ``cudaq.qview``. Its system span is
``max(mode_map) + 1``, including spectators and gaps. Default words are
contiguous slices packed immediately after that span, in
``schedule.parameter_names`` order; they too must end at or below qubit
``2**31 - 1``. Explicit ``parameter_registers`` must be an ordered
sequence (list, tuple, range or array), not a set, mapping or iterator,
with one contiguous increasing sequence of ``angle_bits`` qubit indices
per slot, each from 0 through ``2**31 - 1``; words cannot overlap each
other or any part of the system span. ``num_qubits`` is the required
combined span, including any placement gaps. QROM addresses, lookup
scratch, and external controls are additional. Separately allocated
registers can be combined at the call site, as in
``consume_words([*system, *words])``, when the combined order matches
``structure.mode_map`` and ``parameter_registers``.

The numerical matrix supplies reference angles and table entries; loaded
quantum words determine every executed parameter, including slots whose
host value is zero. Full schedules retain ``m*m`` slots: ``m`` diagonal
phases followed by a mixing angle and relative phase for each mixing
block. Partial schedules retain ``2*m-1`` slots: mixing angles followed by
output phases. Rows sharing a consumer must share its structure, parameter
order, and encoding, and every row is packed with the consumer's
``parameter_registers``, not those of the plan that produced its codes.
The planner's phases are functions of the matrix values, so value-equal
inputs give the same words; real orthogonal inputs give phases that are
multiples of ``pi``.

``kernel()`` and ``adjoint_kernel()`` use the same base register layout,
and ``cudaq.adjoint`` of ``kernel()`` matches ``adjoint_kernel()`` for
classical and programmable plans. Ordinary ``cudaq.control`` can add an
external control. Alternatively, ``controlled_kernel(*, adjoint=False)``
accepts one combined register with enable at index zero and every base
index shifted by one; allocate ``plan.num_qubits + 1`` qubits for this
wrapper. The disabled branch is identity. Serialized plans contain host
data, omit cached runtime kernels, and are validated again when loaded.

.. warning::

   A view narrower than ``num_qubits`` (``num_qubits + 1`` for
   ``controlled_kernel``) is not rejected. CUDA-Q does not bounds-check
   views, so the kernel silently addresses the parent-register qubits that
   follow the view, such as a QROM address or scratch qubit.

Encoding and diagnostics
------------------------

``AngleEncoding(width)`` is immutable. It uses unsigned little-endian
full-turn fixed point, no offset, and nearest-even rounding:

.. math::

   \Delta = 2\pi / 2^b, \qquad
   x = \operatorname{round}_{\mathrm{nearest\ even}}(\theta/\Delta)
        \bmod 2^b, \qquad \theta(x)=x\Delta.

``encode`` accepts finite real angles, including negative and wrapped
values, and returns a Python integer. ``decode`` accepts an integer in
``[0, 2**width)`` and returns radians. ``step``, ``weights``, ``width``, and
``bit_order == "little"`` expose the convention; bit ``k`` has weight
``2*pi * 2**(k-width)``. The width limit matches double-valued host angles.
This full-turn rule applies to complete Givens gates and exact number
phases; a standalone Pauli rotation has exact period ``4*pi``.

Host angles and diagnostics use double precision. Outside one signed full
turn, encoding uses trigonometric argument reduction to preserve the phase
of large finite angles; directly taking a remainder by floating-point
``2*pi`` would lose that phase. ``encode`` converts its input to a double
and rounds in double precision, so from about 43 bits, and for some
floating-point near-ties at smaller widths, a code can differ from the
exact nearest code by one step, less than about ``1e-15`` radians beyond
half a step. Exact half-step ties of the computed quotient within one
signed turn, which includes every planner angle, still resolve to the
even code: ``3*math.pi/8`` encodes to 2 at width 3, while the wrapped
``3*math.pi/8 + 2*math.pi`` is perturbed by argument reduction and
encodes to 1. The grid error bounds below describe angle quantization
only. They exclude floating-point error of less than about ``1e-15``
radians per tape occurrence, so a bound summing ``C`` occurrences is not
certified below about ``C * 1e-15`` radians.

.. list-table:: Plan metadata
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Meaning
   * - ``schedule.parameters`` / ``schedule.parameter_names``
     - Ordered host angles in radians / slot names
   * - ``schedule.operations``
     - Chronological ``GivensOperation`` records, exported from
       ``cudaq_algorithms.basis_change``, with logical ``first`` and
       ``second`` modes; ``scale`` is ``-1`` for the leading phase of a
       complex mixing block
   * - ``schedule.reconstruction_error``
     - Maximum orbital-matrix entry discrepancy, or reference-column
       discrepancy: a finite nonnegative real number, validated on
       construction and ``None`` for structure-only schedules (a value
       without parameters is rejected); excluded from schedule and plan
       equality and hashing
   * - ``angle_encoding`` / ``angle_bits`` / ``is_programmable``
     - Encoding metadata and execution mode
   * - ``encoded_words`` / ``represented_parameters``
     - Host codes and their decoded radian values
   * - ``quantization_error`` / ``quantization_error_bound``
     - Circular parameter report and conservative Fock-space operator-norm bound

Structure-only plans have no numerical parameters, codes, reconstruction
residual, or quantization diagnostics. Classical plans have no encoding or
quantization diagnostics. Input tolerance and reconstruction residual are
separate from quantization and synthesis error. Schedule parameters,
``orbital_matrix`` overrides, and represented parameters must be
one-dimensional real vectors with one finite entry per slot: integer or
floating-point arrays, or sequences (including object arrays) of real
scalars such as ``Fraction``. Boolean, string, bytes, and complex entries
are rejected, even inside an otherwise numeric list.

``estimate_fermion_basis_change_error(schedule, represented_parameters,
*, synthesis_error_bound=0.)`` uses shortest signed circular deltas modulo
``2*pi``. Each tape occurrence contributes its absolute parameter delta;
a full schedule's relative phase occurs twice. The supplied synthesis
budget applies to the complete circuit and remains a separate report
field. The total basis-change bound is clipped at two. Conjugating a
norm-one reflection has at most twice that error, also clipped at two.
For coherent selection, take the maximum error over table branches.

Nearest rounding gives the conservative quantization bound

.. math::

   \epsilon_{\mathrm{quant}} \le \min(2, C\pi/2^b),\qquad
   C = \begin{cases}
     m+3m(m-1)/2 & \text{full},\\
     2m-1 & \text{partial}.
   \end{cases}

It compares the represented circuit with its unquantized schedule,
including the partial completion. It excludes host reconstruction and
synthesis error. Thus ``angle_bits=16`` alone does not guarantee a chosen
total accuracy.

Explicit QROM composition and reflections
-----------------------------------------

The plan owns no QROM object. The caller packs each plan's ``encoded_words``
into table rows, constructs ``QROM``, allocates output and clean scratch,
and composes lookup, consumer, and inverse lookup. Lookup preserves the
address; the consumer preserves each word's computational-basis value.
Inverse lookup restores initially zero words and scratch while leaving the
intended address/system entanglement.

QROM writes row bit ``k`` to qubit ``k`` of the view passed as its output.
``parameter_registers`` indices count from the start of the register ``q``
passed to the consumer's ``kernel()``, not from a parent allocation. If the
output view starts at index ``base`` of ``q``, bit ``i`` of slot ``s`` goes
to output position ``parameter_registers[s][i] - base``, and positions in
layout gaps stay zero. This packs one row's codes, such as another plan's
``encoded_words``, for any consumer layout:

.. code-block:: python

   def pack_row(codes, consumer, base):
       return sum(((code >> i) & 1) << (index - base)
                  for code, word in zip(codes, consumer.parameter_registers)
                  for i, index in enumerate(word))

Pass ``q[base:consumer.num_qubits]`` as the output, with ``output_bits``
equal to ``consumer.num_qubits - base``, for a ``base`` from the system
span through the lowest word qubit. The register of ``controlled_kernel``
starts with the enable, so there the output view is
``q[base + 1:consumer.num_qubits + 1]``. The default layout with
``base = consumer.structure.num_system_qubits`` places slot ``s`` at output
positions ``angle_bits * s`` onward, so a row reduces to
``sum(code << (angle_bits * s) for s, code in enumerate(codes))``. The
example below uses that special case.

For the selected occupation reflection :math:`Q_j = W_j Z_r W_j^\dagger`,
the chronological sequence is lookup, ``plan.adjoint_kernel()``
(:math:`W_j^\dagger` for the loaded words), ``Z_r`` on physical qubit
``mode_map[r]``, ``plan.kernel()`` (:math:`W_j`), and inverse lookup. Here
``r`` is the ``reference_mode`` of a partial plan, or the chosen mode of a
full plan (the example below uses ``r = 0``). The reflection acts on the
existing many-body state. Invalid padded addresses with an all-zero angle
row still apply ``Z_r``; identity behavior on those addresses requires a
validity control on the central Z. Different selected vectors need not be
orthogonal because their address labels distinguish the branches.

The runnable example below uses the public factory and explicit QROM
packing:

.. literalinclude:: ../examples/python/fermion_basis_change.py
   :language: python
   :start-after: [Begin Documentation]

Resources and scope
-------------------

``estimate_fermion_basis_change_resources(schedule, *, angle_bits=None,
num_program_qubits=None)`` counts the complete static tape, including
zero-valued slots. Each Givens uses two Pauli rotations. At physical
endpoint separation ``d``, its paired Jordan-Wigner scaffold uses ``4*d``
CNOTs, eight Hadamards, and four S/S-dagger gates. Each number phase uses
one elementary rotation. Programmable execution uses ordinary
bit-controlled Rz gates and exact bit-controlled R1 gates; no companion
core feature is required.

With ``angle_bits=b``, ``num_bit_controlled_rotations`` is ``b`` times
``num_elementary_rotations`` and the default program-qubit count is
``b * num_parameters``. Without a width both defaults are zero. An explicit
``num_program_qubits`` overrides only the reported program-qubit count;
with a width it must be at least ``angle_bits`` (one word). The plan's
kernels always need every word resident, and streaming remains deferred.
These are static counts for the uncontrolled kernels, excluding QROM,
external controls, hardware scheduling, and fault-tolerant synthesis.

``cudaq.control`` and ``controlled_kernel`` put the enable on every emitted
gate: each Hadamard becomes a controlled H, each S or S-dagger a controlled
S or S-dagger, each parity CNOT a Toffoli, each Rz a controlled Rz, and
each R1 a controlled R1. A programmable plan's bit-controlled Rz and R1
rotations become doubly controlled. A comparison between QROM and fixed
circuits controlled this way must use these controlled counts; a
hand-built circuit that controls only the rotations needs its own count.
Arbitrary qubit-unitary synthesis and many-body state preparation are out
of scope; streaming words, alternative encodings, automatic total-error
budgeting, and specialized synthesis are deferred.
