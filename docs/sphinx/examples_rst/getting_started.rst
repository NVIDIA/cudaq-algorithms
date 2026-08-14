Getting-started walkthrough
===============================

An ordered learning path through the library. Each example builds on the one
before it, from the single core idea -- block-encode a Hamiltonian and walk it
-- up to bringing your own encoding. Every example verifies its own claims
against an independent dense reference, so each script runs to completion or
fails loudly.

.. seealso::

   The :doc:`user guide <../guide/block_encodings>` pages cover each
   primitive in depth; :doc:`../getting_started/index` sets up the
   installation these scripts assume.

Quickstart: block-encode a Hamiltonian and walk it
---------------------------------------------------

The five-minute tour of the core idea. A Hamiltonian is not a circuit, so you
block-encode it: `PauliLCU` hides `H / alpha` inside a larger unitary on a few
ancilla qubits, and the qubitization `Walk` turns powers of that unitary into
Chebyshev moments checked against a dense matrix.

.. literalinclude:: ../examples/python/01_quickstart_block_encoding.py
   :language: python
   :start-after: [Begin Documentation]

Hamiltonian simulation, two independent ways
--------------------------------------------

Compute the time evolution ``exp(-iHt)|psi>`` with two different primitives and
check both against dense linear algebra: `QSVT` with Jacobi-Anger phase factors,
and `Trotter` product formulas with an error-vs-steps trade-off. Because the two
constructions are completely different, their agreement is a strong test.

.. literalinclude:: ../examples/python/02_hamiltonian_simulation.py
   :language: python
   :start-after: [Begin Documentation]

From a molecule to its ground-state energy
------------------------------------------

The full classical to quantum to classical loop, end to end: PySCF mean field,
Jordan-Wigner qubit Hamiltonian, `PauliLCU` block encoding, `Walk` moments, and
a classical Krylov solve for the ground-state energy checked against FCI.

.. literalinclude:: ../examples/python/03_chemistry_to_ground_state.py
   :language: python
   :start-after: [Begin Documentation]

Double factorization and the BlockEncoding protocol
---------------------------------------------------

Double factorization rewrites the two-electron tensor as a sum of low-rank
leaves, giving a compression dial you can watch trade accuracy for a cheaper
block encoding. The same pipeline scales to larger molecules and plugs into the
structural `BlockEncoding` protocol shared by every consumer.

.. literalinclude:: ../examples/python/04_double_factorization_and_the_protocol.py
   :language: python
   :start-after: [Begin Documentation]

State preparation and injection
-------------------------------

Every primitive factory (`Walk`, `QSVT`, `Trotter`) accepts a `state_prep`
kernel argument. Pass one and the factory returns a zero-argument, hardware-shaped
circuit with no statevector crossing the API boundary -- the seam that makes the primitives
hardware-ready and the seam a tensor-network state-prep compiler would plug into.

.. literalinclude:: ../examples/python/05_state_prep_and_injection.py
   :language: python
   :start-after: [Begin Documentation]

Bring your own block encoding
-----------------------------

`Walk` and `QSVT` are generic over the `BlockEncoding` protocol: any object
exposing the right members works, with no inheritance. This example implements
the protocol from scratch for a two-term single-qubit LCU and watches the walk
measure correct Chebyshev moments on it.

.. literalinclude:: ../examples/python/06_bring_your_own_encoding.py
   :language: python
   :start-after: [Begin Documentation]

Solve a linear system with QSVT
-------------------------------

The payoff of QSVT's "pick a polynomial" design: example 2 approximated
``exp(-ixt)`` for time evolution; swap the phase sequence for one
approximating ``1/x`` and the *same* `PauliLCU` + `QSVT` machinery becomes a
quantum linear solver -- the QSVT reading of HHL. A 5x5 symmetric
positive-definite system is padded to three qubits, block-encoded, and
inverted with a Childs-Kothari-Somma polynomial (degree set by the condition
number); the good-subspace state is proportional to ``A^-1 b``, with the
scale recovered classically. Verified against ``numpy.linalg.solve``.
Requires ``qsppack`` for the phase factors, like example 2.

.. literalinclude:: ../examples/python/07_matrix_inversion_qsvt.py
   :language: python
   :start-after: [Begin Documentation]

Quantum phase estimation with three energy oracles
--------------------------------------------------

Quantum phase estimation reads the eigenphases of a unitary into a counting
register: for :math:`U|E_j\rangle = e^{2\pi i \phi_j}|E_j\rangle`, an
:math:`m`-qubit register and the controlled powers :math:`U^{2^k}` build a
phase gradient that the inverse QFT converts into a peak near the integer
:math:`y = 2^m \phi_j`. On a superposed input QPE samples eigenvalues with the
input state's spectral weights -- it does *not* choose the ground state.
H\ :sub:`2` works well because Hartree-Fock has dominant ground-state overlap.

This example wraps one QPE shell -- the same Hartree-Fock preparation, the
same six-qubit counting register, the same hand-written inverse QFT -- around
three different constructions of the unitary whose phase reveals energy:

.. list-table::
   :header-rows: 1

   * - Method
     - Unitary measured by QPE
     - Energy decoder
   * - Exact matrix
     - :math:`U_t = e^{-iHt}` (dense, simulator-only reference)
     - :math:`E = -\operatorname{wrap}(2\pi y/M)/t`
   * - Product formula
     - :math:`U_{\mathrm{Trot}}(t) \approx e^{-iHt}` via `trotter.apply_trotter`
     - same linear decoder
   * - Qubitization walk
     - :math:`W` from the `PauliLCU` block encoding
     - :math:`E = -\alpha\cos\theta`

The product-formula oracle shows a subtlety of composing library evolution
under control: `Trotter` evolves the nonidentity part of the Hamiltonian, so
its omitted identity coefficient -- a global phase for free evolution -- must
be restored as a controlled relative phase inside the QPE loop.

The walk oracle decodes through the qubitization convention
:math:`\cos\theta_j = -E_j/\alpha`, so each energy appears at mirror phases
:math:`\pm\theta_j`; the decoder folds conjugate bins before taking the mode.
Its controlled-power schedule is the centered form of Berry et al.,
`PRX Quantum 6, 020327 (2025) <https://doi.org/10.1103/PRXQuantum.6.020327>`_,
which replaces :math:`W^n` by :math:`W^{n - M/2}`: one unconditional
:math:`W^\dagger`, a controlled :math:`W` for bit zero, and a coherent choice
of :math:`W^{\pm 2^{k-1}}` for every upper bit.

A four-qubit problem does not need qubitization -- the dense reference is
cheaper classically. The walk is here because fault-tolerant algorithms
commonly assume structured PREPARE/SELECT access, and this shell is exactly
how such an oracle drops into QPE. Six counting qubits keep the script fast
on the CPU simulator; every modal estimate is asserted to land within one
phase cell (about 34 mHa here) of the exact ground energy. Resolution scales
with the register: at 12 counting qubits the same circuits reach roughly
0.1--0.5 mHa.

.. literalinclude:: ../examples/python/08_quantum_phase_estimation.py
   :language: python
   :start-after: [Begin Documentation]
