Block-encoding examples
===========================

Hide a Hamiltonian inside a larger unitary using a Pauli sum, sparse matrix
access, or a structured double-factorized representation. These examples
exercise the shared consumers and compare the resulting encodings.

.. seealso::

   :doc:`../guide/block_encodings` — the ``PauliLCU`` API and the
   ``BlockEncoding`` protocol these examples exercise.

Walkthrough of the PauliLCU block-encoding API
----------------------------------------------

A Hamiltonian is a weighted sum of simple operations. To use it in quantum
algorithms, we need a larger circuit that contains a scaled copy of that sum
in the part where its extra qubits are zero. This is a block encoding. This
example follows one small Hamiltonian from its coefficients to an executable
block encoding, then uses that encoding in a quantum walk and a polynomial
transformation.

The comparison asks what changes when the same Hamiltonian uses exact state
preparation or finite-precision alias sampling. You should learn how
preparation weights, the selected operations, and the normalization determine
the encoded matrix; how extra qubits contribute to cost; and how to compare
accuracy across implementations. The NumPy checks separate errors caused by
rounded coefficients from errors in executing the circuit, so the printed
resource savings or costs can be interpreted alongside the approximation.

A guided tour of the `PauliLCU` block-encoding API: build the encoding from a
spin operator, inspect its sizes and normalization, and exercise the device
kernels (`pauli_lcu.prepare`, the select, and the walk) it exposes.

The final sections compare exact and alias PREPARE/SELECT at ``mu=2``,
``3`` and ``4`` using the same Hamiltonian and kernel-factory consumers.
The output summarizes Hamiltonian approximation error, executed algorithms,
and resource cost. On this fixture ``mu=3`` has the same rounded weights as
``mu=2``, while ``mu=4`` reduces the error by about 2.6 times. The resource
table accounts for alias workspace and lists every gate family. A separate
small fixture prints odd/even measured moments. Independent NumPy checks
remain active and are quiet on success; pass ``--verbose`` for the detailed
diagnostics. See :doc:`../guide/block_encodings` for the accuracy/resource
tradeoff and the distinction between rounding and circuit error.

.. literalinclude:: ../examples/python/pauli_lcu_demo.py
   :language: python
   :start-after: [Begin Documentation]

Sparse matrix data and arithmetic oracles
------------------------------------------

A sparse matrix has only a few nonzero entries in each row. This example asks
how to give a quantum algorithm access to those entries without first
expressing the entire matrix as a sum of Pauli operations. It constructs the
same small matrix from stored matrix data and from arithmetic rules describing
its bands, showing two ways to supply the locations and values needed for a
sparse block encoding.

You should learn what the matrix-access oracles must return, how a banded
pattern handles boundaries, and why the two access methods can have different
normalization and workspace costs. Both encodings feed the same Walk and QSVT
interfaces. The checks reconstruct the matrix represented by finite-precision
value angles and compare the circuit with that matrix, so differences in data
access and numerical precision remain visible when assessing the results.

The default output displays the requested and rounded matrices, matrices
reconstructed from both circuits, and their results on the same complex
input vector. A four-point endpoint input makes clipped band boundaries
visible. Circuit errors and a compact resource comparison finish the run.
Use ``--verbose`` for individual validation checks and constructor/register
diagnostics. All checks run in either mode, and failures are always reported.

Construct the same small SPD matrix from SciPy data and clipped banded
oracles, inspect slots/normalization/workspace, and compare each encoded
block to an independently quantized NumPy reference. Both routes execute
Walk and QSVT consumers. A four-point arithmetic case verifies boundary
clipping; alternative input forms and dilation are shown through metadata.

Example 07 on :doc:`getting_started` inverts a sparse encoding of this kind
with QSVT and separates coefficient, polynomial and circuit error budgets.

.. literalinclude:: ../examples/python/sparse_block_encoding.py
   :language: python
   :start-after: [Begin Documentation]

The double-factorized block encoding vs a flat PauliLCU
-------------------------------------------------------

A molecular Hamiltonian contains many interaction coefficients, and the way
those coefficients are organized affects the quantum circuit built from them.
This example uses double factorization to express the interactions through
simpler contributions in rotated orbital bases. Starting from real molecular
integrals, it compares that construction with a Pauli expansion and asks how
keeping fewer factors or refining them changes the encoded Hamiltonian and its
resource requirements.

You should learn how the choice of orbital basis turns the selected
interactions into simple phase operations, and how QROM-loaded angle words let
a quantum index choose that basis coherently. The fixed-angle and programmable
constructions share the same block-encoding interface, allowing the same Walk
and QSVT consumers to use them. Independent references separate factorization
and coefficient errors from angle rounding and circuit error, so readers can
evaluate accuracy together with normalization, qubit counts, and the cost of
loading, using, and clearing the rotation data.

Individual validation checks are shown with ``--verbose``. All checks still
run by default, and failures are always reported.

Runs the worked double-factorized encoding (the bring-your-own-encoding
example ``df_encoding.py``, next to this script) head to head with a flat
`PauliLCU` on real molecules from PySCF -- comparing alpha and term counts,
sweeping the truncation dial, re-optimizing kept leaves with RC-DF, and
verifying the encoded blocks against a sparse Jordan-Wigner reference.

How the example encoding works, in brief: the factorized Hamiltonian is
regrouped so every term is diagonal in some rotated orbital basis -- the
one-body part in the eigenbasis of the corrected one-body matrix, each
factorization leaf in its own eigenbasis. SELECT walks through these
frames in its fixed backend with *uncontrolled* Givens networks (only Z words and
sign phases carry ancilla controls, so unselected frames telescope to the
identity), and the encoding's normalization reproduces the published LCU
one-norm of ``double_factorization_one_norm(..., "lcu")`` exactly, up to
the identity term. Compressing the factorization (fewer leaves) shrinks
the term count and, typically though not monotonically, the
normalization -- the knob a flat Pauli expansion does not have. The full
construction is documented in the example module's docstrings.

The default H2 run also selects one real spatial frame coherently with
QROM-loaded angle words. It compares fixed and represented operators,
clears angle/ladder workspace, and executes Walk and QSVT against NumPy
references. Use ``--rotation-backend {fixed,qrom,both}`` and
``--angle-bits 4``. A host-only four/six/eight-bit sweep separates angle
rounding from circuit error; larger QROM cases are resource-only.
See the guide's real spatial frame section for register conventions and
the distinction from the papers' complete DF algorithms.

.. literalinclude:: ../examples/python/df_block_encoding.py
   :language: python
   :start-after: [Begin Documentation]
