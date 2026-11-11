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

A guided tour of the `PauliLCU` block-encoding API: build the encoding from a
spin operator, inspect its sizes and normalization, and exercise the device
kernels (`pauli_lcu.prepare`, the select, and the walk) it exposes.

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

Individual validation checks are shown with ``--verbose``. All checks still
run by default, and failures are always reported.

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

Runs the worked double-factorized encoding (the bring-your-own-encoding
example ``df_encoding.py``, next to this script) head to head with a flat
`PauliLCU` on real molecules from PySCF -- comparing alpha and term counts,
sweeping the truncation dial, re-optimizing kept leaves with RC-DF, and
verifying the encoded blocks against a sparse Jordan-Wigner reference.

How the example encoding works, in brief: the factorized Hamiltonian is
regrouped so every term is diagonal in some rotated orbital basis -- the
one-body part in the eigenbasis of the corrected one-body matrix, each
factorization leaf in its own eigenbasis. SELECT walks through these
frames with *uncontrolled* Givens-rotation networks (only the Z words and
sign phases carry ancilla controls, so unselected frames telescope to the
identity), and the encoding's normalization reproduces the published LCU
one-norm of ``double_factorization_one_norm(..., "lcu")`` exactly, up to
the identity term. Compressing the factorization (fewer leaves) shrinks
the term count and, typically though not monotonically, the
normalization -- the knob a flat Pauli expansion does not have. The full
construction is documented in the example module's docstrings.

.. literalinclude:: ../examples/python/df_block_encoding.py
   :language: python
   :start-after: [Begin Documentation]
