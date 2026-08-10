Trotter (product-formula simulation)
====================================

Product-formula time evolution for Hamiltonians expressed as sums of Pauli
strings, implemented as a pure-Python peer of the LCU/QSVT primitives:
term extraction, host-side planning and ordering, resource estimation, and
the circuit primitive itself. Requires only the `cudaq` Python package.

Trotterization and QSVT are the two Hamiltonian-simulation routes offered by
the library: this page covers the product-formula route, and
:doc:`qubitization_qsvt` covers the qubitization / QSVT route (exact within
a polynomial approximation, at the cost of an ancilla register).

.. code-block:: text

   python/cudaq_algorithms/trotter.py     term extraction, ordering,
                                          resources,
                                          apply_trotter kernel
   python/cudaq_algorithms/sim_utils.py   shared simulation-only helpers
                                          (evolve lives here)
   tests/python/test_trotter.py           dense-reference test suite

The relocated, runnable end-to-end example is
`docs/sphinx/examples/python/trotter_chemistry.py` (a chemistry-style
walkthrough of the primitive).

Using the primitive
-------------------

.. code-block:: python

   from cudaq_algorithms import trotter

   # Flexible Hamiltonian input: SpinOperator, single spin term,
   # {"XZI...": coeff} mapping, or (coeff, word) pairs.
   evolution = trotter.Trotter(
       hamiltonian,
       ordering=trotter.TrotterOrdering.COEFFICIENT_MAGNITUDE_DESCENDING)

   evolution.kernel(time=0.8, steps=4, order=2)   # ready @cudaq.kernel():
                                                  # |0...0> -> evolved state
   evolution.resources(steps=4, order=2)  # TrotterResourceEstimate
   evolution.num_terms, evolution.identity_coefficient
   evolution.words, evolution.coefficients

`Trotter` extracts and validates the Pauli terms on the host once at
construction (dropping identity terms into `identity_coefficient`) and
applies the requested term ordering; the evolution parameters `time`,
`steps`, and `order` are supplied per kernel request, mirroring the other
primitives (`Walk.kernel(power=...)`, `QSVT.kernel(sequence)`).

Product-formula orders
~~~~~~~~~~~~~~~~~~~~~~~~

- `order=1` — first-order Trotter: one :math:`\exp(-i c_i (t/\mathrm{steps}) P_i)` sweep per
  step; error :math:`O(t^2/\mathrm{steps})`.
- `order=2` (default) — symmetric second-order (Strang) splitting: a
  half-angle forward sweep followed by a half-angle reverse sweep; error
  :math:`O(t^3/\mathrm{steps}^2)`.
- `order=4` — Forest-Ruth fourth-order formula: three symmetric
  second-order sub-steps with time fractions `w1`, `w0`, `w1`; error
  :math:`O(t^5/\mathrm{steps}^4)`.

The Forest-Ruth weights (`w1 = 1/(2 - 2^(1/3))`, `w0 = 1 - 2*w1`) are
precomputed private module constants: CUDA-Q kernels cannot call host-only
math such as cube roots, so the kernel consumes the constants directly.

Circuit-level optimization deliberately deferred (documented, not
implemented): merging the back-to-back half-rotations at sweep and step
boundaries of the order-2/4 formulas (~1/num_terms of all rotations, each
a CX ladder on hardware); no effect on simulator results.

State preparation injection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`kernel()` takes an optional `state_prep` kernel with signature
`(qubits: cudaq.qview)`: the returned circuit allocates the register in
``|0...0>``, runs `state_prep` on it, then evolves — still zero-argument and
directly sampleable, with no statevector anywhere:

.. code-block:: python

   @cudaq.kernel
   def my_prep(qubits: cudaq.qview):
       rx(0.37, qubits[0])
       ry(-0.52, qubits[1])

   kernel = evolution.kernel(time=0.8, steps=4, order=2, state_prep=my_prep)
   counts = cudaq.sample(kernel)

`state_prep` must act only on the register it is handed (width
`num_qubits`). The same parameter exists on the LCU/qubitization/QSVT
factories; see :doc:`state_prep` for the chemistry-style preparations that
plug in here, and :doc:`block_encodings` for the encoding factories that
share the contract.

Composing inside user kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The flattened primitive is the escape hatch for composition with state
preparation or measurement in a custom kernel:

.. code-block:: python

   import cudaq
   from cudaq_algorithms import trotter

   @cudaq.kernel
   def my_kernel(coeffs: list[float], words: list[cudaq.pauli_word],
                 t: float, steps: int, order: int):
       q = cudaq.qvector(4)
       # ... state preparation ...
       trotter.apply_trotter(coeffs, words, t, steps, order, q)

`evolution.coefficients` / `evolution.words` supply the flattened arrays.

Identity terms and the global phase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For `H = c I + H'`, the circuit applies the product formula for `H'` only;
`exp(-i c t)` cannot be realized as a circuit on the evolved register. The
phase is an unobservable global phase for a single unconditioned evolution
but a real relative phase for controlled or interference-based algorithms —
`identity_coefficient` is reported on the `Trotter` object so callers
can account for it.

Simulation-only helper
----------------------

Statevector conveniences live in the shared `cudaq_algorithms.sim_utils`
module (simulation-only: it uses `cudaq.get_state`, which does not exist on
hardware targets). The Trotter-specific helper is `evolve`:

.. code-block:: python

   from cudaq_algorithms import sim_utils

   evolved = sim_utils.evolve(evolution, ket, time=0.8, steps=4, order=2)
   # approximates exp(-i H t)|ket>, identity phase included

Unlike the circuit primitive, `evolve` reintroduces the identity phase by
default (`include_identity_phase=True`), so the result approximates the
full `exp(-i H t)|ket>`.

`evolve` delegates to `Trotter.state_kernel(time, steps, order)` — a
`@cudaq.kernel(state)` factory sharing the same validation and marshaling
as `Trotter.kernel` — and raises `ValueError` for invalid parameters
(including a ket whose dimension does not match `num_qubits`).

Testing
-------

`tests/python/test_trotter.py` pins correctness against independent dense
references: exact matrix exponentials via diagonalization, and an explicit
Pauli-rotation simulator for per-order product formulas. Coverage includes
kernel interop with flattened arguments, invalid-input handling, per-order
error thresholds, asymptotic error-scaling slope fits (order-p error ~
dt^p), exactness for commuting Hamiltonians, a 14-term 4-qubit
chemistry-style Hamiltonian, identity/global-phase handling, and every
accepted input form of the term-extraction front end.

Known CUDA-Q Python constraints
--------------------------------

Two upstream compiler behaviors shape the implementation:

- `return` inside a Python kernel is silently ignored
  (`cuda-quantum#4845 <https://github.com/NVIDIA/cuda-quantum/issues/4845>`_);
  the `apply_trotter` body is a single positively-guarded block instead of
  early-return guards.
- Captured empty lists cannot be marshaled
  (`cuda-quantum#4847 <https://github.com/NVIDIA/cuda-quantum/issues/4847>`_);
  identity-only Hamiltonians special-case the kernel factories.
