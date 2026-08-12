***************
Getting Started
***************

CUDA-Q Algorithms is a **primitive-first** library of fault-tolerant quantum
algorithms built on `CUDA-Q <https://nvidia.github.io/cuda-quantum/>`_. It is
pure Python: quantum primitives are ``@cudaq.kernel`` device functions, and the
classical preprocessing is NumPy/SciPy.

What's in the library
=====================

The library is a small set of layers that feed each other:

- **Chemistry preprocessing** (classical) turns a molecule — via PySCF, Psi4,
  or an FCIDUMP file — into integral tensors and a qubit Hamiltonian, and can
  compress that Hamiltonian by double factorization before anything quantum
  happens.
- **Block encodings** embed a Hamiltonian into a unitary circuit so that
  quantum algorithms can consume it. ``PauliLCU`` ships with the library, and
  a structural protocol lets you plug in an encoding of your own.
- **Algorithm primitives** build circuits from those inputs: qubitization
  walks (``Walk``), the quantum singular value transformation (``QSVT``),
  and Suzuki–Trotter product formulas (``Trotter``).
- **State preparation** produces the initial states those circuits act on:
  Hartree–Fock references, unitary-coupled-cluster ansätze, and
  Givens-rotation Slater determinants.

The goal is to provide the fault-tolerant era's standard circuit building
blocks as small, hardware-shaped, independently validated kernels that
compose — with each other and with your own kernels — rather than as
end-to-end application workflows. A typical path through the library reads
left to right: molecule → qubit Hamiltonian → block encoding → walk/QSVT
circuit → expectation values or spectra.

Installation
============

The package is a single, CUDA-agnostic ``py3-none-any`` wheel that depends on
the ``cudaq`` metapackage (which selects the matching ``cuda-quantum-cuNN`` for
your platform):

.. code-block:: console

   $ pip install cudaq-algorithms

.. note::

   Until the first PyPI release, install from a release wheel with
   ``pip install cudaq-algorithms --find-links <wheel directory>`` or build
   from source (below).

Building from source
--------------------

The library needs no compilation — it is pure Python. Point ``PYTHONPATH`` at
the ``python/`` directory of a checkout (with ``cudaq`` importable) and run the
tests:

.. code-block:: console

   $ PYTHONPATH=python python3 -m pytest tests/python -q

The dense-reference tolerances require an fp64 simulator; ``conftest.py``
selects ``qpp-cpu`` by default (override with ``CUDAQ_DEFAULT_SIMULATOR``).

The primitive-first philosophy
==============================

Every algorithm is exposed as a **factory that emits a CUDA-Q kernel** — a
quantum device function, compiled and run by CUDA-Q (see the
`CUDA-Q kernel basics <https://nvidia.github.io/cuda-quantum/latest/using/basics/kernel_intro.html>`_
if the term is new). The constructor holds the problem, and a method call
holds the choices.

.. code-block:: python

   from cudaq_algorithms import PauliLCU, Walk, QSVT, Trotter

   encoding = PauliLCU(hamiltonian)      # block-encode H / alpha
   walk = Walk(encoding).kernel(power=3) # qubitization walk W^3
   evolve = QSVT(encoding).kernel(phases)
   trotter = Trotter(hamiltonian).kernel(time=1.0, steps=10, order=2)

Consumers such as :class:`~cudaq_algorithms.qubitization.Walk` and
:class:`~cudaq_algorithms.qsvt.QSVT` are generic over the **structural**
``BlockEncoding`` protocol (:mod:`cudaq_algorithms.block_encoding`) — implement
the documented factory surface and any encoding plugs in, no inheritance
required. See :doc:`../guide/block_encodings` for the protocol and a
"bring your own encoding" walkthrough.

Where to go next
================

- :doc:`../guide/block_encodings` — ``PauliLCU``, the ``BlockEncoding``
  protocol, and a worked double-factorized example encoding.
- :doc:`../guide/trotter` and :doc:`../guide/qubitization_qsvt` — the two
  Hamiltonian-simulation routes.
- :doc:`../guide/preprocessing` — molecule / integrals to a qubit Hamiltonian.
- :doc:`../guide/state_prep` — Hartree-Fock, UCC, and Givens state preparation.
- :doc:`../conventions` — qubit ordering, integral tensors, and normalization
  conventions (read this before validating any numerics).
