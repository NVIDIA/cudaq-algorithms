***************
Getting Started
***************

CUDA-Q Algorithms is a **primitive-first** library of fault-tolerant quantum
algorithms built on `CUDA-Q <https://nvidia.github.io/cuda-quantum/>`_. It is
pure Python: quantum primitives are ``@cudaq.kernel`` device functions, and the
classical preprocessing is NumPy/SciPy.

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

Every algorithm is exposed as a **factory that emits a CUDA-Q kernel**: the
constructor holds the problem, and a method call holds the choices.

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

- :doc:`../guide/preprocessing` — molecule / integrals to a qubit Hamiltonian.
- :doc:`../guide/state_prep` — Hartree-Fock, UCC, and Givens state preparation.
- :doc:`../guide/block_encodings` — ``PauliLCU``, the ``BlockEncoding``
  protocol, and a worked double-factorized example encoding.
- :doc:`../guide/trotter` and :doc:`../guide/qubitization_qsvt` — the two
  Hamiltonian-simulation routes.
- :doc:`../conventions` — qubit ordering, integral tensors, and normalization
  conventions (read this before validating any numerics).
