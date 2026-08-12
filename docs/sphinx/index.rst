*****************
CUDA-Q Algorithms
*****************

A primitive-first, fault-tolerant-quantum-computing (FTQC) algorithms library
built on `CUDA-Q <https://nvidia.github.io/cuda-quantum/>`_: block encodings,
qubitization, QSVT, Trotterization, double factorization, fermion-to-qubit
transforms, and state preparation — composed from small, reusable device
kernels.

.. toctree::
   :caption: Getting Started
   :maxdepth: 2

   getting_started/index

.. toctree::
   :caption: User Guide
   :maxdepth: 2

   guide/preprocessing
   guide/block_encodings
   guide/trotter
   guide/qubitization_qsvt
   guide/state_prep

.. toctree::
   :caption: Examples
   :maxdepth: 2

   examples_rst/getting_started
   examples_rst/preprocessing
   examples_rst/block_encodings
   examples_rst/state_prep
   examples_rst/hamiltonian_simulation

.. toctree::
   :caption: Reference
   :maxdepth: 2

   conventions
   api/python_api
