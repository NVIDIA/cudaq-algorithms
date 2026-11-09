Coherent circuit primitives
===========================

These building blocks compose inside CUDA-Q kernels. The runnable examples
compare their complete outputs with independent classical references.

Alias preparation and generic LCU composition
---------------------------------------------

`AliasSamplingPrepare` produces a weighted index marginal with quantum
garbage. It combines classical integer alias tables, a QROM lookup, an
unsigned comparator, and controlled swaps.

.. code-block:: python

   from cudaq_algorithms.primitives import AliasSamplingPrepare

   sampler = AliasSamplingPrepare([1.0, 0.0, 2.0**0.5], mu=2)
   prepare = sampler.kernel()
   unprepare = sampler.adjoint_kernel()

`kernel()` and `adjoint_kernel()` are zero-argument methods (`QROM.kernel`
is a property); the kernels they return both take
`(index: qview, garbage: qview)`. Allocate widths from `num_index`
and `num_garbage`; both registers must start at zero for preparation.

What is prepared
~~~~~~~~~~~~~~~~

Weights are nonnegative with a positive sum. They are padded to
`K = 2**num_index` bins, with zero probability for zero-weight and padded
bins. The index-register measurement probabilities approximate the normalized
weights. This is not a clean state `sum(sqrt(p_k) |k>)` with zero garbage.

The garbage layout is

.. code-block:: text

   [alias(num_index) | keep(mu) | reference(mu) | flag
    | QROM ladder(qrom.num_ladder) | carry]

The table and reference data remain entangled with the index. Conditional
garbage states need not be orthogonal. Only the QROM ladder and comparator
carry are clean between preparation and its inverse. The example reconstructs
the complete state from classical branches and reports the index purity.

Integer tables and discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each original bin k and uniformly distributed reference word r in
`0 <= r < 2**mu`, the selected bin is k if `r < keep[k]`, otherwise
`alias[k]`. Enumerate these `K*2**mu` branches to obtain exact rational
probabilities. Self-aliased full bins remain exact even though their keep
threshold is clamped to mu bits.

`keep`, `alias`, `probabilities`, `table_probabilities`, and
`discretization_bound` are public inspection properties. Per-bin probability
error is bounded by `1.5/(K*2**mu)`. Increasing precision tightens this bound,
but individual rounding errors need not decrease strictly at every step.

Preparation followed by its supplied adjoint restores every register.
Discarding garbage, or reflecting only on the index, does not provide that
contract.

Compose a non-Pauli LCU
~~~~~~~~~~~~~~~~~~~~~~~

The example next encodes :math:`H=0.6X_0-0.4CZ_{0,1}`. Both terms are
Hermitian involutions, but CZ is not a single Pauli word.

.. code-block:: python

   from cudaq_algorithms import LCUBlockEncoding

   encoding = LCUBlockEncoding(
       num_system=2, weights=[0.6, 0.4], mu=2,
       bodies=[
           [("x", 0)],
           [("and_tt", 0, 1, 0), ("z_w", 0),
            ("and_tt", 0, 1, 0), ("sign",)],
       ])

Body programs are host-side data compiled at factory time. Here `and_tt`
computes both system bits into clean work[0]. The leaf-controlled `z_w`
applies the selected CZ phase, the second AND restores work, and `sign`
supplies the negative coefficient. The free compute/uncompute gates cancel
on inactive addresses. The caller is responsible for making extended bodies
Hermitian involutions with complete workspace restoration.

Weights remain positive; their signs belong in SELECT. The exact one-norm
`alpha` is preserved, while `discretized_weights` specifies the weights
actually encoded by the finite tables. At mu=2 this fixture realizes
`0.625 X_0 - 0.375 CZ_01`. The runnable check builds that operator directly
from NumPy matrices rather than calibrating it from circuit output.

The good block of PREPARE/SELECT/UNPREPARE is the represented Hamiltonian
divided by alpha. After nontrivial SELECT, UNPREPARE generally leaves some
amplitude outside that block. SELECT work is nevertheless restored. Project
and reflect using the complete `num_ancilla` register, including index,
preparation garbage, and SELECT work.

Consumers and costs
~~~~~~~~~~~~~~~~~~~~

The same `Walk` and `QSVT` interfaces consume this encoding. Compare their
circuits to the represented operator, and report its discrepancy from the
ideal operator separately. Walk good blocks use `T_k(-H_mu/alpha)`; measured
moments use the positive-H convention.

An extended non-Pauli body does not provide `select_observable` for odd
`Walk.moment` orders. Even moments can require an exponentially expanded
reflection observable, so this example validates small statevector blocks.

A standalone sampler can choose QROM variants. `LCUBlockEncoding` pins its
internal sampler to the ``select`` QROM variant to reuse the clean ladder as
the SELECT walk's ladder. The example
reports each PREPARE direction as its QROM price plus `2*mu-1` Toffolis and
`num_index` separate Fredkins. These are logical counts, without a rotation
or controlled-swap synthesis assumption.

See :doc:`../examples_rst/primitives` for the full script and
:doc:`block_encodings` for the generic encoding protocol.
