Preprocessing: molecule to qubit Hamiltonian
============================================

This page walks the classical front end of the library: turning an
electronic-structure problem into the objects the quantum primitives
consume. The pipeline is

.. code-block:: text

    integrals (FCIDUMP / PySCF / Psi4)
        -> fermion-to-qubit transform (Jordan-Wigner / Bravyi-Kitaev)
        -> qubit Hamiltonian (cudaq.SpinOperator)
        -> PauliLCU / Walk / QSVT / Trotter

with an optional **double-factorization** compression of the
two-electron integrals inserted before the transform. Everything here is
pure Python and needs no compiled extension.

All integral tensors follow the same convention: a chemist-notation
``(pq|rs)`` two-electron tensor over real spatial orbitals, paired with an
``(n, n)`` core Hamiltonian and a scalar (nuclear-repulsion / core)
energy. The tensor conventions — qubit ordering, chemist vs. physicist
notation, spin expansion — are specified in :doc:`../conventions`; read
that page before validating any numerics.

Integral sources (FCIDUMP / PySCF / Psi4)
-----------------------------------------

`cudaq_algorithms.chemistry` provides three loaders, all returning the
same ``(one_body, eri, core_energy)`` triple in chemist ``(pq|rs)``
notation over real spatial orbitals — exactly the arguments
:func:`cudaq_algorithms.chemistry.qubit_hamiltonian` and the
double-factorization module expect. Pass the returned ``core_energy`` (or
``nuclear_repulsion``) as ``scalar_offset``; it is kept classical rather
than encoded into the operator.

FCIDUMP
~~~~~~~

:func:`~cudaq_algorithms.chemistry.from_fcidump` parses the *text* of an FCIDUMP file — the de-facto
interchange format written by Molpro, PySCF, Psi4, and Block/DMRG codes.
The caller does the file I/O, so the parse stays pure and testable:

.. code-block:: python

    from pathlib import Path
    from cudaq_algorithms import chemistry

    one_body, eri, core = chemistry.from_fcidump(
        Path("molecule.fcidump").read_text())
    h = chemistry.qubit_hamiltonian(one_body, eri, scalar_offset=core)
    # or hand the same tensors to a block encoding, e.g. the
    # DoubleFactorizedEncoding example (docs/sphinx/examples/python)

The reader has no third-party dependency (pure text + NumPy) and fills the
eight-fold index symmetry from the symmetry-unique records FCIDUMP stores.
FCIDUMP indices are Fortran 1-based; a ``value i j k l`` record with all of
``i, j, k, l`` nonzero is a two-electron integral, with ``k == l == 0`` a
one-electron integral ``h_ij``, and with all indices zero the core energy.
Only real (RHF/ROHF) FCIDUMP files are supported; unrestricted variants
(Molpro's ``IUHF=1`` or Psi4's ``UHF=.TRUE.``) carry a spin-resolved
integral set with a different index symmetry and are rejected up front by a
header guard.

PySCF
~~~~~

:func:`~cudaq_algorithms.chemistry.from_pyscf` extracts chemist-notation MO integrals plus the
nuclear-repulsion energy from a converged **restricted** mean field:

.. code-block:: python

    from pyscf import gto, scf
    from cudaq_algorithms import chemistry

    mol = gto.M(atom=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7474))],
                basis="sto-3g", symmetry=False)
    mean_field = scf.RHF(mol).run()

    one_body, eri, nuclear_repulsion = chemistry.from_pyscf(mean_field)
    h = chemistry.qubit_hamiltonian(one_body, eri,
                                    scalar_offset=nuclear_repulsion)

The one-body block is built from ``mean_field.get_hcore()`` — the AO-basis
core Hamiltonian the mean field actually used (kinetic + nuclear plus any
ECP, X2C, or QM/MM contribution) — rotated into the MO basis, and the
two-electron tensor comes from ``ao2mo.full`` restored to the dense
``(n, n, n, n)`` form. Restricted references only (a single ``mo_coeff``
matrix); an unrestricted/UHF reference is rejected.

Psi4
~~~~

:func:`~cudaq_algorithms.chemistry.from_psi4` is the Psi4 counterpart, taking a converged
restricted wavefunction — e.g. the second return value of
``psi4.energy("scf", return_wfn=True)`` — and returning the identical
``(one_body, eri, nuclear_repulsion)`` triple, so either loader drives
:func:`cudaq_algorithms.chemistry.qubit_hamiltonian` unchanged:

.. code-block:: python

    import psi4
    from cudaq_algorithms import chemistry

    _, wavefunction = psi4.energy("scf", return_wfn=True)
    one_body, eri, nuclear_repulsion = chemistry.from_psi4(wavefunction)
    h = chemistry.qubit_hamiltonian(one_body, eri,
                                    scalar_offset=nuclear_repulsion)

The core Hamiltonian is read from ``wavefunction.H()`` and the MO
integrals from ``MintsHelper.mo_eri`` (already in chemist ``(pq|rs)``
ordering, matching the PySCF path). The wavefunction must be restricted
(uses ``Ca``) and computed in C1 symmetry (``symmetry c1`` in the Psi4
geometry); an irrep-blocked or unrestricted wavefunction is rejected.

Fermion-to-qubit transforms (Jordan-Wigner and Bravyi-Kitaev)
-------------------------------------------------------------

`cudaq_algorithms.fermion` compiles fermionic integrals to qubit
operators, in pure Python — no compiled extension required.

.. code-block:: python

    from cudaq_algorithms.fermion import jordan_wigner, bravyi_kitaev

    h = jordan_wigner(one_body, two_body, scalar_offset=e_nuc)   # SpinOperator
    h = bravyi_kitaev(one_body, two_body, scalar_offset=e_nuc)

Both accept an ``(n, n)`` one-body tensor, optionally with an
``(n, n, n, n)`` two-body tensor (or a two-body tensor alone); entries are
the coefficients of ``adag_i a_j`` and ``adag_i adag_j a_k a_l`` over ``n``
spin orbitals:

.. math::

   H \;=\; \texttt{scalar\_offset} \cdot I
   \;+\; \sum_{ij} h_{ij} \, a_i^\dagger a_j
   \;+\; \sum_{ijkl} V_{ijkl} \, a_i^\dagger a_j^\dagger a_k a_l

`scalar_offset` is added as an identity term (e.g. nuclear repulsion);
input entries and compiled terms below `tolerance` (default ``1e-15``) are
dropped. The result is a `cudaq.SpinOperator`, ready for
`PauliLCU`/`Walk`/`QSVT` or `Trotter`.

The chemistry bridge (spatial integrals to a qubit Hamiltonian)
---------------------------------------------------------------

`cudaq_algorithms.chemistry` closes the loop between the classical
preprocessing and the quantum primitives. It consumes the same conventions
throughout — an ``(n, n)`` core Hamiltonian and an ``(n, n, n, n)``
chemist-notation ``(pq|rs)`` tensor over real spatial orbitals —
spin-expands them (interleaved spins: ``2p`` up, ``2p + 1`` down), and
applies the Jordan-Wigner transform.

:func:`cudaq_algorithms.chemistry.spin_orbital_tensors` performs the spin
expansion by itself, returning ``(one_body_so, two_body_so)`` over ``2n``
spin orbitals — the fermionic tensors as consumed by `fermion.jordan_wigner`
— for users who need them directly. It validates that `eri` obeys the
real-orbital chemist permutation symmetry
(``(pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)`` and their compositions), which
is what makes the resulting qubit Hamiltonian Hermitian; pass
``validate_symmetry=False`` to skip the check (e.g. for genuinely complex
integrals).

:func:`cudaq_algorithms.chemistry.qubit_hamiltonian` combines the spin
expansion and the Jordan-Wigner transform into a single call, returning a
`cudaq.SpinOperator`. ``scalar_offset`` is added as an identity term (the
nuclear-repulsion energy); ``tolerance`` prunes negligible terms inside the
transform.

.. code-block:: python

    from cudaq_algorithms import PauliLCU, chemistry

    h = chemistry.qubit_hamiltonian(one_body, eri,
                                    scalar_offset=nuclear_repulsion)
    encoding = PauliLCU(h)      # block encoding of h / alpha -> Walk / QSVT

The end-to-end path — mean field to block-encoded ground-state estimation —
is worked in the ``03_chemistry_to_ground_state.py`` example script under
``docs/sphinx/examples/python/``.

Classical double factorization (optional compression)
------------------------------------------------------

`cudaq_algorithms.double_factorization` factorizes the two-electron
integrals into a sum of low-rank, diagonalizable pieces — the
representation used by double-factorized block encodings, qubitization, and
measurement schemes. It implements the explicit (X-DF) and compressed
(C-DF) variants of Cohn, Motta, and Parrish, *Quantum Filter
Diagonalization with Compressed Double-Factorized Hamiltonians*, PRX
Quantum **2**, 040352 (2021)
(`arXiv:2104.08957 <https://arxiv.org/abs/2104.08957>`_).

Inserted before the transform, it is an **optional compression** of the
`eri` tensor: reconstruct a truncated tensor with `reconstruct_eri` and
feed it to `qubit_hamiltonian` in place of the exact one (see the chemistry
bridge above). The double-factorized *block encoding* that consumes a
factorization directly -- the worked `DoubleFactorizedEncoding` example --
is described on :doc:`block_encodings`.

Convention
~~~~~~~~~~

The two-electron integrals are supplied in **chemist notation** ``(pq|rs)``
over real spatial orbitals (8-fold symmetric), as an ``(n, n, n, n)`` array
`eri` with ``eri[p, q, r, s] == (pq|rs)`` (e.g. from
``pyscf.ao2mo.restore("s1", ...)``). Double factorization writes

.. math::

   (pq|rs) \;\approx\; \sum_t \sum_{k,l}
   U^t_{pk} U^t_{qk} \, Z^t_{kl} \, U^t_{rl} U^t_{sl}

with orthogonal **leaf rotations** ``U^t`` (which compile into a
Givens-rotation fabric) and symmetric **core** matrices ``Z^t``.

Backends (NVIDIA math libraries)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Heavy linear algebra (eigendecompositions, tensor contractions, least
squares) runs on the NVIDIA math libraries — cuSOLVER and cuBLAS via
**CuPy** — when a GPU is available, and falls back to **NumPy/SciPy**
otherwise. Every entry point accepts ``backend="auto"`` (default),
``"cupy"``, or ``"numpy"``.

``"auto"`` is **size-aware**: below an empirical orbital-count crossover the
GPU's per-kernel launch and host-sync latency makes it slower than NumPy,
so `auto` stays on the CPU for small problems and switches to the GPU only
once the work is large enough to amortize that overhead (and the GPU lead
then grows with ``n``). The crossovers differ by method — C-DF
(contraction-heavy) crosses around ``n ≈ 18``, while X-DF (a cheap Cholesky
loop + tiny per-leaf eigh) stays CPU-favorable until ``n ≈ 56``. Pass
``backend="cupy"`` or ``"numpy"`` to force a backend regardless of size.

X-DF vs C-DF
~~~~~~~~~~~~

- **X-DF** is exact and cheap. The first factorization of the ERI
  supermatrix into symmetric leaves ``(pq|rs) = sum_t L^t_pq L^t_rs``
  defaults to **pivoted Cholesky**: the ERI matrix is positive
  semidefinite (a Gram matrix of orbital densities), so Cholesky is
  rank-revealing — it keeps leaves while the residual pivot exceeds
  `threshold` and stops at the true rank (at most the symmetric-pair
  dimension ``n(n+1)/2``), which is cheaper than a full eigendecomposition.
  (Pass ``first_factorization="eigendecomposition"`` for the
  symmetric-eigendecomposition variant,
  ``(pq|rs) = sum_t lambda_t V^t_pq V^t_rs``, required for indefinite
  inputs.) Each symmetric leaf is then eigendecomposed to give ``U^t`` and
  the rank-one core ``Z^t``. At full rank the reconstruction is exact to
  machine precision, independent of the first-factor method.
- **C-DF** lifts the rank-one restriction on ``Z^t`` and minimizes
  ``O = 1/2 || eri - reconstruction ||_F^2`` over a fixed number of leaves.
  The leaf rotations are parameterized as ``U^t = exp(X^t)`` (antisymmetric
  ``X^t``) and optimized with L-BFGS, while the symmetric cores are solved
  in closed form at each step (the two-step scheme of the paper,
  warm-started from X-DF). C-DF reaches a target accuracy with substantially
  fewer leaves than X-DF.

RC-DF regularization
~~~~~~~~~~~~~~~~~~~~~

Setting ``regularization=rho > 0`` enables **regularized C-DF** (RC-DF,
Oumarou et al., *Quantum* **8**, 1371 (2024),
`arXiv:2212.07957 <https://arxiv.org/abs/2212.07957>`_), adding the L2
penalty ``rho * sum_{t,k,l} (Z^t_kl)^2`` to the objective (Eq. 17). The
penalty is folded into the inner core solve as a ridge term, so it actually
shrinks the cores ``Z^t`` (it also conditions the otherwise rank-deficient
inner system). Smaller cores lower the Hamiltonian one-norm ``lambda`` — and
hence the qubitization runtime and measurement variance — at a modest cost
in reconstruction accuracy. ``rho`` is an absolute coefficient whose useful
scale depends on the integral magnitude (the paper uses ``~1e-6`` to
``1e-3``); pick it by trading off `factorization_error` against
`double_factorization_one_norm`. By the envelope theorem the analytic
optimization gradient is unchanged in form, so RC-DF runs at the same
per-iteration cost as plain C-DF.

Inner core solve: explicit vs matrix-free
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At each L-BFGS step the symmetric cores ``Z^t`` are recovered exactly by a
linear solve for fixed rotations. Two solvers are available via
`inner_solver`:

- ``"lstsq"`` (default) forms an explicit ``n^4 x num_params`` design
  matrix and solves it with least squares. Simple and robust for
  small/medium systems, but the design matrix does not scale.
- ``"cg"`` is the **matrix-free conjugate-gradient** solve (RC-DF,
  `arXiv:2212.07957 <https://arxiv.org/abs/2212.07957>`_, Eqs. 25-30). It
  applies the normal-equations operator
  ``A(Z)^t = sum_{t'} M_{tt'} Z^{t'} M_{tt'}^T`` with
  ``M_{tt'} = (U^t^T U^{t'})`` elementwise-squared — only ``n x n`` matrix
  products, never materializing the design matrix — so it scales to large
  orbital counts. `cg_tolerance` and `cg_max_iterations` control the CG
  iteration.

Both solvers produce the same reconstruction; when the inner system is full
rank (e.g. ``regularization > 0``) they produce the same cores to machine
precision.

The CG inner solve is **iteration-bound**: its cost is roughly linear in
the number of CG iterations, which is set by the requested tolerance and
the conditioning, not by the matvec. Two accelerators (active only for
``inner_solver="cg"``) cut the per-step cost without changing the final
accuracy:

- **`cg_warm_start=True`** seeds each L-BFGS step's CG from the previous
  step's cores. The cores move slowly between steps, so the warm start
  collapses the iteration count.
- **`cg_optimization_tolerance`** (default ``max(cg_tolerance, 1e-6)``)
  solves the *in-loop* systems only loosely — an inexact inner solve, which
  is sound here because the envelope theorem only needs an approximate
  gradient — while the single **final** solve is tightened to
  `cg_tolerance`. So the returned cores are accurate even though the
  optimization steps used a cheap inner solve.

API
~~~

All functions return or operate on a `DoubleFactorization` dataclass with
`num_orbitals`, `leaf_rotations` (``U^t``), `leaf_cores` (``Z^t``),
`num_leaves`, and a `reconstruct_eri` method.

`explicit_double_factorization(eri, threshold=1e-8, max_num_leaves=None, second_factor_threshold=0.0, first_factorization="cholesky", backend="auto")`
    X-DF (rank-one cores). First factorization defaults to **pivoted
    Cholesky**; pass ``first_factorization="eigendecomposition"`` for the
    eigendecomposition variant.

`compressed_double_factorization(eri, num_leaves, max_iterations=2000, tolerance=1e-10, regularization=0.0, inner_solver="lstsq", cg_tolerance=1e-10, cg_max_iterations=None, cg_warm_start=True, cg_optimization_tolerance=None, backend="auto")`
    C-DF by least-squares optimization (warm-started from X-DF).
    ``regularization=rho>0`` enables **RC-DF** (see above).
    ``inner_solver="cg"`` selects the matrix-free inner core solve, with
    ``cg_warm_start`` / ``cg_optimization_tolerance`` accelerators.

`reconstruct_eri(factorization)`
    Rebuild the ``(pq|rs)`` tensor from a factorization.

`factorization_error(eri, factorization)`
    Frobenius norm of the reconstruction residual.

`modified_one_body_integrals(one_body, eri)`
    The DF one-body correction ``kappa_pq = h_pq - 1/2 sum_r (pr|qr)``.

`double_factorization_one_norm(factorization, one_body_eigenvalues, convention="lcu")`
    One-norm ``lambda`` of the DF Hamiltonian; ``convention="lcu"``
    (Pauli-rotation) or ``"burg"`` (qubitization).

Feeding a compressed tensor to the transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`reconstruct_eri` returns a tensor in the same chemist ``(pq|rs)``
convention `qubit_hamiltonian` consumes, so a compressed factorization
drops straight into the chemistry bridge:

.. code-block:: python

    from cudaq_algorithms import PauliLCU, chemistry
    from cudaq_algorithms import double_factorization as df

    factorization = df.compressed_double_factorization(eri, num_leaves=T)
    h = chemistry.qubit_hamiltonian(one_body,
                                    df.reconstruct_eri(factorization),
                                    scalar_offset=nuclear_repulsion)
    encoding = PauliLCU(h)      # block encoding of h / alpha -> Walk / QSVT

`chemistry.qubit_hamiltonian` returns a `cudaq.SpinOperator`, so the
truncated Hamiltonian feeds `PauliLCU`, `Walk`, and `QSVT` directly. The
payoff of compression shows up on the quantum side as a smaller LCU
normalization ``alpha`` (QSVT degree scales like ``alpha * t`` for time
evolution) and fewer Pauli terms (SELECT cost), at the price of a spectrum
shift controlled by the tensor reconstruction error.
`chemistry.qubit_hamiltonian` uses `fermion.jordan_wigner`; the classical
factorization itself does not.

The ``double_factorization.py`` example script under
``docs/sphinx/examples/python/`` factorizes H2O/STO-3G integrals from PySCF
and compares X-DF and C-DF reconstruction errors at equal leaf counts; the
``df_compression_to_qsvt.py`` script sweeps X-DF leaf counts on H2/STO-3G
and tabulates tensor error, ``alpha``, term count, and the exact
ground-state shift at each truncation.

.. seealso::

   - :doc:`block_encodings` — the worked `DoubleFactorizedEncoding`
     example encoding that consumes a factorization directly.
   - :doc:`state_prep` — references injected into the algorithm factories.
   - :doc:`../conventions` — integral-tensor conventions (qubit ordering,
     chemist notation, spin expansion).
