# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# [Begin Documentation]
"""A user-written block encoding on real molecules, vs the built-in PauliLCU.

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

The encoding here is user-level code: ``df_encoding.py`` next door
implements ``DoubleFactorizedEncoding`` against the public
``BlockEncoding`` protocol. Because the protocol is structural, the same
``Walk`` consumer drives the user-written encoding and the built-in
``PauliLCU`` identically.

Integrals come from PySCF (``pip install pyscf``) via
``chemistry.from_pyscf``, for a menu of molecular configurations::

    python3 df_block_encoding.py [config] [--circuits]
        [--rotation-backend {fixed,qrom,both}] [--angle-bits 4]

    h2         H2 / STO-3G           2 orbitals ->  4 system qubits  (default)
    h2o-cas44  H2O / STO-3G CAS(4,4) 4 orbitals ->  8 system qubits
    h4         linear H4 / STO-3G    4 orbitals ->  8 system qubits
    lih        LiH / STO-3G          6 orbitals -> 12 system qubits
    h2o        H2O / STO-3G          7 orbitals -> 14 system qubits
    h2o-631g   H2O / 6-31G          13 orbitals -> 26 system qubits

Every configuration tells the classical story: double-factorize the ERI,
compare the DF encoding's normalization ``alpha`` and term count against
the flat Pauli-expansion ``PauliLCU`` baseline, and sweep the truncation
dial (fewer leaves -> fewer terms; alpha typically, though not
monotonically, shrinks -- see the note in the sweep. The flat expansion
has no such knob). All but the largest configuration then re-optimize the
kept leaves with RC-DF at the same budgets
(``compressed_double_factorization`` with a small ridge) -- the second
dial: optimize, don't just truncate. Small configurations also
run the circuits: the encoded block is checked against a sparse
Jordan-Wigner reference Hamiltonian, and the same ``Walk`` consumer
measures Chebyshev moments through both encodings. Larger configurations
skip circuit execution (statevector cost is printed)
-- the classical preprocessing scales; the simulator is what does not.
``lih`` sits on the boundary: pass ``--circuits`` to run it anyway
(hours on a typical CPU).

The H2 default executes both fixed and QROM-programmed spatial frames,
then reports a host-only precision sweep. Larger QROM programs are
resource-only; the size policy above applies to the fixed backend. Tight
circuit checks target the represented operator, with angle rounding
reported separately. No compiled extension is needed.
Add --verbose to show individual validation checks; failures are always reported.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import scipy.sparse as sp

import cudaq

from cudaq_algorithms import (PauliLCU, PhaseSequence, QSVT, Walk, chemistry,
                              state_from)
from cudaq_algorithms import double_factorization as df

# The encoding under demonstration is a sibling example file, not a package
# module (CUDA-Q kernels need real .py files, so a plain import is fine).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from df_encoding import DoubleFactorizedEncoding

cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

try:
    from pyscf import ao2mo, gto, mcscf, scf
except ImportError:
    sys.exit("This example builds its molecules with PySCF: pip install pyscf")

# ----------------------------------------------------------------------
# The molecule menu. Each builder returns (one_body, eri, core_energy,
# mean-field energy) with chemist (pq|rs) MO integrals -- exactly what
# chemistry.qubit_hamiltonian and DoubleFactorizedEncoding consume.
# ----------------------------------------------------------------------

_H2O_GEOMETRY = """
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692
"""


def _rhf(atom: str, basis: str):
    molecule = gto.M(atom=atom, basis=basis, symmetry=False, verbose=0)
    return scf.RHF(molecule).run()


def _full_space(atom: str, basis: str):
    mean_field = _rhf(atom, basis)
    one_body, eri, nuclear = chemistry.from_pyscf(mean_field)
    return one_body, eri, nuclear, float(mean_field.e_tot)


def _h2o_cas44():
    """H2O in a (4 electron, 4 orbital) active space.

    PySCF's CASCI effective integrals fold the inactive (core) orbitals
    into a 4-orbital one-body matrix and scalar -- real water at 8 system
    qubits, small enough for the full dense-checked circuit story.
    """
    mean_field = _rhf(_H2O_GEOMETRY, "sto-3g")
    cas = mcscf.CASCI(mean_field, ncas=4, nelecas=4)
    one_body, core_energy = cas.get_h1eff()
    eri = ao2mo.restore(1, cas.get_h2eff(), 4)
    return (np.asarray(one_body), np.asarray(eri), float(core_energy),
            float(mean_field.e_tot))


_H4_CHAIN = "; ".join(f"H 0 0 {i:.1f}" for i in range(4))

CONFIGS = {
    "h2": {
        "label": "H2 / STO-3G",
        "build": lambda: _full_space("H 0 0 0; H 0 0 0.7414", "sto-3g"),
        "mode": "full",
    },
    "h2o-cas44": {
        "label": "H2O / STO-3G, CAS(4e,4o) active space",
        "build": _h2o_cas44,
        "mode": "full",
    },
    "h4": {
        # Smallest system where leaf truncation shows a real alpha-vs-error
        # trade-off; a standard strong-correlation benchmark (stretched
        # chain: 1.0 A spacing vs 0.74 A equilibrium).
        "label": "linear H4 (1.0 A spacing) / STO-3G",
        "build": lambda: _full_space(_H4_CHAIN, "sto-3g"),
        "mode": "full",
    },
    "lih": {
        "label": "LiH / STO-3G",
        "build": lambda: _full_space("Li 0 0 0; H 0 0 1.595", "sto-3g"),
        "mode": "flagged",  # circuits only with --circuits (hours)
    },
    "h2o": {
        "label": "H2O / STO-3G (full space)",
        "build": lambda: _full_space(_H2O_GEOMETRY, "sto-3g"),
        "mode": "classical",
    },
    "h2o-631g": {
        "label": "H2O / 6-31G (full space)",
        "build": lambda: _full_space(_H2O_GEOMETRY, "6-31g"),
        "mode": "classical",
        "cdf": False,  # L-BFGS at 13 orbitals is a coffee break, not a demo
    },
}

# ----------------------------------------------------------------------
# Sparse Jordan-Wigner reference (interleaved spins, qubit 0 least
# significant -- the library's convention). Sparse matrices keep the
# reference cheap out to the 12-qubit LiH configuration.
# ----------------------------------------------------------------------


def _sparse_annihilators(num_qubits: int) -> list[sp.csr_matrix]:
    z2 = sp.csr_matrix(np.diag([1.0, -1.0]))
    identity = sp.identity(2, format="csr")
    lowering = sp.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]]))
    out = []
    for mode in range(num_qubits):
        ops = ([z2] * mode + [lowering] + [identity] *
               (num_qubits - mode - 1))[::-1]
        matrix = sp.identity(1, format="csr")
        for op in ops:
            matrix = sp.kron(matrix, op, format="csr")
        out.append(matrix)
    return out


def reference_hamiltonian(one_body: np.ndarray,
                          eri: np.ndarray) -> sp.csr_matrix:
    """H = sum h_pq E_pq + 1/2 sum (pq|rs) (E_pq E_rs - delta_qr E_ps)."""
    n = one_body.shape[0]
    num_qubits = 2 * n
    lower = _sparse_annihilators(num_qubits)
    raise_ = [op.conj().T.tocsr() for op in lower]

    def excite(p, q):
        return (raise_[2 * p] @ lower[2 * q] +
                raise_[2 * p + 1] @ lower[2 * q + 1])

    excitations = [[excite(p, q) for q in range(n)] for p in range(n)]
    dim = 1 << num_qubits
    h = sp.csr_matrix((dim, dim), dtype=complex)
    for p in range(n):
        for q in range(n):
            h = h + one_body[p, q] * excitations[p][q]
            for r in range(n):
                for s in range(n):
                    term = excitations[p][q] @ excitations[r][s]
                    if q == r:
                        term = term - excitations[p][s]
                    h = h + 0.5 * eri[p, q, r, s] * term
    return h.tocsr()


def chebyshev_moment(h_scaled: sp.csr_matrix, ket: np.ndarray,
                     order: int) -> float:
    """<ket| T_k(H/alpha) |ket> by the vector three-term recurrence."""
    t_prev, t_cur = ket, h_scaled @ ket
    if order == 0:
        return float(np.real(ket.conj() @ t_prev))
    for _ in range(order - 1):
        t_prev, t_cur = t_cur, 2.0 * (h_scaled @ t_cur) - t_prev
    return float(np.real(ket.conj() @ t_cur))


def check(label: str, condition: bool, *, verbose=False):
    if verbose or not condition:
        print(f"  [check] {label} ... {'OK' if condition else 'FAILED'}")
    if not condition:
        sys.exit(1)


# ----------------------------------------------------------------------
# The story, per configuration
# ----------------------------------------------------------------------


def fock_matrix(orbital):
    """Independent second quantization by determinant minors."""
    occupied = [[p for p in range(len(orbital)) if bits >> p & 1]
                for bits in range(1 << len(orbital))]
    result = np.zeros((len(occupied), len(occupied)), dtype=complex)
    for s, rows in enumerate(occupied):
        for t, cols in enumerate(occupied):
            if len(rows) == len(cols):
                result[s, t] = np.linalg.det(orbital[np.ix_(rows, cols)])
    return result


def frame_hamiltonian(encoding, *, represented):
    """Retained coefficients/Z words conjugated by independent Fock frames."""
    frames = (encoding.represented_frame_matrices
              if represented else encoding.ideal_frame_matrices)
    lifts = [fock_matrix(np.kron(frame, np.eye(2))) for frame in frames]
    dimension = 1 << encoding.num_system
    result = np.zeros((dimension, dimension), dtype=complex)
    for coefficient, qubits, frame in encoding.terms:
        diagonal = np.array([(-1)**sum((bits >> q) & 1 for q in qubits)
                             for bits in range(dimension)])
        g = lifts[frame]
        result += coefficient * (g * diagonal) @ g.conj().T
    return result


def angle_error_bound(encoding):
    bounds = encoding.rotation_report["frame_error_bounds"]
    return sum(
        abs(c) * min(2., 2 * bounds[frame])
        for c, qubits, frame in encoding.terms if qubits)


def signal_response(x, phases):
    """Scalar 2x2 projector-phase response; no quantum output calibration."""
    s = np.sqrt(max(0., 1 - x * x))
    w = np.array([[-x, -s], [s, -x]], dtype=complex)
    result = np.diag([np.exp(1j * phases[0]), 1.])
    for phi in phases[1:]:
        result = np.diag([np.exp(1j * phi), 1.]) @ w @ result
    return result[0, 0]


def report_programmed_frames(one_body,
                             eri,
                             factorized,
                             core_energy,
                             angle_bits,
                             circuits,
                             *,
                             verbose=False):
    """Small H2 numerical capstone; larger programs are resource reports."""
    encoding = DoubleFactorizedEncoding(one_body,
                                        factorized.factorization,
                                        rotation_backend="qrom",
                                        angle_bits=angle_bits)
    report = encoding.rotation_report
    total = encoding.num_system + encoding.num_ancilla
    print(f"\n  QROM frames: b={angle_bits}, alpha={encoding.alpha:.12g}, "
          f"{report['mixing_slots']} real spatial mixing slots")
    print(f"  Layout: {encoding.num_system} system + "
          f"{report['term_index_width']} term index + "
          f"{report['program_width']} angle + {report['ladder_width']} ladder "
          f"= {total} qubits; controlled forms add one")
    print(f"  SELECT QROM: variant={report['qrom_variant']}, "
          f"table rows={report['table_size']}; "
          f"{report['lookup_calls']} lookup/unlookup calls per SELECT")
    print(f"  PREPARE: {report['prepare_rotations']} tree rotations; "
          f"SELECT: {report['programmable_spatial_calls']} bit-controlled "
          "spatial rotations (two spin sectors each); external control "
          "acts only on Z words/signs, not on lookup or rotations.")
    print("  Gate counts here are logical operations, without rotation "
          "synthesis or paper-optimized normalization.")
    check("fixed and QROM retain identical coefficients and alpha",
          encoding.terms == factorized.terms
          and encoding.alpha == factorized.alpha,
          verbose=verbose)
    # Dense determinant-minor references grow exponentially. Only the default
    # two-spatial-mode case runs them; larger molecules retain the original
    # classical truncation/compression and fixed-backend demonstrations.
    if len(one_body) > 2:
        print("  QROM circuits and dense angle sweep: resource-only at this "
              f"size ({total} qubits). Angle operator-error bound "
              f"{angle_error_bound(encoding):.6g} Ha.")
        return
    for j, (_, qubits, frame) in enumerate(encoding.terms):
        width = report["term_index_width"]
        address = int(f"{j:0{width}b}"[::-1], 2)
        print(
            f"    term {j:2d} -> physical address {address:2d}, frame {frame}, "
            f"Z{qubits}, packed angle row {report['angle_rows'][address]}")

    input_h = reference_hamiltonian(one_body, eri).toarray()
    df_h = reference_hamiltonian(one_body,
                                 df.reconstruct_eri(
                                     factorized.factorization)).toarray()
    kept = frame_hamiltonian(factorized, represented=False)
    represented = frame_hamiltonian(encoding, represented=True)
    angle_error = np.linalg.norm(represented - kept, 2)
    bound = angle_error_bound(encoding)
    print(
        f"  Operator errors (Ha): factorization={np.linalg.norm(df_h-input_h,2):.3e}, "
        f"threshold={np.linalg.norm(kept-df_h,2):.3e}, "
        f"angle={angle_error:.3e}, bound={bound:.3e} (+1e-10 floating slack)")
    check("angle error satisfies its derived operator bound",
          angle_error <= bound + 1e-10,
          verbose=verbose)
    print(
        f"  NumPy ground energies, including classical core offset: "
        f"fixed={np.linalg.eigvalsh(kept)[0] + core_energy:.12f}, "
        f"represented={np.linalg.eigvalsh(represented)[0] + core_energy:.12f} Ha"
    )
    print(
        "  These are classical diagonalizations; action checks do not measure an energy."
    )
    print("  Host-only precision sweep (no circuit execution):")
    ideal_energy = np.linalg.eigvalsh(kept)[0]
    for bits in (4, 6, 8):
        candidate = DoubleFactorizedEncoding(one_body,
                                             factorized.factorization,
                                             rotation_backend="qrom",
                                             angle_bits=bits)
        h_b = frame_hamiltonian(candidate, represented=True)
        print(
            f"    b={bits}: operator error={np.linalg.norm(h_b-kept,2):.3e}, "
            f"ground-energy error={abs(np.linalg.eigvalsh(h_b)[0]-ideal_energy):.3e} Ha, "
            f"{candidate.num_system + candidate.num_ancilla} qubits")
    if not circuits or total > 20:
        print(
            f"  QROM circuit execution skipped; resource-only ({total} qubits)."
        )
        return
    rng = np.random.default_rng(7)
    ket = rng.normal(size=len(kept)) + 1j * rng.normal(size=len(kept))
    ket = (ket / np.linalg.norm(ket)).astype(np.complex128)

    def verify(label, kernel, expected):
        actual = np.asarray(cudaq.get_state(kernel,
                                            state_from(ket)))[:len(ket)]
        error = float(np.max(np.abs(actual - expected)))
        if verbose or not np.isfinite(error) or error > 1e-10:
            print(f"  {label}: max |diff|={error:.3e}, tolerance=1e-10")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
        check(label, error <= 1e-10, verbose=verbose)

    scaled = represented / encoding.alpha
    verify("QROM <0|U|0> = H_b/alpha", encoding.encode_kernel(), scaled @ ket)
    for order in (1, 2):
        polynomial = -scaled if order == 1 else 2 * scaled @ scaled - np.eye(
            len(kept))
        verify(f"QROM Walk T_{order}(-H_b/alpha)", encoding.walk_kernel(order),
               polynomial @ ket)
    phases = [.3, -.4, .25]
    values, vectors = np.linalg.eigh(scaled)
    expected = (vectors * [signal_response(x, phases)
                           for x in values]) @ (vectors.conj().T @ ket)
    verify("QROM QSVT projector response",
           QSVT(encoding).kernel(PhaseSequence(phases)), expected)
    print(
        "  QROM measured moments omitted: the expanded reflection observable "
        "grows exponentially with workspace.")


def run(key: str,
        force_circuits: bool,
        rotation_backend="both",
        angle_bits=4,
        *,
        verbose=False):
    config = CONFIGS[key]
    one_body, eri, core_energy, scf_energy = config["build"]()
    n = one_body.shape[0]
    num_system = 2 * n

    print(f"=== {key}: {config['label']} ===")
    print(f"RHF energy {scf_energy:.6f} Ha; core/nuclear constant "
          f"{core_energy:.6f} Ha (added classically, not encoded)")
    print(f"{n} spatial orbitals -> {num_system} system qubits")

    # --- classical story: both encodings, and the truncation dial --------
    flat = PauliLCU(chemistry.qubit_hamiltonian(one_body, eri))
    factorization = df.explicit_double_factorization(eri, threshold=0.0)
    factorized = DoubleFactorizedEncoding(one_body, factorization)

    print(f"\n  flat PauliLCU:            alpha = {flat.alpha:10.4f}, "
          f"{flat.num_terms} Pauli terms, {flat.num_ancilla} ancillas")
    print(f"  DoubleFactorizedEncoding: alpha = {factorized.alpha:10.4f}, "
          f"{factorized.num_terms} Z-word terms in {factorized.num_frames} "
          f"frames ({factorization.num_leaves} leaves), "
          f"{factorized.num_givens_rotations} Givens rotations, "
          f"{factorized.num_ancilla} ancillas")

    print("\n  Truncating the factorization (the knob PauliLCU lacks):")
    total = factorization.num_leaves
    if total <= 12:
        leaf_counts = list(range(1, total + 1))
    else:
        leaf_counts = sorted(
            set(np.linspace(1, total, 10).astype(int).tolist()))
    errors = []
    for leaves in leaf_counts:
        truncated = df.explicit_double_factorization(eri,
                                                     threshold=0.0,
                                                     max_num_leaves=leaves)
        encoding = DoubleFactorizedEncoding(one_body, truncated)
        error = df.factorization_error(eri, truncated)
        errors.append(error)
        print(f"    {leaves:3d} leaves: alpha = {encoding.alpha:10.4f}, "
              f"{encoding.num_terms:5d} terms, tensor error {error:.2e}")
    # Note: alpha is NOT guaranteed monotone in the leaf count -- dropping a
    # leaf also reshapes the one-body singles absorbed into kappa, so alpha
    # can overshoot at intermediate truncations. The tensor error IS
    # monotone (nested pivoted-Cholesky truncation).
    check("truncation error is non-increasing in leaves",
          all(a >= b - 1e-9 for a, b in zip(errors, errors[1:])),
          verbose=verbose)
    check("full-rank X-DF reconstructs the ERI exactly",
          errors[-1] < 1e-8,
          verbose=verbose)

    # RC-DF: the other dial. X-DF truncation keeps the FIRST leaves of an
    # exact factorization; C-DF re-optimizes the leaves you keep (L-BFGS
    # over the rotations, closed-form cores) for the same budget. The small
    # ridge (regularization=1e-4, i.e. RC-DF) matters: unregularized C-DF
    # can exploit gauge freedom to fit better with ENORMOUS core entries --
    # alpha blows up by orders of magnitude -- the pathology the
    # regularized variant exists to prevent.
    if config.get("cdf", True):
        if n <= 4:
            budgets = sorted(
                {max(1, round(total * f))
                 for f in (0.25, 0.5, 0.75)} - {total})
        else:
            budgets = [max(1, round(total / 3))]  # L-BFGS gets expensive
        budgets = [b for b in budgets if b < total]  # rank-1: nothing to do
        print("\n  RC-DF at the same leaf budgets (optimize the kept "
              "leaves, don't just truncate):")
        eri_norm = float(np.linalg.norm(eri))  # scale-invariant gate
        wins = []
        for leaves in budgets:
            truncated = df.explicit_double_factorization(eri,
                                                         threshold=0.0,
                                                         max_num_leaves=leaves)
            xdf_error = df.factorization_error(eri, truncated)
            compressed = df.compressed_double_factorization(
                eri, num_leaves=leaves, regularization=1e-4)
            cdf_error = df.factorization_error(eri, compressed)
            cdf_alpha = DoubleFactorizedEncoding(one_body, compressed).alpha
            ratio = xdf_error / max(cdf_error, 1e-16)
            if cdf_error < 1e-10:
                better = "exact fit"
            elif ratio >= 1.0:
                better = f"{ratio:.1f}x better"
            else:
                # Near full rank the truncation error is already ~ the
                # ridge scale, so the regularization bias dominates: the
                # ridge trades a small fit penalty for bounded cores
                # (sane alpha). The win to assert is at AGGRESSIVE budgets.
                better = f"{1.0 / ratio:.1f}x worse (ridge bias; X-DF "\
                         "already near-exact here)"
            if xdf_error > 1e-2 * eri_norm:
                wins.append(cdf_error <= xdf_error * 1.001 + 1e-12)
            print(f"    {leaves:3d} leaves: X-DF error {xdf_error:.2e}  "
                  f"RC-DF error {cdf_error:.2e}  ({better}), "
                  f"RC-DF alpha = {cdf_alpha:.4f}")
        # Reported, not asserted: L-BFGS guarantees descent of the
        # REGULARIZED objective, not of the raw fit error, so the win is
        # empirical (robust in practice at significant truncation error,
        # but a hard exit here could misfire on someone else's molecule).
        if wins and all(wins):
            if verbose:
                print("  [check] RC-DF fits at least as well wherever "
                      "truncation error is still significant ... OK")
        elif wins:
            print("  [note] RC-DF trailed truncated X-DF at a "
                  "significant-error budget -- the ridge trades fit for "
                  "bounded cores; try a smaller regularization.")
        else:
            print("  (no budget in the significant-error regime; "
                  "RC-DF win not reported)")
    else:
        print("\n  RC-DF comparison skipped at this size (the L-BFGS "
              "optimization is the expensive path; see "
              "examples/double_factorization/).")

    # --- circuit story (size-gated) ---------------------------------------
    mode = config["mode"]
    circuits = mode == "full" or (mode == "flagged" and force_circuits)
    if rotation_backend in ("qrom", "both"):
        report_programmed_frames(one_body,
                                 eri,
                                 factorized,
                                 core_energy,
                                 angle_bits,
                                 circuits,
                                 verbose=verbose)
    if rotation_backend == "qrom":
        return
    total_qubits = num_system + factorized.num_ancilla
    if mode == "classical" or (mode == "flagged" and not force_circuits):
        amplitudes = f"2^{total_qubits}"
        print(f"\n  Circuit execution skipped: {num_system} system + "
              f"{factorized.num_ancilla} ancilla = {total_qubits} qubits "
              f"({amplitudes} amplitudes per state).")
        if mode == "flagged":
            print("  Rerun with --circuits to run them anyway "
                  "(hours on a typical CPU).")
        print("  The classical preprocessing above is the part that scales;"
              " the statevector simulator is the part that does not.")
        return

    print(
        f"\n  Fixed circuits: {num_system} system + {factorized.num_ancilla} "
        f"index + 0 angle + 0 ladder = {total_qubits} qubits; "
        "controlled forms add one")
    dim = 1 << num_system
    hamiltonian = reference_hamiltonian(one_body, eri)
    rng = np.random.default_rng(7)
    ket = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
    ket = (ket / np.linalg.norm(ket)).astype(np.complex128)

    # The encoded block <0|U|0> must equal H/alpha, on a random state,
    # against the independent sparse Jordan-Wigner reference.
    state = np.array(
        cudaq.get_state(factorized.encode_kernel(), state_from(ket)))
    block = state[:dim]
    expected = (hamiltonian @ ket) / factorized.alpha
    block_error = float(np.max(np.abs(block - expected)))
    if verbose or not np.isfinite(block_error) or block_error >= 1e-10:
        print(f"  encoded block vs sparse JW reference: max |diff| = "
              f"{block_error:.2e}")
    check("<0|U|0> == H/alpha", block_error < 1e-10, verbose=verbose)

    print("\n  Even Chebyshev moments <T_k(H/alpha)> through Walk (same"
          "\n  consumer, either encoding; alphas differ, H is the same):")
    for order in (0, 2, 4):
        measured_flat = Walk(flat).moment(ket, order)
        measured_df = Walk(factorized).moment(ket, order)
        exact_flat = chebyshev_moment(hamiltonian / flat.alpha, ket, order)
        exact_df = chebyshev_moment(hamiltonian / factorized.alpha, ket, order)
        print(f"    T_{order}:  PauliLCU {measured_flat:+.8f} "
              f"(exact {exact_flat:+.8f})   DF {measured_df:+.8f} "
              f"(exact {exact_df:+.8f})")
        check(f"T_{order} moments match the reference",
              abs(measured_flat - exact_flat) < 1e-8
              and abs(measured_df - exact_df) < 1e-8,
              verbose=verbose)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="configurations:\n" +
        "\n".join(f"  {key:10s} {config['label']}"
                  for key, config in CONFIGS.items()))
    parser.add_argument("config",
                        nargs="?",
                        default="h2",
                        choices=sorted(CONFIGS),
                        help="molecular configuration (default: h2)")
    parser.add_argument("--circuits",
                        action="store_true",
                        help="run circuits on boundary-size configurations "
                        "(lih; hours on a typical CPU)")
    parser.add_argument("--rotation-backend",
                        choices=("fixed", "qrom", "both"),
                        default="both",
                        help="DF rotation backend (default: both)")
    parser.add_argument("--angle-bits",
                        type=int,
                        default=4,
                        help="QROM full-turn angle bits (default: 4)")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show each validation check (failures are always reported)")
    arguments = parser.parse_args()
    run(arguments.config,
        arguments.circuits,
        arguments.rotation_backend,
        arguments.angle_bits,
        verbose=arguments.verbose)


if __name__ == "__main__":
    main()
