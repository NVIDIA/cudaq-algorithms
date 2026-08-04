# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A user-written block encoding on real molecules, vs the built-in PauliLCU.

The encoding here is user-level code: ``df_encoding.py`` next door
implements ``DoubleFactorizedEncoding`` against the public
``BlockEncoding`` protocol. Because the protocol is structural, the same
``Walk`` consumer drives the user-written encoding and the built-in
``PauliLCU`` identically.

Integrals come from PySCF (``pip install pyscf``) via
``chemistry.from_pyscf``, for a menu of molecular configurations::

    python3 df_block_encoding.py [config] [--circuits]

    h2         H2 / STO-3G           2 orbitals ->  4 system qubits  (default)
    h2o-cas44  H2O / STO-3G CAS(4,4) 4 orbitals ->  8 system qubits
    h4         linear H4 / STO-3G    4 orbitals ->  8 system qubits
    lih        LiH / STO-3G          6 orbitals -> 12 system qubits
    h2o        H2O / STO-3G          7 orbitals -> 14 system qubits
    h2o-631g   H2O / 6-31G          13 orbitals -> 26 system qubits

Every configuration tells the classical story: double-factorize the ERI,
compare the DF encoding's normalization ``alpha`` and term count against
the flat Pauli-expansion ``PauliLCU`` baseline, sweep the truncation dial
(fewer leaves -> fewer terms; alpha typically, though not monotonically,
shrinks -- see the note in the sweep. The flat expansion has no such
knob), and then re-optimize the kept leaves with RC-DF at the same
budgets (``compressed_double_factorization`` with a small ridge) -- the
second dial: optimize, don't just truncate. Small configurations also
run the circuits: the encoded block is checked against a sparse
Jordan-Wigner reference Hamiltonian, and the same ``Walk`` consumer
measures Chebyshev moments through both encodings. Larger configurations
skip circuit execution (statevector cost is printed)
-- the classical preprocessing scales; the simulator is what does not.
``lih`` sits on the boundary: pass ``--circuits`` to run it anyway
(hours on a typical CPU).

Runs on the CPU statevector simulator; no compiled extension needed.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import scipy.sparse as sp

import cudaq

from cudaq_algorithms import PauliLCU, Walk, chemistry, state_from
from cudaq_algorithms import double_factorization as df

# The encoding under demonstration is a sibling example file, not a package
# module (CUDA-Q kernels need real .py files, so a plain import is fine).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from df_encoding import DoubleFactorizedEncoding

cudaq.set_target("qpp-cpu")  # fp64: the checks below assert to ~1e-10

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


def check(label: str, condition: bool):
    print(f"  [check] {label} ... {'OK' if condition else 'FAILED'}")
    if not condition:
        sys.exit(1)


# ----------------------------------------------------------------------
# The story, per configuration
# ----------------------------------------------------------------------


def run(key: str, force_circuits: bool):
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
          all(a >= b - 1e-9 for a, b in zip(errors, errors[1:])))
    check("full-rank X-DF reconstructs the ERI exactly", errors[-1] < 1e-8)

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
        wins = []
        for leaves in budgets:
            truncated = df.explicit_double_factorization(eri,
                                                         max_num_leaves=leaves)
            xdf_error = df.factorization_error(eri, truncated)
            compressed = df.compressed_double_factorization(
                eri,
                num_leaves=leaves,
                regularization=1e-4,
                max_iterations=300)
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
            if xdf_error > 1e-2:
                wins.append(cdf_error <= xdf_error * 1.001 + 1e-12)
            print(f"    {leaves:3d} leaves: X-DF error {xdf_error:.2e}  "
                  f"RC-DF error {cdf_error:.2e}  ({better}), "
                  f"RC-DF alpha = {cdf_alpha:.4f}")
        if wins:
            check(
                "RC-DF fits at least as well wherever truncation error is "
                "still significant", all(wins))
        else:
            print("  (no budget in the significant-error regime; "
                  "RC-DF win not asserted)")
    else:
        print("\n  RC-DF comparison skipped at this size (the L-BFGS "
              "optimization is the expensive path; see "
              "examples/double_factorization/).")

    # --- circuit story (size-gated) ---------------------------------------
    mode = config["mode"]
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

    print(f"\n  Circuits: {num_system} system + {factorized.num_ancilla} "
          f"ancilla = {total_qubits} qubits")
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
    print(f"  encoded block vs sparse JW reference: max |diff| = "
          f"{block_error:.2e}")
    check("<0|U|0> == H/alpha", block_error < 1e-10)

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
        check(
            f"T_{order} moments match the reference",
            abs(measured_flat - exact_flat) < 1e-8
            and abs(measured_df - exact_df) < 1e-8)


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
    arguments = parser.parse_args()
    run(arguments.config, arguments.circuits)


if __name__ == "__main__":
    main()
