#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                        #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# [Begin Documentation]
"""Example 4 — Double factorization and the BlockEncoding protocol.

Three ideas, built on real molecules from PySCF:

  A. The compression dial (H2). Double factorization rewrites the
     two-electron tensor as a sum of low-rank "leaves". Keep fewer leaves
     and you trade accuracy for a cheaper block encoding -- a knob you can
     watch, checked against the exact FCI energy at every setting.

  B. It scales, and DF is compact (N2, 20 qubits). The same pipeline runs
     at 20 qubits. There, a flat Pauli LCU block-encodes the Hamiltonian
     as a SELECT over ~3000 Pauli strings; double factorization encodes
     the *same* operator with a few dozen structured leaves. Far fewer
     building blocks, comparable normalization.

  C. Polymorphism through the protocol. `PauliLCU` and
     `DoubleFactorizedEncoding` are unrelated classes, but both satisfy
     `BlockEncoding` -- so `Walk` and `QSVT` consume either with identical
     code.

Prerequisite: PySCF (`pip install pyscf`). Takes a few seconds (the N2
FCI reference is the slow part). Run: python3 04_double_factorization_and_the_protocol.py
"""

from __future__ import annotations

import os
import pathlib
import sys

import cudaq

from cudaq_algorithms import (BlockEncoding, PauliLCU, QSVT, Walk, chemistry)
from cudaq_algorithms import double_factorization as df

# The double-factorized encoding is the worked BlockEncoding example that
# lives beside this script (CUDA-Q kernels need real .py files, so a plain
# path-based import works).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from df_encoding import DoubleFactorizedEncoding


def mean_field(atom: str, basis: str = "sto-3g"):
    from pyscf import gto, scf

    mol = gto.M(atom=atom, basis=basis, symmetry=False)
    return mol, scf.RHF(mol).run(verbose=0)


def fci_energy(one_body, eri, num_electrons, nuclear_repulsion) -> float:
    """Exact (FCI) total energy of the given MO integrals, via PySCF."""
    from pyscf import fci

    electronic, _ = fci.direct_spin1.kernel(one_body, eri, one_body.shape[0],
                                            num_electrons)
    return float(electronic) + nuclear_repulsion


def main() -> int:
    cudaq.set_target(os.environ.get("CUDAQ_DEFAULT_SIMULATOR", "qpp-cpu"))

    try:
        import pyscf  # noqa: F401
    except ImportError:
        print("This example needs PySCF:  pip install pyscf")
        return 0

    # -- A. The compression dial (H2) ------------------------------------
    mol, mf = mean_field("H 0 0 0; H 0 0 0.7414")
    one_body, eri, e_nuc = chemistry.from_pyscf(mf)
    exact = fci_energy(one_body, eri, mol.nelectron, e_nuc)
    full = df.explicit_double_factorization(eri, threshold=0.0)
    print(f"A. H2/STO-3G compression dial (exact FCI = {exact:.6f} Ha)")
    print(f"   {'leaves':>6} {'tensor error':>13} {'alpha':>8} "
          f"{'E error (Ha)':>13}")
    for num_leaves in range(1, full.num_leaves + 1):
        truncated = df.explicit_double_factorization(eri,
                                                     max_num_leaves=num_leaves)
        reconstructed_eri = df.reconstruct_eri(truncated)
        alpha = PauliLCU(
            chemistry.qubit_hamiltonian(one_body,
                                        reconstructed_eri,
                                        scalar_offset=e_nuc)).alpha
        energy = fci_energy(one_body, reconstructed_eri, mol.nelectron, e_nuc)
        print(
            f"   {num_leaves:>6} {df.factorization_error(eri, truncated):>13.2e}"
            f" {alpha:>8.4f} {energy - exact:>+13.2e}")

    # -- B. It scales, and DF is compact (N2, 20 qubits) -----------------
    mol, mf = mean_field("N 0 0 0; N 0 0 1.09")
    one_body, eri, e_nuc = chemistry.from_pyscf(mf)
    hamiltonian = chemistry.qubit_hamiltonian(one_body,
                                              eri,
                                              scalar_offset=e_nuc)
    flat = PauliLCU(hamiltonian)
    factorization = df.explicit_double_factorization(eri, threshold=0.0)
    dfe = DoubleFactorizedEncoding(one_body,
                                   factorization,
                                   scalar_offset=e_nuc)
    print(f"\nB. N2/STO-3G on {hamiltonian.qubit_count} qubits: "
          f"same Hamiltonian, two block encodings")
    print(f"   flat PauliLCU : alpha = {flat.alpha:6.2f}  over "
          f"{hamiltonian.term_count} Pauli terms")
    print(f"   double factn  : alpha = {dfe.alpha:6.2f}  over "
          f"{factorization.num_leaves} leaves")
    print("   -> DF encodes the same operator with ~50x fewer SELECT building "
          "blocks,\n      at a comparable one-norm (in a minimal basis the DF "
          "rank is near\n      full, so alpha barely drops; larger bases "
          "compress far more).")

    # -- C. One protocol, two encodings ----------------------------------
    print(
        "\nC. both are BlockEncodings, so Walk/QSVT consume either unchanged")
    for name, encoding in [("PauliLCU", flat),
                           ("DoubleFactorizedEncoding", dfe)]:
        assert isinstance(encoding, BlockEncoding)
        Walk(encoding)
        QSVT(encoding)  # the exact same construction works on both
        print(f"   {name:26s} system={encoding.num_system} "
              f"ancilla={encoding.num_ancilla} -> Walk + QSVT built")

    print("\nOK — pyscf integrals in, block encodings out; DF is the compact, "
          "scalable one, and both plug into the same primitives (example 6 "
          "shows how to add your own).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
