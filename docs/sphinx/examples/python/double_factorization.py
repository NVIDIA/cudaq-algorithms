# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit (X-DF) and compressed (C-DF) double factorization of the two-electron
integrals, following Cohn, Motta, and Parrish, PRX Quantum 2, 040352 (2021).

Generates restricted Hartree-Fock molecular-orbital integrals for H2O/STO-3G with
PySCF, then double-factorizes the ERI tensor two ways:

  * X-DF: the exact factorization (rank-one cores) via pivoted Cholesky of
    the ERI supermatrix (the default), truncated by residual pivot.
  * C-DF: a least-squares-optimized factorization that reaches comparable
    accuracy with far fewer leaves.

Heavy linear algebra uses the NVIDIA math libraries (cuSOLVER/cuBLAS via CuPy)
when a GPU is available; otherwise it falls back to NumPy/SciPy.
"""
from __future__ import annotations

import numpy as np

import cudaq_algorithms as algorithms

df = algorithms.double_factorization


def water_mo_eri():
    from pyscf import ao2mo, gto, scf
    mol = gto.M(atom="O 0 0 0; H 0 0 0.957; H 0 0.926 -0.24",
                basis="sto-3g",
                verbose=0)
    mf = scf.RHF(mol).run()
    n = mf.mo_coeff.shape[1]
    return np.asarray(ao2mo.restore("s1", ao2mo.kernel(mol, mf.mo_coeff), n))


def main():
    backend = "auto"
    _, backend_name = df.resolve_backend(backend)
    eri = water_mo_eri()
    norm = np.linalg.norm(eri)
    print("Double factorization of H2O/STO-3G two-electron integrals")
    print(f"  backend: {backend_name}")
    print(f"  orbitals: {eri.shape[0]}   ||eri||_F: {norm:.6f}")

    full = df.explicit_double_factorization(eri,
                                            threshold=0.0,
                                            backend=backend)
    print("\nX-DF (explicit):")
    print(f"  full-rank leaves: {full.num_leaves}   "
          f"reconstruction error: {df.factorization_error(eri, full):.3e}")
    for threshold in (1.0e-2, 1.0e-3, 1.0e-4):
        truncated = df.explicit_double_factorization(eri,
                                                     threshold=threshold,
                                                     backend=backend)
        rel = df.factorization_error(eri, truncated) / norm
        print(
            f"  threshold {threshold:.0e}: leaves={truncated.num_leaves:2d}  "
            f"rel error={rel:.3e}")

    print("\nC-DF (compressed) vs X-DF at equal leaf count:")
    for num_leaves in (2, 4):
        explicit = df.explicit_double_factorization(eri,
                                                    threshold=0.0,
                                                    max_num_leaves=num_leaves,
                                                    backend=backend)
        compressed = df.compressed_double_factorization(eri,
                                                        num_leaves=num_leaves,
                                                        max_iterations=600,
                                                        backend=backend)
        x_rel = df.factorization_error(eri, explicit) / norm
        c_rel = df.factorization_error(eri, compressed) / norm
        print(f"  leaves={num_leaves:2d}:  X-DF rel error={x_rel:.3e}   "
              f"C-DF rel error={c_rel:.3e}")
        assert c_rel <= x_rel + 1.0e-6


if __name__ == "__main__":
    main()
