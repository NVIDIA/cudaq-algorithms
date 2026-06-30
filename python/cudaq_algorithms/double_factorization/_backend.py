# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Array-backend selection for double factorization.

Prefers the NVIDIA math libraries (cuSOLVER / cuBLAS via CuPy) when a GPU is
available, and falls back to NumPy otherwise. All public double-factorization
APIs accept a ``backend`` argument of ``"auto"`` (default), ``"cupy"``, or
``"numpy"``.
"""
from __future__ import annotations

import numpy as np

try:  # CuPy is optional; absence simply forces the NumPy path.
    import cupy as _cupy  # type: ignore
except Exception:  # pragma: no cover - environment without CuPy
    _cupy = None

# Empirical CPU/GPU crossovers (orbital count ``n``) for ``backend="auto"``.
# Below these the GPU's per-kernel launch + host-sync latency makes CuPy slower
# than NumPy; above them the batched cuSOLVER/cuBLAS linear algebra wins and the
# lead grows with ``n``. X-DF (a cheap Cholesky loop + tiny per-leaf eigh) stays
# CPU-favorable far longer than the contraction-heavy C-DF optimization. See
# benchmarks/double_factorization/ and DOUBLE_FACTORIZATION_CPP_PORT_NOTES.md.
AUTO_GPU_MIN_ORBITALS_COMPRESSED = 18
AUTO_GPU_MIN_ORBITALS_EXPLICIT = 56


def cupy_gpu_available() -> bool:
    """Return True when CuPy is importable and at least one GPU is visible."""
    if _cupy is None:
        return False
    try:
        return _cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover - driver/runtime issues
        return False


def resolve_backend(backend: str = "auto", problem_size=None, gpu_min_size=0):
    """Return ``(array_module, name)`` for the requested backend.

    ``"auto"`` selects CuPy when a GPU is available *and* the problem is large
    enough to amortize GPU launch/sync overhead -- i.e. ``problem_size`` (the
    orbital count ``n``) is at least ``gpu_min_size`` -- otherwise NumPy. With no
    ``problem_size`` hint it keeps the legacy behavior: CuPy whenever a GPU is
    present. ``"cupy"`` / ``"numpy"`` force the backend regardless of size.
    """
    if backend == "numpy":
        return np, "numpy"
    if backend == "cupy":
        if not cupy_gpu_available():
            raise RuntimeError(
                "double_factorization error - the 'cupy' backend was requested "
                "but no CuPy/GPU is available.")
        return _cupy, "cupy"
    if backend == "auto":
        if cupy_gpu_available() and (problem_size is None
                                     or problem_size >= gpu_min_size):
            return _cupy, "cupy"
        return np, "numpy"
    raise ValueError(
        f"double_factorization error - unknown backend '{backend}'; expected "
        "'auto', 'cupy', or 'numpy'.")


def to_device(array, xp):
    """Move a host/device array onto the backend ``xp``."""
    if xp is np:
        return np.asarray(array)
    return xp.asarray(array)


def to_numpy(array) -> np.ndarray:
    """Return a host NumPy copy of a NumPy or CuPy array."""
    if _cupy is not None and isinstance(array, _cupy.ndarray):
        return _cupy.asnumpy(array)
    return np.asarray(array)


def expm_skew_symmetric(generator, xp):
    """Matrix exponential of a real antisymmetric matrix, returning an orthogonal
    matrix. Uses a Hermitian eigendecomposition (cuSOLVER on the CuPy backend),
    avoiding any dependence on a general matrix-exponential routine."""
    generator = xp.asarray(generator)
    # i * X is Hermitian for real antisymmetric X, so eigh applies.
    hermitian = 1j * generator
    eigenvalues, eigenvectors = xp.linalg.eigh(hermitian)
    phases = xp.exp(-1j * eigenvalues)
    rotated = (eigenvectors * phases) @ eigenvectors.conj().T
    return xp.real(rotated)
