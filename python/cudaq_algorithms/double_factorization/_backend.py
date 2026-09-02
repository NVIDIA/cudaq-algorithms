# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Array-backend selection for double factorization.

Prefers the NVIDIA math libraries (cuSOLVER / cuBLAS via CuPy) when a GPU is
available, and falls back to NumPy otherwise. All public double-factorization
APIs accept a ``backend`` argument of ``"auto"`` (default), ``"cupy"``, or
``"numpy"``.
"""
from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

# The array namespace used for a computation: the ``numpy`` module or, when
# a GPU is available, ``cupy``. CuPy is optional, so there is no importable
# static type covering both namespaces.
ArrayModule: TypeAlias = Any

# An array living on the active backend: ``numpy.ndarray`` on the NumPy
# backend, ``cupy.ndarray`` on the CuPy backend.
DeviceArray: TypeAlias = Any

try:  # CuPy is optional; absence simply forces the NumPy path.
    import cupy as _cupy  # type: ignore
except Exception:  # pragma: no cover - environment without CuPy
    _cupy = None

# Empirical CPU/GPU crossovers (orbital count ``n``) for ``backend="auto"``.
# Below these the GPU's per-kernel launch + host-sync latency makes CuPy slower
# than NumPy; above them the batched cuSOLVER/cuBLAS linear algebra wins and the
# lead grows with ``n``. X-DF (a cheap Cholesky loop + tiny per-leaf eigh) stays
# CPU-favorable far longer than the contraction-heavy C-DF optimization. See
# benchmarks/double_factorization/.
AUTO_GPU_MIN_ORBITALS_COMPRESSED = 18
AUTO_GPU_MIN_ORBITALS_EXPLICIT = 56


def cupy_gpu_available() -> bool:
    """Return True when CuPy can run a kernel on a visible GPU.

    ``getDeviceCount`` only talks to the driver, so a driver-only install
    (no NVRTC / CUDA toolkit) would otherwise look usable and then fail on
    the first compiled kernel. The tiny ``arange`` forces that compile.
    """
    if _cupy is None:
        return False
    try:
        if _cupy.cuda.runtime.getDeviceCount() <= 0:
            return False
        _cupy.arange(1) + 1
        return True
    except Exception:  # pragma: no cover - driver/runtime/NVRTC issues
        return False


def resolve_backend(backend: str = "auto",
                    problem_size: int | None = None,
                    gpu_min_size: int = 0) -> tuple[ArrayModule, str]:
    """Return ``(array_module, name)`` for the requested backend.

    ``"auto"`` selects CuPy when the problem is large enough to amortize GPU
    launch/sync overhead -- i.e. ``problem_size`` (the orbital count ``n``) is
    at least ``gpu_min_size`` -- *and* a GPU is available; otherwise NumPy. The
    size check is evaluated first so a below-threshold problem does not create
    a CUDA context. With no ``problem_size`` hint it keeps the legacy behavior:
    CuPy whenever a GPU is present. ``"cupy"`` / ``"numpy"`` force the backend
    regardless of size.
    """
    if backend == "numpy":
        return np, "numpy"
    if backend == "cupy":
        if not cupy_gpu_available():
            raise RuntimeError("the 'cupy' backend was requested but no "
                               "CuPy/GPU is available.")
        return _cupy, "cupy"
    if backend == "auto":
        if ((problem_size is None or problem_size >= gpu_min_size)
                and cupy_gpu_available()):
            return _cupy, "cupy"
        return np, "numpy"
    raise ValueError(f"unknown backend '{backend}'; expected 'auto', 'cupy', "
                     "or 'numpy'.")


def to_device(array: ArrayLike, xp: ArrayModule) -> DeviceArray:
    """Move a host/device array onto the backend ``xp``."""
    if xp is np:
        return np.asarray(array)
    return xp.asarray(array)


def to_numpy(array) -> np.ndarray:
    """Return a host NumPy copy of a NumPy or CuPy array."""
    if _cupy is not None and isinstance(array, _cupy.ndarray):
        return _cupy.asnumpy(array)
    return np.asarray(array)


def expm_skew_symmetric_batched(generators: ArrayLike,
                                xp: ArrayModule) -> DeviceArray:
    """Batched matrix exponential over a stack of real antisymmetric
    matrices ``generators`` (shape ``(num_leaves, n, n)``), returning
    orthogonal matrices. ``i * X`` is Hermitian for real antisymmetric
    ``X``, so one batched Hermitian eigendecomposition covers the whole
    stack -- on the CuPy backend a single cuSOLVER call rather than a
    Python-driven loop, with no dependence on a general
    matrix-exponential routine.

    Stacked ``eigh`` on current CuPy handles matrix dimensions well
    beyond cuSOLVER's batched-Jacobi 32 limit (verified in review on
    CuPy 13.6 / CUDA 12.9 up to n = 64, matching ``scipy.linalg.expm``
    to ~1e-12), so no per-leaf fallback is needed here."""
    generators = xp.asarray(generators)
    hermitian = 1j * generators
    eigenvalues, eigenvectors = xp.linalg.eigh(
        hermitian)  # batched over axis 0
    phases = xp.exp(-1j * eigenvalues)  # (num_leaves, n)
    # Scale each leaf's eigenvector columns by its phases, then V diag(phase) V^H.
    scaled = eigenvectors * phases[:, None, :]
    rotated = scaled @ xp.conj(eigenvectors).transpose(0, 2, 1)
    return xp.real(rotated)
