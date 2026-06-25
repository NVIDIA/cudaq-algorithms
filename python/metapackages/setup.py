# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import ctypes
import glob
import os
import site
import sys
import tomllib
from setuptools import setup
from typing import Optional

package_name = "cudaq-algorithms"


def _log(msg: str) -> None:
    sys.stdout.write(f"[{package_name}] {msg}\n")
    sys.stdout.flush()


def _get_version_from_library(
    libnames: list[str],
    funcname: str,
    nvrtc: bool = False,
) -> Optional[int]:
    """Returns the library version from a list of candidate libraries."""

    for libname in libnames:
        try:
            _log(f"Looking for library: {libname}")
            runtime_so = ctypes.CDLL(libname)
            break
        except Exception as e:
            _log(f"Failed to open {libname}: {e}")
    else:
        _log("No more candidate libraries to try")
        return None

    func = getattr(runtime_so, funcname, None)
    if func is None:
        raise Exception(f"{libname}: {funcname} could not be found")
    func.restype = ctypes.c_int

    if nvrtc:
        func.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        major = ctypes.c_int()
        minor = ctypes.c_int()
        retval = func(major, minor)
        version = major.value * 1000 + minor.value * 10
    else:
        func.argtypes = [ctypes.POINTER(ctypes.c_int)]
        version_ref = ctypes.c_int()
        retval = func(version_ref)
        version = version_ref.value

    if retval != 0:
        raise Exception(f"{libname}: {funcname} returned error: {retval}")
    _log(f"Detected version: {version}")
    return version


def _get_cuda_version() -> Optional[int]:
    """Returns the detected CUDA version or None."""

    libnames = [
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "libnvrtc.so.11.2",
        "libnvrtc.so.11.1",
        "libnvrtc.so.11.0",
    ]
    _log(f"Trying to detect CUDA version from libraries: {libnames}")
    try:
        version = _get_version_from_library(libnames, "nvrtcVersion", True)
    except Exception as e:
        _log(f"Error: {e}")
    else:
        if version is not None:
            _log("Autodetection succeeded")
            return version

    libnames = [
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11.0",
    ]
    _log(f"Trying to detect CUDA version from libraries: {libnames}")
    try:
        version = _get_version_from_library(libnames, "cudaRuntimeGetVersion")
    except Exception as e:
        _log(f"Error: {e}")
    else:
        if version is not None:
            _log("Autodetection succeeded")
            return version

    try:
        _log("Trying to detect CUDA version using NVIDIA Management Library")
        from pynvml import nvmlInit, nvmlSystemGetCudaDriverVersion
        nvmlInit()
        version = nvmlSystemGetCudaDriverVersion()
    except Exception as e:
        _log(f"Error: {e}")
    else:
        _log(f"Detected version: {version}")
        _log("Autodetection succeeded")
        return version

    _log("Autodetection failed")
    return None


def _check_package_installed(pkg_name: str) -> bool:
    normalized_package_name = pkg_name.replace("-", "_")

    def _check_in_directory(directory: str) -> bool:
        search_pattern = os.path.join(
            directory, f"{normalized_package_name}-*.dist-info")
        matches = glob.glob(search_pattern)
        if matches:
            _log(f"Found matches for {pkg_name} in {directory}:")
            for match in matches:
                _log(match)
            return True
        return False

    user_site_packages = site.getusersitepackages()
    if os.path.exists(user_site_packages) and _check_in_directory(
            user_site_packages):
        return True

    for directory in site.getsitepackages():
        if _check_in_directory(directory):
            return True

    _log(f"No matches found for {pkg_name}")
    return False


def _infer_best_package() -> str:
    installed = []
    for pkg in [package_name, f"{package_name}-cu12", f"{package_name}-cu13"]:
        _log(f"Looking for existing installation of {pkg}.")
        if _check_package_installed(pkg):
            installed.append(pkg)

    cuda_version = _get_cuda_version()
    if cuda_version is None:
        _log("CUDA version not detected. Assuming CUDA 12.0.")
        concrete_package = f"{package_name}-cu12"
    elif cuda_version < 12000:
        raise Exception(
            f"Your CUDA version ({cuda_version}) is not supported. Minimum required: CUDA 12.0."
        )
    elif cuda_version < 13000:
        concrete_package = f"{package_name}-cu12"
    elif cuda_version <= 14000:
        concrete_package = f"{package_name}-cu13"
    else:
        raise Exception(f"Your CUDA version ({cuda_version}) is too new.")

    _log(f"Identified {concrete_package} as the best package.")
    conflicting = ", ".join(pkg for pkg in installed
                            if pkg != concrete_package)
    if conflicting:
        raise Exception(
            f"You have a conflicting {package_name} version installed. "
            f"Please remove the following package(s): {conflicting}")
    return concrete_package


setup_dir = os.path.dirname(os.path.abspath(__file__))
install_requires = []
opt_deps = {}

if os.environ.get("CUDAQ_META_WHEEL_BUILD", "0") != "1":
    with open(os.path.join(setup_dir, "_version.txt"), "r",
              encoding="utf-8") as f:
        version = f.read().strip()

    best_package = _infer_best_package()
    install_requires = [f"{best_package}=={version}"]

    suffix = None
    if "cu12" in best_package:
        suffix = "cu12"
    elif "cu13" in best_package:
        suffix = "cu13"

    with open(os.path.join(setup_dir, f"pyproject.toml.{suffix}"), "rb") as f:
        data = tomllib.load(f)
        opt_deps = data.get("project", {}).get("optional-dependencies", {})
    _log(f"Optional dependencies: {opt_deps}")

setup(
    zip_safe=False,
    install_requires=install_requires,
    extras_require=opt_deps,
)
