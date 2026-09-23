# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent numeric comparison and a subprocess application tracer.

The trace establishes observed package calls, not adversarial proof of data
dependency. Numerical references remain in the controller, outside the sandbox.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import runpy
import sys
import stat
import zipfile

import numpy as np


def read_regular(path, limit=50_000_000):
    """No symlinks, devices, blocking FIFOs, or unbounded parent-side reads."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("artifact must be a bounded regular file")
        content = handle.read(limit + 1)
        if len(content) > limit:
            raise ValueError("artifact exceeded its byte limit")
        return content


def compare_output(path, expected, spec):
    references = {key: np.asarray(value) for key, value in expected.items()}
    if set(references) != set(spec["outputs"]):
        raise ValueError("reference fields do not match the task contract")
    if any(not np.isfinite(value).all() for value in references.values()):
        raise ValueError("nonfinite reference")
    result = {"passed": False, "fields": {}}
    try:
        content = read_regular(path)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 50_000_000:
                raise ValueError("expanded output exceeds 50 MB")
        with np.load(io.BytesIO(content), allow_pickle=False) as loaded:
            for name, reference in references.items():
                if name not in loaded:
                    result["fields"][name] = {
                        "passed": False,
                        "reason": "missing field"
                    }
                    continue
                actual = np.asarray(loaded[name])
                if actual.shape != reference.shape or not np.isfinite(
                        actual).all():
                    result["fields"][name] = {
                        "passed": False,
                        "reason": "shape mismatch or nonfinite output"
                    }
                    continue
                if name in spec.get("phase_invariant_fields", []):
                    overlap = np.vdot(reference, actual)
                    if abs(overlap) > 0:
                        actual = actual * np.conj(overlap) / abs(overlap)
                result["fields"][name] = {
                    "passed":
                    bool(
                        np.allclose(actual,
                                    reference,
                                    atol=spec["atol"],
                                    rtol=spec["rtol"])),
                    "max_abs_error":
                    float(np.max(abs(actual -
                                     reference))) if actual.size else 0.0,
                    "shape":
                    list(actual.shape),
                }
    except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile,
            KeyError, RuntimeError) as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        return result
    result["passed"] = bool(result["fields"]) and all(
        v["passed"] for v in result["fields"].values())
    return result


def required_calls_seen(required, observed):
    missing = sorted(set(required) - set(observed))
    return {"passed": not missing, "missing": missing}


def trace_application(app, input_path, output_path, trace_path, package_root):
    """Run an unmodified real application; record trusted package frame names."""
    root = Path(package_root).resolve()
    calls = set()
    kernels = set()
    pending_kernels = {}

    def profile(frame, event, arg):
        if event not in ("call", "return", "c_call"):
            return
        if event == "call":
            module = frame.f_globals.get("__name__", "")
            if (module.startswith("cudaq_algorithms")
                    and frame.f_code.co_filename.startswith(str(root) + "/")):
                calls.add(module + "." + frame.f_code.co_qualname)
        # JIT device functions do not execute their Python bodies. Record the
        # name of each real CUDA-Q decorator whose compile method is observed.
        if (event == "call" and frame.f_code.co_name == "compile"
                and frame.f_globals.get("__name__")
                == "cudaq.kernel.kernel_decorator"):
            obj = frame.f_locals.get("self")
            if type(obj).__module__ == "cudaq.kernel.kernel_decorator":
                name = getattr(obj, "name", None)
                module = getattr(obj, "kernelModuleName", None)
                parent = frame.f_back
                while parent:
                    if (parent.f_globals.get("__name__")
                            == "cudaq.runtime.state"
                            and parent.f_code.co_name == "get_state"):
                        if isinstance(name, str) and isinstance(module, str):
                            pending_kernels.setdefault(
                                id(parent), set()).add(module + "." + name)
                    parent = parent.f_back
        if (event == "return"
                and frame.f_globals.get("__name__") == "cudaq.runtime.state"
                and frame.f_code.co_name == "get_state" and arg is not None):
            kernels.update(pending_kernels.pop(id(frame), set()))

    old_argv = sys.argv
    sys.argv = [
        str(app), "--input",
        str(input_path), "--output",
        str(output_path)
    ]
    sys.setprofile(profile)
    try:
        runpy.run_path(str(app), run_name="__main__")
    finally:
        sys.setprofile(None)
        sys.argv = old_argv
        Path(trace_path).write_text(
            json.dumps(
                {
                    "calls": sorted(calls),
                    "compiled_kernels": sorted(kernels)
                },
                indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("app", "input", "output", "trace", "package_root"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args()
    trace_application(args.app, args.input, args.output, args.trace,
                      args.package_root)
