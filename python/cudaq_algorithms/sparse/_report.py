# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``encode_sparse`` pricing report carried by the encodings it builds."""

from __future__ import annotations


def report_summary(report: dict | None) -> str:
    """One-line ``repr`` suffix for an ``encode_sparse`` report (empty when
    the encoding was built directly)."""
    if report is None:
        return ""
    parts = []
    for path in ("oracle", "lcu"):
        if path == report["path"]:
            continue
        alpha = report["alpha_alternatives"][path]
        note = " (ineligible)" if report["ineligible"][path] else ""
        parts.append(f"{path}: alpha={alpha:.6g}{note}")
    return (f" [encode_sparse: path={report['path']!r}, "
            f"alpha={report['alpha']:.6g}, nnz={report['nnz']}, "
            f"max_row_nnz={report['max_row_nnz']}, slots={report['slots']}" +
            (", dilated" if report["dilated"] else "") + "; vs " +
            "; ".join(parts) + "]")
