# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``encode_sparse`` report carried by the encodings it builds."""

from __future__ import annotations


def report_summary(report: dict | None) -> str:
    """One-line ``repr`` suffix for an ``encode_sparse`` report (empty when
    the encoding was built directly)."""
    if report is None:
        return ""
    return (f" [encode_sparse: alpha={report['alpha']:.6g}, "
            f"nnz={report['nnz']}, max_row_nnz={report['max_row_nnz']}, "
            f"slots={report['slots']}, qubits={report['qubits']}" +
            (", dilated" if report["dilated"] else "") + "]")
