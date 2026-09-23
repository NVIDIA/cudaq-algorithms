# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the separately scored scientific coverage-completion campaign.

Usage: extended.py prepare|preflight|run|report --output PATH.
The historical and natural-language campaign registries remain unchanged.
"""
import classical_cases
import extended_classical
import extended_stateprep
import run

SUITE = "coverage_completion"


def case_registry():
    cases = {
        spec["id"]: (module, spec)
        for module in (extended_stateprep, extended_classical)
        for spec in module.case_specs()
    }
    psi4 = next(spec for spec in classical_cases.case_specs()
                if spec["id"] == "psi4_energy")
    cases[psi4["id"]] = (classical_cases, psi4)
    return cases


def main(argv=None):
    original, label = run.CASES, run.SUITE
    try:
        run.CASES, run.SUITE = case_registry(), SUITE
        run.main(argv)
    finally:
        run.CASES, run.SUITE = original, label


if __name__ == "__main__":
    main()
