# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lean delivery: an executable dense reference early, resolved fetch commands,
no literal placeholders, a lean entry file and unchanged routing decisions.

Written together with the 2026-10-22 patch; not yet executed at the time of
writing (see the patch summary)."""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SKILL = ROOT / "skills/cudaq-algorithms"
SCRIPT = SKILL / "scripts/route.py"
SPEC = importlib.util.spec_from_file_location("route_lean_delivery", SCRIPT)
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)
EVALS = {
    case["id"]: case["prompt"]
    for case in json.loads((SKILL / "evals/evals.json").read_text())["evals"]
}
NUMERICAL = [
    "science-trotter-conservation-versus-accuracy",
    "science-simulation-probabilistic-cooling",
    "science-qsvt-low-energy-filter-choice",
    "science-qsvt-calibration-robustness",
]
CONTRACT_ONLY = [
    "double-factorization-host-contracts",
    "state-preparation-uccsd-open-shell-parity",
    "source-version-drift",
]
PLACEHOLDERS = ("<skill-dir>", "<this skill's directory>")


def cli(prompt, *options):
    return subprocess.run([sys.executable, str(SCRIPT), *options, prompt],
                          check=True, capture_output=True, text=True).stdout


def context(prompt):
    records = list(
        dict.fromkeys(router.route(prompt) + router.focused_records(prompt)))
    return router.reference_context(records, prompt)


@pytest.mark.parametrize("case", NUMERICAL)
def test_numerical_prompts_get_one_executable_dense_reference_early(case):
    output = context(EVALS[case])
    recipes = [
        block for block in re.findall(r"```python\n(.*?)\n```", output,
                                      re.DOTALL)
        if "def dense_pauli_matrix(" in block
    ]
    assert len(recipes) == 1
    namespace = {}
    exec(recipes[0], namespace)  # runs check_dense_ordering(3)
    dense = namespace["dense_pauli_matrix"]
    assert dense([(1.0, "ZI")], 2)[1, 1] == -1  # Z on qubit 0 flips index 1
    assert dense([(1.0, "IZ")], 2)[1, 1] == 1  # not qubit 1
    first_family = re.search(r"\n### /.*?/references/(?!quick-contract)",
                             output)
    assert output.index("### Dense Pauli reference") < first_family.start()
    assert len(output) <= router.MAX_CONTEXT_CHARS


@pytest.mark.parametrize("case", CONTRACT_ONLY)
def test_contract_prompts_do_not_get_the_dense_reference(case):
    assert "### Dense Pauli reference" not in context(EVALS[case])


@pytest.mark.parametrize("case", NUMERICAL + CONTRACT_ONLY)
def test_cli_output_has_no_literal_placeholders(case):
    output = cli(EVALS[case])
    for placeholder in PLACEHOLDERS:
        assert placeholder not in output


def test_fully_omitted_sections_get_a_resolved_fetch_command():
    output = cli(EVALS["science-trotter-conservation-versus-accuracy"])
    assert "## Verification\n\n[Additional blocks omitted" not in output
    commands = re.findall(r"^Complete section: (.+)$", output, re.MULTILINE)
    assert commands
    for command in commands[:3]:
        result = subprocess.run(command, shell=True, check=True,
                                capture_output=True, text=True, cwd="/")
        assert result.stdout.strip()


def test_entry_file_is_lean_and_has_no_templated_command():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    # The revised patch retains mandatory completion sections and targets
    # an entry below 12 KB; the earlier 10.5 KB estimate was not implemented.
    assert len(text.encode("utf-8")) <= 12000
    assert "<skill-dir>" not in text
    assert 'python3 scripts/route.py "<original request>"' in text
    assert "#dense-operator-self-test" in text


def test_participation_clause_present_where_classical_shortcuts_were_allowed():
    assert "names CUDA-Q Algorithms" in (
        SKILL / "references/validation.md").read_text()
    assert "does not name CUDA-Q Algorithms" in (
        SKILL / "references/qsvt/qsvt.md").read_text()
    assert "names CUDA-Q Algorithms" in (
        SKILL / "references/trotter/trotter.md").read_text()


def test_routing_decisions_match_the_frozen_control_router():
    frozen = (ROOT / ".venv-nvskills/decision-tree-ablation-20261021/"
              "variants/control/scripts/route.py")
    if not frozen.is_file():
        pytest.skip("frozen control router not present")
    spec = importlib.util.spec_from_file_location("frozen_control_router",
                                                  frozen)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    for prompt in EVALS.values():
        assert router.route(prompt) == control.route(prompt)
        assert router.focused_records(prompt) == control.focused_records(
            prompt)
