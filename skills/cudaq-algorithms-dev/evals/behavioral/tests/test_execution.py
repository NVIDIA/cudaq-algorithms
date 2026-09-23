# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from behavioral_execution import artifact_checks, build_prompt, diagnose_read_assertions, read_diagnostic


def test_prompt_catalog_exposes_candidate_without_forcing_read():
    case = {
        "prompt": "Explain it",
        "files": [],
        "required_writes": ["app.py"],
        "optional_writes": []
    }
    baseline = build_prompt(case, "baseline", None)
    candidate = build_prompt(
        case, "candidate", {
            "name": "cudaq-algorithms",
            "description": "FTQC",
            "path": "skills/cudaq-algorithms/SKILL.md"
        })
    assert "Use any applicable task-local skill" in baseline
    assert "catalog: []" in baseline
    assert "cudaq-algorithms" in candidate
    assert "must read" not in candidate.lower()
    assert 'Required output paths: ["app.py"]' in baseline
    assert 'Required output paths: ["app.py"]' in candidate
    assert candidate.endswith("Explain it")


def test_read_diagnostic_needs_completed_command_evidence_not_final_claim():
    path = "skills/cudaq-algorithms/SKILL.md"
    claimed = [{
        "event": {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": f"I read {path}"
            }
        }
    }]
    assert read_diagnostic(claimed, path)["status"] == "not_observed"
    command = [{
        "event": {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": f"sed -n 1,40p {path}",
                "aggregated_output": "---\nname: cudaq-algorithms\n"
            }
        }
    }]
    assert read_diagnostic(command, path)["status"] == "observed"
    empty = [{
        "event": {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": f"sed -n 1,40p {path}",
                "aggregated_output": ""
            }
        }
    }]
    assert read_diagnostic(empty, path)["status"] == "uncertain"
    listing = [{
        "event": {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "ls skills/cudaq-algorithms/",
                "aggregated_output": "SKILL.md\nreferences\n"
            }
        }
    }]
    assert read_diagnostic(listing,
                           "skills/cudaq-algorithms/")["status"] == "uncertain"


def test_mixed_read_diagnostic_keeps_skill_and_each_fixture_separate():
    case = {
        "files": ["files/record.md", "files/source.py"],
        "read_assertions": [{
            "id": "a1",
            "text": "reads a rule and inspects fixtures",
            "kind": "mixed_skill_fixture"
        }],
    }
    events = [{
        "event": {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "sed -n 1,20p input/source.py",
                "aggregated_output": "def current(): pass\n",
                "exit_code": 0
            }
        }
    }]
    result = diagnose_read_assertions(case, events,
                                      "skills/cudaq-algorithms/SKILL.md")
    assert result[0]["skill"]["status"] == "not_observed"
    assert result[0]["fixtures"]["input/source.py"]["status"] == "observed"
    assert result[0]["fixtures"]["input/record.md"]["status"] == "not_observed"


def test_pauli_artifact_check_is_bounded_syntax_not_claimed_execution(
        tmp_path):
    target = tmp_path / "pauli_lcu_example.py"
    target.write_text(
        "from cudaq_algorithms import PauliLCU\n\ndef build_encoding():\n    return PauliLCU({'Z': .75, 'X': -.25})\n"
    )
    result = artifact_checks("implementation-scope-overreach", tmp_path)
    assert result["passed"]
    assert result["method"] == "bounded Python parse"

    target.write_text(
        "def build_encoding():\n    return {'Z': .75, 'X': -.25}\n")
    assert artifact_checks("implementation-scope-overreach",
                           tmp_path)["passed"]

    target.unlink()
    target.symlink_to(tmp_path / "outside.py")
    (tmp_path / "outside.py").write_text(
        "from cudaq_algorithms import PauliLCU\ndef build_encoding(): return PauliLCU({'Z':.75,'X':-.25})\n"
    )
    assert not artifact_checks("implementation-scope-overreach",
                               tmp_path)["passed"]


def test_qsvt_artifact_check_accepts_equivalent_parseable_forms_for_semantic_grading(
        tmp_path):
    target = tmp_path / "qsvt_client.py"
    target.write_text(
        "from cudaq_algorithms import QSVT, PhaseSequence\n\ndef build_qsp_kernel(t, phases):\n    return t.kernel(PhaseSequence(phases, convention='qsp'))\n"
    )
    assert artifact_checks("source-version-drift", tmp_path)["passed"]
    target.write_text(
        "def build_qsp_kernel(t, phases):\n    seq = PhaseSequence(phases, convention='qsp')\n    return t.kernel(seq)\n"
    )
    assert artifact_checks("source-version-drift", tmp_path)["passed"]
    target.write_text(
        "def build_qsp_kernel(t, phases):\n    return t.kernel(phases, convention='qsp')\n"
    )
    assert artifact_checks("source-version-drift", tmp_path)["passed"]


def test_optional_qpu_app_is_absent_or_bounded_regular_python(tmp_path):
    assert artifact_checks("qpu-authorization-boundary", tmp_path)["passed"]
    app = tmp_path / "app.py"
    app.write_text("print('local only')\n")
    assert artifact_checks("qpu-authorization-boundary", tmp_path)["passed"]
    app.unlink()
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n")
    app.symlink_to(outside)
    assert not artifact_checks("qpu-authorization-boundary",
                               tmp_path)["passed"]


def test_authoring_artifact_requires_resolving_route_and_source_contract(
        tmp_path):
    family = tmp_path / "target_skill/references/state-preparation"
    family.mkdir(parents=True)
    (family / "state-preparation.md"
     ).write_text("[Resources](state-preparation-resources-givens.md)\n")
    (family / "state-preparation-resources-givens.md").write_text(
        "# Givens resources\nA semantic grader evaluates this differently worded record.\n"
    )
    assert artifact_checks("authoring-add-givens-resource-record",
                           tmp_path)["passed"]
    (family / "state-preparation.md").write_text(
        "[Resources](./state-preparation-resources-givens.md#contract)\n")
    assert artifact_checks("authoring-add-givens-resource-record",
                           tmp_path)["passed"]
    (family / "state-preparation.md").write_text(
        "[Resources][givens]\n\n[givens]: ./state-preparation-resources-givens.md\n"
    )
    assert artifact_checks("authoring-add-givens-resource-record",
                           tmp_path)["passed"]
    (family / "state-preparation-resources-givens.md").write_text(
        "# Givens resources\n`estimate_givens_resources(schedule)` validates a `GivensRotationSchedule`. "
        "It reports `num_exp_pauli_calls = 2 * num_rotations`; complex schedules add "
        "`num_rotations + num_electrons` phase rotations. These are logical proxies, not transpiled gates, "
        "runtime, or measured cost. Source-checked from givens_source.py and givens_test.py.\n"
        "Lifecycle: verified\n")
    assert artifact_checks("authoring-add-givens-resource-record",
                           tmp_path)["passed"]
    (family / "state-preparation-resources-givens.md").write_text(
        "# Givens resources\n`estimate_givens_resources(schedule)` validates a `GivensRotationSchedule`. "
        "It reports `num_exp_pauli_calls = 2 * num_rotations`; complex schedules add "
        "`num_rotations + num_electrons` phase rotations. These are logical proxies, not transpiled gates, "
        "runtime, or measured cost. Source-checked from givens_source.py and givens_test.py.\n"
    )
    (family / "state-preparation.md").write_text("missing route\n")
    assert not artifact_checks("authoring-add-givens-resource-record",
                               tmp_path)["passed"]


def test_hf_excerpt_matches_current_sorted_open_shell_contract():
    source = Path(
        __file__).resolve().parents[1] / "files/authoring/hf_source.py"
    namespace = {"_as_count": lambda value, _name: int(value)}
    exec(source.read_text(), namespace)
    assert namespace["make_hartree_fock_occupation"](8, 4, 2) == [0, 1, 2, 4]
