# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path

import pytest

from behavioral_cases import (
    build_run_plan,
    behavioral_inventory,
    check_scope,
    load_authored_cases,
    stage_case,
)

ROOT = Path(__file__).resolve().parents[2]


def test_all_authored_cases_load_without_mutating_source_contract():
    cases = load_authored_cases(ROOT / "evals.json")
    assert len(cases) == 42
    assert sum(len(case["assertions"]) for case in cases) == 220
    assert [case["id"] for case in cases
            if case["expected_skill"] is None] == [
                "negative-cudaq-install",
                "negative-generic-quantum-concept",
                "negative-generic-cudaq-kernel",
            ]
    assert all("shared_assertions" in case and "read_assertions" in case
               for case in cases)
    assert sum(len(case["read_assertions"]) for case in cases) == 43
    mixed = {
        item["id"]: item["read_assertions"]
        for item in cases if any(assertion["kind"] == "mixed_skill_fixture"
                                 for assertion in item["read_assertions"])
    }
    assert set(mixed) == {
        "source-version-drift", "implementation-scope-overreach"
    }
    by_id = {case["id"]: case for case in cases}
    assert "both material ambiguities" in by_id[
        "material-ambiguity-clarification"]["shared_assertions"][0]["text"]
    assert "contract as unavailable" in by_id["source-unavailable-fallback"][
        "shared_assertions"][0]["text"]
    assert "staged recorded contract" in by_id["source-version-drift"][
        "shared_assertions"][0]["text"]
    assert "staged change request" in by_id["implementation-scope-overreach"][
        "shared_assertions"][0]["text"]


def test_stage_case_copies_only_named_fixture_and_authorized_target(tmp_path):
    fixture_root = tmp_path / "files"
    fixture_root.mkdir()
    (fixture_root / "request.md").write_text("request")
    (fixture_root / "starter.py").write_text("old = True\n")
    case = {
        "id": "implementation",
        "files": ["files/request.md", "files/starter.py"],
        "authorized_targets": {
            "files/starter.py": "starter.py"
        },
    }
    workspace = tmp_path / "workspace"
    stage_case(case, workspace, fixture_root.parent, arm="baseline")

    assert (workspace / "input/request.md").read_text() == "request"
    assert (workspace / "input/starter.py").read_text() == "old = True\n"
    assert (workspace / "starter.py").read_text() == "old = True\n"
    assert not (workspace / "skills").exists()
    assert sorted(p.name for p in workspace.iterdir()) == [
        ".tmp", "input", "starter.py"
    ]


def test_candidate_skill_is_read_only_and_baseline_has_no_skill(tmp_path):
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    skill = tmp_path / "candidate"
    skill.mkdir()
    (skill /
     "SKILL.md").write_text("---\nname: demo\ndescription: demo\n---\n")
    case = {"id": "advice", "files": [], "authorized_targets": {}}

    candidate = tmp_path / "candidate-work"
    stage_case(case,
               candidate,
               fixture_root,
               arm="candidate",
               skill_root=skill)
    baseline = tmp_path / "baseline-work"
    stage_case(case, baseline, fixture_root, arm="baseline", skill_root=skill)

    assert (candidate / "skills/demo/SKILL.md").is_file()
    assert (candidate / "skills/demo/SKILL.md").stat().st_mode & 0o222 == 0
    assert not (baseline / "skills").exists()


def test_authoring_candidate_stages_guidance_policy_conventions_without_completed_records(
        tmp_path):
    skill = tmp_path / "candidate"
    for directory in ("references", "authoring", "assets", "coverage"):
        (skill / directory).mkdir(parents=True)
    (skill /
     "SKILL.md").write_text("---\nname: demo\ndescription: demo\n---\n")
    (skill / "references/conventions").mkdir()
    (skill / "references/conventions/resource-levels.md").write_text("levels")
    (skill / "references/completed-answer.md").write_text("answer")
    (skill / "authoring/architecture.md").write_text("author")
    (skill / "assets/template.md").write_text("template")
    (skill / "coverage/policy.md").write_text("policy")
    (skill / "coverage/history.md").write_text("history")
    workspace = tmp_path / "work"
    stage_case({
        "id": "author",
        "files": [],
        "authorized_targets": {}
    },
               workspace,
               tmp_path,
               arm="candidate",
               skill_root=skill,
               authoring=True)
    staged = workspace / "skills/demo"
    assert (staged / "references/conventions/resource-levels.md").is_file()
    assert not (staged / "references/completed-answer.md").exists()
    assert (staged / "authoring/architecture.md").is_file()
    assert (staged / "coverage/policy.md").is_file()
    assert not (staged / "coverage/history.md").exists()


def test_nested_authoring_targets_are_materialized(tmp_path):
    (tmp_path / "files").mkdir()
    (tmp_path / "files/record.md").write_text("starter")
    case = {
        "id": "authoring",
        "files": ["files/record.md"],
        "authorized_targets": {
            "files/record.md": "target_skill/references/family/record.md"
        }
    }
    workspace = tmp_path / "workspace"
    stage_case(case, workspace, tmp_path, arm="baseline")
    assert (
        workspace /
        "target_skill/references/family/record.md").read_text() == "starter"


def test_scope_requires_authorized_change_and_rejects_every_other_change(
        tmp_path):
    before = {"starter.py": "old", "input/request.md": "fixed"}
    assert check_scope(before, {
        **before, "starter.py": "new"
    },
                       required={"starter.py"},
                       optional=set())["passed"]
    assert not check_scope(
        before, before, required={"starter.py"}, optional=set())["passed"]
    assert not check_scope(before, {
        **before, "surprise.txt": "x"
    },
                           required=set(),
                           optional=set())["passed"]
    assert check_scope(before, {
        **before, ".tmp/work": "x"
    },
                       required=set(),
                       optional=set())["passed"]


def test_nonregular_workspace_nodes_are_scoped_but_scratch_nodes_are_ignored(
        tmp_path):
    (tmp_path / ".tmp").mkdir()
    before = behavioral_inventory(tmp_path)
    os.mkfifo(tmp_path / "outside.fifo")
    os.mkfifo(tmp_path / ".tmp/scratch.fifo")
    after = behavioral_inventory(tmp_path)
    result = check_scope(before, after, required=set(), optional=set())
    assert not result["passed"]
    assert result["unauthorized"] == ["outside.fifo"]


def test_balanced_plan_has_serial_rotating_arms_and_no_stop_on_pass():
    plan = build_run_plan(["a", "b"], repetitions=3, seed=9)
    assert len(plan) == 12
    blocks = [plan[index:index + 2] for index in range(0, len(plan), 2)]
    assert all({item["arm"]
                for item in block} == {"baseline", "candidate"}
               for block in blocks)
    assert all(
        len({(item["case"], item["repetition"])
             for item in block}) == 1 for block in blocks)
    assert [block[0]["arm"] for block in blocks].count("baseline") == 3
    assert [block[0]["arm"] for block in blocks].count("candidate") == 3


def test_fixture_paths_must_be_relative_and_exist(tmp_path):
    path = tmp_path / "evals.json"
    path.write_text(
        json.dumps({
            "evals": [{
                "id": "bad",
                "prompt": "x",
                "expected_skill": None,
                "files": ["../secret"],
                "assertions": ["x"]
            }]
        }))
    with pytest.raises(ValueError, match="unsafe fixture"):
        load_authored_cases(path)
