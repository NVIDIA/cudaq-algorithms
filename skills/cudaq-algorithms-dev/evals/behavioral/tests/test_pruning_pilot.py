# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
MODULE_PATH = HERE / "pruning_pilot.py"


def load_module():
    spec = importlib.util.spec_from_file_location("pruning_pilot_under_test",
                                                  MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_schedule_is_exactly_two_skill_arms_for_both_models():
    pilot = load_module()

    assert [row.as_dict() for row in pilot.SCHEDULE] == [
        {
            "model": "qwen3-coder-30b",
            "condition": "current"
        },
        {
            "model": "qwen3-coder-30b",
            "condition": "pruned"
        },
        {
            "model": "qwen3-8b",
            "condition": "pruned"
        },
        {
            "model": "qwen3-8b",
            "condition": "current"
        },
    ]
    assert pilot.BUDGETS == {
        "arm_timeout_seconds": 300,
        "max_backend_requests_per_arm": 20,
        "max_output_tokens_per_request": 2048,
        "backend_timeout_seconds": 120,
        "context_window": 32768,
        "compact_limit": 24576,
        "worker_retries": 0,
    }
    assert pilot.GENERATION_CONFIG == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }


def test_operational_inventory_allows_intentional_pruning_and_rejects_evals(
        tmp_path):
    pilot = load_module()
    skill = tmp_path / "skill"
    (skill / "references").mkdir(parents=True)
    (skill / "evals").mkdir()
    (skill / "authoring").mkdir()
    (skill / "coverage").mkdir()
    (skill / "scripts").mkdir()
    (skill / "__pycache__").mkdir()
    (skill / "SKILL.md").write_text("skill\n", encoding="utf-8")
    (skill / "references/kept.md").write_text("kept\n", encoding="utf-8")
    (skill / "evals/private.md").write_text("private\n", encoding="utf-8")
    (skill / "authoring/maintainer.md").write_text("maintainer\n",
                                                   encoding="utf-8")
    (skill / "coverage/policy.md").write_text("coverage\n", encoding="utf-8")
    (skill / "scripts/tool.py").write_text("tool\n", encoding="utf-8")
    (skill / "__pycache__/cache.pyc").write_bytes(b"cache")

    assert set(pilot.operational_inventory(skill)) == {
        "SKILL.md",
        "references/kept.md",
    }

    (skill / "references/link.md").symlink_to(skill / "SKILL.md")
    with pytest.raises(ValueError, match="symlink"):
        pilot.operational_inventory(skill)


def test_recorded_capture_bindings_require_exact_schedule_and_frozen_hashes(
        tmp_path):
    pilot = load_module()
    output = tmp_path / "output"
    pair_roots = {}
    attempts = []
    for index, scheduled in enumerate(pilot.SCHEDULE, 1):
        key = f"{scheduled.model}/{scheduled.condition}"
        root = output / "pairs" / key
        capture = root / "captures/candidate"
        capture.mkdir(parents=True)
        raw = f"capture-{index}\n".encode()
        (capture / "capture.json").write_bytes(raw)
        pair_roots[key] = str(root)
        attempts.append({
            "index": index,
            **scheduled.as_dict(),
            "result": {
                "capture": {
                    "status": "complete",
                    "capture_json_sha256": pilot.sha256_bytes(raw),
                    "source_inventory_sha256": "s" * 64,
                }
            },
        })
    manifest = {
        "schedule": [row.as_dict() for row in pilot.SCHEDULE],
        "pair_roots": pair_roots,
        "source_inventory_sha256": "s" * 64,
    }

    bindings = pilot.validated_attempt_captures(manifest,
                                                {"attempts": attempts})

    assert list(bindings) == [
        f"{row.model}/{row.condition}" for row in pilot.SCHEDULE
    ]
    assert bindings["qwen3-coder-30b/current"][
        "capture_json_sha256"] == pilot.sha256_bytes(b"capture-1\n")

    (Path(bindings["qwen3-coder-30b/current"]["path"]) /
     "capture.json").write_bytes(b"mutated\n")
    with pytest.raises(RuntimeError, match="capture.json changed"):
        pilot.validated_attempt_captures(manifest, {"attempts": attempts})


@pytest.mark.parametrize("mutation",
                         ["duplicate", "reordered", "incomplete", "source"])
def test_recorded_capture_bindings_reject_invalid_attempt_evidence(
        tmp_path, mutation):
    pilot = load_module()
    output = tmp_path / "output"
    pair_roots = {}
    attempts = []
    for index, scheduled in enumerate(pilot.SCHEDULE, 1):
        key = f"{scheduled.model}/{scheduled.condition}"
        root = output / "pairs" / key
        capture = root / "captures/candidate"
        capture.mkdir(parents=True)
        raw = f"capture-{index}\n".encode()
        (capture / "capture.json").write_bytes(raw)
        pair_roots[key] = str(root)
        attempts.append({
            "index": index,
            **scheduled.as_dict(),
            "result": {
                "capture": {
                    "status": "complete",
                    "capture_json_sha256": pilot.sha256_bytes(raw),
                    "source_inventory_sha256": "s" * 64,
                }
            },
        })
    manifest = {
        "schedule": [row.as_dict() for row in pilot.SCHEDULE],
        "pair_roots": pair_roots,
        "source_inventory_sha256": "s" * 64,
    }
    if mutation == "duplicate":
        attempts[-1] = dict(attempts[0])
    elif mutation == "reordered":
        attempts[0], attempts[1] = attempts[1], attempts[0]
    elif mutation == "incomplete":
        attempts[0]["result"]["capture"]["status"] = "failed"
    else:
        attempts[0]["result"]["capture"]["source_inventory_sha256"] = "x" * 64

    with pytest.raises(RuntimeError):
        pilot.validated_attempt_captures(manifest, {"attempts": attempts})


def test_manifest_freezes_models_skills_prompt_bridge_and_helpers(tmp_path):
    pilot = load_module()
    current = tmp_path / "current"
    pruned = tmp_path / "pruned"
    for root, text in ((current, "current\n"), (pruned, "pruned\n")):
        root.mkdir()
        (root / "SKILL.md").write_text(text, encoding="utf-8")

    prompt = "A custom structural BlockEncoding works with Walk but fails in a simulation helper."
    manifest = pilot.manifest_contract(
        output=tmp_path / "output",
        current_skill=current,
        pruned_skill=pruned,
        source_inventory={"README.md": "a" * 64},
        prompt=prompt,
        dependency_hashes={"helper": "b" * 64},
    )

    assert manifest["case_id"] == "E03"
    assert manifest["prompt"] == prompt
    assert manifest["prompt_sha256"] == pilot.sha256_bytes(prompt.encode())
    assert manifest["repaired_bridge"]["sha256"] == (
        "27eda9757b13dc024f44a4ea671f392beeb5d28de8f06a3b82b3ed235b351411")
    assert manifest["skill_inventories"]["current"]["SKILL.md"] != manifest[
        "skill_inventories"]["pruned"]["SKILL.md"]
    assert manifest["schedule"] == [row.as_dict() for row in pilot.SCHEDULE]
    assert manifest["generation_config"] == pilot.GENERATION_CONFIG
    assert manifest["budgets"] == pilot.BUDGETS
    assert manifest["scientific_checker"]["case_id"] == "E03"
    assert manifest["scientific_checker"]["targeted"] == {
        "expected_collected": 32,
        "allowed_skips": 0,
    }


def test_source_inventory_hash_matches_real_capture_helper_canonicalization():
    pilot = load_module()
    capture_path = pilot.OLD_EVALS / "strategy/capture.py"
    if not capture_path.is_file():
        pytest.skip(
            "set CUDAQ_PRUNING_OLD_EVALS to the preserved old evals tree")
    spec = importlib.util.spec_from_file_location("capture_hash_contract",
                                                  capture_path)
    assert spec is not None and spec.loader is not None
    capture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(capture)
    inventory = {"unicode-α.md": "a" * 64, "README.md": "b" * 64}

    assert pilot._inventory_sha256(inventory) == capture._digest(
        capture._json(inventory))


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({
            "completed": True
        }, False),
        (
            {
                "completed": False,
                "failure_stage": "turn_completion",
                "model_input_audit": {
                    "request_count": 20
                },
                "elapsed_seconds": 250,
            },
            True,
        ),
        (
            {
                "completed": False,
                "failure_stage": "turn_completion",
                "model_input_audit": {
                    "request_count": 2
                },
                "elapsed_seconds": 300.1,
            },
            True,
        ),
        (
            {
                "completed": False,
                "failure_stage": "effective_config_gate",
                "model_input_audit": {
                    "request_count": 0
                },
                "elapsed_seconds": 1,
            },
            False,
        ),
    ],
)
def test_budget_censoring_is_distinct_from_other_failures(result, expected):
    pilot = load_module()
    assert pilot.is_budget_censored(result) is expected


def test_cli_prepare_preflight_run_and_check_are_separate_actions():
    pilot = load_module()
    parser = pilot.parser()

    prepare = parser.parse_args([
        "prepare",
        "--output",
        "/tmp/out",
        "--current-skill",
        "/tmp/current",
        "--pruned-skill",
        "/tmp/pruned",
    ])
    assert prepare.action == "prepare"
    for action in ("preflight", "run", "check"):
        assert parser.parse_args([action, "--output",
                                  "/tmp/out"]).action == action


def test_real_preserved_command_uses_8b_catalog_endpoint_and_bounded_config(
        tmp_path):
    pilot = load_module()
    _, dual, bridge, helper = pilot._dependencies()
    spec = next(model for model in dual.MODELS if model.key == "qwen3-8b")
    root = tmp_path / "pair"
    (root / "arms/candidate/sqlite").mkdir(parents=True)
    (root / "arms/candidate/log").mkdir()

    class Collector:
        config_args = []

    with pilot._bound_model(helper, dual, bridge, spec, root):
        config = helper.LocalBridgeConfig(
            client_token="x" * 32,
            log_path=root / "bridge.jsonl",
            max_requests=20,
            max_output_tokens=2048,
            backend_timeout_seconds=120,
        )
        command = helper._command("candidate", "http://127.0.0.1:9999/v1",
                                  Collector(), root)

    joined = " ".join(command)
    assert config.upstream_url == "http://127.0.0.1:18081/v1/chat/completions"
    assert config.allowed_model == "local-qwen3-8b"
    assert config.max_requests == 20
    assert config.max_output_tokens == 2048
    assert 'model="local-qwen3-8b"' in joined
    assert 'model_provider="local_qwen3_8b"' in joined
    assert f"model_catalog_json={json.dumps(str(root / 'model-catalog.json'))}" in joined
