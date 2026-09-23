# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

import selective_discovery as selective


def _skill(root: Path, description: str, record: str = "record\n") -> Path:
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: cudaq-algorithms\ndescription: {description}\n---\nbody\n"
    )
    (root / "references").mkdir()
    (root / "references/qsvt.md").write_text(record)
    return root


def _case(identity="positive", *, expected=True):
    return {
        "id": identity,
        "prompt": "Answer the question",
        "files": ["files/evidence.txt"],
        "assertions": [],
        "shared_assertions": [],
        "read_assertions": [],
        "authorized_targets": {},
        "required_writes": [],
        "optional_writes": [],
        "expected_records": ["references/qsvt.md"],
        "expected_activation": expected,
        "routing_required": True,
        "activation_required": True,
    }


def _patch_runtime(monkeypatch):
    monkeypatch.setattr(
        selective, "_runtime_provenance", lambda _python: {
            "python": {
                "sha256": "python",
                "version": "Python 3.12"
            },
            "codex": {
                "sha256": "codex",
                "version": "codex-cli 0.144.4"
            },
        })
    monkeypatch.setattr(
        selective, "_worker_runtime_metadata", lambda _python: {
            "prefix": "/clean",
            "base_prefix": "/usr",
            "distributions": [],
            "discoverable": {
                "cudaq": False,
                "cudaq_algorithms": False,
                "pip": False
            },
        })


def test_prepare_freezes_contract_and_builds_interleaved_opaque_pairs(
        tmp_path, monkeypatch):
    fixture_root = tmp_path / "evals"
    (fixture_root / "files").mkdir(parents=True)
    (fixture_root / "files/evidence.txt").write_text("fixed\n")
    monkeypatch.setattr(selective, "EVAL_ROOT", fixture_root)
    _patch_runtime(monkeypatch)
    current = _skill(tmp_path / "current", "current words")
    keywords = _skill(tmp_path / "keywords", "keyword words")
    cases = [_case("positive"), _case("negative", expected=False)]
    output = tmp_path / "campaign"

    manifest = selective.prepare(output, {
        "cases": cases,
        "snapshots": {
            "current": str(current),
            "keywords": str(keywords)
        },
        "allowed_snapshot_differences": ["SKILL.md"],
    },
                                 python="/clean/python")

    assert manifest["repetitions"] == 3
    assert manifest["parallel"] == 4
    assert manifest["worker_timeout_s"] == 180
    assert manifest["model"] == "gpt-5.5"
    assert manifest["reasoning_effort"] == "low"
    assert len(manifest["schedule"]) == 12
    for left, right in zip(manifest["schedule"][::2],
                           manifest["schedule"][1::2]):
        assert (left["case"], left["repetition"]) == (right["case"],
                                                      right["repetition"])
        assert {left["variant"], right["variant"]} == {"current", "keywords"}
        assert left["worker_run"]["arm"] == right["worker_run"][
            "arm"] == "candidate"
    worker_json = json.dumps(
        [row["worker_run"] for row in manifest["schedule"]])
    assert "current" not in worker_json and "keywords" not in worker_json
    assert manifest["fixture_inventory"] == {
        "files/evidence.txt":
        selective._hash(fixture_root / "files/evidence.txt")
    }
    assert manifest["evaluator_inventory"]
    assert manifest["snapshot_inventory"]["current"]
    assert manifest["cases"] == cases

    cases[0]["prompt"] = "mutated after preparation"
    frozen = json.loads((output / "manifest.json").read_text())
    assert frozen["cases"][0]["prompt"] == "Answer the question"


@pytest.mark.parametrize("mutate, message", [
    (lambda spec: spec["cases"].append(dict(spec["cases"][0])), "unique"),
    (lambda spec: spec["cases"][0].update(expected_activation="yes"),
     "expected_activation"),
    (lambda spec: spec["cases"][0].update(expected_records=["../secret"]),
     "expected_records"),
])
def test_prepare_rejects_invalid_or_duplicate_cases(tmp_path, monkeypatch,
                                                    mutate, message):
    fixture_root = tmp_path / "evals"
    (fixture_root / "files").mkdir(parents=True)
    (fixture_root / "files/evidence.txt").write_text("fixed\n")
    monkeypatch.setattr(selective, "EVAL_ROOT", fixture_root)
    _patch_runtime(monkeypatch)
    spec = {
        "cases": [_case()],
        "snapshots": {
            "current": str(_skill(tmp_path / "current", "same")),
            "keywords": str(_skill(tmp_path / "keywords", "same")),
        },
        "allowed_snapshot_differences": ["SKILL.md"],
    }
    mutate(spec)
    with pytest.raises(ValueError, match=message):
        selective.prepare(tmp_path / "campaign", spec, python="/clean/python")


def test_prepare_rejects_unapproved_skill_content_difference(
        tmp_path, monkeypatch):
    monkeypatch.setattr(selective, "EVAL_ROOT", tmp_path / "evals")
    _patch_runtime(monkeypatch)
    current = _skill(tmp_path / "current", "same", record="old\n")
    keywords = _skill(tmp_path / "keywords", "same", record="changed\n")
    with pytest.raises(
            ValueError,
            match="unapproved snapshot differences.*references/qsvt.md"):
        selective.prepare(tmp_path / "campaign", {
            "cases": [{
                **_case(), "files": []
            }],
            "snapshots": {
                "current": str(current),
                "keywords": str(keywords)
            },
            "allowed_snapshot_differences": ["SKILL.md"],
        },
                          python="/clean/python")


def test_prepare_rejects_nonfixed_experiment_controls(tmp_path, monkeypatch):
    monkeypatch.setattr(selective, "EVAL_ROOT", tmp_path / "evals")
    _patch_runtime(monkeypatch)
    snapshots = {
        "current": str(_skill(tmp_path / "current", "same")),
        "keywords": str(_skill(tmp_path / "keywords", "same")),
    }
    with pytest.raises(ValueError, match="repetitions must be fixed at 3"):
        selective.prepare(tmp_path / "campaign", {
            "cases": [{
                **_case(), "files": []
            }],
            "snapshots": snapshots,
            "repetitions": 2,
        },
                          python="/clean/python")


def _command(command, output="text\n", exit_code=0, elapsed=1.5):
    return {
        "elapsed_s": elapsed,
        "event": {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": command,
                "aggregated_output": output,
                "exit_code": exit_code,
            }
        }
    }


def test_observe_reads_requires_exact_successful_content_read_and_records_order(
):
    skill = "skills/cudaq-algorithms/SKILL.md"
    relative_record = "references/qsvt.md"
    record = f"skills/cudaq-algorithms/{relative_record}"
    events = [
        _command("ls skills/cudaq-algorithms", "SKILL.md\n", elapsed=.5),
        _command(f"sed -n 1,20p {skill}.bak", "wrong\n", elapsed=.7),
        _command(f"sed -n 1,20p {skill}", "", elapsed=.9),
        _command(f"cat {skill}", "skill text\n", elapsed=1.2),
        _command(f"sed -n 1,20p {record}", "record text\n", elapsed=2.4),
    ]
    observed = selective.observe_reads(events, [relative_record],
                                       worker_completed=True)
    assert observed["activation"] is True
    assert observed["skill"]["status"] == "observed"
    assert observed["skill"]["event_index"] == 4
    assert observed["skill"]["elapsed_s"] == 1.2
    assert observed["records"][relative_record]["status"] == "observed"
    assert observed["records"][relative_record]["event_index"] == 5
    assert observed["first_skill_content_read"]["order"] < observed[
        "first_expected_record_read"]["order"]
    assert observed["command_count"] == 5


def test_observe_reads_distinguishes_failed_empty_absent_and_unavailable():
    skill = "skills/cudaq-algorithms/SKILL.md"
    failed = selective.observe_reads([_command(f"cat {skill}", "denied", 1)],
                                     [],
                                     worker_completed=True)
    assert failed["activation"] == "unknown"
    assert failed["skill"]["status"] == "failed"
    empty = selective.observe_reads([_command(f"cat {skill}", "", 0)], [],
                                    worker_completed=True)
    assert empty["activation"] == "unknown"
    assert empty["skill"]["status"] == "empty"
    absent = selective.observe_reads([_command("pwd")], [],
                                     worker_completed=True)
    assert absent["activation"] is False
    assert absent["skill"]["status"] == "not_observed"
    unavailable = selective.observe_reads([], [], worker_completed=False)
    assert unavailable["activation"] == "unknown"
    assert unavailable["skill"]["status"] == "unavailable"


def test_observe_reads_accepts_absolute_workspace_path_without_partial_suffix(
):
    relative_record = "references/qsvt.md"
    full = "/tmp/opaque/runs/attempt-001/skills/cudaq-algorithms/references/qsvt.md"
    observed = selective.observe_reads([_command(f"cat {full}")],
                                       [relative_record],
                                       worker_completed=True)
    assert observed["records"][relative_record]["status"] == "observed"
    partial = selective.observe_reads([_command(f"cat {full}.backup")],
                                      [relative_record],
                                      worker_completed=True)
    assert partial["records"][relative_record]["status"] == "not_observed"


@pytest.mark.parametrize("command", [
    "rg --files skills/cudaq-algorithms/SKILL.md",
    "grep -l name skills/cudaq-algorithms/SKILL.md",
    'echo "cat skills/cudaq-algorithms/SKILL.md"',
])
def test_observe_reads_does_not_promote_name_only_or_quoted_fake_reader(
        command):
    observed = selective.observe_reads(
        [_command(command, "skills/cudaq-algorithms/SKILL.md\n")], [],
        worker_completed=True)
    assert observed["activation"] is False
    assert observed["skill"]["status"] == "not_observed"


def test_observe_reads_marks_unrecognized_target_access_unknown():
    observed = selective.observe_reads([
        _command("mystery-reader skills/cudaq-algorithms/SKILL.md", "maybe\n")
    ], [],
                                       worker_completed=True)
    assert observed["activation"] == "unknown"
    assert observed["skill"]["status"] == "unrecognized"


def test_observe_reads_handles_actual_bash_lc_wrapper_and_keeps_target_association(
):
    skill = "skills/cudaq-algorithms/SKILL.md"
    actual = _command(f'/bin/bash -lc "sed -n \'1,220p\' {skill}"',
                      "---\nname: cudaq-algorithms\n")
    observed = selective.observe_reads([actual], [], worker_completed=True)
    assert observed["activation"] is True
    assert observed["skill"]["status"] == "observed"

    other = "skills/cudaq-algorithms/references/catalog.md"
    associated = _command(f'/bin/bash -lc "echo {skill}; cat {other}"',
                          "skill name\nother contents\n")
    not_observed = selective.observe_reads(
        [associated], [],
        worker_completed=True,
        known_skill_paths=["SKILL.md", "references/catalog.md"])
    assert not_observed["skill"]["status"] == "observed"
    skill_only = selective._observation([associated], [skill], available=True)
    assert skill_only["status"] == "not_observed"


def test_observe_reads_handles_wrapped_expected_record_and_unrecognized_reader(
):
    relative = "references/qsvt/qsvt-sequence.md"
    full = f"skills/cudaq-algorithms/{relative}"
    read = selective.observe_reads(
        [_command(f"/bin/bash -lc \"sed -n '1,260p' {full}\"")], [relative],
        worker_completed=True)
    assert read["records"][relative]["status"] == "observed"
    unknown = selective.observe_reads(
        [_command(f'/bin/bash -lc "mystery-reader {full}"')], [relative],
        worker_completed=True)
    assert unknown["records"][relative]["status"] == "unrecognized"


@pytest.mark.parametrize("item_update", [
    {
        "exit_code": None
    },
    {
        "exit_code": False
    },
    {
        "aggregated_output": None
    },
])
def test_observe_reads_does_not_promote_incomplete_native_event(item_update):
    event = _command("cat skills/cudaq-algorithms/SKILL.md")
    event["event"]["item"].update(item_update)
    observed = selective.observe_reads([event], [], worker_completed=True)
    assert observed["activation"] == "unknown"
    assert observed["skill"]["status"] == "unrecognized"


def test_verify_frozen_rejects_manifest_case_or_control_mutation(
        tmp_path, monkeypatch):
    case = {**_case(), "files": []}
    manifest = {
        "cases": [case],
        "case_contract_sha256": selective._case_hash([case]),
        "schedule": selective._build_schedule([case["id"]]),
        "model": "gpt-5.5",
        "reasoning_effort": "low",
        "repetitions": 3,
        "seed": 20260914,
        "parallel": 4,
        "worker_timeout_s": 180,
        "stop_on_pass": False,
    }
    monkeypatch.setattr(selective, "_verify_fixtures", lambda _manifest: None)
    monkeypatch.setattr(selective, "_verify_snapshots_and_evaluators",
                        lambda _manifest: None)
    monkeypatch.setattr(selective, "_runtime_provenance", lambda _python: {})
    monkeypatch.setattr(selective, "_worker_runtime_metadata",
                        lambda _python: {})
    manifest["runtime_provenance"] = manifest["worker_runtime"] = {}
    selective._verify_frozen(manifest, "/clean/python")
    manifest["cases"][0]["prompt"] = "tampered"
    with pytest.raises(RuntimeError, match="case contract drift"):
        selective._verify_frozen(manifest, "/clean/python")


def test_activation_group_separates_optional_fixture_cases():
    assert selective._activation_group({
        "expected_activation": True,
        "activation_required": True
    }) == "positive"
    assert selective._activation_group({
        "expected_activation": True,
        "activation_required": False
    }) == "optional"
    assert selective._activation_group({
        "expected_activation": False,
        "activation_required": True
    }) == "negative"


def test_run_uses_one_shot_pairs_and_preserves_failure_then_continues(
        tmp_path, monkeypatch):
    output = tmp_path / "campaign"
    runs = output / "runs"
    runs.mkdir(parents=True)
    snapshot = _skill(tmp_path / "frozen", "same")
    fixture_root = tmp_path / "evals"
    (fixture_root / "files").mkdir(parents=True)
    (fixture_root / "files/evidence.txt").write_text("fixed\n")
    monkeypatch.setattr(selective, "EVAL_ROOT", fixture_root)
    case = _case()
    schedule = [
        {
            "pair": 1,
            "variant": "current",
            "case": "positive",
            "repetition": 1,
            "worker_run": {
                "id": "attempt-001",
                "case": "positive",
                "repetition": 1,
                "arm": "candidate"
            }
        },
        {
            "pair": 1,
            "variant": "keywords",
            "case": "positive",
            "repetition": 1,
            "worker_run": {
                "id": "attempt-002",
                "case": "positive",
                "repetition": 1,
                "arm": "candidate"
            }
        },
    ]
    manifest = {
        "cases": [case],
        "schedule": schedule,
        "parallel": 1,
        "worker_timeout_s": 180,
        "model": "gpt-5.5",
        "fixture_inventory": {
            "files/evidence.txt":
            selective._hash(fixture_root / "files/evidence.txt")
        },
        "snapshot_paths": {
            "current": str(snapshot),
            "keywords": str(snapshot)
        },
        "snapshot_inventory": {
            "current": selective._tree_inventory(snapshot),
            "keywords": selective._tree_inventory(snapshot)
        },
        "evaluator_inventory": {},
        "runtime_provenance": {},
        "worker_runtime": {},
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    (output / "preflight.json").write_text(json.dumps({"passed": True}))
    calls = []

    def fake_worker(worker_run, _case, _manifest, root, _python):
        calls.append(worker_run["id"])
        destination = root / "runs" / worker_run["id"]
        destination.mkdir()
        if len(calls) == 1:
            (destination / "events.jsonl").write_text(
                json.dumps(
                    _command("cat skills/cudaq-algorithms/SKILL.md", "denied",
                             1)) + "\n")
            raise RuntimeError("boom")
        (destination / "events.jsonl").write_text("")
        result = {
            **worker_run, "worker_attempted": True,
            "worker_completed": True,
            "status": "completed",
            "worker_elapsed_s": 2,
            "worker_telemetry": {
                "totals": {
                    "total_tokens": 7
                }
            }
        }
        (destination / "result.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(selective, "_worker_result", fake_worker)
    monkeypatch.setattr(selective, "_verify_frozen",
                        lambda *_args, **_kwargs: None)
    selective.run(output, python="/clean/python")
    assert calls == ["attempt-001", "attempt-002"]
    failed = json.loads((runs / "attempt-001/result.json").read_text())
    continued = json.loads((runs / "attempt-002/result.json").read_text())
    assert failed["status"] == "infrastructure_failure"
    assert failed["variant"] == "current"
    assert failed["discovery"]["skill"]["status"] == "failed"
    assert continued["variant"] == "keywords"
    with pytest.raises(RuntimeError, match="one-shot"):
        selective.run(output, python="/clean/python")


def test_report_groups_positive_negative_activation_and_native_resources(
        tmp_path):
    output = tmp_path / "campaign"
    runs = output / "runs"
    runs.mkdir(parents=True)
    cases = [
        _case("positive", expected=True),
        _case("negative", expected=False)
    ]
    schedule = []
    rows = []
    values = [
        ("positive", "current", False, 3, 10, 4),
        ("positive", "keywords", True, 2, 8, 3),
        ("negative", "current", False, 1, 5, 1),
        ("negative", "keywords", "unknown", None, None, 0),
    ]
    for index, (case, variant, activation, elapsed, tokens,
                commands) in enumerate(values, 1):
        run_id = f"attempt-{index:03d}"
        worker = {
            "id": run_id,
            "case": case,
            "repetition": 1,
            "arm": "candidate"
        }
        schedule.append({
            "pair": index,
            "variant": variant,
            "case": case,
            "repetition": 1,
            "worker_run": worker
        })
        row = {
            **worker,
            "variant": variant,
            "worker_attempted": True,
            "worker_completed": elapsed is not None,
            "status":
            "completed" if elapsed is not None else "infrastructure_failure",
            "worker_elapsed_s": elapsed,
            "worker_telemetry": {
                "totals": {
                    "input_tokens": tokens,
                    "cached_input_tokens": 2 if tokens is not None else None,
                    "output_tokens": 1 if tokens is not None else None,
                    "reasoning_output_tokens":
                    0 if tokens is not None else None,
                    "total_tokens": tokens,
                }
            },
            "discovery": {
                "activation":
                activation,
                "command_count":
                commands,
                "skill": {
                    "status":
                    "observed" if activation is True else "not_observed"
                },
                "records": {},
                "first_skill_content_read": ({
                    "elapsed_s": 1.25
                } if activation is True else None),
                "first_expected_record_read": ({
                    "elapsed_s": 1.75
                } if activation is True else None)
            },
        }
        destination = runs / run_id
        destination.mkdir()
        (destination / "result.json").write_text(json.dumps(row))
        rows.append(row)
    (output / "manifest.json").write_text(
        json.dumps({
            "cases": cases,
            "schedule": schedule
        }))

    summary = selective.report(output)

    assert summary["activation"]["positive"]["current"] == {
        "true": 0,
        "false": 1,
        "unknown": 0
    }
    assert summary["activation"]["positive"]["keywords"] == {
        "true": 1,
        "false": 0,
        "unknown": 0
    }
    assert summary["activation"]["negative"]["keywords"]["unknown"] == 1
    positive_keywords = summary["per_case"]["positive"]["keywords"]
    assert positive_keywords["elapsed_s"]["sum"] == 2
    assert positive_keywords["native_totals"]["total_tokens"] == {
        "sum": 8,
        "available": 1,
        "attempts": 1
    }
    assert positive_keywords["native_totals"]["cached_input_tokens"][
        "sum"] == 2
    assert positive_keywords["native_totals"]["uncached_input_tokens"][
        "sum"] == 6
    assert positive_keywords["command_count"] == {
        "sum": 3,
        "available": 1,
        "attempts": 1
    }
    assert positive_keywords["first_skill_read_s"] == {
        "sum": 1.25,
        "available": 1,
        "attempts": 1
    }
    assert positive_keywords["first_expected_record_read_s"] == {
        "sum": 1.75,
        "available": 1,
        "attempts": 1
    }
    assert summary["selective_outcomes"]["required_positive_misses"][
        "current"] == 1
    assert summary["selective_outcomes"]["negative_unwanted_loads"][
        "current"] == 0
    assert summary["limitations"]["per_file_tokens"] is None
    assert (output / "summary.json").is_file()
    assert "diagnostic" in (output / "summary.md").read_text().lower()


def test_report_lists_scheduled_attempt_missing_result_without_fabricating_usage(
        tmp_path):
    output = tmp_path / "campaign"
    output.mkdir()
    worker = {
        "id": "attempt-001",
        "case": "positive",
        "repetition": 1,
        "arm": "candidate"
    }
    (output / "manifest.json").write_text(
        json.dumps({
            "cases": [_case()],
            "schedule": [{
                "pair": 1,
                "variant": "current",
                "case": "positive",
                "repetition": 1,
                "worker_run": worker
            }],
        }))
    partial = output / "runs/attempt-001"
    partial.mkdir(parents=True)
    (partial / "events.jsonl").write_text("partial\n")
    summary = selective.report(output)
    assert summary["finalized_attempts"] == 0
    assert summary["missing_attempts"] == [{
        "id":
        "attempt-001",
        "case":
        "positive",
        "variant":
        "current",
        "partial_evidence": ["events.jsonl"],
    }]
    assert "Missing scheduled attempts: 1." in (output /
                                                "summary.md").read_text()


def test_reanalyze_writes_distinct_corrected_report_with_consumed_provenance(
        tmp_path):
    output = tmp_path / "campaign"
    run_id = "attempt-001"
    run_dir = output / "runs" / run_id
    run_dir.mkdir(parents=True)
    case = {**_case(), "files": []}
    worker = {
        "id": run_id,
        "case": case["id"],
        "repetition": 1,
        "arm": "candidate"
    }
    module_path = str(Path(selective.__file__).resolve())
    manifest = {
        "cases": [case],
        "schedule": [{
            "pair": 1,
            "variant": "current",
            "case": case["id"],
            "repetition": 1,
            "worker_run": worker
        }],
        "snapshot_inventory": {
            "current": {
                "SKILL.md": "hash"
            },
            "keywords": {}
        },
        "evaluator_inventory": {
            module_path: "original-parser-hash"
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    original_result = {
        **worker,
        "variant": "current",
        "worker_completed": True,
        "status": "completed",
        "worker_elapsed_s": 4.5,
        "worker_telemetry": {
            "totals": {
                "input_tokens": 12,
                "cached_input_tokens": 5,
                "output_tokens": 3,
                "reasoning_output_tokens": 1,
                "total_tokens": 15
            }
        },
        "discovery": {
            "activation": "unknown"
        },
    }
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(original_result))
    event = _command(
        '/bin/bash -lc "sed -n \'1,220p\' skills/cudaq-algorithms/SKILL.md"')
    events_path = run_dir / "events.jsonl"
    events_path.write_text(json.dumps(event) + "\n")
    before = {
        path: path.read_bytes()
        for path in (output / "manifest.json", result_path, events_path)
    }
    destination = output / "parser-reanalysis-v2"

    payload = selective.reanalyze(output, destination)

    assert payload["summary"]["activation"]["positive"]["current"]["true"] == 1
    assert payload["summary"]["per_case"]["positive"]["current"][
        "native_totals"]["total_tokens"]["sum"] == 15
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["original_parser_sha256"] == "original-parser-hash"
    assert provenance["corrected_parser_sha256"] == selective._hash(
        Path(selective.__file__))
    assert set(provenance["consumed_files"]) == {
        "manifest.json", f"runs/{run_id}/result.json",
        f"runs/{run_id}/events.jsonl"
    }
    assert all(path.read_bytes() == content
               for path, content in before.items())
    assert not (destination / "runs").exists()
    with pytest.raises(FileExistsError, match="destination already exists"):
        selective.reanalyze(output, destination)
