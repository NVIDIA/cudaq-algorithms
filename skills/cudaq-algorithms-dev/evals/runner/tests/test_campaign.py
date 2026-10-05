import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner import campaign


def test_schedule_covers_every_pair_without_duplicate():
    cases = [{"id": f"case-{i}"} for i in range(62)]
    rows = campaign.schedule(cases, list(range(5)))
    assert len(rows) == 620
    assert len({(r["case_id"], r["seed"], r["arm"]) for r in rows}) == 620
    for offset in range(0, len(rows), 2):
        assert rows[offset]["case_id"] == rows[offset + 1]["case_id"]
        assert {r["arm"]
                for r in rows[offset:offset + 2]} == {"baseline", "skill"}


def test_resume_never_replaces_started_or_terminal_attempt(tmp_path):
    row = {"case_id": "case-0", "seed": 0, "arm": "skill"}
    path = campaign.attempt_path(tmp_path, "nemo", row)
    assert campaign.reserve(path)
    assert not campaign.reserve(path)
    (path / "result.json").write_text('{"outcome":"backend_error"}')
    assert not campaign.reserve(path)
    assert json.loads(
        (path / "result.json").read_text())["outcome"] == "backend_error"


@pytest.mark.parametrize("alias", ["../escape", "a/b", "a\\b", "", ".", ".."])
def test_path_components_are_validated(tmp_path, alias):
    with pytest.raises(ValueError):
        campaign.attempt_path(tmp_path, alias, {
            "case_id": "x",
            "seed": 0,
            "arm": "skill"
        })


def test_worker_sees_original_prompt_and_not_private_rubric():
    case = {
        "id": "x",
        "prompt": "Original question.",
        "assertions": ["SECRET"],
        "expected_output": "PRIVATE",
        "files": []
    }
    messages = campaign.worker_messages(case, "baseline", "listed", "unused")
    assert messages[-1] == {"role": "user", "content": "Original question."}
    assert "SECRET" not in json.dumps(messages)
    assert "PRIVATE" not in json.dumps(messages)
    assert "/skill/SKILL.md" not in json.dumps(messages)
    assert "/skill/SKILL.md" in json.dumps(
        campaign.worker_messages(case, "skill", "listed", "BODY"))
    assert "BODY" not in json.dumps(
        campaign.worker_messages(case, "skill", "listed", "BODY"))


def test_missing_usage_is_unknown_even_after_known_turn():
    assert campaign.total_tokens([{"usage": {"total_tokens": 3}}, {}]) is None
    assert campaign.total_tokens([]) is None
    assert campaign.total_tokens([{"usage": None}]) is None
    assert campaign.total_tokens([{
        "usage": {
            "total_tokens": 3
        }
    }, {
        "usage": {
            "total_tokens": 4
        }
    }]) == 7


def test_prepare_preserves_virtualenv_python_symlink(tmp_path):
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    root = tmp_path / "campaign"
    spec = campaign.prepare(
        root,
        campaign.EVALS.parents[2], [{
            "id": "fake",
            "model": "fake/model"
        }],
        pilot=True,
        runtime_python=python,
        case_ids=["science-block-encoding-heralded-observables"])
    assert spec["settings"]["runtime_python"] == str(python)
    assert spec["case_contracts"] == campaign.read_json(root /
                                                        "case_contracts.json")


def test_backend_failure_is_terminal_and_has_evidence(tmp_path, monkeypatch):
    from runner import providers, workspace
    attempt = tmp_path / "attempts" / "nemo" / "case" / "0" / "baseline"
    attempt.mkdir(parents=True)

    def fail(*args, **kwargs):
        raise providers.ProviderError("backend", 503, "unavailable", True)

    monkeypatch.setattr(providers, "complete", fail)
    monkeypatch.setattr(
        workspace, "stage_workspace", lambda *a, **kw: {
            "workspace_path": str(tmp_path),
            "skill_path": None
        })
    spec = {
        "settings": {
            "request_timeout": 1
        },
        "protocol": {
            "budget_seconds": 1,
            "tool_call_limit": 2,
            "skill_exposure": "listed"
        }
    }
    result = campaign.run_attempt(tmp_path, attempt, {
        "id": "case",
        "prompt": "Hi",
        "files": []
    }, {"alias": "nemo"}, {
        "seed": 0,
        "arm": "baseline",
        "case_id": "case"
    }, spec)
    assert result["outcome"] == "backend_error"
    assert result["resources"]["tokens"] is None
    assert (attempt / "result.json").exists()
    assert (attempt / "transcript.jsonl").stat().st_size > 0


def test_two_arm_pipeline_executes_tools_and_resume_does_not_repeat(
        tmp_path, monkeypatch):
    from runner import providers, workspace
    root = tmp_path / "campaign"
    campaign.prepare(root,
                     campaign.EVALS.parents[2], [{
                         "id": "fake",
                         "model": "fake/model"
                     }],
                     pilot=True,
                     case_ids=["science-block-encoding-heralded-observables"],
                     budget=10)
    monkeypatch.setattr(workspace, "preflight", lambda **kw: {
        "available": True,
        "isolated": True
    })
    monkeypatch.setattr(providers, "preflight", lambda *a, **kw: {"ok": True})
    original = workspace.run_command
    monkeypatch.setattr(
        workspace, "run_command",
        lambda command, path, **kw: original(command,
                                             path,
                                             timeout=kw["timeout"],
                                             isolation="trusted",
                                             log_dir=kw["log_dir"]))
    requests = []

    def complete(model, messages, tools, timeout):
        requests.append(messages[-1])
        if not any(m["role"] == "tool" for m in messages):
            message = {
                "role":
                "assistant",
                "content":
                None,
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name":
                        "shell",
                        "arguments":
                        json.dumps({
                            "command":
                            "python3 -c 'from pathlib import Path; Path(\"artifact.txt\").write_text(str(2+2)); print(2+2)'"
                        })
                    }
                }]
            }
        else:
            assert '"stdout": "4\\n"' in messages[-1]["content"]
            message = {
                "role":
                "assistant",
                "content":
                "Observed 4; this is a test fixture, not a scientific answer."
            }
        return {
            "choices": [{
                "message": message,
                "finish_reason": "stop"
            }],
            "usage": {
                "total_tokens": 7
            }
        }

    monkeypatch.setattr(providers, "complete", complete)
    status = campaign.run(root, emit=lambda _: None)
    assert status["fake"]["answered"] == 2
    assert len(requests) == 4
    for path in (root / "attempts").glob("*/*/*/*"):
        assert (path / "workspace/artifact.txt").read_text() == "4"
        result = campaign.read_json(path / "result.json")
        assert result["resources"]["tokens"] == 14
        assert result["resources"]["tool_calls"] == 1
        assert len(list((path / "tool-logs").glob("*.stdout.log"))) == 1
    campaign.run(root, emit=lambda _: None)
    assert len(requests) == 4


def test_changed_protocol_refuses_resume(tmp_path):
    root = tmp_path / "campaign"
    spec = campaign.prepare(root,
                            campaign.EVALS.parents[2], [{
                                "id": "fake",
                                "model": "fake/model"
                            }],
                            pilot=True)
    spec["protocol"]["skill_exposure"] = "injected"
    campaign.write_json(root / "campaign.json", spec)
    with pytest.raises(ValueError, match="configuration changed"):
        campaign.verify_inputs(root)


def test_terminal_record_alone_cannot_be_reserved_again(tmp_path):
    (tmp_path / "result.json").write_text('{"outcome":"answered"}')
    assert not campaign.reserve(tmp_path)
