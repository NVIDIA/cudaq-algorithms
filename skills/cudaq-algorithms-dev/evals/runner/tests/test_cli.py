import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner import cli


def test_preflight_saves_partial_results_and_cleanly_cancels(
        monkeypatch, tmp_path, capsys):
    output = tmp_path / "preflight.json"
    monkeypatch.setattr(cli, "models_from", lambda args: [{
        "id": "nemo"
    }, {
        "id": "kimi"
    }])

    def preflight(model, timeout, *, progress):
        result = {
            "id": model["id"],
            "status": "running",
            "active_phase": "tool_call",
            "ok": None,
            "completion": {
                "ok": True,
                "latency_seconds": 1
            }
        }
        progress(result)
        saved = json.loads(output.read_text())
        assert saved["models"][-1]["completion"]["ok"]
        if model["id"] == "kimi":
            assert saved["models"][0]["ok"]
            raise KeyboardInterrupt
        return {
            **result, "status": "complete",
            "active_phase": None,
            "ok": True,
            "tool_call": {
                "ok": True,
                "latency_seconds": 1
            }
        }

    monkeypatch.setattr(cli.providers, "preflight", preflight)
    assert cli.main(["preflight", "--output", str(output)]) == 130
    saved = json.loads(output.read_text())["models"]
    assert saved[0]["ok"] is True
    assert saved[1]["completion"]["ok"] is True
    assert saved[1]["status"] == "interrupted"
    assert saved[1]["ok"] is False
    assert "saved" in capsys.readouterr().err.lower()


def test_preflight_reports_heartbeat_during_blocking_request(
        monkeypatch, tmp_path, capsys):
    saw_waiting = threading.Event()
    monkeypatch.setattr(cli, "HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(cli, "models_from", lambda args: [{"id": "kimi"}])
    original_print = print

    def observe_print(*args, **kwargs):
        original_print(*args, **kwargs)
        if "still waiting" in " ".join(str(arg) for arg in args):
            saw_waiting.set()

    monkeypatch.setattr(cli, "print", observe_print, raising=False)

    def preflight(model, timeout, *, progress):
        progress({
            "id": "kimi",
            "status": "running",
            "active_phase": "completion",
            "ok": None
        })
        assert saw_waiting.wait(
            2), "No heartbeat while provider request was blocked"
        return {
            "id": "kimi",
            "status": "complete",
            "active_phase": None,
            "ok": True
        }

    monkeypatch.setattr(cli.providers, "preflight", preflight)
    assert cli.main(["preflight", "--output",
                     str(tmp_path / "report.json")]) == 0
    assert "completion" in capsys.readouterr().err


def test_pending_only_campaign_is_not_success(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli.campaign, "run",
        lambda *a, **kw: {"nemo": {
            "pending": 620,
            "unfinished": 0
        }})
    assert cli.main(["run", str(tmp_path)]) != 0


def test_limited_successful_batch_can_leave_remaining_work(
        monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli.campaign, "run", lambda *a, **kw:
        {"nemo": {
            "pending": 618,
            "answered": 2,
            "unfinished": 0
        }})
    assert cli.main(["run", str(tmp_path), "--limit", "2"]) == 0


@pytest.mark.parametrize("option,value", [("--limit", "0"), ("--limit", "-1")])
def test_nonpositive_limit_is_rejected(monkeypatch, tmp_path, option, value):
    monkeypatch.setattr(
        cli.campaign, "run",
        lambda *a, **kw: pytest.fail("must validate before running"))
    assert cli.main(["run", str(tmp_path), option, value]) != 0
