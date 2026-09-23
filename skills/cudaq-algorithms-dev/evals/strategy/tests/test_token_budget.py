# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real tokenizer checks; set NEMOTRON_TOKENIZER_ASSETS to the pinned assets."""
import importlib.util
import os
from pathlib import Path
import shutil

import pytest

MODULE = Path(__file__).resolve().parents[1] / "token_budget.py"
counter_module = None
if MODULE.exists():
    spec = importlib.util.spec_from_file_location("strategy_token_budget",
                                                  MODULE)
    counter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(counter_module)


@pytest.fixture
def assets():
    path = os.environ.get("NEMOTRON_TOKENIZER_ASSETS")
    if not path:
        pytest.skip(
            "set NEMOTRON_TOKENIZER_ASSETS for the real pinned-tokenizer checks"
        )
    return Path(path)


def request():
    return {
        "model":
        "nvidia/nemotron-3-super-120b-a12b",
        "messages": [{
            "role": "user",
            "content": "Return JSON with ok=false and count=999."
        }],
        "max_tokens":
        128,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    }


def test_matches_observed_probe_and_reserves_safety_output_and_schema(assets):
    assert counter_module is not None, "pinned token counter is not implemented"
    counter = counter_module.NemotronCounter(assets)
    value = request()
    value["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": "capability_probe",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ok": {
                        "type": "boolean",
                        "enum": [True]
                    },
                    "count": {
                        "type": "integer",
                        "enum": [7]
                    }
                },
                "required": ["ok", "count"],
                "additionalProperties": False
            }
        }
    }
    measured = counter.count_request(value)
    assert measured["input_tokens"] == 28  # Live NVIDIA probe reported 28.
    assert measured["response_format_tokens"] == 83
    assert measured["input_token_upper_bound"] == 1135
    assert measured["total_budget_tokens"] == 1263
    assert measured["within_limit"] is True
    assert measured["safety_margin_tokens"] == 1024
    assert len(measured["asset_hashes"]) == 4


@pytest.mark.parametrize("fault", ["model", "thinking", "output", "content"])
def test_unsupported_requests_fail_closed(assets, fault):
    assert counter_module is not None, "pinned token counter is not implemented"
    counter = counter_module.NemotronCounter(assets)
    value = request()
    if fault == "model": value["model"] = "different-model"
    if fault == "thinking":
        value["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True
    if fault == "output": value["max_tokens"] = 4097
    if fault == "content":
        value["messages"][0]["content"] = [{"type": "image"}]
    with pytest.raises(ValueError):
        counter.count_request(value)


def test_altered_assets_are_rejected_before_tokenization(assets, tmp_path):
    assert counter_module is not None, "pinned token counter is not implemented"
    copied = tmp_path / "assets"
    shutil.copytree(assets, copied)
    (copied / "tokenizer_config.json").write_text("{}")
    with pytest.raises(ValueError, match="asset"):
        counter_module.NemotronCounter(copied)


def test_large_input_is_counted_completely_and_rejected_by_budget(assets):
    assert counter_module is not None, "pinned token counter is not implemented"
    counter = counter_module.NemotronCounter(assets)
    value = request()
    value["messages"][0]["content"] = "alpha " * 65_000
    measured = counter.count_request(value)
    assert measured["input_tokens"] > 64_000
    assert measured["within_limit"] is False
