# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline Nemotron judge budget using four pinned official tokenizer assets.

No network, model weights, remote Python code, or filesystem writes. The
rendering matched retained NVIDIA prompt usage (probe 28; E01 4635/6093).
Provider-side changes are not measured here: reserve the complete serialized
response schema plus 1024 tokens beyond the official rendered chat count.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import stat

MODEL = "nvidia/nemotron-3-super-120b-a12b"
TOKENIZER_REPOSITORY = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
REVISION = "2dc98e2afe4face0e4ce40972a915c45368bd34a"
CONTEXT_TOKENS = 64000
SAFETY_MARGIN_TOKENS = 1024
ASSET_HASHES = {
    "tokenizer.json":
    "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7",
    "tokenizer_config.json":
    "10f93eabcb9b1602fbb991d6308e787ce1df28ee9cd7a1c6d1e8c3f338b957bc",
    "chat_template.jinja":
    "575fb74f54ed264df9047d0ecce3c98938aae953fb4f50356675706264cbb68a",
    "special_tokens_map.json":
    "e9435fefd6d838fd9fcbbc44b97a8e3ff322be7f6dfb7e4fd2468586574bb52b",
}


class NemotronCounter:
    """Load trusted bytes once; count the exact request given to the SDK."""

    def __init__(self, assets_dir):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        directory = Path(assets_dir).absolute()
        if any(path.is_symlink() for path in (directory, *directory.parents)):
            raise ValueError("unsafe tokenizer asset directory")
        assets = {}
        for name, expected in ASSET_HASHES.items():
            path = directory / name
            info = path.lstat()
            if not stat.S_ISREG(
                    info.st_mode
            ) or info.st_nlink != 1 or info.st_size > 20_000_000:
                raise ValueError("unsafe tokenizer asset")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("tokenizer asset hash mismatch")
            assets[name] = raw
        self._tokenizer = Tokenizer.from_str(
            assets["tokenizer.json"].decode("utf-8"))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        environment = ImmutableSandboxedEnvironment(trim_blocks=True,
                                                    lstrip_blocks=True)
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, ensure_ascii=False, **kwargs)
        self._template = environment.from_string(
            assets["chat_template.jinja"].decode("utf-8"))
        self.versions = {
            name: importlib.metadata.version(name)
            for name in ("tokenizers", "Jinja2")
        }

    def count_request(self, kwargs):
        """Return the measured chat count and a conservative budget reservation.

        input_token_upper_bound = chat + serialized response_format + 1024;
        total_budget_tokens additionally reserves all requested output tokens.
        This bound is calibrated to retained provider observations, not a claim
        that arbitrary future provider-side prompt modifications are known.
        """
        if kwargs.get("model") != MODEL:
            raise ValueError("unsupported token-count model")
        output = kwargs.get("max_tokens")
        if type(output) is not int or not 1 <= output <= 4096:
            raise ValueError("unsupported output budget")
        if kwargs.get("extra_body") != {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
        }:
            raise ValueError("unsupported chat-template policy")
        messages = kwargs.get("messages")
        if (not isinstance(messages, list) or not messages
                or [m.get("role") for m in messages if isinstance(m, dict)
                    ] not in (["user"], ["system", "user"]) or
                any(not isinstance(m, dict) or set(m) != {"role", "content"}
                    or not isinstance(m["content"], str) for m in messages)
                or kwargs.get("tools") or kwargs.get("functions")):
            raise ValueError("unsupported text-only judge messages")
        rendered = self._template.render(messages=messages,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
        count = lambda text: len(
            self._tokenizer.encode(text, add_special_tokens=False).ids)
        input_tokens = count(rendered)
        response_format = kwargs.get("response_format")
        schema_tokens = count(
            json.dumps(response_format, ensure_ascii=False,
                       allow_nan=False)) if response_format is not None else 0
        upper = input_tokens + schema_tokens + SAFETY_MARGIN_TOKENS
        total = upper + output
        return {
            "schema_version": 1,
            "model": MODEL,
            "tokenizer_repository": TOKENIZER_REPOSITORY,
            "revision": REVISION,
            "asset_hashes": dict(ASSET_HASHES),
            "versions": dict(self.versions),
            "input_tokens": input_tokens,
            "response_format_tokens": schema_tokens,
            "safety_margin_tokens": SAFETY_MARGIN_TOKENS,
            "input_token_upper_bound": upper,
            "reserved_output_tokens": output,
            "total_budget_tokens": total,
            "context_token_limit": CONTEXT_TOKENS,
            "within_limit": total <= CONTEXT_TOKENS,
            "rendered_utf8_bytes": len(rendered.encode()),
            "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest()
        }
