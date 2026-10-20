# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for caller-supplied structural JSON contracts, not science."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "skills/cudaq-algorithms/scripts/check_output.py"


@pytest.fixture(scope="module")
def checker():
    spec = importlib.util.spec_from_file_location("output_contract_checker", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def field(pointer="/result", kind="number", shape="[]"):
    return (pointer, kind, shape)


def code(checker, text, fields=None, format="json"):
    report = checker.check_text(text, [field()] if fields is None else fields, format)
    assert "structural" in report["scope"].lower()
    assert "scientific" in report["scope"].lower()
    assert report["counts"]["errors"] == len(report["errors"])
    assert report["incomplete"] == bool(report["unresolved"])
    return checker.exit_code(report), report


@pytest.mark.parametrize("text,fields", [
    ('{"result": 3}', [field()]),
    ('{"result": 3.5}', [field()]),
    ('{"result": [[1,2],[3,4]]}', [field(shape="[2,2]")]),
    ('{"result": []}', [field(shape="[0]")]),
    ('{"result": [[],[]]}', [field(shape="[2,0]")]),
    ('{"result": true}', [field(kind="boolean")]),
    ('{"result": 3}', [field(kind="integer")]),
    ('{"result": "hello"}', [field(kind="string")]),
    ('{"result": null}', [field(kind="null")]),
    ('{"result": {"extra": 1}}', [field(kind="object")]),
    ('[[1,2],[3,4]]', [field("", shape="[2,2]")]),
    ('{"a/b":{"~key": [1,2]}}', [field("/a~1b/~0key/1")]),
    ('{"": 1, "01":2,"-":3}', [field("/"), field("/01"), field("/-")]),
    ('{"result": 1,"extra":{"unchecked":"anything"}}', [field()]),
    ('{"result": 1}', [field(), field()]),
])
def test_supported_contracts(checker, text, fields):
    status, report = code(checker, text, fields)
    assert status == 0
    assert report["counts"]["checked"] == len(set(fields))


@pytest.mark.parametrize("text,fields", [
    ('{"result": {"row":[1,2]}}', [field(shape="[1,2]")]),
    ('{"result": [[1,2],[3]]}', [field(shape="[2,2]")]),
    ('{"result": [1,2]}', [field(shape="[3]")]),
    ('{"result": [1,"2"]}', [field(shape="[2]")]),
    ('{"result": true}', [field()]),
    ('{"result": false}', [field(kind="integer")]),
    ('{"result": 1.0}', [field(kind="integer")]),
    ('{"result": "3"}', [field()]),
    ('{"result": []}', [field()]),
    ('{}', [field()]),
    ('{"result": 2}', [field("/result/child")]),
    ('{"result": [2]}', [field("/result/1")]),
])
def test_shape_type_and_missing_errors(checker, text, fields):
    status, report = code(checker, text, fields)
    assert status == 1
    assert report["errors"]
    assert report["counts"]["checked"] == 0


@pytest.mark.parametrize("text", [
    '', '{', '{"result":1} trailing', '{"result":1}\n{"result":2}',
    '{"result":1,"result":2}', '{"result":{"a":1,"a":2}}',
    '{"result":NaN}', '{"result":Infinity}', '{"result":-Infinity}',
])
def test_invalid_or_nonfinite_json(checker, text):
    assert code(checker, text)[0] == 1


@pytest.mark.parametrize("literal", ["1e400", "-1e400", "1e10000", "-1e10000"])
def test_finite_numeric_representation_limit_is_incomplete(checker, literal):
    status, report = code(checker, '{"result":' + literal + '}')
    assert status == 2
    assert report["incomplete"]
    assert not report["errors"]


def test_cli_finite_numeric_representation_limit_is_incomplete(tmp_path):
    proc, report = run_cli(tmp_path, '{"result":1e400}')
    assert proc.returncode == 2
    assert report["incomplete"] and not report["errors"]


@pytest.mark.parametrize("fields", [
    [], [field(kind="array")], [field(shape="[true]")],
    [field(shape="[-1]")], [field(shape="[1.0]")],
    [field(shape="{}")], [field(shape="bad")], [field(shape="[NaN]")],
    [field("result")], [field("/bad~")], [field("/bad~2")],
    [field(), field(kind="string")], [field(), field(shape="[1]")],
])
def test_invalid_contracts_are_incomplete(checker, fields):
    status, report = code(checker, '{"result":1}', fields)
    assert status == 2
    assert report["incomplete"]
    assert not report["errors"]


@pytest.mark.parametrize("index", ["01", "-1", "-", "+0", "1.0", "١"])
def test_invalid_array_pointer_indices(checker, index):
    assert code(checker, '[1,2]', [field("/" + index)])[0] == 2


def test_large_array_index_is_reported_not_crash(checker):
    assert code(checker, '[1]', [field("/" + "9" * 5000)])[0] in (1, 2)


@pytest.mark.parametrize("text,shape", [
    ('[]', '[0,4]'), ('[[],[]]', '[2,0,4]'),
])
def test_unobservable_trailing_dimensions_are_incomplete(checker, text, shape):
    status, report = code(checker, text, [field("", shape=shape)])
    assert status == 2
    assert report["counts"]["checked"] == 0


def test_errors_and_incomplete_are_both_retained(checker):
    status, report = code(checker, '{"a":[],"b":"bad"}',
                          [field("/a", shape="[0,2]"), field("/b")])
    assert status == 1
    assert report["errors"] and report["unresolved"]


def test_last_line_selection_never_falls_back(checker):
    assert code(checker, 'progress\n{"result":1}\n \n', format="last-json-line")[0] == 0
    assert code(checker, '{"result":1}\nnot JSON\n', format="last-json-line")[0] == 1
    assert code(checker, 'progress\n{"result":1}', format="json")[0] == 1
    assert code(checker, '{"result":1}', format="unsupported")[0] == 2


def test_depth_limit_ignores_brackets_in_strings(checker):
    assert code(checker, json.dumps({"result": "[" * 100}), [field(kind="string")])[0] == 0
    assert code(checker, '[' * 65 + '1' + ']' * 65, [field("")])[0] == 2
    assert code(checker, '[' * 64 + '1' + ']' * 64,
                [field("", shape=json.dumps([1] * 64))])[0] == 0


def test_input_size_and_large_integer_are_bounded(checker, tmp_path):
    large = tmp_path / "large.json"
    large.write_bytes(b" " * (checker.MAX_BYTES + 1))
    assert checker.exit_code(checker.check_file(large, [field()])) == 2
    assert code(checker, ' ' * (checker.MAX_BYTES + 1))[0] == 2
    # Python's optional integer-string conversion limit must never crash the CLI.
    status, _ = code(checker, '{"result":' + '9' * 5000 + '}', [field(kind="integer")])
    assert status in (0, 2)
    assert code(checker, '{"result":' + '9' * 300 + '}', [field(kind="integer")])[0] == 0


def test_file_io_and_preservation(checker, tmp_path):
    source = tmp_path / "result.json"
    original = b'{"result": [1,2], "script": "__import__(\\"os\\").system(\\"false\\")"}'
    source.write_bytes(original)
    assert checker.exit_code(checker.check_file(source, [field(shape="[2]")])) == 0
    assert source.read_bytes() == original
    assert checker.exit_code(checker.check_file(tmp_path / "missing", [field()])) == 2
    assert checker.exit_code(checker.check_file(tmp_path, [field()])) == 2
    source.write_bytes(b'\xff')
    assert checker.exit_code(checker.check_file(source, [field()])) == 1


def test_wide_array_and_mixed_evidence_within_one_field(checker):
    assert code(checker, '[' + ','.join(['0'] * 100000) + ']',
                [field("", shape="[100000]")])[0] == 0
    status, report = code(checker, '[[],[1]]', [field("", shape="[2,0,3]")])
    assert status == 1
    assert report["errors"] and report["unresolved"]


def test_contract_resource_limits(checker):
    assert code(checker, '[]', [field("", shape=json.dumps([0] * 65))])[0] == 2
    assert code(checker, '{}', [field('/' + 'x' * 8192)])[0] == 2
    assert code(checker, '{}', [field('/' + str(n)) for n in range(257)])[0] == 2


def run_cli(tmp_path, text, fields=None, extra=()):
    source = tmp_path / "result.json"
    source.write_text(text)
    args = [sys.executable, "-B", str(SCRIPT), "--input", str(source), "--format", "json"]
    for args_field in [field()] if fields is None else fields:
        args.extend(["--field", *args_field])
    proc = subprocess.run(args + list(extra), capture_output=True, text=True)
    assert proc.stdout == ""
    return proc, json.loads(proc.stderr)


@pytest.mark.parametrize("text,fields,expected", [
    ('{"result":1}', [field()], 0),
    ('{"result":{}}', [field(shape="[1]")], 1),
    ('{"result":[]}', [field(shape="[0,2]")], 2),
    ('{"result":1}', [], 2),
    ('{"result":1}', [field(kind="unsupported")], 2),
    ('{"a":[],"b":"bad"}', [field("/a", shape="[0,1]"), field("/b")], 1),
])
def test_cli_exit_codes_and_stderr(tmp_path, text, fields, expected):
    proc, report = run_cli(tmp_path, text, fields)
    assert proc.returncode == expected
    assert "structural" in report["scope"].lower()
    assert "scientific" in report["scope"].lower()


def test_cli_diagnostics_do_not_dump_values(tmp_path):
    proc, report = run_cli(tmp_path, '{"result":"private payload value"}')
    assert proc.returncode == 1
    assert "private payload value" not in proc.stderr
    assert report["errors"][0]["pointer"] == "/result"


def test_cli_invalid_argument_is_stderr_json(tmp_path):
    proc, report = run_cli(tmp_path, '{"result":1}', extra=("--not-an-option",))
    assert proc.returncode == 2 and report["incomplete"]


@pytest.mark.parametrize("columns", [40, 70, 80, 120])
def test_cli_help_describes_contract_syntax(monkeypatch, columns):
    monkeypatch.setenv("COLUMNS", str(columns))
    proc = subprocess.run([sys.executable, "-B", str(SCRIPT), "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stderr == ""
    # argparse wraps prose according to terminal width; all terms must remain.
    help_text = " ".join(proc.stdout.split())
    for text in ("JSON Pointer", "number", "integer", "string", "boolean",
                 "null", "object", "[]", "[2,3]", "empty", "root"):
        assert text in help_text


def test_same_shell_preserves_producer_failure_and_output(tmp_path):
    capture = tmp_path / "capture.txt"
    quoted_check = shlex.join([sys.executable, "-B", str(SCRIPT), "--input", str(capture),
                              "--format", "last-json-line", "--field", "/result", "number", "[]"])
    python = shlex.quote(sys.executable)
    for producer, expected, stdout in [
        ('import sys; print("{\\"result\\":1}"); sys.exit(7)', 7, ''),
        ('print("progress"); print("{\\"result\\":1}")', 0, 'progress\n{"result":1}\n'),
    ]:
        command = f'{python} -B -c {shlex.quote(producer)} >{shlex.quote(str(capture))} && {quoted_check} && cat {shlex.quote(str(capture))}'
        proc = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
        assert proc.returncode == expected
        assert proc.stdout == stdout
        if expected == 7:
            assert proc.stderr == ""


def test_helper_imports_no_scientific_or_execution_modules():
    import ast
    tree = ast.parse(SCRIPT.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split('.')[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imports.add((node.module or '').split('.')[0])
    assert imports <= {"__future__", "argparse", "json", "math", "os", "pathlib", "stat", "sys"}
