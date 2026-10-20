#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check explicit structural constraints against saved JSON, using only stdlib.

No producer is executed and no payload is changed. Extra fields, meanings, row
identities, units, values and scientific correctness are not validated. A caller
must supply the requested contract, rather than infer it from observed output.
Exit 0: supplied constraints matched; 1: malformed data/definite mismatch;
2: invalid contract, unavailable data or incomplete check. Diagnostics use stderr.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import stat
import sys

MAX_BYTES = 8 * 1024 * 1024
MAX_DEPTH = 64
MAX_FIELDS = 256
MAX_CONTRACT_TEXT = 8192
TYPES = {"number", "integer", "string", "boolean", "null", "object"}
SCOPE = ("Structural constraints supplied by the caller only; extra fields, "
         "semantic meaning, numerical and scientific correctness are unverified.")


class InvalidData(ValueError):
    pass


class Incomplete(ValueError):
    pass


def _report():
    return {"scope": SCOPE, "constraints": [], "counts": {}, "errors": [],
            "unresolved": [], "incomplete": False}


def _finish(report, checked=0):
    report["incomplete"] = bool(report["unresolved"])
    report["counts"] = {"constraints": len(report["constraints"]),
                        "checked": checked, "errors": len(report["errors"]),
                        "incomplete": len(report["unresolved"])}
    return report


def exit_code(report):
    """Definite errors take precedence, while incomplete evidence is retained."""
    return 1 if report["errors"] else 2 if report["incomplete"] else 0


def _issue(report, kind, reason, **details):
    report[kind].append({"reason": reason, **details})


def _pointer_tokens(pointer):
    if not isinstance(pointer, str) or len(pointer) > MAX_CONTRACT_TEXT:
        raise Incomplete("pointer must be a bounded string")
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise Incomplete("JSON Pointer must be empty or begin with slash")
    tokens = []
    for token in pointer[1:].split("/"):
        index = 0
        while index < len(token):
            if token[index] == "~":
                if index + 1 == len(token) or token[index + 1] not in "01":
                    raise Incomplete("JSON Pointer permits only ~0 and ~1 escapes")
                index += 1
            index += 1
        tokens.append(token.replace("~1", "/").replace("~0", "~"))
    if len(tokens) > MAX_DEPTH:
        raise Incomplete("pointer exceeds supported nesting limit")
    return tokens


def _contracts(fields):
    if not isinstance(fields, (list, tuple)) or not 1 <= len(fields) <= MAX_FIELDS:
        raise Incomplete(f"supply 1 to {MAX_FIELDS} explicit field constraints")
    result = {}
    for item in fields:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise Incomplete("each field requires POINTER TYPE JSON_SHAPE")
        pointer, kind, shape_text = item
        _pointer_tokens(pointer)
        if not isinstance(kind, str) or kind not in TYPES:
            raise Incomplete("unsupported leaf type")
        if not isinstance(shape_text, str) or len(shape_text) > MAX_CONTRACT_TEXT:
            raise Incomplete("shape must be a bounded JSON list of dimensions")
        try:
            shape = json.loads(shape_text)
        except (ValueError, RecursionError):
            raise Incomplete("shape must be a JSON list of nonnegative integers") from None
        if (not isinstance(shape, list) or len(shape) > MAX_DEPTH
                or any(type(n) is not int or n < 0 for n in shape)):
            raise Incomplete("shape must have at most 64 nonnegative integer dimensions")
        constraint = {"pointer": pointer, "type": kind, "shape": shape}
        if pointer in result and constraint != result[pointer]:
            raise Incomplete("conflicting constraints for the same pointer")
        result[pointer] = constraint
    return list(result.values())


def _guard_depth(text):
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise Incomplete("JSON exceeds supported 64-level nesting limit")
        elif char in "]}":
            depth -= 1


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidData("duplicate JSON object key")
        result[key] = value
    return result


def _constant(_value):
    raise InvalidData("nonfinite JSON numbers are unsupported")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise Incomplete("finite JSON number exceeds the supported float range")
    return number


def _payload(text, format):
    if format == "last-json-line":
        lines = text.splitlines()
        text = next((line for line in reversed(lines) if line.strip()), "")
    _guard_depth(text)
    try:
        return json.loads(text, object_pairs_hook=_object,
                          parse_constant=_constant, parse_float=_float)
    except (InvalidData, Incomplete):
        raise
    except json.JSONDecodeError as error:
        raise InvalidData(f"malformed JSON at line {error.lineno}, column {error.colno}") from None
    except (ValueError, OverflowError, RecursionError, MemoryError):
        raise Incomplete("JSON exceeds supported parser or numeric representation limits") from None


def _kind(value):
    return {dict: "object", list: "array", str: "string", bool: "boolean",
            int: "integer", float: "number", type(None): "null"}.get(type(value), "unsupported")


def _lookup(payload, pointer):
    value = payload
    for token in _pointer_tokens(pointer):
        if isinstance(value, dict):
            if token not in value:
                raise InvalidData("required field is absent")
            value = value[token]
        elif isinstance(value, list):
            if (not token or any(c not in "0123456789" for c in token)
                    or len(token) > 1 and token[0] == "0"):
                raise Incomplete("array pointer index must be canonical nonnegative decimal")
            # Compare decimal strings first: an arbitrarily large index need not
            # be converted to a Python integer just to determine it is absent.
            bound = str(len(value))
            if len(token) > len(bound) or len(token) == len(bound) and token >= bound:
                raise InvalidData("required array index is absent")
            value = value[int(token)]
        else:
            raise InvalidData("pointer traverses a non-container value")
    return value


def _matches(value, kind):
    if kind == "number":
        return type(value) in (int, float)
    return _kind(value) == kind


def _check_field(value, constraint):
    """Return at most one definite and one incomplete diagnostic per field."""
    shape, kind = constraint["shape"], constraint["type"]
    failures, unresolved = [], []
    # Iterator frames keep additional traversal storage proportional to depth,
    # even when a bounded input contains a very wide array.
    pending = [(iter((value,)), 0)]
    while pending:
        values, depth = pending[-1]
        try:
            item = next(values)
        except StopIteration:
            pending.pop()
            continue
        if depth == len(shape):
            if not _matches(item, kind) and not failures:
                failures.append({"reason": "leaf type mismatch", "expected": kind,
                                 "actual": _kind(item)})
        elif not isinstance(item, list):
            if not failures:
                failures.append({"reason": "array required at declared dimension",
                                 "dimension": depth, "expected": "array", "actual": _kind(item)})
        elif len(item) != shape[depth]:
            if not failures:
                failures.append({"reason": "array length mismatch", "dimension": depth,
                                 "expected_length": shape[depth], "actual_length": len(item)})
        elif not item and depth + 1 < len(shape):
            if not unresolved:
                unresolved.append({"reason": "empty array cannot establish trailing dimensions",
                                   "dimension": depth})
        else:
            pending.append((iter(item), depth + 1))
    return failures, unresolved


def _prepare(fields, format):
    report = _report()
    try:
        report["constraints"] = _contracts(fields)
        if format not in ("json", "last-json-line"):
            raise Incomplete("format must be json or last-json-line")
    except Incomplete as error:
        _issue(report, "unresolved", str(error))
    return report


def _check_text(text, report, format):
    try:
        if not isinstance(text, str):
            raise Incomplete("input must be decoded JSON text")
        if len(text) > MAX_BYTES or len(text.encode("utf-8")) > MAX_BYTES:
            raise Incomplete("input exceeds the 8 MiB limit")
        payload = _payload(text, format)
    except (Incomplete, MemoryError) as error:
        _issue(report, "unresolved", str(error) or "insufficient memory to check input")
        return _finish(report)
    except (InvalidData, UnicodeError) as error:
        _issue(report, "errors", str(error) if isinstance(error, InvalidData) else "invalid UTF-8 text")
        return _finish(report)
    checked = 0
    for constraint in report["constraints"]:
        pointer = constraint["pointer"]
        try:
            value = _lookup(payload, pointer)
            errors, unresolved = _check_field(value, constraint)
            report["errors"].extend({"pointer": pointer, **item} for item in errors)
            report["unresolved"].extend({"pointer": pointer, **item} for item in unresolved)
            checked += not errors and not unresolved
        except InvalidData as error:
            _issue(report, "errors", str(error), pointer=pointer)
        except (Incomplete, MemoryError) as error:
            _issue(report, "unresolved", str(error) or "insufficient memory to check field", pointer=pointer)
    return _finish(report, checked)


def check_text(text, fields, format="json"):
    """Validate text against (pointer, type, JSON-shape-string) triples."""
    report = _prepare(fields, format)
    return _finish(report) if report["unresolved"] else _check_text(text, report, format)


def check_file(path, fields, format="json"):
    """Read a regular saved file with bounded bytes; never execute or modify it."""
    report = _prepare(fields, format)
    if report["unresolved"]:
        return _finish(report)
    try:
        path = Path(path)
        if not stat.S_ISREG(path.stat().st_mode):
            raise Incomplete("input must be a regular saved file")
        with path.open("rb") as source:
            raw = source.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise Incomplete("input exceeds the 8 MiB limit")
        text = raw.decode("utf-8")
    except UnicodeError:
        _issue(report, "errors", "input is not valid UTF-8")
        return _finish(report)
    except (OSError, ValueError, TypeError, MemoryError) as error:
        _issue(report, "unresolved", str(error) if isinstance(error, Incomplete)
               else "input could not be read as a regular saved file")
        return _finish(report)
    return _check_text(text, report, format)


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        # Do not echo malformed arguments, which can contain the output itself.
        raise Incomplete("invalid command arguments; use --help for syntax")


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--input", required=True, help="existing saved output file")
    parser.add_argument("--format", choices=("json", "last-json-line"), required=True)
    parser.add_argument("--field", action="append", nargs=3, default=[],
                        metavar=("POINTER", "TYPE", "JSON_SHAPE"),
                        help=("repeat a required field: POINTER is JSON Pointer "
                              "(/name, ~0 for tilde, ~1 for slash; empty string "
                              "selects the root); TYPE is number, integer, "
                              "string, boolean, null or object; JSON_SHAPE is "
                              "a list of nonnegative dimensions, e.g. [2,3], "
                              "or [] for a leaf; empty arrays cannot establish "
                              "further dimensions"))
    try:
        args = parser.parse_args(argv)
        report = check_file(args.input, args.field, args.format)
    except Incomplete as error:
        report = _report()
        _issue(report, "unresolved", str(error))
        _finish(report)
    print(json.dumps(report, ensure_ascii=True, separators=(",", ":")), file=sys.stderr)
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
