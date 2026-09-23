# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Privacy-minimal native Codex token telemetry for the end-to-end evaluator.

The collector accepts invocation-scoped OTLP/HTTP JSON logs from Codex and,
optionally, the already-parsed JSONL events emitted by ``codex exec --json``.
Only token counts and compaction markers are retained.  Raw OTLP payloads,
prompts, tool text, authentication data, and environment data are discarded.
"""

from __future__ import annotations

from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
from typing import Any

_TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
_OTEL_TOKEN_FIELDS = {
    "input_tokens": "input_token_count",
    "cached_input_tokens": "cached_token_count",
    "output_tokens": "output_token_count",
    "reasoning_output_tokens": "reasoning_token_count",
}
_CAMEL_TOKEN_FIELDS = {
    "input_tokens": "inputTokens",
    "cached_input_tokens": "cachedInputTokens",
    "output_tokens": "outputTokens",
    "reasoning_output_tokens": "reasoningOutputTokens",
    "total_tokens": "totalTokens",
}
_MAX_BODY_BYTES = 4 * 1024 * 1024


def _as_nonnegative_int(value: Any) -> int | None:
    """Decode an OTLP/native integer without turning absence into zero."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str):
        try:
            result = int(value)
        except ValueError:
            return None
        return result if result >= 0 else None
    return None


def _otlp_value(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    for key in ("intValue", "stringValue", "boolValue", "doubleValue"):
        if key in value:
            return value[key]
    return None


def _attributes(record: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    raw_attributes = record.get("attributes", [])
    if not isinstance(raw_attributes, list):
        return result
    for attribute in raw_attributes:
        if not isinstance(attribute, Mapping):
            continue
        key = attribute.get("key")
        if isinstance(key, str):
            result[key] = _otlp_value(attribute.get("value"))
    return result


def _usage_from_mapping(raw: Any,
                        *,
                        camel_case: bool = False) -> dict[str, int | None]:
    if not isinstance(raw, Mapping):
        return {field: None for field in _TOKEN_FIELDS}
    names = _CAMEL_TOKEN_FIELDS if camel_case else {
        field: field
        for field in _TOKEN_FIELDS
    }
    usage = {
        field: _as_nonnegative_int(raw.get(name))
        for field, name in names.items()
    }
    if usage["total_tokens"] is None:
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        if input_tokens is not None and output_tokens is not None:
            # Cached input is a subset of input; reasoning output is a subset
            # of output.  Neither is added again.
            usage["total_tokens"] = input_tokens + output_tokens
    return usage


class _CollectorHTTPServer(ThreadingHTTPServer):
    daemon_threads = False
    allow_reuse_address = False

    def __init__(self, collector: "TelemetryCollector") -> None:
        self._connections: set[socket.socket] = set()
        self._connections_lock = threading.Lock()
        super().__init__(("127.0.0.1", 0), _OTLPHandler)
        self.collector = collector

    def get_request(self) -> tuple[socket.socket, Any]:
        request, address = super().get_request()
        with self._connections_lock:
            self._connections.add(request)
        return request, address

    def shutdown_request(self, request: socket.socket) -> None:
        try:
            super().shutdown_request(request)
        finally:
            with self._connections_lock:
                self._connections.discard(request)

    def server_close(self) -> None:
        # Stop accepted uploads as well as the listening socket. Joining all
        # handlers makes the event snapshot stable when the context returns.
        with self._connections_lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        super().server_close()


class _OTLPHandler(BaseHTTPRequestHandler):
    server: _CollectorHTTPServer

    def handle(self) -> None:
        self.connection.settimeout(5)
        try:
            super().handle()
        except OSError:
            # The collector closes incomplete uploads during shutdown.
            return

    def do_POST(
            self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path != "/v1/logs":
            self.send_error(404)
            return
        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            self.send_error(415)
            return
        length = _as_nonnegative_int(self.headers.get("Content-Length"))
        if length is None:
            self.send_error(411)
            return
        if length > _MAX_BODY_BYTES:
            self.send_error(413)
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(400)
            return
        if not isinstance(payload, Mapping):
            self.send_error(400)
            return
        self.server.collector.record_otlp(payload)
        response = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, _format: str, *args: Any) -> None:
        # Never copy request metadata into stderr or benchmark artifacts.
        return


class TelemetryCollector:
    """Collect sanitized native telemetry for one Codex attempt.

    Start the context before reading :attr:`config_args`, append those args to
    ``codex exec`` (alongside ``--ignore-user-config``), and pass every parsed
    ``--json`` event to :meth:`record_event`.  The latter supplies cumulative
    counter and compaction cross-checks without retaining event text.

    Set ``compaction_notifications_supported`` only after independently
    establishing that the supplied transport emits every compaction. The
    installed exec JSONL route does not establish that support.
    """

    def __init__(self,
                 *,
                 compaction_notifications_supported: bool = False) -> None:
        self.events: list[dict[str, Any]] = []
        self._seen: set[tuple[Any, ...]] = set()
        self._lock = threading.RLock()
        self._server: _CollectorHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._native_event_stream_observed = False
        self._compaction_notifications_supported = compaction_notifications_supported
        self._unidentified_sequence = 0
        self._modern_compaction_turns: set[tuple[Any, ...]] = set()
        self._legacy_compactions: dict[tuple[Any, ...], dict[str, Any]] = {}

    def __enter__(self) -> "TelemetryCollector":
        with self._lock:
            if self._server is not None:
                raise RuntimeError("TelemetryCollector is already active")
            self._server = _CollectorHTTPServer(self)
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="codex-otel-loopback",
                daemon=True,
            )
            self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)
        with self._lock:
            self._server = None
            self._thread = None

    @property
    def endpoint(self) -> str:
        with self._lock:
            if self._server is None:
                raise RuntimeError("TelemetryCollector must be active")
            host, port = self._server.server_address
        return f"http://{host}:{port}/v1/logs"

    @property
    def config_args(self) -> list[str]:
        endpoint = self.endpoint
        exporter = ("otel.exporter={otlp-http={"
                    f'endpoint="{endpoint}",protocol="json"'
                    "}}")
        return [
            "-c",
            exporter,
            "-c",
            "otel.log_user_prompt=false",
            "-c",
            'otel.metrics_exporter="none"',
            "-c",
            'otel.trace_exporter="none"',
        ]

    def _append(self, event: dict[str, Any], identity: tuple[Any,
                                                             ...]) -> None:
        with self._lock:
            if identity in self._seen:
                return
            self._seen.add(identity)
            self.events.append(event)

    def _unique_identity(self, source: str) -> tuple[str, int]:
        with self._lock:
            self._unidentified_sequence += 1
            return source, self._unidentified_sequence

    def record_otlp(self, payload: Mapping[str, Any]) -> None:
        """Sanitize an OTLP/HTTP JSON log batch immediately."""
        resource_logs = payload.get("resourceLogs", [])
        if not isinstance(resource_logs, list):
            return
        for resource_log in resource_logs:
            if not isinstance(resource_log, Mapping):
                continue
            scope_logs = resource_log.get("scopeLogs", [])
            if not isinstance(scope_logs, list):
                continue
            for scope_log in scope_logs:
                if not isinstance(scope_log, Mapping):
                    continue
                records = scope_log.get("logRecords", [])
                if not isinstance(records, list):
                    continue
                for record in records:
                    if isinstance(record, Mapping):
                        self._record_otlp_log(record)

    def _record_otlp_log(self, record: Mapping[str, Any]) -> None:
        attributes = _attributes(record)
        body = _otlp_value(record.get("body"))
        event_name = attributes.get("event.name", body)
        if event_name != "codex.sse_event" or attributes.get(
                "event.kind") != "response.completed":
            return
        usage = {
            field: _as_nonnegative_int(attributes.get(attribute))
            for field, attribute in _OTEL_TOKEN_FIELDS.items()
        }
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        usage["total_tokens"] = (input_tokens +
                                 output_tokens if input_tokens is not None
                                 and output_tokens is not None else None)
        identifying_values = tuple(
            record.get(key) for key in ("timeUnixNano", "observedTimeUnixNano",
                                        "traceId", "spanId"))
        if any(value is not None for value in identifying_values):
            identity = ("otel", ) + identifying_values + tuple(usage.values())
        else:
            # Standard OTLP records carry a timestamp.  Without any native
            # identity it is safer not to collapse two equal real requests.
            identity = self._unique_identity("otel-unidentified")
        self._append({
            "kind": "request_usage",
            "source": "otel",
            **usage
        }, identity)

    def record_event(self, event: Mapping[str, Any]) -> None:
        """Sanitize one parsed Codex JSONL or app-server notification."""
        if not isinstance(event, Mapping):
            return
        with self._lock:
            self._native_event_stream_observed = True

        method = event.get("method")
        event_type = event.get("type")
        params = event.get("params")
        params = params if isinstance(params, Mapping) else {}

        if method == "thread/tokenUsage/updated":
            token_usage = params.get("tokenUsage")
            if isinstance(token_usage, Mapping):
                usage = _usage_from_mapping(token_usage.get("total"),
                                            camel_case=True)
                thread_id = params.get("threadId")
                turn_id = params.get("turnId")
                identity = (
                    "cumulative",
                    "app_server",
                    thread_id,
                    turn_id,
                    tuple(usage.values()),
                )
                self._append(
                    {
                        "kind": "cumulative_usage",
                        "source": "app_server",
                        "thread_id": thread_id,
                        "turn_id": turn_id,
                        **usage,
                    },
                    identity,
                )
            return

        if method == "turn/completed":
            turn = params.get("turn")
            turn = turn if isinstance(turn, Mapping) else {}
            thread_id = params.get("threadId")
            turn_id = turn.get("id")
            status = turn.get("status")
            self._append(
                {
                    "kind": "turn_completion",
                    "source": "app_server",
                    "thread_id": thread_id,
                    "turn_id": turn_id,
                    "completed": status == "completed",
                },
                ("turn_completion", "app_server", thread_id, turn_id, status),
            )
            return

        if event_type in {"turn.completed", "turn_complete"}:
            usage = _usage_from_mapping(event.get("usage"))
            identity = ("cumulative", "exec_jsonl", tuple(usage.values()))
            self._append(
                {
                    "kind": "cumulative_usage",
                    "source": "exec_jsonl",
                    **usage
                },
                identity,
            )
            return

        if event_type == "token_count":
            info = event.get("info")
            info = info if isinstance(info, Mapping) else {}
            usage = _usage_from_mapping(info.get("total_token_usage"))
            identity = ("cumulative", "token_count", tuple(usage.values()))
            self._append(
                {
                    "kind": "cumulative_usage",
                    "source": "token_count",
                    **usage
                },
                identity,
            )
            return

        item = params.get("item")
        item_is_compaction = (method == "item/completed"
                              and isinstance(item, Mapping)
                              and item.get("type") == "contextCompaction")
        if method == "thread/compacted" or event_type == "context_compacted" or item_is_compaction:
            native_identity = (
                params.get("threadId"),
                params.get("turnId"),
                item.get("id") if isinstance(item, Mapping) else None,
            )
            identity = (("compaction", method, event_type,
                         *native_identity) if any(value is not None
                                                  for value in native_identity)
                        else self._unique_identity("compaction-unidentified"))
            compacted = {"kind": "compaction", "source": "native_event"}
            # The deprecated thread notification and the modern item event
            # describe the same compaction. Prefer identified item events;
            # they also distinguish multiple compactions within one turn.
            turn = native_identity[:2]
            with self._lock:
                if all(value is not None for value in turn):
                    if item_is_compaction:
                        self._modern_compaction_turns.add(turn)
                        legacy = self._legacy_compactions.pop(turn, None)
                        if legacy is not None:
                            self.events[:] = [
                                event for event in self.events
                                if event is not legacy
                            ]
                    elif method == "thread/compacted":
                        if turn in self._modern_compaction_turns:
                            return
                        self._legacy_compactions.setdefault(turn, compacted)
                self._append(compacted, identity)

    def summary(self) -> dict[str, Any]:
        """Return JSON-serializable totals, peak coverage, and limitations."""
        with self._lock:
            events = [dict(event) for event in self.events]
            native_stream_observed = self._native_event_stream_observed
            compaction_supported = self._compaction_notifications_supported

        requests = [
            event for event in events if event["kind"] == "request_usage"
        ]
        snapshots = [
            event for event in events if event["kind"] == "cumulative_usage"
        ]
        completions = [
            event for event in events if event["kind"] == "turn_completion"
        ]
        compaction_events = [
            event for event in events if event["kind"] == "compaction"
        ]

        request_totals: dict[str, int | None] = {}
        for field in _TOKEN_FIELDS:
            values = [event.get(field) for event in requests]
            request_totals[field] = (sum(values) if values
                                     and all(value is not None
                                             for value in values) else None)
        # Re-derive the inclusive total so subsets are never double-counted.
        if (request_totals["input_tokens"] is not None
                and request_totals["output_tokens"] is not None):
            request_totals["total_tokens"] = (request_totals["input_tokens"] +
                                              request_totals["output_tokens"])

        # A native exec completion carries its own authoritative usage.  The
        # app-server splits the same evidence across a successful completion
        # notification and a cumulative snapshot, so bind those by both IDs.
        # Never substitute request-level OTLP counts for either transport.
        final_snapshots = [
            event for event in snapshots if event["source"] == "exec_jsonl"
        ]
        app_server_snapshot = None
        if completions:
            app_server_snapshots = [
                event for event in snapshots if event["source"] == "app_server"
            ]
            valid_completion_ids = all(
                isinstance(event.get("thread_id"), str)
                and bool(event.get("thread_id")) and isinstance(
                    event.get("turn_id"), str) and bool(event.get("turn_id"))
                for event in completions)
            valid_snapshot_ids = all(
                isinstance(event.get("thread_id"), str)
                and bool(event.get("thread_id")) and isinstance(
                    event.get("turn_id"), str) and bool(event.get("turn_id"))
                for event in app_server_snapshots)
            observed_pairs = {
                (event.get("thread_id"), event.get("turn_id"))
                for event in [*completions, *app_server_snapshots]
            }
            if (valid_completion_ids and valid_snapshot_ids
                    and len(observed_pairs) == 1 and all(
                        event.get("completed") is True
                        for event in completions)):
                target = next(iter(observed_pairs))
                matching_snapshots = [
                    event for event in app_server_snapshots
                    if (event.get("thread_id"), event.get("turn_id")) == target
                ]
                if matching_snapshots and all(
                        matching_snapshots[-1].get(field) is not None
                        for field in _TOKEN_FIELDS):
                    app_server_snapshot = matching_snapshots[-1]
        snapshot = (final_snapshots[-1]
                    if final_snapshots else app_server_snapshot)
        native_turn_completed = snapshot is not None
        totals = {
            field: snapshot.get(field) if snapshot else None
            for field in _TOKEN_FIELDS
        }
        observed_cumulative_totals = {
            field: snapshots[-1].get(field) if snapshots else None
            for field in _TOKEN_FIELDS
        }

        observed_inputs = [
            event["input_tokens"] for event in requests
            if event.get("input_tokens") is not None
        ]
        peak = max(observed_inputs) if observed_inputs else None
        observed_inputs_complete = bool(requests) and len(
            observed_inputs) == len(requests)

        comparable_fields: list[str] = []
        differences: dict[str, int] = {}
        if snapshot is not None:
            for field in _TOKEN_FIELDS:
                left = snapshot.get(field)
                right = request_totals.get(field)
                if left is not None and right is not None:
                    comparable_fields.append(field)
                    if left != right:
                        differences[field] = left - right
        crosscheck_available = bool(comparable_fields)
        crosscheck_complete = len(comparable_fields) == len(
            _TOKEN_FIELDS) and not differences
        peak_complete = (native_turn_completed and observed_inputs_complete and
                         all(field in comparable_fields
                             for field in ("input_tokens", "output_tokens"))
                         and not differences)
        compactions_complete = native_turn_completed and compaction_supported

        completeness = {
            "totals":
            all(totals[field] is not None for field in _TOKEN_FIELDS),
            "cached_input_tokens": totals["cached_input_tokens"] is not None,
            "reasoning_output_tokens": totals["reasoning_output_tokens"]
            is not None,
            "observed_peak_request_input_tokens": peak_complete,
            "compactions": compactions_complete,
            "cumulative_crosscheck": crosscheck_complete,
        }
        missing_reasons: dict[str, str] = {}
        for field in _TOKEN_FIELDS:
            if totals[field] is None:
                missing_reasons[field] = (
                    f"{field} was absent from the final native completion usage event"
                    if native_turn_completed else
                    "no matching successful native completion/usage pair was observed"
                )
        if not peak_complete:
            missing_reasons["observed_peak_request_input_tokens"] = (
                "the observed OTLP input peak is not verified by a complete, matching native turn"
            )
        if not compaction_supported:
            missing_reasons["compactions"] = (
                "compaction notification coverage is not established for the supplied native transport"
            )
        elif not native_turn_completed:
            missing_reasons["compactions"] = (
                "the native event stream did not reach an exec turn completion; only observed markers are known"
                if native_stream_observed else
                "the native event stream was not supplied to the collector")
        if not crosscheck_available:
            missing_reasons["cumulative_crosscheck"] = (
                "both cumulative and request-level counters were not available"
            )
        elif differences:
            missing_reasons["cumulative_crosscheck"] = (
                "request-level OTLP counters did not match cumulative native counters"
            )
        elif not crosscheck_complete:
            missing_reasons["cumulative_crosscheck"] = (
                "the available counters match, but one or more token fields were not comparable"
            )

        result = {
            "totals": totals,
            "native_turn_completed": native_turn_completed,
            "observed_request_totals": request_totals,
            "observed_cumulative_totals": observed_cumulative_totals,
            "observed_peak_request_input_tokens": peak,
            "peak_observation": {
                "requests_observed":
                len(requests),
                "requests_with_input_tokens":
                len(observed_inputs),
                "all_observed_requests_have_input_tokens":
                observed_inputs_complete,
            },
            "compactions":
            len(compaction_events) if compactions_complete else None,
            "observed_compactions": len(compaction_events),
            "request_usage_events": len(requests),
            "cumulative_snapshots": len(snapshots),
            "completeness": completeness,
            "missing_reasons": missing_reasons,
            "counter_crosscheck": {
                "available": crosscheck_available,
                "matches": not differences if crosscheck_available else None,
                "compared_fields": comparable_fields,
                "differences": differences if crosscheck_available else None,
            },
        }
        # Assert the public contract here as a guard against accidental future
        # additions of non-serializable telemetry types.
        json.dumps(result)
        return result


__all__ = ["TelemetryCollector"]
