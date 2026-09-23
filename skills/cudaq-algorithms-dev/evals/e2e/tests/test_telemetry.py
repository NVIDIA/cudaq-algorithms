# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal native-event contracts for benchmark token instrumentation."""

import importlib.util
import json
from pathlib import Path
import socket
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _attribute(key, value):
    if isinstance(value, int):
        encoded = {"intValue": str(value)}
    else:
        encoded = {"stringValue": value}
    return {"key": key, "value": encoded}


def _otlp_batch(*,
                timestamp,
                input_tokens,
                cached_tokens=None,
                output_tokens=None,
                reasoning_tokens=None,
                extra_records=()):
    attributes = [
        _attribute("event.name", "codex.sse_event"),
        _attribute("event.kind", "response.completed"),
        _attribute("input_token_count", input_tokens),
    ]
    for key, value in (
        ("cached_token_count", cached_tokens),
        ("output_token_count", output_tokens),
        ("reasoning_token_count", reasoning_tokens),
    ):
        if value is not None:
            attributes.append(_attribute(key, value))
    usage_record = {
        "timeUnixNano": str(timestamp),
        "body": {
            "stringValue": "codex.sse_event"
        },
        "attributes": attributes,
    }
    return {
        "resourceLogs": [{
            "resource": {
                "attributes": [
                    _attribute("service.name", "codex_cli_rs"),
                ]
            },
            "scopeLogs": [{
                "scope": {
                    "name": "codex_otel"
                },
                "logRecords": [usage_record, *extra_records],
            }],
        }],
    }


def _app_server_usage(*, thread_id="thread-1", turn_id="turn-1", total=None):
    if total is None:
        total = {
            "totalTokens": 120,
            "inputTokens": 100,
            "cachedInputTokens": 40,
            "outputTokens": 20,
            "reasoningOutputTokens": 7,
        }
    return {
        "method": "thread/tokenUsage/updated",
        "params": {
            "threadId": thread_id,
            "turnId": turn_id,
            "tokenUsage": {
                "total": total
            },
        },
    }


def _app_server_completion(*,
                           thread_id="thread-1",
                           turn_id="turn-1",
                           status="completed"):
    return {
        "method": "turn/completed",
        "params": {
            "threadId": thread_id,
            "turn": {
                "id": turn_id,
                "status": status
            },
        },
    }


class TelemetryContracts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spec = importlib.util.find_spec("telemetry")

    def collector(self):
        self.assertIsNotNone(self.spec, "telemetry implementation is missing")
        from telemetry import TelemetryCollector
        return TelemetryCollector()

    def test_cumulative_snapshots_replace_instead_of_sum(self):
        collector = self.collector()
        collector.record_event({
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "tokenUsage": {
                    "total": {
                        "totalTokens": 120,
                        "inputTokens": 100,
                        "cachedInputTokens": 40,
                        "outputTokens": 20,
                        "reasoningOutputTokens": 7,
                    },
                    "last": {
                        "totalTokens": 120,
                        "inputTokens": 100,
                        "cachedInputTokens": 40,
                        "outputTokens": 20,
                        "reasoningOutputTokens": 7,
                    },
                    "modelContextWindow": 200000,
                },
            },
        })
        collector.record_event({
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-2",
                "tokenUsage": {
                    "total": {
                        "totalTokens": 185,
                        "inputTokens": 150,
                        "cachedInputTokens": 60,
                        "outputTokens": 35,
                        "reasoningOutputTokens": 12,
                    },
                    "last": {
                        "totalTokens": 65,
                        "inputTokens": 50,
                        "cachedInputTokens": 20,
                        "outputTokens": 15,
                        "reasoningOutputTokens": 5,
                    },
                    "modelContextWindow": 200000,
                },
            },
        })

        summary = collector.summary()
        self.assertIn("observed_cumulative_totals", summary)
        self.assertEqual(
            summary["observed_cumulative_totals"], {
                "input_tokens": 150,
                "cached_input_tokens": 60,
                "output_tokens": 35,
                "reasoning_output_tokens": 12,
                "total_tokens": 185,
            })
        self.assertTrue(
            all(value is None for value in summary["totals"].values()))
        self.assertEqual(summary["cumulative_snapshots"], 2)

    def test_completed_app_server_turn_uses_only_its_matching_snapshot(self):
        collector = self.collector()
        collector.record_otlp(
            _otlp_batch(
                timestamp=125,
                input_tokens=100,
                cached_tokens=40,
                output_tokens=20,
                reasoning_tokens=7,
            ))
        collector.record_event(_app_server_usage())
        collector.record_event(_app_server_completion())

        summary = collector.summary()
        self.assertTrue(summary["native_turn_completed"])
        self.assertEqual(
            summary["totals"], {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "output_tokens": 20,
                "reasoning_output_tokens": 7,
                "total_tokens": 120,
            })
        self.assertTrue(summary["completeness"]["totals"])
        self.assertTrue(summary["completeness"]["cumulative_crosscheck"])
        self.assertTrue(
            summary["completeness"]["observed_peak_request_input_tokens"])

    def test_app_server_completion_requires_success_matching_ids_and_usage(
            self):
        cases = {
            "failed":
            (_app_server_usage(), _app_server_completion(status="failed")),
            "interrupted": (
                _app_server_usage(),
                _app_server_completion(status="interrupted"),
            ),
            "mismatched_turn": (
                _app_server_usage(),
                _app_server_completion(turn_id="turn-2"),
            ),
            "usage_missing_turn_id": (
                _app_server_usage(turn_id=None),
                _app_server_completion(),
            ),
            "completion_missing_thread_id": (
                _app_server_usage(),
                _app_server_completion(thread_id=None),
            ),
            "completion_missing_turn_id": (
                _app_server_usage(),
                _app_server_completion(turn_id=None),
            ),
            "missing_usage": (
                _app_server_usage(total={}),
                _app_server_completion(),
            ),
        }
        for name, events in cases.items():
            with self.subTest(name=name):
                collector = self.collector()
                for event in events:
                    collector.record_event(event)
                summary = collector.summary()
                self.assertFalse(summary["native_turn_completed"])
                self.assertTrue(
                    all(value is None for value in summary["totals"].values()))
                self.assertFalse(summary["completeness"]["totals"])

    def test_latest_matching_app_server_snapshot_must_be_complete(self):
        collector = self.collector()
        collector.record_event(_app_server_usage())
        collector.record_event(_app_server_usage(total={"inputTokens": 101}))
        collector.record_event(_app_server_completion())

        summary = collector.summary()
        self.assertFalse(summary["native_turn_completed"])
        self.assertTrue(
            all(value is None for value in summary["totals"].values()))

    def test_later_failed_app_server_turn_does_not_resurrect_prior_success(
            self):
        collector = self.collector()
        collector.record_event(_app_server_usage())
        collector.record_event(_app_server_completion())
        collector.record_event(
            _app_server_usage(
                turn_id="turn-2",
                total={
                    "totalTokens": 55,
                    "inputTokens": 50,
                    "cachedInputTokens": 10,
                    "outputTokens": 5,
                    "reasoningOutputTokens": 2,
                },
            ))
        collector.record_event(
            _app_server_completion(
                turn_id="turn-2",
                status="failed",
            ))

        summary = collector.summary()
        self.assertFalse(summary["native_turn_completed"])
        self.assertTrue(
            all(value is None for value in summary["totals"].values()))

    def test_later_failure_for_same_app_server_turn_overrides_success(self):
        collector = self.collector()
        collector.record_event(_app_server_usage())
        collector.record_event(_app_server_completion())
        collector.record_event(_app_server_completion(status="failed"))

        summary = collector.summary()
        self.assertFalse(summary["native_turn_completed"])
        self.assertTrue(
            all(value is None for value in summary["totals"].values()))

    def test_later_success_does_not_override_app_server_terminal_failure(self):
        for status in ("failed", "interrupted"):
            with self.subTest(status=status):
                collector = self.collector()
                collector.record_event(_app_server_usage())
                collector.record_event(_app_server_completion(status=status))
                collector.record_event(_app_server_completion())

                summary = collector.summary()
                self.assertFalse(summary["native_turn_completed"])
                self.assertTrue(
                    all(value is None for value in summary["totals"].values()))

    def test_multiple_completed_app_server_turns_fail_closed(self):
        collector = self.collector()
        collector.record_event(_app_server_usage())
        collector.record_event(
            _app_server_usage(
                thread_id="thread-2",
                turn_id="turn-2",
            ))
        collector.record_event(
            _app_server_completion(
                thread_id="thread-2",
                turn_id="turn-2",
            ))
        collector.record_event(_app_server_completion())

        summary = collector.summary()
        self.assertFalse(summary["native_turn_completed"])
        self.assertTrue(
            all(value is None for value in summary["totals"].values()))

    def test_incomplete_second_app_server_turn_invalidates_prior_completion(
            self):
        collector = self.collector()
        collector.record_event(_app_server_usage())
        collector.record_event(_app_server_completion())
        collector.record_event(
            _app_server_usage(
                thread_id="thread-2",
                turn_id="turn-2",
            ))

        summary = collector.summary()
        self.assertFalse(summary["native_turn_completed"])
        self.assertTrue(
            all(value is None for value in summary["totals"].values()))

    def test_cache_and_reasoning_subsets_are_not_double_counted(self):
        collector = self.collector()
        collector.record_otlp(
            _otlp_batch(
                timestamp=101,
                input_tokens=100,
                cached_tokens=40,
                output_tokens=20,
                reasoning_tokens=7,
            ))

        summary = collector.summary()
        self.assertIn("observed_request_totals", summary)
        self.assertEqual(
            summary["observed_request_totals"], {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "output_tokens": 20,
                "reasoning_output_tokens": 7,
                "total_tokens": 120,
            })
        self.assertEqual(summary["observed_peak_request_input_tokens"], 100)

    def test_missing_usage_fields_are_null_with_reasons(self):
        collector = self.collector()
        collector.record_otlp(
            _otlp_batch(
                timestamp=102,
                input_tokens=91,
                output_tokens=None,
            ))

        summary = collector.summary()
        self.assertIsNone(summary["totals"]["cached_input_tokens"])
        self.assertIsNone(summary["totals"]["output_tokens"])
        self.assertIsNone(summary["totals"]["reasoning_output_tokens"])
        self.assertIsNone(summary["totals"]["total_tokens"])
        self.assertFalse(summary["completeness"]["totals"])
        self.assertIn("cached_input_tokens", summary["missing_reasons"])
        self.assertIn("output_tokens", summary["missing_reasons"])
        self.assertIn("reasoning_output_tokens", summary["missing_reasons"])
        self.assertIn("total_tokens", summary["missing_reasons"])

    def test_duplicate_batches_and_compactions_are_deduplicated(self):
        collector = self.collector()
        batch = _otlp_batch(
            timestamp=103,
            input_tokens=70,
            cached_tokens=10,
            output_tokens=9,
            reasoning_tokens=4,
        )
        collector.record_otlp(batch)
        collector.record_otlp(batch)
        compacted = {
            "method": "thread/compacted",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-3"
            },
        }
        collector.record_event(compacted)
        collector.record_event(compacted)

        summary = collector.summary()
        self.assertEqual(summary["request_usage_events"], 1)
        self.assertIn("observed_request_totals", summary)
        self.assertEqual(summary["observed_request_totals"]["input_tokens"],
                         70)
        self.assertEqual(summary["observed_compactions"], 1)

    def test_distinct_unidentified_exec_compactions_are_counted(self):
        collector = self.collector()
        collector.record_event({"type": "context_compacted"})
        collector.record_event({"type": "context_compacted"})

        summary = collector.summary()
        self.assertIn("observed_compactions", summary)
        self.assertEqual(summary["observed_compactions"], 2)

    def test_incomplete_stream_does_not_claim_final_totals_or_coverage(self):
        collector = self.collector()
        collector.record_event({
            "type": "thread.started",
            "thread_id": "thread-1"
        })
        collector.record_otlp(
            _otlp_batch(
                timestamp=120,
                input_tokens=100,
                cached_tokens=30,
                output_tokens=20,
                reasoning_tokens=5,
            ))

        summary = collector.summary()
        self.assertTrue(
            all(value is None for value in summary["totals"].values()))
        self.assertFalse(summary["native_turn_completed"])
        self.assertEqual(summary["observed_request_totals"]["total_tokens"],
                         120)
        self.assertEqual(summary["observed_peak_request_input_tokens"], 100)
        self.assertFalse(summary["completeness"]["totals"])
        self.assertFalse(
            summary["completeness"]["observed_peak_request_input_tokens"])
        self.assertFalse(summary["completeness"]["compactions"])
        self.assertIsNone(summary["compactions"])

    def test_missing_final_field_is_not_filled_from_mismatched_requests(self):
        collector = self.collector()
        collector.record_otlp(
            _otlp_batch(
                timestamp=121,
                input_tokens=120,
                cached_tokens=100,
                output_tokens=20,
                reasoning_tokens=15,
            ))
        collector.record_event({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 20,
                "cached_input_tokens": 0,
                "output_tokens": 5
            },
        })

        summary = collector.summary()
        self.assertEqual(summary["totals"]["total_tokens"], 25)
        self.assertIsNone(summary["totals"]["reasoning_output_tokens"])
        self.assertEqual(
            summary["observed_request_totals"]["reasoning_output_tokens"], 15)
        self.assertFalse(summary["completeness"]["totals"])
        self.assertFalse(summary["completeness"]["reasoning_output_tokens"])
        self.assertIn("reasoning_output_tokens", summary["missing_reasons"])

    def test_missing_request_does_not_claim_peak_coverage(self):
        collector = self.collector()
        collector.record_otlp(
            _otlp_batch(
                timestamp=122,
                input_tokens=100,
                cached_tokens=30,
                output_tokens=20,
                reasoning_tokens=5,
            ))
        collector.record_event({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 300,
                "cached_input_tokens": 60,
                "output_tokens": 50,
                "reasoning_output_tokens": 10
            },
        })

        summary = collector.summary()
        self.assertEqual(summary["observed_peak_request_input_tokens"], 100)
        self.assertFalse(
            summary["completeness"]["observed_peak_request_input_tokens"])
        self.assertTrue(summary["peak_observation"]
                        ["all_observed_requests_have_input_tokens"])
        self.assertEqual(
            summary["peak_observation"]["requests_with_input_tokens"], 1)

    def test_matching_partial_fields_do_not_claim_full_counter_crosscheck(
            self):
        collector = self.collector()
        collector.record_otlp(
            _otlp_batch(
                timestamp=123,
                input_tokens=100,
                output_tokens=20,
            ))
        collector.record_event({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20
            },
        })

        summary = collector.summary()
        self.assertTrue(summary["counter_crosscheck"]["matches"])
        self.assertFalse(summary["completeness"]["cumulative_crosscheck"])
        self.assertIn("cumulative_crosscheck", summary["missing_reasons"])

    def test_final_jsonl_is_authoritative_over_larger_intermediate_snapshot(
            self):
        collector = self.collector()
        collector.record_event({
            "type": "token_count",
            "info": {
                "total_token_usage": {
                    "input_tokens": 200,
                    "output_tokens": 50
                },
            }
        })
        collector.record_event({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
                "reasoning_output_tokens": 5,
            }
        })

        self.assertEqual(collector.summary()["totals"]["total_tokens"], 110)

    def test_completed_exec_stream_does_not_prove_compaction_coverage(self):
        collector = self.collector()
        collector.record_event({
            "type": "thread.started",
            "thread_id": "thread-1"
        })
        collector.record_event({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
                "reasoning_output_tokens": 5,
            }
        })

        summary = collector.summary()
        self.assertIsNone(summary["compactions"])
        self.assertFalse(summary["completeness"]["compactions"])
        self.assertEqual(summary["observed_compactions"], 0)
        self.assertIn("compactions", summary["missing_reasons"])

    def test_verified_compaction_transport_can_report_completed_count(self):
        from telemetry import TelemetryCollector
        collector = TelemetryCollector(compaction_notifications_supported=True)
        collector.record_event({"type": "context_compacted"})
        collector.record_event({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 10,
                "reasoning_output_tokens": 5,
            }
        })

        summary = collector.summary()
        self.assertEqual(summary["compactions"], 1)
        self.assertTrue(summary["completeness"]["compactions"])

    def test_compaction_aliases_do_not_double_count_and_distinct_items_survive(
            self):
        for modern_first in (False, True):
            with self.subTest(modern_first=modern_first):
                collector = self.collector()
                params = {"threadId": "thread-1", "turnId": "turn-1"}
                legacy = {"method": "thread/compacted", "params": params}
                modern = {
                    "method": "item/completed",
                    "params": {
                        **params,
                        "item": {
                            "id": "item-1",
                            "type": "contextCompaction"
                        },
                    }
                }
                for event in ((modern, legacy) if modern_first else
                              (legacy, modern)):
                    collector.record_event(event)
                collector.record_event(modern)
                collector.record_event({
                    "method": "item/completed",
                    "params": {
                        **params,
                        "item": {
                            "id": "item-2",
                            "type": "contextCompaction"
                        },
                    }
                })

                summary = collector.summary()
                self.assertIn("observed_compactions", summary)
                self.assertEqual(summary["observed_compactions"], 2)

    def test_same_timestamp_with_different_usage_is_not_a_duplicate(self):
        collector = self.collector()
        collector.record_otlp(
            _otlp_batch(
                timestamp=110,
                input_tokens=20,
                cached_tokens=0,
                output_tokens=2,
                reasoning_tokens=0,
            ))
        collector.record_otlp(
            _otlp_batch(
                timestamp=110,
                input_tokens=30,
                cached_tokens=10,
                output_tokens=3,
                reasoning_tokens=1,
            ))

        self.assertEqual(collector.summary()["request_usage_events"], 2)

    def test_counter_mismatch_is_incomplete_and_keeps_cumulative_totals(self):
        collector = self.collector()
        collector.record_otlp(
            _otlp_batch(
                timestamp=108,
                input_tokens=20,
                cached_tokens=0,
                output_tokens=0,
                reasoning_tokens=0,
            ))
        collector.record_otlp(
            _otlp_batch(
                timestamp=109,
                input_tokens=40,
                cached_tokens=10,
                output_tokens=5,
                reasoning_tokens=2,
            ))
        collector.record_event({
            "type": "turn.completed",
            "usage": {
                "input_tokens": 40,
                "cached_input_tokens": 10,
                "output_tokens": 5,
                "reasoning_output_tokens": 2,
            },
        })

        summary = collector.summary()
        self.assertEqual(summary["totals"]["total_tokens"], 45)
        self.assertFalse(summary["counter_crosscheck"]["matches"])
        self.assertEqual(summary["counter_crosscheck"]["differences"], {
            "input_tokens": -20,
            "total_tokens": -20,
        })
        self.assertFalse(summary["completeness"]["cumulative_crosscheck"])
        self.assertIn("cumulative_crosscheck", summary["missing_reasons"])

    def test_only_usage_metadata_survives_sanitization(self):
        prompt_record = {
            "timeUnixNano":
            "104",
            "body": {
                "stringValue": "codex.user_prompt"
            },
            "attributes": [
                _attribute("event.name", "codex.user_prompt"),
                _attribute("prompt", "SECRET PROMPT"),
            ],
        }
        tool_record = {
            "timeUnixNano":
            "105",
            "body": {
                "stringValue": "codex.tool_result"
            },
            "attributes": [
                _attribute("event.name", "codex.tool_result"),
                _attribute("output", "SECRET TOOL OUTPUT"),
                _attribute("environment", "SECRET=VALUE"),
            ],
        }
        collector = self.collector()
        collector.record_otlp(
            _otlp_batch(
                timestamp=106,
                input_tokens=30,
                cached_tokens=0,
                output_tokens=5,
                reasoning_tokens=2,
                extra_records=(prompt_record, tool_record),
            ))

        serialized = json.dumps(collector.events)
        self.assertEqual(len(collector.events), 1)
        self.assertNotIn("SECRET", serialized)
        self.assertNotIn("prompt", serialized.lower())
        self.assertNotIn("tool", serialized.lower())

    def test_context_manager_is_loopback_only_and_closes_endpoint(self):
        collector = self.collector()
        with self.assertRaises(RuntimeError):
            _ = collector.config_args

        payload = json.dumps(
            _otlp_batch(
                timestamp=107,
                input_tokens=12,
                cached_tokens=0,
                output_tokens=3,
                reasoning_tokens=1,
            )).encode()
        with collector:
            args = collector.config_args
            joined = " ".join(args)
            self.assertIn("127.0.0.1", joined)
            self.assertIn("/v1/logs", joined)
            self.assertIn("otel.log_user_prompt=false", joined)
            endpoint = collector.endpoint
            with self.assertRaises(HTTPError) as rejected:
                urlopen(Request(endpoint.replace("/v1/logs", "/other"),
                                data=payload,
                                headers={"Content-Type": "application/json"}),
                        timeout=1)
            self.assertEqual(rejected.exception.code, 404)
            with urlopen(Request(endpoint,
                                 data=payload,
                                 headers={"Content-Type": "application/json"}),
                         timeout=1) as response:
                self.assertEqual(response.status, 200)

        self.assertEqual(collector.summary()["request_usage_events"], 1)
        with self.assertRaises(URLError):
            urlopen(Request(endpoint,
                            data=payload,
                            headers={"Content-Type": "application/json"}),
                    timeout=.2)

    def test_exit_closes_accepted_incomplete_upload_before_snapshot(self):
        collector = self.collector()
        accepted = threading.Event()
        payload = json.dumps(
            _otlp_batch(
                timestamp=124,
                input_tokens=12,
                cached_tokens=0,
                output_tokens=3,
                reasoning_tokens=1,
            )).encode()
        client = None
        try:
            with collector:
                server = collector._server
                original = server.finish_request

                def acknowledge_accept(request, address):
                    accepted.set()
                    return original(request, address)

                with patch.object(server,
                                  "finish_request",
                                  side_effect=acknowledge_accept):
                    client = socket.create_connection(server.server_address,
                                                      timeout=2)
                    headers = (
                        "POST /v1/logs HTTP/1.0\r\nContent-Type: application/json\r\n"
                        f"Content-Length: {len(payload)}\r\n\r\n").encode()
                    client.sendall(headers + payload[:-1])
                    self.assertTrue(accepted.wait(timeout=2))
            try:
                client.sendall(payload[-1:])
                response = client.recv(4096)
            except (BrokenPipeError, ConnectionResetError):
                response = b""
            self.assertNotIn(b"200 OK", response)
            self.assertEqual(collector.summary()["request_usage_events"], 0)
        finally:
            if client is not None:
                client.close()


if __name__ == "__main__":
    unittest.main()
