"""Offline HTTP contract and preflight checks; no endpoint access required."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class Response(io.BytesIO):

    def read1(self, size=-1):
        return self.read(size)


class Clock:

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ProviderTests(unittest.TestCase):

    def test_nemotron_catalog_sends_documented_agent_template_flags(self):
        path = Path(__file__).resolve().parents[2] / 'models.json'
        model = next(m for m in json.loads(path.read_text())['models']
                     if m['id'] == 'nemotron-ultra')
        model['api_key_env'] = 'TEST_PROVIDER_KEY'
        seen = []
        with patch.object(self.p, '_open', self.transport(self.reply(), seen)):
            self.p.complete(model, [{
                'role': 'user',
                'content': 'hello'
            }], [], 3)
        body = json.loads(seen[0][0].data)
        self.assertEqual(body['chat_template_kwargs'], {
            'enable_thinking': True,
            'force_nonempty_content': True
        })

    def test_retry_after_http_date_and_long_delay_are_not_shortened(self):
        from email.utils import formatdate
        with patch.object(self.p.time, 'time', return_value=1000.):
            self.assertEqual(
                self.p._retry_after(formatdate(1120, usegmt=True)), 120.)
            self.assertEqual(self.p._retry_after('120'), 120.)
            self.assertEqual(self.p._retry_after(formatdate(900, usegmt=True)),
                             0.)

    def test_preflight_publishes_completed_phase_before_next_request(self):
        updates = []
        replies = [
            self.reply(),
            self.reply(None,
                       tool_calls=[{
                           "id": "call",
                           "type": "function",
                           "function": {
                               "name": "echo",
                               "arguments": '{"value":"preflight"}'
                           }
                       }])
        ]

        def complete(*args):
            if len(replies) == 1:
                self.assertTrue(updates[-1]["completion"]["ok"])
                self.assertEqual(updates[-1]["active_phase"], "tool_call")
            return replies.pop(0)

        with patch.object(self.p, "complete", complete):
            result = self.p.preflight(self.model,
                                      5,
                                      progress=lambda value: updates.append(
                                          json.loads(json.dumps(value))))
        self.assertTrue(result["ok"])
        self.assertEqual(updates[-1]["status"], "complete")
        self.assertNotIn("private-test-key", json.dumps(updates))

    def test_preflight_publishes_partial_result_when_interrupted(self):
        updates = []
        with patch.object(self.p,
                          "complete",
                          side_effect=[self.reply(),
                                       KeyboardInterrupt()]):
            with self.assertRaises(KeyboardInterrupt):
                self.p.preflight(self.model,
                                 5,
                                 progress=lambda value: updates.append(
                                     json.loads(json.dumps(value))))
        self.assertTrue(updates[-1]["completion"]["ok"])
        self.assertEqual(updates[-1]["status"], "interrupted")
        self.assertEqual(updates[-1]["tool_call"]["error"]["kind"],
                         "interrupted")
        self.assertFalse(updates[-1]["ok"])

    def setUp(self):
        from runner import providers
        self.p = providers
        self.model = {
            "id": "example",
            "model": "publisher/model",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "TEST_PROVIDER_KEY",
            "max_tokens": 100,
            "extra_body": {
                "temperature": 0.2
            }
        }
        self.env = patch.dict(os.environ,
                              {"TEST_PROVIDER_KEY": "private-test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def reply(self, content="OK", **kwargs):
        return {
            "model":
            "actual-model",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": content,
                    **kwargs
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "total_tokens": 12
            }
        }

    def transport(self, reply, seen):

        def open_request(request, timeout):
            seen.append((request, timeout))
            return Response(json.dumps(reply).encode())

        return open_request

    def test_exact_request_and_raw_usage_are_preserved(self):
        seen = []
        reply = self.reply()
        with patch.object(self.p, "_open", self.transport(reply, seen)):
            result = self.p.complete(self.model, [{
                "role": "user",
                "content": "hello"
            }], None, 3)
        request, timeout = seen[0]
        body = json.loads(request.data)
        self.assertEqual(request.full_url,
                         "https://example.invalid/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"),
                         "Bearer private-test-key")
        self.assertEqual(body["model"], "publisher/model")
        self.assertEqual(body["max_tokens"], 100)
        self.assertFalse(body["stream"])
        self.assertNotIn("tools", body)
        self.assertEqual(result, reply)
        self.assertLessEqual(timeout, 3)

    def test_extra_body_cannot_silently_replace_model(self):
        self.model["extra_body"] = {"model": "another-model"}
        with self.assertRaises(self.p.ProviderError) as raised:
            self.p.complete(self.model, [], None, 1)
        self.assertEqual(raised.exception.kind, "configuration")

    def test_environment_key_precedes_explicit_file(self):
        self.model["api_key_file"] = "/does-not-exist"
        with patch.object(self.p, "_open", self.transport(self.reply(), [])):
            self.p.complete(self.model, [], None, 1)

    def test_explicit_file_fallback_and_missing_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key"
            path.write_text("file-key\n")
            self.model["api_key_file"] = str(path)
            with patch.dict(os.environ, {"TEST_PROVIDER_KEY": ""}):
                seen = []
                with patch.object(self.p, "_open",
                                  self.transport(self.reply(), seen)):
                    self.p.complete(self.model, [], None, 1)
                self.assertEqual(seen[0][0].get_header("Authorization"),
                                 "Bearer file-key")
                path.unlink()
                with self.assertRaises(self.p.ProviderError) as raised:
                    self.p.complete(self.model, [], None, 1)
                self.assertEqual(raised.exception.kind,
                                 "authentication_config")

    def test_http_errors_are_structured_and_do_not_echo_body_or_key(self):
        for status, retryable in [(401, False), (429, True), (503, True),
                                  (400, False)]:
            error = HTTPError("https://example.invalid", status,
                              "private-test-key", {},
                              io.BytesIO(b"private-test-key"))
            with patch.object(self.p, "_open", side_effect=error):
                with self.assertRaises(self.p.ProviderError) as raised:
                    self.p.complete(self.model, [], None, 1)
            self.assertEqual(raised.exception.status, status)
            self.assertEqual(raised.exception.retryable, retryable)
            self.assertNotIn("private-test-key", str(raised.exception))
            self.assertNotIn("private-test-key",
                             json.dumps(raised.exception.as_dict()))

    def http_failure(self, status, body, headers=None, timeout=1):
        stream = body if hasattr(body, 'read') else io.BytesIO(body)
        error = HTTPError('https://example.invalid', status,
                          'private-test-key', headers or {}, stream)
        with patch.object(self.p, '_open', side_effect=error):
            with self.assertRaises(self.p.ProviderError) as raised:
                self.p.complete(self.model, [{'role': 'user',
                                             'content': 'private prompt'}],
                                None, timeout)
        self.assertTrue(stream.closed)
        return raised.exception.as_dict()

    def test_http_error_records_safe_request_id_and_overload_category(self):
        error = self.http_failure(503, json.dumps({
            'error': {'message': 'Service temporarily overloaded',
                      'type': 'Service Unavailable', 'code': 503},
            'request_id': 'untrusted-body-id', 'status': 200,
            'kind': 'success', 'retryable': False,
            'prompt': 'private prompt', 'secret': 'private-test-key'
        }).encode(), {'Nvcf-Reqid': '62ff7c74-1532-4277-b43a-2a519e31c23b',
                      'Retry-After': '3.5', 'Authorization': 'private-test-key'})
        self.assertEqual(error, {
            'kind': 'http', 'status': 503,
            'message': 'Provider returned HTTP 503.', 'retryable': True,
            'retry_after_seconds': 3.5,
            'request_id': '62ff7c74-1532-4277-b43a-2a519e31c23b',
            'provider_reason': 'overloaded'
        })

    def test_http_error_classifies_only_exact_known_error_fields(self):
        cases = [
            (500, {'error': {'message': 'Internal server error'}}, 'internal_error'),
            (429, {'status': 429, 'title': 'Too Many Requests'}, 'rate_limited'),
            (400, {'error': {'code': 'rate_limit_exceeded'}}, 'rate_limited'),
            (400, {'error': {'type': 'overloaded_error'}}, 'overloaded'),
            (400, {'error': {'message': 'Service temporarily overloaded private prompt'}}, 'unknown'),
            (400, {'prompt': 'Service temporarily overloaded'}, 'unknown'),
            (400, {'error': {'message': {'secret': 'private-test-key'}}}, 'unknown'),
            (503, {}, 'internal_error'),
        ]
        for status, body, expected in cases:
            with self.subTest(status=status, body=body):
                error = self.http_failure(status, json.dumps(body).encode())
                self.assertEqual(error.get('provider_reason'), expected)
                self.assertNotIn('private prompt', json.dumps(error))
                self.assertNotIn('private-test-key', json.dumps(error))

    def test_http_error_accepts_only_bounded_allowlisted_request_headers(self):
        for headers, expected in [
            ({'x-request-id': 'req_123:abc.def-456'}, 'req_123:abc.def-456'),
            ({'nvcf-reqid': 'provider-123'}, 'provider-123'),
            ({'X-Request-ID': 'request-456'}, 'request-456'),
            ({'Request-Id': 'not-allowlisted'}, None),
            ({'Nvcf-Reqid': 'bad\r\nprivate-test-key'}, None),
            ({'Nvcf-Reqid': 'private-test-key'}, None),
            ({'Nvcf-Reqid': 'prefix-private-test-key-suffix'}, None),
            ({'Nvcf-Reqid': 'a' * 129}, None),
            ({'Nvcf-Reqid': 'two, ids'}, None),
            ({'Nvcf-Reqid': 'é'}, None),
            ({'Nvcf-Reqid': ['not', 'a', 'string']}, None),
            ({'Nvcf-Reqid': 'bad id', 'x-request-id': 'fallback'}, 'fallback'),
        ]:
            with self.subTest(headers=headers):
                error = self.http_failure(503, b'{}', headers)
                self.assertEqual(error.get('request_id'), expected)
                self.assertEqual(error.get('provider_reason'), 'internal_error')

    def test_http_error_invalid_or_oversized_body_preserves_http_failure(self):
        class TrackedBody(io.BytesIO):
            consumed = 0

            def read1(self, size=-1):
                chunk = super().read1(size)
                self.consumed += len(chunk)
                return chunk

        for raw in (b'not-json private-test-key', b'\xff', b'[]',
                    b'[' * 2000 + b']' * 2000,
                    b'{"error":{"message":"Service temporarily overloaded"},'
                    b'"padding":"' + b'x' * 20000 + b'"}'):
            with self.subTest(size=len(raw)):
                body = TrackedBody(raw)
                error = self.http_failure(503, body, {'Retry-After': '2'})
                self.assertEqual(error['status'], 503)
                self.assertEqual(error.get('provider_reason'), 'internal_error')
                self.assertEqual(error['retry_after_seconds'], 2.)
                self.assertLessEqual(body.consumed, 16385)
                self.assertNotIn('private-test-key', json.dumps(error))

    def test_http_error_body_timeout_preserves_status_and_retry_after(self):
        class TimeoutBody(io.BytesIO):

            def read1(self, size=-1):
                raise TimeoutError('private-test-key private prompt')

        error = self.http_failure(503, TimeoutBody(), {
            'Nvcf-Reqid': 'request-123', 'Retry-After': '2'
        })
        self.assertEqual(error['kind'], 'http')
        self.assertEqual(error['status'], 503)
        self.assertEqual(error.get('request_id'), 'request-123')
        self.assertEqual(error.get('provider_reason'), 'internal_error')
        self.assertEqual(error['retry_after_seconds'], 2.)

    def test_http_error_body_reads_use_short_shared_deadline(self):
        clock = Clock()
        timeouts = []

        class SlowBody(io.BytesIO):

            def read1(self, size=-1):
                clock.now += 0.1
                return super().read1(min(size, 1))

        body = SlowBody(b'{"error":{"message":"Service temporarily overloaded"}}')
        body.fp = SimpleNamespace(raw=SimpleNamespace(
            _sock=SimpleNamespace(settimeout=timeouts.append)))
        with patch.object(self.p.time, 'monotonic', clock.monotonic):
            error = self.http_failure(503, body, timeout=10)
        self.assertEqual(error['status'], 503)
        self.assertEqual(error.get('provider_reason'), 'internal_error')
        self.assertGreater(len(timeouts), 0)
        self.assertLessEqual(timeouts[0], 0.25)
        self.assertLessEqual(clock.now, 0.3 + 1e-9)
        self.assertTrue(all(a > b for a, b in zip(timeouts, timeouts[1:])))

    def test_http_error_body_without_timeout_support_is_not_read(self):
        class UnboundedBody:
            closed = False

            def read(self, size=-1):
                raise AssertionError('A potentially blocking body was read')

            def close(self):
                self.closed = True

        error = self.http_failure(503, UnboundedBody())
        self.assertEqual(error['status'], 503)
        self.assertEqual(error.get('provider_reason'), 'internal_error')

    def test_http_error_malformed_retry_header_preserves_original_status(self):
        error = self.http_failure(503, b'{}',
                                  {'Retry-After': ['private-test-key']})
        self.assertEqual(error['status'], 503)
        self.assertEqual(error.get('provider_reason'), 'internal_error')
        self.assertNotIn('retry_after_seconds', error)

    def test_http_error_does_not_read_after_original_request_deadline(self):
        clock = Clock()

        class ExpiredBody(io.BytesIO):

            def read1(self, size=-1):
                raise AssertionError('Read after original request deadline')

        body = ExpiredBody(b'{}')

        def expired(request, timeout):
            clock.now = 1.
            raise HTTPError(request.full_url, 503, 'unavailable', {}, body)

        with patch.object(self.p.time, 'monotonic', clock.monotonic), \
                patch.object(self.p, '_open', expired):
            with self.assertRaises(self.p.ProviderError) as raised:
                self.p.complete(self.model, [], None, 1)
        self.assertEqual(raised.exception.status, 503)
        self.assertEqual(raised.exception.as_dict().get('provider_reason'),
                         'internal_error')
        self.assertTrue(body.closed)

    def test_network_error_is_redacted(self):
        with patch.object(self.p,
                          "_open",
                          side_effect=URLError("private-test-key")):
            with self.assertRaises(self.p.ProviderError) as raised:
                self.p.complete(self.model, [], None, 1)
        self.assertTrue(raised.exception.retryable)
        self.assertNotIn("private-test-key", str(raised.exception))

    def test_timeout_is_typed(self):
        with patch.object(self.p,
                          "_open",
                          side_effect=TimeoutError("private-test-key")):
            with self.assertRaises(self.p.ProviderError) as raised:
                self.p.complete(self.model, [], None, 1)
        self.assertEqual(raised.exception.kind, "timeout")

    def test_oversized_and_invalid_responses_are_rejected(self):
        for raw, expected in [(b"x" * 33, "response_too_large"),
                              (b"not-json", "invalid_response"),
                              (b"{}", "invalid_response")]:
            with patch.object(self.p, "MAX_RESPONSE_BYTES",
                              32), patch.object(self.p,
                                                "_open",
                                                return_value=Response(raw)):
                with self.assertRaises(self.p.ProviderError) as raised:
                    self.p.complete(self.model, [], None, 1)
            self.assertEqual(raised.exception.kind, expected)

    def test_preflight_really_requests_text_and_a_tool_call(self):
        replies = [
            self.reply(),
            self.reply(None,
                       tool_calls=[{
                           "id": "call-1",
                           "type": "function",
                           "function": {
                               "name": "echo",
                               "arguments": '{"value":"preflight"}'
                           }
                       }])
        ]
        seen = []

        def open_request(request, timeout):
            seen.append(json.loads(request.data))
            return Response(json.dumps(replies.pop(0)).encode())

        with patch.object(self.p, "_open", open_request):
            result = self.p.preflight(self.model, timeout=2)
        self.assertTrue(result["ok"])
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[1]["tools"][0]["function"]["name"], "echo")
        self.assertNotIn("tool_choice", seen[1])
        self.assertEqual(result["completion"]["returned_model"],
                         "actual-model")
        self.assertTrue(result["tool_call"]["ok"])
        self.assertNotIn("private-test-key", json.dumps(result))

    def test_preflight_detects_plain_text_instead_of_tool_call(self):
        with patch.object(self.p, "_open", self.transport(self.reply(), [])):
            result = self.p.preflight(self.model, 1)
        self.assertFalse(result["ok"])
        self.assertTrue(result["completion"]["ok"])
        self.assertFalse(result["tool_call"]["ok"])

    def test_preflight_invalid_url_does_not_echo_embedded_credentials(self):
        self.model[
            "base_url"] = "https://user:private-test-key@example.invalid/v1?key=private-test-key"
        result = self.p.preflight(self.model, 1)
        self.assertFalse(result["ok"])
        self.assertNotIn("private-test-key", json.dumps(result))

    def test_preflight_records_malformed_tool_calls_as_capability_failure(
            self):
        with patch.object(self.p, "_open",
                          self.transport(self.reply(None, tool_calls=3), [])):
            result = self.p.preflight(self.model, 1)
        self.assertFalse(result["tool_call"]["ok"])

    def test_complete_returns_numeric_retry_after_without_retrying(self):
        for header, expected in [("3.5", 3.5), ("1000000", 1000000.0),
                                 ("-1", None), ("nan", None), ("inf", None),
                                 ("private-test-key", None)]:
            seen = []

            def unavailable(request, timeout):
                seen.append(request)
                raise HTTPError(request.full_url, 429, "private-test-key",
                                {"Retry-After": header},
                                io.BytesIO(b"private-test-key"))

            with patch.object(self.p, "_open", unavailable):
                with self.assertRaises(self.p.ProviderError) as raised:
                    self.p.complete(self.model, [], None, 5)
            self.assertEqual(len(seen), 1)
            self.assertEqual(raised.exception.retry_after_seconds, expected)
            self.assertNotIn("private-test-key",
                             json.dumps(raised.exception.as_dict()))

    def test_preflight_retries_transient_http_with_shared_deadline_and_same_model(
            self):
        clock = Clock()
        seen = []
        tool_reply = self.reply(None,
                                tool_calls=[{
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"value":"preflight"}'
                                    }
                                }])

        def transport(request, timeout):
            body = json.loads(request.data)
            seen.append((body, timeout))
            clock.now += 0.25
            if len(seen) in (1, 2):
                headers = {"Retry-After": "1.5"} if len(seen) == 1 else {}
                raise HTTPError(request.full_url,
                                429 if len(seen) == 1 else 503, "unavailable",
                                headers, io.BytesIO())
            return Response(
                json.dumps(tool_reply if body.get("tools") else self.reply()).
                encode())

        with patch.object(self.p, "_open", transport), patch.object(
                self.p.time, "monotonic",
                clock.monotonic), patch.object(self.p.time, "sleep",
                                               clock.sleep):
            result = self.p.preflight(self.model, 10)
        self.assertTrue(result["ok"])
        self.assertEqual([body["model"] for body, _ in seen],
                         ["publisher/model"] * 4)
        self.assertEqual([timeout for _, timeout in seen], [10, 8.25, 6, 10])
        self.assertEqual(clock.sleeps, [1.5, 2])
        phase = result["completion"]
        self.assertEqual(phase["request_count"], 3)
        self.assertEqual(phase["retry_count"], 2)
        self.assertEqual(phase["backoff_seconds"], 3.5)
        self.assertEqual(phase["attempts"][0]["error"]["status"], 429)
        self.assertEqual(phase["attempts"][1]["error"]["status"], 503)
        self.assertEqual(phase["attempts"][0]["backoff_seconds"], 1.5)
        self.assertEqual(phase["latency_seconds"], 4.25)

    def test_preflight_authentication_failure_is_not_retried(self):
        clock = Clock()
        seen = []

        def transport(request, timeout):
            seen.append(request)
            raise HTTPError(request.full_url, 401, "unauthorized", {},
                            io.BytesIO())

        with patch.object(self.p, "_open", transport), patch.object(
                self.p.time, "monotonic",
                clock.monotonic), patch.object(self.p.time, "sleep",
                                               clock.sleep):
            result = self.p.preflight(self.model, 10)
        self.assertFalse(result["ok"])
        self.assertEqual(len(seen), 2)
        self.assertEqual(clock.sleeps, [])
        self.assertEqual(result["completion"]["retry_count"], 0)
        self.assertEqual(result["completion"]["error"]["status"], 401)

    def test_preflight_does_not_wait_or_retry_past_phase_deadline(self):
        clock = Clock()
        seen = []

        def transport(request, timeout):
            seen.append(timeout)
            clock.now += 0.75
            raise HTTPError(request.full_url, 408, "request timeout", {},
                            io.BytesIO())

        with patch.object(self.p, "_open", transport), patch.object(
                self.p.time, "monotonic",
                clock.monotonic), patch.object(self.p.time, "sleep",
                                               clock.sleep):
            result = self.p.preflight(self.model, 1)
        self.assertFalse(result["ok"])
        self.assertEqual(seen, [1, 1])
        self.assertEqual(clock.sleeps, [])
        self.assertEqual(result["completion"]["request_count"], 1)
        self.assertEqual(result["completion"]["retry_stop_reason"],
                         "phase_deadline")

    def test_preflight_never_exceeds_two_retries_per_phase(self):
        for status in (408, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                clock = Clock()
                seen = []

                def transport(request, timeout):
                    seen.append(timeout)
                    raise HTTPError(request.full_url, status, "unavailable",
                                    {}, io.BytesIO())

                with patch.object(self.p, "_open", transport), patch.object(
                        self.p.time, "monotonic",
                        clock.monotonic), patch.object(self.p.time, "sleep",
                                                       clock.sleep):
                    result = self.p.preflight(self.model, 30)
                self.assertFalse(result["ok"])
                self.assertEqual(len(seen), 6)
                self.assertEqual(clock.sleeps, [2, 2, 2, 2])
                self.assertEqual(result["completion"]["retry_count"], 2)
                self.assertEqual(result["tool_call"]["retry_count"], 2)
                self.assertEqual(result["completion"]["error"]["status"],
                                 status)

    def test_catalog_contains_ten_known_models_without_credentials(self):
        catalog = json.loads(
            (Path(__file__).resolve().parents[2] / "models.json").read_text())
        models = catalog["models"]
        self.assertEqual(len(models), 10)
        keyed = {model["id"]: model for model in models}
        self.assertEqual(keyed["nemotron-ultra"]["model"],
                         "nvidia/nemotron-3-ultra-550b-a55b")
        self.assertEqual(keyed["kimi"]["model"], "moonshotai/kimi-k3")
        self.assertTrue(
            all(x["api_key_env"] == "NVIDIA_API_KEY" for x in models))
        self.assertTrue(all("api_key_file" not in x for x in models))


if __name__ == "__main__":
    unittest.main()
