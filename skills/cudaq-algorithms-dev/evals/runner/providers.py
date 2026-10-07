"""Bounded, synchronous OpenAI-compatible chat transport using only the stdlib.

Credentials are read at invocation time and are never included in evidence or
errors. ``complete`` sends one request; preflight alone retries selected HTTP
failures within each phase's deadline. No model substitutions are made.
"""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import re
import socket
import time
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_ERROR_BYTES = 16 * 1024
ERROR_READ_SECONDS = 0.25
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_PROVIDER_REASONS = frozenset(
    ("overloaded", "rate_limited", "internal_error", "unknown"))
PREFLIGHT_RETRY_STATUSES = (408, 429, 500, 502, 503, 504)
MAX_PREFLIGHT_RETRIES = 2


def _retry_after(value) -> float | None:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            seconds = max(0., date.timestamp() - time.time())
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    # Never retry before the server's requested time. Callers stop rather than
    # shorten this delay when it does not fit their remaining budget.
    return seconds


class ProviderError(Exception):
    """A safe-to-record provider failure without response bodies or credentials."""

    def __init__(self,
                 kind: str,
                 status: int | None,
                 message: str,
                 retryable: bool = False,
                 retry_after_seconds: float | None = None,
                 request_id: str | None = None,
                 provider_reason: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status
        self.message = message
        self.retryable = retryable
        self.retry_after_seconds = _retry_after(retry_after_seconds)
        self.request_id = request_id if isinstance(
            request_id, str) and _REQUEST_ID.fullmatch(request_id) else None
        self.provider_reason = provider_reason if isinstance(
            provider_reason,
            str) and provider_reason in _PROVIDER_REASONS else None

    def as_dict(self) -> dict:
        result = {
            "kind": self.kind,
            "status": self.status,
            "message": self.message,
            "retryable": self.retryable
        }
        if self.retry_after_seconds is not None:
            result["retry_after_seconds"] = self.retry_after_seconds
        if self.request_id is not None:
            result["request_id"] = self.request_id
        if self.provider_reason is not None:
            result["provider_reason"] = self.provider_reason
        return result


def _error_request_id(headers, credential: str) -> str | None:
    # Header names are case insensitive even for plain-dict test transports.
    for name in ("nvcf-reqid", "x-request-id"):
        values = [
            value for key, value in headers.items()
            if isinstance(key, str) and key.lower() == name
        ]
        if len(values) != 1:
            continue
        value = values[0]
        if (isinstance(value, str) and _REQUEST_ID.fullmatch(value)
                and credential not in value):
            return value
    return None


def _error_reason(error: HTTPError, deadline: float) -> str:
    """Classify bounded diagnostic input; never retain free-form provider text."""
    fallback = ("rate_limited" if error.code == 429 else
                "internal_error" if 500 <= error.code < 600 else "unknown")
    response = error.fp
    sock = getattr(getattr(getattr(response, "fp", None), "raw", None),
                   "_sock", None)
    # BytesIO is nonblocking. For network streams require a socket whose read
    # timeout can be reduced; otherwise keep the status-derived category.
    if not isinstance(response, io.BytesIO) and sock is None:
        return fallback
    try:
        raw = _read_response(response,
                             min(deadline,
                                 time.monotonic() + ERROR_READ_SECONDS),
                             max_bytes=MAX_ERROR_BYTES)
        body = json.loads(raw)
    except (OSError, HTTPException, ValueError, TypeError, RecursionError,
            ProviderError):
        return fallback
    if not isinstance(body, dict) or error.code == 429:
        return fallback
    detail = body.get("error")
    fields = [body.get(name) for name in ("title", "detail", "message")]
    if isinstance(detail, dict):
        fields.extend(detail.get(name) for name in ("message", "type", "code"))
    elif isinstance(detail, str):
        fields.append(detail)
    known = {
        "Service temporarily overloaded": "overloaded",
        "overloaded_error": "overloaded",
        "rate_limit_exceeded": "rate_limited",
        "rate_limit_error": "rate_limited",
        "Too Many Requests": "rate_limited",
        "Internal server error": "internal_error",
        "Internal Server Error": "internal_error",
        "internal_server_error": "internal_error",
    }
    for value in fields:
        if isinstance(value, str) and value in known:
            return known[value]
    return fallback


class _NoRedirect(HTTPRedirectHandler):

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward an Authorization header to a redirected destination.
        return None


def _open(request: Request, timeout: float):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def _credential(model: dict) -> str:
    key = os.environ.get(model.get("api_key_env") or "NVIDIA_API_KEY",
                         "").strip()
    if not key and model.get("api_key_file"):
        try:
            with Path(model["api_key_file"]).open(encoding="utf-8") as stream:
                key = stream.read(16385).strip()
        except (OSError, UnicodeError, TypeError):
            raise ProviderError(
                "authentication_config", None,
                "The configured credential file could not be read.") from None
    if not key or len(key) > 16384 or "\n" in key or "\r" in key:
        raise ProviderError(
            "authentication_config", None,
            "A valid API credential is required in the configured environment variable or file."
        )
    return key


def _read_response(response, deadline: float, *, max_bytes=None) -> bytes:
    max_bytes = MAX_RESPONSE_BYTES if max_bytes is None else max_bytes
    chunks = []
    size = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError()
        # urllib wraps an HTTPResponse around a socket. Update its timeout on
        # every read so a slowly trickling body cannot renew the full deadline.
        sock = getattr(getattr(getattr(response, "fp", None), "raw", None),
                       "_sock", None)
        if sock is not None:
            sock.settimeout(remaining)
        reader = getattr(response, "read1", response.read)
        chunk = reader(min(65536, max_bytes + 1 - size))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        size += len(chunk)
        if size > max_bytes:
            raise ProviderError(
                "response_too_large", None,
                "The provider response exceeded the size limit.")


def complete(model: dict, messages: list, tools: list | None,
             timeout: float) -> dict:
    """Send exactly one completion request and preserve its choices and usage."""
    if isinstance(timeout, bool) or not isinstance(
            timeout,
        (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ProviderError("configuration", None,
                            "timeout must be a positive finite number.")
    base = model.get("base_url", "")
    try:
        parsed = urlsplit(base)
    except (TypeError, ValueError):
        raise ProviderError("configuration", None,
                            "Invalid provider base URL.") from None
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query
            or parsed.fragment
            or (parsed.scheme == "http"
                and parsed.hostname not in ("localhost", "127.0.0.1", "::1"))):
        raise ProviderError(
            "configuration", None,
            "Use an HTTPS base URL without credentials, query, or fragment; HTTP is allowed only on loopback."
        )
    if not isinstance(model.get("model"), str) or not model["model"].strip():
        raise ProviderError("configuration", None,
                            "An explicit provider model ID is required.")
    extra = model.get("extra_body") or {}
    if not isinstance(extra, dict) or set(extra) & {
            "model", "messages", "tools", "stream", "max_tokens"
    }:
        raise ProviderError(
            "configuration", None,
            "extra_body cannot override model, messages, tools, stream, or max_tokens."
        )
    max_tokens = model.get("max_tokens", 8192)
    if not isinstance(max_tokens, int) or isinstance(max_tokens,
                                                     bool) or max_tokens <= 0:
        raise ProviderError("configuration", None,
                            "max_tokens must be a positive integer.")
    body = {
        **extra, "model": model["model"],
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens
    }
    if tools is not None:
        body["tools"] = tools
    try:
        encoded = json.dumps(body, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        raise ProviderError(
            "configuration", None,
            "Request values must be JSON serializable.") from None
    credential = _credential(model)
    request = Request(base.rstrip("/") + "/chat/completions",
                      data=encoded,
                      headers={
                          "Authorization": "Bearer " + credential,
                          "Content-Type": "application/json",
                          "Accept": "application/json"
                      },
                      method="POST")
    deadline = time.monotonic() + timeout
    try:
        with _open(request,
                   timeout=max(0.001,
                               deadline - time.monotonic())) as response:
            raw = _read_response(response, deadline)
    except HTTPError as exc:
        status = exc.code
        headers = exc.headers or {}
        retry_after = _retry_after(headers.get("Retry-After"))
        request_id = _error_request_id(headers, credential)
        try:
            provider_reason = _error_reason(exc, deadline)
        finally:
            try:
                exc.close()
            except OSError:
                pass
        kind = "authentication" if status in (
            401, 403) else "rate_limit" if status == 429 else "http"
        raise ProviderError(kind,
                            status,
                            f"Provider returned HTTP {status}.",
                            status in (408, 429) or 500 <= status < 600,
                            retry_after_seconds=retry_after,
                            request_id=request_id,
                            provider_reason=provider_reason) from None
    except (TimeoutError, socket.timeout):
        raise ProviderError("timeout", None,
                            "Provider request exceeded its timeout.",
                            True) from None
    except URLError as exc:
        kind = "timeout" if isinstance(exc.reason,
                                       (TimeoutError,
                                        socket.timeout)) else "network"
        raise ProviderError(kind, None, "Provider connection failed.",
                            True) from None
    except OSError:
        raise ProviderError("network", None, "Provider connection failed.",
                            True) from None
    try:
        result = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ProviderError("invalid_response", None,
                            "Provider response was not valid JSON.") from None
    if (not isinstance(result, dict)
            or not isinstance(result.get("choices"), list)
            or not result["choices"]
            or not isinstance(result["choices"][0], dict)
            or not isinstance(result["choices"][0].get("message"), dict)):
        raise ProviderError(
            "invalid_response", None,
            "Provider response did not contain a chat-completion choice.")
    return result


def _preflight_request(model, messages, tools, timeout):
    """One phase: at most three requests sharing one request/backoff deadline."""
    attempts, backoff, result, error = [], 0.0, None, None
    stop_reason = None
    if isinstance(timeout, bool) or not isinstance(
            timeout,
        (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        error = ProviderError("configuration", None,
                              "timeout must be a positive finite number.")
    else:
        deadline = time.monotonic() + timeout
        for index in range(MAX_PREFLIGHT_RETRIES + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error = error or ProviderError(
                    "timeout", None, "Preflight phase deadline exhausted.",
                    True)
                stop_reason = "phase_deadline"
                break
            started = time.monotonic()
            attempt = {"request_index": index + 1}
            try:
                result = complete(model, messages, tools, remaining)
                error = None
                attempt["returned_model"] = result.get("model")
            except ProviderError as exc:
                error = exc
                attempt["error"] = exc.as_dict()
            attempt["latency_seconds"] = round(time.monotonic() - started, 3)
            attempts.append(attempt)
            if error is None or not error.retryable or error.status not in PREFLIGHT_RETRY_STATUSES:
                break
            if index == MAX_PREFLIGHT_RETRIES:
                stop_reason = "retry_limit"
                break
            delay = error.retry_after_seconds if error.retry_after_seconds is not None else 2.0
            if delay >= deadline - time.monotonic():
                stop_reason = "phase_deadline"
                break
            before_sleep = time.monotonic()
            time.sleep(delay)
            waited = time.monotonic() - before_sleep
            attempt["scheduled_backoff_seconds"] = delay
            attempt["backoff_seconds"] = round(waited, 3)
            backoff += waited
    details = {
        "request_count": len(attempts),
        "retry_count": max(0,
                           len(attempts) - 1),
        "backoff_seconds": round(backoff, 3),
        "attempts": attempts
    }
    if stop_reason:
        details["retry_stop_reason"] = stop_reason
    return result, error, details


def preflight(model: dict, timeout: float = 30, *, progress=None) -> dict:
    """Check text completion and actual echo tool calling without executing it.

    Each phase shares one timeout across up to two retries of HTTP 408/429 and
    transient server errors (500/502/503/504),
    including backoff. Other failures and capability mismatches are not retried.
    Evidence includes retry errors, backoff, exact returned model IDs, usage and
    timing, but never raw model text, credential paths, or credentials.
    """
    config = {
        **model, "max_tokens": min(model.get("max_tokens", 512), 512),
        "extra_body": dict(model.get("extra_body") or {})
    }
    try:
        parsed = urlsplit(model.get("base_url", ""))
        safe_base_url = (model.get("base_url")
                         if not (parsed.username or parsed.password
                                 or parsed.query or parsed.fragment) else None)
    except (TypeError, ValueError):
        safe_base_url = None
    evidence = {
        "id": model.get("id"),
        "status": "running",
        "ok": None,
        "requested_model": model.get("model"),
        "base_url": safe_base_url,
        "timeout_per_request_seconds": timeout,
        "timeout_per_phase_seconds": timeout,
        "retry_policy": {
            "max_retries_per_phase":
            MAX_PREFLIGHT_RETRIES,
            "http_statuses":
            list(PREFLIGHT_RETRY_STATUSES),
            "fallback_backoff_seconds":
            2.0,
            "retry_after_policy":
            "honor seconds or HTTP-date within phase deadline"
        }
    }
    tool = {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Return the supplied value unchanged.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string"
                    }
                },
                "required": ["value"],
                "additionalProperties": False
            }
        }
    }
    for phase in ("completion", "tool_call"):
        start = time.monotonic()
        evidence["active_phase"] = phase
        if progress is not None:
            progress(evidence)
        config["extra_body"].pop("tool_choice", None)
        # Exercise ordinary tool selection, as the campaign does. Some providers
        # support tool calls but reject an explicitly forced named tool_choice.
        prompt = "Reply with exactly OK." if phase == "completion" else 'Call the echo tool with value "preflight". Do not answer in prose.'
        try:
            result, error, retry_details = _preflight_request(
                config, [{
                    "role": "user",
                    "content": prompt
                }], None if phase == "completion" else [tool], timeout)
        except KeyboardInterrupt:
            evidence[phase] = {
                "ok": False,
                "latency_seconds": round(time.monotonic() - start, 3),
                "error": {
                    "kind": "interrupted",
                    "status": None,
                    "retryable": False,
                    "message": "Preflight interrupted by user."
                }
            }
            evidence.update(status="interrupted", ok=False)
            if progress is not None:
                progress(evidence)
            raise
        if error is None:
            message = result["choices"][0]["message"]
            ok = bool(
                isinstance(message.get("content"), str)
                and message["content"].strip())
            if phase == "tool_call":
                ok = False
                calls = message.get("tool_calls")
                for call in calls if isinstance(calls, list) else []:
                    try:
                        function = call["function"]
                        args = json.loads(function["arguments"])
                        ok = ok or (function["name"] == "echo" and args == {
                            "value": "preflight"
                        })
                    except (KeyError, TypeError, ValueError):
                        continue
            row = {
                "ok": ok,
                "returned_model": result.get("model"),
                "usage": result.get("usage"),
                "finish_reason": result["choices"][0].get("finish_reason")
            }
            if not ok:
                row["error"] = {
                    "kind":
                    "capability",
                    "status":
                    None,
                    "retryable":
                    False,
                    "message":
                    "No nonempty completion." if phase == "completion" else
                    "The requested echo tool call was not returned."
                }
        else:
            row = {"ok": False, "error": error.as_dict()}
        row.update(retry_details)
        row["latency_seconds"] = round(time.monotonic() - start, 3)
        evidence[phase] = row
        evidence["active_phase"] = None
        if progress is not None:
            progress(evidence)
    evidence[
        "ok"] = evidence["completion"]["ok"] and evidence["tool_call"]["ok"]
    evidence["status"] = "complete"
    if progress is not None:
        progress(evidence)
    return evidence
