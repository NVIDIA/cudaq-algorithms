# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge transport failures must not lose or replace successful observations."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner import assessment, campaign, cli, providers


@pytest.fixture
def grading(tmp_path, monkeypatch):
    root = tmp_path / 'campaign'
    campaign.prepare(root,
                     campaign.EVALS.parents[2], [{
                         'id': 'worker',
                         'model': 'worker/model'
                     }],
                     pilot=True,
                     case_ids=['explicit-primitive-reasoning'])
    case = campaign.read_json(root / 'suite.json')['evals'][0]
    attempt = root / 'attempts/worker' / case['id'] / '0/baseline'
    attempt.mkdir(parents=True)
    campaign.write_json(attempt / 'result.json', {
        'case_id': case['id'],
        'outcome': 'answered'
    })
    (attempt / 'transcript.jsonl').write_text('An answer was recorded.\n')
    unknown = {
        'value': None,
        'evidence': [],
        'reason': 'Insufficient evidence.'
    }
    grade = {
        'case_id':
        case['id'],
        'assertions': [{
            'index': i,
            **unknown
        } for i in range(len(case['assertions']))],
        **{
            key: unknown
            for key in ('expected_output', 'claimed_success', 'critical_failures')
        },
        'tripwires': {
            key: unknown
            for key in assessment.TRIPWIRES
        },
    }
    reply = {
        'model':
        'exact/judge',
        'usage': {
            'total_tokens': 8
        },
        'choices': [{
            'message': {
                'content': json.dumps(grade)
            },
            'finish_reason': 'stop'
        }]
    }
    clock = SimpleNamespace(now=0., sleeps=[])

    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(cli.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(cli.time, 'sleep', sleep)
    return root, attempt, reply, clock


def transport_records(attempt):
    return [
        json.loads(path.read_text())
        for path in attempt.glob('judge-transport-*.json')
    ]


def test_grade_retries_rate_limit_with_same_payload_and_shared_deadline(
        grading, monkeypatch):
    root, attempt, reply, clock = grading
    requests = []
    judge = {'model': 'exact/judge'}

    def complete(model, messages, tools, timeout):
        requests.append(
            (model, json.loads(json.dumps(messages)), tools, timeout))
        clock.now += 1
        if len(requests) == 1:
            raise providers.ProviderError('rate_limit',
                                          429,
                                          'HTTP 429',
                                          True,
                                          retry_after_seconds=3)
        return reply

    monkeypatch.setattr(providers, 'complete', complete)
    assert cli.grade(root, judge, 10) == {'new_assessments': 1}
    assert [item[3] for item in requests] == [10, 6]
    assert requests[0][:3] == requests[1][:3]
    assert clock.sleeps == [3]
    assert campaign.read_json(attempt / 'judge-response.json') == reply
    record, = transport_records(attempt)
    assert record['request_count'] == 2 and record['retry_count'] == 1
    assert record['attempts'][0]['error']['status'] == 429
    assert record['backoff_seconds'] == 3
    assert record['wall_seconds'] == 5


@pytest.mark.parametrize(
    'kind,status,retryable,after,expected_calls,expected_sleeps,stop', [
        ('rate_limit', 429, True, None, 3, [2, 4], 'retry_limit'),
        ('timeout', None, True, None, 3, [2, 4], 'retry_limit'),
        ('http', 503, True, 40, 1, [], 'deadline'),
        ('authentication', 401, False, None, 1, [], 'nonretryable'),
        ('authentication', 403, True, None, 1, [], 'nonretryable'),
        ('configuration', None, False, None, 1, [], 'nonretryable'),
    ])
def test_grade_transport_limits_and_nonretryable_errors(
        grading, monkeypatch, kind, status, retryable, after, expected_calls,
        expected_sleeps, stop):
    root, attempt, _, clock = grading
    requests = []

    def complete(model, messages, tools, timeout):
        requests.append(timeout)
        clock.now += 1
        raise providers.ProviderError(kind,
                                      status,
                                      'safe failure',
                                      retryable,
                                      retry_after_seconds=after)

    monkeypatch.setattr(providers, 'complete', complete)
    with pytest.raises(providers.ProviderError):
        cli.grade(root, {'model': 'exact/judge'}, 20)
    assert len(requests) == expected_calls
    assert clock.sleeps == expected_sleeps
    assert not (attempt / 'judge-response.json').exists()
    record, = transport_records(attempt)
    assert record['request_count'] == expected_calls
    assert record['stop_reason'] == stop
    assert record['wall_seconds'] == clock.now


def test_grade_does_not_request_after_backoff_oversleeps_deadline(
        grading, monkeypatch):
    root, attempt, _, clock = grading

    def complete(*args):
        raise providers.ProviderError('rate_limit', 429, 'HTTP 429', True)

    monkeypatch.setattr(providers, 'complete', complete)
    monkeypatch.setattr(cli.time, 'sleep', lambda _: setattr(clock, 'now', 11))
    with pytest.raises(providers.ProviderError):
        cli.grade(root, {'model': 'exact/judge'}, 10)
    record, = transport_records(attempt)
    assert record['request_count'] == 1
    assert record['stop_reason'] == 'deadline'
    assert record['backoff_seconds'] == 11


def test_grade_counts_failed_request_latency_against_retry_budget(
        grading, monkeypatch):
    root, attempt, _, clock = grading

    def complete(*args):
        clock.now += 9
        raise providers.ProviderError('http', 503, 'HTTP 503', True)

    monkeypatch.setattr(providers, 'complete', complete)
    with pytest.raises(providers.ProviderError):
        cli.grade(root, {'model': 'exact/judge'}, 10)
    record, = transport_records(attempt)
    assert record['request_count'] == 1 and record['wall_seconds'] == 9
    assert record['stop_reason'] == 'deadline'
    assert clock.sleeps == []


def test_grade_interrupted_backoff_retains_partial_elapsed_wait(
        grading, monkeypatch):
    root, attempt, _, clock = grading

    def complete(*args):
        raise providers.ProviderError('http', 503, 'HTTP 503', True)

    def sleep(seconds):
        clock.now += .5
        raise KeyboardInterrupt

    monkeypatch.setattr(providers, 'complete', complete)
    monkeypatch.setattr(cli.time, 'sleep', sleep)
    with pytest.raises(KeyboardInterrupt):
        cli.grade(root, {'model': 'exact/judge'}, 10)
    record, = transport_records(attempt)
    assert record['request_count'] == 1 and record['backoff_seconds'] == .5
    assert record['stop_reason'] == 'interrupted'
    assert not (attempt / 'judge-response.json').exists()


def test_grade_resume_reuses_raw_response_and_rejects_changed_evidence(
        grading, monkeypatch):
    root, attempt, reply, _ = grading
    monkeypatch.setattr(providers, 'complete', lambda *args: reply)
    cli.grade(root, {'model': 'exact/judge'}, 10)
    raw = (attempt / 'judge-response.json').read_bytes()
    provenance = (attempt / 'judge-response.provenance.json').read_bytes()
    for name in ('assessment.json', 'assessment.provenance.json'):
        (attempt / name).unlink()
    monkeypatch.setattr(
        providers, 'complete',
        lambda *args: pytest.fail('retained response must be reused'))
    assert cli.grade(root, {'model': 'different/judge'}, 10) == {
        'new_assessments': 1
    }
    assert (attempt / 'judge-response.json').read_bytes() == raw
    assert (attempt / 'assessment.provenance.json').read_bytes() == provenance
    for name in ('assessment.json', 'assessment.provenance.json'):
        (attempt / name).unlink()
    (attempt / 'transcript.jsonl').write_text('Changed evidence.\n')
    with pytest.raises(ValueError, match='changed evidence'):
        cli.grade(root, {'model': 'exact/judge'}, 10)


@pytest.mark.parametrize('invalid', ['{not json}', 'fabricated quote'])
def test_malformed_judgment_is_retained_and_never_transport_retried(
        grading, monkeypatch, invalid):
    root, attempt, reply, _ = grading
    if invalid == 'fabricated quote':
        grade = json.loads(reply['choices'][0]['message']['content'])
        grade['assertions'][0].update(value=True,
                                      evidence=['not in transcript'])
        invalid = json.dumps(grade)
    reply['choices'][0]['message']['content'] = invalid

    def complete(*args):
        assert not transport_records(
            attempt), 'invalid judgment must not be retried'
        return reply

    monkeypatch.setattr(providers, 'complete', complete)
    with pytest.raises(ValueError):
        cli.grade(root, {'model': 'exact/judge'}, 10)
    retained = (attempt / 'judge-response.json').read_bytes()
    monkeypatch.setattr(
        providers, 'complete',
        lambda *args: pytest.fail('invalid raw response must be retained'))
    with pytest.raises(ValueError):
        cli.grade(root, {'model': 'exact/judge'}, 10)
    assert (attempt / 'judge-response.json').read_bytes() == retained
    assert not (attempt / 'assessment.json').exists()
