"""Fault-injected transport time accounting; no API calls or real sleeps."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner import campaign, cli, providers, workspace


def unavailable(retry_after=None):
    return providers.ProviderError('http', 503, 'Overloaded', True,
                                   retry_after_seconds=retry_after)


def run_timeline(tmp_path, monkeypatch, requests, *, task_budget=20,
                 wait_budget=100, retries=None, sleep_extra=0,
                 interrupt_sleep=False, verify=False, respect_timeout=True):
    now, seen, sleeps = [0.], [], []
    sequence = iter(requests)

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += .5 if interrupt_sleep else seconds + sleep_extra
        if interrupt_sleep:
            raise KeyboardInterrupt

    monkeypatch.setattr(campaign, 'time', SimpleNamespace(
        monotonic=lambda: now[0], time=lambda: 1000 + now[0], sleep=sleep))
    monkeypatch.setattr(workspace, 'stage_workspace', lambda *a: {
        'workspace_path': str(tmp_path), 'skill_path': None})

    def complete(model, messages, tools, timeout):
        seen.append({'messages': json.dumps(messages), 'timeout': timeout})
        seconds, response = next(sequence)
        if respect_timeout and seconds > timeout:
            now[0] += timeout
            raise providers.ProviderError('timeout', None, 'Timed out', True)
        now[0] += seconds
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, dict):
            return response
        return {'choices': [{'message': {'role': 'assistant', 'content': 'Done'},
                             'finish_reason': 'stop'}],
                'usage': {'total_tokens': 5}}

    monkeypatch.setattr(providers, 'complete', complete)
    settings = {'request_timeout': 180}
    if wait_budget is not None:
        settings['backend_wait_budget_seconds'] = wait_budget
        settings['transport_max_retries'] = retries
    elif retries is not None:
        settings['transport_max_retries'] = retries
    spec = {'settings': settings, 'protocol': {
        'budget_seconds': task_budget, 'tool_call_limit': 5,
        'skill_exposure': 'listed'}}
    if verify:
        spec['case_contracts'] = {'test': {'executable_check': True}}

        def execute(*args, deadline, **kwargs):
            assert deadline == 23
            now[0] += 3
            campaign.write_json(tmp_path / 'check.json', {'passed': True})
            return {'status': 'passed', 'evidence_file': 'check.json'}

        monkeypatch.setattr(campaign.verification, 'execute', execute)
    result = campaign.run_attempt(tmp_path, tmp_path,
        {'id': 'test', 'prompt': 'Original', 'files': []}, {'alias': 'nemo'},
        {'case_id': 'test', 'seed': 0, 'arm': 'baseline'}, spec)
    events = [json.loads(line) for line in
              (tmp_path / 'transcript.jsonl').read_text().splitlines()]
    resources = result['resources']
    assert resources['wall_seconds'] == pytest.approx(
        resources['task_seconds'] + resources['backend_wait_seconds'])
    assert campaign.read_json(tmp_path / 'result.json') == result
    return result, seen, sleeps, events


def test_recovers_beyond_previous_retry_cap_without_restarting(tmp_path, monkeypatch):
    result, seen, sleeps, events = run_timeline(
        tmp_path, monkeypatch, [(1, unavailable())] * 15 + [(4, None)],
        task_budget=10, wait_budget=1800)
    assert result['outcome'] == 'answered'
    assert len(seen) == 16 and len({r['messages'] for r in seen}) == 1
    assert result['resources']['task_seconds'] == 4
    assert result['resources']['backend_wait_seconds'] == 15 + sum(sleeps)
    assert result['resources']['tokens'] is None
    assert len([e for e in events if e['event'] == 'transport_error']) == 15


def test_successful_request_latency_is_charged_without_queue_estimates(tmp_path, monkeypatch):
    result, seen, _, _ = run_timeline(tmp_path, monkeypatch, [(19, None)])
    assert result['outcome'] == 'answered'
    assert result['resources']['task_seconds'] == 19
    assert result['resources']['backend_wait_seconds'] == 0
    assert result['resources']['tokens'] == 5


def test_measured_failed_requests_and_sleep_are_separate_from_task(tmp_path, monkeypatch):
    result, _, _, events = run_timeline(tmp_path, monkeypatch,
        [(3, unavailable()), (4, None)], sleep_extra=.25)
    assert result['outcome'] == 'answered'
    assert result['resources']['task_seconds'] == 4
    assert result['resources']['backend_wait_seconds'] == 5.25
    error = next(e for e in events if e['event'] == 'transport_error')
    assert error['latency_seconds'] == 3
    backoff = next(e for e in events if e['event'] == 'transport_backoff')
    assert backoff['scheduled_seconds'] == 2
    assert backoff['seconds'] == 2.25


@pytest.mark.parametrize('limit,expected_requests', [(0, 1), (1, 2), (10, 11)])
def test_explicit_retry_caps_are_still_honored(tmp_path, monkeypatch, limit, expected_requests):
    result, seen, _, _ = run_timeline(tmp_path, monkeypatch,
        [(1, unavailable())] * 20, retries=limit, wait_budget=1800)
    assert result['outcome'] == 'backend_error'
    assert len(seen) == expected_requests
    assert result['error']['retryable'] is True


def test_retry_after_cannot_exceed_remaining_wait_allowance(tmp_path, monkeypatch):
    result, seen, sleeps, _ = run_timeline(tmp_path, monkeypatch,
        [(2, unavailable(10))], wait_budget=10)
    assert result['outcome'] == 'backend_error'
    assert len(seen) == 1 and sleeps == []
    assert result['error']['retry_after_seconds'] == 10
    assert result['resources']['backend_wait_seconds'] == 2


@pytest.mark.parametrize('retry_after', [0, .1, 1])
def test_small_retry_after_does_not_disable_new_policy_pacing(tmp_path, monkeypatch, retry_after):
    result, seen, sleeps, _ = run_timeline(tmp_path, monkeypatch,
        [(1, unavailable(retry_after)), (1, unavailable(retry_after)), (2, None)])
    assert result['outcome'] == 'answered'
    assert sleeps == [2, 4] and len(seen) == 3
    assert result['resources']['backend_wait_seconds'] == 8


def test_legacy_retry_after_zero_keeps_original_behavior(tmp_path, monkeypatch):
    result, seen, sleeps, _ = run_timeline(tmp_path, monkeypatch,
        [(1, unavailable(0)), (2, None)], wait_budget=None)
    assert result['outcome'] == 'answered'
    assert sleeps == [0] and len(seen) == 2
    assert result['resources']['task_seconds'] == 3


def test_request_timeout_cannot_exceed_remaining_wait_allowance(tmp_path, monkeypatch):
    result, seen, sleeps, _ = run_timeline(tmp_path, monkeypatch,
        [(20, unavailable())], task_budget=20, wait_budget=3)
    assert result['outcome'] == 'backend_error'
    assert seen[0]['timeout'] == 3
    assert sleeps == []
    assert result['resources']['backend_wait_seconds'] == 3
    assert result['resources']['task_seconds'] == 0


def test_sleep_overrun_is_recorded_and_cannot_start_another_request(tmp_path, monkeypatch):
    result, seen, _, _ = run_timeline(tmp_path, monkeypatch,
        [(1, unavailable()), (1, None)], wait_budget=5, sleep_extra=10)
    assert result['outcome'] == 'backend_error'
    assert len(seen) == 1
    assert result['resources']['backend_wait_seconds'] == 13
    assert result['resources']['task_seconds'] == 0


def test_wall_cap_stops_work_even_if_sleep_overshoots_both_budgets(tmp_path, monkeypatch):
    result, seen, _, _ = run_timeline(tmp_path, monkeypatch,
        [(1, unavailable()), (1, None)], task_budget=4,
        wait_budget=5, sleep_extra=10)
    assert result['outcome'] == 'budget_timeout'
    assert len(seen) == 1
    assert result['error']['status'] == 503
    assert result['error']['stop_reason'] == 'wall_budget'
    assert result['resources']['wall_seconds'] == 13


def test_success_returned_after_task_deadline_is_not_accepted(tmp_path, monkeypatch):
    result, seen, _, _ = run_timeline(tmp_path, monkeypatch,
        [(21, None)], respect_timeout=False)
    assert result['outcome'] == 'budget_timeout'
    assert result['final_answer'] is None
    assert seen[0]['timeout'] == 20
    assert result['resources']['task_seconds'] == 21
    assert result['resources']['backend_wait_seconds'] == 0


def test_successful_turns_deplete_task_budget_across_conversation(tmp_path, monkeypatch):
    tool_reply = {'choices': [{'message': {'role': 'assistant', 'content': None,
        'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {
            'name': 'read_file', 'arguments': '{"path":"/workspace/missing"}'}}]},
        'finish_reason': 'tool_calls'}], 'usage': {'total_tokens': 8}}
    result, seen, _, _ = run_timeline(tmp_path, monkeypatch,
        [(17, tool_reply), (1, unavailable()), (2, None)])
    assert result['outcome'] == 'answered'
    assert [request['timeout'] for request in seen] == [20, 3, 3]
    assert result['resources']['task_seconds'] == 19
    assert result['resources']['backend_wait_seconds'] == 3
    assert result['resources']['tool_calls'] == 1


def test_partial_interrupted_sleep_is_counted_and_saved(tmp_path, monkeypatch):
    result, seen, _, _ = run_timeline(tmp_path, monkeypatch,
        [(1, unavailable())], interrupt_sleep=True)
    assert result['outcome'] == 'error'
    assert result['error']['kind'] == 'interrupted'
    assert len(seen) == 1
    assert result['resources']['backend_wait_seconds'] == 1.5
    assert result['resources']['tokens'] is None


def test_partial_interrupted_request_is_counted_and_saved(tmp_path, monkeypatch):
    result, _, _, events = run_timeline(tmp_path, monkeypatch,
        [(2, KeyboardInterrupt())])
    assert result['outcome'] == 'error'
    assert result['resources']['backend_wait_seconds'] == 2
    assert result['resources']['tokens'] is None
    assert any(e['event'] == 'transport_error' and
               e['error']['kind'] == 'interrupted' for e in events)


def test_legacy_campaign_without_new_setting_keeps_charged_time_and_two_retries(tmp_path, monkeypatch):
    result, seen, sleeps, _ = run_timeline(tmp_path, monkeypatch,
        [(1, unavailable())] * 4, wait_budget=None)
    assert result['outcome'] == 'backend_error'
    assert len(seen) == 3 and sleeps == [2, 4]
    assert result['resources']['backend_wait_seconds'] == 0
    assert result['resources']['task_seconds'] == 9


def test_zero_wait_budget_disables_recovery_without_disabling_initial_request(tmp_path, monkeypatch):
    result, seen, sleeps, _ = run_timeline(tmp_path, monkeypatch,
        [(3, unavailable()), (1, None)], wait_budget=0)
    assert result['outcome'] == 'backend_error'
    assert len(seen) == 1 and sleeps == []
    assert result['resources']['backend_wait_seconds'] == 0
    assert result['resources']['task_seconds'] == 3


def test_verification_receives_task_deadline_extended_only_by_measured_wait(tmp_path, monkeypatch):
    result, _, _, _ = run_timeline(tmp_path, monkeypatch,
        [(1, unavailable()), (4, None)], verify=True)
    assert result['resources']['task_seconds'] == 7
    assert result['resources']['wall_seconds'] == 10
    completion = campaign.read_json(tmp_path / 'verification.json')['verified_completion']
    assert completion['task_seconds'] == 7
    assert completion['wall_seconds'] == 10
    assert completion['failed_attempts'] == 0


def test_new_campaign_freezes_bounded_wait_default_and_optional_retry_count(tmp_path):
    spec = campaign.prepare(tmp_path / 'campaign', campaign.EVALS.parents[2],
        [{'id': 'fake', 'model': 'fake/model'}], pilot=True,
        case_ids=['science-block-encoding-heralded-observables'])
    assert spec['settings']['backend_wait_budget_seconds'] == 1800
    assert spec['settings']['transport_max_retries'] is None
    policy = json.loads(spec['environment']['transport_policy'])
    assert policy['backend_wait_budget_seconds'] == 1800
    assert policy['task_budget_seconds'] == 900
    assert policy['wall_cap_seconds'] == 2700
    assert policy['request_timeout_seconds'] == 180
    assert policy['max_retries_per_request'] is None
    args = cli.parser().parse_args(['init', str(tmp_path / 'another'),
                                   '--runtime-python', sys.executable])
    assert args.backend_wait_budget_seconds == 1800
    assert args.transport_max_retries is None


@pytest.mark.parametrize('invalid', [-1, float('inf'), float('nan'), True])
def test_invalid_wait_budget_rejected_before_creating_campaign(tmp_path, invalid):
    path = tmp_path / 'campaign'
    with pytest.raises(ValueError, match='backend_wait_budget_seconds'):
        campaign.prepare(path, campaign.EVALS.parents[2],
            [{'id': 'fake', 'model': 'fake/model'}],
            backend_wait_budget_seconds=invalid)
    assert not path.exists()
