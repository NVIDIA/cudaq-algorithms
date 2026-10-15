import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner import campaign, providers, workspace


def run_sequence(tmp_path, monkeypatch, failures, budget=120, retries=5):
    current = [0.]
    delays = []
    seen = []

    def sleep(delay):
        delays.append(delay)
        current[0] += delay

    monkeypatch.setattr(
        campaign, 'time',
        SimpleNamespace(monotonic=lambda: current[0],
                        time=lambda: 1000 + current[0],
                        sleep=sleep))
    monkeypatch.setattr(
        workspace, 'stage_workspace', lambda *a: {
            'workspace_path': str(tmp_path),
            'skill_path': None
        })
    events = list(failures)

    def complete(model, messages, tools, timeout):
        seen.append(json.dumps(messages))
        if events:
            raise events.pop(0)
        return {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'Done'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'total_tokens': 5
            }
        }

    monkeypatch.setattr(providers, 'complete', complete)
    result = campaign.run_attempt(tmp_path, tmp_path, {
        'id': 'test',
        'prompt': 'Original',
        'files': []
    }, {'alias': 'nemo'}, {
        'case_id': 'test',
        'seed': 0,
        'arm': 'baseline'
    }, {
        'settings': {
            'request_timeout': 90,
            'transport_max_retries': retries
        },
        'protocol': {
            'budget_seconds': budget,
            'tool_call_limit': 5,
            'skill_exposure': 'listed'
        }
    })
    return result, delays, seen


def test_recovers_after_observed_500_429_500_sequence_without_restarting_task(
        tmp_path, monkeypatch):
    errors = [
        providers.ProviderError('http', 500, '500', True),
        providers.ProviderError('rate_limit', 429, '429', True),
        providers.ProviderError('http', 500, '500', True)
    ]
    result, delays, seen = run_sequence(tmp_path, monkeypatch, errors)
    assert result['outcome'] == 'answered'
    assert delays == [2, 4, 8]
    assert len(seen) == 4 and len(set(seen)) == 1
    assert result['resources']['task_seconds'] == 14
    assert result['resources']['tokens'] is None


def test_honors_retry_after_without_early_request(tmp_path, monkeypatch):
    error = providers.ProviderError('rate_limit',
                                    429,
                                    '429',
                                    True,
                                    retry_after_seconds=70)
    result, delays, seen = run_sequence(tmp_path, monkeypatch, [error])
    assert result['outcome'] == 'answered'
    assert delays == [70]


@pytest.mark.parametrize('budget,retries', [(3, 5), (120, 0)])
def test_retries_remain_bounded(tmp_path, monkeypatch, budget, retries):
    errors = [
        providers.ProviderError('http', 500, '500', True) for _ in range(10)
    ]
    result, delays, seen = run_sequence(tmp_path, monkeypatch, errors, budget,
                                        retries)
    assert result['outcome'] == 'backend_error'
    assert sum(delays) < budget
    assert len(seen) <= retries + 1


def test_header_beyond_remaining_budget_does_not_retry_early(
        tmp_path, monkeypatch):
    error = providers.ProviderError('rate_limit',
                                    429,
                                    '429',
                                    True,
                                    retry_after_seconds=300)
    result, delays, seen = run_sequence(tmp_path,
                                        monkeypatch, [error],
                                        budget=120)
    assert result['outcome'] == 'backend_error'
    assert len(seen) == 1 and delays == []
