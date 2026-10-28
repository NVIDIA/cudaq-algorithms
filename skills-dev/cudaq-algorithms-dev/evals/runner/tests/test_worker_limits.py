# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real worker output capture before transcript serialization."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner import campaign, providers


@pytest.mark.parametrize('streams,limit', [('stdout', 1024), ('stderr', 1024),
                                           ('combined', 1024),
                                           ('stdout', 70000)])
def test_worker_limits_real_output_before_serializing_it(
        tmp_path, monkeypatch, streams, limit):
    (tmp_path / 'source').mkdir()
    (tmp_path / 'fixtures').mkdir()
    attempt = tmp_path / 'attempts/test/case/0/baseline'
    attempt.mkdir(parents=True)
    monkeypatch.setattr(campaign,
                        'MAX_TOOL_OUTPUT_BYTES',
                        limit,
                        raising=False)
    script = {
        'stdout': 'import os; os.write(1, b"x"*100000)',
        'stderr': 'import os; os.write(2, b"x"*100000)',
        'combined': 'import os; os.write(1, b"a"*700); os.write(2, b"b"*700)',
    }[streams]
    command = "python3 -c '" + script + "'"
    replies = []

    def complete(model, messages, tools, timeout):
        if messages[-1]['role'] == 'tool':
            replies.append(messages[-1]['content'])
            message = {'content': 'Observed bounded output.'}
        else:
            message = {
                'content':
                None,
                'tool_calls': [{
                    'id': 'call',
                    'type': 'function',
                    'function': {
                        'name': 'shell',
                        'arguments': json.dumps({'command': command})
                    }
                }]
            }
        return {
            'choices': [{
                'message': message,
                'finish_reason': 'stop'
            }],
            'usage': {
                'total_tokens': 1
            }
        }

    monkeypatch.setattr(providers, 'complete', complete)
    result = campaign.run_attempt(tmp_path, attempt, {
        'id': 'case',
        'prompt': 'Run the requested command.',
        'files': []
    }, {'alias': 'test'}, {
        'case_id': 'case',
        'seed': 0,
        'arm': 'baseline'
    }, {
        'settings': {
            'isolation': 'trusted',
            'request_timeout': 10
        },
        'protocol': {
            'budget_seconds': 10,
            'tool_call_limit': 2,
            'skill_exposure': 'listed'
        }
    })
    assert result['outcome'] == 'answered'
    events = [
        json.loads(line)
        for line in (attempt / 'transcript.jsonl').read_text().splitlines()
    ]
    output, = [event['output'] for event in events if event['event'] == 'tool']
    assert output['output_limit_exceeded'] is True
    assert len(output['stdout']) <= limit and len(output['stderr']) <= limit
    for stream in ('stdout', 'stderr'):
        raw = Path(output[stream + '_log']).read_bytes()
        assert output[stream + '_bytes'] == len(raw)
        assert output[stream + '_truncated'] is (len(raw) > limit)
        assert output[stream] == raw[:limit].decode()
    assert output['stdout_bytes'] + output['stderr_bytes'] > limit
    assert output['termination_reason'] in ('process_exit', 'output_limit')
    assert (attempt / 'transcript.jsonl').stat().st_size < 2 * limit + 6000
    if len(json.dumps(output)) > 48000:
        notice = json.loads(replies[0].rsplit('\n', 1)[1])
        assert notice['model_context_truncated']
        for key in ('stdout_log', 'stderr_log', 'output_limit_exceeded',
                    'termination_reason', 'stdout_truncated',
                    'stderr_truncated'):
            assert notice[key] == output[key]
        assert len(replies[0]) < 50000
    else:
        assert json.loads(replies[0]) == output
