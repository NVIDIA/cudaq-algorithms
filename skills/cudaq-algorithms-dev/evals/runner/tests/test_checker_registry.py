import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner import campaign, checkers


def test_exact_checker_coverage_and_preregistered_specs():
    contracts = json.loads(
        (campaign.EVALS / 'case_contracts.json').read_text())
    specifications = checkers.specifications()
    assert set(specifications) == {
        key
        for key, value in contracts.items() if value['executable_check']
    }
    assert len(specifications) == 24
    for key, spec in specifications.items():
        assert spec['oracle'] and spec['tolerances'] and spec['output']
        public = checkers.public_contract(key)
        assert public == spec['output']
        assert 'oracle' not in public


def test_private_checker_sources_are_fingerprinted():
    files = campaign.evaluator_inventory()
    assert 'evals/runner/checkers/common.py' in files
    assert 'evals/runner/checkers/dynamics.py' in files
    assert 'evals/runner/verification.py' in files


def test_campaign_freezes_checker_contracts_before_execution(tmp_path):
    spec = campaign.prepare(
        tmp_path / 'campaign',
        campaign.EVALS.parents[2], [{
            'id': 'fake',
            'model': 'fake/model'
        }],
        pilot=True,
        case_ids=['repository-implementation-third-moment'])
    assert spec['checker_contracts'] == checkers.specifications()
    assert spec['verification_policy'] == 'after_final_answer_v1'
