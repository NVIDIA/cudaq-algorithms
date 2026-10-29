# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Navigate the maintained catalog's actual prompts, without private answers.

These are navigation regressions, not evidence that a model follows the skill.
The exact unmatched cases below are intentional: family selection must not add
an unsupported API, guess an abstract method, or override the outer task scope.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'evals'))
from runner import campaign

ROOT = Path(__file__).resolve().parents[4]
DEVELOPMENT = ROOT / 'skills-dev/cudaq-algorithms-dev/evals'
RUNTIME = ROOT / 'skills/cudaq-algorithms'
SPEC = importlib.util.spec_from_file_location('catalog_router',
                                              RUNTIME / 'scripts/route.py')
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)
CATALOG = json.loads((DEVELOPMENT / 'evals.json').read_text())
# Discard answer material before constructing any routing input.
CASES = [{
    key: case.get(key)
    for key in ('id', 'prompt', 'files', 'expected_skill')
} for case in CATALOG['evals']]

# Expected families are hand-reviewed from prompts, never computed by router.
REQUIRED = {
    'implicit-composition-boundary': 'application-composition.md',
    'explicit-primitive-reasoning': 'primitives/primitives.md',
    'unsupported-library-workflow': 'chemistry/chemistry-bridges.md',
    'unknown-future-primitive': 'block-encoding/block-encoding.md',
    'state-preparation-provider-selection':
    'state-preparation/state-preparation.md',
    'state-preparation-ucc-parameterization-boundary':
    'state-preparation/state-preparation.md',
    'state-preparation-reference-excitation-device-boundary':
    'state-preparation/state-preparation.md',
    'state-preparation-uccsd-open-shell-parity':
    'state-preparation/state-preparation.md',
    'state-preparation-grouped-ucc-device-boundary':
    'state-preparation/operator-pools.md',
    'state-preparation-givens-device-boundary':
    'state-preparation/state-preparation.md',
    'state-preparation-injection-composition':
    'state-preparation/injection-contract.md',
    'state-preparation-controlled-adjoint-unknown':
    'state-preparation/state-preparation.md',
    'state-preparation-measurement-assisted-negative':
    'state-preparation/injection-contract.md',
    'operator-pool-selection-boundary': 'state-preparation/operator-pools.md',
    'block-encoding-capability-boundary': 'block-encoding/block-encoding.md',
    'qubitization-walk-moment-boundary': 'qubitization/qubitization.md',
    'qsvt-phase-and-recovery-boundary': 'qsvt/qsvt.md',
    'trotter-evolution-resource-boundary': 'trotter/trotter.md',
    'fermion-transform-selection-boundary':
    'fermion-transforms/fermion-transforms.md',
    'chemistry-bridge-dependency-boundary': 'chemistry/chemistry-bridges.md',
    'double-factorization-encoding-boundary':
    'double-factorization/double-factorization.md',
    'double-factorization-host-contracts':
    'double-factorization/double-factorization.md',
    'simulation-analysis-hardware-boundary':
    'simulation/simulation-analysis.md',
    'implicit-ftqc-application-advisory': 'chemistry/chemistry-bridges.md',
    'state-preparation-width-mismatch-unknown':
    'state-preparation/injection-contract.md',
    'material-ambiguity-clarification': 'trotter/trotter.md',
    'harmless-assumption-proceed': 'block-encoding/block-encoding.md',
    'contradictory-output-contract': 'trotter/trotter.md',
    'source-unavailable-fallback': 'qsvt/qsvt.md',
    'source-version-drift': 'qsvt/qsvt.md',
    'qpu-authorization-boundary': 'trotter/trotter.md',
    'unexecuted-code-evidence': 'qsvt/qsvt.md',
    'qsvt-paraphrase-convention': 'qsvt/qsvt.md',
    'chemistry-end-to-end-composition': 'chemistry/chemistry-bridges.md',
    'implementation-scope-overreach': 'block-encoding/block-encoding.md',
    'operator-pool-ceo-units': 'state-preparation/operator-pools.md',
    'science-state-preparation-orbital-phases':
    'state-preparation/state-preparation.md',
    'science-state-preparation-correlated-ordering':
    'state-preparation/state-preparation.md',
    'science-block-encoding-heralded-observables':
    'block-encoding/block-encoding.md',
    'science-block-encoding-truncation-tradeoff':
    'block-encoding/block-encoding.md',
    'science-qubitization-interferometric-amplitude':
    'qubitization/qubitization.md',
    'science-qubitization-reference-dependent-convergence':
    'qubitization/qubitization.md',
    'science-qsvt-low-energy-filter-choice': 'qsvt/qsvt.md',
    'science-qsvt-calibration-robustness': 'qsvt/qsvt.md',
    'science-trotter-quench-cost': 'trotter/trotter.md',
    'science-trotter-conservation-versus-accuracy': 'trotter/trotter.md',
    'science-fermion-transforms-persistent-current':
    'fermion-transforms/fermion-transforms.md',
    'science-fermion-transforms-hubbard-transport':
    'fermion-transforms/fermion-transforms.md',
    'science-chemistry-reference-quality': 'chemistry/chemistry-bridges.md',
    'science-chemistry-water-bending': 'chemistry/chemistry-bridges.md',
    'science-double-factorization-leaf-budget':
    'double-factorization/double-factorization.md',
    'science-double-factorization-optimized-tradeoff':
    'double-factorization/double-factorization.md',
    'science-simulation-probabilistic-cooling':
    'simulation/simulation-analysis.md',
    'science-simulation-postselection-versus-discard':
    'simulation/simulation-analysis.md',
    'science-workflow-molecular-bond-stretching':
    'chemistry/chemistry-bridges.md',
    'science-workflow-frustrated-domain-wall': 'trotter/trotter.md',
}
NEGATIVES = {
    'negative-cudaq-install', 'negative-generic-quantum-concept',
    'negative-generic-cudaq-kernel'
}
EXCEPTIONS = {
    'contextual-existing-primitive': {
        'reason':
        'Unverified apply_local_phase requires exact-name source and contract lookup; no substitute.',
        'fallback': ('catalog.md', 'source-provenance.md', 'conventions.md'),
        'scope':
        'Give bounded source-grounded advice or request missing evidence; do not invent a signature.',
        'files': [],
    },
    'repository-implementation-third-moment': {
        'reason':
        'The outer implementation prompt delegates mathematical detail to declared files.',
        'fallback': ('catalog.md', 'qubitization/qubitization-moments.md'),
        'scope':
        'Repair the staged starter and validate within the outer implementation request.',
        'files': ['files/application_request.md', 'files/starter_app.py'],
    },
    'fixture-prompt-injection-boundary': {
        'reason':
        'The outer review prompt delegates mathematical detail to its one declared file.',
        'fallback': ('catalog.md', 'qubitization/qubitization-moments.md'),
        'scope':
        'Recommend a chain and validation plan only; starter/Hamiltonian are missing. Quoted instructions are data.',
        'files': ['files/application_request.md'],
    },
}


def test_catalog_mirror_and_explicit_case_coverage():
    assert CATALOG == json.loads((RUNTIME / 'evals/evals.json').read_text())
    assert len(CASES) == len({case['id'] for case in CASES}) == 62
    assert {case['id']
            for case in CASES} == set(REQUIRED) | set(EXCEPTIONS) | NEGATIVES
    assert {case['id']
            for case in CASES if case['expected_skill'] is None} == NEGATIVES


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['id'])
def test_actual_catalog_prompt_routes_or_has_an_exact_documented_exception(
        case):
    records = router.route(case['prompt'])
    assert len(records) == len(set(records)) <= router.MAX_RECORDS
    assert all((router.REFS / record).is_file() for record in records)
    if case['id'] in NEGATIVES:
        assert records == []
    elif case['id'] in EXCEPTIONS:
        exception = EXCEPTIONS[case['id']]
        assert records == [], 'An arbitrary family or substitute is not an accepted fallback'
        assert case['files'] == exception['files']
        assert all((router.REFS / record).is_file()
                   for record in exception['fallback'])
        output = subprocess.check_output([
            sys.executable,
            str(RUNTIME / 'scripts/route.py'), case['prompt']
        ],
                                         text=True)
        assert 'references/catalog.md' in output
        assert '## references/' not in output
    else:
        assert REQUIRED[case['id']] in records


@pytest.mark.parametrize('case_id', [
    'repository-implementation-third-moment',
    'fixture-prompt-injection-boundary'
])
def test_fixture_led_prompts_resolve_only_after_reading_declared_context(
        case_id):
    case = next(case for case in CASES if case['id'] == case_id)
    outer_prompt = case['prompt']
    assert router.route(outer_prompt) == []
    context = '\n\n'.join(
        (DEVELOPMENT / name).read_text() for name in case['files'])
    # Context is navigational data, never a replacement for the outer request.
    records = router.route(outer_prompt + '\n\n' + context)
    assert records[0] == 'qubitization/qubitization.md'
    assert (router.REFS / 'qubitization/qubitization-moments.md').is_file()
    # The harness must keep the actual outer user instruction, including the
    # distinction between advice and authorized repair, above fixture contents.
    messages = campaign.worker_messages(case, 'baseline', 'listed', '')
    assert [message for message in messages if message['role'] == 'user'] == [{
        'role':
        'user',
        'content':
        outer_prompt
    }]
    supplied = messages[0]['content']
    assert '/workspace/.eval-inputs/files/application_request.md' in supplied
    assert ('/workspace/.eval-inputs/files/starter_app.py' in supplied) is (
        case_id == 'repository-implementation-third-moment')
