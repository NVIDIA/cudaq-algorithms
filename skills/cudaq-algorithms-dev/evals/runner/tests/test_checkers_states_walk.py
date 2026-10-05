"""Independent fixtures and intentionally wrong scientific payloads."""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm, sqrtm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_states_walk_checker_is_available():
    assert importlib.util.find_spec('runner.checkers.states_walk') is not None


@pytest.fixture
def checker():
    from runner.checkers import states_walk
    return states_walk


def packed(x):
    x = np.asarray(x, dtype=complex)
    return np.stack([x.real, x.imag], axis=-1).tolist()


def kron_word(word):
    matrices = {
        'I': np.eye(2),
        'X': np.array([[0, 1], [1, 0]]),
        'Y': np.array([[0, -1j], [1j, 0]]),
        'Z': np.diag([1, -1])
    }
    out = np.array([[1.]])
    for letter in word[::-1]:
        out = np.kron(out, matrices[letter])
    return out


def terms_matrix(terms):
    return sum(c * kron_word(w) for c, w in terms)


def term(n, coefficient, **ops):
    word = ['I'] * n
    for q, op in ops.items():
        word[int(q)] = op
    return [coefficient, ''.join(word)]


def product(factors):
    out = np.array([1.])
    for factor in factors[::-1]:
        out = np.kron(out, factor)
    return out


def jw(n):
    # Jordan-Wigner tensor matrices, independent of checker's occupation-bit construction.
    lowering = np.array([[0., 1.], [0., 0.]])
    result = []
    for q in range(n):
        op = np.array([[1.]])
        for k in reversed(range(n)):
            op = np.kron(
                op, lowering
                if k == q else np.diag([1., -1.]) if k < q else np.eye(2))
        result.append(op)
    return result


def orbital_payload():
    a = jw(6)
    h = np.zeros((64, 64), complex)
    for j in range(6):
        hop = -(1 if j % 2 == 0 else .6) * np.exp(
            .7j / 6) * a[(j + 1) % 6].conj().T @ a[j]
        h += hop + hop.conj().T
    indices = [i for i in range(64) if i.bit_count() == 3]
    values, vectors = np.linalg.eigh(h[np.ix_(indices, indices)])
    state = np.zeros(64, complex)
    state[indices] = vectors[:, 0]
    coherence = [
        np.vdot(state, a[(j + 1) % 6].conj().T @ a[j] @ state)
        for j in range(6)
    ]
    return {
        'prepared_state': packed(state),
        'energy': float(values[0]),
        'coherences': packed(coherence)
    }


def correlated_payload():
    a = jw(6)
    numbers = [x.conj().T @ x for x in a]
    double = sum(numbers[j] @ numbers[j + 1] for j in [0, 2, 4])
    h = 2 * double
    for j in range(4):
        hop = a[j + 2].conj().T @ a[j]
        h = h - hop - hop.conj().T
    t1 = a[2].conj().T @ a[0] + a[3].conj().T @ a[1]
    t2 = a[4].conj().T @ a[5].conj().T @ a[1] @ a[0]
    ref = np.eye(64)[:, 3]
    u1, u2 = expm(.28 * (t1 - t1.conj().T)), expm(-.19 * (t2 - t2.conj().T))
    states = np.array([u2 @ u1 @ ref, u1 @ u2 @ ref])
    return {
        'prepared_states': packed(states),
        'energies': [np.vdot(s, h @ s).real for s in states],
        'double_occupancies': [np.vdot(s, double @ s).real for s in states],
        'mutual_fidelity': abs(np.vdot(*states))**2
    }


def herald_payload():
    terms = [term(4, .4)]
    for j in range(3):
        terms.extend(
            term(4, c, **{
                str(j): p,
                str(j + 1): q
            }) for c, p, q in [(.25, 'X',
                                'X'), (.25, 'Y',
                                       'Y'), (.175, 'Z',
                                              'Z'), (.075, 'X',
                                                     'Y'), (-.075, 'Y', 'X')])
    terms += [term(4, .1, **{str(j): 'Z'}) for j in range(4)]
    h = terms_matrix(terms)
    m = sum((-1)**j * kron_word('I' * j + 'Z' + 'I' * (3 - j)) / 4
            for j in range(4))
    states = [
        np.eye(16)[:, 10],
        product([np.array([1, 1j**j]) / np.sqrt(2) for j in range(4)])
    ]
    alpha = sum(abs(c) for c, _ in terms)
    branches = [h @ s / alpha for s in states]
    return {
        'lcu_terms':
        terms,
        'alpha':
        alpha,
        'probabilities': [np.vdot(s, s).real for s in branches],
        'magnetizations':
        [np.vdot(s, m @ s).real / np.vdot(s, s).real for s in branches]
    }


def truncation_payload(bounds=False):
    terms = [term(6, 2.4)]
    couplings = [1, .08, .004, .0006, .0001]
    terms += [
        term(6, -couplings[j - i - 1], **{
            str(i): 'Z',
            str(j): 'Z'
        }) for i in range(6) for j in range(i + 1, 6)
    ]
    terms += [term(6, -.7, **{str(j): 'X'}) for j in range(6)]
    exact = np.linalg.eigvalsh(terms_matrix(terms))[0]
    rows = []
    for cutoff, outside in itertools.product([0., .0005, .001, .01],
                                             [False, True]):
        retained = [[c, w] for c, w in terms
                    if abs(c) >= cutoff and not (outside and w == 'IIIIII')]
        energy = np.linalg.eigvalsh(
            terms_matrix(retained))[0] + (2.4 if outside else 0)
        bias = sum(abs(c) for c, _ in terms
                   if abs(c) < cutoff) if bounds else abs(energy - exact)
        row = {
            'cutoff': cutoff,
            'offset_outside': outside,
            'lcu_terms': retained,
            'alpha': sum(abs(c) for c, _ in retained),
            'bias_kind': 'bound' if bounds else 'observed',
            'bias': float(bias),
            'certified': bool(bias <= .001)
        }
        if not bounds:
            row['ground_energy'] = float(energy)
        rows.append(row)
    return {'candidates': rows, 'recommendation': 3}


def return_payload():
    terms = [term(4, .3)]
    terms += [term(4, -.7, **{str(j): 'Z', str(j + 1): 'Z'}) for j in range(3)]
    terms += [
        term(4, c, **{str(j): p}) for j in range(4)
        for c, p in [(-.5, 'X'), (.2, 'Z')]
    ]
    h = terms_matrix(terms)
    psi = product([
        np.array([
            np.cos(np.pi * (j + 1) / 10),
            np.exp(.5j * np.pi * j) * np.sin(np.pi * (j + 1) / 10)
        ]) for j in range(4)
    ])
    return {
        'amplitudes':
        packed([np.vdot(psi,
                        expm(-1j * t * h) @ psi) for t in [.4, .8, 1.2]]),
        'evaluated_moment_orders':
        list(range(24))
    }


def krylov_payload():
    terms = [
        term(4, .25, **{
            str(j): p,
            str(j + 1): p
        }) for j in range(3) for p in 'XYZ'
    ]
    terms += [term(4, -.1, **{str(j): 'Z'}) for j in range(4)]
    h = terms_matrix(terms)
    states = np.zeros((2, 16), complex)
    states[0, [5, 10]] = np.array([1, 1j]) / np.sqrt(2)
    states[1, [1, 2, 4, 8]] = np.array([1, 1j, 1, -1j]) / 2
    estimates = []
    for psi in states:
        row = []
        # SVD power Krylov construction differs from oracle's iterative orthogonalization.
        for dimension in [1, 2, 4, 6, 8]:
            cols = [psi]
            for _ in range(dimension - 1):
                cols.append(h @ cols[-1])
            q, singular, _ = np.linalg.svd(np.array(cols).T,
                                           full_matrices=False)
            q = q[:, singular > 1e-11]
            row.append(np.linalg.eigvalsh(q.conj().T @ h @ q)[0])
        estimates.append(row)
    return {
        'estimates': np.asarray(estimates).tolist(),
        'global_ground_energy': float(np.linalg.eigvalsh(h)[0]),
        'reachable_ground_energies': [float(row[-1]) for row in estimates]
    }


def filter_payload():
    terms = [term(4, -1., **{str(j): 'Z', str(j + 1): 'Z'}) for j in range(3)]
    terms += [
        term(4, c, **{str(j): p}) for j in range(4)
        for c, p in [(-.6, 'X'), (-.15, 'Z')]
    ]
    h = terms_matrix(terms)
    evals, vecs = np.linalg.eigh(h)
    low = vecs[:, evals < -2] @ vecs[:, evals < -2].conj().T
    psi = np.ones(16) / 4
    scaled = h / 6
    coupling = np.asarray(sqrtm(np.eye(16) - scaled @ scaled), dtype=complex)
    step = np.block([[-scaled, -coupling], [coupling, -scaled]])
    probabilities, weights = [], []
    for phases in [[.15, -.30, .45], [.20, -.50, .10, .40]]:
        state = np.r_[psi, np.zeros(16)].astype(complex)
        for index, phase in enumerate(phases):
            if index:
                state = step @ state
            state[:16] *= np.exp(2j * phase)
        good = state[:16]
        probability = np.vdot(good, good).real
        probabilities.append(probability)
        weights.append(np.vdot(good, low @ good).real / probability)
    return {
        'probabilities':
        probabilities,
        'low_energy_weights':
        weights,
        'unfiltered_weight':
        np.vdot(psi, low @ psi).real,
        'selected_filter':
        max((i for i, p in enumerate(probabilities) if p >= .1),
            key=lambda i: weights[i],
            default=None)
    }


def calibration_payload(metric='raw_l2', ket_order='q3q2q1q0'):
    psi = np.zeros(16, complex)
    indices = [0, 3, 10, 15]
    if ket_order == 'q0q1q2q3':
        indices = [int(f'{i:04b}'[::-1], 2) for i in indices]
    psi[indices] = np.array([1, 1j, 1, -1]) / 2
    vectors, errors = [], []
    for delta in [0., .05, .15]:
        h = .6 * kron_word('XXII') + .8 * kron_word(
            'IZZZ') + delta * kron_word('ZIII')
        # Independently recover the real and imaginary input pieces through the full dilation.
        scaled = h / (1.4 + delta)
        coupling = np.asarray(sqrtm(np.eye(16) - scaled @ scaled),
                              dtype=complex)
        step = np.block([[-scaled, -coupling], [coupling, -scaled]])
        recovered = []
        for input_piece in [psi.real, psi.imag]:
            raw = []
            for phases in [[1.1548959086655175], [.17, .25508841602995813]]:
                state = np.r_[input_piece, np.zeros(16)].astype(complex)
                for index, phase in enumerate(phases):
                    if index:
                        state = step @ state
                    state[:16] *= np.exp(2j * phase)
                raw.append(state[:16] * np.exp(-1j * sum(phases)))
            recovered.append(2 * (raw[0].real + 1j * raw[1].imag))
        vector = recovered[0] + 1j * recovered[1]
        exact = expm(-.63j * h) @ psi
        vectors.append(vector)
        if metric == 'raw_l2':
            error = np.linalg.norm(vector - exact)
        else:
            v = vector / np.linalg.norm(vector)
            if metric == 'normalized_l2':
                error = np.linalg.norm(v - exact)
            elif metric == 'phase_aligned_l2':
                error = np.sqrt(max(0., 2 - 2 * abs(np.vdot(exact, v))))
            else:
                error = max(0., 1 - abs(np.vdot(exact, v))**2)
                if metric in ['root_infidelity', 'trace_distance']:
                    error = np.sqrt(error)
        errors.append(float(error))
    return {
        'evolved_states': packed(vectors),
        'ket_order': ket_order,
        'metric': metric,
        'errors': errors,
        'below_threshold': [e < .001 for e in errors]
    }


CASES = [
    ('science-state-preparation-orbital-phases', orbital_payload),
    ('science-state-preparation-correlated-ordering', correlated_payload),
    ('science-block-encoding-heralded-observables', herald_payload),
    ('science-block-encoding-truncation-tradeoff', truncation_payload),
    ('science-qubitization-interferometric-amplitude', return_payload),
    ('science-qubitization-reference-dependent-convergence', krylov_payload),
    ('science-qsvt-low-energy-filter-choice', filter_payload),
    ('science-qsvt-calibration-robustness', calibration_payload),
]


@pytest.mark.parametrize('case_id,make_payload', CASES)
def test_independent_scientific_fixture_is_accepted(checker, case_id,
                                                    make_payload):
    checker.check(case_id, make_payload())


@pytest.mark.parametrize('case_id,make_payload', CASES)
def test_missing_executed_outputs_are_rejected(checker, case_id, make_payload):
    from runner.checkers.common import CheckFailure
    with pytest.raises(CheckFailure):
        checker.check(case_id, {'claimed_residual': 0.})


@pytest.mark.parametrize('case_id,make_payload,key',
                         [(*CASES[0], 'energy'), (*CASES[1], 'energies'),
                          (*CASES[2], 'magnetizations'),
                          (*CASES[4], 'amplitudes'), (*CASES[5], 'estimates'),
                          (*CASES[6], 'low_energy_weights'),
                          (*CASES[7], 'errors')])
def test_corrupted_requested_value_is_rejected(checker, case_id, make_payload,
                                               key):
    from runner.checkers.common import CheckFailure
    payload = make_payload()
    value = np.asarray(payload[key])
    payload[key] = (value + .03).tolist()
    with pytest.raises(CheckFailure):
        checker.check(case_id, payload)


def test_orbital_global_phase_is_allowed_but_conjugation_is_wrong(checker):
    from runner.checkers.common import CheckFailure
    payload = orbital_payload()
    state = np.asarray(payload['prepared_state']) @ np.array([1, 1j])
    payload['prepared_state'] = packed(state * np.exp(.7j))
    checker.check(CASES[0][0], payload)
    payload['prepared_state'] = packed(state.conj())
    with pytest.raises(CheckFailure):
        checker.check(CASES[0][0], payload)


def test_correlated_states_cannot_be_swapped(checker):
    from runner.checkers.common import CheckFailure
    payload = correlated_payload()
    payload['prepared_states'].reverse()
    with pytest.raises(CheckFailure):
        checker.check(CASES[1][0], payload)


def test_alternative_lcu_decomposition_uses_its_own_alpha(checker):
    from runner.checkers.common import CheckFailure
    payload = herald_payload()
    old_alpha = payload['alpha']
    payload['lcu_terms'] += [[.2, 'XXXX'], [-.2, 'XXXX']]
    payload['alpha'] += .4
    payload['probabilities'] = (np.asarray(payload['probabilities']) *
                                (old_alpha / payload['alpha'])**2).tolist()
    checker.check(CASES[2][0], payload)
    payload['alpha'] = old_alpha
    with pytest.raises(CheckFailure):
        checker.check(CASES[2][0], payload)


def test_truncation_accepts_conservative_bounds_without_eigensolve(checker):
    checker.check(CASES[3][0], truncation_payload(bounds=True))


def test_truncation_restores_offset_and_rejects_false_certification(checker):
    from runner.checkers.common import CheckFailure
    payload = truncation_payload()
    payload['candidates'][1]['ground_energy'] -= 2.4
    with pytest.raises(CheckFailure):
        checker.check(CASES[3][0], payload)
    payload = truncation_payload(bounds=True)
    payload['candidates'][-1]['certified'] = True
    with pytest.raises(CheckFailure):
        checker.check(CASES[3][0], payload)


def test_walk_budget_allows_analytic_zeroth_but_not_25_evaluated_moments(
        checker):
    from runner.checkers.common import CheckFailure
    payload = return_payload()
    payload['evaluated_moment_orders'] = list(range(1, 25))
    checker.check(CASES[4][0], payload)
    payload['evaluated_moment_orders'] = list(range(25))
    with pytest.raises(CheckFailure):
        checker.check(CASES[4][0], payload)


@pytest.mark.parametrize('metric', [
    'normalized_l2', 'phase_aligned_l2', 'infidelity', 'root_infidelity',
    'trace_distance'
])
def test_calibration_declared_state_metrics_are_supported(checker, metric):
    checker.check(CASES[7][0], calibration_payload(metric))


def test_calibration_allows_explicit_alternative_ket_order(checker):
    checker.check(CASES[7][0], calibration_payload(ket_order='q0q1q2q3'))


def test_calibration_rejects_real_input_recovery_on_complex_state(checker):
    from runner.checkers.common import CheckFailure
    payload = calibration_payload()
    payload['evolved_states'] = packed(
        np.asarray(payload['evolved_states'])[..., 0])
    with pytest.raises(CheckFailure):
        checker.check(CASES[7][0], payload)
