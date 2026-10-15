"""Physical outputs pass; corruptions fail without trusting claimed error bars."""

import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).parents[2]))
from runner.checkers import common


@pytest.fixture
def subject():
    return importlib.import_module('runner.checkers.dynamics')


def word(n, **ops):
    return ''.join(ops.get(str(q), 'I') for q in range(n))


def matrix(w):
    one = {
        'I': np.eye(2),
        'X': np.array([[0, 1], [1, 0]]),
        'Y': np.array([[0, -1j], [1j, 0]]),
        'Z': np.diag([1, -1])
    }
    out = np.ones((1, 1), complex)
    for p in w[::-1]:
        out = np.kron(out, one[p])
    return out


def make_terms(n, bonds, fields):
    pairs = []
    for i, j, letters, c in bonds:
        pairs.append((word(n, **{str(i): letters[0], str(j): letters[1]}), c))
    for i, letter, c in fields:
        pairs.append((word(n, **{str(i): letter}), c))
    return dict(pairs)


def quench_terms():
    return make_terms(6, [(j, j + 1, 'ZZ', -1) for j in range(5)],
                      [(j, p, c) for p, c in [('X', -.8), ('Z', -.15)]
                       for j in range(6)])


def transport_terms():
    return make_terms(5,
                      [(j, j + 1, p + p, .6) for j in range(4) for p in 'XY'],
                      [(j, 'Z', c)
                       for j, c in enumerate([.2, -.1, .3, -.2, .1])])


def dense(terms):
    return sum(c * matrix(w) for w, c in terms.items())


def pf_state(terms, order, steps, time, state, weights=(1, )):
    """Test oracle: form each dense exponential once, multiply step unitary."""
    dim = len(state)
    step = np.eye(dim, dtype=complex)
    sequence = list(order) + list(reversed(order))
    for weight in weights:
        for w in sequence:
            step = expm(
                -.5j * time / steps * weight * terms[w] * matrix(w)) @ step
    return np.linalg.matrix_power(step, steps) @ state


def quench_payload(reverse=False, weights=None):
    terms = quench_terms()
    order = list(terms)[::-1] if reverse else list(terms)
    if weights is None:
        a = 1 / (2 - 2**(1 / 3))
        weights = [a, 1 - 2 * a, a]
    initial = np.ones(64) / 8
    magnetization = sum(matrix(word(6, **{str(j): 'Z'})) for j in range(6)) / 6
    exact = expm(-.8j * dense(terms)) @ initial
    reference = float(np.vdot(exact, magnetization @ exact).real)
    rows = []
    for formula in [2, 4]:
        for steps in [1, 2, 4, 8, 16, 32]:
            state = pf_state(terms, order, steps, .8, initial,
                             weights if formula == 4 else [1])
            value = float(np.vdot(state, magnetization @ state).real)
            rows.append(
                dict(order=formula,
                     steps=steps,
                     magnetization=value,
                     two_qubit_gates=20 * steps *
                     (len(weights) if formula == 4 else 1)))
    feasible = [
        row for row in rows if abs(row['magnetization'] - reference) < 1e-3
    ]
    winner = min(feasible, key=lambda row: row['two_qubit_gates'])
    return dict(term_order=order,
                fourth_order_weights=weights,
                cost_model='unmerged CNOT ladder: two CNOTs per ZZ rotation',
                reference_magnetization=reference,
                candidates=rows,
                selected={key: winner[key]
                          for key in ['order', 'steps']})


def transport_payload(reverse_kets=False):
    terms = transport_terms()
    bond = list(terms)
    global_order = sorted(terms,
                          key=lambda w: 0 if 'X' in w else 1
                          if 'Y' in w else 2)
    initial_basis = [24, 6] if reverse_kets else [3, 12]
    initial = np.zeros(32, complex)
    initial[initial_basis] = [1 / np.sqrt(2), 1j / np.sqrt(2)]
    exact = expm(-1.2j * dense(terms)) @ initial
    center = np.array([(s >> 2) & 1 for s in range(32)])
    outside = np.array([s.bit_count() != 2 for s in range(32)])
    rows = []
    for name, order in [('bond', bond), ('xx_then_yy', global_order)]:
        for steps in [2, 4, 8, 16]:
            state = pf_state(terms, order, steps, 1.2, initial)
            probs = abs(state)**2
            rows.append(
                dict(grouping=name,
                     steps=steps,
                     leakage=float(probs[outside].sum()),
                     occupation=float(probs @ center)))
    return dict(initial_basis=initial_basis,
                term_orders={
                    'bond': bond,
                    'xx_then_yy': global_order
                },
                candidates=rows,
                reference_occupation=float(abs(exact)**2 @ center))


@pytest.mark.parametrize('reverse', [False, True])
def test_quench_accepts_different_valid_orderings(subject, reverse):
    subject.check('science-trotter-quench-cost', quench_payload(reverse))


def test_quench_mean_sz_uses_declared_units_for_predictions_and_selection(
        subject):
    payload = quench_payload()
    alternative = quench_payload(reverse=True)
    row_index = next(i for i, row in enumerate(payload['candidates'])
                     if (row['order'], row['steps']) == (4, 1))
    payload['candidates'][row_index] = alternative['candidates'][row_index]
    payload['candidates'][row_index]['term_order'] = alternative['term_order']
    # This valid mixed-order comparison changes the cheapest feasible setting
    # when the absolute error target is interpreted in mean-Sz units.
    eligible_z = [
        row for row in payload['candidates']
        if abs(row['magnetization'] -
               payload['reference_magnetization']) < 1e-3
    ]
    winner_z = min(eligible_z, key=lambda row: row['two_qubit_gates'])
    payload['magnetization_convention'] = 'mean_sz'
    payload['reference_magnetization'] /= 2
    for row in payload['candidates']:
        row['magnetization'] /= 2
    eligible = [
        row for row in payload['candidates']
        if abs(row['magnetization'] -
               payload['reference_magnetization']) < 1e-3
    ]
    winner = min(eligible, key=lambda row: row['two_qubit_gates'])
    payload['selected'] = {key: winner[key] for key in ('order', 'steps')}
    assert payload['selected'] != {
        key: winner_z[key]
        for key in ('order', 'steps')
    }
    subject.check('science-trotter-quench-cost', payload)
    payload['selected'] = {key: winner_z[key] for key in ('order', 'steps')}
    with pytest.raises(common.CheckFailure):
        subject.check('science-trotter-quench-cost', payload)


@pytest.mark.parametrize('convention', ['total_z', 'mean_x', None, []])
def test_quench_rejects_invalid_magnetization_convention(subject, convention):
    payload = quench_payload()
    payload['magnetization_convention'] = convention
    with pytest.raises(common.CheckFailure):
        subject.check('science-trotter-quench-cost', payload)


def test_quench_accepts_five_stage_suzuki(subject):
    a = 1 / (4 - 4**(1 / 3))
    payload = quench_payload(weights=[a, a, 1 - 4 * a, a, a])
    subject.check('science-trotter-quench-cost', payload)


def test_quench_accepts_per_candidate_order_and_composition(subject):
    payload = quench_payload()
    a = 1 / (4 - 4**(1 / 3))
    alternative = quench_payload(reverse=True, weights=[a, a, 1 - 4 * a, a, a])
    for index, row in enumerate(payload['candidates']):
        # Leave some rows on the global defaults; use independently evaluated
        # different circuits for both second- and fourth-order candidates.
        if row['steps'] in (1, 4, 16):
            replacement = copy.deepcopy(alternative['candidates'][index])
            replacement['term_order'] = alternative['term_order']
            if row['order'] == 4:
                replacement['fourth_order_weights'] = alternative[
                    'fourth_order_weights']
            payload['candidates'][index] = replacement
    eligible = [
        row for row in payload['candidates']
        if abs(row['magnetization'] -
               payload['reference_magnetization']) < 1e-3
    ]
    selected = min(eligible, key=lambda row: row['two_qubit_gates'])
    payload['selected'] = {key: selected[key] for key in ('order', 'steps')}
    subject.check('science-trotter-quench-cost', payload)


@pytest.mark.parametrize('override', ['term_order', 'fourth_order_weights'])
def test_quench_validates_per_candidate_overrides(subject, override):
    payload = quench_payload()
    row = next(row for row in payload['candidates'] if row['order'] == 4)
    row[override] = [] if override == 'term_order' else [1 / 3] * 3
    with pytest.raises(common.CheckFailure):
        subject.check('science-trotter-quench-cost', payload)


@pytest.mark.parametrize('change',
                         ['output', 'winner', 'weights', 'missing', 'cost'])
def test_quench_rejects_corruptions(subject, change):
    payload = quench_payload()
    if change == 'output':
        payload['candidates'][0]['magnetization'] += .1
    elif change == 'winner':
        payload['selected'] = {'order': 2, 'steps': 32}
    elif change == 'weights':
        payload['fourth_order_weights'] = [1 / 3] * 3
    elif change == 'missing':
        payload['candidates'].pop()
    else:
        payload['candidates'][0]['two_qubit_gates'] = -1
    with pytest.raises(common.CheckFailure):
        subject.check('science-trotter-quench-cost', payload)


@pytest.mark.parametrize('reverse', [False, True])
def test_conservation_accepts_both_declared_ket_conventions(subject, reverse):
    subject.check('science-trotter-conservation-versus-accuracy',
                  transport_payload(reverse))


def test_conservation_rejects_mislabeled_grouping(subject):
    payload = transport_payload()
    payload['term_orders']['bond'] = payload['term_orders']['xx_then_yy']
    with pytest.raises(common.CheckFailure):
        subject.check('science-trotter-conservation-versus-accuracy', payload)


def test_conservation_rejects_mean_number_substitution(subject):
    payload = transport_payload()
    payload['candidates'][4]['leakage'] = 0
    with pytest.raises(common.CheckFailure):
        subject.check('science-trotter-conservation-versus-accuracy', payload)


def test_conservation_rejects_accurate_reference_instead_of_circuit(subject):
    payload = transport_payload()
    payload['candidates'][0]['occupation'] = payload['reference_occupation']
    with pytest.raises(common.CheckFailure):
        subject.check('science-trotter-conservation-versus-accuracy', payload)


def ring_reference(phi=np.pi / 3):
    basis = [s for s in range(16) if s.bit_count() == 2]
    h = np.zeros((6, 6), complex)
    current = h.copy()
    potentials = [.2, -.1, .15, -.25]
    for col, s in enumerate(basis):
        h[col, col] = sum(potentials[j] * ((s >> j) & 1) + .8 *
                          ((s >> j) & 1) * ((s >> ((j + 1) % 4)) & 1)
                          for j in range(4))
        for j in range(4):
            k = (j + 1) % 4
            if (s >> j) & 1 and not (s >> k) & 1:
                t = s ^ (1 << j)
                sign = (-1)**((s & ((1 << j) - 1)).bit_count() +
                              (t & ((1 << k) - 1)).bit_count())
                row = basis.index(t | (1 << k))
                hopping = -np.exp(1j * phi / 4) * sign
                h[row, col] += hopping
                h[col, row] += hopping.conjugate()
                current[row, col] += -1j * hopping / 4
                current[col, row] += (-1j * hopping / 4).conjugate()
    values, vectors = np.linalg.eigh(h)
    ground = vectors[:, 0]
    return float(values[0]), float(np.vdot(ground, current @ ground).real)


def test_persistent_current_matches_finite_difference_and_rejects_sign(
        subject):
    energy, current = ring_reference()
    derivative = -(ring_reference(np.pi / 3 + 1e-5)[0] -
                   ring_reference(np.pi / 3 - 1e-5)[0]) / 2e-5
    assert abs(current - derivative) < 1e-9
    payload = {
        mapping: dict(energy=energy, current=current)
        for mapping in ['jw', 'bk']
    }
    subject.check('science-fermion-transforms-persistent-current', payload)
    payload['bk']['current'] *= -1
    with pytest.raises(common.CheckFailure):
        subject.check('science-fermion-transforms-persistent-current', payload)


def hubbard_reference():
    # Construct only the Nup=Ndown=2 sector directly from Fock transitions.
    basis = [
        s for s in range(256)
        if sum((s >> q) & 1 for q in [0, 2, 4, 6]) == 2 and sum(
            (s >> q) & 1 for q in [1, 3, 5, 7]) == 2
    ]
    h = np.zeros((36, 36), complex)
    charge = np.array([[((s >> (2 * j)) & 1) + ((s >> (2 * j + 1)) & 1)
                        for j in range(4)] for s in basis])
    double = np.array([[((s >> (2 * j)) & 1) * ((s >> (2 * j + 1)) & 1)
                        for j in range(4)] for s in basis])
    h[np.diag_indices(36)] = charge @ [.3, 0, -.2, .1] + 3 * double.sum(axis=1)
    for col, s in enumerate(basis):
        for j, hopping in enumerate([1, .7, 1.2]):
            for spin in [0, 1]:
                p, q = 2 * j + spin, 2 * (j + 1) + spin
                if (s >> p) & 1 and not (s >> q) & 1:
                    t = s ^ (1 << p)
                    sign = (-1)**((s & ((1 << p) - 1)).bit_count() +
                                  (t & ((1 << q) - 1)).bit_count())
                    row = basis.index(t | (1 << q))
                    h[row, col] = h[col, row] = -hopping * sign
    initial = np.zeros(36)
    initial[basis.index(51)] = 1
    probs = abs(expm(-1.2j * h) @ initial)**2
    return {
        'charge': (probs @ charge).tolist(),
        'double_occupancy': (probs @ double).tolist()
    }


def test_hubbard_checks_all_eight_outputs_in_both_encodings(subject):
    results = hubbard_reference()
    assert abs(sum(results['charge']) - 4) < 1e-12
    payload = {'jw': copy.deepcopy(results), 'bk': copy.deepcopy(results)}
    subject.check('science-fermion-transforms-hubbard-transport', payload)
    payload['bk']['double_occupancy'][3] += 1e-3
    with pytest.raises(common.CheckFailure):
        subject.check('science-fermion-transforms-hubbard-transport', payload)


def cooling_payload():
    terms = make_terms(4, [(0, 1, 'ZZ', 1), (1, 2, 'ZZ', 1), (2, 3, 'ZZ', 1),
                           (3, 0, 'ZZ', -1)],
                       [(q, 'X', -.7) for q in range(4)] + [(0, 'Z', .2)])
    h = dense(terms)
    initial = np.ones(16) / 4
    p, e = [], []
    for beta in [.05, .1, .2]:
        branch = (initial - beta * h @ initial) / (1 + 7 * beta)
        p.append(float(np.vdot(branch, branch).real))
        e.append(float(np.vdot(branch, h @ branch).real / p[-1]))
    selected = max((i for i in range(3) if e[i] <= -3.3), key=lambda i: p[i])
    return {
        'probabilities': p,
        'energies': e,
        'selected_beta': [.05, .1, .2][selected]
    }


def test_cooling_yield_selection_and_missing_normalization(subject):
    payload = cooling_payload()
    subject.check('science-simulation-probabilistic-cooling', payload)
    payload['probabilities'][1] *= 1.7**2
    with pytest.raises(common.CheckFailure):
        subject.check('science-simulation-probabilistic-cooling', payload)


def test_postselection_channel_reference_and_wrong_purity(subject):
    initial = np.array([(-1)**sum(((s >> j) & 1) * ((s >> (j + 1)) & 1)
                                  for j in range(3)) for s in range(16)]) / 4
    terms = make_terms(4, [(0, 1, 'XX', .7), (1, 2, 'ZZ', .4),
                           (2, 3, 'YX', -.2)], [])
    observable = matrix('XZII')
    branch = dense(terms) @ initial / 1.3
    probability = float(np.vdot(branch, branch).real)
    rho = sum(
        abs(c) / 1.3 *
        np.outer(matrix(w) @ initial, (matrix(w) @ initial).conj())
        for w, c in terms.items())
    payload = dict(
        heralding_probability=probability,
        conditional_expectation=float(
            np.vdot(branch, observable @ branch).real / probability),
        discarded_expectation=float(np.trace(rho @ observable).real),
        discarded_purity=float(np.trace(rho @ rho).real))
    subject.check('science-simulation-postselection-versus-discard', payload)
    payload['discarded_purity'] = 1
    with pytest.raises(common.CheckFailure):
        subject.check('science-simulation-postselection-versus-discard',
                      payload)


def domain_payload():
    initial = np.zeros(64)
    initial[56] = 1
    imbalance = np.array([
        sum((1 if j < 3 else -1) * (1 - 2 * ((s >> j) & 1))
            for j in range(6)) / 6 for s in range(64)
    ])
    correlation = np.array([(1 - 2 * ((s >> 2) & 1)) * (1 - 2 * ((s >> 3) & 1))
                            for s in range(64)])
    values = []
    for j2 in [0, .4]:
        terms = make_terms(6, [(j, j + distance, p + p, c / 4)
                               for distance, c in [(1, 1), (2, j2)]
                               for j in range(6 - distance)
                               for p in 'XYZ'], [])
        rows = []
        for t in [0, .5, 1, 2, 3]:
            probs = abs(expm(-1j * t * dense(terms)) @ initial)**2
            rows.append([float(probs @ imbalance), float(probs @ correlation)])
        values.append(rows)
    return {
        'imbalance': np.array(values)[:, :, 0].tolist(),
        'central_correlation': np.array(values)[:, :, 1].tolist()
    }


def test_domain_wall_both_grids_and_orientation(subject):
    payload = domain_payload()
    subject.check('science-workflow-frustrated-domain-wall', payload)
    payload['imbalance'][0][0] = -1
    with pytest.raises(common.CheckFailure):
        subject.check('science-workflow-frustrated-domain-wall', payload)


@pytest.mark.parametrize('case_id,payload', [
    ('science-simulation-probabilistic-cooling', {
        'probabilities': [float('nan')] * 3,
        'energies': [-4] * 3,
        'selected_beta': .1
    }),
    ('science-workflow-frustrated-domain-wall', {
        'maximum_error': 0
    }),
    ('science-fermion-transforms-persistent-current', {
        'jw': {
            'energy': -2,
            'current': 0
        },
        'bk': {
            'energy': -2,
            'current': 0
        }
    }),
])
def test_claimed_accuracy_and_nonfinite_outputs_do_not_pass(
        subject, case_id, payload):
    with pytest.raises(common.CheckFailure):
        subject.check(case_id, payload)
