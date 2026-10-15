"""Private dynamics oracles, built from NumPy/SciPy without library imports.

The numerical check intentionally does not certify which library operations
produced an artifact. The original rubric judge assesses that evidence and
resource-counting assumptions. Candidate Pauli orderings are supplied as data,
then checked and evaluated independently; no preferred ordering is prescribed.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import expm

from .common import (annihilators, close, pauli, real_array, require, sector)

_ORDER_FORMAT = (
    'A list of all nonidentity Hamiltonian Pauli words, each exactly once, '
    'in the forward half-step order; word character j acts on site/qubit j. '
    'Commuting groups may be flattened in any internal order.')

CASES = {
    'science-trotter-quench-cost': {
        'output': {
            'term_order':
            'Default for candidates without their own term_order. ' +
            _ORDER_FORMAT,
            'fourth_order_weights':
            'Default real palindromic composition weights for S4(dt)=S2(w1*dt)...S2(wm*dt), sum(w)=1 and sum(w^3)=0; report the weights actually used. Fourth-order candidates may override these individually.',
            'cost_model':
            'Nonempty text describing the common two-qubit decomposition/counting assumptions, including any cancellations.',
            'magnetization_convention':
            'Optional mean_z (default), meaning sum_j <Z_j>/6, or mean_sz, meaning sum_j <Z_j>/12 because Sz=Z/2. Use this same convention for the reference, every candidate, and the absolute 1e-3 accuracy target.',
            'reference_magnetization':
            'Exact/reference mean longitudinal magnetization at time 0.8 in the declared magnetization_convention, real scalar.',
            'candidates':
            'Twelve objects, one per order in [2,4] and steps in [1,2,4,8,16,32], containing order, steps, magnetization (mean longitudinal), and two_qubit_gates (nonnegative integer). Each may include term_order overriding the global default. Fourth-order rows may also include fourth_order_weights overriding that default, under the same validity requirements.',
            'selected':
            'Object {order,steps} selecting any minimum-cost candidate whose absolute magnetization error is <1e-3; null only if none qualify.',
        },
        'oracle':
        'Independent Pauli action evaluates the declared symmetric second-order and valid fourth-order composition circuits for every candidate; dense exact quench checks reference and feasibility in the declared mean-Z or mean-Sz units. Counts are checked for valid integer shape and minimum selection only; the original rubric separately validates the declared gate-count model and library-produced construction.',
        'tolerances': {
            'predictions_absolute': 1e-7,
            'reference_absolute': 1e-8,
            'feasibility_strict_absolute': 1e-3,
            'weight_identities': 1e-10
        },
    },
    'science-trotter-conservation-versus-accuracy': {
        'output': {
            'initial_basis':
            'Two integer computational-basis indices for the two written kets, either [3,12] (rightmost ket digit is site 0) or [24,6] (leftmost is site 0); amplitudes are [1,i]/sqrt(2).',
            'term_orders':
            'Object with keys bond and xx_then_yy, each ' + _ORDER_FORMAT +
            ' In bond, each bond XX and YY must be adjacent; in xx_then_yy, all XX precede all YY. Fields may be placed as declared.',
            'reference_occupation':
            'Exact/reference <(I-Z2)/2> at time 1.2, real scalar.',
            'candidates':
            'Eight objects for grouping in [bond,xx_then_yy] and steps in [2,4,8,16], containing grouping, steps, occupation, leakage (total probability outside the two-excitation sector).',
        },
        'oracle':
        'Independent Pauli actions evaluate each declared symmetric product formula, central occupation, and full probability outside N=2. Dense exponential establishes the convention-matched physical reference; original rubric assesses conclusions and library construction.',
        'tolerances': {
            'predictions_absolute': 1e-7,
            'reference_absolute': 1e-8
        },
    },
    'science-fermion-transforms-persistent-current': {
        'output': {
            'jw':
            'Object {energy: real scalar, current: real scalar} for physical N=2 ground state; current is -<dH/dphi>.',
            'bk': 'Same fields for Bravyi-Kitaev physical N=2 ground state.',
        },
        'oracle':
        'Independent Fock-space annihilation matrices construct the periodic complex-hopping Hamiltonian and its analytic flux derivative in N=2; the lowest eigenvector fixes current. Both representations and their mutual agreement are checked; original rubric assesses actual encoding construction and independent evidence.',
        'tolerances': {
            'energy_absolute': 1e-8,
            'current_absolute': 1e-8,
            'representation_agreement_absolute': 1e-8
        },
    },
    'science-fermion-transforms-hubbard-transport': {
        'output': {
            'jw':
            'Object {charge: real[4], double_occupancy: real[4]}, ordered physical sites 0,1,2,3 at time 1.2.',
            'bk': 'Same two arrays for the Bravyi-Kitaev evolution.',
        },
        'oracle':
        'Independent Fock-space hopping operators and diagonal interactions in the Nup=Ndown=2 sector generate exact evolution from doubly occupied sites 0 and 2. All site observables and mutual representation agreement are checked; original rubric assesses the encoding construction and approximation evidence.',
        'tolerances': {
            'observables_absolute': 1e-6,
            'representation_agreement_absolute': 1e-6
        },
    },
    'science-simulation-probabilistic-cooling': {
        'output': {
            'probabilities':
            'Real[3] successful yields, beta ordered [0.05,0.10,0.20].',
            'energies':
            'Real[3] conditional mean energies in the same beta order.',
            'selected_beta':
            'Real scalar beta with greatest yield among energies <=-3.3, or null if no candidate qualifies.',
        },
        'oracle':
        'Exact Pauli matrices directly apply each prescribed (I-beta H)/(1+7 beta) to the plus state and independently determine squared branch norms, conditional energies, and maximum-yield feasible choice. Original rubric assesses library block-encoding construction.',
        'tolerances': {
            'probability_strict_absolute': 1e-6,
            'energy_absolute': 1e-6
        },
    },
    'science-simulation-postselection-versus-discard': {
        'output': {
            'conditional_expectation':
            'Real scalar <X0 Z1> conditioned on all-zero ancillas.',
            'discarded_expectation':
            'Real scalar <X0 Z1> after discarding ancillas.',
            'heralding_probability':
            'Real scalar all-zero ancilla probability.',
            'discarded_purity':
            'Real scalar Tr(rho_system^2) after discarding ancillas.',
        },
        'oracle':
        'Independent explicit cluster amplitudes and Pauli matrices give the successful H/1.3 branch; the full Pauli-LCU discarded channel is sum_j |c_j|/1.3 P_j rho P_j. All four physical quantities are checked; original rubric assesses use of the actual library dilation.',
        'tolerances': {
            'all_observables_absolute': 1e-6
        },
    },
    'science-workflow-frustrated-domain-wall': {
        'output': {
            'imbalance':
            'Real[2,5], rows J2=[0,0.4], columns times=[0,0.5,1,2,3], using the question\'s normalized imbalance.',
            'central_correlation':
            'Real[2,5] <Z2 Z3>, same coupling/time order.',
        },
        'oracle':
        'Independent Heisenberg Pauli matrices with the 1/4 spin factor and exact dense evolution from sites 0-2 up and 3-5 down check every observable/time/coupling. Original rubric assesses the quantum circuit approach, library contribution, and finite-time interpretation.',
        'tolerances': {
            'observables_strict_absolute': 1e-3
        },
    },
}


def _scalar(payload, key):
    return float(real_array(payload, key, ()))


def _word(n, operators):
    return ''.join(operators.get(q, 'I') for q in range(n))


def _matrix(terms):
    n = len(next(iter(terms)))
    return pauli(n, [(c, {
        j: p
        for j, p in enumerate(w) if p != 'I'
    }) for w, c in terms.items()])


def _quench_terms():
    terms = {_word(6, {j: 'Z', j + 1: 'Z'}): -1.0 for j in range(5)}
    terms.update({
        _word(6, {j: p}): c
        for p, c in [('X', -.8), ('Z', -.15)]
        for j in range(6)
    })
    return terms


def _transport_terms():
    terms = {_word(5, {j: p, j + 1: p}): .6 for j in range(4) for p in 'XY'}
    terms.update({
        _word(5, {j: 'Z'}): c
        for j, c in enumerate([.2, -.1, .3, -.2, .1])
    })
    return terms


def _order(value, terms, label):
    require(
        isinstance(value, list) and all(isinstance(w, str) for w in value),
        f'{label}: expected a list of Pauli words')
    require(
        len(value) == len(terms) and set(value) == set(terms),
        f'{label}: each Hamiltonian term must occur exactly once')
    return value


@lru_cache(maxsize=128)
def _pauli_permutation(word):
    # Columns map |s> -> phase[s] |s XOR flip>; invert that permutation
    # so phase[source] * state[source] is the result in output order.
    n = len(word)
    flip = sum(1 << j for j, p in enumerate(word) if p in 'XY')
    phase = np.ones(2**n, dtype=complex)
    states = np.arange(2**n)
    for j, p in enumerate(word):
        if p in 'YZ':
            phase *= 1 - 2 * ((states >> j) & 1)
        if p == 'Y':
            phase *= 1j
    source = states ^ flip
    return source, phase[source]


def _product_state(terms, order, weights, steps, time, initial):
    state = initial.astype(complex).copy()
    rotations = []
    for weight in weights:
        for word in order + list(reversed(order)):
            angle = .5 * time / steps * weight * terms[word]
            source, phase = _pauli_permutation(word)
            rotations.append(
                (np.cos(angle), -1j * np.sin(angle), source, phase))
    for _ in range(steps):
        for cosine, sine, source, phase in rotations:
            state = cosine * state + sine * phase * state[source]
    return state


def _rows(payload, expected, keys):
    rows = payload.get('candidates')
    require(
        isinstance(rows, list) and len(rows) == len(expected),
        f'candidates: expected {len(expected)} rows')
    indexed = {}
    for row in rows:
        require(isinstance(row, dict), 'each candidate must be an object')
        key = tuple(row.get(name) for name in keys)
        require(
            all(
                isinstance(value, (str, int)) and not isinstance(value, bool)
                for value in key),
            'candidate identifiers must be integers or strings')
        require(
            key in expected and key not in indexed,
            f'candidate identifiers are missing, duplicated or unsupported: {key}'
        )
        indexed[key] = row
    require(set(indexed) == set(expected), 'candidate grid is incomplete')
    return indexed


def _weights(payload):
    raw = payload.get('fourth_order_weights')
    require(
        isinstance(raw, list) and len(raw) >= 1,
        'fourth_order_weights: expected a nonempty composition')
    weights = real_array(payload, 'fourth_order_weights', (len(raw), ))
    close(weights, weights[::-1], 1e-10, 'composition symmetry')
    close(np.asarray(weights.sum()), np.asarray(1.), 1e-10, 'composition time')
    close(np.asarray(np.sum(weights**3)), np.asarray(0.), 1e-10,
          'fourth-order cancellation')
    return weights.tolist()


def _check_quench(payload):
    terms = _quench_terms()
    convention = payload.get('magnetization_convention', 'mean_z')
    require(
        isinstance(convention, str) and convention in ('mean_z', 'mean_sz'),
        'magnetization_convention: expected mean_z or mean_sz')
    order = _order(payload.get('term_order'), terms, 'term_order')
    weights = _weights(payload)
    require(
        isinstance(payload.get('cost_model'), str)
        and bool(payload['cost_model'].strip()),
        'cost_model: declare common counting assumptions')
    initial = np.ones(64) / 8
    z = np.array(
        [sum(1 - 2 * ((s >> j) & 1) for j in range(6)) / 6 for s in range(64)])
    if convention == 'mean_sz':
        z /= 2
    exact = expm(-.8j * _matrix(terms)) @ initial
    reference = float(abs(exact)**2 @ z)
    close(np.asarray(_scalar(payload, 'reference_magnetization')),
          np.asarray(reference), 1e-8, 'reference magnetization')
    expected = {(o, n) for o in [2, 4] for n in [1, 2, 4, 8, 16, 32]}
    rows = _rows(payload, expected, ['order', 'steps'])
    feasible = {}
    for (formula, steps), row in rows.items():
        candidate_order = (_order(row['term_order'], terms,
                                  'candidate term_order')
                           if 'term_order' in row else order)
        if 'fourth_order_weights' in row:
            require(
                formula == 4,
                'fourth_order_weights applies only to fourth-order candidates')
            candidate_weights = _weights(row)
        else:
            candidate_weights = weights if formula == 4 else [1.]
        state = _product_state(terms, candidate_order, candidate_weights,
                               steps, .8, initial)
        magnetization = float(abs(state)**2 @ z)
        close(np.asarray(_scalar(row, 'magnetization')),
              np.asarray(magnetization), 1e-7,
              f'order {formula}, steps {steps}: circuit magnetization')
        cost = row.get('two_qubit_gates')
        require(
            isinstance(cost, int) and not isinstance(cost, bool) and cost >= 0,
            'two_qubit_gates: expected nonnegative integer estimate')
        if abs(magnetization - reference) < 1e-3:
            feasible[(formula, steps)] = cost
    selected = payload.get('selected')
    if not feasible:
        require(selected is None,
                'no candidate meets the magnetization target')
    else:
        require(isinstance(selected, dict), 'selected: expected {order,steps}')
        choice = (selected.get('order'), selected.get('steps'))
        require(
            all(
                isinstance(x, int) and not isinstance(x, bool)
                for x in choice), 'selected: order and steps must be integers')
        require(choice in feasible,
                'selected candidate misses magnetization target')
        require(feasible[choice] == min(feasible.values()),
                'selected candidate is not a minimum-cost feasible choice')


def _check_transport(payload):
    terms = _transport_terms()
    initial_basis = real_array(payload, 'initial_basis', (2, ))
    require(
        initial_basis.tolist() in ([3., 12.], [24., 6.]),
        'initial_basis: expected one of the declared ket-to-site conventions')
    initial = np.zeros(32, complex)
    initial[initial_basis.astype(int)] = [1 / np.sqrt(2), 1j / np.sqrt(2)]
    term_orders = payload.get('term_orders')
    require(isinstance(term_orders, dict),
            'term_orders: expected both grouping policies')
    orders = {
        name: _order(term_orders.get(name), terms, name)
        for name in ['bond', 'xx_then_yy']
    }
    for j in range(4):
        xword, yword = [_word(5, {j: p, j + 1: p}) for p in 'XY']
        require(
            abs(orders['bond'].index(xword) -
                orders['bond'].index(yword)) == 1,
            'bond: corresponding XX and YY terms must be adjacent')
    ordered = orders['xx_then_yy']
    require(
        max(i for i, w in enumerate(ordered) if 'X' in w)
        < min(i for i, w in enumerate(ordered) if 'Y' in w),
        'xx_then_yy: all XX terms must precede all YY terms')
    center = np.array([(s >> 2) & 1 for s in range(32)])
    outside = np.array([s.bit_count() != 2 for s in range(32)])
    exact = expm(-1.2j * _matrix(terms)) @ initial
    reference = float(abs(exact)**2 @ center)
    close(np.asarray(_scalar(payload, 'reference_occupation')),
          np.asarray(reference), 1e-8, 'reference occupation')
    rows = _rows(payload, {(grouping, n)
                           for grouping in orders
                           for n in [2, 4, 8, 16]}, ['grouping', 'steps'])
    for (grouping, steps), row in rows.items():
        state = _product_state(terms, orders[grouping], [1.], steps, 1.2,
                               initial)
        probabilities = abs(state)**2
        close(np.asarray(_scalar(row, 'occupation')),
              np.asarray(probabilities @ center), 1e-7,
              f'{grouping}, steps {steps}: central occupation')
        close(np.asarray(_scalar(row, 'leakage')),
              np.asarray(probabilities[outside].sum()), 1e-7,
              f'{grouping}, steps {steps}: genuine sector leakage')


@lru_cache(maxsize=1)
def _persistent_reference():
    a = annihilators(4)
    numbers = [op.conj().T @ op for op in a]
    h = np.zeros((16, 16), complex)
    derivative = h.copy()
    phase = np.exp(1j * np.pi / 12)
    for j, epsilon in enumerate([.2, -.1, .15, -.25]):
        k = (j + 1) % 4
        forward = -phase * a[k].conj().T @ a[j]
        h += forward + forward.conj(
        ).T + .8 * numbers[j] @ numbers[k] + epsilon * numbers[j]
        derivative += 1j * forward / 4 + (1j * forward / 4).conj().T
    indices = sector(4, 2)
    block = np.ix_(indices, indices)
    energies, vectors = np.linalg.eigh(h[block])
    ground = vectors[:, 0]
    return float(
        energies[0]), float(-np.vdot(ground, derivative[block] @ ground).real)


def _check_persistent(payload):
    energy, current = _persistent_reference()
    outputs = []
    for mapping in ['jw', 'bk']:
        row = payload.get(mapping)
        value = np.array([_scalar(row, 'energy'), _scalar(row, 'current')])
        close(value, np.array([energy, current]), 1e-8,
              f'{mapping}: physical energy/current')
        outputs.append(value)
    close(outputs[0], outputs[1], 1e-8, 'encoding agreement')


@lru_cache(maxsize=1)
def _hubbard_reference():
    indices = sector(8, 4, (2, 2))
    a = [op[:, indices] for op in annihilators(8)]
    occupation = ((indices[:, None] >> np.arange(8)) & 1).astype(float)
    charge = occupation[:, ::2] + occupation[:, 1::2]
    double = occupation[:, ::2] * occupation[:, 1::2]
    h = np.diag(charge @ [.3, 0., -.2, .1] +
                3 * double.sum(axis=1)).astype(complex)
    for j, hopping in enumerate([1., .7, 1.2]):
        for spin in [0, 1]:
            forward = -hopping * a[2 * j + spin].conj().T @ a[2 *
                                                              (j + 1) + spin]
            h += forward + forward.conj().T
    initial = np.zeros(len(indices))
    initial[np.flatnonzero(indices == 51)[0]] = 1
    probabilities = abs(expm(-1.2j * h) @ initial)**2
    return probabilities @ charge, probabilities @ double


def _check_hubbard(payload):
    charge, double = _hubbard_reference()
    values = []
    for mapping in ['jw', 'bk']:
        row = payload.get(mapping)
        c = real_array(row, 'charge', (4, ))
        d = real_array(row, 'double_occupancy', (4, ))
        close(c, charge, 1e-6, f'{mapping}: site charges')
        close(d, double, 1e-6, f'{mapping}: double occupancies')
        values.append(np.concatenate([c, d]))
    close(values[0], values[1], 1e-6, 'encoding agreement')


@lru_cache(maxsize=1)
def _cooling_reference():
    h = pauli(4, [(1., {
        0: 'Z',
        1: 'Z'
    }), (1., {
        1: 'Z',
        2: 'Z'
    }), (1., {
        2: 'Z',
        3: 'Z'
    }), (-1., {
        3: 'Z',
        0: 'Z'
    }), (.2, {
        0: 'Z'
    })] + [(-.7, {
        j: 'X'
    }) for j in range(4)])
    initial = np.ones(16) / 4
    probabilities, energies = [], []
    for beta in [.05, .1, .2]:
        branch = (initial - beta * h @ initial) / (1 + 7 * beta)
        probability = float(np.vdot(branch, branch).real)
        probabilities.append(probability)
        energies.append(float(np.vdot(branch, h @ branch).real / probability))
    return np.array(probabilities), np.array(energies)


def _check_cooling(payload):
    probabilities, energies = _cooling_reference()
    close(real_array(payload, 'probabilities', (3, )),
          probabilities,
          1e-6,
          'successful yields',
          strict=True)
    close(real_array(payload, 'energies', (3, )), energies, 1e-6,
          'conditional energies')
    feasible = np.flatnonzero(energies <= -3.3)
    if not len(feasible):
        require(
            payload.get('selected_beta') is None,
            'no beta meets the energy threshold')
    else:
        best = max(feasible, key=lambda i: probabilities[i])
        close(np.asarray(_scalar(payload, 'selected_beta')),
              np.asarray([.05, .1, .2][best]), 1e-12,
              'maximum-yield feasible beta')


@lru_cache(maxsize=1)
def _postselection_reference():
    initial = np.array([(-1)**sum(((s >> j) & 1) * ((s >> (j + 1)) & 1)
                                  for j in range(3)) for s in range(16)]) / 4
    coefficients = [.7, .4, -.2]
    operators = [
        pauli(4, [(1., ops)]) for ops in [{
            0: 'X',
            1: 'X'
        }, {
            1: 'Z',
            2: 'Z'
        }, {
            2: 'Y',
            3: 'X'
        }]
    ]
    outputs = [operator @ initial for operator in operators]
    branch = sum(c * state for c, state in zip(coefficients, outputs)) / 1.3
    probability = float(np.vdot(branch, branch).real)
    rho = sum(
        abs(c) / 1.3 * np.outer(state, state.conj())
        for c, state in zip(coefficients, outputs))
    observable = pauli(4, [(1., {0: 'X', 1: 'Z'})])
    return {
        'conditional_expectation':
        float(np.vdot(branch, observable @ branch).real / probability),
        'discarded_expectation':
        float(np.trace(rho @ observable).real),
        'heralding_probability':
        probability,
        'discarded_purity':
        float(np.trace(rho @ rho).real),
    }


def _check_postselection(payload):
    for key, expected in _postselection_reference().items():
        close(np.asarray(_scalar(payload, key)), np.asarray(expected), 1e-6,
              key)


@lru_cache(maxsize=1)
def _domain_reference():
    initial = np.zeros(64)
    initial[56] = 1
    bits = (np.arange(64)[:, None] >> np.arange(6)) & 1
    z = 1 - 2 * bits
    imbalance = (z[:, :3].sum(axis=1) - z[:, 3:].sum(axis=1)) / 6
    correlation = z[:, 2] * z[:, 3]
    outputs = np.zeros((2, 5, 2))
    for coupling_index, coupling in enumerate([0., .4]):
        h = pauli(6, [(c / 4, {
            j: p,
            j + distance: p
        }) for distance, c in [(1, 1.), (2, coupling)]
                      for j in range(6 - distance) for p in 'XYZ'])
        eigenvalues, eigenvectors = np.linalg.eigh(h)
        coordinates = eigenvectors.conj().T @ initial
        for time_index, time in enumerate([0., .5, 1., 2., 3.]):
            state = eigenvectors @ (np.exp(-1j * time * eigenvalues) *
                                    coordinates)
            probabilities = abs(state)**2
            outputs[coupling_index, time_index] = [
                probabilities @ imbalance, probabilities @ correlation
            ]
    return outputs[:, :, 0], outputs[:, :, 1]


def _check_domain(payload):
    imbalance, correlation = _domain_reference()
    close(real_array(payload, 'imbalance', (2, 5)),
          imbalance,
          1e-3,
          'domain-wall imbalance',
          strict=True)
    close(real_array(payload, 'central_correlation', (2, 5)),
          correlation,
          1e-3,
          'central correlation',
          strict=True)


_CHECKS = {
    'science-trotter-quench-cost': _check_quench,
    'science-trotter-conservation-versus-accuracy': _check_transport,
    'science-fermion-transforms-persistent-current': _check_persistent,
    'science-fermion-transforms-hubbard-transport': _check_hubbard,
    'science-simulation-probabilistic-cooling': _check_cooling,
    'science-simulation-postselection-versus-discard': _check_postselection,
    'science-workflow-frustrated-domain-wall': _check_domain,
}


def check(case_id, payload):
    """Check genuine numerical outputs; raise CheckFailure on any violation."""
    require(case_id in _CHECKS, f'unsupported dynamics case: {case_id}')
    require(isinstance(payload, dict),
            'scientific result must be a JSON object')
    _CHECKS[case_id](payload)
