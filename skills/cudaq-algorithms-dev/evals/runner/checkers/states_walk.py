"""Private scientific checks for preparations, block encodings, walks and QSVT.

All references are NumPy/SciPy calculations.  Worker-visible ``output`` entries
specify serialization only; numerical oracles and tolerances remain private.
Method/library provenance is assessed separately against the original rubric.
"""
import itertools

import numpy as np
from scipy.linalg import expm

from .common import (CheckFailure, annihilators, close, complex_array, pauli,
                     real_array, require)

_BASIS = (
    'Serialize amplitudes in computational basis index sum_q n_q*2**q '
    '(q0 least significant); fermionic creation operators are ordered by '
    'increasing mode index. Convert other register/gauge conventions to this '
    'basis for serialization. Complex numbers are [real, imaginary].')
_LCU = (
    'Nonempty list of [real_coefficient, Pauli_word] pairs from the encoding; '
    'word character q acts on qubit q (q0 FIRST). Retain separate terms if '
    'the chosen valid LCU decomposition has duplicates.')
_TOL = 5e-7

CASES = {
    'science-state-preparation-orbital-phases': {
        'output': {
            'prepared_state':
            'Complex [64] executed or faithfully evaluated preparation state. '
            + _BASIS,
            'energy':
            'Numerical ground-state energy.',
            'coherences':
            'Complex [6], entry j is <a†_(j+1 mod 6) a_j>, with the closing bond last.'
        },
        'oracle':
        'Single-particle Hermitian eigensolve, occupied-orbital Slater minors and an independent Jordan-Wigner observable calculation.',
        'tolerances': {
            'observable_atol': _TOL,
            'infidelity_strict': 1e-8
        }
    },
    'science-state-preparation-correlated-ordering': {
        'output': {
            'prepared_states':
            'Complex [2,64] in order U2 U1|reference>, U1 U2|reference> (rightmost excitation acts first). '
            + _BASIS,
            'energies':
            'Real [2] in that order, using canonical hopping -sum(a†_(j+1,s) a_(j,s)+h.c.). Convert equivalent gauge conventions consistently.',
            'double_occupancies':
            'Real [2] total sum_j <n_(2j)n_(2j+1)> in that order.',
            'mutual_fidelity':
            'Squared normalized overlap |<psi_21|psi_12>|^2 (convert a reported root fidelity to this serialization).'
        },
        'oracle':
        'Fermionic occupation-bit operators, exact generator exponentials and canonical Hubbard observables; arbitrary overall preparation phases are immaterial.',
        'tolerances': {
            'atol': _TOL,
            'state_infidelity': 1e-8
        }
    },
    'science-block-encoding-heralded-observables': {
        'output': {
            'lcu_terms':
            _LCU,
            'alpha':
            'Positive LCU coefficient 1-norm of the declared decomposition, including the scalar offset.',
            'probabilities':
            'Real [2] successful-branch probabilities, Neel then spiral input.',
            'magnetizations':
            'Real [2] conditional staggered magnetizations, Neel then spiral input.'
        },
        'oracle':
        'Independent Pauli matrices, validation of the supplied LCU sum and its 1-norm, and direct H|psi> branch observables.',
        'tolerances': {
            'atol': _TOL
        }
    },
    'science-block-encoding-truncation-tradeoff': {
        'output': {
            'candidates':
            'List of eight objects, one for every cutoff and offset choice. Each has cutoff (0,0.0005,0.001,0.01), offset_outside (boolean), lcu_terms ('
            + _LCU +
            '), alpha (positive coefficient 1-norm), bias_kind ("observed" or "bound"), bias (nonnegative physical energy-bias magnitude or certified upper bound), certified (boolean indicating demonstrated bias <=1e-3). Observed-bias rows also contain ground_energy, with the scalar offset restored. Bounds need no eigensolve or ground_energy. Optional num_terms/num_ancilla describe the chosen Pauli LCU.',
            'recommendation':
            'Zero-based index into candidates of a configuration certified within the requested inclusive bias allowance.'
        },
        'oracle':
        'All eight independent truncated Pauli matrices; exact small dense spectra validate observed biases or upper bounds; scalar bookkeeping and supplied encoding normalization are checked.',
        'tolerances': {
            'atol': _TOL,
            'bias_allowance_inclusive': 1e-3
        }
    },
    'science-qubitization-interferometric-amplitude': {
        'output': {
            'amplitudes':
            'Complex [3] G(t), each entry [real, imaginary], in time order 0.4,0.8,1.2; retain the scalar-energy phase.',
            'evaluated_moment_orders':
            'Complete list of nonnegative integer orders of the walk moments actually evaluated, with at most 24 entries. Include order 0 if evaluated; an analytically known zeroth moment can be omitted. List every evaluation; do not silently exclude nontrivial moments.'
        },
        'oracle':
        'Independent dense Hermitian evolution of the complex product state and strict absolute complex-amplitude errors; the evaluated moment budget is checked.',
        'tolerances': {
            'amplitude_error_strict': 1e-6
        }
    },
    'science-qubitization-reference-dependent-convergence': {
        'output': {
            'estimates':
            'Real [2,5] moment-based Krylov energies: references in prompt order, dimensions 1,2,4,6,8. Use stable saturated values after rank exhaustion.',
            'global_ground_energy':
            'Numerical global ground energy of the full Hamiltonian.',
            'reachable_ground_energies':
            'Real [2] lowest energies having nonzero spectral support in each supplied reference.'
        },
        'oracle':
        'Independent reorthogonalized dense Krylov projections and dense spectral support, without a generalized eigenproblem over a singular Gram matrix.',
        'tolerances': {
            'atol': _TOL
        }
    },
    'science-qsvt-low-energy-filter-choice': {
        'output': {
            'probabilities':
            'Real [2] successful-branch probabilities for the phase lists in prompt order.',
            'low_energy_weights':
            'Real [2] conditional probabilities of the strict event E < -2 for those filters.',
            'unfiltered_weight':
            'Probability of E < -2 for the unfiltered plus-state.',
            'selected_filter':
            'Zero-based index 0 or 1 of the eligible filter with greatest conditional low-energy weight, or null if neither has success probability >=0.1. No unspecified usefulness threshold is assumed.'
        },
        'oracle':
        'Independent eigenvalues and input spectral weights with a 2x2 forward-walk QSP transfer recurrence. Known global convention phases do not affect probabilities.',
        'tolerances': {
            'atol': _TOL
        }
    },
    'science-qsvt-calibration-robustness': {
        'output': {
            'evolved_states':
            'Complex [3,16] reconstructed approximate vectors for delta=0,0.05,0.15, each entry [real, imaginary], serialized in the q0-least-significant computational basis. For raw_l2 preserve the unnormalized vector and physical phase; other metrics may normalize/phase-align only as declared by metric.',
            'ket_order':
            'Interpretation of the written input kets: "q3q2q1q0" or "q0q1q2q3". State arrays themselves always use q0-least-significant indices.',
            'metric':
            'One of raw_l2 (||v-u||), normalized_l2 (||v/||v||-u||), phase_aligned_l2 (min_phase ||e^(i phase)v/||v||-u||), infidelity (1-|<u|v/||v||>|^2), root_infidelity (sqrt(infidelity)), trace_distance (same as root_infidelity for these pure states). Here u=exp(-i H_delta*0.63)|psi> is normalized.',
            'errors':
            'Real [3] errors under the declared metric.',
            'below_threshold':
            'Boolean [3] decisions for strict error <1e-3, in perturbation order.'
        },
        'oracle':
        'Independent complex-linear closed-form response of the degree-zero cosine and degree-one sine sequences, plus dense exact propagators at every perturbation. Normalized and phase-insensitive metrics are judged according to their declared meaning.',
        'tolerances': {
            'atol': _TOL,
            'error_threshold_strict': 1e-3
        }
    },
}


def _scalar(payload, key):
    return float(real_array(payload, key, ()))


def _normed(vector, label):
    norm = np.linalg.norm(vector)
    require(norm > 0, f'{label}: zero vector')
    return vector / norm


def _fidelity(first, second):
    return min(
        1.,
        abs(np.vdot(_normed(first, 'state'), _normed(second, 'reference')))**2)


def _expectation(state, operator):
    return float(np.vdot(state, operator @ state).real)


def _product(factors):
    state = np.array([1.], dtype=complex)
    for factor in reversed(factors):
        state = np.kron(state, factor)
    return state


def _orbital(payload):
    h = np.zeros((6, 6), dtype=complex)
    for j in range(6):
        k = (j + 1) % 6
        h[k, j] = -(1. if j % 2 == 0 else .6) * np.exp(.7j / 6)
        h[j, k] = h[k, j].conjugate()
    values, orbitals = np.linalg.eigh(h)
    target = np.zeros(64, dtype=complex)
    for occupied in itertools.combinations(range(6), 3):
        target[sum(1 << q for q in occupied)] = np.linalg.det(
            orbitals[list(occupied), :3])
    prepared = complex_array(payload, 'prepared_state', (64, ))
    close(np.linalg.norm(prepared), np.array(1.), 1e-8, 'preparation norm')
    infidelity = max(0., 1 - _fidelity(prepared, target))
    require(infidelity < 1e-8,
            f'preparation infidelity {infidelity:g} is not below 1e-8')
    close(real_array(payload, 'energy', ()), values[:3].sum(), _TOL,
          'ground energy')
    a = annihilators(6)
    expected = np.array([
        np.vdot(target, a[(j + 1) % 6].conj().T @ a[j] @ target)
        for j in range(6)
    ])
    # A permitted imperfect preparation can perturb coherences by O(sqrt(infidelity)).
    close(complex_array(payload, 'coherences',
                        (6, )), expected, _TOL + 2 * np.sqrt(infidelity),
          'oriented nearest-neighbor coherences')


def _correlated(payload):
    a = annihilators(6)
    occupation = [x.conj().T @ x for x in a]
    double = sum(occupation[j] @ occupation[j + 1] for j in (0, 2, 4))
    h = 2 * double
    for j in range(4):
        hopping = a[j + 2].conj().T @ a[j]
        h = h - hopping - hopping.conj().T
    t1 = a[2].conj().T @ a[0] + a[3].conj().T @ a[1]
    t2 = a[4].conj().T @ a[5].conj().T @ a[1] @ a[0]
    u1 = expm(.28 * (t1 - t1.conj().T))
    u2 = expm(-.19 * (t2 - t2.conj().T))
    reference = np.zeros(64, complex)
    reference[3] = 1
    expected = [u2 @ u1 @ reference, u1 @ u2 @ reference]
    prepared = complex_array(payload, 'prepared_states', (2, 64))
    for i, state in enumerate(prepared):
        close(np.linalg.norm(state), np.array(1.), 1e-8, f'ordering {i} norm')
        require(1 - _fidelity(state, expected[i]) < 1e-8,
                f'ordering {i}: incorrect prepared state')
    close(real_array(payload, 'energies', (2, )),
          np.array([_expectation(s, h) for s in expected]), _TOL,
          'ordered energies')
    close(real_array(payload, 'double_occupancies', (2, )),
          np.array([_expectation(s, double) for s in expected]), _TOL,
          'ordered double occupancies')
    close(real_array(payload, 'mutual_fidelity', ()),
          np.array(_fidelity(*expected)), _TOL, 'squared mutual fidelity')


def _lcu(payload, n, expected):
    terms = payload.get('lcu_terms')
    require(
        isinstance(terms, list) and len(terms) > 0,
        'lcu_terms must be a nonempty list')
    decoded, coefficients = [], []
    for term in terms:
        require(
            isinstance(term, (list, tuple)) and len(term) == 2,
            'LCU terms are [coefficient, word] pairs')
        coefficient = _scalar({'coefficient': term[0]}, 'coefficient')
        word = term[1]
        require(
            isinstance(word, str) and len(word) == n
            and set(word) <= set('IXYZ'), 'invalid Pauli word')
        decoded.append((coefficient, {
            q: letter
            for q, letter in enumerate(word) if letter != 'I'
        }))
        coefficients.append(coefficient)
    close(pauli(n, decoded), expected, _TOL, 'declared LCU operator')
    alpha = _scalar(payload, 'alpha')
    require(alpha > 0, 'alpha must be positive')
    close(np.array(alpha), np.array(sum(abs(c) for c in coefficients)), _TOL,
          'LCU normalization')
    if 'num_terms' in payload:
        close(real_array(payload, 'num_terms', ()), np.array(len(terms)), 0,
              'LCU term count')
    if 'num_ancilla' in payload:
        close(real_array(payload, 'num_ancilla', ()),
              np.array(max(1, (len(terms) - 1).bit_length())), 0,
              'Pauli LCU ancilla count')
    return alpha


def _herald(payload):
    terms = [(.4, {})]
    for j in range(3):
        terms += [(c, {
            j: p,
            j + 1: q
        }) for c, p, q in [(.25, 'X',
                            'X'), (.25, 'Y',
                                   'Y'), (.175, 'Z',
                                          'Z'), (.075, 'X',
                                                 'Y'), (-.075, 'Y', 'X')]]
    terms += [(.1, {j: 'Z'}) for j in range(4)]
    h = pauli(4, terms)
    alpha = _lcu(payload, 4, h)
    neel = np.zeros(16, complex)
    neel[10] = 1
    spiral = _product([np.array([1, 1j**j]) / np.sqrt(2) for j in range(4)])
    magnetization = pauli(4, [((-1)**j / 4, {j: 'Z'}) for j in range(4)])
    probabilities, conditioned = [], []
    for state in (neel, spiral):
        branch = h @ state / alpha
        probability = np.vdot(branch, branch).real
        probabilities.append(probability)
        conditioned.append(_expectation(branch, magnetization) / probability)
    close(real_array(payload, 'probabilities', (2, )), np.array(probabilities),
          _TOL, 'heralding probabilities')
    close(real_array(payload, 'magnetizations', (2, )), np.array(conditioned),
          _TOL, 'conditional magnetizations')


def _truncation(payload):
    couplings = (1., .08, .004, .0006, .0001)
    terms = [(2.4, {})] + [(-couplings[j - i - 1], {
        i: 'Z',
        j: 'Z'
    }) for i in range(6) for j in range(i + 1, 6)]
    terms += [(-.7, {j: 'X'}) for j in range(6)]
    original_energy = float(np.linalg.eigvalsh(pauli(6, terms))[0])
    rows = payload.get('candidates')
    require(
        isinstance(rows, list) and len(rows) == 8,
        'candidates must cover all eight cutoff/offset combinations')
    seen, certified_rows = set(), []
    for row in rows:
        require(isinstance(row, dict), 'candidate must be an object')
        cutoff = _scalar(row, 'cutoff')
        require(cutoff in (0., .0005, .001, .01), 'unexpected cutoff')
        outside = row.get('offset_outside')
        require(isinstance(outside, bool), 'offset_outside must be boolean')
        key = cutoff, outside
        require(key not in seen, 'duplicate cutoff/offset candidate')
        seen.add(key)
        retained = [(c, ops) for c, ops in terms
                    if abs(c) >= cutoff and not (outside and not ops)]
        matrix = pauli(6, retained)
        _lcu(row, 6, matrix)
        energy = float(
            np.linalg.eigvalsh(matrix)[0] + (2.4 if outside else 0.))
        bias = _scalar(row, 'bias')
        require(bias >= 0, 'bias magnitude/bound must be nonnegative')
        exact_bias = abs(energy - original_energy)
        kind = row.get('bias_kind')
        require(kind in ('observed', 'bound'),
                'bias_kind must distinguish observation from bound')
        if kind == 'observed':
            close(real_array(row, 'ground_energy', ()), np.array(energy), _TOL,
                  'restored physical ground energy')
            close(np.array(bias), np.array(exact_bias), _TOL,
                  'physical ground-energy bias')
            certified = exact_bias <= 1e-3
        else:
            require(
                bias + 1e-10 >= exact_bias,
                'claimed bound is smaller than the independently computed bias'
            )
            certified = bias <= 1e-3
            if 'ground_energy' in row:
                close(real_array(row, 'ground_energy', ()), np.array(energy),
                      _TOL, 'restored physical ground energy')
        require(
            isinstance(row.get('certified'), bool)
            and row['certified'] == certified,
            'incorrect certification (a loose upper bound alone does not certify accuracy)'
        )
        certified_rows.append(certified)
    selection = payload.get('recommendation')
    require(
        type(selection) is int and 0 <= selection < len(rows),
        'recommendation must index candidates')
    require(certified_rows[selection],
            'recommended configuration is not certified within 1e-3')


def _return_amplitudes(payload):
    terms = [(.3, {})] + [(-.7, {j: 'Z', j + 1: 'Z'}) for j in range(3)]
    terms += [(c, {
        j: op
    }) for j in range(4) for c, op in ((-.5, 'X'), (.2, 'Z'))]
    h = pauli(4, terms)
    state = _product([
        np.array([
            np.cos(np.pi * (j + 1) / 10),
            np.exp(.5j * np.pi * j) * np.sin(np.pi * (j + 1) / 10)
        ]) for j in range(4)
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    weights = abs(eigenvectors.conj().T @ state)**2
    expected = np.array([
        np.sum(weights * np.exp(-1j * t * eigenvalues)) for t in (.4, .8, 1.2)
    ])
    close(complex_array(payload, 'amplitudes', (3, )),
          expected,
          1e-6,
          'complex return amplitudes',
          strict=True)
    orders = payload.get('evaluated_moment_orders')
    require(
        isinstance(orders, list) and 0 < len(orders) <= 24,
        'complete evaluated-moment list must contain 1 to 24 entries')
    require(all(type(order) is int and order >= 0 for order in orders),
            'moment orders must be nonnegative integers')


def _krylov_energies(h, state):
    basis, energies = [], []
    candidate = state.copy()
    for dimension in range(1, 9):
        for _ in range(2):
            for vector in basis:
                candidate -= vector * np.vdot(vector, candidate)
        norm = np.linalg.norm(candidate)
        if norm > 1e-11:
            basis.append(candidate / norm)
        q = np.array(basis).T
        energies.append(float(np.linalg.eigvalsh(q.conj().T @ h @ q)[0]))
        candidate = h @ basis[-1]
    return np.array([energies[d - 1] for d in (1, 2, 4, 6, 8)])


def _convergence(payload):
    terms = [(.25, {j: op, j + 1: op}) for j in range(3) for op in 'XYZ']
    terms += [(-.1, {j: 'Z'}) for j in range(4)]
    h = pauli(4, terms)
    states = np.zeros((2, 16), dtype=complex)
    states[0, [5, 10]] = np.array([1, 1j]) / np.sqrt(2)
    states[1, [1, 2, 4, 8]] = np.array([1, 1j, 1, -1j]) / 2
    expected = np.array([_krylov_energies(h, state) for state in states])
    values, vectors = np.linalg.eigh(h)
    reachable = np.array([
        np.min(values[abs(vectors.conj().T @ state)**2 > 1e-12])
        for state in states
    ])
    close(real_array(payload, 'estimates', (2, 5)), expected, _TOL,
          'Krylov energies with rank saturation')
    close(real_array(payload, 'global_ground_energy', ()), values[0], _TOL,
          'global ground energy')
    close(real_array(payload, 'reachable_ground_energies', (2, )), reachable,
          _TOL, 'reference-supported ground energies')


def _qsp_response(phases, x):
    # Reflection is diag(-1, 1), hence the forward step's upper-left element is -x.
    s = np.sqrt(max(0., 1 - x * x))
    step = np.array([[-x, -s], [s, -x]], complex)
    vector = np.array([np.exp(1j * phases[0]), 0.], complex)
    for phase in phases[1:]:
        vector = np.diag([np.exp(1j * phase),
                          np.exp(-1j * phase)]) @ step @ vector
    return vector[0]


def _filters(payload):
    terms = [(-1., {j: 'Z', j + 1: 'Z'}) for j in range(3)]
    terms += [(c, {
        j: op
    }) for j in range(4) for c, op in ((-.6, 'X'), (-.15, 'Z'))]
    values, vectors = np.linalg.eigh(pauli(4, terms))
    weights = abs(vectors.conj().T @ (np.ones(16) / 4))**2
    low = values < -2
    probabilities, conditioned = [], []
    for phases in ((.15, -.30, .45), (.20, -.50, .10, .40)):
        response = np.array([_qsp_response(phases, x) for x in values / 6])
        filtered = weights * abs(response)**2
        probability = float(filtered.sum())
        probabilities.append(probability)
        conditioned.append(float(filtered[low].sum() / probability))
    close(real_array(payload, 'probabilities', (2, )), np.array(probabilities),
          _TOL, 'filter success probabilities')
    close(real_array(payload, 'low_energy_weights', (2, )),
          np.array(conditioned), _TOL, 'conditional low-energy weights')
    close(real_array(payload, 'unfiltered_weight', ()),
          np.array(weights[low].sum()), _TOL, 'unfiltered low-energy weight')
    eligible = [
        i for i, probability in enumerate(probabilities) if probability >= .1
    ]
    selected = payload.get('selected_filter')
    require('selected_filter' in payload, 'missing selected_filter')
    if not eligible:
        require(selected is None,
                'no filter satisfies the requested success probability')
    else:
        require(
            type(selected) is int and selected in eligible,
            'selected filter is not eligible')
        require(
            conditioned[selected]
            >= max(conditioned[i] for i in eligible) - _TOL,
            'selected filter does not maximize conditional weight')


def _state_error(approximate, exact, metric):
    if metric == 'raw_l2':
        return float(np.linalg.norm(approximate - exact))
    approximate = _normed(approximate, 'evolved state')
    if metric == 'normalized_l2':
        return float(np.linalg.norm(approximate - exact))
    overlap = min(1., abs(np.vdot(exact, approximate)))
    if metric == 'phase_aligned_l2':
        return float(np.sqrt(max(0., 2 - 2 * overlap)))
    infidelity = max(0., 1 - overlap**2)
    return float(infidelity if metric == 'infidelity' else np.sqrt(infidelity))


def _calibration(payload):
    metric = payload.get('metric')
    require(
        metric in ('raw_l2', 'normalized_l2', 'phase_aligned_l2', 'infidelity',
                   'root_infidelity', 'trace_distance'),
        'unsupported or missing error metric')
    ket_order = payload.get('ket_order')
    require(ket_order in ('q3q2q1q0', 'q0q1q2q3'),
            'missing or unsupported input ket order')
    indices = [0, 3, 10, 15]
    if ket_order == 'q0q1q2q3':
        indices = [int(f'{index:04b}'[::-1], 2) for index in indices]
    state = np.zeros(16, complex)
    state[indices] = np.array([1, 1j, 1, -1]) / 2
    submitted = complex_array(payload, 'evolved_states', (3, 16))
    expected_errors = []
    for row, delta in enumerate((0., .05, .15)):
        h = pauli(4, [(.6, {
            0: 'X',
            1: 'X'
        }), (.8, {
            1: 'Z',
            2: 'Z',
            3: 'Z'
        }), (delta, {
            0: 'Z'
        })])
        # Degree-0/1 responses imply this complex-linear operator, with the
        # circuit's known QSP global phases removed before component recovery.
        approximate = (2 * np.cos(1.1548959086655175) * state -
                       2j * np.sin(.17 + .25508841602995813) * (h @ state) /
                       (1.4 + delta))
        actual = submitted[row]
        if metric == 'raw_l2':
            close(actual, approximate, _TOL,
                  f'delta={delta}: raw complex-linear reconstruction')
        elif metric == 'normalized_l2':
            close(_normed(actual, 'evolved state'),
                  _normed(approximate, 'expected state'), _TOL,
                  f'delta={delta}: normalized reconstruction')
        else:
            require(
                1 - _fidelity(actual, approximate) <= 1e-10,
                f'delta={delta}: reconstructed state differs beyond an allowed phase/normalization'
            )
        exact = expm(-.63j * h) @ state
        expected_errors.append(_state_error(approximate, exact, metric))
    errors = real_array(payload, 'errors', (3, ))
    close(errors, np.array(expected_errors), _TOL,
          'declared evolved-state errors')
    require(np.all(errors >= 0), 'errors must be nonnegative')
    decisions = payload.get('below_threshold')
    require(
        isinstance(decisions, list) and len(decisions) == 3
        and all(type(x) is bool for x in decisions),
        'below_threshold must contain three booleans')
    require(decisions == [error < 1e-3 for error in expected_errors],
            'incorrect below-threshold decisions for the declared metric')


_HANDLERS = dict(
    zip(CASES, (_orbital, _correlated, _herald, _truncation,
                _return_amplitudes, _convergence, _filters, _calibration)))


def check(case_id, payload):
    """Validate executed numerical outputs, raising CheckFailure on failure."""
    require(case_id in _HANDLERS, f'unsupported states/walk case: {case_id}')
    require(isinstance(payload, dict), 'output must be a JSON object')
    try:
        _HANDLERS[case_id](payload)
    except CheckFailure:
        raise
    except (TypeError, ValueError, KeyError, IndexError, OverflowError) as exc:
        raise CheckFailure(f'invalid scientific output: {exc}') from exc
