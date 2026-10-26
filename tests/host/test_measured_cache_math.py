"""Independent mathematical gate, before production lowering is introduced."""
import sys
import subprocess
from pathlib import Path
import pytest
import qrom_channel_reference as ref


@pytest.mark.parametrize('n', range(1, 4))
@pytest.mark.parametrize('w', range(1, 5))
def test_clean_cache_kraus_identity(n, w):
    ref.check_cache(n, w, controlled=False)


def test_controlled_cache_kraus_identity_with_reference():
    ref.check_cache(2, 2, controlled=True, reference=True)


@pytest.mark.parametrize('k', [1, 2, 4])
def test_phase_onehot_reference(k):
    ref.check_phase(k)


@pytest.mark.parametrize(
    'mutation',
    ['missing_fixup', 'missing_control', 'packed_order', 'missing_reset'])
def test_mutation_sentinels(mutation):
    ref.check_mutation(mutation)


def test_no_cudaq():
    # A fresh process proves the host dependency boundary even when native
    # tests in this pytest session have already imported CUDA-Q.
    code = '''
import sys
from pathlib import Path
from qrom_host_loader import load_qrom_host_module
import qrom_channel_reference
for name in ('_unary_iteration_ir', '_qrom_common', '_qrom_plan',
             '_qrom_cache_erasure', '_qrom_measured_plan',
             '_select_copy_plan', '_qrom_retained_plan'):
    if (Path('../../python/cudaq_algorithms/primitives') / (name + '.py')).exists():
        load_qrom_host_module(name)
assert 'cudaq' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', code],
                   cwd=Path(__file__).parent,
                   check=True)


@pytest.mark.parametrize('controlled', [False, True])
def test_packed_cache_partial_block(controlled):
    import numpy as np
    data = (1, 2, 3)
    expected = ref.direct_xor_isometry(data, 3, 2, controlled, 4) / 4
    for outcome in range(16):
        actual = ref.cache_branch(data,
                                  3,
                                  2,
                                  outcome,
                                  controlled,
                                  block_size=2)
        ref.assert_close(actual, expected)
        ref.assert_close(np.sum(abs(actual)**2, axis=0),
                         np.full(actual.shape[1], 1 / 16))


def test_tape_interpreter_retains_slots_across_temporary_measurements():
    import numpy as np
    from types import SimpleNamespace
    # Cache starts at |1>, tree AND is zero. The intervening tree result must
    # not overwrite cache slot 0; its retained result repairs the minus sign.
    plan = SimpleNamespace(ops=((24, 0, 0, 0), (22, 1, 2, 0), (26, 2, 0, 0)),
                           parity_masks=((0, ), ),
                           num_results=1)
    state = np.zeros(16, complex)
    state[2 | 8] = 1
    expected = np.zeros(16, complex)
    expected[8] = .5
    actual = ref.execute_plan(plan,
                              state, (1, 0),
                              address=(0, ),
                              ladder=(1, 2, 3),
                              output=())
    ref.assert_close(actual, expected)


def candidate_plan(data, controlled, k, *, dirty=False):
    """Assemble existing pure walk/rebase and production erasure (n=3,b=1,B=2).

    This is the candidate under test, not an expected map. Task 5 replaces
    assembly with the public pure factory; expected XOR remains direct.
    """
    from qrom_host_loader import load_qrom_host_module as load
    from types import SimpleNamespace
    ir = load('_unary_iteration_ir')
    common = load('_qrom_common')
    cache = load('_qrom_cache_erasure')
    words = tuple(
        sum((data[2 * j + i] if 2 * j + i < len(data) else 0) << i
            for i in range(2)) for j in range((len(data) + 1) // 2))
    bodies = [[('x', t) for t in range(2) if word >> t & 1] for word in words]
    walk = common.rebase_block_walk(ir._emit_measured_walk(
        2, len(words), controlled, bodies, False),
                                    address_offset=1,
                                    output_bits=0,
                                    bank_offset=2)
    route = ((3, 3, 2, 0), (4, 2, 0, 3), (3, 3, 2, 0))
    echo = route + ((20, 2, 0, 0), ) + route[::-1]
    if dirty:
        return SimpleNamespace(ops=walk + echo + walk + echo,
                               parity_masks=(),
                               num_results=0)
    erase = cache._plan_cache_erasure(words,
                                      2,
                                      2,
                                      controlled=controlled,
                                      phase_block_size=k,
                                      cache_offset=2,
                                      tree_offset=0,
                                      address_offset=1)
    return SimpleNamespace(ops=walk + echo + erase.ops,
                           parity_masks=erase.parity_masks,
                           num_results=erase.num_results)


def check_emitted_plan(plan, data, controlled, *, dirty=False, twice=False):
    import itertools
    import numpy as np
    c = int(controlled)
    logical = c + 4 + (2 if dirty else 0)
    # Dirty bank comes before clean tree; all clean ancillas are trailing bits.
    ladder = (logical, logical + 1, c + 4,
              c + 5) if dirty else tuple(range(logical, logical + 4))
    wires = dict(address=tuple(range(c, c + 3)),
                 output=(c + 3, ),
                 ladder=ladder,
                 control=0 if controlled else None)
    expected = ref.direct_xor_isometry(data, 3, 1, controlled,
                                       2 if dirty else 4, 2 if dirty else 0)
    initial = np.zeros_like(expected)
    initial[:1 << logical] = np.eye(1 << logical)
    q = sum(op[0] in (22, 23, 24, 25) for op in plan.ops)
    assert q <= 12, 'larger fixtures belong in host count tests'
    scale = 2**(-q / 2)
    completeness = np.zeros((1 << logical, ) * 2, complex)
    rng = np.random.default_rng(817)
    # Two columns are amplitudes of an untouched spectator. All address,
    # output and control values (including padding) have complex support.
    psi = rng.normal(size=(1 << logical,
                           2)) + 1j * rng.normal(size=(1 << logical, 2))
    psi /= np.linalg.norm(psi)
    maximum = 0.
    for history in itertools.product((0, 1), repeat=q):
        branch = ref.execute_plan(plan, initial, history, **wires)
        target = scale * expected
        ref.assert_close(branch, target)
        completeness += branch.conj().T @ branch
        ref.assert_close(np.linalg.norm(branch @ psi)**2, 2.**-q)
        ref.assert_close(branch @ psi, target @ psi)
        if twice:
            # First-pass proof holds for every logical state. Check a second
            # invocation with fresh slots for every second history as well.
            repeated = ref.execute_plan(plan, expected @ psi, history, **wires)
            ref.assert_close(repeated, scale * (initial @ psi))
        maximum = max(maximum, float(np.max(abs(branch - target))))
    ref.assert_close(completeness, np.eye(1 << logical))
    return q, maximum


@pytest.mark.parametrize('controlled', [False, True])
@pytest.mark.parametrize('k', [1, 2])
@pytest.mark.parametrize('m', [1, 3, 5, 8])
def test_emitted_complete_lookup_all_histories(controlled, k, m):
    data = (1, 0, 0, 1, 1, 1, 0, 1)[:m]
    check_emitted_plan(candidate_plan(data, controlled, k),
                       data,
                       controlled,
                       twice=True)


@pytest.mark.parametrize('controlled', [False, True])
def test_emitted_dirty_echo_all_bank_inputs_and_histories(controlled):
    data = (1, 0, 1)
    check_emitted_plan(candidate_plan(data, controlled, 1, dirty=True),
                       data,
                       controlled,
                       dirty=True,
                       twice=True)


@pytest.mark.parametrize('mutation', [
    'missing_fixup', 'packed_order', 'missing_reset', 'wrong_address',
    'missing_unroute'
])
def test_actual_tape_mutations_are_detected(mutation):
    from types import SimpleNamespace
    data = (1, 0, 0, 1, 1)
    p = candidate_plan(data, True, 2)
    ops = list(p.ops)
    masks = p.parity_masks
    if mutation == 'missing_fixup':
        ops = [op for op in ops if op[0] not in (26, 27)]
    elif mutation == 'packed_order':
        masks = tuple(tuple(1 - t for t in mask) for mask in masks)
    elif mutation == 'missing_reset':
        # Reintroduce a measured bank's |1> after its guaranteed reset.
        at = next(i for i, op in enumerate(ops) if op[0] == 24)
        ops.insert(at + 1, (1, 2, 0, 0))
    elif mutation == 'wrong_address':
        at = next(i for i, op in enumerate(ops) if op[0] == 22)
        op, a, b, c = ops[at]
        ops[at] = (op, a, b, 0)
    else:
        at = next(i for i, op in enumerate(ops) if op[0] == 20)
        del ops[at + 1:at + 4]
    bad = SimpleNamespace(ops=tuple(ops),
                          parity_masks=masks,
                          num_results=p.num_results)
    with pytest.raises(AssertionError):
        check_emitted_plan(bad, data, True)


@pytest.mark.parametrize('controlled', [False, True])
@pytest.mark.parametrize('k', [1, 2])
def test_emitted_erasure_direct_loaded_subspace_all_histories(controlled, k):
    import itertools
    import numpy as np
    from qrom_host_loader import load_qrom_host_module as load
    # Deliberately asymmetric cache words, one partial row and padded row.
    words = (1, 2, 3)
    c = int(controlled)
    dim = 1 << (c + 2)
    state = np.zeros((dim << 4, dim), complex)
    expected = np.zeros_like(state)
    expected[:dim] = np.eye(dim)
    for source in range(dim):
        high = source >> c
        enabled = source & 1 if controlled else 1
        word = words[high] if high < len(words) else 0
        state[source | ((enabled * word) << (c + 4)), source] = 1
    p = load('_qrom_cache_erasure')._plan_cache_erasure(words,
                                                        2,
                                                        2,
                                                        controlled=controlled,
                                                        phase_block_size=k,
                                                        cache_offset=2,
                                                        tree_offset=0)
    q = p.resources.num_measurements
    completeness = np.zeros((dim, dim), complex)
    for history in itertools.product((0, 1), repeat=q):
        branch = ref.execute_plan(p,
                                  state,
                                  history,
                                  address=(c, c + 1),
                                  ladder=tuple(range(c + 2, c + 6)),
                                  output=(),
                                  control=0 if controlled else None)
        ref.assert_close(branch, expected * 2**(-q / 2))
        completeness += branch.conj().T @ branch
    ref.assert_close(completeness, np.eye(dim))
