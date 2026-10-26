"""Pure cache/phase tapes and independent logical resource checks."""
import pytest
from qrom_host_loader import load_qrom_host_module as load


@pytest.mark.parametrize('controlled', [False, True])
@pytest.mark.parametrize('words,bits,k', [([1, 2, 1], 2, 2), ([0], 2, 1),
                                          ([3], 2, 2), ([3, 3, 3], 2, 1),
                                          ([1 << 70], 71, 2)])
def test_cache_counts_and_dependencies(controlled, words, bits, k):
    cache = load('_qrom_cache_erasure')
    ir = load('_unary_iteration_ir')
    p = cache._plan_cache_erasure(words,
                                  2,
                                  bits,
                                  controlled=controlled,
                                  phase_block_size=k,
                                  cache_offset=2,
                                  tree_offset=0)
    s = k.bit_length() - 1
    rows = (len(words) + k - 1) // k
    tree = rows + (2 - s) - 2 + int(controlled) - (rows - 1).bit_count()
    assert p.resources.toffoli_count == tree + k - 1
    assert p.resources.tree_and_count == tree
    assert p.resources.temporary_and_count == tree + k - 1
    assert p.resources.num_measurements == bits + tree + k - 1
    assert p.num_results == p.resources.classical_bits == bits
    ir.validate_feedback_ops(p.ops,
                             parity_masks=p.parity_masks,
                             num_results=bits,
                             num_address=2,
                             num_ladder=2 + bits,
                             num_output=0,
                             controlled=controlled)
    assert [x.name for x in p.resources.stages] == [
        'cache_measure', 'phase_unary_prepare', 'phase_walk',
        'phase_unary_erase'
    ]
    if bits > 64: assert (70, ) in p.parity_masks


def test_output_measurement_prefix():
    cache = load('_qrom_cache_erasure')
    ir = load('_unary_iteration_ir')
    p = cache._plan_cache_measurement(3,
                                      cache_offset=5,
                                      cache_register='output')
    assert p.ops == tuple(
        (ir._OP_MEASURE_RESET_OUTPUT, 5 + i, i, 0) for i in range(3))
    with pytest.raises(ValueError):
        cache._plan_cache_erasure([1],
                                  2,
                                  2,
                                  controlled=False,
                                  phase_block_size=2,
                                  cache_offset=0,
                                  tree_offset=0,
                                  cache_register='output')


def test_87_toffoli_fixture():
    cache = load('_qrom_cache_erasure')
    p = cache._plan_cache_erasure([255] * 64,
                                  6,
                                  8,
                                  controlled=False,
                                  phase_block_size=8,
                                  cache_offset=6,
                                  tree_offset=0,
                                  address_offset=2)
    assert p.resources.toffoli_count + 62 + 12 == 87
    assert p.resources.num_measurements + 62 == 83
    assert p.resources.num_clean == 14
    assert p.resources.tree_and_count == 6
    assert p.resources.temporary_and_count == 13


def test_feedback_rejects_uninitialized_and_bad_roles():
    ir = load('_unary_iteration_ir')
    kwargs = dict(parity_masks=((0, ), ),
                  num_results=1,
                  num_address=1,
                  num_ladder=2,
                  num_output=0,
                  controlled=False)
    with pytest.raises(ValueError, match='initialized'):
        ir.validate_feedback_ops([(26, 0, 0, 0)], **kwargs)
    with pytest.raises(ValueError):
        ir.validate_feedback_ops([(24, 2, 0, 0)], **kwargs)
    with pytest.raises(ValueError):
        ir.validate_feedback_ops([(24, 0, 0, 0), (27, 0, 2, 0)], **kwargs)


@pytest.mark.parametrize('h', range(1, 6))
@pytest.mark.parametrize('controlled', [False, True])
def test_independent_counts_for_every_partial_tree(h, controlled):
    cache = load('_qrom_cache_erasure')
    for count in range(1, (1 << h) + 1):
        for s in range(h):
            k = 1 << s
            p = cache._plan_cache_erasure([(j * 7 + 1) % (1 << k)
                                           for j in range(count)],
                                          h,
                                          k,
                                          controlled=controlled,
                                          phase_block_size=k,
                                          cache_offset=h,
                                          tree_offset=0)
            rows = (count + k - 1) // k
            tree = rows + h - s - 2 + int(controlled) - (rows - 1).bit_count()
            assert (p.resources.tree_and_count,
                    p.resources.temporary_and_count, p.resources.toffoli_count,
                    p.resources.num_measurements) == (tree, tree + k - 1,
                                                      tree + k - 1,
                                                      tree + k - 1 + k)


def test_phase_rebases_all_shared_address_and_ladder_operands():
    cache = load('_qrom_cache_erasure')
    ir = load('_unary_iteration_ir')
    kwargs = dict(controlled=True, phase_block_size=2)
    original = cache._plan_phase_correction(((0, ), (1, ), (), (0, 1)),
                                            3,
                                            unary_offset=3,
                                            tree_offset=0,
                                            **kwargs)
    shifted = cache._plan_phase_correction(((0, ), (1, ), (), (0, 1)),
                                           3,
                                           unary_offset=10,
                                           tree_offset=7,
                                           address_offset=4,
                                           **kwargs)
    expected = []
    for opcode, *operands in original.ops:
        for pos in ir._ADDRESS_OPERANDS.get(opcode, ()):
            operands[pos] += 4
        for pos in ir._LADDER_OPERANDS.get(opcode, ()):
            operands[pos] += 7
        expected.append((opcode, *operands))
    assert shifted.ops == tuple(expected)


@pytest.mark.parametrize('ops,masks,results', [
    ([(24, 0, 1, 0)], ((0, ), ), 1),
    ([(24, 0, 0, 1)], ((0, ), ), 1),
    ([(24, 0, 0, 0), (24, 1, 0, 0)], ((0, ), ), 1),
    ([(24, 0, 0, 0), (26, 0, 1, 0)], ((0, ), ), 1),
    ([(24, 0, 0, 0), (26, 0, 0, 1)], ((0, ), ), 1),
    ([(24, 0, 0, 0), (27, 0, 0, 0)], ((0, ), ), 1),
    ([(24, 0, 0, 0)], ((1, ), ), 1),
    ([(24, 0, 0, 0)], ((0, 0), ), 1),
    ([(9, 0, 0, 0)], (), 0),
    ([(25, 0, 0, 0)], (), 1),
    ([(28, 0, 0, 0)], (), 0),
])
def test_feedback_operand_validation(ops, masks, results):
    with pytest.raises(ValueError):
        load('_unary_iteration_ir').validate_feedback_ops(ops,
                                                          parity_masks=masks,
                                                          num_results=results,
                                                          num_address=1,
                                                          num_ladder=2,
                                                          num_output=0,
                                                          controlled=False)


def test_phase_external_slots_and_deduplicated_masks():
    cache = load('_qrom_cache_erasure')
    ir = load('_unary_iteration_ir')
    p = cache._plan_phase_correction(((70, 0), (0, 70), ()),
                                     2,
                                     controlled=False,
                                     phase_block_size=1,
                                     unary_offset=2,
                                     tree_offset=0)
    assert p.parity_masks == ((0, 70), ())
    assert p.num_results == 71
    ir.validate_feedback_ops(p.ops,
                             parity_masks=p.parity_masks,
                             num_results=71,
                             num_address=2,
                             num_ladder=2,
                             num_output=0,
                             controlled=False,
                             initialized_results=(0, 70))
    q = cache._plan_cache_erasure([1],
                                  2,
                                  71,
                                  controlled=False,
                                  phase_block_size=1,
                                  cache_offset=2,
                                  tree_offset=0)
    assert q.num_results == q.resources.classical_bits == 71


@pytest.mark.parametrize('kwargs', [
    dict(address_bits=0),
    dict(phase_block_size=3),
    dict(phase_block_size=4),
    dict(unary_offset=0),
    dict(controlled=1),
    dict(tree_offset=-1),
    dict(cache_register='bad'),
    dict(cache_words=[4]),
    dict(cache_words=[]),
    dict(cache_words=[0] * 5),
    dict(cache_bits=0),
])
def test_cache_input_validation(kwargs):
    arguments = dict(cache_words=[1],
                     address_bits=2,
                     cache_bits=2,
                     controlled=False,
                     phase_block_size=2,
                     cache_offset=2,
                     tree_offset=0)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        load('_qrom_cache_erasure')._plan_cache_erasure(**arguments)
