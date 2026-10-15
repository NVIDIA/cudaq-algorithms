"""Host contracts and tapes captured before extracting the shared builders."""
import json
from pathlib import Path
import pytest
from qrom_host_loader import load_qrom_host_module as load

ROOT = Path(
    __file__).resolve().parents[2] / 'python/cudaq_algorithms/primitives'


def test_baseline_tapes():
    ir = load('_unary_iteration_ir')
    for case in json.loads(
            Path(__file__).with_name('qrom_baseline_tapes.json').read_text()):
        n, t, c, b = (case[k] for k in ('n', 't', 'c', 'bodies'))
        calls = []
        bodies = ir.materialize_bodies(
            lambda k: (calls.append(k) or iter(b[k])), t)
        assert calls == list(range(t))
        assert ir._emit_walk(n, t, c, lambda k: bodies[k]) == list(
            map(tuple, case['coherent']))
        for reverse, key in [(False, 'forward'), (True, 'reverse')]:
            assert ir._emit_measured_walk(n, t, c, bodies, reverse) == list(
                map(tuple, case[key]))
        assert ir.infer_num_work(ir._emit_walk(n, t, c, lambda k: bodies[k]),
                                 None) == 3


def test_validation_and_geometry():
    common = load('_qrom_common')
    ir = load('_unary_iteration_ir')
    assert common.validate_qrom_data(iter([0, 1.0, True]), 2.0,
                                     1.0) == ((0, 1, 1), 2, 1)
    assert ir.validate_unary_inputs(2.0, 3.0) == (2, 3)
    g = common.BlockGeometry(5, 3, 19, 4)
    assert (g.low_bits, g.high_bits, g.num_blocks) == (2, 3, 5)
    for data, n, b in [([], 2, 1), ([2], 2, 1), ([-1], 2, 1), ([.5], 2, 1),
                       ([0] * 5, 2, 1)]:
        with pytest.raises(ValueError):
            common.validate_qrom_data(iter(data), n, b)


@pytest.mark.parametrize('gate', [('x', ), ('x', -1), ('unknown', 0),
                                  ('x_w', 0), ('and_tt', 0, 1, -1)])
def test_body_validation(gate):
    with pytest.raises(ValueError):
        load('_unary_iteration_ir').emit_body_gates([], 0, 0, [gate])


def test_rebase_operands_and_token_boundary():
    ir = load('_unary_iteration_ir')
    common = load('_qrom_common')
    for opcode, positions in ir._ADDRESS_OPERANDS.items():
        expected = [1, 2, 3]
        for pos in positions:
            expected[pos] += 7
        assert common.rebase_block_walk([(opcode, 1, 2, 3)],
                                        address_offset=7,
                                        output_bits=2,
                                        bank_offset=10) == ((opcode,
                                                             *expected), )
    assert common.rebase_block_walk(
        [(ir._OP_BODY_X, 0, 1, 0), (ir._OP_BODY_X, 0, 2, 0)],
        address_offset=0,
        output_bits=2,
        bank_offset=10) == ((ir._OP_BODY_X, 0, 1, 0), (ir._OP_CX_LADDER_LADDER,
                                                       0, 10, 0))


def test_resources_distinguish_routing():
    p = load('_qrom_plan')
    r = p.QROMResources((p.QROMStageResources(
        'tree', 5, 5, 5, 5), p.QROMStageResources('routing', 12)), 7, 4)
    assert (r.toffoli_count, r.tree_and_count, r.temporary_and_count,
            r.num_measurements, r.num_ladder) == (17, 5, 5, 5, 11)
    with pytest.raises(Exception):
        r.num_clean = 3


def load_constructor(filename, source=None):
    """Execute factory logic with the runtime binding replaced by a tape sink.

    This never imports CUDA-Q and deliberately does not claim kernel coverage.
    """
    import ast
    from types import SimpleNamespace
    ir = load('_unary_iteration_ir')
    namespace = dict(vars(ir))
    namespace.update(vars(load('_qrom_common')))
    namespace.update(vars(load('_qrom_plan')))
    namespace['_plan_measured_qrom'] = load(
        '_qrom_measured_plan')._plan_measured_qrom

    def walk(n,
             t,
             body,
             controlled=False,
             include_adjoint=True,
             num_work=None):
        bodies = ir.materialize_bodies(body, t)
        ops = (ir._emit_measured_walk(n, t, controlled, bodies, False)
               if filename == '_qrom_measured.py' else ir._emit_walk(
                   n, t, controlled, lambda k: bodies[k]))
        return SimpleNamespace(
            kernel=tuple(ops),
            ops=tuple(ops),
            num_ladder=n,
            toffoli_count=sum(op[0] in ir._TOFFOLI_OPCODES for op in ops),
            num_measurements=sum(op[0] in ir._MEASURED_OPCODES for op in ops),
            tree_and_count=sum(op[0] in (ir._OP_CCX, ir._OP_CCX_CTRL)
                               for op in ops))

    namespace.update(_mint_interpreter=lambda ops, **kw: tuple(ops),
                     unary_iteration_kernels=walk,
                     measured_unary_iteration_kernels=walk)
    tree = ast.parse(source if source is not None else (ROOT /
                                                        filename).read_text())
    tree.body = [
        node for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    exec(
        compile(tree,
                filename,
                'exec',
                flags=__import__('__future__').annotations.compiler_flag),
        namespace)
    return namespace['MeasuredQROM' if filename ==
                     '_qrom_measured.py' else 'QROM']


def test_qrom_constructor_baseline_tapes():
    for case in json.loads(
            Path(__file__).with_name(
                'qrom_constructor_baseline.json').read_text()):
        factory = load_constructor(case['filename'])
        options = dict(case['kwargs'])
        if options.get('variant') == 'auto':
            # #64 fixtures pin legacy explicit tapes; #65 auto includes copy.
            options['variant'] = case['counts'][2]
            if case['counts'][3] is not None:
                options['block_size'] = case['counts'][3]
        obj = factory(iter(case['data']), case['n'], case['b'], **options)
        assert obj.kernel == tuple(map(tuple, case['ops']))
        assert (obj.toffoli_count, obj.num_ladder, obj.variant,
                obj.block_size) == tuple(case['counts'])
        assert obj.resources.toffoli_count == obj.toffoli_count
        assert obj.resources.num_ladder == obj.num_ladder
        if case['filename'] == '_qrom_measured.py':
            assert obj.resources.temporary_and_count == obj.tree_and_count == obj.num_measurements
            assert obj.resources.num_dirty == obj.num_dirty


@pytest.mark.parametrize('width', [0, -1, 1.5])
def test_work_width_rejects_insufficient_or_fractional(width):
    ir = load('_unary_iteration_ir')
    with pytest.raises(ValueError, match="body's work usage"):
        ir.infer_num_work([(ir._OP_COPY_TW, 0, 1, 0)], width)


@pytest.mark.parametrize('filename', ['_qrom.py', '_qrom_measured.py'])
def test_constructor_integral_inputs_and_bool_contract(filename):
    factory = load_constructor(filename)
    obj = factory(iter([True, 0.0]), 2.0, 1.0, variant='select')
    assert obj.data == (1, 0)
    if filename == '_qrom_measured.py':
        with pytest.raises(ValueError, match='controlled must be a bool'):
            factory([0], 2, 1, controlled=1)
        with pytest.raises(ValueError, match='dirty_banks must be a bool'):
            factory([0], 2, 1, dirty_banks=1)
        assert factory([0], 2, 1, variant='select_swap',
                       block_size=2.0).block_size == 2
    else:
        # PR #65 already normalizes integral block sizes (native extension test).
        assert factory([0], 2, 1, variant='select_swap',
                       block_size=2.0).block_size == 2


def test_pure_modules_never_import_runtime():
    import ast
    for name in ['_unary_iteration_ir', '_qrom_common', '_qrom_plan']:
        tree = ast.parse((ROOT / (name + '.py')).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != 'cudaq' for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module not in ('cudaq', '_unary_iteration',
                                           '_unary_iteration_measured')


@pytest.mark.parametrize('measured', [False, True])
def test_actual_unary_factory_materializes_callback_once(measured):
    import ast
    import dataclasses
    import __future__
    ir = load('_unary_iteration_ir')
    namespace = dict(vars(ir))
    namespace.update(dataclass=dataclasses.dataclass,
                     _mint_interpreter=lambda ops, *args: tuple(ops))
    filename = '_unary_iteration_measured.py' if measured else '_unary_iteration.py'
    tree = ast.parse((ROOT / filename).read_text())
    tree.body = [
        node for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.FunctionDef)
                 and node.name == '_mint_interpreter')
    ]
    exec(
        compile(tree,
                filename,
                'exec',
                flags=__future__.annotations.compiler_flag), namespace)
    calls = []

    def body(k):
        calls.append(k)
        return iter([('x', 0), ('copy_tw', 0, 2), ('z_w', 2)])

    factory = namespace['measured_unary_iteration_kernels'
                        if measured else 'unary_iteration_kernels']
    result = factory(3.0, 5.0, body, controlled=True, include_adjoint=True)
    assert calls == list(range(5))
    assert result.num_work == 3
    assert result.kernel_adj is not None
