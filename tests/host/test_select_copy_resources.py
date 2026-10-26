"""Independent instruction counts and budgeted SELECT-COPY search."""
import pytest
from qrom_host_loader import load_qrom_host_module

sc = load_qrom_host_module('_select_copy_plan')
ret = load_qrom_host_module('_qrom_retained_plan')
ir = load_qrom_host_module('_unary_iteration_ir')


def counted(ops, *, restore=False):
    trees = sum(op in (ir._OP_CCX, ir._OP_CCX_CTRL) for op, *_ in ops)
    bank = sum(op == ir._OP_CCX_LADDER_LADDER for op, *_ in ops)
    copies = sum(op == ir._OP_CCX_LADDER_LADDER_TARGET for op, *_ in ops)
    measured = sum(op in ir._MEASURED_OPCODES for op, *_ in ops)
    return trees + bank + copies, trees, trees + (bank
                                                  if restore else 0), measured


def totals(r):
    return r.toffoli_count, r.tree_and_count, r.temporary_and_count, r.num_measurements


def test_select_copy_costs_count_independent_operations():
    for a in range(2, 7):
        for b in range(1, 7):
            for n in (1, (1 << a) - 1, 1 << a):
                data = tuple((i * 13 + 3) % (1 << b) for i in range(n))
                for low in range(1, a):
                    for dirty in (False, True):
                        for control in (False, True):
                            for width in range(1, b + 1):
                                packets = sc.packet_layout(b,
                                                           packet_width=width)
                                args = dict(block_size=1 << low,
                                            packets=packets,
                                            controlled=control,
                                            dirty_banks=dirty)
                                plan = sc.plan_select_copy(data, a, b, **args)
                                priced = sc.select_copy_resources(
                                    n, a, b, **args)
                                assert totals(priced) == counted(plan.ops,
                                                                 restore=dirty)
                                assert priced == plan.resources
                                assert priced.classical_bits == plan.num_results == 0


def test_retained_resources_include_begin_end_and_cache():
    for a in range(2, 6):
        for b in range(1, 5):
            tables = ((1, ) * ((1 << a) - 1), (0, 1))
            for raw in (tables, ret.difference_tables(tables, a, b)):
                for low in range(1, a):
                    for dirty in (False, True):
                        for control in (False, True):
                            args = dict(block_size=1 << low,
                                        controlled=control,
                                        dirty_banks=dirty)
                            plan = ret.plan_retained_xor(raw, a, b, **args)
                            priced = ret.retained_resources(
                                len(raw), max(map(len, raw)), a, b, **args)
                            ops = tuple(op for p in (plan.begin, *plan.steps,
                                                     plan.end) for op in p.ops)
                            assert totals(priced) == counted(ops)
                            assert priced == plan.resources
                            assert priced.temporary_and_count == priced.tree_and_count
                            assert priced.num_ladder == max(
                                low, a - low) + b + ((1 << low) - 1) * b
                            # Every raw step includes one unconditional cache CNOT per output bit.
                            cache_offset = max(low, a - low)
                            for data, p in zip(raw, plan.steps):
                                assert sum(
                                    op == ir._OP_BODY_X
                                    for op, *_ in p.ops) == b + sum(
                                        data[q].bit_count()
                                        for q in range(0, len(data), 1 << low))
                                assert p.ops[-b:] == tuple(
                                    (ir._OP_BODY_X, cache_offset + t, t, 0)
                                    for t in range(b))
                            assert priced.classical_bits == 0


def test_retained_budget_search_matches_exhaustive(monkeypatch):
    tables = ((1, 2, 3, 4, 5), (0, 3))
    for a in range(3, 7):
        for b in range(3, 7):
            for dirty in (False, True):
                for control in (False, True):
                    candidates = [(B,
                                   ret.plan_retained_xor(tables,
                                                         a,
                                                         b,
                                                         block_size=B,
                                                         controlled=control,
                                                         dirty_banks=dirty))
                                  for B in (1 << l for l in range(1, a))]
                    for budget in {
                            p.resources.num_ladder + d
                            for _, p in candidates
                            for d in (-1, 0)
                    }:
                        feasible = [(B, p) for B, p in candidates
                                    if p.resources.num_ladder <= budget]
                        kwargs = dict(controlled=control,
                                      dirty_banks=dirty,
                                      max_ancillas=budget)
                        if not feasible:
                            with pytest.raises(ValueError, match='budget'):
                                ret.select_retained_qrom(
                                    tables, a, b, **kwargs)
                        else:
                            B, p = min(feasible,
                                       key=lambda item:
                                       (item[1].resources.toffoli_count, item[
                                           1].resources.num_ladder, item[0]))
                            winner = ret.select_retained_qrom(
                                tables, a, b, **kwargs)
                            assert (winner.block_size, winner.plan) == (B, p)


def test_adaptive_search_matches_exhaustive_feasible_candidates():
    measured = load_qrom_host_module('_qrom_measured_plan')
    for a in range(2, 7):
        for b in range(1, 7):
            for n in ((1 << a) - 1, 1 << a):
                data = tuple((i * 7 + 1) % (1 << b) for i in range(n))
                for dirty in (False, True):
                    for control in (False, True):
                        legacy = []
                        for c in measured._price_candidates(
                                a,
                                n,
                                b,
                                variant='auto',
                                block_size=None,
                                controlled=control,
                                dirty_banks=dirty,
                                cache_uncompute='auto',
                                phase_block_size=None):
                            p = measured._emit_selection(
                                data, a, b, c, control, dirty)
                            legacy.append((c, p))
                        copies = []
                        layouts = {
                            sc.packet_layout(b, alpha=i)
                            for i in range(1, b + 1)
                        } | {
                            sc.packet_layout(b, packet_width=i)
                            for i in range(1, b + 1)
                        }
                        for low in range(1, a):
                            for layout in layouts:
                                p = sc.plan_select_copy(data,
                                                        a,
                                                        b,
                                                        block_size=1 << low,
                                                        packets=layout,
                                                        controlled=control,
                                                        dirty_banks=dirty)
                                copies.append((1 << low, layout, p))
                        budgets = {
                            p.resources.num_ladder + d
                            for _, _, p in copies
                            for d in (-1, 0)
                        } | {
                            p.resources.num_ladder + d
                            for _, p in legacy
                            for d in (-1, 0)
                        }
                        for budget in budgets:
                            old = [(c, p) for c, p in legacy
                                   if p.resources.num_ladder <= budget]
                            new = [(B, l, p) for B, l, p in copies
                                   if p.resources.num_ladder <= budget]
                            args = dict(controlled=control,
                                        dirty_banks=dirty,
                                        cache_uncompute='auto',
                                        max_ancillas=budget)
                            if not old and not new:
                                with pytest.raises(ValueError, match='budget'):
                                    measured._plan_measured_qrom(
                                        data, a, b, **args)
                                continue
                            winner = measured._plan_measured_qrom(
                                data, a, b, **args)
                            best_old = min(old,
                                           key=lambda x: x[0].key,
                                           default=None)
                            best_new = min(
                                new,
                                key=lambda x:
                                (x[2].resources.toffoli_count, x[2].resources.
                                 num_ladder, x[0], x[1].widths),
                                default=None)
                            if best_old and (
                                    not best_new or best_old[0].cost
                                    <= best_new[2].resources.toffoli_count):
                                assert winner.variant == best_old[0].variant
                                assert winner.plan == best_old[1]
                            else:
                                assert winner.variant == 'select_copy'
                                assert (winner.block_size, winner.packets,
                                        winner.plan) == best_new


def test_select_copy_options_and_explicit_budget():
    measured = load_qrom_host_module('_qrom_measured_plan')
    args = dict(variant='select_copy',
                block_size=2,
                packet_width=2,
                dirty_banks=True)
    p = measured._plan_measured_qrom((1, 2, 3), 3, 3, **args)
    assert p.packets.widths == (2, 1)
    assert measured._plan_measured_qrom(
        (1, 2, 3), 3, 3, max_ancillas=p.plan.resources.num_ladder, **args) == p
    with pytest.raises(ValueError, match='budget'):
        measured._plan_measured_qrom(
            (1, 2, 3),
            3,
            3,
            max_ancillas=p.plan.resources.num_ladder - 1,
            **args)
    for extra in (dict(cache_uncompute='measure'), dict(phase_block_size=1),
                  dict(alpha=2)):
        with pytest.raises(ValueError):
            measured._plan_measured_qrom((1, 2, 3), 3, 3, **args, **extra)


def test_candidate_pricing_emits_only_selected_tape(monkeypatch):
    measured = load_qrom_host_module('_qrom_measured_plan')
    emitted = []
    old_emit, copy_emit = measured._emit_selection, measured.plan_select_copy

    def old(*args, **kwargs):
        emitted.append('legacy')
        return old_emit(*args, **kwargs)

    def copy(*args, **kwargs):
        emitted.append('copy')
        return copy_emit(*args, **kwargs)

    monkeypatch.setattr(measured, '_emit_selection', old)
    monkeypatch.setattr(measured, 'plan_select_copy', copy)
    measured._plan_measured_qrom(tuple(range(64)), 6, 6, max_ancillas=12)
    assert len(emitted) == 1
    retained_emit = ret.plan_retained_xor
    emitted.clear()

    def retained(*args, **kwargs):
        emitted.append('retained')
        return retained_emit(*args, **kwargs)

    monkeypatch.setattr(ret, 'plan_retained_xor', retained)
    ret.select_retained_qrom((tuple(range(64)), ), 6, 6, max_ancillas=20)
    assert emitted == ['retained']


def test_full_tree_schedule_ledger():
    # Full-tree, uniform packets, uncontrolled dirty-bank schedule only.
    # No small-B asymptotic substitution: H=N/B-2 and L=B-2 are exact.
    N, B, b, width = 64, 4, 6, 2
    packets = sc.packet_layout(b, packet_width=width)
    plan = sc.plan_select_copy(tuple(range(N)),
                               6,
                               b,
                               block_size=B,
                               packets=packets,
                               controlled=False,
                               dirty_banks=True)
    p, H, L = 3, 14, 2
    trees = (p + 1) * H + (p + 1) * L
    restore, generic = (B - 1) * width, (B - 1) * b
    assert totals(plan.resources) == (trees + restore + generic, trees,
                                      trees + restore, trees + restore)
    assert plan.resources.toffoli_count - plan.resources.temporary_and_count == generic
    assert plan.resources.temporary_and_count - plan.resources.tree_and_count == restore
    # Logical 4T/7T model, not a claim about emitted CUDA-Q synthesis.
    assert 4 * plan.resources.temporary_and_count + 7 * generic == 406


def test_paper_a1_matching_regime_stage_constants():
    # Motlagh/Pocrnic arXiv:2605.20334v1 Appendix A, HTML Eq.(9), PDF A1.
    # Compare only full-power-of-two tables, fixed-width dirty packets and
    # uncontrolled walks. Its M-1 block estimate differs from our exact M-2.
    N, B, b, mu = 64, 4, 10, 4
    alpha = (b + mu - 1) // mu
    plan = sc.plan_select_copy(tuple(range(N)),
                               6,
                               b,
                               block_size=B,
                               packets=sc.packet_layout(b, packet_width=mu),
                               controlled=False,
                               dirty_banks=True)
    paper = (alpha + 1) * (
        (N + B - 1) // B + B - 3) + (B - 1) * (mu * (b // mu + 1) + b % mu)
    exact = counted(plan.ops, restore=True)[0]
    assert exact == plan.resources.toffoli_count == 106
    assert paper == 110
    assert paper - exact == alpha + 1
    high = sum(
        s.toffoli_count for s in plan.resources.stages
        if s.name.startswith('high_transition') or s.name == 'bank_unload')
    assert high == (alpha + 1) * (N // B - 2)
    assert (alpha + 1) * (N // B - 1) - high == paper - exact
    assert plan.resources.num_clean == max(2,
                                           4) + 1  # explicit Restore scratch
