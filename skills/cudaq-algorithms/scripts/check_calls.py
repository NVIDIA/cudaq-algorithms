#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only source-signature probe; never imports or executes inspected code.

Checked means only that explicit arguments bind to a declared Python signature.
Types, values, CUDA-Q compilation, runtime behavior and numerics are unverified.
CLI status: 1 = definite binding errors; 2 = unresolved calls or no calls checked;
0 = reported call bindings checked without errors/unknowns, not code validity.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import inspect
import json
from pathlib import Path


@dataclass
class Ref:
    kind: str
    name: str = ''
    module: str = ''
    node: ast.AST | None = None
    path: str = ''
    reason: str = ''
    line: int = 0
    scope: Scope | None = None


@dataclass
class Scope:
    parent: Scope | None = None
    bindings: dict[str, Ref] = field(default_factory=dict)
    wildcard: bool = False
    tainted: set[str] = field(default_factory=set)

    def bind(self, name, ref):
        if name in self.bindings:
            self.bindings[name] = Ref('unknown', name, reason='multiple bindings or shadowed alias')
        else:
            self.bindings[name] = ref

    def lookup(self, name, line=None):
        if name in self.tainted:
            return Ref('unknown', name, reason='global/nonlocal rebinding may shadow this alias')
        if self.wildcard:
            return Ref('unknown', name, reason='wildcard import or dynamic namespace modification')
        if name in self.bindings:
            ref = self.bindings[name]
            if line is not None and ref.line > line:
                return Ref('unknown', name, reason='binding appears after this call')
            return ref
        if self.parent:
            return self.parent.lookup(name, line)
        return Ref('unknown', name, reason='no explicit resolvable binding')


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = dotted(node.value)
        return head + '.' + node.attr if head else None
    return None


class Bindings(ast.NodeVisitor):
    """Conservative lexical bindings: assignments/rebindings are never inferred."""
    def __init__(self, scope, module='', package='', path=''):
        self.scope, self.module, self.package, self.path = scope, module, package, path
        self.conditional = 0

    def put(self, name, ref):
        if self.conditional:
            ref = Ref('unknown', name, reason='binding depends on control flow')
        self.scope.bind(name, ref)

    def visit_Import(self, node):
        for item in node.names:
            name = item.asname or item.name.split('.')[0]
            target = item.name if item.asname else name
            self.put(name, Ref('module', target, module=target, line=node.lineno))

    def visit_ImportFrom(self, node):
        if node.level:
            parts = self.package.split('.') if self.package else []
            keep = len(parts) - node.level + 1
            target = '.'.join(parts[:keep]) if keep > 0 else ''
            if target and node.module:
                target += '.' + node.module
        else:
            target = node.module or ''
        for item in node.names:
            if item.name == '*':
                self.scope.wildcard = True
                continue
            name = item.asname or item.name
            self.put(name, Ref('export', item.name, module=target, line=node.lineno))

    def visit_FunctionDef(self, node):
        self.put(node.name, Ref('function', node.name, self.module, node, self.path,
                               line=node.lineno, scope=self.scope))
        # A nested function body is a different lexical scope.
        for expression in node.decorator_list + node.args.defaults + [v for v in node.args.kw_defaults if v]:
            self.visit(expression)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.put(node.name, Ref('class', node.name, self.module, node, self.path, line=node.lineno))
        for expression in node.decorator_list + node.bases + [kw.value for kw in node.keywords]:
            self.visit(expression)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.put(node.id, Ref('unknown', node.id, reason='assigned name or shadowed alias', line=node.lineno))

    def visit_Attribute(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            name = dotted(node)
            if name:
                self.put(name.split('.')[0], Ref('unknown', name, reason='attribute mutation may replace an API'))
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id in ('exec', 'eval'):
            self.scope.wildcard = True
        if isinstance(node.func, ast.Name) and node.func.id == 'setattr' and node.args:
            name = dotted(node.args[0])
            if name:
                self.put(name.split('.')[0], Ref('unknown', name, reason='dynamic attribute mutation'))
        self.generic_visit(node)

    def visit_Subscript(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)) and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Name) and node.value.func.id in ('globals', 'locals'):
                self.scope.wildcard = True
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.put(node.name, Ref('unknown', node.name, reason='exception target shadows alias'))
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name:
            self.put(node.name, Ref('unknown', node.name, reason='match capture shadows alias'))
        self.generic_visit(node)

    visit_MatchStar = visit_MatchAs

    def visit_MatchMapping(self, node):
        if node.rest:
            self.put(node.rest, Ref('unknown', node.rest, reason='match mapping capture shadows alias'))
        self.generic_visit(node)

    def comprehension(self, node):
        # Generator targets live in the comprehension scope. Assignment
        # expressions still conservatively affect the enclosing lexical scope.
        for generator in node.generators:
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for expression in ([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]):
            self.visit(expression)

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = comprehension

    def visit_Global(self, node):
        for name in node.names:
            self.put(name, Ref('unknown', name, reason='global/nonlocal binding not inferred'))

    visit_Nonlocal = visit_Global

    def control_flow(self, node):
        self.conditional += 1
        self.generic_visit(node)
        self.conditional -= 1

    visit_If = visit_For = visit_AsyncFor = visit_While = control_flow
    visit_Try = visit_TryStar = visit_With = visit_AsyncWith = visit_Match = control_flow

    def visit_Lambda(self, node):
        for value in node.args.defaults + [v for v in node.args.kw_defaults if v]:
            self.visit(value)


class Sources:
    def __init__(self, package_root, package_name):
        self.name = package_name
        self.root = Path(package_root).resolve()
        if not (self.root / '__init__.py').is_file():
            nested = self.root.joinpath(*package_name.split('.'))
            if (nested / '__init__.py').is_file():
                self.root = nested.resolve()
        self.cache = {}
        self.loaded_modules = set()

    def local(self, module):
        return module == self.name or module.startswith(self.name + '.')

    def source_path(self, module):
        if not self.local(module):
            return None
        parts = module[len(self.name):].strip('.').split('.') if module != self.name else []
        if any(not part.isidentifier() for part in parts):
            return None
        base = self.root.joinpath(*parts)
        for path in (base / '__init__.py', base.with_suffix('.py')):
            # Symlinks cannot make resolution leave the supplied package tree.
            if path.resolve().is_relative_to(self.root) and path.is_file():
                return path
        return None

    def scope(self, module):
        if module in self.cache:
            return self.cache[module]
        path = self.source_path(module)
        if path is None:
            return None
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            self.cache[module] = None
            return None
        scope = Scope()
        package = module if path.name == '__init__.py' else module.rpartition('.')[0]
        collector = Bindings(scope, module, package, str(path))
        for statement in tree.body:
            collector.visit(statement)
        self.cache[module] = scope
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                self.loaded_modules.update(item.name for item in statement.names)
        return scope

    def resolve(self, ref, seen=()):
        if ref.kind != 'export':
            return ref
        key = (ref.module, ref.name)
        if key in seen or len(seen) >= 64:
            return Ref('unknown', ref.name, ref.module, reason='cyclic or excessively deep reexport')
        if not self.local(ref.module):
            return Ref('external', ref.module + '.' + ref.name, reason='outside inspected package')
        scope = self.scope(ref.module)
        if scope and ref.name in scope.bindings:
            bound = scope.lookup(ref.name)
            # `from . import child` in __init__.py loads that submodule;
            # it is not a recursive import of a function with the same name.
            if bound.kind == 'export' and (bound.module, bound.name) == key:
                child = ref.module + '.' + ref.name
                if self.source_path(child):
                    return Ref('module', child, child)
            return self.resolve(bound, seen + (key,))
        # Python `from package import submodule` may load a real submodule.
        child = ref.module + '.' + ref.name
        if self.source_path(child):
            return Ref('module', child, child)
        return Ref('unknown', ref.name, ref.module, reason='unresolved API or dynamic export')

    def expression(self, node, scope):
        if isinstance(node, ast.Name):
            return self.resolve(scope.lookup(node.id, node.lineno))
        if isinstance(node, ast.Attribute):
            base = self.expression(node.value, scope)
            if base.kind == 'module':
                if not self.local(base.module):
                    return Ref('external', base.module + '.' + node.attr, reason='outside inspected package')
                exports = self.scope(base.module)
                if exports and node.attr in exports.bindings:
                    return self.resolve(exports.lookup(node.attr))
                child = base.module + '.' + node.attr
                if child in self.loaded_modules and self.source_path(child):
                    return Ref('module', child, child)
                return Ref('unknown', child, reason='module attribute is not an explicit source export')
            return Ref('unknown', dotted(node) or '<attribute>', reason=base.reason or 'method/attribute target not inferred')
        return Ref('unknown', '<dynamic call>', reason='computed call target not inferred')


def signature(ref, sources):
    node = ref.node
    if node.decorator_list:
        if len(node.decorator_list) != 1:
            return None, 'multiple decorators may transform the call contract'
        decorator = sources.expression(node.decorator_list[0], ref.scope)
        if decorator.kind != 'external' or decorator.name != 'cudaq.kernel':
            return None, 'unsupported decorator; effective signature is unproved'
    args = node.args
    params = []
    positional = args.posonlyargs + args.args
    first_default = len(positional) - len(args.defaults)
    for index, argument in enumerate(positional):
        kind = inspect.Parameter.POSITIONAL_ONLY if index < len(args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD
        default = inspect.Parameter.empty if index < first_default else '<source default>'
        params.append(inspect.Parameter(argument.arg, kind, default=default))
    if args.vararg:
        params.append(inspect.Parameter(args.vararg.arg, inspect.Parameter.VAR_POSITIONAL))
    for argument, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(inspect.Parameter(argument.arg, inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if default is None else '<source default>'))
    if args.kwarg:
        params.append(inspect.Parameter(args.kwarg.arg, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(params), None


class Calls(ast.NodeVisitor):
    def __init__(self, sources, scope, output):
        self.sources, self.scope, self.output = sources, scope, output

    def visit_Call(self, node):
        ref = self.sources.expression(node.func, self.scope)
        display = dotted(node.func) or ast.unparse(node.func)
        name = (ref.module + '.' if ref.module else '') + ref.name if ref.kind in ('function', 'class', 'export') else ref.name
        item = dict(line=node.lineno, call=display, name=name, evidence={})
        category, reason = 'unknown', ref.reason or 'unsupported constructor or non-function call target'
        if ref.kind == 'class':
            item['evidence'] = dict(path=ref.path or '<snippet>', line=ref.node.lineno, declaration='class ' + ref.name)
        if ref.kind == 'function':
            item['evidence'] = dict(path=ref.path or '<snippet>', line=ref.node.lineno,
                declaration=ast.unparse(ref.node.args), qualified_name=(ref.module + '.' if ref.module else '') + ref.name)
            try:
                declared, reason = signature(ref, self.sources)
            except (TypeError, ValueError) as error:
                declared, reason = None, 'source signature cannot be represented: ' + str(error)
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(kw.arg is None for kw in node.keywords):
                reason = 'dynamic *args/**kwargs cannot be bound without evaluation'
            elif declared is not None:
                keywords = [kw.arg for kw in node.keywords]
                try:
                    if len(set(keywords)) != len(keywords):
                        raise TypeError('duplicate keyword argument')
                    declared.bind(*[None for _ in node.args], **dict.fromkeys(keywords))
                except TypeError as error:
                    category, reason = 'errors', str(error)
                else:
                    category, reason = 'checked', 'explicit arguments bind to source signature only; runtime/types/numerics unverified'
        item['reason'] = reason
        self.output[category].append(item)
        self.generic_visit(node)

    def function(self, node):
        for expression in node.decorator_list + node.args.defaults + [v for v in node.args.kw_defaults if v]:
            self.visit(expression)
        parent = self.scope
        child = Scope(parent=parent, tainted=parent.tainted)
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs + [a for a in (node.args.vararg, node.args.kwarg) if a]:
            child.bind(arg.arg, Ref('unknown', arg.arg, reason='function parameter shadows any outer alias'))
        collector = Bindings(child)
        for statement in node.body:
            collector.visit(statement)
        self.scope = child
        for statement in node.body:
            self.visit(statement)
        self.scope = parent

    visit_FunctionDef = visit_AsyncFunctionDef = function

    def visit_Lambda(self, node):
        # Defaults are evaluated in the enclosing scope before parameters bind.
        for expression in node.args.defaults + [v for v in node.args.kw_defaults if v is not None]:
            self.visit(expression)
        child = Scope(parent=self.scope, tainted=self.scope.tainted)
        for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs + [a for a in (node.args.vararg, node.args.kwarg) if a]:
            child.bind(arg.arg, Ref('unknown', arg.arg, reason='lambda parameter shadows any outer alias'))
        old, self.scope = self.scope, child
        self.visit(node.body)
        self.scope = old

    def visit_ClassDef(self, node):
        for expression in node.decorator_list + node.bases + [kw.value for kw in node.keywords]:
            self.visit(expression)
        # Class descriptors and method binding are outside this narrow probe.
        child = Scope(parent=self.scope, wildcard=True, tainted=self.scope.tainted)
        old, self.scope = self.scope, child
        for statement in node.body:
            self.visit(statement)
        self.scope = old

    def comprehension(self, node):
        # The first iterable is evaluated in the enclosing scope; generator
        # targets shadow names inside filters, later iterables, and the result.
        self.visit(node.generators[0].iter)
        child = Scope(parent=self.scope, tainted=self.scope.tainted)
        collector = Bindings(child)
        for generator in node.generators:
            collector.visit(generator.target)
        old, self.scope = self.scope, child
        for index, generator in enumerate(node.generators):
            if index:
                self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for expression in ([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]):
            self.visit(expression)
        self.scope = old

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = comprehension


def check_source(code: str, package_root: Path, package_name: str = 'cudaq_algorithms') -> dict:
    output = dict(checked=[], errors=[], unknown=[])
    if not package_name or any(not part.isidentifier() for part in package_name.split('.')):
        output['unknown'].append(dict(line=0, call='<configuration>', name=package_name,
            reason='package_name must be a dotted Python identifier', evidence={}))
        return output
    try:
        tree = ast.parse(code, filename='<snippet>')
    except SyntaxError as error:
        output['unknown'].append(dict(line=error.lineno or 0, call='<syntax>', name='<snippet>',
            reason='snippet cannot be parsed: ' + error.msg, evidence={}))
        return output
    tainted = {name for node in ast.walk(tree) if isinstance(node, (ast.Global, ast.Nonlocal)) for name in node.names}
    sources, scope = Sources(package_root, package_name), Scope(tainted=tainted)
    collector = Bindings(scope)
    for statement in tree.body:
        collector.visit(statement)
        if isinstance(statement, ast.Import):
            sources.loaded_modules.update(item.name for item in statement.names)
    Calls(sources, scope, output).visit(tree)
    for values in output.values():
        values.sort(key=lambda item: (item['line'], item['call']))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--package-name', default='cudaq_algorithms')
    parser.add_argument('--summary', action='store_true',
                        help='print counts, all errors, and unresolved reason counts; preserve exit status')
    parser.add_argument('snippet', type=Path)
    args = parser.parse_args()
    try:
        code = args.snippet.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as error:
        result = dict(checked=[], errors=[], unknown=[dict(line=0, call='<read>', name=str(args.snippet), reason=str(error), evidence={})])
    else:
        result = check_source(code, args.package_root, args.package_name)
    display = result
    if args.summary:
        reasons = {}
        for item in result['unknown']:
            reason = item['reason']
            reasons[reason] = reasons.get(reason, 0) + 1
        display = dict(
            counts={key: len(items) for key, items in result.items()},
            errors=result['errors'],
            unknown_reasons=dict(sorted(reasons.items())),
            note='Argument-binding counts only; runtime, types and numerics remain unverified. '
                 'Rerun without --summary for individual checked and unresolved evidence.')
    print(json.dumps(display, indent=2))
    if result['errors']:
        return 1
    if result['unknown'] or not result['checked']:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
