"""Registered private executable checks and public artifact interfaces."""

from importlib import import_module

MODULES = ('regression', 'states_walk', 'dynamics', 'chemistry')

# Import statements that prove a private-reference dependency is usable in the
# configured runtime. Every checker needs the common ones; a module declares
# any further names in its own ``DEPENDENCIES`` tuple.
IMPORT_PROBES = {
    'numpy': 'import numpy',
    'scipy': 'import scipy',
    'pyscf': 'import pyscf; from pyscf import gto, scf, fci'
}
COMMON_DEPENDENCIES = ('numpy', 'scipy')


def module_for(case_id):
    if not case_id.startswith('science-'):
        name = 'regression'
    elif any(
            case_id.startswith('science-' + prefix)
            for prefix in ('state-preparation-', 'block-encoding-',
                           'qubitization-', 'qsvt-')):
        name = 'states_walk'
    elif any(
            case_id.startswith('science-' + prefix)
            for prefix in ('chemistry-', 'double-factorization-',
                           'workflow-molecular-')):
        name = 'chemistry'
    else:
        name = 'dynamics'
    module = import_module('.' + name, __name__)
    return module if case_id in module.CASES else None


def dependencies(case_id):
    """Private-reference dependencies of the registered checker for a case."""
    module = module_for(case_id)
    if module is None:
        raise ValueError(f'No registered checker: {case_id}')
    extra = tuple(getattr(module, 'DEPENDENCIES', ()))
    for name in COMMON_DEPENDENCIES + extra:
        if name not in IMPORT_PROBES:
            raise ValueError(f'No import probe for dependency: {name}')
    return COMMON_DEPENDENCIES + extra


def specifications():
    result = {}
    for name in MODULES:
        module = import_module('.' + name, __name__)
        for case, specification in module.CASES.items():
            if case in result:
                raise ValueError(f'duplicate checker: {case}')
            result[case] = specification
    return result


def public_contract(case_id):
    module = module_for(case_id)
    return module.CASES[case_id]['output'] if module else None


def check(case_id, payload):
    module = module_for(case_id)
    if module is None:
        raise ValueError(f'No registered checker: {case_id}')
    module.check(case_id, payload)
