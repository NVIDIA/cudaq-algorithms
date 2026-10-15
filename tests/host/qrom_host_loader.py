"""Load pure QROM planners without importing public CUDA-Q packages."""
import importlib.util
from pathlib import Path
import sys
import types

_ROOT = Path(
    __file__).resolve().parents[2] / 'python/cudaq_algorithms/primitives'
_PACKAGE = '_qrom_host_primitives'


def load_qrom_host_module(name: str):
    if not name.isidentifier():
        raise ValueError('expected a primitive module basename')
    if _PACKAGE not in sys.modules:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(_ROOT)]
        sys.modules[_PACKAGE] = package
    fullname = f'{_PACKAGE}.{name}'
    if fullname in sys.modules:
        return sys.modules[fullname]
    spec = importlib.util.spec_from_file_location(fullname,
                                                  _ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    cudaq_was_loaded = 'cudaq' in sys.modules
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(fullname, None)
        raise
    assert cudaq_was_loaded or 'cudaq' not in sys.modules
    return module
