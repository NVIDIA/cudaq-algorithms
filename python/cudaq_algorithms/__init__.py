import importlib as _importlib

try:
    from ._pycudaq_algorithms import *
    from ._pycudaq_algorithms import __version__
    from ._pycudaq_algorithms import stateprep as _stateprep_extension
except ImportError:
    raise

_stateprep_wrappers = _importlib.import_module(".stateprep", __name__)

# Keep the extension module visible for CUDA-Q device-kernel registration, but
# use the Python wrapper for host-side overload dispatch.
stateprep = _stateprep_extension
stateprep.make_givens_rotation_schedule = (
    _stateprep_wrappers.make_givens_rotation_schedule)
stateprep.make_slater_determinant_plan = (
    _stateprep_wrappers.make_slater_determinant_plan)
