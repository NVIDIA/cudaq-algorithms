from cudaq_algorithms._pycudaq_algorithms import stateprep as _stateprep

uccsd = _stateprep.uccsd
uccgsd = _stateprep.uccgsd
upccgsd = _stateprep.upccgsd
ceo = _stateprep.ceo
apply_givens_rotation = _stateprep.apply_givens_rotation
apply_phase_givens_rotation = _stateprep.apply_phase_givens_rotation
prepare_complex_slater_determinant = _stateprep.prepare_complex_slater_determinant
prepare_slater_determinant = _stateprep.prepare_slater_determinant

GivensRotation = _stateprep.GivensRotation
GivensRotationSchedule = _stateprep.GivensRotationSchedule
SlaterDeterminantPlan = _stateprep.SlaterDeterminantPlan
GivensStatePrepResources = _stateprep.GivensStatePrepResources

get_uccsd_excitations = _stateprep.get_uccsd_excitations
get_num_uccsd_parameters = _stateprep.get_num_uccsd_parameters
get_uccgsd_pauli_lists = _stateprep.get_uccgsd_pauli_lists
get_upccgsd_pauli_lists = _stateprep.get_upccgsd_pauli_lists
get_ceo_pauli_lists = _stateprep.get_ceo_pauli_lists
get_givens_rotation_indices = _stateprep.get_givens_rotation_indices
get_givens_rotation_phases = _stateprep.get_givens_rotation_phases
get_givens_rotation_angles = _stateprep.get_givens_rotation_angles
validate_slater_determinant_plan = _stateprep.validate_slater_determinant_plan
estimate_givens_stateprep_resources = (
    _stateprep.estimate_givens_stateprep_resources)
estimate_givens_rotation_schedule_resources = (
    _stateprep.estimate_givens_rotation_schedule_resources)

make_uccsd_operator_pool = _stateprep.make_uccsd_operator_pool
make_uccgsd_operator_pool = _stateprep.make_uccgsd_operator_pool
make_upccgsd_operator_pool = _stateprep.make_upccgsd_operator_pool
make_ceo_operator_pool = _stateprep.make_ceo_operator_pool

_make_givens_rotation_schedule = _stateprep.make_givens_rotation_schedule
_make_complex_givens_rotation_schedule = (
    _stateprep.make_complex_givens_rotation_schedule)
_make_slater_determinant_plan = _stateprep.make_slater_determinant_plan
_make_complex_slater_determinant_plan = (
    _stateprep.make_complex_slater_determinant_plan)


def _contains_complex(values):
    if isinstance(values, complex):
        return True
    if isinstance(values, (str, bytes)):
        return False
    try:
        return any(_contains_complex(value) for value in values)
    except TypeError:
        return False


def _is_complex_input(values):
    is_complex_array = (hasattr(values, "dtype")
                        and getattr(values.dtype, "kind", None) == "c")
    if hasattr(values, "tolist"):
        values = values.tolist()
    return values, is_complex_array or _contains_complex(values)


def make_givens_rotation_schedule(occupied_orbitals, tolerance=1.0e-12):
    occupied_orbitals, is_complex = _is_complex_input(occupied_orbitals)
    if is_complex:
        return _make_complex_givens_rotation_schedule(occupied_orbitals,
                                                      tolerance)
    return _make_givens_rotation_schedule(occupied_orbitals, tolerance)


def make_slater_determinant_plan(occupied_orbitals, tolerance=1.0e-12):
    occupied_orbitals, is_complex = _is_complex_input(occupied_orbitals)
    if is_complex:
        return _make_complex_slater_determinant_plan(occupied_orbitals,
                                                     tolerance)
    return _make_slater_determinant_plan(occupied_orbitals, tolerance)


__all__ = [
    "uccsd",
    "uccgsd",
    "upccgsd",
    "ceo",
    "apply_givens_rotation",
    "apply_phase_givens_rotation",
    "prepare_complex_slater_determinant",
    "prepare_slater_determinant",
    "GivensRotation",
    "GivensRotationSchedule",
    "SlaterDeterminantPlan",
    "GivensStatePrepResources",
    "get_uccsd_excitations",
    "get_num_uccsd_parameters",
    "get_uccgsd_pauli_lists",
    "get_upccgsd_pauli_lists",
    "get_ceo_pauli_lists",
    "make_givens_rotation_schedule",
    "make_slater_determinant_plan",
    "validate_slater_determinant_plan",
    "estimate_givens_stateprep_resources",
    "estimate_givens_rotation_schedule_resources",
    "get_givens_rotation_indices",
    "get_givens_rotation_phases",
    "get_givens_rotation_angles",
    "make_uccsd_operator_pool",
    "make_uccgsd_operator_pool",
    "make_upccgsd_operator_pool",
    "make_ceo_operator_pool",
]
