from dataclasses import dataclass

from cudaq_algorithms._pycudaq_algorithms import stateprep as _stateprep

uccsd = _stateprep.uccsd
uccgsd = _stateprep.uccgsd
upccgsd = _stateprep.upccgsd
hartree_fock = _stateprep.hartree_fock
hartree_fock_occupation = _stateprep.hartree_fock_occupation
fixed_parameter_ucc = _stateprep.fixed_parameter_ucc
ceo = _stateprep.ceo

HartreeFockResources = _stateprep.HartreeFockResources
_CppFixedParameterUCCPlan = _stateprep.FixedParameterUCCPlan
FixedParameterUCCResources = _stateprep.FixedParameterUCCResources


@dataclass(frozen=True)
class FixedParameterUCCPlan:
    num_qubits: int
    parameters: list[float]
    pauli_words: list
    coefficients: list[list[float]]


get_uccsd_excitations = _stateprep.get_uccsd_excitations
get_num_uccsd_parameters = _stateprep.get_num_uccsd_parameters
get_uccgsd_pauli_lists = _stateprep.get_uccgsd_pauli_lists
get_upccgsd_pauli_lists = _stateprep.get_upccgsd_pauli_lists
get_ceo_pauli_lists = _stateprep.get_ceo_pauli_lists

make_hartree_fock_occupation = _stateprep.make_hartree_fock_occupation
validate_hartree_fock_occupation = _stateprep.validate_hartree_fock_occupation
estimate_hartree_fock_resources = _stateprep.estimate_hartree_fock_resources
estimate_hartree_fock_occupation_resources = _stateprep.estimate_hartree_fock_occupation_resources


def make_fixed_parameter_ucc_plan(pauli_words,
                                  coefficients,
                                  parameters,
                                  num_qubits=0,
                                  coefficient_tolerance=1.0e-12):
    if coefficient_tolerance < 0.0:
        raise ValueError(
            "fixed_parameter_ucc error - coefficient tolerance must be non-negative."
        )
    if len(parameters) != len(pauli_words) or len(pauli_words) != len(
            coefficients):
        raise ValueError(
            "fixed_parameter_ucc error - parameters, Pauli-word groups, and coefficient groups must have the same length."
        )
    for word_group, coefficient_group in zip(pauli_words, coefficients):
        if len(word_group) != len(coefficient_group):
            raise ValueError(
                "fixed_parameter_ucc error - each Pauli-word group must match its coefficient group."
            )
    return FixedParameterUCCPlan(num_qubits=int(num_qubits),
                                 parameters=list(parameters),
                                 pauli_words=list(pauli_words),
                                 coefficients=list(coefficients))


make_fixed_parameter_uccsd_plan = _stateprep.make_fixed_parameter_uccsd_plan
make_fixed_parameter_uccgsd_plan = _stateprep.make_fixed_parameter_uccgsd_plan
make_fixed_parameter_upccgsd_plan = _stateprep.make_fixed_parameter_upccgsd_plan


def validate_fixed_parameter_ucc_plan(plan, coefficient_tolerance=1.0e-12):
    # Accept both the C++ plan (from the uccsd/uccgsd/upccgsd makers) and the
    # Python plan returned by make_fixed_parameter_ucc_plan, so all the helpers
    # compose. (The C++-bound validator only accepts the C++ plan type.)
    if isinstance(plan, _CppFixedParameterUCCPlan):
        return _stateprep.validate_fixed_parameter_ucc_plan(
            plan, coefficient_tolerance)
    if coefficient_tolerance < 0.0:
        raise ValueError(
            "fixed_parameter_ucc error - coefficient tolerance must be non-negative."
        )
    if len(plan.parameters) != len(plan.pauli_words) or len(
            plan.pauli_words) != len(plan.coefficients):
        raise ValueError(
            "fixed_parameter_ucc error - parameters, Pauli-word groups, and coefficient groups must have the same length."
        )
    for word_group, coefficient_group in zip(plan.pauli_words,
                                             plan.coefficients):
        if len(word_group) != len(coefficient_group):
            raise ValueError(
                "fixed_parameter_ucc error - each Pauli-word group must match its coefficient group."
            )


def estimate_fixed_parameter_ucc_resources(plan):
    if isinstance(plan, _CppFixedParameterUCCPlan):
        return _stateprep.estimate_fixed_parameter_ucc_resources(plan)
    resources = FixedParameterUCCResources()
    resources.num_qubits = plan.num_qubits
    resources.num_excitations = len(plan.pauli_words)
    resources.num_pauli_rotations = sum(
        len(group) for group in plan.pauli_words)
    resources.max_pauli_rotations_per_excitation = max(
        (len(group) for group in plan.pauli_words), default=0)
    return resources


make_uccsd_operator_pool = _stateprep.make_uccsd_operator_pool
make_uccgsd_operator_pool = _stateprep.make_uccgsd_operator_pool
make_upccgsd_operator_pool = _stateprep.make_upccgsd_operator_pool
make_ceo_operator_pool = _stateprep.make_ceo_operator_pool

__all__ = [
    "uccsd",
    "uccgsd",
    "upccgsd",
    "hartree_fock",
    "hartree_fock_occupation",
    "fixed_parameter_ucc",
    "ceo",
    "HartreeFockResources",
    "FixedParameterUCCPlan",
    "FixedParameterUCCResources",
    "get_uccsd_excitations",
    "get_num_uccsd_parameters",
    "get_uccgsd_pauli_lists",
    "get_upccgsd_pauli_lists",
    "get_ceo_pauli_lists",
    "make_hartree_fock_occupation",
    "validate_hartree_fock_occupation",
    "estimate_hartree_fock_resources",
    "estimate_hartree_fock_occupation_resources",
    "make_fixed_parameter_ucc_plan",
    "make_fixed_parameter_uccsd_plan",
    "make_fixed_parameter_uccgsd_plan",
    "make_fixed_parameter_upccgsd_plan",
    "validate_fixed_parameter_ucc_plan",
    "estimate_fixed_parameter_ucc_resources",
    "make_uccsd_operator_pool",
    "make_uccgsd_operator_pool",
    "make_upccgsd_operator_pool",
    "make_ceo_operator_pool",
]
