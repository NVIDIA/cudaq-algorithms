"""Route scientific intent without embedding benchmark questions or answers."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "skills/cudaq-algorithms/scripts/route.py"
SPEC = importlib.util.spec_from_file_location("skill_route", SCRIPT)
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)


@pytest.mark.parametrize("prompt,required", [
    ("Given two geometries and a Gaussian basis, compare ground energies "
     "in a frozen-core active space.",
     {"chemistry/chemistry-bridges.md", "application-composition.md"}),
    ("For this spin chain, follow a domain-wall quench and determine when "
     "the local magnetization changes sign.",
     {"trotter/trotter.md", "application-composition.md"}),
    ("After discarding the ancilla, how does the density matrix differ "
     "from conditioning on success?",
     {"simulation/simulation-analysis.md"}),
    ("Use the supplied QSP phases for spectral filtering of a spin chain "
     "after a quench.", {"qsvt/qsvt.md"}),
    ("Construct a Krylov energy estimate from Chebyshev moments.",
     {"qubitization/qubitization.md"}),
    ("Map this hopping current with Bravyi-Kitaev.",
     {"fermion-transforms/fermion-transforms.md"}),
    ("Does a controlled SELECT require an adjoint block encoding?",
     {"block-encoding/block-encoding.md"}),
    ("Build a complex Slater determinant with Givens rotations.",
     {"state-preparation/state-preparation.md"}),
    ("Does UCCGSD supply an operator pool?",
     {"state-preparation/operator-pools.md"}),
])
def test_routes(prompt, required):
    records = router.route(prompt)
    assert required <= set(records)
    assert len(records) == len(set(records)) <= router.MAX_RECORDS
    assert all((router.REFS / record).is_file() for record in records)


@pytest.mark.parametrize("prompt", [
    "Install CUDA-Q and configure its GPU backend.",
    "Draw a Bell circuit with two qubits.",
    "Explain what a quantum computer is.",
])
def test_unmatched_requests_do_not_choose_a_science_family(prompt):
    assert router.route(prompt) == []


def test_named_methods_take_priority_over_default_dynamics():
    assert router.route(
        "Compare QSVT and Chebyshev walk predictions of an Ising quench."
    )[:2] == ["qsvt/qsvt.md", "qubitization/qubitization.md"]


def test_conditional_analysis_is_not_displaced_by_default_dynamics():
    records = router.route(
        "Use QSP phases for an Ising quench and report good_subspace success "
        "probability after discarding ancillas."
    )
    assert records[0] == "qsvt/qsvt.md"
    assert "simulation/simulation-analysis.md" in records
    assert "trotter/trotter.md" not in records


@pytest.mark.parametrize("prompt", [
    "Compute the spectrum in a computational basis.",
    "Compare Jordan-Wigner and Bravyi-Kitaev ground energies in an occupation basis.",
    "Compare the bond energies of a spin chain after a quench.",
])
def test_nonchemical_basis_and_bonds_do_not_select_chemistry(prompt):
    assert "chemistry/chemistry-bridges.md" not in router.route(prompt)


def test_atomic_inputs_can_select_the_molecular_chain_with_a_named_method():
    records = router.route(
        "Given these atoms and coordinates, use walk moments to estimate "
        "ground energies during bond stretching."
    )
    assert records == ["qubitization/qubitization.md",
                       "chemistry/chemistry-bridges.md", "application-composition.md"]


@pytest.mark.parametrize("prompt", [
    "Prepare two ordered correlated states for a Hubbard chain and compare energies.",
    "Measure persistent current of a hopping model using Jordan-Wigner.",
    r"Find conditional magnetization for a heralded input written as \bigotimes_j psi_j.",
])
def test_static_state_tasks_do_not_default_to_time_evolution(prompt):
    assert "trotter/trotter.md" not in router.route(prompt)
