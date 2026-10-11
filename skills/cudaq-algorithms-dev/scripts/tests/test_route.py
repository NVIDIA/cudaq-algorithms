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
     "from conditioning on success?", {"simulation/simulation-analysis.md"}),
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
    ("Use QROM for the SELECT of a PauliLCU block encoding.",
     {"primitives/primitives.md", "block-encoding/block-encoding.md"}),
    ("Add a constant with a ripple-carry adder and compare two registers.",
     {"primitives/primitives.md"}),
    ("Use unary iteration to apply a per-address body.",
     {"primitives/primitives.md"}),
    ("Build an alias sampling PREPARE for these LCU coefficients.",
     {"primitives/primitives.md", "block-encoding/block-encoding.md"}),
    ("What is the ground state energy of the 2D Fermi-Hubbard model at "
     "half filling?",
     {"qubitization/qubitization.md", "application-composition.md"}),
    ("Ground state energy of H2O in STO-3G.",
     {"chemistry/chemistry-bridges.md", "application-composition.md"}),
    ("Run quantum phase estimation on LiH to get the ground state energy.",
     {"chemistry/chemistry-bridges.md", "qubitization/qubitization.md"}),
    ("Why does my fp32 nvidia target miss the 1e-12 tolerance that fp64 meets?",
     {"conventions/numerical-comparison.md"}),
    ("The kernel rejects a ternary and an empty captured list; can I use "
     "cudaq.adjoint instead?", {"conventions/kernel-boundaries.md"}),
    ("Price the select_swap QROM and its toffoli_count.",
     {"primitives/primitives.md", "primitives/primitives-qrom.md"}),
    ("Compose cmp_ge_register with add_constant_qft on one register.",
     {"primitives/primitives.md", "primitives/primitives-arithmetic.md"}),
    ("Build an AliasSamplingPrepare with mu bits and check its "
     "discretization bound.",
     {"primitives/primitives.md", "primitives/primitives-alias-sampling.md"}),
])
def test_routes(prompt, required):
    records = router.route(prompt)
    assert required <= set(records)
    assert len(records) == len(set(records)) <= router.MAX_RECORDS
    assert all((router.REFS / record).is_file() for record in records)


def test_every_table_record_exists():
    # route() does not filter missing records, so a renamed or moved
    # reference fails here instead of dropping silently from the output.
    missing = [
        record for record in router.all_records()
        if not (router.REFS / record).is_file()
    ]
    assert missing == []


@pytest.mark.parametrize("prompt,excluded", [
    ("How do I use the controlled walk step kernel of a block encoding in "
     "my own kernel?", "state-preparation/state-preparation.md"),
    ("What sign convention does the qubitization walk use for the block?",
     "qsvt/qsvt.md"),
    ("Compute the Givens rotation angles to prepare a Slater determinant.",
     "qsvt/qsvt.md"),
    ("Is Trotter or QSVT the better speedup for this spin chain evolution?",
     "double-factorization/double-factorization.md"),
    ("Apply the QFT to the counting register of the walk.",
     "primitives/primitives.md"),
    ("Does a controlled SELECT require an adjoint block encoding?",
     "conventions/kernel-boundaries.md"),
])
def test_generic_protocol_words_do_not_cross_route(prompt, excluded):
    assert excluded not in router.route(prompt)


def test_phase_estimation_gets_a_building_block_note():
    assert any("not a packaged primitive" in note
               for note in router.notes("Run QPE on this Hamiltonian."))
    assert router.notes("Construct a Krylov energy estimate.") == []


@pytest.mark.parametrize("token,expected", [
    ("LiH", True),
    ("H2O", True),
    ("N2", True),
    ("CH4", True),
    ("QSVT", False),
    ("UCC", False),
    ("HF", False),
    ("H", False),
    ("Be", False),
])
def test_chemical_formula_recognition(token, expected):
    assert router.is_chemical_formula(token) is expected


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
        "probability after discarding ancillas.")
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
        "ground energies during bond stretching.")
    assert records == [
        "qubitization/qubitization.md", "chemistry/chemistry-bridges.md",
        "application-composition.md"
    ]


@pytest.mark.parametrize("prompt", [
    "Prepare two ordered correlated states for a Hubbard chain and compare energies.",
    "Measure persistent current of a hopping model using Jordan-Wigner.",
    r"Find conditional magnetization for a heralded input written as \bigotimes_j psi_j.",
])
def test_static_state_tasks_do_not_default_to_time_evolution(prompt):
    assert "trotter/trotter.md" not in router.route(prompt)
