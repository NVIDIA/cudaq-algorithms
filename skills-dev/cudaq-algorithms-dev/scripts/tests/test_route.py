# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route scientific intent without embedding benchmark questions or answers."""
import importlib.util
import subprocess
import sys
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
    ("Build an alias sampling PREPARE for these LCU coefficients.", {
        "primitives/primitives.md", "primitives/primitives-alias-sampling.md",
        "block-encoding/block-encoding.md"
    }),
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
    ("Prepare an LCU encoding from the supplied coefficients.",
     "primitives/primitives-alias-sampling.md"),
])
def test_generic_protocol_words_do_not_cross_route(prompt, excluded):
    assert excluded not in router.route(prompt)


def test_phase_estimation_gets_a_building_block_note():
    assert any("not a packaged primitive" in note
               for note in router.notes("Run QPE on this Hamiltonian."))
    assert router.notes("Construct a Krylov energy estimate.") == []


@pytest.mark.parametrize("token,expected", [
    ("LiH", True),
    ("H2", True),
    ("H2O", True),
    ("N2", True),
    ("CH4", True),
    ("QSVT", False),
    ("UCC", False),
    ("HF", False),
    ("H", False),
    ("H0", False),
    ("H1", False),
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


@pytest.mark.parametrize("prompt", [
    "Find the lowest energy of a two-site Heisenberg dimer.",
    "Find the ground state of an XXZ dimer in a fixed magnetization sector.",
])
def test_spin_dimers_select_the_lattice_energy_chain(prompt):
    records = router.route(prompt)
    assert records == [
        "qubitization/qubitization.md", "application-composition.md"
    ]


@pytest.mark.parametrize("label", ["H0", "H1"])
def test_hamiltonian_labels_do_not_add_a_molecular_chain(label):
    records = router.route(
        f"Estimate the ground energy of {label} using Chebyshev moments.")
    assert records == ["qubitization/qubitization.md"]


@pytest.mark.parametrize("prompt", [
    "Find the ground energy of an argon dimer.",
    "Compute the binding energy of helium dimers.",
    "Find the ground energy of a copper-dimer.",
    "Find the binding energy of an argon dimer in a Gaussian basis.",
    "Compare ground energies of a molecular dimer along a bond stretch.",
    "Find the ground energy of He2.",
    "Find the ground energy of H2.",
    "Find the ground energy of H2O.",
    "Find the ground energy of LiH.",
])
def test_molecular_context_retains_the_chemistry_chain(prompt):
    assert router.route(prompt) == [
        "chemistry/chemistry-bridges.md", "application-composition.md"
    ]


def test_dimer_without_physical_context_does_not_choose_a_family():
    assert router.route("Find the dimer ground energy.") == []


def test_named_method_precedes_the_spin_dimer_default():
    records = router.route(
        "Use QSVT to filter the ground state of a Heisenberg dimer.")
    assert records[0] == "qsvt/qsvt.md"
    assert "chemistry/chemistry-bridges.md" not in records


@pytest.mark.parametrize("prompt", [
    "Build alias-sampling PREPARE for an LCU block encoding.",
    "Use AliasSamplingPrepare for the PauliLCU coefficient distribution.",
    "Use AliasSamplingPrepare for LCU state preparation.",
    "Build alias-sampling PREPARE and inject it into an LCU block encoding.",
])
def test_named_alias_prepare_keeps_its_focused_record(prompt):
    records = router.route(prompt)
    assert set(records) == {
        "block-encoding/block-encoding.md", "primitives/primitives.md",
        "primitives/primitives-alias-sampling.md"
    }
    assert len(records) <= router.MAX_RECORDS


@pytest.mark.parametrize("prompt", [
    "Inject a state preparation kernel into an LCU block encoding.",
    "Prepare a Slater determinant and inject it into the walk kernel.",
])
def test_generic_state_injection_retains_its_contract(prompt):
    records = router.route(prompt)
    assert records[0] == "state-preparation/state-preparation.md"
    assert "state-preparation/injection-contract.md" in records
    assert "primitives/primitives.md" not in records


def test_named_method_precedes_alias_prepare_with_state_injection():
    records = router.route(
        "Use QSVT with alias-sampling PREPARE and inject a state.")
    assert records == [
        "qsvt/qsvt.md", "primitives/primitives.md",
        "primitives/primitives-alias-sampling.md"
    ]


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


def route_cli(prompt, *options):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *options, prompt],
        check=True,
        capture_output=True,
        text=True).stdout


@pytest.mark.parametrize("prompt,first", [
    ("Compare occupation accuracy across Trotter step counts.",
     "trotter/trotter.md"),
    ("Compare excitation gaps after explicit factorization of ERIs.",
     "double-factorization/double-factorization.md"),
    ("Use QSVT for an occupation observable with precision and statevector checks.",
     "qsvt/qsvt.md"),
])
def test_observable_words_do_not_select_state_preparation(prompt, first):
    records = router.route(prompt)
    assert records[0] == first
    assert "state-preparation/state-preparation.md" not in records


@pytest.mark.parametrize("prompt", [
    "Prepare an occupation basis state and inject it into Trotter.",
    "Build a double excitation kernel for state preparation.",
    "Use make_hartree_fock_occupation to construct a reference determinant.",
])
def test_actual_occupation_and_excitation_preparation_is_retained(prompt):
    assert "state-preparation/state-preparation.md" in router.route(prompt)


@pytest.mark.parametrize("prompt", [
    "Prepare a Bell state and sample its counts.",
    "Show a standalone Bell kernel and inspect its statevector.",
    "Write a basic CUDA-Q kernel to prepare two qubits and sample shots.",
])
def test_standalone_kernel_scope_is_not_changed_by_analysis_words(prompt):
    assert router.route(prompt) == []
    output = route_cli(prompt, "--full")
    assert "outside" in output.lower()
    assert "## references/" not in output
    assert "Selectable operation contracts" not in output


@pytest.mark.parametrize("prompt,required", [
    ("Inject a Bell state into Trotter and inspect its statevector.",
     "trotter/trotter.md"),
    ("Apply PauliLCU to a Bell input and measure good_subspace probability.",
     "block-encoding/block-encoding.md"),
    ("Use sim_utils.evolve on a Bell state.", "trotter/trotter.md"),
    ("Prepare a Bell input for a spin chain quench.", "trotter/trotter.md"),
    ("Use chemistry.qubit_hamiltonian and evaluate it on a Bell input.",
     "chemistry/chemistry-bridges.md"),
])
def test_bell_inputs_to_packaged_operations_stay_in_scope(prompt, required):
    assert required in router.route(prompt)


@pytest.mark.parametrize("symbol,family,focused", [
    ("explicit_double_factorization",
     "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-explicit.md"),
    ("compressed_double_factorization",
     "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-compressed.md"),
    ("reconstruct_eri", "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-reconstruction.md"),
    ("factorization_error", "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-error.md"),
    ("modified_one_body_integrals",
     "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-modified-one-body.md"),
    ("double_factorization_one_norm",
     "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-one-norm.md"),
    ("TrotterOrdering", "trotter/trotter.md", "trotter/trotter-planning.md"),
    ("make_trotter_terms", "trotter/trotter.md",
     "trotter/trotter-planning.md"),
    ("Trotter.kernel", "trotter/trotter.md",
     "trotter/trotter-kernel-factory.md"),
    ("Trotter.state_kernel", "trotter/trotter.md",
     "trotter/trotter-state-kernel-factory.md"),
    ("apply_trotter", "trotter/trotter.md", "trotter/trotter-apply-kernel.md"),
    ("Trotter.resources", "trotter/trotter.md",
     "trotter/trotter-resources-planned.md"),
    ("estimate_trotter_resources", "trotter/trotter.md",
     "trotter/trotter-resources-raw.md"),
    ("sim_utils.evolve", "trotter/trotter.md",
     "simulation/simulation-evolve.md"),
    ("sim.evolve", "trotter/trotter.md", "simulation/simulation-evolve.md"),
])
def test_named_operations_get_direct_focused_paths(symbol, family, focused):
    prompt = f"Use {symbol}."
    assert router.route(prompt)[0] == family
    assert str(router.REFS / focused) in route_cli(prompt)


@pytest.mark.parametrize("prompt", [
    "Use explicit Cholesky double factorization to compare truncation errors.",
    "Compare double factorization with explicit leaves across truncation thresholds.",
])
def test_named_factorization_operation_does_not_require_symbol_spelling(
        prompt):
    assert str(
        router.REFS /
        "double-factorization/double-factorization-explicit.md") in route_cli(
            prompt)


@pytest.mark.parametrize("primitive,focused", [
    ("QROM", "primitives/primitives-qrom.md"),
    ("alias-sampling PREPARE", "primitives/primitives-alias-sampling.md"),
])
def test_named_primitive_focus_survives_starting_guide_limit(
        primitive, focused):
    prompt = f"Use QSVT and a Chebyshev walk with {primitive}."
    assert len(router.route(prompt)) == router.MAX_RECORDS == 3
    assert str(router.REFS / focused) in route_cli(prompt)


@pytest.mark.parametrize("prompt", [
    "Compare Trotter evolution across grouping orders and step counts.",
    "Compare explicit double factorization leaves using energy gaps.",
    "Does Trotter evolution conserve the particle number?",
])
def test_scientific_operation_does_not_suggest_unasked_boundary_claims(prompt):
    output = route_cli(prompt)
    assert "Candidate premise rows" not in output


@pytest.mark.parametrize("prompt,claim", [
    ("Does Trotter.resources report hardware depth?",
     "formula-level logical proxies"),
    ("Are explicit and compressed factorization interchangeable?",
     "L-BFGS-B optimization"),
])
def test_boundary_questions_retain_relevant_premise_rows(prompt, claim):
    assert claim in route_cli(prompt)


def test_named_operation_hint_precedes_related_premise_hints():
    output = route_cli("Does Trotter.state_kernel apply the identity phase?")
    assert output.index(
        str(router.REFS /
            "trotter/trotter-state-kernel-factory.md")) < output.index(
                "Candidate premise rows")


def test_shared_checkpoints_do_not_consume_starting_guides():
    prompt = "Compare QSVT and a Chebyshev walk for an Ising quench."
    assert len(router.route(prompt)) == router.MAX_RECORDS
    output = route_cli(prompt)
    assert str(router.REFS / "workflow.md") in output
    assert str(router.REFS / "validation.md") in output
    assert output.index(str(router.REFS / "workflow.md")) > output.index(
        str(router.REFS / router.route(prompt)[-1]))


def test_shared_checkpoint_paths_are_covered_by_record_existence_check():
    assert {"workflow.md", "validation.md"} <= set(router.all_records())


@pytest.mark.parametrize("prompt", [
    "Apply a PauliLCU block to supplied input states and report acceptance "
    "probability.",
    "Apply PauliLCU to a supplied ket and report the accepted-branch probability.",
    "Validate sim_utils.action on unequal complex product-state factors "
    "assigned to each site.",
    "Check tensor-product qubit ordering of an input statevector used by Walk.",
    "Compute QSVT action on entangled input-state amplitudes.",
    "Compare Trotter evolution of a prepared state.",
])
def test_input_state_workflows_expose_ordering_checkpoint(prompt):
    output = route_cli(prompt)
    assert str(router.REFS / "conventions/ordering.md") in output
    assert "#product-state-amplitudes" in output


def test_input_checkpoint_preserves_full_starting_guide_budget():
    prompt = "Compare QSVT and a Chebyshev walk on a supplied input statevector."
    records = router.route(prompt)
    assert records == [
        "qsvt/qsvt.md", "qubitization/qubitization.md",
        "simulation/simulation-analysis.md"
    ]
    output = route_cli(prompt)
    assert output.count("## references/") == 3
    assert output.index(str(
        router.REFS / "conventions/ordering.md")) > output.index(
            str(router.REFS / records[-1]))


@pytest.mark.parametrize("prompt", [
    "Show a standalone Bell kernel and inspect its statevector.",
    "Which kernel signatures must a block encoding provider implement?",
    "What register width does a PauliLCU encoding require for qubits on "
    "separated sites?",
    "Estimate Trotter resources for a qubit register.",
    "Compare double factorization leaf budgets.",
])
def test_non_input_workflows_do_not_add_ordering_checkpoint(prompt):
    assert str(router.REFS / "conventions/ordering.md") not in route_cli(prompt)


def test_input_checkpoint_path_has_record_existence_coverage():
    assert "conventions/ordering.md" in router.all_records()
