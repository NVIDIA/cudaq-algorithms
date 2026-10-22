# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route scientific intent without embedding benchmark questions or answers."""
import importlib.util
import re
import shutil
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


@pytest.mark.parametrize("prompt", [
    "Prepare a Slater determinant ground state for a hopping model and "
    "compute its energy and coherences.",
    "Construct a Slater determinant for a spin lattice, then measure its "
    "energy expectation and variance.",
    "Build a Hartree-Fock reference for a Hubbard chain and evaluate its energy.",
    "Prepare a state with Givens rotations and measure its hopping energy.",
    "Construct a UCC trial state for a Hubbard lattice and report its energy.",
])
def test_preparing_a_known_state_does_not_infer_an_energy_solver(prompt):
    records = router.route(prompt)
    assert records[0] == "state-preparation/state-preparation.md"
    assert "qubitization/qubitization.md" not in records
    output = route_cli(prompt)
    assert f"### {router.REFS / 'state-preparation/state-preparation.md'}:" in output
    assert f"### {router.REFS / 'qubitization/qubitization.md'}:" not in output


@pytest.mark.parametrize("outcome", [
    "estimate the interacting ground energy",
    "find the actual ground state",
    "compute the excitation spectrum",
    "estimate the lowest eigenvalue",
    "compare spectral gaps",
    "estimate the lowest energy of the full Hamiltonian",
    "calculate its excited-state energies",
])
def test_preparation_retains_a_separate_spectral_estimation_stage(outcome):
    records = router.route(
        f"Prepare a Slater reference for a Hubbard lattice, then {outcome}.")
    assert "state-preparation/state-preparation.md" in records
    assert "qubitization/qubitization.md" in records


@pytest.mark.parametrize("method,family", [
    ("Chebyshev moments", "qubitization/qubitization.md"),
    ("QPE", "qubitization/qubitization.md"),
    ("QSP", "qsvt/qsvt.md"),
    ("Trotter", "trotter/trotter.md"),
])
def test_preparation_diagnostics_preserve_requested_downstream_methods(
        method, family):
    records = router.route(
        "Prepare a Slater determinant for a hopping model, "
        f"then use {method} and compute its energy and coherences.")
    assert "state-preparation/state-preparation.md" in records
    assert family in records


@pytest.mark.parametrize("prompt", [
    "Use the supplied Slater reference to estimate the Hubbard ground energy.",
    "Do not prepare a Slater reference; find the Hubbard ground state.",
    "Prepare a report of the Hubbard spectrum from a Slater reference.",
    "Build a reference table of Hartree-Fock energies for a Hubbard model.",
    "Prepare a report about Hartree-Fock states and Hubbard energies.",
])
def test_existing_references_and_spectral_reports_keep_energy_routing(prompt):
    assert router.route(prompt)[0] == "qubitization/qubitization.md"


def test_diagnostics_do_not_hide_a_later_lowest_energy_request():
    records = router.route(
        "Prepare a Slater state for a Hubbard model and measure its energy "
        "expectation, then estimate the lowest energy of the full Hamiltonian."
    )
    assert "qubitization/qubitization.md" in records


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
    "Prepare correlated trial states using fermionic excitation operators "
    "and Jordan-Wigner, then measure Hubbard energies.",
    "Construct two correlated states with single and double excitations; "
    "use Jordan-Wigner to compare lattice energy expectations.",
    "Build ordered preparations from fermionic excitation generators and "
    "use Bravyi-Kitaev to evaluate spin-lattice energies.",
    "Apply the supplied fermionic excitation operators to a reference state "
    "and compare Hubbard energies using Jordan-Wigner.",
    "Compose double-excitation rotations into a correlated state and "
    "measure its Hubbard energy using Jordan-Wigner.",
    "Prepare correlated states by applying fermionic excitation operators "
    "and use Jordan-Wigner to compare Hubbard energy expectations.",
])
def test_excitation_preparation_precedes_default_energy_under_guide_cap(
        prompt):
    records = router.route(prompt)
    preparation = "state-preparation/state-preparation.md"
    transform = "fermion-transforms/fermion-transforms.md"
    assert records[:2] == [transform, preparation]
    assert len(records) == len(set(records)) <= router.MAX_RECORDS
    if "qubitization/qubitization.md" in records:
        assert records.index(preparation) < records.index(
            "qubitization/qubitization.md")
    output = route_cli(prompt)
    assert "For an **ordered correlated preparation**" in output
    assert "Requested task completion" in output


@pytest.mark.parametrize("prompt", [
    "Compute fermionic excitation energies for a Hubbard model using "
    "Jordan-Wigner.",
    "Use Jordan-Wigner to compare the single-excitation spectrum and "
    "double-excitation gaps of a lattice Hamiltonian.",
    "Measure Hubbard energy on an already prepared correlated input state "
    "obtained from fermionic excitation operators, using Jordan-Wigner.",
    "The supplied state was prepared with fermionic excitation operators. "
    "Compute its Hubbard energy using Jordan-Wigner.",
    "Prepare a spectral report of fermionic excitation energies in a "
    "Hubbard model using Jordan-Wigner.",
    "Use QSP to prepare the ground state from a Hubbard excitation-energy "
    "spectrum and Jordan-Wigner coefficients.",
    "Build a spectrum of the already prepared state with fermionic "
    "excitation operators using Jordan-Wigner for a Hubbard model.",
    "Do not prepare a correlated state with fermionic excitation operators; "
    "measure Hubbard energy on the supplied input using Jordan-Wigner.",
    "Do not apply fermionic excitation operators to the supplied state; "
    "measure its Hubbard energy using Jordan-Wigner.",
    "Do not actually prepare a state using fermionic excitations; "
    "compare Hubbard excitation energies using Jordan-Wigner.",
    "There is no need to prepare a state with fermionic excitations; "
    "just compare Hubbard excitation energies using Jordan-Wigner.",
])
def test_excitation_spectra_and_existing_inputs_do_not_promote_preparation(
        prompt):
    assert "state-preparation/state-preparation.md" not in router.route(prompt)


@pytest.mark.parametrize("method,family", [
    ("QSP", "qsvt/qsvt.md"),
    ("QSVT", "qsvt/qsvt.md"),
    ("Chebyshev moments", "qubitization/qubitization.md"),
    ("Trotter", "trotter/trotter.md"),
    ("PauliLCU block encoding", "block-encoding/block-encoding.md"),
])
def test_named_method_survives_composed_excitation_preparation(method, family):
    records = router.route(
        "Prepare a correlated state with fermionic excitation operators "
        f"using Jordan-Wigner, then use {method} for Hubbard energy estimates."
    )
    assert records == [
        family, "fermion-transforms/fermion-transforms.md",
        "state-preparation/state-preparation.md"
    ]


def test_declined_preparation_does_not_hide_a_later_positive_operation():
    records = router.route(
        "There is no need to prepare a state with single excitations; "
        "instead prepare a correlated state with double excitations and "
        "compare Hubbard energies using Jordan-Wigner.")
    assert records[:2] == [
        "fermion-transforms/fermion-transforms.md",
        "state-preparation/state-preparation.md"
    ]


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


def test_named_recovery_with_bell_input_keeps_its_algorithm_contract():
    prompt = "Use recover_real_time_evolution with a Bell input state."
    assert not router.standalone_kernel(prompt)
    assert "qsvt/qsvt.md" in router.route(prompt)
    assert "outside this skill" not in route_cli(prompt, "--paths")


@pytest.mark.parametrize("name", ["QPE", "quantum phase estimation"])
def test_phase_estimation_with_bell_input_retains_building_block_note(name):
    output = route_cli(f"Use {name} with a Bell input state.", "--paths")
    assert "not a packaged primitive" in output
    assert "walk-step kernels" in output
    assert "outside this skill" not in output


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
    assert len(re.findall(r"(?m)^## references/", output)) == 3
    assert output.index(str(router.REFS /
                            "conventions/ordering.md")) > output.index(
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
    assert str(router.REFS /
               "conventions/ordering.md") not in route_cli(prompt)


def test_input_checkpoint_path_has_record_existence_coverage():
    assert "conventions/ordering.md" in router.all_records()


@pytest.mark.parametrize("prompt", [
    "How do I connect a state constructor, encoded operator and spectral "
    "transformation? Explain whether their interfaces are composable.",
    "Check compatibility of several FTQC building blocks before implementation.",
    "Which interfaces must agree when composing quantum primitives?",
    "Explain whether state preparation, encoded operators, and spectral "
    "transformations have compatible interfaces.",
    "How can I compose quantum primitives with state preparation?",
])
def test_abstract_composition_exposes_checks_without_selecting_an_algorithm(
        prompt):
    assert router.route(prompt) == ["application-composition.md"]
    output = route_cli(prompt)
    assert "execution layer" in output
    assert "Molecular structure to an energy estimate" not in output


def test_default_delivers_relevant_sections_without_the_whole_guide():
    prompt = "Construct an HF preparation for a molecular qubit Hamiltonian."
    output = route_cli(prompt)
    assert "Selected reference text" in output
    assert "Requested task completion" in output
    assert "## Workflow" in output
    assert "Resource estimator routing" not in output.split(
        "Selected reference text", 1)[1]
    assert len(output) < 20000


def test_paths_option_retains_navigation_without_loading_reference_bodies():
    output = route_cli("Compare Trotter evolution and its circuit cost.",
                       "--paths")
    assert str(router.REFS / "trotter/trotter.md") in output
    assert "Selected reference text" not in output
    assert "1. Specify the Hamiltonian" not in output


def test_section_excerpts_respect_code_fences_and_expose_omissions(
        tmp_path, monkeypatch):
    # A large example must not crowd out the later validation instructions,
    # and Markdown-looking lines inside code must not become section headings.
    record = tmp_path / "test.md"
    record.write_text(
        "# Guide\n\n## Workflow\n\nRead this complete paragraph.\n\n"
        "```python\n## Verification\n" + "x = 'example'\n" * 1000 +
        "```\n\n## Verification\n\nCheck the actual result.\n\n"
        "## Unrelated\n\nDO NOT LOAD THIS SECTION\n")
    monkeypatch.setattr(router, "REFS", tmp_path)
    output = router.reference_context(["test.md"],
                                      "Compute a result",
                                      limit=1600)
    assert "Read this complete paragraph." in output
    assert "Check the actual result." in output
    assert "DO NOT LOAD THIS SECTION" not in output
    assert "x = 'example'" not in output
    assert "omitted" in output.lower()
    assert "### references/test.md:3" in output
    assert len(output) <= 1600


@pytest.mark.parametrize("prompt", [
    "Connect two web services and check whether their interfaces are compatible.",
    "Explain musical composition.",
])
def test_composition_vocabulary_alone_does_not_select_quantum_guides(prompt):
    assert router.route(prompt) == []


def test_supplied_filter_method_does_not_load_a_default_energy_algorithm():
    prompt = "Use the supplied QSP phases to improve low-energy weight in a Heisenberg chain."
    records = router.route(prompt)
    assert "qsvt/qsvt.md" in records
    assert "qubitization/qubitization.md" not in records
    assert "Krylov" not in route_cli(prompt).split("Selected reference text",
                                                   1)[1]


def test_named_hf_preparation_loads_its_contract_without_a_givens_recipe():
    output = route_cli(
        "Construct an HF preparation for a molecular qubit Hamiltonian.")
    assert str(
        router.REFS /
        "state-preparation/state-preparation-kernel-hartree-fock-occupation.md"
    ) in output
    assert "run_slater" not in output.split("Selected reference text", 1)[1]


def test_multi_stage_task_keeps_the_complete_deliverable_checklist():
    output = route_cli(
        "Using QSP phases, compare low-energy preparation for a spin chain and report whether the improvement is sufficient."
    )
    context = output.split("Selected reference text", 1)[1]
    assert "a feasibility bound alone does not define usefulness" in context


def test_hf_focus_preserves_other_explicit_preparation_stages():
    output = route_cli(
        "Prepare a Hartree-Fock reference, then apply a Givens rotation and a UCC ansatz."
    )
    assert "run_slater" in output.split("Selected reference text", 1)[1]


def test_installation_prefix_does_not_change_bounded_operation_content(
        tmp_path, monkeypatch):
    prompt = ("Prepare a Hartree-Fock reference, then apply a Givens rotation "
              "and a UCC ansatz.")
    records = router.route(prompt) + router.focused_records(prompt)
    original_refs = router.REFS
    contexts = []
    for prefix in (tmp_path / "short", tmp_path / ("long-root-" * 12)):
        refs = prefix / "cudaq-algorithms" / "references"
        shutil.copytree(original_refs, refs)
        monkeypatch.setattr(router, "REFS", refs)
        context = router.reference_context(records, prompt)
        contexts.append(context)
        assert len(context) <= router.MAX_CONTEXT_CHARS
        assert "run_slater" in context
        labels = re.findall(r"(?m)^### (references/[^:\n]+):(\d+)$", context)
        assert labels
        for relative, line in labels:
            source = refs.parent / relative
            assert source.is_file()
            assert 1 <= int(line) <= len(source.read_text().splitlines())
    assert contexts[0] == contexts[1]


def test_cli_resolves_relative_excerpt_labels_from_one_explicit_root():
    output = route_cli(
        "Prepare a Hartree-Fock reference, then apply Givens rotations.")
    root_line = re.search(r"(?m)^Skill directory: (.+)$", output)
    assert root_line
    assert root_line.start() < output.index("## Selected reference text")
    root = Path(root_line.group(1))
    assert root == router.REFS.parent
    labels = re.findall(r"(?m)^### (references/[^:\n]+):(\d+)$", output)
    assert labels
    assert all((root / relative).is_file() for relative, _ in labels)
    assert f"Open: {router.REFS / 'state-preparation/state-preparation.md'}" in output


def test_unused_excerpt_budget_is_reused_for_a_complete_contract_table(
        tmp_path, monkeypatch):
    (tmp_path /
     "family.md").write_text("## Workflow\n\nUse the selected operation.\n")
    table = "| Input | Requirement |\n| --- | --- |\n" + "| argument | must match the declared representation |\n" * 30
    (tmp_path / "focused.md").write_text("## Scientific contract\n\n" + table)
    monkeypatch.setattr(router, "REFS", tmp_path)
    output = router.reference_context(["family.md", "focused.md"],
                                      "Use a named operation",
                                      limit=2500)
    assert table.strip() in output
    assert len(output) <= 2500


@pytest.mark.parametrize("prompt", [
    "Design how molecular integrals, compressed factors and a dynamics "
    "estimator fit together; explain each interface before implementation.",
    "Outline a pipeline from chemist ERIs through a low-rank approximation "
    "to time-dependent observables. Discuss hand-offs; no code changes.",
    "I need an architecture connecting molecular integrals through a "
    "compressed representation to dynamics. Do not write code.",
])
def test_advisory_architecture_delivers_representation_and_method_options(
        prompt):
    output = route_cli(prompt)
    assert "Selected reference text" in output
    context = output.split("Selected reference text", 1)[1]
    assert "## Composition procedure" in context
    assert "| Compressed chemistry approximation |" in context
    assert "host-side factorization data, not a quantum kernel or block encoding" in context
    assert "| factorize / chemist ERI tensor by X-DF |" in context
    assert "| compress / chemist ERI tensor by C-DF or RC-DF |" in context
    assert "double-factorization/double-factorization-compressed.md" not in router.focused_records(
        prompt)


def test_advisory_named_methods_keep_exact_contracts_and_full_output_checks():
    prompt = (
        "Sketch the architecture joining compressed double factorization of "
        "molecular integrals with QSP phases, Trotter and a qubitization walk. "
        "Explain stage boundaries and validation without running code.")
    assert router.route(prompt) == [
        "qsvt/qsvt.md", "double-factorization/double-factorization.md",
        "trotter/trotter.md"
    ]
    context = router.reference_context(
        router.route(prompt) + router.focused_records(prompt), prompt)
    assert len(context) <= router.MAX_CONTEXT_CHARS
    assert "## Composition procedure" in context
    assert "host-side factorization data, not a quantum kernel or block encoding" in context
    assert "No packaged quantum kernel or end-to-end resource estimator is returned." in " ".join(
        context.split())
    assert 'default convention is `"qsvt"`' in context
    assert "sim.evolve(ev, ket, time, steps=3, order=2)" in context
    assert "double-factorization/double-factorization-explicit.md" not in router.focused_records(
        prompt)
    assert "Requested output artifacts" in context
    assert "payload at another destination does not establish this interface's validity" in " ".join(
        context.split())
    assert "## Molecular structure to an energy estimate" not in context


@pytest.mark.parametrize("prompt", [
    "Plan the interfaces for an integration of a state constructor, block "
    "encoding and QSVT; no execution, just validation contracts.",
    "Review the compatibility of Slater preparation, PauliLCU and a "
    "Chebyshev walk in a multi-stage design.",
])
def test_advisory_nonchemical_composition_avoids_unrelated_compression(prompt):
    output = route_cli(prompt)
    assert "## Composition procedure" in output
    assert "| Compressed chemistry approximation |" not in output
    assert "double-factorization/double-factorization.md" not in router.focused_records(
        prompt)


@pytest.mark.parametrize("prompt,required", [
    ("Sketch an architecture using explicit double factorization (X-DF) of "
     "molecular integrals followed by QSVT dynamics; explain stage contracts.",
     "double-factorization/double-factorization-explicit.md"),
    ("Sketch a reusable molecular workflow: double factorization, a block "
     "encoding and supplied QSP phases. Explain contracts before implementation.",
     "double-factorization/double-factorization.md"),
])
def test_advisory_respects_explicit_and_unspecified_factorization(
        prompt, required):
    focused = router.focused_records(prompt)
    assert required in focused
    assert "double-factorization/double-factorization-compressed.md" not in focused
    output = route_cli(prompt)
    assert "## Composition procedure" in output
    assert "host-side factorization data, not a quantum kernel or block encoding" in output


@pytest.mark.parametrize("prompt", [
    "Design interfaces from molecular integrals through a low-rank SVD "
    "approximation to a Trotter dynamics estimator.",
    "Outline the pipeline from molecular integrals through a low-rank "
    "tensor-train representation to QSVT dynamics.",
    "Plan a pipeline from molecular integrals to QSVT dynamics without "
    "double factorization or other compression.",
    "Explain the architecture for molecular integrals and QSVT dynamics; "
    "there is no need for compressed factors.",
])
def test_advisory_does_not_substitute_factorization_for_alternatives_or_negation(
        prompt):
    output = route_cli(prompt)
    assert "## Composition procedure" in output
    assert "| Compressed chemistry approximation |" not in output
    assert "double-factorization/double-factorization.md" not in router.focused_records(
        prompt)
    assert "double-factorization/double-factorization-compressed.md" not in router.focused_records(
        prompt)


def test_advisory_later_positive_compression_clause_remains_actionable():
    prompt = (
        "Design a pipeline for molecular integrals and dynamics. Do not "
        "compress the first input; but use compressed double factorization "
        "for the second input and explain the stage boundaries.")
    output = route_cli(prompt)
    assert "## Composition procedure" in output
    assert "| Compressed chemistry approximation |" in output
    assert "No packaged quantum kernel or end-to-end resource estimator is returned." in " ".join(
        output.split())


@pytest.mark.parametrize("prompt", [
    "Explain the architecture connecting molecular integrals, compressed "
    "factors and QSVT, then implement the pipeline.",
    "Benchmark QSVT and Chebyshev using a prepared statevector; include "
    "architecture diagrams in the final writeup.",
    "Implement the prescribed pipeline with QSVT, Trotter and a Chebyshev "
    "walk and report numerical errors.",
    "Explain GPU architecture and occupancy for a standalone CUDA-Q kernel.",
    "Plan interfaces for a classical data pipeline without quantum methods.",
    "Design lossless compressed NumPy storage for molecular integral arrays.",
    "Recommend an archive file format for molecular integrals without "
    "changing their values.",
])
def test_execution_and_incidental_architecture_do_not_load_advisory_contracts(
        prompt):
    assert "application-composition.md" not in router.focused_records(prompt)
    assert "## Composition procedure" not in route_cli(prompt)


@pytest.mark.parametrize("mode", [None, "--paths", "--full"])
def test_unmatched_eri_architecture_cli_modes_expose_the_same_records(mode):
    prompt = (
        "Outline a pipeline from chemist ERIs through a compressed "
        "representation to time-dependent observables; explain the hand-offs.")
    output = route_cli(prompt, *([mode] if mode else []))
    assert "No family matched" not in output
    assert "references/application-composition.md" in output
    assert str(router.REFS /
               "double-factorization/double-factorization.md") in output
    if mode == "--paths":
        assert "## Composition procedure" not in output
        assert "Selected reference text" not in output
    elif mode is None:
        assert "## Composition procedure" in output
        assert "host-side factorization data, not a quantum kernel or block encoding" in output
        for relative, line in re.findall(
                r"(?m)^### (references/[^:\n]+):(\d+)$", output):
            source = router.REFS.parent / relative
            assert source.is_file()
            assert 1 <= int(line) <= len(source.read_text().splitlines())


@pytest.mark.parametrize("restriction", [
    "avoid compression and use the full tensor",
    "skip low-rank approximations and retain the supplied integrals",
    "omit optional compression from this design",
    "exclude compressed double factorization",
    "avoid actually using compressed double factorization",
    "proceed without compressed double factorization",
])
def test_advisory_operation_negation_omits_df_contracts(restriction):
    prompt = (
        "Design the architecture from molecular integrals to QSVT dynamics; " +
        restriction + ".")
    focused = router.focused_records(prompt)
    assert "application-composition.md" in focused
    assert "double-factorization/double-factorization.md" not in focused
    assert "double-factorization/double-factorization-compressed.md" not in focused
    output = route_cli(prompt)
    assert "## Composition procedure" in output
    assert "| Compressed chemistry approximation |" not in output
    assert 'DoubleFactorization(method="C-DF")' not in output


@pytest.mark.parametrize("prompt", [
    "Explain a molecular-integral to QSVT pipeline without compressed "
    "double factorization.",
    "Apply QSVT to the supplied Hamiltonian without compressed double "
    "factorization.",
    "Compare QSP conventions; do not actually use C-DF.",
])
def test_negated_compressed_method_has_no_legacy_focused_recipe(prompt):
    assert "double-factorization/double-factorization-compressed.md" not in router.focused_records(
        prompt)
    output = route_cli(prompt)
    assert 'DoubleFactorization(method="C-DF")' not in output
    assert "## Scientific contract\n\nC-DF fixes" not in output


@pytest.mark.parametrize("operation", [
    "use compressed double factorization before QSVT",
    "compare X-DF and C-DF before a QSVT dynamics estimator",
    "avoid changing inputs when using compressed double factorization "
    "before QSVT",
    "skip changing the basis while using C-DF before QSVT",
    "do not change inputs when using C-DF before QSVT",
    "avoid Trotter and use compressed double factorization before QSVT",
    "do not apply Trotter and use compressed double factorization before QSVT",
    "omit low-rank compression for the first input; but use compressed "
    "double factorization for the second input before QSVT",
    "do not skip compressed double factorization before QSVT",
])
def test_affirmative_df_contract_survives_other_negated_operations(operation):
    prompt = "Explain the architecture for molecular integrals: " + operation + "."
    assert "double-factorization/double-factorization-compressed.md" in router.focused_records(
        prompt)
    output = route_cli(prompt)
    assert 'DoubleFactorization(method="C-DF")' in output
    assert "No packaged quantum kernel or end-to-end resource estimator is returned." in " ".join(
        output.split())


def test_explicit_df_focus_is_not_replaced_by_a_negated_compressed_method():
    prompt = (
        "Design a pipeline from molecular integrals to QSVT; avoid compressed "
        "double factorization and use explicit double factorization (X-DF).")
    focused = router.focused_records(prompt)
    assert "double-factorization/double-factorization-explicit.md" in focused
    assert "double-factorization/double-factorization-compressed.md" not in focused


@pytest.mark.parametrize("prompt,record", [
    ("Explain why not call reconstruct_eri on an arbitrary matrix.",
     "double-factorization/double-factorization-reconstruction.md"),
    ("Why should I not use compressed double factorization on complex ERIs?",
     "double-factorization/double-factorization-compressed.md"),
])
def test_negated_contract_questions_retain_the_requested_explanation(
        prompt, record):
    assert record in router.focused_records(prompt)
    assert str(router.REFS / record) in route_cli(prompt)


def source_section(record, title):
    return next(
        body
        for name, _, body in router.markdown_sections((router.REFS /
                                                       record).read_text())
        if name == title)


@pytest.mark.parametrize("prompt,record", [
    ("Compare double factorization approximations using physical ground energies and excitation gaps.",
     "double-factorization/double-factorization.md"),
    ("Compute ground-state energies from compressed ERI with fixed h, scalar and electron count.",
     "double-factorization/double-factorization.md"),
    ("Build a fixed-sector Hamiltonian from double-factorized integrals.",
     "double-factorization/double-factorization.md"),
    ("Compare the many-body eigenvalues of original and reconstructed ERI in an electron sector.",
     "double-factorization/double-factorization.md"),
    ("Use Jordan-Wigner and Bravyi-Kitaev to compare charge transport after evolving a lattice Hamiltonian.",
     "fermion-transforms/fermion-transforms.md"),
    ("Calculate an expectation value with a Jordan-Wigner observable at fixed mode count.",
     "fermion-transforms/fermion-transforms.md"),
    ("Compute fixed-sector ground energies with fermion.bravyi_kitaev.",
     "fermion-transforms/fermion-transforms.md"),
    ("Use Jordan–Wigner and Bravyi–Kitaev to compute lattice charge dynamics.",
     "fermion-transforms/fermion-transforms.md"),
])
def test_physical_transform_delivers_complete_scientific_checkpoint(
        prompt, record):
    output = route_cli(prompt)
    # CLI appends a shared checklist; measure the bounded reference context itself.
    context = router.reference_context(
        router.route(prompt) + router.focused_records(prompt), prompt)
    assert len(context) <= router.MAX_CONTEXT_CHARS
    for title in ("Workflow", "Verification"):
        assert source_section(record, title) in context
        assert source_section(record, title) in output
    for title in ("Requested task completion", "Requested output artifacts"):
        assert source_section("validation.md", title) in context
    assert source_section("workflow.md", "Completion checkpoint") in context
    assert "def fixed_width_matrix(operator, m):" in context
    assert "Current public source and tests are authoritative" in context
    if record.startswith("double-factorization/"):
        assert "chemistry.qubit_hamiltonian(one_body, eri, scalar_offset=constant)" in context
        assert "h, eri_approx, scalar_offset=offset, tolerance=prune_tol)" in context
    else:
        assert source_section(record, "Shared input representation") in context
        assert "def occupation_matrix(h, V, offset=0.0):" in context
        assert "psi_bk[perm] = psi_occ" in context
        assert "### references/trotter/trotter.md:" not in context


@pytest.mark.parametrize("prompt", [
    "Compute the double-factorization LCU one-norm and identity coefficient only.",
    "Compute the DF one-norm only; do not calculate ground energies or excitation gaps.",
    "Report ERI factorization one-norm and sector metadata only; do not evaluate energies or gaps.",
    "Inspect eigenvalues of the ERI supermatrix to choose Cholesky versus eigendecomposition.",
    "Compare ERI compression sizes without physical spectrum calculations.",
    "Compare Bravyi-Kitaev and Jordan-Wigner Pauli weights only; do not calculate dynamics or energies.",
    "Compile ladder tensors to Jordan-Wigner coefficients and report Pauli words only.",
    "Explain a pipeline from Jordan-Wigner transforms through block encoding to physical observables.",
    "Outline a pipeline from ERI compression through double factorization to ground energy; do not execute it.",
    "Compute ground energy from double factorization and implement a PauliLCU block encoding.",
    "Use Jordan-Wigner and Suzuki-Trotter to compute time-dependent density.",
    "Use Bravyi-Kitaev and QSVT evolution to compute a charge trace.",
    "Compute a double-factorization ground energy and construct a Slater state preparation.",
    "Compare double-factorization energies using the Bravyi-Kitaev transform.",
    "Use Jordan-Wigner and a second-order product formula to simulate lattice dynamics and measure density.",
    "Use Bravyi-Kitaev with Chebyshev moments to estimate the ground energy of a Hubbard chain.",
    "Use Jordan-Wigner with Krylov to estimate a fixed-sector spectrum.",
    "Use Jordan-Wigner and sim_utils.evolve to calculate density dynamics.",
    "Compare ERI compression sizes without calculating ground-state energies.",
    "Compare ERI factorization norms without estimating energies.",
])
def test_physical_transform_packet_preserves_other_tasks_and_named_stages(
        prompt):
    records = router.route(prompt) + router.focused_records(prompt)
    assert router.physical_transform_record(records, prompt) is None


def test_physical_transform_small_budget_keeps_the_existing_bound():
    prompt = "Calculate physical ground energies from double-factorized ERI."
    records = router.route(prompt) + router.focused_records(prompt)
    context = router.reference_context(records, prompt, limit=2500)
    assert len(context) <= 2500
    assert context.count("```python") == context.count("\n```\n")
