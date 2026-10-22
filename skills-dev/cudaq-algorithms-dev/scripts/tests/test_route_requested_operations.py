# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Requested stages survive routing, exclusions and bounded contract delivery."""
import importlib.util
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest

SCRIPT = Path(
    __file__).resolve().parents[4] / "skills/cudaq-algorithms/scripts/route.py"
SPEC = importlib.util.spec_from_file_location("requested_operation_router",
                                              SCRIPT)
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)
PREP = "state-preparation/state-preparation.md"
DF = "double-factorization/double-factorization.md"
COMPRESSED = "double-factorization/double-factorization-compressed.md"
NORM = "double-factorization/double-factorization-one-norm.md"


def cli(prompt, *options):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *options, prompt],
        check=True,
        capture_output=True,
        text=True).stdout


@pytest.mark.parametrize("options", [(), ("--compact", ),
                                     ("--stage", "contract")])
def test_requested_preparation_contract_is_complete_or_fetchable(options):
    prompt = (
        "Apply the supplied fermionic excitation operators to a reference "
        "state using Jordan-Wigner, then use QSP for Hubbard energy estimates."
    )
    output = cli(prompt, *options)
    assert PREP in output
    # Retrieve the actual advertised command when a budget omits any part.
    source = (router.REFS / PREP).read_text()
    for title in ("Workflow",
                  "Verification") if not options else ("Workflow", ):
        expected = re.search(r"(?ms)^## " + title + r"\n.*?(?=^## |\Z)",
                             source).group(0).strip()
        if expected not in output:
            commands = re.findall(r"(?:Complete section: |More: )(.+)", output)
            command = next(
                (shlex.split(command) for command in commands
                 if PREP in command and shlex.split(command)[-1] == title),
                None)
            assert command is not None, f"Missing complete {title} and exact fetch"
            fetched = subprocess.run(command,
                                     check=True,
                                     capture_output=True,
                                     text=True).stdout
            assert fetched.rstrip().endswith(expected)
            assert len(re.findall(r"(?m)^```", fetched)) % 2 == 0
    assert "ordered correlated preparation" in expected or title == "Verification"


def test_more_than_three_requested_stages_remain_discoverable():
    prompt = (
        "Prepare a correlated state with fermionic excitation operators using "
        "Jordan-Wigner, build a PauliLCU block encoding, then use QSP "
        "for Hubbard energy estimates.")
    records = router.route(prompt)
    assert len(records) <= 3
    available = set(records + router.focused_records(prompt))
    assert {
        PREP, "fermion-transforms/fermion-transforms.md",
        "block-encoding/block-encoding.md", "qsvt/qsvt.md"
    } <= available


def test_supplied_preparation_seam_does_not_expand_all_named_consumers():
    prompt = (
        "I have a state-preparation kernel to hand to PauliLCU.encode_kernel, "
        "Walk.kernel, QSVT.kernel, and Trotter.kernel. Explain the injection "
        "register and width rules.")
    assert "state-preparation/injection-contract.md" in router.route(prompt)
    assert not {
        "qsvt/qsvt.md", "qubitization/qubitization.md",
        "block-encoding/block-encoding.md"
    } & set(router.focused_records(prompt))


def test_injected_trotter_resource_contract_keeps_injection_over_generic_analysis(
):
    prompt = (
        "Use fourth-order Trotter evolution of a newly allocated/injected state "
        "and a caller-supplied cudaq.State; report hardware runtime from "
        "estimate_trotter_resources.")
    assert "state-preparation/injection-contract.md" in router.route(prompt)
    assert "simulation/simulation-analysis.md" not in router.route(prompt)


def test_excitation_operator_before_preparation_verb_preserves_requested_provider(
):
    prompt = ("Use fermionic excitation operators to prepare the reference, "
              "then use QSP to estimate Hubbard energy.")
    assert PREP in router.route(prompt) + router.focused_records(prompt)
    assert PREP in cli(prompt)


def test_declined_excitation_operator_before_verb_does_not_request_provider():
    prompt = (
        "Do not use fermionic excitation operators to prepare the reference; "
        "use QSP to estimate Hubbard energy from the supplied state.")
    assert PREP not in router.route(prompt) + router.focused_records(prompt)


@pytest.mark.parametrize("outcome", ["lowest eigenvalue", "spectral gaps"])
def test_spectral_goal_requires_a_physical_object(outcome):
    assert "qubitization/qubitization.md" in router.route(
        f"Estimate the {outcome} of a Hubbard lattice.")
    assert "qubitization/qubitization.md" not in router.route(
        f"Estimate the {outcome} of this matrix.")


@pytest.mark.parametrize("prompt", [
    "Measure the Hubbard energy of the supplied prepared state.",
    "Compute the energy expectation of an already prepared Slater state for a Hubbard lattice.",
])
def test_prepared_state_measurement_does_not_infer_spectral_solver(prompt):
    assert "qubitization/qubitization.md" not in router.route(prompt)


@pytest.mark.parametrize("verb", ["evaluate", "report"])
def test_explicit_ground_energy_after_preparation_keeps_spectral_stage(verb):
    prompt = f"Prepare a Slater state and {verb} the ground energy of the Hubbard model."
    assert "qubitization/qubitization.md" in router.route(prompt)
    assert PREP in router.route(prompt)


@pytest.mark.parametrize("restriction", [
    "without C-DF",
    "avoid compressed double factorization",
    "do not actually use C-DF",
    "exclude RC-DF",
    "but not C-DF",
    "no C-DF",
    "C-DF is excluded",
])
def test_excluded_compression_is_absent_from_focused_and_exact_recipes(
        restriction):
    prompt = f"Use X-DF to factorize these ERIs and compare QSP conventions; {restriction}."
    assert COMPRESSED not in router.focused_records(prompt)
    output = cli(prompt)
    assert 'method="C-DF"' not in output
    assert str(router.REFS / COMPRESSED) not in output


@pytest.mark.parametrize("options", [(), ("--compact", ),
                                     ("--stage", "contract")])
def test_exclusion_does_not_reenter_through_family_or_premise_recipes(options):
    prompt = (
        "Can X-DF compress the ERI tensor into a converged result controlled "
        "by num_leaves; do not use C-DF.")
    output = cli(prompt, *options)
    assert "For explicit-versus-compressed comparisons" not in output
    assert 'method="C-DF"' not in output
    assert str(router.REFS / COMPRESSED) not in output


@pytest.mark.parametrize("operation", [
    "avoid Trotter and use C-DF",
    "do not change inputs when using C-DF",
    "do not skip C-DF",
    "avoid C-DF for this input; use C-DF for the other input",
    "explain why not C-DF",
    "compare X-DF and C-DF",
    "explain why C-DF is excluded for these inputs",
    "C-DF is not excluded",
    "C-DF is excluded for one input; use C-DF for the other input",
])
def test_affirmative_and_explanatory_compression_survives_negated_other_actions(
        operation):
    assert COMPRESSED in router.focused_records(operation)


@pytest.mark.parametrize("prompt", [
    "Compute the double-factorization LCU one-norm and identity coefficient only.",
    "Compute the LCU norm and identity coefficient of the double factorization only.",
    "Compute the DF LCU norm and identity coefficient only.",
])
def test_norm_and_identity_request_selects_only_the_host_operation(prompt):
    assert router.route(prompt) == [DF]
    assert NORM in router.focused_records(prompt)
    output = cli(prompt)
    assert "double_factorization_one_norm" in output
    assert "excludes the scalar/identity contribution" in output
    assert "## references/trotter/" not in output
    assert "## references/block-encoding/" not in output


def test_factorization_cost_and_identity_does_not_request_trotter():
    prompt = "Estimate the double-factorization cost and identity coefficient only."
    assert router.route(prompt) == [DF]
    assert "## references/trotter/" not in cli(prompt)


@pytest.mark.parametrize("obj", ["statevector", "matrix"])
def test_unrelated_one_norm_does_not_infer_double_factorization(obj):
    prompt = f"Compute the one-norm of this {obj}."
    assert DF not in router.route(prompt)
    assert NORM not in router.focused_records(prompt)


@pytest.mark.parametrize("stage,family", [
    ("build a PauliLCU block encoding", "block-encoding/block-encoding.md"),
    ("use Trotter for time evolution", "trotter/trotter.md"),
    ("use QSP to estimate the energy", "qsvt/qsvt.md"),
])
def test_norm_request_preserves_an_additional_requested_stage(stage, family):
    prompt = f"Compute the double-factorization LCU one-norm and identity coefficient; then {stage}."
    assert family in router.route(prompt)
    assert NORM in router.focused_records(prompt)
