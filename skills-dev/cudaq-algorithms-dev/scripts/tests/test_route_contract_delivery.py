# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required contracts survive excerpt limits and precede optional examples."""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "skills/cudaq-algorithms/scripts/route.py"
SPEC = importlib.util.spec_from_file_location("route_contract_delivery",
                                              SCRIPT)
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)
EVALS = json.loads(
    (ROOT / "skills/cudaq-algorithms/evals/evals.json").read_text())["evals"]


def request_context(prompt, **kwargs):
    records = list(
        dict.fromkeys(router.route(prompt) + router.focused_records(prompt)))
    return router.reference_context(records, prompt, **kwargs)


@pytest.mark.parametrize("prompt", [
    "Run a numerical QSVT calibration for an asymmetric Pauli Hamiltonian.",
    "Simulate PauliLCU probabilistic cooling and compare with a dense matrix.",
    "Compare Trotter quench dynamics with an independent dense Hamiltonian.",
] + [
    case["prompt"] for case in EVALS if case["id"] in {
        "science-qsvt-calibration-robustness",
        "science-simulation-probabilistic-cooling",
    }
])
def test_numerical_routes_supply_an_executable_little_endian_reference(prompt):
    output = request_context(prompt)
    blocks = re.findall(r"```python\n(.*?)\n```", output, re.DOTALL)
    recipes = [block for block in blocks if "def dense_pauli_matrix(" in block]
    assert len(recipes) == 1
    namespace = {}
    exec(recipes[0], namespace)
    dense = namespace["dense_pauli_matrix"]
    # Independent bit action: X at q0, Y at q2 (including its complex phase),
    # and a scalar on a register with an idle middle qubit.
    actual = dense([(0.3, "XII"), (-0.7, "IIY"), (0.2, "III")], 3)
    expected = np.zeros((8, 8), dtype=complex)
    for column in range(8):
        expected[column ^ 1, column] += 0.3
        expected[column ^ 4, column] += -0.7j * (-1 if column & 4 else 1)
        expected[column, column] += 0.2
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert dense([(1.0, "III")], 3).shape == (8, 8)
    with pytest.raises(ValueError):
        dense([(1.0, "X")], 3)


def test_tensor_matrix_contract_does_not_load_a_pauli_matrix_recipe():
    prompt = next(case["prompt"] for case in EVALS
                  if case["id"] == "double-factorization-host-contracts")
    assert "### Dense Pauli reference" not in request_context(prompt)


def test_qsp_delivers_the_existing_exact_call_and_explicit_convention():
    output = request_context(
        "Apply the supplied raw QSP phase sequence to a normalized ket with "
        "QSVT and compare the good-subspace probability with a dense check.")
    exact_call = 'sim.transform(qsvt, ket, PhaseSequence(phases, convention="qsp"))'
    canonical = (router.REFS / "quick-contract.md").read_text()
    assert exact_call in canonical
    assert exact_call in output
    assert 'default convention is `"qsvt"`' in output
    assert "quick-contract.md:" in output
    assert "frozen_core_integrals" not in output
    assert "Chemistry sector energies" not in output
    assert "ev = Trotter" not in output


@pytest.mark.parametrize("prompt", [
    'Use QSVT projector phases with convention="qsvt" to transform a state.',
    'Apply PhaseSequence with its default convention.',
    'Apply the supplied raw QSP angles to an input ket.',
])
def test_declared_phase_tags_get_the_sequence_representation_contract(prompt):
    assert "qsvt/qsvt-sequence.md" in router.focused_records(prompt)
    output = request_context(prompt)
    assert "projector_phases" in output
    assert '"qsvt"' in output and '"qsp"' in output


def test_chemistry_keeps_sector_rules_and_active_electron_remapping():
    output = request_context(
        "For a frozen-core molecular Hamiltonian, construct a compatible HF "
        "preparation and compare reference and correlated energies in the "
        "requested electron and spin sector.")
    normalized = " ".join(output.split())
    assert "requested electron/spin sector" in normalized
    assert "scalar included exactly once" in normalized
    assert "Subtract `len(core)` from each spin's electron count." in normalized
    assert "Remap occupied orbitals into the active list" in normalized
    assert "`S_z = (N_up - N_down)/2`, not `S^2`" in normalized
    assert "sector before diagonalization" in normalized
    assert "unconstrained Fock-space minimum" in normalized
    assert "sim.transform(qsvt" not in output


def test_exact_contract_precedes_family_examples_when_they_are_included():
    output = request_context(
        "Use QSP phase angles for QSVT filtering of an input statevector.")
    exact_call = 'PhaseSequence(phases, convention="qsp")'
    assert exact_call in output
    examples = [match.start() for match in re.finditer(r"(?m)^```", output)]
    if examples:
        assert output.index(exact_call) < examples[0]


def test_abstract_composition_does_not_pick_an_algorithm_example():
    prompt = ("Explain which interfaces must agree when composing a state "
              "constructor, encoded operator and spectral transformation.")
    assert router.route(prompt) == ["application-composition.md"]
    output = request_context(prompt)
    assert "execution layer" in output
    assert "quick-contract.md:" not in output
    assert "sim.transform(qsvt" not in output
    assert "frozen_core_integrals" not in output
    assert "walk.kernel(power=k)" not in output


def test_injection_delivers_capability_and_boundaries_without_provider_recipe(
):
    output = request_context(
        "Does a state_prep kernel width mismatch raise a clear error? Compare "
        "a measurement-assisted callable and a unitary callable for Walk and QSVT."
    )
    normalized = " ".join(output.split())
    assert "Exact width match required, unenforced at factory time." in normalized
    assert "Zero-argument consumer mode." in output
    assert "Controlled consumers run the preparation once, uncontrolled" in output
    assert "| Width-mismatch behavior | unverified |" in output
    assert "run_slater" not in output
    assert "For an **orbital determinant**" not in output


def test_artifact_contract_can_come_from_context_outside_the_routing_query():
    output = request_context(
        "Compare second- and fourth-order Trotter dynamics of a spin chain "
        "using magnetization accuracy and circuit counts.")
    assert "Requested output artifacts" in output
    assert "Check required fields, types, array shapes" in output
    assert "json.loads(completed.stdout.splitlines()[-1])" in output
    assert len(output) <= router.MAX_CONTEXT_CHARS


def test_nested_contract_sections_are_not_repeated(tmp_path, monkeypatch):
    (tmp_path / "focused.md").write_text(
        "# Provider contract\n\nThe provider owns this unique precondition.\n\n"
        "## Kernel contract\n\nKeep this unique kernel requirement.\n\n"
        "### Argument contract\n\nPreserve this unique argument requirement.\n"
    )
    monkeypatch.setattr(router, "REFS", tmp_path)
    output = router.reference_context(["focused.md"], "Use a provider")
    for phrase in ("unique precondition", "unique kernel requirement",
                   "unique argument requirement"):
        assert output.count(phrase) == 1


def test_an_early_fenced_example_cannot_displace_required_later_prose(
        tmp_path, monkeypatch):
    # Both the complete example and the later prose individually fit, but not
    # together. First-fit source order used to retain the optional example.
    code = "```python\n" + "example_step()\n" * 45 + "```"
    required = (
        "Preserve the declared representation and check the physical sector "
        "before comparing the result. " * 6).strip()
    (tmp_path / "family.md"
     ).write_text("## Workflow\n\nRead the required assumptions.\n\n" + code +
                  "\n\n" + required + "\n")
    monkeypatch.setattr(router, "REFS", tmp_path)
    output = router.reference_context(["family.md"],
                                      "Compute a result",
                                      limit=1250)
    assert required in output
    assert code not in output
    assert "example_step()" not in output
    assert "omitted" in output.lower()
    assert len(output) <= 1250


@pytest.mark.parametrize("limit", [1800, 5000, 12000])
def test_real_contract_context_stays_bounded_with_complete_fences(limit):
    output = request_context(
        "Construct an HF preparation for a molecular Hamiltonian and compare "
        "a QSVT sequence using raw QSP angles with exact sector energies.",
        limit=limit)
    assert len(output) <= limit
    fence = None
    for line in output.splitlines():
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if not marker:
            continue
        run = marker.group(1)
        if fence is None:
            fence = run
        elif (run[0] == fence[0] and len(run) >= len(fence)
              and not line[marker.end():].strip()):
            fence = None
    assert fence is None
