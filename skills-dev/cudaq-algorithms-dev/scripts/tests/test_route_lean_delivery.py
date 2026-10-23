# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check executable dense references, resolved fetches and lean entry text."""
import functools
import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SKILL = ROOT / "skills/cudaq-algorithms"
SCRIPT = SKILL / "scripts/route.py"
SPEC = importlib.util.spec_from_file_location("route_lean_delivery", SCRIPT)
router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(router)
EVALS = {
    case["id"]: case["prompt"]
    for case in json.loads((SKILL / "evals/evals.json").read_text())["evals"]
}
NUMERICAL = [
    "science-trotter-conservation-versus-accuracy",
    "science-simulation-probabilistic-cooling",
    "science-qsvt-low-energy-filter-choice",
    "science-qsvt-calibration-robustness",
]
CONTRACT_ONLY = [
    "double-factorization-host-contracts",
    "state-preparation-uccsd-open-shell-parity",
    "source-version-drift",
]
PLACEHOLDERS = ("<skill-dir>", "<this skill's directory>")


def cli(prompt, *options):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *options, prompt],
        check=True,
        capture_output=True,
        text=True).stdout


def context(prompt):
    records = list(
        dict.fromkeys(router.route(prompt) + router.focused_records(prompt)))
    return router.reference_context(records, prompt)


@pytest.mark.parametrize("case", NUMERICAL)
def test_numerical_prompts_get_one_executable_dense_reference_early(case):
    output = context(EVALS[case])
    recipes = [
        block
        for block in re.findall(r"```python\n(.*?)\n```", output, re.DOTALL)
        if "def dense_pauli_matrix(" in block
    ]
    assert len(recipes) == 1
    namespace = {}
    exec(recipes[0], namespace)  # runs check_dense_ordering(3)
    dense = namespace["dense_pauli_matrix"]
    assert dense([(1.0, "ZI")], 2)[1, 1] == -1  # Z on qubit 0 flips index 1
    assert dense([(1.0, "IZ")], 2)[1, 1] == 1  # not qubit 1
    first_family = re.search(r"\n### /.*?/references/(?!quick-contract)",
                             output)
    assert output.index("### Dense Pauli reference") < first_family.start()
    assert len(output) <= router.MAX_CONTEXT_CHARS


@pytest.mark.parametrize("case", CONTRACT_ONLY)
def test_contract_prompts_do_not_get_the_dense_reference(case):
    assert "### Dense Pauli reference" not in context(EVALS[case])


@pytest.mark.parametrize("case", NUMERICAL + CONTRACT_ONLY)
def test_cli_output_has_no_literal_placeholders(case):
    output = cli(EVALS[case])
    for placeholder in PLACEHOLDERS:
        assert placeholder not in output


def test_fully_omitted_sections_get_a_resolved_fetch_command():
    output = cli(EVALS["science-trotter-conservation-versus-accuracy"])
    assert "## Verification\n\n[Additional blocks omitted" not in output
    commands = re.findall(r"^Complete section: (.+)$", output, re.MULTILINE)
    assert commands
    for command in commands[:3]:
        result = subprocess.run(command,
                                shell=True,
                                check=True,
                                capture_output=True,
                                text=True,
                                cwd="/")
        assert result.stdout.strip()


def test_entry_file_is_lean_and_has_no_templated_command():
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    # The revised patch retains mandatory completion sections and targets
    # an entry below 12 KB; the earlier 10.5 KB estimate was not implemented.
    assert len(text.encode("utf-8")) <= 12000
    assert "<skill-dir>" not in text
    assert 'python3 scripts/route.py "<original request>"' in text
    assert "#dense-operator-self-test" in text


def test_participation_clause_present_where_classical_shortcuts_were_allowed():
    assert "names CUDA-Q Algorithms" in (
        SKILL / "references/validation.md").read_text()
    assert "does not name CUDA-Q Algorithms" in (
        SKILL / "references/qsvt/qsvt.md").read_text()
    assert "names CUDA-Q Algorithms" in (
        SKILL / "references/trotter/trotter.md").read_text()


DELIVERY = "## Delivery and completion checks"
PARSE = "json.loads(completed.stdout.splitlines()[-1])"


@functools.lru_cache(maxsize=None)
def cached_cli(prompt, *options):
    return cli(prompt, *options)


def heading_block(output, marker):
    """Delivered text from a heading line to the next heading outside fences."""
    if marker not in output:
        return None
    lines = output[output.index(marker):].splitlines()
    kept, fence = lines[:1], False
    for line in lines[1:]:
        if re.match(r"^\s{0,3}(`{3,}|~{3,})", line):
            fence = not fence
        elif not fence and re.match(r"^#{2,3} ", line):
            break
        kept.append(line)
    return "\n".join(kept).rstrip()


def listed_records(output):
    """Starting guides, focused operation records and premise-named records."""
    found = set(re.findall(r"(?m)^## references/(\S+)$", output))
    for header in ("## Focused operation records (reuse excerpts below)",
                   "## Focused records named by the matching rows"):
        if header in output:
            for line in output.split(header, 1)[1].strip("\n").splitlines():
                if not line.strip():
                    break
                found.add(Path(line).relative_to(router.REFS).as_posix())
    named = re.search(r"(?m)^  Records named by these rows: (.+)$", output)
    if named:
        found.update(named.group(1).split("; "))
    return found


def selected_records(prompt):
    """The records main() passes on, in its order (premise-named last)."""
    records = list(
        dict.fromkeys(router.route(prompt) + router.focused_records(prompt)))
    named = listed_records(cached_cli(prompt, "--paths")) - set(records)
    return records + sorted(named)


def fetch_targets(output):
    targets = set()
    for command in re.findall(r"(?m)^Complete section: (.+)$", output):
        args = shlex.split(command)
        assert args[:2] == [sys.executable, str(SCRIPT)]
        record = args[args.index("--record") + 1]
        targets.update((record, args[index + 1])
                       for index, arg in enumerate(args) if arg == "--section")
    return targets


@pytest.mark.parametrize("case", sorted(EVALS))
def test_default_output_is_capped_and_delivers_checks_before_excerpts(case):
    prompt = EVALS[case]
    output = cached_cli(prompt)
    # Install paths count at one canonical length (see the relocation test).
    assert router.output_budget_size(
        output) <= router.DEFAULT_OUTPUT_BYTES == 8000
    assert output.count("\n") <= router.DEFAULT_OUTPUT_LINES == 180
    if not router.route(prompt):
        assert DELIVERY not in output
        return
    lines = output.splitlines()
    start = lines.index(DELIVERY)
    end = next(index for index in range(start, len(lines))
               if lines[index].startswith(("- Completion checkpoint:",
                                           "- Contract checkpoint:")))
    while lines[end + 1].startswith("  "):
        end += 1
    assert end < 120  # A reader that stops at `head -120` keeps the checks.
    records = selected_records(prompt)
    if router.composition_only(records, prompt):
        assert PARSE not in output
    else:
        parse = next(index for index, line in enumerate(lines)
                     if PARSE in line)
        assert start < parse < 120
    # Exact calls and the dense reference stay unchanged and above the block.
    context = router.reference_context(records, prompt)
    for marker in ("### Exact calls", "### Dense Pauli reference"):
        expected = heading_block(context, marker)
        assert heading_block(output, marker) == expected
        if expected:
            assert output.index(marker) < output.index(DELIVERY)
    if "## Reference excerpts" in output:
        assert output.index(DELIVERY) < output.index("## Reference excerpts")


@pytest.mark.parametrize("case", sorted(EVALS))
def test_default_output_keeps_the_selected_records(case):
    # Navigation-only output is unchanged, so it pins the selected records.
    prompt = EVALS[case]
    assert listed_records(cached_cli(prompt)) == listed_records(
        cached_cli(prompt, "--paths"))


@pytest.mark.parametrize("case", sorted(EVALS))
def test_each_selected_section_is_delivered_whole_or_fetchable(case):
    prompt = EVALS[case]
    output = cached_cli(prompt)
    if not router.route(prompt):
        return
    targets = fetch_targets(output)
    for label, body, record, title, required in router.context_sections(
            selected_records(prompt), prompt):
        if required or (record, title) == ("quick-contract.md", "Exact calls"):
            continue
        blocks = router.markdown_blocks(router.resolve_placeholders(body))
        if all(
                re.match(r"^#{1,6}\s+", line) for block in blocks
                for line in block.splitlines() if line.strip()):
            continue  # A heading-only preamble carries no contract text.
        whole = label + "\n\n".join(blocks) + "\n"
        assert whole in output or (record, title) in targets, (record, title)
    paths = cached_cli(prompt, "--paths")
    if "## Candidate premise rows" in paths:
        rows = [
            line for line in paths.split("## Candidate premise rows", 1)
            [1].splitlines() if line.startswith("| ") and not line.startswith(
                ("| If the request", "| ---"))
        ]
        delivery = output.split(DELIVERY, 1)[1].split("\n## ", 1)[0]
        for row in rows:
            claim = row.strip().strip("|").split(" | ")[0].strip()
            assert claim in delivery  # The claim is a pointer, never a verdict.
            assert row in output or any(
                title.startswith("Premise check") for _, title in targets)


@pytest.mark.parametrize("case", [
    "science-qsvt-low-energy-filter-choice",
    "state-preparation-reference-excitation-device-boundary",
])
def test_grouped_fetch_commands_return_each_complete_section(case):
    commands = re.findall(r"(?m)^Complete section: (.+)$",
                          cached_cli(EVALS[case]))
    assert any(command.count("--section") > 1 for command in commands)
    for command in commands:
        args = shlex.split(command)
        fetched = subprocess.run(args,
                                 cwd="/",
                                 check=True,
                                 capture_output=True,
                                 text=True).stdout
        record = args[args.index("--record") + 1]
        source = router.markdown_sections((router.REFS / record).read_text())
        for title in (args[index + 1] for index, arg in enumerate(args)
                      if arg == "--section"):
            line, body = next(
                (line, body) for name, line, body in source if name == title)
            assert f"## {router.REFS / record}:{line}\n{body}" in fetched


# test_route.py prompts whose inline content sits nearest the budget.
C_DF_OPERATIONS = [
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
]
C_DF_PROMPTS = [
    "Explain the architecture for molecular integrals: " + operation + "."
    for operation in C_DF_OPERATIONS
]
HF_PROMPT = "Construct an HF preparation for a molecular qubit Hamiltonian."
RELOCATED = sorted(EVALS.items()) + [
    (f"c-df-{index}", prompt) for index, prompt in enumerate(C_DF_PROMPTS)
] + [("hf-preparation", HF_PROMPT)]


@pytest.fixture(scope="module")
def long_skill(tmp_path_factory):
    # At least 100 more characters than this checkout's skill directory,
    # whatever the checkout path (CI paths are longer than a dev container's).
    base = tmp_path_factory.mktemp("relocated")
    missing = len(str(SKILL)) + 100 - len(str(base / "cudaq-algorithms"))
    # Split the padding so each path component stays well below NAME_MAX.
    segments = -(-max(missing, 0) // 60)
    for _ in range(segments):
        base = base / ("long-install-root-" * 4)[:59]
    skill = base / "cudaq-algorithms"
    shutil.copytree(SKILL / "references", skill / "references")
    (skill / "scripts").mkdir()
    shutil.copyfile(SCRIPT, skill / "scripts/route.py")
    return skill


@functools.lru_cache(maxsize=None)
def relocated_cli(skill, prompt):
    return subprocess.run(
        [sys.executable,
         str(skill / "scripts/route.py"), prompt],
        cwd="/",
        check=True,
        capture_output=True,
        text=True).stdout


@pytest.mark.parametrize("name, prompt",
                         RELOCATED,
                         ids=[n for n, _ in RELOCATED])
def test_section_selection_does_not_depend_on_the_install_path(
        long_skill, name, prompt):
    here = cached_cli(prompt)
    there = relocated_cli(long_skill, prompt)
    assert len(str(long_skill)) - len(str(SKILL)) >= 100
    # Same sections, order and fetch commands; only the printed paths differ.
    assert there.replace(str(long_skill),
                         "<skill>") == here.replace(str(SKILL), "<skill>")
    commands = re.findall(r"(?m)^Complete section: (.+)$", there)
    for command in commands:
        assert shlex.split(command)[:2] == [
            sys.executable,
            str(long_skill / "scripts/route.py")
        ]


@pytest.mark.parametrize("prompt", C_DF_PROMPTS)
def test_affirmative_compressed_contract_survives_a_long_install_path(
        long_skill, prompt):
    # Mirrors test_route.py::test_affirmative_df_contract_survives_...
    output = relocated_cli(long_skill, prompt)
    assert 'DoubleFactorization(method="C-DF")' in output
    assert ("No packaged quantum kernel or end-to-end resource estimator is "
            "returned.") in " ".join(output.split())


def test_relevant_guide_sections_survive_a_long_install_path(long_skill):
    # Mirrors test_route.py::test_default_delivers_relevant_sections_...
    output = relocated_cli(long_skill, HF_PROMPT)
    assert "Selected reference text" in output
    assert "Requested task completion" in output
    assert "## Workflow" in output
    assert "Resource estimator routing" not in output.split(
        "Selected reference text", 1)[1]
    assert len(output) < 20000
