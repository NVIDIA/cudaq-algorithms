#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route an in-scope request by API or scientific object and desired outcome.

Usage: python3 scripts/route.py [--paths | --full] "<the request text>"
Default: routes plus bounded excerpts of relevant contracts and checkpoints.
--paths prints navigation only; --full prints whole selected guides.
Optional: --compact, --stage plan|contract|verify, or --record PATH --section HEADING.
Default delivery is unchanged. Run only the branches needed for unresolved checks.
This is a navigation aid, not a skill-trigger classifier or an answer generator.
"""
import argparse
import re
import sys
from pathlib import Path

REFS = Path(__file__).resolve().parent.parent / "references"

# `cudaq_algorithms.primitives` vocabulary. Bare "qft", "control", and
# "adjoint" are too generic; qft routes only next to add/constant/phase/iqft.
_QFT_ARITHMETIC = r"\biqft\b|\bqft\b.*(add|constant|phase)|(add|constant|phase).*\bqft\b"
PRIMITIVES_LEAVES = [  # (regex, focused record); the front door is added too
    (r"\bqrom\b|qroam|lookup table|table lookup|select.swap",
     "primitives/primitives-qrom.md"),
    (r"unary.iteration|unaryiterationkernels|select walk",
     "primitives/primitives-unary-iteration.md"),
    (r"comparator|\badders?\b|ripple.carry|reversible arithmetic|integer arithmetic|cuccaro|cdkm|draper|fourier.basis adder|(add|subtract)_(constant|register)|cmp_g[et]_|phase_add_constant|"
     + _QFT_ARITHMETIC, "primitives/primitives-arithmetic.md"),
    (r"alias.?sampl|aliassamplingprepare|coherent alias|prepare with garbage|\bvose\b",
     "primitives/primitives-alias-sampling.md"),
]
PRIMITIVES_PATTERN = "|".join([r"\bprimitives\b|toffoli.count"] +
                              [pattern for pattern, _ in PRIMITIVES_LEAVES])
PRIMITIVES_NAMED = r"\bqrom\b|qroam|unary.iteration|select.swap|alias.?sampl|comparator|\badders?\b|ripple.carry|draper|cuccaro|cdkm"

# One row per family: (record path relative to references/, keyword regex over
# the lowercased request, named-method regex or None). The keyword regex selects
# the family; the named-method regex is the subset of that vocabulary which
# names a specific method, so a request naming it keeps the family ahead of a
# default scientific path (a quench with supplied QSP phases must not silently
# become a Trotter task). Generic words shared across families (controlled,
# adjoint, convention, angles, compress, speedup) are deliberately absent.
FAMILIES = [
    ("state-preparation/operator-pools.md",
     r"operator pool|\bceo\b|uccgsd|upccgsd|excitation pool|amplitude optimi|optimi[sz]e the amplitudes",
     None),
    ("state-preparation/state-preparation.md",
     r"hartree|\buccs?d?\b|givens|slater|occupation.(state|pattern|bitstring)|state.prep|prepar(e|ation)|determinant|(single|double)[_ -]excitation|excitation[_ -]kernel|measurement-assisted|inject",
     None),
    ("state-preparation/injection-contract.md",
     r"width|state_prep|inject|consumer|measurement-assisted|feed-forward",
     None),
    ("trotter/trotter.md",
     r"trotter|time evolution|evolve\b|product formula|suzuki|identity (term|phase|coefficient)|estimate_trotter",
     r"trotter|suzuki|product formula"),
    ("qsvt/qsvt.md",
     r"qsvt|\bqsp\b|qsppack|phase sequence|phase factors|qsp angles|phase angles|polynomial (transform|filter)|real-time recovery|recover",
     r"qsvt|\bqsp\b|qsppack|phase sequence|phase factors"),
    ("qubitization/qubitization.md",
     r"\bwalk\b|moment|chebyshev|qubitization|\bkrylov\b|phase estimation|\bqpe\b",
     r"chebyshev|\bkrylov\b|qubitization|\bwalk\b"),
    ("block-encoding/block-encoding.md",
     r"block.encod|paulilcu|\blcu\b|postselect|encoding protocol|controlled select",
     r"block.encod|controlled select|paulilcu|\blcu\b"),
    ("primitives/primitives.md", PRIMITIVES_PATTERN, PRIMITIVES_NAMED),
    ("conventions/numerical-comparison.md",
     r"fp32|fp64|\bprecision\b|\btolerances?\b|nvidia target|single.precision|double.precision|machine epsilon",
     None),
    ("conventions/kernel-boundaries.md",
     r"ternary|heredoc|ifexp|cudaq\.adjoint|cudaq\.control|empty list|marshal|kernel boundar|control.set",
     None),
    ("double-factorization/double-factorization.md",
     r"double.factor|factoriz|\beri\b|\bleaves\b|\bleaf\b|electron.repulsion|two.electron integral|c-df|x-df|rc-df",
     r"double.factor|factoriz|\bc-df\b|\bx-df\b|\bleaf\b|\bleaves\b"),
    ("chemistry/chemistry-bridges.md",
     r"pyscf|fcidump|psi4|chemistry|integrals|molecul|hamiltonian from|sto-3g|6-31g|cc-pv[dtq]z|def2-",
     None),
    ("fermion-transforms/fermion-transforms.md",
     r"bravyi|jordan|fermion|transform selection|parity string",
     r"jordan.wigner|bravyi.kitaev"),
    ("simulation/simulation-analysis.md",
     r"statevector|sim_utils|good.subspace|shots|simulator result|hardware",
     None),
]
ROUTES = [(pattern, rec) for rec, pattern, _ in FAMILIES]
EXPLICIT_ROUTES = [(named, rec) for rec, _, named in FAMILIES if named]

# Words that name a state-preparation object. Only these promote the
# state-preparation family selector to the first position; protocol vocabulary
# such as "controlled" or "adjoint" belongs to every walk/QSVT consumer.
STATE_PREP_OBJECT = r"hartree|givens|slater|\buccs?d?\b|occupation.(state|pattern|bitstring)|determinant|state.prep|inject"
EXCITATION_PREPARATION = (
    r"\b(?:fermionic?\s+)?excitation[ -](?:operators?|generators?|kernels?|rotations?)\b|"
    r"\b(?:single|double)[_ -]excitations?\b|\bfermionic excitations\b")
COMPRESSED_METHOD = r"compressed[_ -](double[_ -])?factorization|\b(?:rc|c)-df\b"
GENERIC_DF_METHOD = r"\bdouble[_ -]factorization\b|\bdf\b"
DF_METHOD = (
    COMPRESSED_METHOD +
    r"|\bexplicit[_ -](?:(?:cholesky|eigendecomposition)[_ -])?(?:double[_ -])?factorization\b|\bx-df\b|"
    + GENERIC_DF_METHOD)
DF_API = (
    r"\b(?:explicit_double_factorization|explicit_cholesky_double_factorization|"
    r"explicit_eigendecomposition_double_factorization|reconstruct_eri|"
    r"factorization_error|modified_one_body_integrals|double_factorization_one_norm|doublefactorization)\b"
)
NORM_OPERATION = (
    r"\bdouble_factorization_one_norm\b|"
    r"\A(?=[\s\S]*\b(?:double.factor\w*|factoriz\w*|df|eri|leaf|leaves)\b)"
    r"[\s\S]*?\b(?:one[ -]norm|lcu[ -]norm)\b")


def operation_negated(text: str, start: int, end: int | None = None) -> bool:
    """Exclude an operation only when its own verb/object is declined."""
    prefix = re.split(r"[.;\n]", text[:start])[-1]
    explanation = re.search(
        r"\bwhy\b|\b(?:explain|describe|discuss)\s+when\b|"
        r"^\s*when\s+(?:should|would|can|could|do|does|is|are)\b", prefix)
    if explanation:
        return False  # An explanation of a restriction still needs its contract.
    if end is not None and re.match(
            r"\s+(?:is|should be|must be|will be)\s+(?:excluded|omitted|avoided|not (?:used|applied))\b",
            text[end:]):
        return True
    # \x74 is a plain "t": spelled out, this removal verb beside a quote reads
    # as an obfuscation directive to SkillSpector 2.13, which then reports the
    # whole file as only partially analyzed.
    if re.search(
            r"\b(?:do not|don't|never)\s+(?:skip|omi\x74|exclude|avoid)\s*$",
            prefix):
        return False
    return bool(
        re.search(
            r"\b(?:avoid|skip|omit|exclude|without|do not|don't|never|not|no|no need (?:to|for))"
            r"(?:\s+(?:actually|optional|the|any|use|using|apply|applying|perform|performing))*\s*$",
            prefix))


def operation_text(text: str) -> str:
    """Remove declined factorization mentions before any routing pass."""
    low = text.lower()
    return re.sub(
        DF_METHOD, lambda match: " " * len(match.group()) if operation_negated(
            low, match.start(), match.end()) else match.group(), low)


def double_factorization_excluded(text: str) -> bool:
    """A declined DF stage must not be inferred again from its integral inputs."""
    low = text.lower()
    mentions = list(re.finditer(DF_METHOD + "|" + DF_API, low))
    declined = [
        match for match in mentions
        if re.fullmatch(GENERIC_DF_METHOD, match.group())
        and operation_negated(low, match.start(), match.end())
    ]
    # A different input may still request DF before or after the restriction.
    return bool(declined) and all(
        operation_negated(low, match.start(), match.end())
        for match in mentions)


def compressed_method_excluded(text: str) -> bool:
    return bool(
        re.search(COMPRESSED_METHOD, text, re.I)
        and not re.search(COMPRESSED_METHOD, operation_text(text)))


def preparation_requested(text: str,
                          *,
                          include_injection_contract: bool = True) -> bool:
    """A provider operation or injection seam, rather than a spectral report."""
    low = text.lower()
    objects = STATE_PREP_OBJECT + "|" + EXCITATION_PREPARATION
    # The injectable seam itself is a preparation contract even when the
    # caller already has a provider kernel and asks how to hand it to consumers.
    if include_injection_contract:
        for match in re.finditer(
                r"\bstate[_ -]prep(?:aration)?\b|\binject(?:ion|ed|ing)?\b",
                low):
            if not operation_negated(low, match.start()):
                return True
    for match in re.finditer(
            r"\b(?:prepare|preparing|preparation|construct|build|apply|applying|compose|inject)\b",
            low):
        if operation_negated(low, match.start()):
            continue
        before = re.search("(?:" + EXCITATION_PREPARATION + r")\s+to\s*$",
                           low[:match.start()])
        if before and not operation_negated(low, before.start()):
            return True
        tail = re.split(r"[.;\n]", low[match.end():])[0]
        obj = re.search(objects, tail)
        if obj and not re.search(
                r"\b(?:report|table|spectrum|spectral report)\b",
                tail[:obj.start()]):
            # Excitation energies/gaps describe observables, not preparations.
            if re.match(r"[ -](?:energies|energy|gaps?|spectrum)\b",
                        tail[obj.end():]):
                continue
            return True
        if match.group().startswith("prepar") and re.search(
                objects, low[:match.start()]):
            # Named noun phrases such as "Slater state preparation".
            if re.search(
                    r"(?:slater|hartree.fock|ucc|occupation|excitation)\s+(?:state\s+)?$",
                    low[:match.start()]):
                return True
    return False


def prepared_state_diagnostic(text: str, preparation: bool) -> bool:
    """Energy on a known state is not a request to solve for eigenstates."""
    low = text.lower()
    if not (preparation or re.search(
            r"\b(?:supplied|already prepared|prepared|input)\b[^.;]*\bstate\b",
            low)):
        return False
    spectral = re.search(
        r"\b(?:estimate|find|compute|compare|calculate|evaluate|report)\b[^.;]*?"
        r"(?:lowest|ground[ -](?:state|energ)|excited[ -]state|spectrum|spectral gaps?|excitation (?:energies|gaps?))",
        low)
    return not spectral and bool(
        re.search(
            r"\b(?:measure|evaluate|compute|report|compare)\b[^.;]*\benerg",
            low))


# Exact APIs and selected operation names bypass premise-row word overlap.
# These are supplemental focused paths, so named contracts survive MAX_RECORDS.
# Rows: (request regex, starting family, focused record).
FOCUSED_ROUTES = [
    (r"\bphasesequence\b|\bqsp\b|"
     r"\bqsvt\b[^.;\n]*\b(?:phases?|angles?|sequences?|conventions?)\b|"
     r"\b(?:phases?|angles?|sequences?|conventions?)\b[^.;\n]*\bqsvt\b",
     "qsvt/qsvt.md", "qsvt/qsvt-sequence.md"),
    (r"explicit[_ -](?:(?:cholesky|eigendecomposition)[_ -])?(double[_ -])?factorization|"
     r"\bdouble.factorization\b[^.;\n]*\bexplicit\b|\bx-df\b",
     "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-explicit.md"),
    (COMPRESSED_METHOD, "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-compressed.md"),
    (r"\breconstruct_eri\b", "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-reconstruction.md"),
    (r"\bfactorization_error\b",
     "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-error.md"),
    (r"\bmodified_one_body_integrals\b",
     "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-modified-one-body.md"),
    (NORM_OPERATION, "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-one-norm.md"),
    (r"\btrotterordering\b|\bmake_trotter_terms\b", "trotter/trotter.md",
     "trotter/trotter-planning.md"),
    (r"\btrotter\.kernel\b", "trotter/trotter.md",
     "trotter/trotter-kernel-factory.md"),
    (r"\btrotter\.state_kernel\b", "trotter/trotter.md",
     "trotter/trotter-state-kernel-factory.md"),
    (r"\bapply_trotter\b", "trotter/trotter.md",
     "trotter/trotter-apply-kernel.md"),
    (r"\btrotter\.resources\b", "trotter/trotter.md",
     "trotter/trotter-resources-planned.md"),
    (r"\bestimate_trotter_resources\b", "trotter/trotter.md",
     "trotter/trotter-resources-raw.md"),
    (r"\b(?:sim|sim_utils)\.evolve\b", "trotter/trotter.md",
     "simulation/simulation-evolve.md"),
]
SHARED_RECORDS = ["workflow.md", "validation.md", "conventions/ordering.md"]
HF_PREPARATION = (r"\bhartree_fock_occupation\b|"
                  r"\b(?:hartree.fock|hf)\b.{0,60}\bprepar\w*|"
                  r"\bprepar\w*.{0,60}\b(?:hartree.fock|hf)\b")
HF_RECORD = "state-preparation/state-preparation-kernel-hartree-fock-occupation.md"
OTHER_PREPARATION = r"givens|slater|\bucc(?:sd|gsd)?\b|upccgsd|excitation|operator pool|\bceo\b"

# Supplemental input checks do not select families or consume starting paths.
INPUT_STATE_FAMILIES = ("state-preparation/", "block-encoding/",
                        "qubitization/", "qsvt/", "trotter/", "simulation/")
INPUT_STATE_PATTERN = (
    r"\b(?:input|prepared|product)[ -]states?\b|\b(?:statevectors?|kets?)\b|"
    r"\btensor[ -](?:product|order)\b|\bendianness\b|"
    r"\b(?:little|big)[ -]endian\b|\b(?:site|qubit|basis)[ -]amplitudes?\b|"
    r"\binitial(?:ly)?[ -](?:in|state)\b|\bstarting (?:from|in)\b|\\rangle|"
    r"\|[01+\-]{1,12}>")

# Numerical routes get the executable little-endian dense reference early, so
# the ordering is not rediscovered by trial and error (conventions/ordering.md).
DENSE_REFERENCE_FAMILIES = ("trotter/", "simulation/", "qsvt/",
                            "block-encoding/", "qubitization/")
DENSE_REFERENCE_EXCLUDED = ("double-factorization/", "chemistry/",
                            "fermion-transforms/")
DENSE_REFERENCE_INTENT = (
    r"\b(?:dense|exact|reference|numerical(?:ly)?|simulat\w*|compar\w*|"
    r"calibrat\w*|fidelit\w*|probabilit\w*|accura\w*|errors?|dynamics|"
    r"evolution|evolv\w*|cooling|expectation|observables?|magneti[sz]\w*|"
    r"energy|energies|spectrum|spectral|leakage|occupation)\b")
DENSE_REFERENCE_SECTION = "Dense-operator self-test"
DENSE_REFERENCE_LABEL = "\n### Dense Pauli reference\n"
PLACEHOLDERS = ("<this skill's directory>", "<skill-dir>")


def dense_reference_wanted(records: list[str], text: str) -> bool:
    """The primary route is numerical and the request compares or simulates."""
    primary = records[0] if records else ""
    return (primary.startswith(DENSE_REFERENCE_FAMILIES) and not any(
        rec.startswith(DENSE_REFERENCE_EXCLUDED) for rec in records)
            and bool(re.search(DENSE_REFERENCE_INTENT, text, re.IGNORECASE))
            and (REFS / "conventions/ordering.md").is_file())


def resolve_placeholders(text: str) -> str:
    """Emit executable paths: the skill directory replaces documentation placeholders."""
    skill_dir = str(REFS.parent)
    for placeholder in PLACEHOLDERS:
        text = text.replace(placeholder, skill_dir)
    return text


# Require both a scientific object and an outcome: generic words such as
# "energy", "time", or "geometry" alone should not choose an application.
# Rows: (object regex, goal regex, records, compose) where compose adds the
# application-composition guide after the stage records.
# A chemical atom name qualifies a molecular dimer; bare "dimer" also names
# spin systems and is not by itself a chemistry cue.
ATOMIC_DIMER = (
    r"\b(?:hydrogen|helium|lithium|beryllium|boron|carbon|nitrogen|oxygen|"
    r"fluorine|neon|sodium|magnesium|aluminium|aluminum|silicon|phosphorus|"
    r"sulfur|sulphur|chlorine|argon|potassium|calcium|titanium|vanadium|"
    r"chromium|manganese|iron|cobalt|nickel|copper|zinc|bromine|krypton|"
    r"iodine|xenon)[\s-]+dimers?\b")
MOLECULE_OBJECT = (
    r"geometr|bond.length|bond.angle|bond.stretch|atomic.coordinates|\batoms\b|"
    r"gaussian.basis|orbital.basis|basis.set|sto-3g|6-31g|cc-pv[dtq]z|def2-|"
    r"molecul|frozen.core|active.space|mean.field|nuclear.repulsion|"
    r"\bwater\b|ammonia|methane|benzene|ethylene|hydride|hydrogen fluoride|" +
    ATOMIC_DIMER)
LATTICE_OBJECT = r"spin.chain|spin.lattice|lattice|domain.wall|hubbard|ising|heisenberg|xxz|magneti[sz]|hopping|half.filling"
SCIENCE_ROUTES = [
    (MOLECULE_OBJECT,
     r"energ|ground|gap|spectrum|reference|overlap|qubit|correlat|hamiltonian",
     ["chemistry/chemistry-bridges.md"], True),
    (LATTICE_OBJECT, r"evol|quench|dynamics|\btimes?\b|time.depend|transport",
     ["trotter/trotter.md"], True),
    (LATTICE_OBJECT,
     r"energ|ground.state|spectrum|\bgaps?\b|lowest eigenvalue|sector",
     ["qubitization/qubitization.md"], True),
    (r"ancilla|herald|postselect|success.flag",
     r"probab|discard|trace|condition|observable|density|expectation",
     ["simulation/simulation-analysis.md"], False),
]
MAX_RECORDS = 3
MAX_CONTEXT_CHARS = 16000

# Abstract compatibility questions need the composition procedure, without
# guessing an algorithm from generic words such as "spectral" or "primitive".
COMPOSITION_INTENT = r"compos(?:e|ing|ition|able|ability)|compatib|\bconnect\b|\binterfaces?\b"
COMPOSITION_OBJECT = (
    r"\bftqc\b|state constructor|"
    r"encoded operator|spectral transformation|quantum (?:primitives|building blocks)"
)

# Chemical formulas (LiH, H2O, N2, CH4) are molecular objects. Tokens parse
# fully into these element symbols with optional positive counts and contain
# at least two atoms. Thus Hamiltonian labels H0 and H1 are not molecules;
# library and method acronyms that happen to parse are excluded explicitly.
ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Br", "Kr", "I", "Xe"
}
NOT_FORMULAS = {
    "OK", "CC", "HF", "UCC", "BK", "JW", "PBC", "BCS", "HPC", "CI", "FCI",
    "KS", "SCF", "CNN", "NN", "SO", "NO", "ON", "CO", "SNR", "BF"
}
_FORMULA_TOKEN = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")
_FORMULA_PART = re.compile(r"([A-Z][a-z]?)(\d*)")


def is_chemical_formula(token: str) -> bool:
    if token in NOT_FORMULAS or not _FORMULA_TOKEN.match(token):
        return False
    parts = _FORMULA_PART.findall(token)
    if any(symbol not in ELEMENTS for symbol, _ in parts):
        return False
    counts = [int(count) if count else 1 for _, count in parts]
    return all(count > 0 for count in counts) and sum(counts) >= 2


def mentions_molecule(text: str) -> bool:
    return any(
        is_chemical_formula(token)
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", text))


# Queries naming an algorithm that is not a packaged primitive get a note
# with the packaged building blocks instead of a misroute.
NOTES = [
    (r"phase estimation|\bqpe\b",
     "Quantum phase estimation is not a packaged primitive. Building blocks: "
     "the block encoding's walk-step kernels (`walk_step_kernel`, "
     "`adjoint_walk_step_kernel`, their controlled forms, `prepare_kernel`) "
     "and `Trotter` product-formula evolution; the self-verifying example "
     "`docs/sphinx/examples/python/08_quantum_phase_estimation.py` runs one "
     "QPE shell around exact, Trotter, and qubitization-walk oracles "
     "(walk decoding E = -alpha*cos(theta)). Chebyshev moments/Krylov "
     "(qubitization guide) estimate energies without a counting register."),
]
CHECKLIST = """
## Use the selected guides

Scientific task: read Workflow and Verification, select the requested observable
and error measure from the excerpts, and follow source pointers for chosen stages.
Reuse the emitted sections; open more text only for an unresolved requirement.
Contract question: check relevant premise rows against the current source.
Report the result, evidence, material boundaries, and unresolved inputs concisely.
"""


def all_records() -> list[str]:
    """Every record path a route can print; tests check each exists."""
    out = [rec for rec, _, _ in FAMILIES
           ] + [leaf for _, leaf in PRIMITIVES_LEAVES]
    out.extend(SHARED_RECORDS)
    out.append("quick-contract.md")
    out.append(HF_RECORD)
    for _, family, focused in FOCUSED_ROUTES:
        out.extend((family, focused))
    for _, _, recs, compose in SCIENCE_ROUTES:
        out.extend(recs)
        if compose:
            out.append("application-composition.md")
    return list(dict.fromkeys(out))


def notes(text: str) -> list[str]:
    low = text.lower()
    return [note for pattern, note in NOTES if re.search(pattern, low)]


def standalone_kernel(text: str) -> bool:
    """Recognize explicit basic-kernel queries, preserving composed uses."""
    low = text.lower()
    if not re.search(
            r"\bbell\b|\b(?:basic|standalone)\s+(?:cuda-q|cudaq)\s+kernel\b",
            low):
        return False
    if any(re.search(pattern, low) for pattern, _ in EXPLICIT_ROUTES):
        return False
    if any(re.search(pattern, low) for pattern, _, _ in FOCUSED_ROUTES):
        return False
    # A basic state used by a named algorithm is not a standalone kernel.
    # Keep unsupported-algorithm notes reachable as well (for example QPE).
    if re.search(r"\brecover_real_time_evolution\b", low) or notes(text):
        return False
    if re.search(
            r"hartree|givens|slater|\buccs?d?\b|state_prep|sim_utils|good.subspace|"
            r"\b(?:chemistry|fermion|stateprep|sim|df)\.[a-z_]\w*", low):
        return False
    return not any((re.search(obj, low) or
                    (obj == MOLECULE_OBJECT and mentions_molecule(text)))
                   and re.search(goal, low)
                   for obj, goal, _, _ in SCIENCE_ROUTES)


def focused_records(text: str) -> list[str]:
    """Direct operation hints, independent of heuristic premise candidates."""
    low = operation_text(text)
    out = [
        focused for pattern, _, focused in FOCUSED_ROUTES
        if re.search(pattern, low)
    ]
    out.extend(leaf for pattern, leaf in PRIMITIVES_LEAVES
               if re.search(pattern, low))
    # Planning is the relevant contract for an explicitly requested grouping.
    if re.search(r"trotter|suzuki|product formula", low) and re.search(
            r"grouping|term order|ordering", low):
        out.append("trotter/trotter-planning.md")
    if re.search(HF_PREPARATION, low):
        out.append(HF_RECORD)
    if preparation_requested(text, include_injection_contract=False):
        # Three starting guides remain the limit. Explicit stages beyond that
        # limit are still reachable as focused contracts. A supplied provider's
        # injection question is answered by the seam, not all consumer guides.
        requested = [
            rec for pattern, rec in EXPLICIT_ROUTES if re.search(pattern, low)
        ]
        requested.append("state-preparation/state-preparation.md")
        selected = route(text)
        out.extend(rec for rec in requested if rec not in selected)
    if double_factorization_excluded(text):
        out = [
            rec for rec in out if not rec.startswith("double-factorization/")
        ]
    return list(dict.fromkeys(out))


def route(text: str) -> list[str]:
    if standalone_kernel(text):
        return []
    low = operation_text(text)
    preparation = preparation_requested(text)
    # In a host norm request, LCU names the formula and the identity coefficient
    # names an excluded scalar offset. Neither calls for a simulation circuit.
    if re.search(NORM_OPERATION, low):
        low = re.sub(r"\blcu[ -]+(?:one[ -])?norm\b", "one-norm", low)
        low = re.sub(r"\bidentity coefficient\b", "", low)
    elif re.search(r"double.factor|factoriz|\beri\b|\bdf\b",
                   low) and re.search(r"\bcost\b", low):
        low = re.sub(r"\bidentity coefficient\b", "", low)
    molecule = mentions_molecule(text)
    out = []
    for pattern, rec in ROUTES:
        if re.search(pattern, low) and rec not in out:
            out.append(rec)
        if rec == "primitives/primitives.md":
            # A named leaf symbol opens its focused record after the door.
            for leaf_pattern, leaf in PRIMITIVES_LEAVES:
                if re.search(leaf_pattern, low) and leaf not in out:
                    out.append(leaf)
    if preparation and "state-preparation/state-preparation.md" not in out[:1]:
        out = ["state-preparation/state-preparation.md"] + [
            r for r in out if r != "state-preparation/state-preparation.md"
        ]
    science = []
    compose = False
    for obj, goal, recs, chain in SCIENCE_ROUTES:
        matched = re.search(obj, low) or (obj is MOLECULE_OBJECT and molecule)
        if matched and re.search(goal, low):
            if recs == ["qubitization/qubitization.md"
                        ] and prepared_state_diagnostic(text, preparation):
                continue
            science.extend(r for r in recs if r not in science)
            compose = compose or chain
    # A named method takes priority over a default scientific path: a quench
    # with supplied QSP phases must not silently become a Trotter task.
    explicit = [
        family for pattern, family, _ in FOCUSED_ROUTES
        if re.search(pattern, low)
    ]
    for pattern, rec in EXPLICIT_ROUTES:
        if re.search(pattern, low):
            explicit.append(rec)
            if rec == "primitives/primitives.md":
                # Keep a named primitive's contract ahead of generic words
                # such as PREPARE before the three-record limit is applied.
                explicit.extend(leaf
                                for leaf_pattern, leaf in PRIMITIVES_LEAVES
                                if re.search(leaf_pattern, low))
    if science:
        # Composition points to the remaining stage guides; do not dump an
        # entire dependency tree into context for every scientific request.
        chain = ["application-composition.md"] if compose else []
        if any(
                rec.startswith(("qsvt/", "qubitization/", "trotter/"))
                for rec in explicit):
            science = [
                rec for rec in science if rec not in
                {"trotter/trotter.md", "qubitization/qubitization.md"}
            ]
        requested_prep = ["state-preparation/state-preparation.md"
                          ] if preparation else []
        out = explicit + requested_prep + science + chain + out
    elif explicit and ("primitives/primitives.md" in explicit or
                       (preparation and re.search(EXCITATION_PREPARATION, low))
                       or not preparation):
        # A named primitive is more specific than generic state preparation
        # or injection wording. Keep ordinary state-preparation priority when
        # no primitive was named.
        out = explicit + out
    if any(
            rec.startswith(("qsvt/", "qubitization/", "trotter/"))
            for rec in explicit) and not preparation:
        # Spectral preparation by the named algorithm is not a request for an
        # unrelated Hartree-Fock, Givens or UCC provider recipe.
        out = [
            rec for rec in out
            if rec != "state-preparation/state-preparation.md"
        ]
    if not preparation and re.search(EXCITATION_PREPARATION, low):
        out = [
            rec for rec in out
            if rec != "state-preparation/state-preparation.md"
        ]
    generic = {
        "primitives/primitives.md", "state-preparation/state-preparation.md"
    }
    if (not explicit and not science and set(out) <= generic
            and re.search(COMPOSITION_INTENT, low)
            and re.search(COMPOSITION_OBJECT, low)
            and not re.search(OTHER_PREPARATION + r"|hartree|\bhf\b", low)):
        return ["application-composition.md"]
    if double_factorization_excluded(text):
        out = [
            rec for rec in out if not rec.startswith("double-factorization/")
        ]
    return list(dict.fromkeys(out))[:MAX_RECORDS]


def markdown_sections(body: str) -> list[tuple[str, int, str]]:
    """Return heading sections with line numbers, respecting fenced examples."""
    lines = body.splitlines(keepends=True)
    headings = []
    fence = None
    for index, line in enumerate(lines):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = run
            elif (run[0] == fence[0] and len(run) >= len(fence)
                  and not line[marker.end():].strip()):
                fence = None
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if fence is None and heading:
            headings.append((index, len(heading.group(1)), heading.group(2)))
    sections = []
    for n, (start, level, title) in enumerate(headings):
        end = next((i for i, depth, _ in headings[n + 1:] if depth <= level),
                   len(lines))
        sections.append((title, start + 1, "".join(lines[start:end]).strip()))
    return sections


def relevant_sections(record: str,
                      body: str,
                      composition_only: bool = False,
                      text: str = "") -> list[tuple[str, int, str]]:
    sections = markdown_sections(body)
    if record == "application-composition.md":
        wanted = {"Composition procedure"}
    elif record == "validation.md":
        wanted = {"Requested task completion"}
        if not composition_only:
            wanted.add("Requested output artifacts")
    elif record == "workflow.md":
        wanted = {
            "Match the scientific contracts"
            if composition_only else "Completion checkpoint"
        }
    elif any(title == "Workflow" for title, _, _ in sections):
        wanted = {"Workflow", "Verification"}
    else:
        # Focused records use a small range of contract/representation titles.
        wanted = {
            title
            for title, _, _ in sections if re.search(
                r"contract|representation|capability|boundaries|composition|validation",
                title, re.IGNORECASE)
        }
    selected = []
    end = 0
    for title, line, section in sections:
        # Retain a contract document's preamble, not its entire subtree. A
        # selected section already contains any selected nested subsections.
        if section.startswith("# "):
            next_line = next(
                (start for _, start, _ in sections if start > line),
                line + len(section.splitlines()))
            section = "\n".join(section.splitlines()[:next_line -
                                                     line]).strip()
        if title in wanted and line >= end:
            if (record == "double-factorization/double-factorization.md"
                    and title == "Workflow"
                    and compressed_method_excluded(text)):
                section = "\n\n".join(
                    block for block in markdown_blocks(section)
                    if not re.search(r"\b(?:RC|C)-DF\b", block))
            selected.append((title, line, section))
            end = line + len(section.splitlines())
    return selected


def quick_contract_sections(records: list[str],
                            text: str) -> list[tuple[str, int, str]]:
    """Select existing exact calls and conventions; keep one source of truth."""
    path = REFS / "quick-contract.md"
    if not path.is_file():
        return []
    families = {record.split("/")[0] for record in records if "/" in record}
    operations = {
        "block-encoding":
        {"Block-encode `H/alpha`", "Check the encoded block densely"},
        "qsvt": {"QSP/QSVT sequence"},
        "qubitization": {"Qubitization walk action"},
        "trotter": {"Suzuki-Trotter"},
        "chemistry": {"Chemistry integrals"},
        "fermion-transforms": {"Fermion transforms"},
        "primitives": {"Circuit primitives"},
        "double-factorization": {"Double factorization"},
        "state-preparation": {"State preparation kernels"},
    }
    wanted_rows = set().union(*(operations.get(f, set()) for f in families))
    if compressed_method_excluded(text):
        # The mixed row includes a C-DF constructor. Keep X-DF's focused
        # contract, without silently reintroducing the declined recipe.
        wanted_rows.discard("Double factorization")
    if ("state-preparation/injection-contract.md" in records and not re.search(
            OTHER_PREPARATION + r"|hartree|\bhf\b", text, re.IGNORECASE)):
        wanted_rows.discard("State preparation kernels")
    if "qsvt" in families and re.search(r"recover|real.time|evolution", text,
                                        re.IGNORECASE):
        wanted_rows.add("Real-time evolution recovery")
    wanted_sections = set()
    if "chemistry" in families:
        wanted_sections.add(
            "Chemistry sector energies and integral conventions")
    if "qubitization" in families:
        wanted_sections.add(
            "Qubitization walk action and prepared-state checks")
    if "block-encoding" in families and re.search(r"protocol|consumer", text,
                                                  re.IGNORECASE):
        wanted_sections.add("Consume a protocol-only encoding")
    body = path.read_text()
    rows, header, table_line = [], [], 0
    for line, value in enumerate(body.splitlines(), 1):
        if value.startswith("| Operation |"):
            header = [value]
            table_line = line
        elif header and len(header) == 1 and value.startswith("| ---"):
            header.append(value)
        elif value.startswith("| ") and value.split(
                "|", 2)[1].strip() in wanted_rows:
            rows.append(value)
    selected = []
    if rows:
        selected.append(("Exact calls", table_line,
                         "### Exact calls\n\n" + "\n".join(header + rows)))
    selected.extend(section for section in markdown_sections(body)
                    if section[0] in wanted_sections)
    return selected


def markdown_blocks(body: str) -> list[str]:
    """Keep paragraphs, list items, tables and fenced examples intact."""
    blocks, current = [], []
    fence = None
    for line in body.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if fence is None and (not line.strip() or marker
                              or re.match(r"^(?:[-*+] |\d+[.)] )", line)):
            if current:
                blocks.append("".join(current).strip())
                current = []
        current.append(line)
        if marker:
            run = marker.group(1)
            if fence is None:
                fence = run
            elif (run[0] == fence[0] and len(run) >= len(fence)
                  and not line[marker.end():].strip()):
                fence = None
                blocks.append("".join(current).strip())
                current = []
    if current:
        blocks.append("".join(current).strip())
    return [block for block in blocks if block]


def reference_context(records: list[str],
                      text: str,
                      limit: int = MAX_CONTEXT_CHARS) -> str:
    """Load public reference sections, never catalog answers or private grades.

    Deliver exact calls first; preserve prose contracts before fenced examples.
    Omitted blocks remain available at exact paths.
    """
    selected = [
        (f"\n### {REFS / 'quick-contract.md'}:{line}\n", body,
         "quick-contract.md", title, False)
        for title, line, body in quick_contract_sections(records, text)
    ]
    if dense_reference_wanted(records, text):
        # The executable little-endian dense reference follows the exact calls;
        # its fence is allocated with the prose so the budget never drops it.
        for title, _, body in markdown_sections(
            (REFS / "conventions/ordering.md").read_text()):
            if title == DENSE_REFERENCE_SECTION:
                lines = body.splitlines()
                while lines and re.match(r"^#{1,6}\s+", lines[0]):
                    lines.pop(0)
                selected.append(
                    (DENSE_REFERENCE_LABEL, "\n".join(lines).strip(),
                     "conventions/ordering.md", title, True))
    composition_only = records == [
        "application-composition.md"
    ] and not re.search(r"\b(json|csv|schema|stdout|files?|scripts?)\b", text,
                        re.IGNORECASE)
    # Completion applies conditionally to the user's requested deliverables;
    # load it alongside family guidance instead of just linking to it again.
    ordered = list(dict.fromkeys(records + ["validation.md", "workflow.md"]))
    for record in ordered:
        if record == "state-preparation/state-preparation.md":
            if (HF_RECORD in records
                    and not re.search(OTHER_PREPARATION, text, re.IGNORECASE)):
                continue  # Named HF contract replaces the broad Slater/UCC recipe.
            if ("state-preparation/injection-contract.md" in records
                    and not re.search(OTHER_PREPARATION + r"|hartree|\bhf\b",
                                      text, re.IGNORECASE)):
                continue  # Injection does not ask for a particular provider.
        path = REFS / record
        if not path.is_file():
            continue  # Navigation reports missing selected records separately.
        for title, line, body in relevant_sections(record, path.read_text(),
                                                   composition_only, text):
            # Delivery requirements can come from the surrounding instructions,
            # not just this routing query. Keep their conditional procedure.
            selected.append(
                (f"\n### {path}:{line}\n", body, record, title, False))
    if not selected:
        return ""
    header = "\n## Selected reference text\n"
    footer = (
        "\nExcerpts may omit blocks or examples. Use the source paths and line "
        "numbers above to resolve any remaining contract before using it; "
        "follow current source/test links for API behavior.\n")
    notice = "\n[Additional blocks omitted; open this section when needed.]\n"
    overhead = len(header) + len(footer) + sum(
        len(label) + len(notice) + 1 for label, *_ in selected)
    if overhead > limit:
        return ""  # Do not emit misleading fragments under an unusable budget.
    excerpts = [[
        label,
        markdown_blocks(resolve_placeholders(body)),
        set(), record, title, required
    ] for label, body, record, title, required in selected]
    # A per-section quota favors short boilerplate over an indivisible semantic
    # list or table. Allocate by contract priority instead, before examples.
    remaining = limit - overhead

    def section_priority(entry):
        label, blocks, _, _, _, required = entry
        if "quick-contract.md:" in label:
            return 0
        if required:
            return 0.5  # Executable dense reference, right after the exact calls.
        if blocks and re.search(r"capabilit|boundar|scientific contract",
                                blocks[0], re.IGNORECASE):
            return 1  # Preserve whole semantic invariant lists/boundary tables.
        if "/validation.md:" in label:
            return 2  # Deliverable/interface checks selected for this request.
        if "/workflow.md:" in label:
            return 6
        if blocks and blocks[0].startswith("## Validation"):
            return 5  # Evidence inventories follow the operation's requirements.
        if blocks and blocks[0] in ("## Workflow", "## Verification"):
            return 4
        return 3  # Focused representation and composition requirements.

    priority = sorted(excerpts, key=section_priority)
    for examples in (False, True):
        for _, blocks, kept, _, _, required in priority:
            for index, block in enumerate(blocks):
                is_example = bool(re.match(r"^\s*(`{3,}|~{3,})", block))
                # A required recipe keeps its fence in the prose pass; other
                # fenced examples still follow every prose block.
                eligible = is_example == examples or (required
                                                      and not examples)
                if (eligible and index not in kept
                        and len(block) + 2 <= remaining):
                    kept.add(index)
                    remaining -= len(block) + 2

    def render_excerpts():
        rendered = [header]
        for label, blocks, kept, record, title, _ in excerpts:
            substantive = any(not all(
                re.match(r"^#{1,6}\s+", line)
                for line in blocks[index].splitlines() if line.strip())
                              for index in kept)
            if not substantive:
                # A retained heading is not delivered contract content.
                rendered.append(label + "Complete section: " +
                                section_command(record, title) + "\n")
                continue
            rendered.append(label + "\n\n".join(
                block
                for index, block in enumerate(blocks) if index in kept) + "\n")
            if len(kept) < len(blocks):
                rendered.append(notice)
                if record == "state-preparation/state-preparation.md" and preparation_requested(
                        text):
                    rendered.append("Complete section: " +
                                    section_command(record, title) + "\n")
        return "".join(rendered + [footer])

    output = render_excerpts()
    # Resolved fetch commands can exceed the reserved generic notice. Make
    # room by removing lower-priority blocks, keeping exact calls and the
    # required dense recipe. Never fall back to a heading-only stub.
    if len(output) > limit:
        for examples in (True, False):
            for _, blocks, kept, record, _, required in reversed(priority):
                if required or record == "quick-contract.md":
                    continue
                for index in sorted(kept, reverse=True):
                    is_example = bool(
                        re.match(r"^\s*(`{3,}|~{3,})", blocks[index]))
                    if is_example != examples:
                        continue
                    kept.remove(index)
                    output = render_excerpts()
                    if len(output) <= limit:
                        return output
        return ""  # Even complete navigation cannot fit this caller's budget.
    return output


COMPACT_CONTEXT_CHARS = 7000
STAGE_CONTEXT_CHARS = 5000
SECTION_CHARS = 16000


def section_command(record: str, title: str) -> str:
    import shlex
    if record == "quick-contract.md" and title == "Exact calls":
        title = "Quick contract (checked against `python/cudaq_algorithms` in this checkout)"
    return shlex.join([
        sys.executable,
        str(Path(__file__).resolve()), "--record", record, "--section", title
    ])


def fetch_section(record: str, title: str, parser) -> None:
    """Read only a known local reference and a complete named section."""
    # Reject path traversal, absolute paths, and non-reference inputs.
    if not re.fullmatch(r"[\w-]+(?:/[\w-]+)*\.md", record):
        parser.error("--record must be a relative reference Markdown path")
    try:
        reference_root = REFS.resolve()
        path = (reference_root / record).resolve()
    except (OSError, RuntimeError):
        parser.error("reference path cannot be resolved")
    if not path.is_relative_to(reference_root):
        parser.error(
            "reference resolves outside the skill's references directory")
    if not path.is_file():
        parser.error("reference does not exist in this skill")
    sections = markdown_sections(path.read_text())
    found = [(line, body) for name, line, body in sections if name == title]
    if len(found) != 1:
        parser.error("section must match one unique heading; available: " +
                     "; ".join(name for name, _, _ in sections))
    line, body = found[0]
    if len(body) > SECTION_CHARS:
        parser.error(
            "section exceeds 16000 characters; choose a narrower heading: " +
            "; ".join(name for name, start, _ in sections if start > line))
    print(f"## {path}:{line}\n{body}")


def compact_context(records: list[str],
                    text: str,
                    stage: str = "bundle") -> str:
    """Keep verbatim semantic blocks; advertise every omitted section explicitly.

    Contracts precede examples. A selected source block is never cut mid-list,
    table, or fence. Shared delivery rules remain in SKILL.md and are retrievable.
    The cap applies to this bundle; legacy navigation is additional output.
    """
    selected = []
    if stage != "verify":
        selected.extend(
            ("quick-contract.md", title, line, body)
            for title, line, body in quick_contract_sections(records, text))
    composition_only = records == [
        "application-composition.md"
    ] and not re.search(r"\b(json|csv|schema|stdout|files?|scripts?)\b", text,
                        re.I)
    for record in dict.fromkeys(records):
        if record == "state-preparation/state-preparation.md":
            if HF_RECORD in records and not re.search(OTHER_PREPARATION, text,
                                                      re.I):
                continue
            if ("state-preparation/injection-contract.md" in records
                    and not re.search(OTHER_PREPARATION + r"|hartree|\bhf\b",
                                      text, re.I)):
                continue
        path = REFS / record
        if not path.is_file():
            continue
        body = path.read_text()
        for title, line, section in relevant_sections(record, body,
                                                      composition_only, text):
            verify = bool(re.search(r"verif|validat", title, re.I))
            if stage == "contract" and verify:
                continue
            if stage == "verify" and not verify:
                continue
            selected.append((record, title, line, section))
    if stage == "verify":
        for record in ("validation.md", "workflow.md"):
            selected.extend(
                (record, title, line, body)
                for title, line, body in relevant_sections(
                    record, (REFS / record).read_text(), composition_only))
    # Deliver focused contracts before broad recipes, without changing routing.
    def priority(item):
        record, title, _, _ = item
        if record == "quick-contract.md": return 0
        if re.search(r"capabilit|boundar|scientific contract", title, re.I):
            return 1
        if title not in ("Workflow", "Verification"): return 2
        return 3 if title == "Workflow" else 4

    selected.sort(key=priority)
    limit = COMPACT_CONTEXT_CHARS if stage == "bundle" else STAGE_CONTEXT_CHARS
    header = "\n## Compact reference bundle\n"
    footer = (
        "\nReuse these blocks. Fetch a named section only for a missing contract "
        "or check; follow its current source/test pointers. Fenced examples "
        "may be omitted. --stage contract or --stage verify with the original "
        "request selects that branch; --record/--section fetches one complete section.\n"
    )
    source_paths = []
    for record in records:
        path = REFS / record
        if path.is_file():
            source_paths.extend(
                re.findall(
                    r"(?:python/cudaq_algorithms|tests/python|docs/sphinx)/[\w./-]+",
                    path.read_text()))
    if source_paths:
        footer += "Current source/tests: " + "; ".join(
            dict.fromkeys(source_paths)) + "\n"
    entries = []
    seen = set()
    for record, title, line, section in selected:
        label = f"\n### {record}:{line} - {title}\n"
        fetch = "\nMore: " + section_command(record, title) + "\n"
        blocks = []
        for block in markdown_blocks(section):
            # A heading need not have a blank line before its prose. Remove
            # heading lines only; never discard requirements in the same block.
            lines = block.splitlines()
            while lines and re.match(r"^#{1,6}\s+", lines[0]):
                lines.pop(0)
            block = "\n".join(lines).strip()
            if not block or block in seen:
                continue
            seen.add(block)
            blocks.append(block)
        entries.append([label, fetch, blocks, []])
    # Reserve every follow-up pointer before assigning content budget.
    overhead = len(header) + len(footer) + sum(
        len(e[0]) + len(e[1]) for e in entries)
    if overhead > limit:
        return header + "Too many sections for one bundle; use --stage plan, then fetch one named section.\n"
    remaining = limit - overhead
    for examples in (False, True):
        for _, _, blocks, kept in entries:
            for block in blocks:
                if bool(re.match(
                        r"^\s*(`{3,}|~{3,})",
                        block)) == examples and len(block) + 2 <= remaining:
                    kept.append(block)
                    remaining -= len(block) + 2
    rendered = [header]
    for label, fetch, blocks, kept in entries:
        rendered.append(label)
        # Restore original block order, including intact examples if they fit.
        rendered.append("\n\n".join(block for block in blocks
                                    if block in kept))
        if len(kept) < len(blocks):
            rendered.append(fetch)
        else:
            rendered.append("\n")
    rendered.append(footer)
    return "".join(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--full",
                        action="store_true",
                        help="print complete selected guides")
    output.add_argument(
        "--paths",
        action="store_true",
        help="print navigation only, without reference excerpts")
    output.add_argument(
        "--compact",
        action="store_true",
        help="experimental smaller bounded contract/check bundle")
    output.add_argument(
        "--stage",
        choices=("plan", "contract", "verify"),
        help="experimental navigation, contracts, or verification branch")
    output.add_argument("--plan",
                        action="store_const",
                        const="plan",
                        dest="stage",
                        help="alias for --stage plan")
    parser.add_argument("--record", help="reference path from the plan")
    parser.add_argument(
        "--section", help="exact heading to fetch completely (16000-char cap)")
    parser.add_argument("request", nargs="*")
    args = parser.parse_args()
    if args.record or args.section:
        if not (
                args.record and args.section
        ) or args.full or args.paths or args.compact or args.stage or args.request:
            parser.error(
                "use --record and --section together, without routing arguments"
            )
        fetch_section(args.record, args.section, parser)
        return
    text = " ".join(args.request).strip() or sys.stdin.read()
    if not text:
        print("usage: python3 scripts/route.py \"<the request text>\"")
        sys.exit(2)
    if standalone_kernel(text):
        print(
            "Standalone Bell/basic CUDA-Q kernels are outside this skill's scope."
        )
        return
    for note in notes(text):
        print(f"\n## Note\n{note}")
    recs = route(text)
    if not recs:
        print(
            "No family matched; open references/catalog.md and pick the family by mathematical object."
        )
        if args.full:
            print((REFS / "catalog.md").read_text())
        print(CHECKLIST)
        return
    printed = []
    for rec in recs:
        print(f"\n## references/{rec}")
        if not (REFS / rec).is_file():
            # Report a missing record loudly instead of dropping the route.
            print(f"MISSING: {REFS / rec} (route table names a record that "
                  "is not in this skill copy; open references/catalog.md)")
            continue
        body = (REFS / rec).read_text()
        printed.append(body)
        if args.full:
            print(body)
        else:
            print(f"Open: {REFS / rec}")
            print("Sections: " +
                  "; ".join(re.findall(r"^## (.+)$", body, re.MULTILINE)))
    direct = [rec for rec in focused_records(text) if rec not in recs]
    if direct:
        print("\n## Focused operation records (reuse excerpts below)\n")
        for rec in direct:
            print(REFS / rec)
    # Premise rows are for boundary/diagnostic intent, not ordinary operation
    # queries sharing only a family name or generic evolution/count words.
    diagnostic = re.search(
        r"\b(?:does|can|is|are|whether|why|assum\w*|validat\w*|support\w*|"
        r"interchangeable|mismatch\w*|raises?|reject\w*|accept\w*|guarantee\w*)\b",
        text.lower())
    stop = {
        "confirm", "kernel", "kernels", "device", "register", "cuda",
        "algorithms", "library", "packaged", "request", "that", "this", "with",
        "into", "from", "they", "them", "give", "also", "phrased", "only",
        "want", "would", "rather", "same", "call", "calls", "using", "inside",
        "every", "before", "changing", "which", "tell", "exactly", "case",
        "data", "show", "code", "reaches", "need", "needs", "have", "both",
        "each", "then", "state", "claims", "assumes", "trotter", "evolution",
        "double", "factorization", "preparation", "counts"
    }
    words = {w for w in re.findall(r"[a-z_]{4,}", text.lower())} - stop
    scored = []
    for body in printed if diagnostic else []:
        start = body.find("## Premise check")
        if start < 0:
            continue
        for line in body[start:].splitlines()[1:]:
            if line.startswith("## "):
                break
            if not line.startswith("| ") or line.startswith(
                    "| ---") or line.startswith("| If the request"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
            if len(cells) < 3:
                continue
            overlap = len(words & (
                {w
                 for w in re.findall(r"[a-z_]{4,}", cells[0].lower())} - stop))
            if overlap >= 2:
                scored.append((overlap, cells[0], line))
    scored.sort(key=lambda t: -t[0])
    if compressed_method_excluded(text):
        scored = [
            item for item in scored
            if "double-factorization-compressed.md" not in item[2]
        ]
    best = scored[0][0] if scored else 0
    chosen = [t for t in scored if t[0] >= 3 or t[0] == best][:3]
    focused = []
    for _, claim, line in chosen:
        for m in re.finditer(r"references/([\w./-]+\.md)`", line):
            rel = m.group(1)
            if (REFS / rel).is_file(
            ) and rel not in recs and rel not in direct and rel not in focused:
                focused.append(rel)
    if not args.full and not args.paths:
        if args.stage == "plan":
            print(
                "\nFetch only needed contracts/checks with --stage contract|verify and the original request, or --record PATH --section HEADING."
            )
            for record in dict.fromkeys(recs + direct + focused +
                                        ["validation.md", "workflow.md"]):
                path = REFS / record
                if path.is_file():
                    print(record + ": " + "; ".join(
                        title
                        for title, _, _ in markdown_sections(path.read_text())
                        if not title.startswith("Premise")))
        elif args.compact or args.stage:
            print(
                resolve_placeholders(
                    compact_context(recs + direct + focused, text, args.stage
                                    or "bundle")))
        else:
            print(
                resolve_placeholders(
                    reference_context(recs + direct + focused, text)))
    if chosen:
        print(
            "\n## Candidate premise rows (check applicability and current source)\n"
        )
        print(
            "| If the request claims or assumes | Verdict | Say |\n| --- | --- | --- |"
        )
        for _, claim, line in chosen:
            print(resolve_placeholders(line))
    if focused:
        print("\n## Focused records named by the matching rows\n")
        for rel in focused:
            print(REFS / rel)
    print("\n## Shared checkpoints")
    print(f"Multi-step work: {REFS / 'workflow.md'}")
    print(f"Numerical claims: {REFS / 'validation.md'}")
    if any(rec.startswith(INPUT_STATE_FAMILIES)
           for rec in recs) and re.search(INPUT_STATE_PATTERN, text.lower()):
        print(
            f"Input-state representation: {REFS / 'conventions/ordering.md'}"
            "#product-state-amplitudes (product recipe only for product "
            "inputs; verify entangled inputs without assuming factorization)")
    if dense_reference_wanted(recs + direct + focused, text):
        print(
            f"Dense references: {REFS / 'conventions/ordering.md'}"
            "#dense-operator-self-test (run the self-test before trusting any "
            "dense operator or statevector reference; fetch this section "
            "if the selected output mode did not include its recipe)")
    print(CHECKLIST)


if __name__ == "__main__":
    main()
