#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route an in-scope request by API or scientific object and desired outcome.

Usage: python3 scripts/route.py [--paths | --full] "<the request text>"
Default: routes plus bounded excerpts of relevant contracts and checkpoints.
--paths prints navigation only; --full prints whole selected guides.
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
     r"double.factor|factoriz|\beri\b|one-norm|one norm|\bleaves\b|\bleaf\b|electron.repulsion|two.electron integral|c-df|x-df|rc-df",
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
    (r"compressed[_ -](double[_ -])?factorization|\b(?:rc|c)-df\b",
     "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-compressed.md"),
    (r"\breconstruct_eri\b", "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-reconstruction.md"),
    (r"\bfactorization_error\b",
     "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-error.md"),
    (r"\bmodified_one_body_integrals\b",
     "double-factorization/double-factorization.md",
     "double-factorization/double-factorization-modified-one-body.md"),
    (r"\bdouble_factorization_one_norm\b",
     "double-factorization/double-factorization.md",
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
DF_METHOD_RECORDS = (
    "double-factorization/double-factorization-explicit.md",
    "double-factorization/double-factorization-compressed.md",
)
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
    r"\b(?:little|big)[ -]endian\b|\b(?:site|qubit|basis)[ -]amplitudes?\b")

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
    (LATTICE_OBJECT, r"energ|ground.state|spectr(?:um|a)|eigenvalue|\bgaps?\b|sector",
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
    if re.search(
            r"hartree|givens|slater|\buccs?d?\b|state_prep|sim_utils|good.subspace|"
            r"\b(?:chemistry|fermion|stateprep|sim|df)\.[a-z_]\w*", low):
        return False
    return not any((re.search(obj, low) or
                    (obj == MOLECULE_OBJECT and mentions_molecule(text)))
                   and re.search(goal, low)
                   for obj, goal, _, _ in SCIENCE_ROUTES)


def affirmative_term(text: str, pattern: str) -> bool:
    """Keep local negation from adding advisory stages or focused DF recipes."""
    modifiers = (r"(?:\s+(?:\w+ly|use|using|apply|applying|need|to|for|"
                 r"asking|you|the|any|optional|additional|compressed|explicit|"
                 r"double|factorization|or|other|calculat(?:e|ing)|comput(?:e|ing)|"
                 r"evaluat(?:e|ing)|estimat(?:e|ing))){0,5}\s*$")
    clauses = re.split(
        r"[.;!?\n]|\b(?:but|then|however)\b|"
        r"\band\s+(?=(?:use|include|apply|compare|retain|keep)\b)",
        text.lower())
    for clause in clauses:
        for match in re.finditer(pattern, clause):
            prefix = clause[:match.start()]
            # A diagnostic "why not use ...?" needs the named contract.
            if re.search(r"\bwhy(?:\s+[\w-]+){0,4}\s+not" + modifiers,
                         prefix):
                return True
            # "Do not skip C-DF" is affirmative. Keep exclude/avoid local to
            # the method, not a different operation such as changing inputs.
            if re.search(r"\b(?:not|never)\s+(?:avoid|skip|omit|exclude)" +
                         modifiers, prefix):
                return True
            if not re.search(
                    r"\b(?:not|never|without|no|avoid|skip|omit|exclude)" +
                    modifiers, prefix):
                return True
    return False


def architecture_records(text: str) -> list[str]:
    """Supplement advisory stage contracts without choosing an implementation."""
    low = text.lower()
    if standalone_kernel(text) or affirmative_term(
            text, r"\b(?:run|execute|implement|benchmark|simulate|write|edit)\b"):
        return []
    stages = r"architecture|pipeline|workflow|chain|stages?|hand.?off|interfaces?|connect|through|between|fit together|compatib"
    intent = r"\barchitecture\b|\b(?:outline|sketch|design|plan|review|explain|describe)\b"
    families = {rec for rec, pattern, _ in FAMILIES if re.search(pattern, low)}
    physical = re.search(r"molecul|\beris?\b|chemist|hamiltonian|spin.chain", low)
    outcome = re.search(r"dynamics|time.depend|observable|energ|spectrum|spectral", low)
    if not (re.search(intent, low) and re.search(stages, low)
            and (len(families) >= 2 or (physical and outcome))):
        return []
    out = ["application-composition.md"]
    named_df = affirmative_term(text, r"double.factor|\b(?:rc|c|x)-df\b")
    integral = re.search(r"molecul|\beris?\b|chemist|two.electron integral", low)
    compression = affirmative_term(text, r"\bcompress\w*|\blow.rank\b")
    alternative = re.search(
        r"\bsvd\b|singular.value|tensor.train|tensor.hypercontract|\btucker\b|"
        r"\bstorage\b|\barchive|\bnpz\b|\blossless\b|file format", low)
    if named_df or (integral and compression and not alternative):
        family = "double-factorization/double-factorization.md"
        out.append(family)
        # Named operations retain their own focus. Generic compression and
        # bare DF expose the family options, not an assumed C-DF optimizer.
        out.extend(focused for pattern, starting, focused in FOCUSED_ROUTES
                   if starting == family and re.search(pattern, low) and (
                       focused not in DF_METHOD_RECORDS
                       or affirmative_term(text, pattern)))
    return list(dict.fromkeys(out))


def focused_records(text: str) -> list[str]:
    """Direct operation hints, independent of heuristic premise candidates."""
    low = text.lower()
    out = [
        focused for pattern, _, focused in FOCUSED_ROUTES
        if re.search(pattern, low) and (
            focused not in DF_METHOD_RECORDS
            or affirmative_term(text, pattern))
    ]
    out.extend(leaf for pattern, leaf in PRIMITIVES_LEAVES
               if re.search(pattern, low))
    # Planning is the relevant contract for an explicitly requested grouping.
    if re.search(r"trotter|suzuki|product formula", low) and re.search(
            r"grouping|term order|ordering", low):
        out.append("trotter/trotter-planning.md")
    if re.search(HF_PREPARATION, low):
        out.append(HF_RECORD)
    out.extend(architecture_records(text))
    return list(dict.fromkeys(out))


def state_construction(text: str) -> bool:
    """Recognize constructing states, not spectra or already prepared inputs."""
    excitation = (r"\b(?:fermionic|single|double)[ -]excitations?\b"
                  r"(?![ -]+(?:energ|gap|spectr))")
    provider = r"hartree.fock|\bhf\b|givens|slater|\bucc(?:sd|gsd)?\b|upccgsd"
    # A construction verb needs a state/preparation as its direct object.
    # Prepositions prevent e.g. "build a spectrum of the prepared state"
    # from becoming a request to construct that state.
    target = (r"(?:(?!\b(?:of|for|from|with|using|via|on|about|regarding|"
              r"already|report|table)\b)[\w-]+\s+){0,8}"
              r"(?:states?|preparations?|determinants?|references?)\b"
              r"(?=\s*(?:$|[,;.:!?]|\b(?:for|of|from|with|using|via|on|in|"
              r"under|by|and|then|to)\b))")
    operation = (r"\b(?:prepare|construct|build|create)\s+" + target +
                 r"|\b(?:apply|compose|exponentiate)\b.*" + excitation +
                 r".*\b(?:to|into|on)\s+" + target)
    for clause in re.split(r"[.;!?\n]", text.lower()):
        if not re.search(excitation + "|" + provider, clause):
            continue
        for match in re.finditer(operation, clause):
            # Keep negation local to this operation, allowing short adverb
            # modifiers without suppressing a later affirmative clause.
            if not re.search(
                    r"(?:\b(?:not|never|without)|n't|\bno\s+need\s+to)"
                    r"(?:\s+\w+ly){0,2}\s*$", clause[:match.start()]):
                return True
    return False


def route(text: str) -> list[str]:
    if standalone_kernel(text):
        return []
    low = text.lower()
    molecule = mentions_molecule(text)
    preparation = state_construction(text)
    operations = ["state-preparation/state-preparation.md"] if preparation else []
    out = []
    for pattern, rec in ROUTES:
        if re.search(pattern, low) and rec not in out:
            out.append(rec)
        if rec == "primitives/primitives.md":
            # A named leaf symbol opens its focused record after the door.
            for leaf_pattern, leaf in PRIMITIVES_LEAVES:
                if re.search(leaf_pattern, low) and leaf not in out:
                    out.append(leaf)
    if any(r.startswith("state-preparation/") for r in out) and re.search(
            STATE_PREP_OBJECT,
            low) and "state-preparation/state-preparation.md" not in out[:1]:
        out = ["state-preparation/state-preparation.md"] + [
            r for r in out if r != "state-preparation/state-preparation.md"
        ]
    science = []
    compose = False
    for obj, goal, recs, chain in SCIENCE_ROUTES:
        matched = re.search(obj, low) or (obj is MOLECULE_OBJECT and molecule)
        if matched and re.search(goal, low):
            # Computing the energy of a constructed state does not require
            # finding an unknown eigenstate. Keep named solver routes and
            # separately requested spectral outcomes; suppress only the
            # inferred solver for preparation and its diagnostic observables.
            if (recs == ["qubitization/qubitization.md"] and preparation
                    and "qubitization/qubitization.md" not in out
                    and re.search(
                        r"expectation|variance|coherenc|densit|"
                        r"\b(?:its|their|expected|mean)\s+(?:[\w-]+\s+){0,2}energ|"
                        r"energ\w*\s+of\s+(?:this|that|the)\s+state", low)
                    and not re.search(
                        r"(?:ground|excited)[ -](?:state[ -])?energ|"
                        r"(?:lowest|highest|minimum|maximum)[ -]energ|spectr(?:um|a)|"
                        r"eigenvalue|\bgaps?\b|"
                        r"\b(?:find|solve|estimate|compute|calculate|determine)\s+"
                        r"(?:[\w-]+\s+){0,4}ground[ -]state|"
                        r"ground[ -]state[ -](?:search|estimation)", low)):
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
        out = explicit + operations + science + chain + out
    else:
        # A named primitive is more specific than generic state preparation
        # or injection wording. Keep ordinary state-preparation priority when
        # no primitive was named.
        out = operations + out
        if explicit and ("primitives/primitives.md" in explicit
                         or not re.search(STATE_PREP_OBJECT, low)):
            out = explicit + out
    if any(
            rec.startswith(("qsvt/", "qubitization/", "trotter/"))
            for rec in explicit) and not (preparation or re.search(
                STATE_PREP_OBJECT, low)):
        # Spectral preparation by the named algorithm is not a request for an
        # unrelated Hartree-Fock, Givens or UCC provider recipe.
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
    if not out and architecture_records(text):
        return ["application-composition.md"]
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


def relevant_sections(
        record: str,
        body: str,
        composition_only: bool = False) -> list[tuple[str, int, str]]:
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
            selected.append((title, line, section))
            end = line + len(section.splitlines())
    return selected


def architecture_sections(record: str, body: str,
                          records: list[str], text: str) -> list[tuple[str, int, str]]:
    """Use existing seam contracts instead of broad advisory family recipes."""
    sections = markdown_sections(body)
    family = "double-factorization/double-factorization.md"
    if record == family:
        wanted = {title for title, _, _ in sections
                  if title.startswith("Shared representation - ")}
        if not any(rec in records for rec in DF_METHOD_RECORDS):
            wanted.add("Selectable operation contracts")
        return [section for section in sections if section[0] in wanted]
    selected = relevant_sections(record, body)
    if (record == "application-composition.md" and family in records
            and affirmative_term(text, r"\bcompress\w*|\blow.rank\b")):
        # One verbatim application row is enough; unrelated chains crowd out
        # the requested method contracts under the shared excerpt budget.
        header = []
        for line, value in enumerate(body.splitlines(), 1):
            if value.startswith(("| Goal |", "| --- | --- | --- |")):
                header.append(value)
            elif value.startswith("| Compressed chemistry approximation |"):
                selected.append(("Compressed chemistry chain", line,
                                 "\n".join(header + [value])))
                break
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


def physical_transform_record(records: list[str], text: str) -> str | None:
    """Prioritize a transform's complete physical check over inferred stages.

    A named downstream algorithm or preparation keeps the normal composed
    context. Bare LCU one-norm is a cost definition, not a block-encoding task.
    """
    if architecture_records(text):
        return None
    df = "double-factorization/double-factorization.md"
    fermion = "fermion-transforms/fermion-transforms.md"
    for pattern, family in EXPLICIT_ROUTES:
        if family in (df, fermion):
            continue
        if family == "block-encoding/block-encoding.md":
            pattern = pattern.replace(r"|\blcu\b", "")
        if affirmative_term(text, pattern):
            return None
    # Exact dotted APIs are matched before clause splitting; preserve their
    # focused contracts even when no broad family keyword names the stage.
    if any(family not in (df, fermion) and re.search(pattern, text, re.IGNORECASE)
           for pattern, family, _ in FOCUSED_ROUTES):
        return None
    if affirmative_term(text, r"\b(?:hartree[-_ ]fock|slater|givens|ucc\w*|state_prep)\b"):
        return None
    fermion_pattern = next(pattern for pattern, family in EXPLICIT_ROUTES
                           if family == fermion)
    named_fermion = affirmative_term(
        text, fermion_pattern + r"|\bfermion[-_ ]transforms?\b")

    def target(pattern):
        # Keep coordinated requested/negated outputs in the same match.
        return affirmative_term(text, rf"(?:{pattern})(?:\s+(?:or|and)\s+(?:{pattern}))*")

    energy = r"\b(?:(?:ground(?:[- ]state)?[- ])?energ(?:y|ies)|(?:excitation[- ])?gaps?)\b"
    if df in records and not named_fermion:
        allowed = ("double-factorization/", "chemistry/", "block-encoding/",
                   "conventions/numerical-comparison.md", "application-composition.md")
        if any(not record.startswith(allowed) for record in records):
            return None
        spectrum = (
            r"\b(?:physical|hamiltonian|many[- ]body|sector)\b.{0,45}\b(?:spectr\w*|eigenvalues?)\b|"
            r"\b(?:spectr\w*|eigenvalues?)\b.{0,45}\b(?:physical|hamiltonian|sector)\b")
        construction = r"\b(?:construct|build|assemble|diagonali[sz]\w*|project)\b.{0,70}\b(?:hamiltonian|sector)\b"
        if target(energy) or target(spectrum) or target(construction):
            return df
    if fermion in records and named_fermion and df not in records:
        allowed = ("fermion-transforms/", "trotter/", "qubitization/", "simulation/",
                   "conventions/numerical-comparison.md", "application-composition.md")
        if any(not record.startswith(allowed) for record in records):
            return None
        physical = energy + r"|\b(?:dynamic\w*|evol\w*|observables?|expectation\w*|charge|occupanc\w*|density|spectr\w*|eigenvalues?|correlat\w*)\b"
        if target(physical):
            return fermion
    return None


def physical_transform_context(records: list[str], text: str, limit: int) -> str:
    """Deliver whole existing scientific sections, including runnable checks."""
    record = physical_transform_record(records, text)
    if record is None:
        return ""
    selected = [record]
    if record.startswith("double-factorization/"):
        selected.append("chemistry/chemistry-bridges.md")
    sections = [("quick-contract.md", line, body)
                for title, line, body in quick_contract_sections(selected, text)
                if title == "Exact calls"]
    if not sections:
        return ""
    wanted = [(record, "Workflow"), (record, "Verification"),
              ("validation.md", "Requested task completion"),
              ("validation.md", "Requested output artifacts"),
              ("workflow.md", "Completion checkpoint")]
    if record.startswith("fermion-transforms/"):
        wanted.insert(0, (record, "Shared input representation"))
    for path, title in wanted:
        source = REFS / path
        if not source.is_file():
            return ""
        matches = [(path, line, body) for name, line, body in
                   markdown_sections(source.read_text()) if name == title]
        if len(matches) != 1:
            return ""  # Changed/missing contracts use the ordinary navigation.
        sections.extend(matches)
    context = ("\n## Selected reference text\n"
               "\nSection paths are relative to the skill directory.\n")
    for path, line, body in sections:
        context += f"\n### references/{path}:{line}\n{body}\n"
    context += ("\nOther selected operation records remain available at the navigation paths. "
                "Follow current source/test links for API behavior.\n")
    return context if len(context) <= limit else ""


def reference_context(records: list[str],
                      text: str,
                      limit: int = MAX_CONTEXT_CHARS) -> str:
    """Load public reference sections, never catalog answers or private grades.

    Deliver exact calls first; preserve prose contracts before fenced examples.
    Paths are relative to the skill directory, printed once by the CLI.
    This keeps content selection independent of the installation prefix.
    """
    physical_context = physical_transform_context(records, text, limit)
    if physical_context:
        return physical_context
    architecture = architecture_records(text)
    records = list(dict.fromkeys(architecture + records))
    selected = [(f"\n### references/quick-contract.md:{line}\n", body)
                for _, line, body in quick_contract_sections(records, text)]
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
        sections = (architecture_sections(record, path.read_text(), architecture, text)
                    if record in architecture else
                    relevant_sections(record, path.read_text(), composition_only))
        for title, line, body in sections:
            # Delivery requirements can come from the surrounding instructions,
            # not just this routing query. Keep their conditional procedure.
            selected.append((f"\n### references/{record}:{line}\n", body))
    if not selected:
        return ""
    header = ("\n## Selected reference text\n"
              "\nSection paths are relative to the skill directory.\n")
    footer = (
        "\nExcerpts may omit blocks or examples. Use the source paths and line "
        "numbers above to resolve any remaining contract before using it; "
        "follow current source/test links for API behavior.\n")
    notice = "\n[Additional blocks omitted; open this section when needed.]\n"
    overhead = len(header) + len(footer) + sum(
        len(label) + len(notice) + 1 for label, _ in selected)
    if overhead > limit:
        return ""  # Do not emit misleading fragments under an unusable budget.
    excerpts = [[label, markdown_blocks(body),
                 set()] for label, body in selected]
    # A per-section quota favors short boilerplate over an indivisible semantic
    # list or table. Allocate by contract priority instead, before examples.
    remaining = limit - overhead

    def section_priority(entry):
        label, blocks, _ = entry
        if "quick-contract.md:" in label:
            return 0
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
        for _, blocks, kept in priority:
            for index, block in enumerate(blocks):
                is_example = bool(re.match(r"^\s*(`{3,}|~{3,})", block))
                if (is_example == examples and index not in kept
                        and len(block) + 2 <= remaining):
                    kept.add(index)
                    remaining -= len(block) + 2
    rendered = [header]
    for label, blocks, kept in excerpts:
        rendered.append(label + "\n\n".join(
            block
            for index, block in enumerate(blocks) if index in kept) + "\n")
        if len(kept) < len(blocks):
            rendered.append(notice)
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
    parser.add_argument("request", nargs="*")
    args = parser.parse_args()
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
        context = reference_context(recs + direct + focused, text)
        if context:
            print(f"\nSkill directory: {REFS.parent}")
            print(context)
    if chosen:
        print(
            "\n## Candidate premise rows (check applicability and current source)\n"
        )
        print(
            "| If the request claims or assumes | Verdict | Say |\n| --- | --- | --- |"
        )
        for _, claim, line in chosen:
            print(line)
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
    print(CHECKLIST)


if __name__ == "__main__":
    main()
