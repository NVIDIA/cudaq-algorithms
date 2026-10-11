#!/usr/bin/env python3
"""Route an in-scope request by API or scientific object and desired outcome.

Usage: python3 scripts/route.py [--full] "<the request text>"
Default: paths, section names, candidate premise rows, and focused record paths.
Open the relevant sections and their source pointers; --full prints whole guides.
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
     r"hartree|\buccs?d?\b|givens|slater|occupation|state.prep|prepar(e|ation)|determinant|excitation|measurement-assisted|inject",
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
STATE_PREP_OBJECT = r"hartree|givens|slater|\buccs?d?\b|occupation|determinant|state.prep|inject"

# Require both a scientific object and an outcome: generic words such as
# "energy", "time", or "geometry" alone should not choose an application.
# Rows: (object regex, goal regex, records, compose) where compose adds the
# application-composition guide after the stage records.
MOLECULE_OBJECT = (
    r"geometr|bond.length|bond.angle|bond.stretch|atomic.coordinates|\batoms\b|"
    r"gaussian.basis|orbital.basis|basis.set|sto-3g|6-31g|cc-pv[dtq]z|def2-|"
    r"molecul|frozen.core|active.space|mean.field|nuclear.repulsion|"
    r"\bwater\b|ammonia|methane|benzene|ethylene|hydride|hydrogen fluoride|dimer"
)
LATTICE_OBJECT = r"spin.chain|spin.lattice|lattice|domain.wall|hubbard|ising|heisenberg|xxz|magneti[sz]|hopping|half.filling"
SCIENCE_ROUTES = [
    (MOLECULE_OBJECT,
     r"energ|ground|gap|spectrum|reference|overlap|qubit|correlat|hamiltonian",
     ["chemistry/chemistry-bridges.md"], True),
    (LATTICE_OBJECT, r"evol|quench|dynamics|\btimes?\b|time.depend|transport",
     ["trotter/trotter.md"], True),
    (LATTICE_OBJECT, r"energ|ground.state|spectrum|\bgap\b|sector",
     ["qubitization/qubitization.md"], True),
    (r"ancilla|herald|postselect|success.flag",
     r"probab|discard|trace|condition|observable|density|expectation",
     ["simulation/simulation-analysis.md"], False),
]
MAX_RECORDS = 3

# Chemical formulas (LiH, H2O, N2, CH4) are molecular objects. Tokens parse
# fully into these element symbols with optional counts and have at least two
# symbols or one count; library and method acronyms that happen to parse are
# excluded explicitly.
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
_FORMULA_PART = re.compile(r"[A-Z][a-z]?\d*")


def is_chemical_formula(token: str) -> bool:
    if token in NOT_FORMULAS or not _FORMULA_TOKEN.match(token):
        return False
    parts = _FORMULA_PART.findall(token)
    if any(part.rstrip("0123456789") not in ELEMENTS for part in parts):
        return False
    return len(parts) >= 2 or any(ch.isdigit() for ch in token)


def mentions_molecule(text: str) -> bool:
    return any(
        is_chemical_formula(token)
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", text))


# Requests naming an algorithm that is not a packaged primitive get a note
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
and error measure, and follow source pointers only for the chosen stages.
Contract question: check relevant premise rows against the current source.
Report the result, evidence, material boundaries, and unresolved inputs concisely.
"""


def all_records() -> list[str]:
    """Every record path a route can print; tests check each exists."""
    out = [rec for rec, _, _ in FAMILIES
           ] + [leaf for _, leaf in PRIMITIVES_LEAVES]
    for _, _, recs, compose in SCIENCE_ROUTES:
        out.extend(recs)
        if compose:
            out.append("application-composition.md")
    return list(dict.fromkeys(out))


def notes(text: str) -> list[str]:
    low = text.lower()
    return [note for pattern, note in NOTES if re.search(pattern, low)]


def route(text: str) -> list[str]:
    low = text.lower()
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
            science.extend(r for r in recs if r not in science)
            compose = compose or chain
    # A named method takes priority over a default scientific path: a quench
    # with supplied QSP phases must not silently become a Trotter task.
    explicit = [
        rec for pattern, rec in EXPLICIT_ROUTES if re.search(pattern, low)
    ]
    if science:
        # Composition points to the remaining stage guides; do not dump an
        # entire dependency tree into context for every scientific request.
        chain = ["application-composition.md"] if compose else []
        if any(
                rec.startswith(("qsvt/", "qubitization/", "trotter/"))
                for rec in explicit):
            science = [rec for rec in science if rec != "trotter/trotter.md"]
        out = explicit + science + chain + out
    elif explicit and not re.search(STATE_PREP_OBJECT, low):
        out = explicit + out
    return list(dict.fromkeys(out))[:MAX_RECORDS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full",
                        action="store_true",
                        help="print complete selected guides")
    parser.add_argument("request", nargs="*")
    args = parser.parse_args()
    text = " ".join(args.request).strip() or sys.stdin.read()
    if not text:
        print("usage: python3 scripts/route.py \"<the request text>\"")
        sys.exit(2)
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
    # focused records named by the premise-check rows that match the request: score rows by shared content words
    stop = {
        "confirm", "kernel", "kernels", "device", "register", "cuda",
        "algorithms", "library", "packaged", "request", "that", "this", "with",
        "into", "from", "they", "them", "give", "also", "phrased", "only",
        "want", "would", "rather", "same", "call", "calls", "using", "inside",
        "every", "before", "changing", "which", "tell", "exactly", "case",
        "data", "show", "code", "reaches", "need", "needs", "have", "both",
        "each", "then", "state", "claims", "assumes"
    }
    words = {w for w in re.findall(r"[a-z_]{4,}", text.lower())} - stop
    scored = []
    for body in printed:
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
            if (REFS /
                    rel).is_file() and rel not in recs and rel not in focused:
                focused.append(rel)
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
        print(
            "\n## Focused records named by the matching rows (open each before answering)\n"
        )
        for rel in focused:
            print(f"cat {REFS / rel}")
    print(CHECKLIST)


if __name__ == "__main__":
    main()
