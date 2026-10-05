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
ROUTES = [  # (regex over the lowercased request, record path relative to references/)
    (r"operator pool|\bceo\b|uccgsd|upccgsd|excitation pool|amplitude optimi|optimi[sz]e the amplitudes", "state-preparation/operator-pools.md"),
    (r"hartree|\buccsd?\b|givens|slater|occupation|state.prep|prepar(e|ation)|determinant|excitation|measurement-assisted|controlled|adjoint|inject", "state-preparation/state-preparation.md"),
    (r"width|state_prep|inject|consumer|controlled|adjoint|measurement-assisted|feed-forward", "state-preparation/injection-contract.md"),
    (r"trotter|time evolution|evolve\b|product formula|suzuki|identity (term|phase|coefficient)|estimate_trotter", "trotter/trotter.md"),
    (r"qsvt|\bqsp\b|qsppack|phase sequence|phases|angles|convention|real-time recovery|recover", "qsvt/qsvt.md"),
    (r"\bwalk\b|moment|chebyshev|qubitization", "qubitization/qubitization.md"),
    (r"block.encod|paulilcu|\blcu\b|postselect|encoding protocol|controlled select", "block-encoding/block-encoding.md"),
    (r"double.factor|factoriz|\beri\b|one-norm|one norm|leaves|electron.repulsion|compress|c-df|x-df|rc-df|speedup", "double-factorization/double-factorization.md"),
    (r"pyscf|fcidump|psi4|chemistry|integrals|molecule|hamiltonian from", "chemistry/chemistry-bridges.md"),
    (r"bravyi|jordan|fermion|transform selection|parity string", "fermion-transforms/fermion-transforms.md"),
    (r"statevector|sim_utils|good.subspace|shots|simulator result|hardware", "simulation/simulation-analysis.md"),
]
# Require both a scientific object and an outcome: generic words such as
# "energy", "time", or "geometry" alone should not choose an application.
SCIENCE_ROUTES = [
    (r"geometr|bond.length|bond.angle|bond.stretch|atomic.coordinates|\batoms\b|gaussian.basis|orbital.basis|basis.set|molecul|frozen.core|active.space|mean.field|nuclear.repulsion",
     r"energ|ground|gap|spectrum|reference|overlap|qubit|correlat|hamiltonian",
     "chemistry/chemistry-bridges.md"),
    (r"spin.chain|spin.lattice|lattice|domain.wall|hubbard|ising|heisenberg|xxz|magneti[sz]|hopping",
     r"evol|quench|dynamics|\btimes?\b|time.depend|transport",
     "trotter/trotter.md"),
    (r"ancilla|herald|postselect|success.flag",
     r"probab|discard|trace|condition|observable|density|expectation",
     "simulation/simulation-analysis.md"),
]
MAX_RECORDS = 3
CHECKLIST = """
## Use the selected guides

Scientific task: read Workflow and Verification, select the requested observable
and error measure, and follow source pointers only for the chosen stages.
Contract question: check relevant premise rows against the current source.
Report the result, evidence, material boundaries, and unresolved inputs concisely.
"""


def route(text: str) -> list[str]:
    low = text.lower()
    out = []
    for pattern, rec in ROUTES:
        if re.search(pattern, low) and rec not in out and (REFS / rec).is_file():
            out.append(rec)
    if any(r.startswith("state-preparation/") for r in out) and "state-preparation/state-preparation.md" not in out[:1]:
        out = ["state-preparation/state-preparation.md"] + [r for r in out if r != "state-preparation/state-preparation.md"]
    science = [rec for obj, goal, rec in SCIENCE_ROUTES
               if re.search(obj, low) and re.search(goal, low)]
    # A named method takes priority over a default scientific path: a quench
    # with supplied QSP phases must not silently become a Trotter task.
    explicit = [rec for pattern, rec in [
        (r"qsvt|\bqsp\b|qsppack|phase sequence", "qsvt/qsvt.md"),
        (r"trotter|suzuki|product formula", "trotter/trotter.md"),
        (r"chebyshev|\bkrylov\b|qubitization|\bwalk\b", "qubitization/qubitization.md"),
        (r"double.factor|factoriz|\bc-df\b|\bx-df\b|\bleaf\b|\bleaves\b", "double-factorization/double-factorization.md"),
        (r"jordan.wigner|bravyi.kitaev", "fermion-transforms/fermion-transforms.md"),
        (r"block.encod|controlled select|paulilcu|\blcu\b", "block-encoding/block-encoding.md"),
    ] if re.search(pattern, low)]
    if science:
        # Composition points to the remaining stage guides; do not dump an
        # entire dependency tree into context for every scientific request.
        chain = ["application-composition.md"] if any(
            rec.startswith(("chemistry/", "trotter/")) for rec in science) else []
        if any(rec.startswith(("qsvt/", "qubitization/", "trotter/")) for rec in explicit):
            science = [rec for rec in science if rec != "trotter/trotter.md"]
        out = explicit + science + chain + out
    elif explicit and not re.search(r"state.prep|inject|slater|hartree|ucc|givens", low):
        out = explicit + out
    return list(dict.fromkeys(rec for rec in out if (REFS / rec).is_file()))[:MAX_RECORDS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="print complete selected guides")
    parser.add_argument("request", nargs="*")
    args = parser.parse_args()
    text = " ".join(args.request).strip() or sys.stdin.read()
    if not text:
        print("usage: python3 scripts/route.py \"<the request text>\""); sys.exit(2)
    recs = route(text)
    if not recs:
        print("No family matched; open references/catalog.md and pick the family by mathematical object.")
        if args.full:
            print((REFS / "catalog.md").read_text())
        print(CHECKLIST)
        return
    printed = []
    for rec in recs:
        body = (REFS / rec).read_text()
        printed.append(body)
        print(f"\n## references/{rec}")
        if args.full:
            print(body)
        else:
            print(f"Open: {REFS / rec}")
            print("Sections: " + "; ".join(re.findall(r"^## (.+)$", body, re.MULTILINE)))
    # focused records named by the premise-check rows that match the request: score rows by shared content words
    stop = {"confirm", "kernel", "kernels", "device", "register", "cuda", "algorithms", "library", "packaged", "request", "that", "this",
            "with", "into", "from", "they", "them", "give", "also", "phrased", "only", "want", "would", "rather", "same", "call", "calls",
            "using", "inside", "every", "before", "changing", "which", "tell", "exactly", "case", "data", "show", "code", "reaches", "need",
            "needs", "have", "both", "each", "then", "state", "claims", "assumes"}
    words = {w for w in re.findall(r"[a-z_]{4,}", text.lower())} - stop
    scored = []
    for body in printed:
        start = body.find("## Premise check")
        if start < 0:
            continue
        for line in body[start:].splitlines()[1:]:
            if line.startswith("## "):
                break
            if not line.startswith("| ") or line.startswith("| ---") or line.startswith("| If the request"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
            if len(cells) < 3:
                continue
            overlap = len(words & ({w for w in re.findall(r"[a-z_]{4,}", cells[0].lower())} - stop))
            if overlap >= 2:
                scored.append((overlap, cells[0], line))
    scored.sort(key=lambda t: -t[0])
    best = scored[0][0] if scored else 0
    chosen = [t for t in scored if t[0] >= 3 or t[0] == best][:3]
    focused = []
    for _, claim, line in chosen:
        for m in re.finditer(r"references/([\w./-]+\.md)`", line):
            rel = m.group(1)
            if (REFS / rel).is_file() and rel not in recs and rel not in focused:
                focused.append(rel)
    if chosen:
        print("\n## Candidate premise rows (check applicability and current source)\n")
        print("| If the request claims or assumes | Verdict | Say |\n| --- | --- | --- |")
        for _, claim, line in chosen:
            print(line)
    if focused:
        print("\n## Focused records named by the matching rows (open each before answering)\n")
        for rel in focused:
            print(f"cat {REFS / rel}")
    print(CHECKLIST)


if __name__ == "__main__":
    main()
