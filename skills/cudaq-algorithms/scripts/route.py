#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route a request to this skill's family records and print them with the answer checklist.
Usage: python3 scripts/route.py "<the request text>"
Prints, in full: the matching family front-door records (at most three), then the premise-check rows that match the
request (claim, verdict, and the bold text to say), the focused records those rows name (as cat commands to run), and the checklist."""
import re
import sys
from pathlib import Path

REFS = Path(__file__).resolve().parent.parent / "references"
ROUTES = [  # (regex over the lowercased request, record path relative to references/)
    (r"operator pool|\bceo\b|uccgsd|upccgsd|excitation pool|amplitude optimi|optimi[sz]e the amplitudes",
     "state-preparation/operator-pools.md"),
    (r"hartree|\buccsd?\b|givens|slater|occupation|state.prep|prepar(e|ation)|determinant|excitation|measurement-assisted|controlled|adjoint|inject",
     "state-preparation/state-preparation.md"),
    (r"width|state_prep|inject|consumer|controlled|adjoint|measurement-assisted|feed-forward",
     "state-preparation/injection-contract.md"),
    (r"trotter|time evolution|evolve\b|product formula|suzuki|identity (term|phase|coefficient)|estimate_trotter",
     "trotter/trotter.md"),
    (r"qsvt|\bqsp\b|qsppack|phase sequence|phases|angles|convention|real-time recovery|recover",
     "qsvt/qsvt.md"),
    (r"\bwalk\b|moment|chebyshev|qubitization",
     "qubitization/qubitization.md"),
    (r"block.encod|paulilcu|\blcu\b|postselect|encoding protocol",
     "block-encoding/block-encoding.md"),
    (r"double.factor|factoriz|\beri\b|one-norm|one norm|leaves|electron.repulsion|compress|c-df|x-df|rc-df|speedup",
     "double-factorization/double-factorization.md"),
    (r"pyscf|fcidump|psi4|chemistry|integrals|molecule|hamiltonian from",
     "chemistry/chemistry-bridges.md"),
    (r"bravyi|jordan|fermion|transform selection|parity string",
     "fermion-transforms/fermion-transforms.md"),
    (r"statevector|sim_utils|good.subspace|shots|simulator result|hardware",
     "simulation/simulation-analysis.md"),
]
MAX_RECORDS = 3
CHECKLIST = """
## Answer checklist (fill every heading)

**Record consulted:** <the record paths printed above and the focused records listed under "Focused records", which you must open>
**Premise check:** <one line per claim the request makes or assumes: "<claim> — verified | false | unknown — <what the record and source say>">
**Boundaries:** <the unsupported / absent / unverified rows that apply, restated; what the library does not check>
**Unknowns:** <what the request must still supply; bold list if the task cannot proceed without it>
"""


def route(text: str) -> list[str]:
    low = text.lower()
    out = []
    for pattern, rec in ROUTES:
        if re.search(pattern,
                     low) and rec not in out and (REFS / rec).is_file():
            out.append(rec)
    if any(r.startswith("state-preparation/") for r in
           out) and "state-preparation/state-preparation.md" not in out[:1]:
        out = ["state-preparation/state-preparation.md"] + [
            r for r in out if r != "state-preparation/state-preparation.md"
        ]
    return out[:MAX_RECORDS]


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or sys.stdin.read()
    if not text:
        print("usage: python3 scripts/route.py \"<the request text>\"")
        sys.exit(2)
    recs = route(text)
    if not recs:
        print(
            "No family matched; open references/catalog.md and pick the family by mathematical object."
        )
        print((REFS / "catalog.md").read_text())
        print(CHECKLIST)
        return
    printed = []
    for rec in recs:
        body = (REFS / rec).read_text()
        printed.append(body)
        print(f"\n===== references/{rec} =====\n")
        print(body)
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
        for line in body[start:].splitlines():
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
            "\n## Premise rows matching this request (apply them; the bold text is the answer to give)\n"
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
