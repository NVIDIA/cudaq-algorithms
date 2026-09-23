# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classical/chemistry applications and evaluator-only independent references.

Oracles use tensor-product Pauli matrices, occupation-basis ladder algebra,
independent FCIDUMP expansion, and independent PySCF FCI energies.
No oracle imports cudaq or cudaq_algorithms. Gold programs are never staged
for task agents; all accept the common JSON input/non-object NPZ interface.
"""
from __future__ import annotations

from functools import lru_cache
import re
import textwrap

import numpy as np
from scipy.linalg import expm


def _pauli(word):
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1])
    }
    result = np.ones((1, 1), complex)
    for letter in word[::-1]:
        result = np.kron(result, matrices[letter])
    return result


def _annihilators(modes):
    result = []
    for mode in range(modes):
        a = np.zeros((2**modes, 2**modes), complex)
        for occupation in range(2**modes):
            if occupation & (1 << mode):
                a[occupation ^ (1 << mode),
                  occupation] = (-1 if
                                 (occupation &
                                  ((1 << mode) - 1)).bit_count() % 2 else 1)
        result.append(a)
    return result


def _fock(one, two, offset):
    """Raw coefficient convention: h_ij a†_i a_j + v_ijkl a†_i a†_j a_k a_l."""
    a = _annihilators(len(one))
    result = offset * np.eye(2**len(one), dtype=complex)
    for i, j in zip(*np.nonzero(one)):
        result += one[i, j] * a[i].conj().T @ a[j]
    for i, j, k, l in zip(*np.nonzero(two)):
        result += two[i, j, k, l] * a[i].conj().T @ a[j].conj().T @ a[k] @ a[l]
    return result


def _chemist_fock(one, eri, offset):
    """Direct spatial integral second quantization, without spin tensor conversion."""
    n = len(one)
    a = _annihilators(2 * n)
    result = offset * np.eye(4**n, dtype=complex)
    for p in range(n):
        for q in range(n):
            for spin in range(2):
                result += one[p,
                              q] * a[2 * p + spin].conj().T @ a[2 * q + spin]
            for r in range(n):
                for s in range(n):
                    for sigma in range(2):
                        for tau in range(2):
                            result += .5 * eri[p, q, r, s] * (
                                a[2 * p + sigma].conj().T @ a[2 * r + tau].
                                conj().T @ a[2 * s + tau] @ a[2 * q + sigma])
    return result


def _bk_permutation(modes):
    """Occupation index -> binary indexed tree parity index via update traversal."""
    result = []
    for occupation in range(2**modes):
        parity = 0
        for mode in range(modes):
            if (occupation >> mode) & 1:
                node = mode + 1
                while node <= modes:
                    parity ^= 1 << (node - 1)
                    node += node & -node
        result.append(parity)
    return np.array(result)


def _complex(p, name):
    return np.asarray(p[name + "_real"]) + 1j * np.asarray(p[name + "_imag"])


def _encode_complex(p, name, value):
    p[name + "_real"] = np.asarray(value).real.tolist()
    p[name + "_imag"] = np.asarray(value).imag.tolist()


def _compression_integrals(p):
    theta = p["orbital_angle"]
    u = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    eri = np.einsum("pk,qk,kl,rl,sl->pqrs", u, u, p["density_core"], u, u)
    return np.asarray(p["one_body"]), eri


def _fcidump_integrals(contents):
    """Independent expansion of the benchmark's real FCIDUMP records."""
    n = int(re.search(r"\bNORB\s*=\s*(\d+)", contents, re.IGNORECASE)[1])
    body = re.split(r"&END|\$END|/", contents, maxsplit=1,
                    flags=re.IGNORECASE)[1]
    one, eri, offset = np.zeros((n, n)), np.zeros((n, n, n, n)), 0.
    for line in body.splitlines():
        if not line.strip() or line.lstrip().startswith(("!", "#")):
            continue
        value, *indices = line.split()
        value = float(value.replace("D", "E").replace("d", "e"))
        i, j, k, l = map(int, indices)
        if (i, j, k, l) == (0, 0, 0, 0):
            offset = value
        elif (k, l) == (0, 0):
            one[i - 1, j - 1] = one[j - 1, i - 1] = value
        else:
            for a, b in ((i - 1, j - 1), (j - 1, i - 1)):
                for c, d in ((k - 1, l - 1), (l - 1, k - 1)):
                    eri[a, b, c, d] = eri[c, d, a, b] = value
    return one, eri, offset


def parameters(case_id: str, variant: int) -> dict:
    if variant not in (0, 1):
        raise ValueError(
            "Only fixed public (0) and held-out (1) variants exist")
    if case_id == "trotter_dynamics":
        p = {
            "words": ["XI", "IZ", "XZ", "YY"],
            "coefficients":
            [.7, -.4, .31, .23] if variant == 0 else [-.53, .27, .41, -.19],
            "identity":
            .37 if variant == 0 else -.29,
            "time":
            .8 if variant == 0 else 1.13,
            "steps":
            3 if variant == 0 else 4,
            "order":
            2 if variant == 0 else 4
        }
        state = np.array([1, .3 + .2j, -.4j, -.2 +
                          .1j] if variant == 0 else [.2j, 1, -.3 + .4j, .2],
                         complex)
        _encode_complex(p, "state", state / np.linalg.norm(state))
        return p
    if case_id == "fermion_transport":
        h = np.diag([-.9, -.2, .35, .7]).astype(complex)
        for i, j, hopping in [(0, 1, .27 + .13j), (1, 2, -.31 + .09j),
                              (2, 3, .22 - .17j), (0, 3, .11j)]:
            h[i, j] = hopping * (1 + .23 * variant)
            h[j, i] = h[i, j].conjugate()
        v = np.zeros((4, 4, 4, 4), complex)
        v[0, 2, 2, 0], v[1, 3, 3, 1] = .42 + .1 * variant, -.18
        # Pair transport supplements density interactions and tests quartic ordering.
        v[0, 1, 3, 2], v[2, 3, 1, 0] = .08 + .05j, .08 - .05j
        ket = np.zeros(16, complex)
        ket[3], ket[5], ket[10] = 1, .4j, -.3 + .1j * variant
        p = {"offset": .31 - .12 * variant, "time": .67 + .24 * variant}
        for name, value in (("one_body", h), ("two_body", v),
                            ("state", ket / np.linalg.norm(ket))):
            _encode_complex(p, name, value)
        return p
    if case_id == "molecular_compression":
        p = {
            "orbital_angle":
            .29 + .18 * variant,
            "density_core": [[.72 + .08 * variant, .19],
                             [.19, .47 + .11 * variant]],
            "one_body": [[-1.12 + .1 * variant, .08], [.08, -.43]],
            "offset":
            .61 - .07 * variant,
            "num_electrons":
            2,
            "num_leaves":
            1
        }
        one, eri = _compression_integrals(p)
        # Store all symmetry records: unambiguous FCIDUMP, including nonzero
        # off-diagonal one-electron and three-equal-index ERI records.
        lines = ["&FCI NORB=2,NELEC=2,MS2=0,", " ORBSYM=1,1, ISYM=1,", "&END"]
        for idx in np.ndindex(eri.shape):
            lines.append(f"{eri[idx]:.17g} " +
                         " ".join(str(i + 1) for i in idx))
        for i in range(2):
            for j in range(i + 1):
                lines.append(f"{one[i,j]:.17g} {i+1} {j+1} 0 0")
        lines.append(f"{p['offset']:.17g} 0 0 0 0")
        # The task input contains FCIDUMP and scientific compression settings;
        # the generating orbital rotation and density core stay private.
        return {
            "fcidump": "\n".join(lines) + "\n",
            "num_electrons": 2,
            "num_leaves": 1
        }
    if case_id in ("pyscf_energy", "psi4_energy"):
        return {
            "distance_bohr": 1.4 if variant == 0 else 1.9,
            "basis": "sto-3g",
            "num_electrons": 2
        }
    raise KeyError(case_id)


@lru_cache(maxsize=4)
def _provider_reference(distance_bohr, basis):
    """PySCF FCI is independent of both evaluated chemistry bridges and JW.

    Both provider cases use these gauge-invariant energies. Psi4 is still a
    required runtime dependency of its application; missing Psi4 blocks it.
    """
    from pyscf import gto, scf, fci, lib
    lib.num_threads(1)
    molecule = gto.M(atom=f"H 0 0 0; H 0 0 {distance_bohr}",
                     unit="Bohr",
                     basis=basis,
                     verbose=0)
    mf = scf.RHF(molecule)
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Independent PySCF RHF did not converge")
    return {
        "ground_energy": float(fci.FCI(mf).kernel()[0]),
        "hf_energy": float(mf.e_tot),
        "nuclear_energy": 1. / distance_bohr
    }


def expected(case_id: str, params: dict) -> dict[str, object]:
    p = params
    if case_id == "trotter_dynamics":
        ket = _complex(p, "state")
        operators = [_pauli(w) for w in p["words"]]
        h = p["identity"] * np.eye(len(ket), dtype=complex)
        for coefficient, op in zip(p["coefficients"], operators):
            h += coefficient * op
        state = ket.copy()
        dt = p["time"] / p["steps"]

        def second_order(state, tau):
            terms = list(zip(p["coefficients"], operators))
            for coefficient, op in terms + terms[::-1]:
                state = expm(-.5j * tau * coefficient * op) @ state
            return state

        for _ in range(p["steps"]):
            if p["order"] == 2:
                state = second_order(state, dt)
            else:
                w1 = 1 / (2 - 2**(1 / 3))
                for scale in (w1, 1 - 2 * w1, w1):
                    state = second_order(state, scale * dt)
        state *= np.exp(-1j * p["identity"] * p["time"])
        repeat = 2 if p["order"] == 2 else 6
        return {
            "state":
            state,
            "probabilities":
            abs(state)**2,
            "exact_error":
            np.linalg.norm(state - expm(-1j * p["time"] * h) @ ket),
            "pauli_rotations":
            len(operators) * repeat * p["steps"],
            "estimated_cx_count":
            sum(2 * max(0,
                        len(w) - w.count("I") - 1)
                for w in p["words"]) * repeat * p["steps"]
        }
    if case_id == "fermion_transport":
        jw = _fock(_complex(p, "one_body"), _complex(p, "two_body"),
                   p["offset"])
        permutation = _bk_permutation(4)
        bk = np.empty_like(jw)
        bk[np.ix_(permutation, permutation)] = jw
        state = expm(-1j * p["time"] * jw) @ _complex(p, "state")
        bk_state = np.empty_like(state)
        bk_state[permutation] = state
        occupations = [
            sum(abs(state[b])**2 * ((b >> q) & 1) for b in range(16))
            for q in range(4)
        ]
        return {
            "jw_matrix": jw,
            "bk_matrix": bk,
            "jw_state": state,
            "bk_state": bk_state,
            "occupations": np.array(occupations)
        }
    if case_id == "molecular_compression":
        one, eri, offset = _fcidump_integrals(p["fcidump"])
        h = _chemist_fock(one, eri, offset)
        sector = [
            i for i in range(len(h)) if i.bit_count() == p["num_electrons"]
        ]
        kappa = one.copy()
        for i, j in np.ndindex(one.shape):
            kappa[i, j] -= .5 * sum(eri[i, r, j, r] for r in range(len(one)))
        return {
            "one_body": one,
            "eri": eri,
            "explicit_eri": eri.copy(),
            "compressed_eri": eri.copy(),
            "compressed_leaves": 1,
            "modified_one_body": kappa,
            "ground_energy": np.linalg.eigvalsh(h[np.ix_(sector, sector)])[0]
        }
    if case_id in ("pyscf_energy", "psi4_energy"):
        return dict(_provider_reference(p["distance_bohr"], p["basis"]))
    raise KeyError(case_id)


def case_specs() -> list[dict]:
    common = "Read JSON from --input and write the requested arrays/scalars to a non-object NPZ at --output. Use the actual cudaq_algorithms package. Support arbitrary values following the input schema. "
    entries = [
        ("trotter_dynamics", ["trotter"],
         "Phase-correct non-eigenstate Suzuki dynamics and cost analysis",
         "Use sim_utils.evolve to evolve the supplied normalized two-qubit state with Suzuki order/steps from the input, preserving the listed Pauli term order (word character zero is qubit zero). Include the nonzero identity shift's phase. Return final state, probabilities, Euclidean error versus exact exp(-itH) evolution, and package resource estimates for this circuit. Global phase is graded.",
         {
             "state": "complex evolved state with identity phase",
             "probabilities": "basis probabilities",
             "exact_error": "L2 error versus exact evolution",
             "pauli_rotations": "unmerged Pauli rotation estimate",
             "estimated_cx_count": "unmerged CNOT estimate"
         }, [
             "cudaq_algorithms.trotter.Trotter.__init__",
             "cudaq_algorithms.trotter.Trotter.resources",
             "cudaq_algorithms.sim_utils.evolve"
         ], []),
        ("fermion_transport", ["fermion-transforms"],
         "Interacting complex-hopping transport in JW and BK encodings",
         "Compile supplied four-mode one_body and two_body tensors in both Jordan-Wigner and Bravyi-Kitaev encodings, including offset. Raw two_body[i,j,k,l] multiplies a†_i a†_j a_k a_l (no additional 1/2). Return both dense matrices; exactly evolve the supplied occupation-basis state using exp(-itH) in each encoding and return each encoded state and physical occupation expectations ordered by mode. State indices use mode zero as least significant bit; BK uses Fenwick-tree parity bits.",
         {
             "jw_matrix": "JW Hamiltonian",
             "bk_matrix": "BK Hamiltonian",
             "jw_state": "evolved occupation-basis state",
             "bk_state": "evolved BK parity-basis state",
             "occupations": "physical mode number expectations"
         }, [
             "cudaq_algorithms.fermion._compilers.jordan_wigner",
             "cudaq_algorithms.fermion._compilers.bravyi_kitaev"
         ], []),
        ("molecular_compression", ["chemistry", "double-factorization"],
         "FCIDUMP to explicit and genuinely compressed DF to molecular energy",
         "Load the FCIDUMP text through the package chemistry API. Compute full explicit DF and actual optimized compressed DF using the requested one leaf, CPU backend. This rank-two ERI admits exact one-leaf C-DF; return parsed one-body/ERI, both reconstructed ERIs, compressed leaf count, DF-modified one-body integrals, and the lowest energy in the requested electron-number sector from the package qubit Hamiltonian built with compressed ERI and the FCIDUMP core shift.",
         {
             "one_body": "parsed spatial core integrals",
             "eri": "parsed chemist ERI",
             "explicit_eri": "full X-DF reconstruction",
             "compressed_eri": "one-leaf optimized C-DF reconstruction",
             "compressed_leaves": "actual C-DF leaf count",
             "modified_one_body": "h_pq - 1/2 sum_r (pr|qr)",
             "ground_energy":
             "fixed-electron ground energy including core shift"
         }, [
             "cudaq_algorithms.chemistry.from_fcidump",
             "cudaq_algorithms.double_factorization._factorization.explicit_double_factorization",
             "cudaq_algorithms.double_factorization._factorization.compressed_double_factorization",
             "cudaq_algorithms.double_factorization._factorization.reconstruct_eri",
             "cudaq_algorithms.double_factorization._factorization.modified_one_body_integrals",
             "cudaq_algorithms.chemistry.qubit_hamiltonian"
         ], []),
    ]
    for provider in ("pyscf", "psi4"):
        memory_note = (
            " For this tiny PySCF molecule, explicitly use in-core integrals "
            "(mol.incore_anyway=True before RHF) and verbose=0; the sandbox "
            "does not expose process-memory metadata."
            if provider == "pyscf" else "")
        entries.append((
            provider + "_energy", ["chemistry"],
            f"Real {provider} molecular provider to correlated energy",
            f"Run real {provider} restricted Hartree-Fock for neutral singlet H2 at the supplied distance in Bohr and basis (C1 symmetry). Extract integrals with chemistry.from_{provider}, then build the packaged qubit Hamiltonian including nuclear repulsion. Return its lowest energy in the two-electron sector, the HF determinant expectation of that same Hamiltonian (interleaved spins), and nuclear energy. Do not substitute another provider for the requested provider."
            + memory_note, {
                "ground_energy": "two-electron FCI energy from qubit matrix",
                "hf_energy": "HF determinant expectation from qubit matrix",
                "nuclear_energy": "nuclear repulsion"
            }, [
                f"cudaq_algorithms.chemistry.from_{provider}",
                "cudaq_algorithms.chemistry.qubit_hamiltonian"
            ], [provider] + (["pyscf"] if provider == "psi4" else [])))
    return [{
        "id":
        i,
        "families":
        families,
        "summary":
        summary,
        "task":
        common + task,
        "dependencies": ["numpy", "scipy", "cudaq", "cudaq_algorithms"] + deps,
        "outputs":
        outputs,
        "atol":
        2e-7 if i == "molecular_compression" else 2e-8,
        "rtol":
        2e-8,
        "required_symbols":
        symbols,
        "required_public_apis": [
            symbol.replace("._compilers.",
                           ".").replace("._factorization.",
                                        ".").removesuffix(".__init__")
            for symbol in symbols
        ]
    } for i, families, summary, task, outputs, symbols, deps in entries]


_BOOTSTRAP = '''\
import argparse
import json
import numpy as np
from scipy.linalg import expm
import cudaq
import cudaq_algorithms as algorithms
cudaq.set_target("qpp-cpu", precision="fp64")
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.input) as handle:
    p = json.load(handle)
def complex_input(name):
    return np.asarray(p[name+"_real"]) + 1j*np.asarray(p[name+"_imag"])
'''


def reference_source(case_id: str) -> str:
    if case_id == "trotter_dynamics":
        body = '''
from cudaq_algorithms import trotter, sim_utils
terms = list(zip(p["coefficients"], p["words"])) + [(p["identity"], "II")]
evolution = trotter.Trotter(terms)
ket = complex_input("state")
state = sim_utils.evolve(evolution, ket, time=p["time"], steps=p["steps"], order=p["order"])
resource = evolution.resources(steps=p["steps"], order=p["order"])
h = sum(c*cudaq.SpinOperator.from_word(w) for c,w in terms)
exact = expm(-1j*p["time"]*np.asarray(h.to_matrix()))@ket
np.savez(args.output, state=state, probabilities=abs(state)**2,
         exact_error=np.linalg.norm(state-exact), pauli_rotations=resource.pauli_rotations,
         estimated_cx_count=resource.estimated_cx_count)
'''
    elif case_id == "fermion_transport":
        body = '''
from cudaq_algorithms.fermion import jordan_wigner, bravyi_kitaev
h, v = complex_input("one_body"), complex_input("two_body")
jw = np.asarray(jordan_wigner(h, v, scalar_offset=p["offset"]).to_matrix())
bk = np.asarray(bravyi_kitaev(h, v, scalar_offset=p["offset"]).to_matrix())
n = len(h)
permutation = []
for index in range(2**n):
    encoded = 0
    for q in range(n):
        lo = q+1-((q+1)&-(q+1))
        bit = sum((index >> j)&1 for j in range(lo,q+1)) % 2
        encoded |= bit << q
    permutation.append(encoded)
ket = complex_input("state")
encoded_ket = np.empty_like(ket)
encoded_ket[permutation] = ket
jw_state = expm(-1j*p["time"]*jw)@ket
bk_state = expm(-1j*p["time"]*bk)@encoded_ket
occupations = [sum(abs(bk_state[permutation[b]])**2*((b >> q)&1)
                   for b in range(2**n)) for q in range(n)]
np.savez(args.output, jw_matrix=jw, bk_matrix=bk, jw_state=jw_state,
         bk_state=bk_state, occupations=occupations)
'''
    elif case_id == "molecular_compression":
        body = '''
from cudaq_algorithms import chemistry, double_factorization as df
one, eri, offset = chemistry.from_fcidump(p["fcidump"])
explicit = df.explicit_double_factorization(eri, threshold=1e-12, backend="numpy")
compressed = df.compressed_double_factorization(eri, num_leaves=p["num_leaves"],
              max_iterations=2000, tolerance=1e-12, backend="numpy")
reconstructed = df.reconstruct_eri(compressed)
h = np.asarray(chemistry.qubit_hamiltonian(one, reconstructed, scalar_offset=offset).to_matrix())
sector = [i for i in range(len(h)) if i.bit_count() == p["num_electrons"]]
np.savez(args.output, one_body=one, eri=eri, explicit_eri=df.reconstruct_eri(explicit),
         compressed_eri=reconstructed, compressed_leaves=compressed.num_leaves,
         modified_one_body=df.modified_one_body_integrals(one, eri),
         ground_energy=np.linalg.eigvalsh(h[np.ix_(sector, sector)])[0])
'''
    elif case_id in ("pyscf_energy", "psi4_energy"):
        if case_id == "pyscf_energy":
            body = '''
from pyscf import gto, scf, lib
lib.num_threads(1)
mol = gto.M(atom=f"H 0 0 0; H 0 0 {p['distance_bohr']}", unit="Bohr", basis=p["basis"], verbose=0)
mol.incore_anyway = True
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
if not mf.converged:
    raise RuntimeError("RHF did not converge")
one, eri, nuclear = algorithms.chemistry.from_pyscf(mf)
'''
        else:
            body = '''
import psi4
psi4.set_num_threads(1)
psi4.core.be_quiet()
psi4.geometry(f"0 1\\nH 0 0 0\\nH 0 0 {p['distance_bohr']}\\nunits bohr\\nsymmetry c1\\nno_reorient\\nno_com")
psi4.set_options({"basis": p["basis"], "reference": "rhf", "scf_type": "pk",
                  "e_convergence": 1e-12, "d_convergence": 1e-12})
_, wfn = psi4.energy("scf", return_wfn=True)
one, eri, nuclear = algorithms.chemistry.from_psi4(wfn)
'''
        body += '''
h = np.asarray(algorithms.chemistry.qubit_hamiltonian(one, eri, scalar_offset=nuclear).to_matrix())
sector = [i for i in range(len(h)) if i.bit_count() == p["num_electrons"]]
hf_index = (1 << p["num_electrons"])-1
np.savez(args.output, ground_energy=np.linalg.eigvalsh(h[np.ix_(sector, sector)])[0],
         hf_energy=h[hf_index,hf_index].real, nuclear_energy=nuclear)
'''
    else:
        raise KeyError(case_id)
    return _BOOTSTRAP + textwrap.dedent(body)
