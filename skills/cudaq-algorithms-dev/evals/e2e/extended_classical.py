# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped host, simulation and raw Trotter outcomes missing from the first suites.

References import no evaluated package. Private gold sources use real public
APIs and are never staged for task agents. All state phases are absolute.
"""
import textwrap

import numpy as np
from scipy.linalg import expm

from classical_cases import _chemist_fock
from quantum_cases import pauli_matrix


def case_specs():
    prefix = "cudaq_algorithms."
    common = " Pauli words list q0 first; q0 is the least significant statevector bit. Preserve absolute complex phase. "
    data = [
        ("trotter_device_surfaces", ["trotter"],
         "Plan and execute the same product-state evolution through three public circuit surfaces.",
         "Use make_trotter_terms to extract/prune terms at tolerance, reporting its original-order coefficients/words and identity separately. "
         "Construct Trotter using TrotterOrdering(ordering). Prepare q0 with Ry(angles[0]) then Rz(phase), q1 with Ry(angles[1]). "
         "Execute Trotter.kernel with injected preparation, Trotter.state_kernel with that cudaq.State, and raw apply_trotter inside a live-register kernel. "
         "All three implement the specified order/steps/time product formula without the identity phase. Return their full states and the ordered plan. "
         "Also execute raw apply_trotter with zero steps and report its unchanged state. "
         "Report raw estimate_trotter_resources fields in order [num_terms,steps,order,pauli_rotations,estimated_cx_count,identity_coefficient], "
         "both for the planned lists and after appending coefficient 0 with word II; the raw estimator counts supplied words without pruning. "
         "Encode every word numerically with I=0,X=1,Y=2,Z=3. Return three host-rejection flags for NaN time, zero steps and order 3 passed to the factory.",
         {
             "raw_coefficients": "extracted coefficients before ordering",
             "raw_words": "numeric word rows before ordering",
             "planned_coefficients": "ordered coefficients",
             "planned_words": "numeric ordered word rows",
             "identity_coefficient": "separate scalar identity",
             "injected_state": "injected factory state",
             "state_input_state": "state-taking factory state",
             "raw_state": "direct raw kernel state",
             "invalid_raw_state": "raw zero-step no-op state",
             "resource_fields": "six estimator fields",
             "zero_probe_resources":
             "six raw estimator fields including zero word",
             "host_rejections": "three rejection flags"
         }, [
             "trotter.make_trotter_terms", "TrotterOrdering",
             "trotter.Trotter.kernel", "trotter.Trotter.state_kernel",
             "trotter.apply_trotter", "trotter.estimate_trotter_resources"
         ], [
             "trotter.make_trotter_terms", "trotter.Trotter.__init__",
             "trotter.Trotter.kernel", "trotter.Trotter.state_kernel",
             "trotter.estimate_trotter_resources"
         ], ["trotter.apply_trotter"],
         "Two two-qubit product-state inputs, second/fourth-order formulas and two orderings; direct device and factory outcomes, pruning/identity, raw no-op and formula-level resources. Not controlled evolution or measured hardware cost."
         ),
        ("df_spin_diagnostics", ["chemistry", "double-factorization"],
         "DF reconstruction diagnostics through spin expansion and a two-electron energy.",
         "Construct DoubleFactorization from the supplied real orthogonal rotations and symmetric cores (method='C-DF'); these are data, not an optimizer task. "
         "Reconstruct its chemist ERI, call factorization_error against target_eri and expand one_body plus the reconstructed ERI using spin_orbital_tensors. "
         "Obtain the Fock-like eigenvalues of one_body minus half the contraction sum_r ERI[p,r,q,r] and compute double_factorization_one_norm in both lcu and burg conventions. "
         "Build the Jordan-Wigner qubit Hamiltonian of one_body/reconstructed ERI with no scalar offset; report its lowest energy in the exactly two-electron sector. "
         "Return five diagnostics: reject rank-1 one_body, mismatched ERI size, an ERI with only [0,1,0,0] increased by .2, and norm convention='unknown'; "
         "then report that a square nonsymmetric one_body (only [0,1] increased by .2) is accepted by spin_orbital_tensors. "
         "Use validation defaults. Norms are named formula-level Hamiltonian normalizations, not gate/runtime bounds.",
         {
             "reconstructed_eri": "reconstructed spatial ERI",
             "one_body_so": "interleaved-spin one-body tensor",
             "two_body_so": "ladder coefficients including half factor",
             "residual": "absolute Frobenius residual",
             "lcu_norm": "LCU formula norm",
             "burg_norm": "gauge-fixed eigenfactor formula norm",
             "two_electron_energy": "lowest fixed-sector energy",
             "diagnostics": "five ordered boundary flags"
         }, [
             "chemistry.spin_orbital_tensors",
             "double_factorization.factorization_error",
             "double_factorization.double_factorization_one_norm"
         ], [
             "chemistry.spin_orbital_tensors",
             "double_factorization._factorization.factorization_error",
             "double_factorization._factorization.double_factorization_one_norm"
         ], [],
         "Two real two-orbital/two-leaf fixtures; spin-index/half-factor tensors, nonzero residual, both named norm formulas and fixed-sector energy. Explicit validation boundaries; no optimizer robustness, encoding or hardware-resource claim."
         ),
        ("good_subspace_layout", ["simulation", "block-encoding"],
         "Extract a real encoded complex state and verify layout, copying and normalization boundaries.",
         "Build PauliLCU from terms, prepare ket_real+i ket_imag, and execute its encode_kernel using an actual cudaq.State. "
         "Call sim_utils.good_subspace directly on the full state. Return its unnormalized block, good_probability=norm(block)^2, "
         "bad_probability from the complementary full-state amplitudes, and full_norm. Also extract from the full array multiplied by scale_real+i scale_imag; "
         "return scaled_block and its block weight, which need not be a probability. "
         "Verify that mutating a returned block does not mutate the full array (copy_independent), and reject a column array, a one-element-short array, and a one-element-long array (three shape_rejections).",
         {
             "block": "unnormalized good block",
             "good_probability": "zero-ancilla probability",
             "bad_probability": "complement probability",
             "full_norm": "full squared norm",
             "scaled_block": "block from scaled full input",
             "scaled_block_weight": "scaled block squared norm",
             "copy_independent": "copy/nonalias flag",
             "shape_rejections": "three rejection flags"
         }, ["sim_utils.good_subspace", "PauliLCU"
             ], ["sim_utils.good_subspace",
                 "pauli_lcu.PauliLCU.__init__"], ["pauli_lcu.apply"],
         "Two two-qubit complex-input LCU circuits; exact good-block layout/phase, complementary probability, nonunit-input block weight, copying and shape rejection. Statevector-only, not shots or QPU execution."
         ),
    ]
    return [
        dict(id=cid,
             families=families,
             summary=summary,
             task=task + common,
             outputs=outputs,
             required_public_apis=[prefix + s for s in public],
             required_symbols=[prefix + s for s in host],
             required_kernels=[prefix + s for s in device],
             atol=2e-9,
             rtol=2e-9,
             dependencies=["cudaq", "cudaq_algorithms", "numpy", "scipy"],
             coverage_scope=scope) for cid, families, summary, task, outputs,
        public, host, device, scope in data
    ]


def parameters(case_id, variant):
    if variant not in (0, 1):
        raise ValueError("variant must be public 0 or held-out 1")
    if case_id == "trotter_device_surfaces":
        return {
            "terms": [[.3 - .38 * variant, "II"], [.61 - 1.14 * variant, "XI"],
                      [-.44 + 1.21 * variant, "ZZ"],
                      [.13 - .41 * variant, "IY"], [1e-10, "YX"], [0., "XX"],
                      [-.1, "II"]],
            "tolerance":
            1e-8,
            "ordering":
            ("preserve_input", "coefficient_magnitude_descending")[variant],
            "time":
            .72 + .21 * variant,
            "steps":
            2 + variant,
            "order": (2, 4)[variant],
            "angles": [.47 + .16 * variant, -.62 + .27 * variant],
            "phase":
            .33 - .61 * variant
        }
    if case_id == "df_spin_diagnostics":
        rotations = [
            np.array([[np.cos(t), -np.sin(t)], [np.sin(t),
                                                np.cos(t)]])
            for t in (.31 + .12 * variant, -.43 + .19 * variant)
        ]
        cores = [
            np.array([[1.2 + .2 * variant, -.24], [-.24, .6 - .11 * variant]]),
            np.array([[.27, .19 + .05 * variant], [.19 + .05 * variant, -.14]])
        ]
        p = {
            "rotations": [r.tolist() for r in rotations],
            "cores": [c.tolist() for c in cores],
            "one_body": [[-.7 - .1 * variant, .16],
                         [.16, -.22 + .07 * variant]]
        }
        eri = reconstruct(p)
        perturbation = np.array([[.7, -.2], [-.2, .4]])
        p["target_eri"] = (
            eri + (.017 + .009 * variant) *
            np.einsum("pq,rs->pqrs", perturbation, perturbation)).tolist()
        return p
    if case_id == "good_subspace_layout":
        ket = np.array([
            .4 + .1j, -.2 + .3j, .7 - .05 * variant, -.1 - .2j * (1 + variant)
        ])
        ket /= np.linalg.norm(ket)
        return {
            "terms": [[.23, "II"], [-.65 + .17 * variant, "XI"], [.41, "YZ"],
                      [.18 + .06 * variant, "IZ"]],
            "ket_real":
            ket.real.tolist(),
            "ket_imag":
            ket.imag.tolist(),
            "scale_real":
            1.8 + .2 * variant,
            "scale_imag":
            -.4 + .1 * variant
        }
    raise ValueError("unknown case: " + case_id)


def reconstruct(p):
    """Sum outer products of orbital projectors, independently of package einsum."""
    n = len(p["one_body"])
    eri = np.zeros((n, ) * 4)
    for rotation, core in zip(p["rotations"], p["cores"]):
        u = np.asarray(rotation)
        for k in range(n):
            left = np.outer(u[:, k], u[:, k])
            for l in range(n):
                right = np.outer(u[:, l], u[:, l])
                eri += core[k][l] * left[:, :, None, None] * right[None,
                                                                   None, :, :]
    return eri


def expected(case_id, p):
    if case_id == "trotter_device_surfaces":
        pairs = [(c, w) for c, w in p["terms"]
                 if c != 0 and abs(c) >= p["tolerance"]]
        identity = sum(c for c, w in pairs if set(w) == {"I"})
        raw = [(c, w) for c, w in pairs if set(w) != {"I"}]
        planned = sorted(
            raw, key=lambda item: abs(item[0]), reverse=True
        ) if p["ordering"] == "coefficient_magnitude_descending" else raw
        a, b = p["angles"]
        phase = p["phase"]
        initial = np.kron([np.cos(b / 2), np.sin(b / 2)], [
            np.cos(a / 2) * np.exp(-.5j * phase),
            np.sin(a / 2) * np.exp(.5j * phase)
        ])
        state = initial.copy()
        dt = p["time"] / p["steps"]
        weight = 1 / (2 - 2**(1 / 3))
        for _ in range(p["steps"]):
            if p["order"] == 1:
                sequence = [(dt, planned)]
            else:
                fractions = [1.] if p["order"] == 2 else [
                    weight, 1 - 2 * weight, weight
                ]
                sequence = [(dt * f / 2, terms) for f in fractions
                            for terms in (planned, list(reversed(planned)))]
            for t, terms in sequence:
                for c, word in terms:
                    state = expm(-1j * t * c * pauli_matrix(word)) @ state
        scale = p["steps"] * {1: 1, 2: 2, 4: 6}[p["order"]]
        cx = sum(2 * max(0,
                         sum(letter != "I" for letter in w) - 1)
                 for _, w in planned) * scale
        fields = [
            len(planned), p["steps"], p["order"],
            len(planned) * scale, cx, identity
        ]
        zero = [
            len(planned) + 1, p["steps"], p["order"],
            (len(planned) + 1) * scale, cx, identity
        ]
        words = lambda terms: np.array(
            [["IXYZ".index(ch) for ch in w] for _, w in terms], dtype=int)
        return {
            "raw_coefficients": np.array([c for c, _ in raw]),
            "raw_words": words(raw),
            "planned_coefficients": np.array([c for c, _ in planned]),
            "planned_words": words(planned),
            "identity_coefficient": identity,
            "injected_state": state,
            "state_input_state": state,
            "raw_state": state,
            "invalid_raw_state": initial,
            "resource_fields": np.array(fields),
            "zero_probe_resources": np.array(zero),
            "host_rejections": np.ones(3, int)
        }
    if case_id == "df_spin_diagnostics":
        one = np.asarray(p["one_body"])
        eri = reconstruct(p)
        n = len(one)
        m = 2 * n
        one_so = np.array(
            [[one[a // 2, b // 2] if a % 2 == b % 2 else 0 for b in range(m)]
             for a in range(m)], complex)
        two_so = np.zeros((m, ) * 4, complex)
        for a, b, c, d in np.ndindex(two_so.shape):
            if a % 2 == d % 2 and b % 2 == c % 2:
                two_so[a, b, c, d] = .5 * eri[a // 2, d // 2, b // 2, c // 2]
        fock_eigs = np.linalg.eigvalsh(one - .5 * np.einsum("prqr->pq", eri))
        lcu = burg = float(sum(abs(fock_eigs)))
        for core in p["cores"]:
            z = np.asarray(core)
            lcu += sum(
                abs(z[k, l]) for k in range(n)
                for l in range(k + 1, n)) + .25 * sum(abs(z.diagonal()))
            eigenvalues, vectors = np.linalg.eigh(z)
            burg += .25 * sum(
                abs(v) * sum(abs(vectors[:, k]))**2
                for k, v in enumerate(eigenvalues))
        h = _chemist_fock(one, eri, 0.)
        sector = [i for i in range(1 << m) if i.bit_count() == 2]
        return {
            "reconstructed_eri":
            eri,
            "one_body_so":
            one_so,
            "two_body_so":
            two_so,
            "residual":
            float(np.linalg.norm(np.asarray(p["target_eri"]) - eri)),
            "lcu_norm":
            lcu,
            "burg_norm":
            burg,
            "two_electron_energy":
            float(np.linalg.eigvalsh(h[np.ix_(sector, sector)])[0]),
            "diagnostics":
            np.ones(5, int)
        }
    if case_id == "good_subspace_layout":
        ket = np.asarray(p["ket_real"]) + 1j * np.asarray(p["ket_imag"])
        alpha = sum(abs(c) for c, _ in p["terms"])
        h = sum(c * pauli_matrix(w) for c, w in p["terms"])
        block = h @ ket / alpha
        probability = float(np.vdot(block, block).real)
        scaled = (p["scale_real"] + 1j * p["scale_imag"]) * block
        return {
            "block": block,
            "good_probability": probability,
            "bad_probability": 1 - probability,
            "full_norm": 1.,
            "scaled_block": scaled,
            "scaled_block_weight": float(np.vdot(scaled, scaled).real),
            "copy_independent": 1,
            "shape_rejections": np.ones(3, int)
        }
    raise ValueError("unknown case: " + case_id)


def reference_source(case_id):
    header = '''
import argparse, json
import numpy as np
import cudaq
cudaq.set_target("qpp-cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.input) as handle:
    p = json.load(handle)
out = {}
def rejected(call):
    try:
        call()
    except (ValueError, TypeError):
        return 1
    return 0
'''
    bodies = {
        "trotter_device_surfaces":
        '''
from cudaq_algorithms import trotter, sim_utils, TrotterOrdering
coefficients, words, identity, n = trotter.make_trotter_terms(p["terms"], p["tolerance"])
evolution = trotter.Trotter(p["terms"], ordering=TrotterOrdering(p["ordering"]), coefficient_tolerance=p["tolerance"])
a, b = map(float, p["angles"])
phase = float(p["phase"])
@cudaq.kernel
def prep(q: cudaq.qview):
    ry(a, q[0])
    rz(phase, q[0])
    ry(b, q[1])
@cudaq.kernel
def initial():
    q = cudaq.qvector(2)
    prep(q)
@cudaq.kernel
def raw(c: list[float], w: list[cudaq.pauli_word], t: float, steps: int, order: int):
    q = cudaq.qvector(2)
    prep(q)
    trotter.apply_trotter(c, w, t, steps, order, q)
t, steps, order = float(p["time"]), int(p["steps"]), int(p["order"])
initial_state = cudaq.get_state(initial)
out["injected_state"] = np.asarray(cudaq.get_state(evolution.kernel(t, steps, order, state_prep=prep)))
out["state_input_state"] = np.asarray(cudaq.get_state(evolution.state_kernel(t, steps, order), initial_state))
out["raw_state"] = np.asarray(cudaq.get_state(raw, evolution.coefficients, evolution.words, t, steps, order))
out["invalid_raw_state"] = np.asarray(cudaq.get_state(raw, evolution.coefficients, evolution.words, t, 0, order))
encode = lambda ws: np.asarray([["IXYZ".index(ch) for ch in w] for w in ws], dtype=int)
out.update(raw_coefficients=np.asarray(coefficients), raw_words=encode(words),
           planned_coefficients=np.asarray(evolution.coefficients), planned_words=encode(evolution.words), identity_coefficient=identity)
fields = lambda r: np.asarray([r.num_terms, r.steps, r.order, r.pauli_rotations, r.estimated_cx_count, r.identity_coefficient])
out["resource_fields"] = fields(trotter.estimate_trotter_resources(evolution.coefficients, evolution.words, steps, order, identity))
out["zero_probe_resources"] = fields(trotter.estimate_trotter_resources(evolution.coefficients+[0.], evolution.words+["II"], steps, order, identity))
out["host_rejections"] = np.asarray([rejected(lambda: evolution.kernel(float("nan"), steps, order)),
    rejected(lambda: evolution.kernel(t, 0, order)), rejected(lambda: evolution.kernel(t, steps, 3))])
''',
        "df_spin_diagnostics":
        '''
from cudaq_algorithms import chemistry
from cudaq_algorithms import double_factorization as df
one = np.asarray(p["one_body"])
n = len(one)
factor = df.DoubleFactorization(n, [np.asarray(r) for r in p["rotations"]], [np.asarray(z) for z in p["cores"]], "C-DF")
eri = df.reconstruct_eri(factor)
out["reconstructed_eri"] = eri
out["one_body_so"], out["two_body_so"] = chemistry.spin_orbital_tensors(one, eri)
out["residual"] = df.factorization_error(np.asarray(p["target_eri"]), factor)
values = np.linalg.eigvalsh(df.modified_one_body_integrals(one, eri))
out["lcu_norm"] = df.double_factorization_one_norm(factor, values, "lcu")
out["burg_norm"] = df.double_factorization_one_norm(factor, values, "burg")
h = np.asarray(chemistry.qubit_hamiltonian(one, eri).to_matrix())
sector = [i for i in range(1 << (2*n)) if i.bit_count() == 2]
out["two_electron_energy"] = np.linalg.eigvalsh(h[np.ix_(sector, sector)])[0]
bad = eri.copy(); bad[0, 1, 0, 0] += .2
asymmetric = one.copy(); asymmetric[0, 1] += .2
out["diagnostics"] = np.asarray([rejected(lambda: chemistry.spin_orbital_tensors(one.ravel(), eri)),
    rejected(lambda: chemistry.spin_orbital_tensors(one, eri[:-1])),
    rejected(lambda: chemistry.spin_orbital_tensors(one, bad)),
    rejected(lambda: df.double_factorization_one_norm(factor, values, "unknown")),
    1-rejected(lambda: chemistry.spin_orbital_tensors(asymmetric, eri))])
''',
        "good_subspace_layout":
        '''
from cudaq_algorithms import PauliLCU, sim_utils
enc = PauliLCU(p["terms"])
ket = np.asarray(p["ket_real"])+1j*np.asarray(p["ket_imag"])
full = np.asarray(cudaq.get_state(enc.encode_kernel(), sim_utils.state_from(ket)), dtype=np.complex128)
block = sim_utils.good_subspace(enc, full)
scaled = sim_utils.good_subspace(enc, full*(p["scale_real"]+1j*p["scale_imag"]))
probe = sim_utils.good_subspace(enc, full)
before = full.copy(); probe[0] += .3+.2j
out.update(block=block, good_probability=np.vdot(block, block).real,
    bad_probability=np.vdot(full[len(block):], full[len(block):]).real,
    full_norm=np.vdot(full, full).real, scaled_block=scaled,
    scaled_block_weight=np.vdot(scaled, scaled).real,
    copy_independent=int(not np.shares_memory(probe, full) and np.array_equal(before, full)))
out["shape_rejections"] = np.asarray([rejected(lambda: sim_utils.good_subspace(enc, full[:, None])),
    rejected(lambda: sim_utils.good_subspace(enc, full[:-1])),
    rejected(lambda: sim_utils.good_subspace(enc, np.append(full, 0.)))])
''',
    }
    if case_id not in bodies:
        raise ValueError("unknown case: " + case_id)
    return textwrap.dedent(header + bodies[case_id] +
                           "\nnp.savez(args.output, **out)\n")
