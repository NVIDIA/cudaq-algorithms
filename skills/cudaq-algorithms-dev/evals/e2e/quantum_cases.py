# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quantum application contracts; numerical oracles import no tested package.

All words are ordered q0 first, with q0 the least significant state index bit.
The small four-mode pools are constructed with ladder-operator algebra, not
library pool outputs or copied Pauli expansions. Gold programs are private
preflight artifacts and must never be staged in a benchmark worker sandbox.
"""
from itertools import combinations
import textwrap

import numpy as np
from scipy.linalg import expm


def pauli_matrix(word):
    """Independent Kronecker construction in CUDA-Q's little-endian order."""
    matrices = {
        "I": np.eye(2),
        "X": np.array([[0, 1], [1, 0]]),
        "Y": np.array([[0, -1j], [1j, 0]]),
        "Z": np.diag([1, -1])
    }
    result = np.ones((1, 1), complex)
    for letter in reversed(word):
        result = np.kron(result, matrices[letter])
    return result


def _hamiltonian(p):
    return sum(float(c) * pauli_matrix(w) for c, w in p["terms"])


def _complex(p, name):
    return np.asarray(p[name + "_real"]) + 1j * np.asarray(p[name + "_imag"])


def slater_state(orbitals):
    """Exterior-product coefficients: one determinant minor per occupation."""
    q = np.asarray(orbitals, complex)
    n, electrons = q.shape
    ket = np.zeros(1 << n, complex)
    for occupied in combinations(range(n), electrons):
        ket[sum(1 << i
                for i in occupied)] = np.linalg.det(q[list(occupied), :])
    return ket


def _lower(mode, fermionic=True):
    matrix = np.zeros((16, 16), complex)
    for index in range(16):
        if index & (1 << mode):
            parity = (index &
                      ((1 << mode) - 1)).bit_count() if fermionic else 0
            matrix[index ^ (1 << mode), index] = (-1)**parity
    return matrix


def pool_generators(case_id):
    """Independent four-spin-orbital generators, in documented pool order.

    A single moves a fermion low->high: G=-i(T-T†). A canonical double
    moves an ordered occupied pair into an ordered virtual pair:
    G=+i(T-T†). UpCCGSD reverses the pair orientation. CEO uses qubit
    ladder operators (no parity strings), coupling opposite-spin exchanges.
    """
    a = [_lower(i) for i in range(4)]

    def single(low, high):
        transfer = a[high].conj().T @ a[low]
        return -1j * (transfer - transfer.conj().T)

    def double(source, target):
        low, high = source
        lo_v, hi_v = target
        transfer = a[lo_v].conj().T @ a[hi_v].conj().T @ a[high] @ a[low]
        return 1j * (transfer - transfer.conj().T)

    if case_id in ("pool_uccsd", "direct_uccsd"):
        return [single(0, 2), single(1, 3), double((0, 1), (2, 3))]
    if case_id == "pool_uccgsd":
        # Sorted descending-pair tuples: ((1,0),(3,2)), ((2,0),(3,1)),
        # ((2,1),(3,0)). Fermionic crossing signs come from Fock algebra.
        return [single(q, p) for p in range(1, 4) for q in range(p)] + [
            double((0, 1), (2, 3)),
            double((0, 2), (1, 3)),
            double((1, 2), (0, 3))
        ]
    if case_id == "pool_upccgsd":
        return [single(0, 2), single(1, 3), double((2, 3), (0, 1))]
    if case_id == "pool_ceo":
        b = [_lower(i, fermionic=False) for i in range(4)]
        alpha = b[2].conj().T @ b[0]
        beta = b[3].conj().T @ b[1]
        return [
            -1j * (alpha - alpha.conj().T), -1j * (beta - beta.conj().T),
            1j * (alpha + alpha.conj().T) @ (beta - beta.conj().T),
            1j * (alpha - alpha.conj().T) @ (beta + beta.conj().T)
        ]
    raise ValueError(f"unknown pool {case_id!r}")


def signal_response(x, phases, directions=None, convention="qsvt"):
    """Scalar 2x2 invariant-subspace product; return actual circuit phase.

    Reflect * U has diagonal -x and off-diagonals -sqrt(1-x²), +sqrt(1-x²).
    The projector convention is diag(exp(i phi),1). Tagged QSP phases
    are doubled, retaining the global phase present in the measured block.
    """
    phases = np.asarray(phases, float) * (2 if convention == "qsp" else 1)
    directions = directions if directions is not None else [0] * (len(phases) -
                                                                  1)
    s = np.sqrt(max(0., 1 - float(x)**2))
    forward = np.array([[-x, -s], [s, -x]], complex)
    state = np.array([np.exp(1j * phases[0]), 0], complex)
    for phase, direction in zip(phases[1:], directions):
        state = (forward.T if direction == 1 else forward) @ state
        state[0] *= np.exp(1j * phase)
    return state[0]


def _filtered(h, ket, alpha, phases, directions=None, convention="qsvt"):
    values, vectors = np.linalg.eigh(h / alpha)
    responses = [
        signal_response(v, phases, directions, convention) for v in values
    ]
    return vectors @ (np.asarray(responses) * (vectors.conj().T @ ket))


def case_specs():
    """Ten end-to-end application prompts, without oracle implementation."""
    base = "Read JSON from --input and save the requested fields in a non-object NPZ at --output. Use actual cudaq_algorithms primitives and CPU qpp-cpu fp64. Pauli words list q0 first, and q0 is the least significant bit of statevector indices. "
    energy_outputs = {
        "state": "prepared system statevector",
        "energy":
        "expectation of the full supplied Hamiltonian, including identity",
        "particle_number": "expected occupied spin orbitals"
    }
    specs = []

    def add(id, families, summary, task, outputs, symbols, phase=()):
        definitions = {
            "stateprep.make_givens_rotation_schedule":
            "stateprep._givens.make_givens_rotation_schedule",
            "stateprep.slater_determinant_kernel":
            "stateprep._givens.slater_determinant_kernel",
            "stateprep.get_fixed_parameter_ucc_pauli_lists":
            "stateprep._hartree_fock.get_fixed_parameter_ucc_pauli_lists",
            "stateprep.hartree_fock_ucc_kernel":
            "stateprep._hartree_fock.hartree_fock_ucc_kernel",
            "stateprep.uccsd": "stateprep._kernels.uccsd",
            "PauliLCU": "pauli_lcu.PauliLCU.__init__",
            "Walk.moments": "qubitization.Walk.moments",
            "Walk.kernel": "qubitization.Walk.kernel",
            "PhaseSequence": "qsvt.PhaseSequence.__init__",
            "QSVT.kernel": "qsvt.QSVT.kernel",
            "recover_real_time_evolution": "qsvt.recover_real_time_evolution",
        }
        for pool_name in ("uccsd", "uccgsd", "upccgsd", "ceo"):
            definitions[
                f"stateprep.make_{pool_name}_operator_pool"] = f"stateprep._pools.make_{pool_name}_operator_pool"
        spec = dict(
            id=id,
            families=[
                "simulation" if f == "simulation-analysis" else f
                for f in families
            ],
            summary=summary,
            task=base + task,
            dependencies=["cudaq", "cudaq_algorithms", "numpy", "scipy"],
            outputs=dict(outputs),
            atol=2e-9,
            rtol=2e-9,
            required_public_apis=["cudaq_algorithms." + s for s in symbols],
            required_symbols=[
                "cudaq_algorithms." + definitions.get(s, s) for s in symbols
            ])
        if phase:
            spec["phase_invariant_fields"] = list(phase)
        if id == "direct_uccsd":
            # Device functions are compiled/inlined, not Python-called. The
            # controller records decorator compile provenance under a root
            # whose real CUDA-Q execution then returns successfully.
            spec["required_kernels"] = spec["required_symbols"]
            spec["required_symbols"] = []
        specs.append(spec)

    add(
        "slater_energy", ["state-preparation", "simulation-analysis"],
        "Complex Slater determinant and electronic energy",
        "Prepare the two-electron Slater determinant specified by orthonormal columns orbitals_real + i orbitals_imag using the packaged Givens schedule and Slater preparation. Measure its energy for terms=[coefficient,word] and particle number. Return state (global phase immaterial), energy, particle_number.",
        energy_outputs, [
            "stateprep.make_givens_rotation_schedule",
            "stateprep.slater_determinant_kernel"
        ], ["state"])
    for pool in ("uccsd", "uccgsd", "upccgsd", "ceo"):
        add(
            "pool_" + pool, ["state-preparation", "simulation-analysis"],
            f"Fixed-amplitude {pool.upper()} electronic state and energy",
            f"Construct the full packaged {pool.upper()} operator pool for n_qubits=4 (CEO takes n_spatial=2; UCCSD uses electrons=2,spin=0). Starting from occupied, apply the fixed-amplitude ordered product exp(+i amplitudes[k] G[k]) in package pool order, then measure the supplied full Hamiltonian and particle number. Use the packaged fixed-parameter UCC preparation. Return state, energy, particle_number. Do not substitute the direct UCCSD circuit, whose angle convention differs.",
            energy_outputs, [
                f"stateprep.make_{pool}_operator_pool",
                "stateprep.get_fixed_parameter_ucc_pauli_lists",
                "stateprep.hartree_fock_ucc_kernel"
            ])
    add(
        "direct_uccsd", ["state-preparation", "simulation-analysis"],
        "Direct UCCSD circuit and electronic energy",
        "Prepare occupied, then execute the packaged direct stateprep.uccsd circuit with amplitudes, electrons=2,spin=0 on n_qubits=4. Parameters follow the package excitation order. Return state, energy of supplied terms, and particle_number. Preserve the direct circuit's half-angle and mixed-double signs.",
        energy_outputs, ["stateprep.uccsd"])
    add(
        "pauli_action", ["block-encoding", "simulation-analysis"],
        "Signed Pauli LCU Hamiltonian action",
        "Build PauliLCU from terms, retaining identity. Use sim_utils.action to prepare ket_real+i ket_imag, execute its encoding, and extract the unnormalized all-zero-ancilla block. Return block=H ket/alpha, action=H ket, alpha=sum absolute coefficients, success_probability=norm(block)^2, and energy=<ket|H|ket>. Preserve relative and absolute complex phases.",
        {
            "block": "unnormalized encoded state",
            "action": "full Hamiltonian action",
            "alpha": "LCU normalization",
            "success_probability": "all-zero-ancilla probability",
            "energy": "full energy"
        }, ["PauliLCU", "sim_utils.action"])
    add(
        "walk_spectrum",
        ["qubitization", "block-encoding", "simulation-analysis"],
        "Chebyshev spectral moments from a quantum walk",
        "Build a PauliLCU and Walk for terms and ket. Measure moments <T_j(H/alpha)> for j=0..count-1 through the packaged Walk API. Also execute power walk steps with ancillas uncomputed and return its unnormalized good block T_power(-H/alpha)ket. Return moments, block, alpha. Include the identity term in H and alpha.",
        {
            "moments": "Chebyshev spectral moments",
            "block": "unnormalized walk block",
            "alpha": "LCU normalization"
        }, ["Walk.moments", "Walk.kernel"])
    add(
        "qsvt_filter", ["qsvt", "block-encoding", "simulation-analysis"],
        "Complex QSVT spectral filter with directed walks",
        "Build a PauliLCU and use sim_utils.transform to execute QSVT on ket using the supplied phases, convention, and directions (0=forward,1=adjoint). Return the unnormalized all-zero-signal block, its success_probability and alpha. The phase data is supplied scientific input; no phase synthesis is requested. Preserve the actual circuit's complex phase and do not normalize postselection.",
        {
            "block": "complex unnormalized filtered state",
            "success_probability": "all-zero-signal probability",
            "alpha": "LCU normalization"
        }, ["PhaseSequence", "QSVT.kernel", "sim_utils.transform"])
    add(
        "qsvt_recovery", ["qsvt", "block-encoding", "simulation-analysis"],
        "Recover real-domain time evolution from two QSP sequences",
        "For the real Hamiltonian H=aX+bZ and real ket, execute both supplied cos_phases and sin_phases as QSP-convention QSVT sequences. They interpolate half-cosine and negative-imaginary half-sine on this Hamiltonian's two eigenvalues. Use the packaged recover_real_time_evolution helper to combine the good blocks into exp(-i H time)ket. Return cos_block, sin_block with their actual circuit phases, and evolved with absolute phase preserved. No phase synthesis is requested.",
        {
            "cos_block": "actual unnormalized cosine QSVT block",
            "sin_block": "actual unnormalized sine QSVT block",
            "evolved": "recovered full time-evolved system state"
        }, ["QSVT.kernel", "recover_real_time_evolution"])
    return specs


def parameters(case_id, variant):
    if case_id not in {s["id"] for s in case_specs()}:
        raise ValueError(f"unknown case {case_id!r}")
    if variant not in (0, 1):
        raise ValueError("variant must be 0 (public) or 1 (held out)")
    rng = np.random.default_rng(4171 + variant)
    if case_id == "slater_energy" or case_id.startswith(
            "pool_") or case_id == "direct_uccsd":
        p = {
            "n_qubits":
            4,
            "n_spatial":
            2,
            "electrons":
            2,
            "spin":
            0,
            "occupied": [0, 1],
            "terms": [[.31, "IIII"], [.6 + .07 * variant, "ZIII"],
                      [-.43, "IIZI"], [.27, "XZXI"], [-.19, "YZYI"],
                      [.14, "IIXX"], [.11, "IYXY"]]
        }
        if case_id == "slater_energy":
            raw = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
            q, _ = np.linalg.qr(raw)
            p.update(orbitals_real=q.real.tolist(),
                     orbitals_imag=q.imag.tolist())
        else:
            count = {"pool_uccgsd": 9, "pool_ceo": 4}.get(case_id, 3)
            p["amplitudes"] = (.35 * rng.normal(size=count) + .12).tolist()
        return p
    if case_id == "qsvt_recovery":
        a, b, t, angle = ((.6, .8, .63, .41), (-.52, .39, .87, -.57))[variant]
        radius, alpha = np.hypot(a, b), abs(a) + abs(b)
        cosine = float(np.arccos(np.cos(radius * t) / 2))
        sine = float(np.arcsin(alpha * np.sin(radius * t) / (2 * radius)))
        return {
            "terms": [[a, "X"], [b, "Z"]],
            "ket_real": [float(np.cos(angle)),
                         float(np.sin(angle))],
            "ket_imag": [0., 0.],
            "time": t,
            "cos_phases": [cosine],
            "sin_phases": [.17 + .03 * variant, sine - .17 - .03 * variant]
        }
    if case_id == "walk_spectrum":
        ket = np.array([.7 + .04 * variant, .3 - .09j * (variant + 1)],
                       complex)
        terms = [[.21 + .03 * variant, "I"], [.51, "X"],
                 [-.32 + .04 * variant, "Z"]]
    else:
        ket = rng.normal(size=4) + 1j * rng.normal(size=4)
        terms = [[.23, "II"], [.7 - .13 * variant, "ZI"], [-.43, "IZ"],
                 [.19, "XX"], [.11 + .07 * variant, "YZ"]]
    ket = ket / np.linalg.norm(ket)
    p = {
        "terms": terms,
        "ket_real": ket.real.tolist(),
        "ket_imag": ket.imag.tolist()
    }
    if case_id == "walk_spectrum":
        p.update(count=7, power=3 + variant)
    if case_id == "qsvt_filter":
        p.update(phases=[.23 + .07 * variant, -.51, .37, .19 - .13 * variant],
                 directions=[0, 1, 0],
                 convention="qsvt" if variant == 0 else "qsp")
    return p


def expected(case_id, params):
    """NumPy/SciPy expectations, independent of CUDA-Q and package outputs."""
    p = params
    h = _hamiltonian(p)
    if case_id == "slater_energy" or case_id.startswith(
            "pool_") or case_id == "direct_uccsd":
        if case_id == "slater_energy":
            ket = slater_state(_complex(p, "orbitals"))
        else:
            ket = np.zeros(16, complex)
            ket[sum(1 << i for i in p["occupied"])] = 1
            scales = [
                -.5, -.5, .5
            ] if case_id == "direct_uccsd" else [1] * len(p["amplitudes"])
            for theta, scale, generator in zip(p["amplitudes"], scales,
                                               pool_generators(case_id)):
                ket = expm(1j * theta * scale * generator) @ ket
        return {
            "state":
            ket,
            "energy":
            float(np.vdot(ket, h @ ket).real),
            "particle_number":
            float(sum(i.bit_count() * abs(c)**2 for i, c in enumerate(ket)))
        }
    ket = _complex(p, "ket")
    alpha = sum(abs(c) for c, _ in p["terms"])
    if case_id == "pauli_action":
        action = h @ ket
        block = action / alpha
        return {
            "block": block,
            "action": action,
            "alpha": alpha,
            "success_probability": float(np.vdot(block, block).real),
            "energy": float(np.vdot(ket, action).real)
        }
    if case_id == "walk_spectrum":
        polys = [np.eye(len(ket)), h / alpha]
        for order in range(2, max(p["count"], p["power"] + 1)):
            polys.append(2 * (h / alpha) @ polys[-1] - polys[-2])
        return {
            "moments":
            np.array([np.vdot(ket, t @ ket).real for t in polys[:p["count"]]]),
            "block": (-1)**p["power"] * polys[p["power"]] @ ket,
            "alpha":
            alpha
        }
    if case_id == "qsvt_filter":
        block = _filtered(h, ket, alpha, p["phases"], p["directions"],
                          p["convention"])
        return {
            "block": block,
            "success_probability": float(np.vdot(block, block).real),
            "alpha": alpha
        }
    if case_id == "qsvt_recovery":
        return {
            "cos_block":
            _filtered(h, ket, alpha, p["cos_phases"], convention="qsp"),
            "sin_block":
            _filtered(h, ket, alpha, p["sin_phases"], convention="qsp"),
            "evolved":
            expm(-1j * p["time"] * h) @ ket
        }
    raise ValueError(f"unknown case {case_id!r}")


_BOOTSTRAP = '''\
import argparse
import json
import numpy as np
import cudaq
from cudaq import spin
from cudaq_algorithms import stateprep, PauliLCU, Walk, QSVT, PhaseSequence, recover_real_time_evolution
from cudaq_algorithms import sim_utils as sim
cudaq.set_target("qpp-cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.input) as file:
    p = json.load(file)
def complex_input(name):
    return np.array(p[name+"_real"])+1j*np.array(p[name+"_imag"])
def observable(terms):
    total = 0.0*spin.i(0)
    for coeff, word in terms:
        op = spin.i(0)
        for index, letter in enumerate(word):
            op = op*{"I": spin.i, "X": spin.x, "Y": spin.y, "Z": spin.z}[letter](index)
        total = total+coeff*op
    return total
'''


def reference_source(case_id):
    """Actual package applications, materialized as app.py before execution."""
    if case_id not in {s["id"] for s in case_specs()}:
        raise ValueError(f"unknown case {case_id!r}")
    if case_id == "slater_energy" or case_id.startswith(
            "pool_") or case_id == "direct_uccsd":
        source = _BOOTSTRAP + 'n = int(p["n_qubits"])\n'
        if case_id == "slater_energy":
            source += 'schedule = stateprep.make_givens_rotation_schedule(complex_input("orbitals"))\nprep = stateprep.slater_determinant_kernel(schedule)\n'
        elif case_id.startswith("pool_"):
            pool = case_id.removeprefix("pool_")
            arguments = 'n, p["electrons"], p["spin"]' if pool == "uccsd" else (
                'p["n_spatial"]' if pool == "ceo" else 'n')
            source += f'pool = stateprep.make_{pool}_operator_pool({arguments})\n'
            source += 'words, coefficients = stateprep.get_fixed_parameter_ucc_pauli_lists(pool, n)\nprep = stateprep.hartree_fock_ucc_kernel(n, p["amplitudes"], words, coefficients, occupied_orbitals=p["occupied"])\n'
        else:
            source += textwrap.dedent('''\
                amplitudes = p["amplitudes"]
                occupied = p["occupied"]
                electrons, spin_value = int(p["electrons"]), int(p["spin"])
                @cudaq.kernel
                def prep(q: cudaq.qview):
                    stateprep.hartree_fock_occupation(q, occupied)
                    stateprep.uccsd(q, amplitudes, electrons, spin_value)
                ''')
        return source + textwrap.dedent('''\
            @cudaq.kernel
            def app_kernel():
                q = cudaq.qvector(n)
                prep(q)
            state = np.asarray(cudaq.get_state(app_kernel), dtype=complex)
            energy = cudaq.observe(app_kernel, observable(p["terms"])).expectation()
            number = sum(.5*(spin.i(i)-spin.z(i)) for i in range(n))
            particle_number = cudaq.observe(app_kernel, number).expectation()
            np.savez(args.output, state=state, energy=energy, particle_number=particle_number)
            ''')
    source = _BOOTSTRAP + 'enc = PauliLCU([(float(c), w) for c, w in p["terms"]])\nket = complex_input("ket")\n'
    if case_id == "pauli_action":
        return source + textwrap.dedent('''\
            block = sim.action(enc, ket)
            action = enc.alpha*block
            np.savez(args.output, block=block, action=action, alpha=enc.alpha, success_probability=float(np.vdot(block, block).real), energy=float(np.vdot(ket, action).real))
            ''')
    if case_id == "walk_spectrum":
        return source + textwrap.dedent('''\
            walk = Walk(enc)
            moments = np.asarray(walk.moments(ket, p["count"]))
            state = cudaq.get_state(walk.kernel(power=p["power"]), sim.state_from(ket))
            block = sim.good_subspace(enc, state)
            np.savez(args.output, moments=moments, block=block, alpha=enc.alpha)
            ''')
    if case_id == "qsvt_filter":
        return source + textwrap.dedent('''\
            seq = PhaseSequence(p["phases"], walk_directions=p["directions"], convention=p["convention"])
            block = sim.transform(QSVT(enc), ket, seq)
            np.savez(args.output, block=block, success_probability=float(np.vdot(block, block).real), alpha=enc.alpha)
            ''')
    return source + textwrap.dedent('''\
        transform = QSVT(enc)
        cos_block = sim.transform(transform, ket, PhaseSequence(p["cos_phases"], convention="qsp"))
        sin_block = sim.transform(transform, ket, PhaseSequence(p["sin_phases"], convention="qsp"))
        evolved = recover_real_time_evolution(cos_block, sin_block, p["cos_phases"], p["sin_phases"])
        np.savez(args.output, cos_block=cos_block, sin_block=sin_block, evolved=evolved)
        ''')
