"""Private molecular/DF oracles using only PySCF and NumPy.

No CUDA-Q Algorithms result participates in these references. Molecular
integrals are generated afresh, and a complete fixed-(Nalpha,Nbeta) FCI matrix
is diagonalized. Candidate DF spectra are evaluated from the returned factors,
with the original physical one-body operator and scalar retained.
"""

from functools import lru_cache
from math import comb

import numpy as np

from .common import CheckFailure, close, real_array, require

_DF_FIELDS = {
    "reference_ground_energy":
    "number, total uncompressed sector ground energy (Ha)",
    "reference_excitation_gap":
    "number, uncompressed sector gap (Ha)",
    "gap_convention":
    "first_level (E[1]-E[0], counting multiplicity) or first_distinct (first E above the ground eigenspace minus E[0]); all states in the stated Nalpha,Nbeta sector, without an extra singlet restriction",
    "gap_degeneracy_tolerance":
    "optional positive number <=1e-7 Ha, default 1e-8; eigenvalues within this absolute distance of E[0] count as the ground eigenspace for first_distinct",
    "candidates":
    "nonempty list of objects: id (unique string), method (explicit or optimized), num_leaves (integer), rotations (num_leaves,4,4 real orthogonal U), cores (num_leaves,4,4 real symmetric Z), reconstructed_eri (4,4,4,4 real, sum_tkl U[t,p,k]U[t,q,k]Z[t,k,l]U[t,r,l]U[t,s,l]), ground_energy and excitation_gap (numbers, Ha). Explicit candidates also give first_factorization (cholesky or eigendecomposition), threshold and second_factor_threshold (nonnegative numbers, default 1e-8 and 0), and optionally retained_leaf_indices (list of indices into the descending-eigenvalue or pivoted-Cholesky leaves; default first num_leaves). State optimization and initialization settings in the artifact/report.",
    "selected_candidate":
    "candidate id meeting the requested errors; null is allowed only in the optimized-tradeoff case when no computed candidate qualifies",
}

CASES = {
    "science-chemistry-reference-quality": {
        "output": {
            "ground_energy":
            "number, total specified LiH active-space ground energy in Ha",
            "reference_energy":
            "number, total energy of the compatible canonical restricted HF reference in Ha",
            "reference_overlap_squared":
            "number, normalized HF weight in the active-space ground eigenspace",
        },
        "oracle":
        "Independent PySCF RHF integrals, explicit frozen-core contraction, full (1,1) active-sector diagonalization and HF ground-space projection.",
        "tolerances": {
            "energy_atol_ha": 1e-6,
            "overlap_atol": 1e-6
        },
    },
    "science-chemistry-water-bending": {
        "output": {
            "geometries":
            "two objects with angle_degrees (90 or 104.5), ground_energy (total active-space Ha), rhf_energy (total RHF Ha)",
            "difference_convention":
            "104.5-minus-90 or 90-minus-104.5",
            "active_energy_difference":
            "number, signed total active-space energy difference in Ha",
            "correlation_contribution":
            "number, same signed difference in (ground_energy-rhf_energy), Ha",
            "preferred_angle_degrees":
            "number, lower correlated-energy angle, 90 or 104.5",
            "correlation_changes_preference":
            "boolean, whether correlated and RHF energy differences have different signs",
        },
        "oracle":
        "Independent geometry-specific PySCF RHF and frozen-core (2,2) FCI, including nuclear and frozen-core constants.",
        "tolerances": {
            "energy_and_difference_atol_ha": 1e-5
        },
    },
    "science-double-factorization-leaf-budget": {
        "output":
        dict(
            _DF_FIELDS,
            minimality_scope=
            "sufficient_only or evaluated_candidates; the latter claims the smallest sufficient leaf count only among the submitted candidates"
        ),
        "oracle":
        "Seeded NumPy model; returned U/Z reconstruction; independent explicit-decomposition consistency and full sector FCI spectra for every candidate, without optimizing the answer on its behalf.",
        "tolerances": {
            "spectral_error_max_ha": 1e-3,
            "reported_value_atol_ha": 1e-7,
            "tensor_atol": 1e-8
        },
    },
    "science-double-factorization-optimized-tradeoff": {
        "output":
        dict(
            _DF_FIELDS,
            candidates=_DF_FIELDS["candidates"] +
            " Include at least one candidate for each (explicit/optimized)x(2/3 leaves). Each also reports one_body_eigenvalues (4 real eigenvalues of F=h-0.5*sum_r ERI[p,r,q,r]+sum_r ERI[p,q,r,r]), identity_coefficient (c+trace(h)+0.5*sum_pq ERI[p,p,q,q]-0.25*sum_pq ERI[p,q,p,q]), and one_norm (number under the declared convention).",
            norm_convention=
            "lcu: sum|eig(F)|+sum_t(sum_k<l|Z[t,k,l]|+0.25*sum_k|Z[t,k,k]|); or burg: sum|eig(F)|+0.25*sum_ti |eig_i(Z[t])|*(sum_k|eigvec_ki(Z[t])|)^2. A custom valid DF estimate may use custom_lcu, supplying each candidate's unitary_lcu evidence as described below.",
            scalar_handling=
            "excluded (known identity shift omitted from estimate) or included_absolute (add abs(identity_coefficient)); identical across all candidates",
            custom_lcu=
            "Only for norm_convention=custom_lcu: each candidate adds unitary_lcu={coefficients:[real], unitaries:[term,256,256,2 real/imag], identity_shift:number}. Sum_j coefficients[j]*unitaries[j] + identity_shift*I must equal the full candidate Hamiltonian (interleaved spin orbitals, q0 least significant); one_norm=sum|coefficients|. Excluded uses identity_shift=identity_coefficient; included_absolute uses identity_shift=0. All terms must be unitary. This permits other valid declared LCU formulas without accepting an unverifiable claimed norm."
        ),
        "oracle":
        "Reconstruct every submitted candidate, independently evaluate full sector spectra and declared DF normalization, then select among admissible computed candidates. No single optimized solution or universal optimum is prescribed.",
        "tolerances": {
            "spectral_error_strict_ha": 1e-3,
            "reported_value_atol": 1e-7,
            "tensor_atol": 1e-8
        },
    },
    "science-workflow-molecular-bond-stretching": {
        "output": {
            "geometries":
            "two objects: distance_angstrom (1.0 or 1.8), ground_energy (reported algorithm total energy, Ha), rhf_energy (total RHF Ha), reference_quality ({kind:'ground_space_weight'|'energy_variance'|'spectral_weights'|'energy_error', value:number or full sorted sector spectral-weight array}; energy_variance is in Ha^2, energy_error is reference energy minus exact ground energy in Ha; repeated degenerate eigenvalues use equal shares of eigenspace weight). Each geometry supplies either (a) convergence (at least two {work:nonnegative number, energy:total estimate in Ha} rows at distinct work amounts) and selected_convergence_index (index of row supplying ground_energy), or (b) convergence_analysis={work:nonnegative number in the declared work_unit, absolute_error_bound:nonnegative number <=0.0016 Ha bounding the absolute error of ground_energy, description:nonempty explanation of the quantitative analysis/certificate and how reference quality determines the work}. The bound is checked against the independently computed same-model ground energy with 1e-10 Ha numerical slack; the rubric reviews its derivation and resource argument. If both evidence forms are supplied, both are validated. By default the reference diagnostic uses canonical restricted HF. To study another HF determinant, supply occupied_alpha_ao and occupied_beta_ao (each 4x2 real occupied orbital coefficients in the atom-ordered STO-3G AO basis, each normalized in the AO overlap metric) and reference_energy (its total expectation in Ha); the diagnostic is checked against that actual determinant. Explain any spin-symmetry implications in the report.",
            "method":
            "nonempty description of the evaluated quantum-energy procedure or quantitative convergence analysis; substantive implementation is also reviewed",
            "work_unit":
            "nonempty definition of the work numbers, distinguishing measured work and scoped resource estimates in the report",
        },
        "oracle":
        "Independent all-orbital PySCF RHF/FCI at both distances; HF spectral weight or variance and final algorithm estimates compared to the same physical model; either numerical convergence rows or quantitative analysis with a bound covering the actual final energy error.",
        "tolerances": {
            "energy_atol_ha": 1.6e-3,
            "reference_energy_atol_ha": 1e-6,
            "reference_quality_atol": 1e-6,
            "analysis_bound_slack_ha": 1e-10
        },
    },
}


def _number(payload, key):
    return float(real_array(payload, key, ()))


def _text(payload, key):
    require(isinstance(payload, dict), "expected an object")
    value = payload.get(key)
    require(
        isinstance(value, str) and value.strip(), f"missing/non-string {key}")
    return value


def _configure_pyscf():
    from pyscf import lib
    lib.num_threads(1)
    try:
        lib.current_memory()
    except OSError:
        # Some process sandboxes do not mount /proc. This only supplies the
        # memory accounting used to choose in-core paths, never a physical
        # quantity or convergence condition. Linux ru_maxrss is in KiB.
        import resource

        def memory_without_proc():
            megabytes = resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss / 1024
            return megabytes, megabytes

        lib.current_memory = memory_without_proc


def _spectrum(h, eri, scalar, nelec):
    from pyscf import fci
    _configure_pyscf()
    n = len(h)
    dim_a, dim_b = (comb(n, k) for k in nelec)
    dim = dim_a * dim_b
    effective = fci.direct_spin1.absorb_h1e(h, eri, n, nelec, .5)
    columns = []
    for index in range(dim):
        vector = np.zeros((dim_a, dim_b))
        vector.flat[index] = 1.
        columns.append(
            fci.direct_spin1.contract_2e(effective, vector, n, nelec).ravel())
    matrix = np.array(columns).T + scalar * np.eye(dim)
    close(matrix, matrix.T, 1e-10, "private FCI Hermiticity")
    return np.linalg.eigh(matrix)


@lru_cache(maxsize=5)
def _molecular_reference(name, coordinate=None):
    from pyscf import ao2mo, gto, scf
    _configure_pyscf()
    if name == "lih":
        atom, frozen, active, nelec = "Li 0 0 0; H 0 0 1.6", 1, 4, (1, 1)
    elif name == "water":
        angle = np.deg2rad(coordinate / 2)
        x, z = .958 * np.sin(angle), .958 * np.cos(angle)
        atom = [("O", (0, 0, 0)), ("H", (x, 0, z)), ("H", (-x, 0, z))]
        frozen, active, nelec = 3, 4, (2, 2)
    else:
        require(name == "h4", "unknown molecular reference")
        atom = [("H", (0, 0, j * coordinate)) for j in range(4)]
        frozen, active, nelec = 0, 4, (2, 2)
    mol = gto.M(atom=atom,
                unit="Angstrom",
                basis="sto-3g",
                charge=0,
                spin=0,
                verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    require(mf.converged, "private RHF oracle failed to converge")
    mo = mf.mo_coeff[:, np.argsort(mf.mo_energy)]
    h = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo), mo.shape[1])
    scalar = mol.energy_nuc()
    for i in range(frozen):
        scalar += 2 * h[i, i]
        for j in range(frozen):
            scalar += 2 * eri[i, i, j, j] - eri[i, j, j, i]
    effective = h[frozen:frozen + active, frozen:frozen + active].copy()
    for p in range(active):
        for q in range(active):
            for i in range(frozen):
                effective[p,
                          q] += 2 * eri[p + frozen, q + frozen, i,
                                        i] - eri[p + frozen, i, i, q + frozen]
    sl = slice(frozen, frozen + active)
    energies, vectors = _spectrum(effective, eri[sl, sl, sl, sl], scalar,
                                  nelec)
    weights = _degeneracy_weights(energies, np.abs(vectors[0, :])**2)
    # The occupied lowest-energy active orbitals form determinant (0,0) in
    # PySCF's alpha/beta string ordering. Degenerate eigenvectors are arbitrary.
    hf = float(np.dot(weights, energies))
    close(hf, mf.e_tot, 2e-9, "private HF/core consistency")
    return {
        "ground": float(energies[0]),
        "hf": hf,
        "weight": float(weights[energies - energies[0] <= 1e-8].sum()),
        "variance": float(np.dot(weights, (energies - hf)**2)),
        "spectral_weights": weights,
        "eigenvalues": energies,
        "eigenvectors": vectors,
        "mo": mo,
        "ao_overlap": mf.get_ovlp()
    }


def _degeneracy_weights(energies, weights):
    weights = weights.copy()
    for i in range(len(energies)):
        group = np.abs(energies - energies[i]) <= 1e-8
        if np.sum(group) > 1:
            weights[group] = np.sum(weights[group]) / np.sum(group)
    return weights


def _reference_quality(ref, row):
    if "occupied_alpha_ao" not in row and "occupied_beta_ao" not in row:
        return dict(ref, energy_error=ref["hf"] - ref["ground"])
    from pyscf.fci import cistring
    alpha = real_array(row, "occupied_alpha_ao", (4, 2))
    beta = real_array(row, "occupied_beta_ao", (4, 2))
    metric = ref["ao_overlap"]
    close(alpha.T @ metric @ alpha, np.eye(2), 1e-8,
          "alpha occupied AO normalization")
    close(beta.T @ metric @ beta, np.eye(2), 1e-8,
          "beta occupied AO normalization")
    strings = cistring.make_strings(range(4), 2)
    occupations = [[q for q in range(4) if int(s) & (1 << q)] for s in strings]
    coefficients = []
    for occupied in (alpha, beta):
        overlap = ref["mo"].T @ metric @ occupied
        coefficients.append(
            np.array([
                np.linalg.det(overlap[indices, :]) for indices in occupations
            ]))
    vector = np.outer(*coefficients).ravel()
    close(
        np.vdot(vector, vector).real, 1., 1e-8, "HF determinant normalization")
    energies = ref["eigenvalues"]
    weights = _degeneracy_weights(
        energies,
        np.abs(ref["eigenvectors"].conj().T @ vector)**2)
    energy = float(np.dot(weights, energies))
    close(_number(row, "reference_energy"), energy, 1e-6,
          "declared HF determinant energy")
    return {
        "weight": float(weights[energies - energies[0] <= 1e-8].sum()),
        "variance": float(np.dot(weights, (energies - energy)**2)),
        "spectral_weights": weights,
        "energy_error": energy - ref["ground"]
    }


@lru_cache(maxsize=2)
def _model(seed):
    rng = np.random.default_rng(seed)
    weights = [.04] * 3 if seed == 7 else [.30, .12, .04, .01]
    eri = np.zeros((4, 4, 4, 4))
    for weight in weights:
        a = rng.standard_normal((4, 4))
        s = (a + a.T) / 2
        eri += weight * np.einsum("pq,rs->pqrs", s, s)
    h = np.diag([-.9, -.5, .1, .4] if seed == 7 else [-.8, -.4, .15, .5])
    edges = [(0, 1, .12), (2, 3, .08)] if seed == 7 else [(0, 1, .08),
                                                          (1, 2, .08),
                                                          (2, 3, .08)]
    for p, q, entry in edges:
        h[p, q] = h[q, p] = entry
    return h, eri, .3 if seed == 7 else .2, (1, 1) if seed == 7 else (2, 2)


def _gap(energies, convention, tolerance):
    if convention == "first_level":
        return float(energies[1] - energies[0])
    differences = energies - energies[0]
    distinct = differences[differences > tolerance]
    require(distinct.size > 0,
            "no excited level under the declared degeneracy tolerance")
    return float(distinct[0])


def _explicit_eri(original, row, leaves):
    """Validate a declared explicit family, allowing both supported paths.

    Comparison uses reconstructed tensors so eigenvector signs, orbital order
    within each leaf and equivalent zero modes do not affect acceptance.
    """
    family = row.get("first_factorization")
    require(family in ("cholesky", "eigendecomposition"),
            "declare the explicit first_factorization")
    threshold = _number(row, "threshold") if "threshold" in row else 1e-8
    second = _number(
        row,
        "second_factor_threshold") if "second_factor_threshold" in row else 0.
    require(threshold >= 0 and second >= 0,
            "factorization thresholds must be nonnegative")
    matrix = original.reshape(16, 16)
    factors = []
    if family == "eigendecomposition":
        values, vectors = np.linalg.eigh(matrix)
        order = np.argsort(np.abs(values))[::-1]
        floor = max(threshold, abs(values[order[0]]) * 1e-14)
        for i in order:
            if abs(values[i]) > floor:
                factors.append((vectors[:, i].reshape(4, 4), values[i]))
    else:
        remainder = matrix.copy()
        floor = max(threshold, np.max(np.diag(matrix)) * 1e-14)
        for _ in range(16):
            pivot = int(np.argmax(np.diag(remainder)))
            value = remainder[pivot, pivot]
            if value <= floor:
                break
            vector = remainder[:, pivot] / np.sqrt(value)
            factors.append((vector.reshape(4, 4), 1.))
            remainder -= np.outer(vector, vector)
    indices = row.get("retained_leaf_indices", list(range(leaves)))
    require(
        isinstance(indices, list) and len(indices) == leaves
        and all(type(i) is int and 0 <= i < len(factors)
                for i in indices) and len(set(indices)) == leaves,
        "invalid retained explicit leaf indices")
    result = np.zeros((4, 4, 4, 4))
    for index in indices:
        factor, scale = factors[index]
        values, vectors = np.linalg.eigh((factor + factor.T) / 2)
        if second > 0:
            values = np.where(
                abs(scale) * np.abs(values).sum() * np.abs(values) > second,
                values, 0.)
        truncated = (vectors * values) @ vectors.T
        result += scale * np.einsum("pq,rs->pqrs", truncated, truncated)
    return result


def _full_hamiltonian(h, eri, scalar):
    # Sparse occupation action independently builds the full JW matrix for
    # optional custom LCU evidence; q0 is least significant.
    result = scalar * np.eye(256)
    terms = []
    for p in range(4):
        for q in range(4):
            for sigma in (0, 1):
                terms.append((h[p, q], [(2 * q + sigma, False),
                                        (2 * p + sigma, True)]))
            for r in range(4):
                for s in range(4):
                    for sigma in (0, 1):
                        for tau in (0, 1):
                            terms.append((.5 * eri[p, q, r, s],
                                          [(2 * q + sigma, False),
                                           (2 * s + tau, False),
                                           (2 * r + tau, True),
                                           (2 * p + sigma, True)]))
    for coefficient, operations in terms:
        if not coefficient:
            continue
        for column in range(256):
            state, sign = column, 1
            for orbital, creation in operations:
                occupied = bool(state & (1 << orbital))
                if occupied == creation:
                    break
                sign *= (-1)**((state & ((1 << orbital) - 1)).bit_count())
                state ^= 1 << orbital
            else:
                result[state, column] += coefficient * sign
    return result


def _norm(row, h, eri, scalar, cores, convention, scalar_handling):
    f = h - .5 * np.einsum("prqr->pq", eri) + np.einsum("pqrr->pq", eri)
    eigenvalues = np.linalg.eigvalsh(f)
    supplied = real_array(row, "one_body_eigenvalues", (4, ))
    close(np.sort(supplied), eigenvalues, 1e-7,
          "candidate Fock-like eigenvalues")
    identity = float(scalar + np.trace(h) + .5 * np.einsum("ppqq->", eri) -
                     .25 * np.einsum("pqpq->", eri))
    close(_number(row, "identity_coefficient"), identity, 1e-7,
          "candidate identity coefficient")
    norm = np.abs(eigenvalues).sum()
    if convention == "lcu":
        norm += sum(
            np.abs(np.triu(z, 1)).sum() + .25 * np.abs(np.diag(z)).sum()
            for z in cores)
    elif convention == "burg":
        for z in cores:
            values, vectors = np.linalg.eigh(z)
            norm += .25 * np.dot(np.abs(values),
                                 np.abs(vectors).sum(axis=0)**2)
    else:
        from .common import complex_array
        evidence = row.get("unitary_lcu", {})
        require(isinstance(evidence, dict),
                "custom LCU evidence must be an object")
        raw = evidence.get("coefficients", [])
        require(
            isinstance(raw, list) and 0 < len(raw) <= 512,
            "custom LCU requires 1 to 512 explicit terms")
        coefficients = real_array(evidence, "coefficients", (len(raw), ))
        matrices = complex_array(evidence, "unitaries", (len(raw), 256, 256))
        shift = _number(evidence, "identity_shift")
        close(shift, identity if scalar_handling == "excluded" else 0., 1e-7,
              "custom identity bookkeeping")
        reconstructed = shift * np.eye(256, dtype=complex)
        for coefficient, matrix in zip(coefficients, matrices):
            close(matrix.conj().T @ matrix, np.eye(256), 1e-8,
                  "custom LCU unitary")
            reconstructed += coefficient * matrix
        close(reconstructed, _full_hamiltonian(h, eri, scalar), 1e-7,
              "custom LCU Hamiltonian")
        return float(np.abs(coefficients).sum())
    if scalar_handling == "included_absolute":
        norm += abs(identity)
    return float(norm)


def _check_df(payload, optimized):
    h, original, scalar, nelec = _model(31 if optimized else 7)
    convention = payload.get("gap_convention")
    require(convention in ("first_level", "first_distinct"),
            "unsupported or absent gap_convention")
    tolerance = _number(payload, "gap_degeneracy_tolerance"
                        ) if "gap_degeneracy_tolerance" in payload else 1e-8
    require(0 < tolerance <= 1e-7,
            "gap_degeneracy_tolerance must be in (0,1e-7]")
    energies, _ = _spectrum(h, original, scalar, nelec)
    ground, gap = float(energies[0]), _gap(energies, convention, tolerance)
    close(_number(payload, "reference_ground_energy"), ground, 1e-7,
          "original ground energy")
    close(_number(payload, "reference_excitation_gap"), gap, 1e-7,
          "original gap")
    rows = payload.get("candidates")
    require(
        isinstance(rows, list) and 0 < len(rows) <= 100,
        "submit 1 to 100 factorization candidates")
    norm_convention, scalar_handling = payload.get(
        "norm_convention"), payload.get("scalar_handling")
    if optimized:
        require(norm_convention in ("lcu", "burg", "custom_lcu"),
                "declare a supported DF norm convention")
        require(scalar_handling in ("excluded", "included_absolute"),
                "declare scalar_handling")
    results, seen = {}, set()
    for row in rows:
        name = _text(row, "id")
        require(name not in results, "duplicate candidate id")
        method, count = row.get("method"), row.get("num_leaves")
        require(
            method in ("explicit",
                       "optimized") if optimized else method == "explicit",
            "incorrect factorization method")
        require(
            type(count) is int
            and (count in (2, 3) if optimized else 0 <= count <= 16),
            "incorrect leaf count")
        if count:
            u = real_array(row, "rotations", (count, 4, 4))
            z = real_array(row, "cores", (count, 4, 4))
        else:
            require(
                row.get("rotations") == [] and row.get("cores") == [],
                "zero-leaf factorization requires empty rotations and cores")
            u, z = np.empty((0, 4, 4)), np.empty((0, 4, 4))
        close(np.einsum("tpi,tpj->tij", u, u), np.tile(np.eye(4),
                                                       (count, 1, 1)), 1e-8,
              "leaf rotations orthogonality")
        close(z, z.transpose(0, 2, 1), 1e-8, "leaf core symmetry")
        reconstructed = np.einsum("tpk,tqk,tkl,trl,tsl->pqrs",
                                  u,
                                  u,
                                  z,
                                  u,
                                  u,
                                  optimize=True)
        close(real_array(row, "reconstructed_eri", (4, 4, 4, 4)),
              reconstructed, 1e-8, "factor-derived ERI")
        if method == "explicit":
            for core in z:
                singular = np.linalg.svd(core, compute_uv=False)
                require(singular[1] <= 1e-8,
                        "explicit leaf core must have rank at most one")
            close(reconstructed, _explicit_eri(original, row, count), 1e-8,
                  "declared explicit factorization")
        candidate_energies, _ = _spectrum(h, reconstructed, scalar, nelec)
        e = float(candidate_energies[0])
        g = _gap(candidate_energies, convention, tolerance)
        close(_number(row, "ground_energy"), e, 1e-7,
              f"{name} actual ground energy")
        close(_number(row, "excitation_gap"), g, 1e-7, f"{name} actual gap")
        norm = _norm(row, h, reconstructed, scalar, z, norm_convention,
                     scalar_handling) if optimized else None
        if optimized:
            close(_number(row, "one_norm"), norm, 1e-7, f"{name} norm")
        error = max(abs(e - ground), abs(g - gap))
        results[name] = {
            "count": count,
            "valid": error < 1e-3 if optimized else error <= 1e-3,
            "norm": norm
        }
        seen.add((method, count))
    if optimized:
        require(
            seen == {(m, n)
                     for m in ("explicit", "optimized")
                     for n in (2, 3)},
            "evaluate explicit and optimized candidates at both leaf counts")
    valid = {name: row for name, row in results.items() if row["valid"]}
    selected = payload.get("selected_candidate")
    if optimized and not valid:
        require(selected is None,
                "no candidate meets both strict spectral bounds")
        return
    require(
        isinstance(selected, str) and selected in valid,
        "selected candidate does not meet both spectral bounds")
    if optimized:
        require(
            valid[selected]["norm"] <= min(r["norm"]
                                           for r in valid.values()) + 1e-7,
            "selected candidate is not a lowest-norm admissible computed candidate"
        )
    else:
        scope = payload.get("minimality_scope")
        require(scope in ("sufficient_only", "evaluated_candidates"),
                "scope the sufficient/minimal leaf-count claim")
        if scope == "evaluated_candidates":
            require(
                valid[selected]["count"] == min(r["count"]
                                                for r in valid.values()),
                "selected leaf count is not minimal among evaluated sufficient candidates"
            )


def _geometry_rows(payload, key, wanted):
    rows = payload.get("geometries")
    require(
        isinstance(rows, list) and len(rows) == 2,
        "exactly two geometry rows are required")
    result = {}
    for row in rows:
        value = _number(row, key)
        matched = [v for v in wanted if abs(value - v) <= 1e-10]
        require(
            len(matched) == 1 and matched[0] not in result,
            f"incorrect or duplicate {key}")
        result[matched[0]] = row
    return result


def check(case_id, payload):
    require(isinstance(payload, dict), "result must be a JSON object")
    if case_id == "science-chemistry-reference-quality":
        ref = _molecular_reference("lih")
        for key, target in (("ground_energy", "ground"), ("reference_energy",
                                                          "hf"),
                            ("reference_overlap_squared", "weight")):
            close(_number(payload, key), ref[target], 1e-6, key)
    elif case_id == "science-chemistry-water-bending":
        rows = _geometry_rows(payload, "angle_degrees", (90., 104.5))
        references = {
            angle: _molecular_reference("water", angle)
            for angle in rows
        }
        for angle, row in rows.items():
            close(_number(row, "ground_energy"), references[angle]["ground"],
                  1e-5, "water ground energy")
            close(_number(row, "rhf_energy"), references[angle]["hf"], 1e-5,
                  "water RHF energy")
        convention = payload.get("difference_convention")
        require(convention in ("104.5-minus-90", "90-minus-104.5"),
                "declare water subtraction convention")
        sign = 1 if convention == "104.5-minus-90" else -1
        difference = references[104.5]["ground"] - references[90.]["ground"]
        rhf_difference = references[104.5]["hf"] - references[90.]["hf"]
        close(_number(payload, "active_energy_difference"), sign * difference,
              1e-5, "water total-energy difference")
        close(_number(payload, "correlation_contribution"),
              sign * (difference - rhf_difference), 1e-5,
              "water correlation contribution")
        close(_number(payload, "preferred_angle_degrees"),
              104.5 if difference < 0 else 90., 1e-10, "water preference")
        require(
            type(payload.get("correlation_changes_preference")) is bool
            and payload["correlation_changes_preference"]
            == bool(difference * rhf_difference < 0),
            "incorrect correlation preference conclusion")
    elif case_id in ("science-double-factorization-leaf-budget",
                     "science-double-factorization-optimized-tradeoff"):
        _check_df(payload, case_id.endswith("optimized-tradeoff"))
    elif case_id == "science-workflow-molecular-bond-stretching":
        _text(payload, "method")
        _text(payload, "work_unit")
        rows = _geometry_rows(payload, "distance_angstrom", (1., 1.8))
        for distance, row in rows.items():
            ref = _molecular_reference("h4", distance)
            close(_number(row, "ground_energy"), ref["ground"], 1.6e-3,
                  "H4 algorithm ground-energy accuracy")
            close(_number(row, "rhf_energy"), ref["hf"], 1e-6,
                  "H4 reference energy")
            diagnostic = _reference_quality(ref, row)
            quality = row.get("reference_quality", {})
            require(isinstance(quality, dict),
                    "reference_quality must be an object")
            kind = quality.get("kind")
            require(
                kind in ("ground_space_weight", "energy_variance",
                         "spectral_weights", "energy_error"),
                "unsupported reference-quality diagnostic")
            if kind == "spectral_weights":
                close(
                    real_array(quality, "value",
                               diagnostic["spectral_weights"].shape),
                    diagnostic["spectral_weights"], 1e-6,
                    "H4 reference spectral weights")
            else:
                key = {
                    "ground_space_weight": "weight",
                    "energy_variance": "variance",
                    "energy_error": "energy_error"
                }[kind]
                close(_number(quality, "value"), diagnostic[key], 1e-6,
                      "H4 reference quality")
            if "convergence_analysis" in row:
                analysis = row["convergence_analysis"]
                _text(analysis, "description")
                require(
                    _number(analysis, "work") >= 0,
                    "analysis work must be nonnegative")
                bound = _number(analysis, "absolute_error_bound")
                require(0 <= bound <= 1.6e-3,
                        "analysis bound must be between zero and 1.6 mHa")
                actual_error = abs(
                    _number(row, "ground_energy") - ref["ground"])
                require(
                    actual_error <= bound + 1e-10,
                    "analysis bound does not cover the independently measured energy error"
                )
            if "convergence" in row or "convergence_analysis" not in row:
                convergence = row.get("convergence")
                require(
                    isinstance(convergence, list) and len(convergence) >= 2,
                    "supply numerical convergence evidence at multiple work amounts or quantitative convergence_analysis"
                )
                work = [_number(entry, "work") for entry in convergence]
                require(
                    min(work) >= 0 and len(set(work)) >= 2,
                    "convergence work must be nonnegative and contain distinct amounts"
                )
                for entry in convergence:
                    _number(entry, "energy")
                index = row.get("selected_convergence_index")
                require(
                    type(index) is int and 0 <= index < len(convergence),
                    "invalid selected_convergence_index")
                close(_number(convergence[index], "energy"),
                      _number(row, "ground_energy"), 1e-8,
                      "selected convergence result")
    else:
        raise CheckFailure(f"unknown chemistry case {case_id}")
