"""Passing executed numerical evidence and physically significant corruptions."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyscf")
from pyscf import fci, gto, lib, mcscf, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runner.checkers import chemistry
from runner.checkers.common import CheckFailure

chemistry._configure_pyscf()


def molecule(atom, core, active, electrons):
    mol = gto.M(atom=atom,
                basis="sto-3g",
                unit="Angstrom",
                spin=0,
                charge=0,
                verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-13)
    assert mf.converged
    cas = mcscf.CASCI(mf, active, electrons)
    cas.ncore = core
    cas.fcisolver.conv_tol = 1e-13
    energy, _, ci, *_ = cas.kernel()
    return float(energy), float(mf.e_tot), float(abs(ci[0, 0])**2)


@pytest.fixture(scope="module")
def lih_payload():
    e, hf, overlap = molecule("Li 0 0 0; H 0 0 1.6", 1, 4, 2)
    return dict(ground_energy=e,
                reference_energy=hf,
                reference_overlap_squared=overlap)


@pytest.fixture(scope="module")
def water_payload():
    rows = []
    for angle in (90., 104.5):
        x, z = .958 * np.sin(np.deg2rad(angle / 2)), .958 * np.cos(
            np.deg2rad(angle / 2))
        e, hf, _ = molecule([("O", (0, 0, 0)), ("H", (x, 0, z)),
                             ("H", (-x, 0, z))], 3, 4, 4)
        rows.append(dict(angle_degrees=angle, ground_energy=e, rhf_energy=hf))
    delta = rows[1]["ground_energy"] - rows[0]["ground_energy"]
    delta_hf = rows[1]["rhf_energy"] - rows[0]["rhf_energy"]
    return dict(geometries=rows,
                difference_convention="104.5-minus-90",
                active_energy_difference=delta,
                correlation_contribution=delta - delta_hf,
                preferred_angle_degrees=104.5 if delta < 0 else 90.,
                correlation_changes_preference=bool(delta * delta_hf < 0))


@pytest.fixture(scope="module")
def h4_payload():
    rows = []
    for distance in (1., 1.8):
        e, hf, overlap = molecule([("H", (0, 0, j * distance))
                                   for j in range(4)], 0, 4, 4)
        rows.append(
            dict(distance_angstrom=distance,
                 ground_energy=e,
                 rhf_energy=hf,
                 reference_quality={
                     "kind": "ground_space_weight",
                     "value": overlap
                 },
                 convergence=[{
                     "work": 1,
                     "energy": hf
                 }, {
                     "work": 36,
                     "energy": e
                 }],
                 selected_convergence_index=1))
    return dict(geometries=rows,
                method="full Hermitian Krylov space",
                work_unit="matrix-vector products")


@pytest.mark.parametrize("case,fixture", [
    ("science-chemistry-reference-quality", "lih_payload"),
    ("science-chemistry-water-bending", "water_payload"),
    ("science-workflow-molecular-bond-stretching", "h4_payload"),
])
def test_molecular_results_pass(case, fixture, request):
    chemistry.check(case, request.getfixturevalue(fixture))


def test_lih_rejects_missing_frozen_core_energy_and_unsquared_overlap(
        lih_payload):
    for field, change in (("ground_energy", .3), ("reference_energy", .01),
                          ("reference_overlap_squared", .01)):
        p = copy.deepcopy(lih_payload)
        p[field] += change
        with pytest.raises(CheckFailure):
            chemistry.check("science-chemistry-reference-quality", p)


def test_water_sign_conventions_and_inconsistent_correlation(water_payload):
    p = copy.deepcopy(water_payload)
    p["difference_convention"] = "90-minus-104.5"
    p["active_energy_difference"] *= -1
    p["correlation_contribution"] *= -1
    chemistry.check("science-chemistry-water-bending", p)
    p["correlation_contribution"] += .002
    with pytest.raises(CheckFailure):
        chemistry.check("science-chemistry-water-bending", p)


def test_h4_rejects_wrong_reference_quality_and_inaccurate_algorithm(
        h4_payload):
    for corruption in ("overlap", "energy", "missing_convergence"):
        p = copy.deepcopy(h4_payload)
        if corruption == "overlap":
            p["geometries"][1]["reference_quality"]["value"] = 1.
        elif corruption == "energy":
            p["geometries"][1]["convergence"][-1]["energy"] += .01
            p["geometries"][1]["ground_energy"] += .01
        else:
            p["geometries"][1]["convergence"] = []
        with pytest.raises(CheckFailure):
            chemistry.check("science-workflow-molecular-bond-stretching", p)


@pytest.fixture
def h4_analysis_payload(h4_payload):
    p = copy.deepcopy(h4_payload)
    for row in p["geometries"]:
        del row["convergence"]
        del row["selected_convergence_index"]
        row["ground_energy"] += 5e-4
        row["convergence_analysis"] = {
            "work":
            36,
            "absolute_error_bound":
            6e-4,
            "description":
            "Complete sector Krylov basis with a 0.6 mHa numerical error allowance; reference spectral weights determine the occupied Krylov support."
        }
    return p


def test_h4_accepts_quantitative_convergence_analysis(h4_analysis_payload):
    chemistry.check("science-workflow-molecular-bond-stretching",
                    h4_analysis_payload)


def test_h4_allows_different_evidence_types_at_each_geometry(
        h4_analysis_payload, h4_payload):
    h4_analysis_payload["geometries"][0] = copy.deepcopy(
        h4_payload["geometries"][0])
    chemistry.check("science-workflow-molecular-bond-stretching",
                    h4_analysis_payload)


@pytest.mark.parametrize("bound", [4e-4, 1.7e-3, -1., float("nan")])
def test_h4_rejects_invalid_quantitative_error_bounds(h4_analysis_payload,
                                                      bound):
    h4_analysis_payload["geometries"][1]["convergence_analysis"][
        "absolute_error_bound"] = bound
    with pytest.raises(CheckFailure):
        chemistry.check("science-workflow-molecular-bond-stretching",
                        h4_analysis_payload)


@pytest.mark.parametrize("missing",
                         ["work", "absolute_error_bound", "description"])
def test_h4_analysis_requires_work_bound_and_explanation(
        h4_analysis_payload, missing):
    del h4_analysis_payload["geometries"][1]["convergence_analysis"][missing]
    with pytest.raises(CheckFailure):
        chemistry.check("science-workflow-molecular-bond-stretching",
                        h4_analysis_payload)


def test_h4_accepts_actual_unrestricted_reference_determinant(h4_payload):
    p = copy.deepcopy(h4_payload)
    for row in p["geometries"]:
        distance = row["distance_angstrom"]
        mol = gto.M(atom=[("H", (0, 0, j * distance)) for j in range(4)],
                    basis="sto-3g",
                    unit="Angstrom",
                    verbose=0)
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-13
        mf.max_cycle = 200
        mf.kernel(dm0=np.array(
            [np.diag([1., 0., 1., 0.]),
             np.diag([0., 1., 0., 1.])]))
        assert mf.converged
        row["occupied_alpha_ao"] = mf.mo_coeff[0][:, :2].tolist()
        row["occupied_beta_ao"] = mf.mo_coeff[1][:, :2].tolist()
        row["reference_energy"] = float(mf.e_tot)
        row["reference_quality"] = {
            "kind": "energy_error",
            "value": float(mf.e_tot - row["ground_energy"])
        }
    chemistry.check("science-workflow-molecular-bond-stretching", p)
    p["geometries"][1]["occupied_alpha_ao"][0][0] += .1
    with pytest.raises(CheckFailure):
        chemistry.check("science-workflow-molecular-bond-stretching", p)


def synthetic(seed):
    rng = np.random.default_rng(seed)
    weights = [.04] * 3 if seed == 7 else [.30, .12, .04, .01]
    tensors = []
    for weight in weights:
        a = rng.standard_normal((4, 4))
        tensors.append(np.sqrt(weight) * (a + a.T) / 2)
    eri = np.einsum("tpq,trs->pqrs", tensors, tensors)
    h = np.diag([-.9, -.5, .1, .4] if seed == 7 else [-.8, -.4, .15, .5])
    for i, j, value in ([(0, 1, .12),
                         (2, 3, .08)] if seed == 7 else [(0, 1, .08),
                                                         (1, 2, .08),
                                                         (2, 3, .08)]):
        h[i, j] = h[j, i] = value
    return h, eri, .3 if seed == 7 else .2, (1, 1) if seed == 7 else (2, 2)


def spectrum(h, eri, scalar, nelec):
    energies, _ = fci.direct_spin1.kernel(h,
                                          eri,
                                          4,
                                          nelec,
                                          ecore=scalar,
                                          nroots=2,
                                          conv_tol=1e-13)
    return float(energies[0]), float(energies[1] - energies[0])


def factors(eri, count, family="eigendecomposition"):
    supermatrix = eri.reshape(16, 16)
    if family == "eigendecomposition":
        values, vectors = np.linalg.eigh(supermatrix)
        order = np.argsort(values)[::-1][:count]
        leaves = [np.sqrt(values[i]) * vectors[:, i] for i in order]
    else:
        remainder = supermatrix.copy()
        leaves = []
        for _ in range(count):
            pivot = np.argmax(np.diag(remainder))
            leaf = remainder[:, pivot] / np.sqrt(remainder[pivot, pivot])
            leaves.append(leaf)
            remainder -= np.outer(leaf, leaf)
    rotations, cores = [], []
    for leaf in leaves:
        values, rotation = np.linalg.eigh(leaf.reshape(4, 4))
        rotations.append(rotation)
        cores.append(np.outer(values, values))
    return np.array(rotations), np.array(cores)


def candidate(h,
              eri,
              scalar,
              nelec,
              count,
              method="explicit",
              family="eigendecomposition",
              factor_data=None):
    u, z = factors(eri, count, family) if factor_data is None else factor_data
    reconstructed = np.einsum("tpk,tqk,tkl,trl,tsl->pqrs", u, u, z, u, u)
    energy, gap = spectrum(h, reconstructed, scalar, nelec)
    f = h.copy()
    for p in range(4):
        for q in range(4):
            f[p, q] += sum(reconstructed[p, q, r, r] -
                           .5 * reconstructed[p, r, q, r] for r in range(4))
    norm = np.abs(np.linalg.eigvalsh(f)).sum()
    for core in z:
        norm += sum(abs(core[k, l]) for k in range(4) for l in range(k + 1, 4))
        norm += .25 * np.abs(np.diag(core)).sum()
    identity = scalar + np.trace(h) + sum(
        .5 * reconstructed[p, p, q, q] - .25 * reconstructed[p, q, p, q]
        for p in range(4) for q in range(4))
    return dict(id=f"{method}-{count}",
                method=method,
                num_leaves=count,
                first_factorization=family,
                threshold=1e-10,
                second_factor_threshold=0.,
                rotations=u.tolist(),
                cores=z.tolist(),
                reconstructed_eri=reconstructed.tolist(),
                ground_energy=energy,
                excitation_gap=gap,
                one_norm=float(norm),
                one_body_eigenvalues=np.linalg.eigvalsh(f).tolist(),
                identity_coefficient=float(identity))


@pytest.fixture(scope="module")
def leaf_payload():
    h, eri, scalar, nelec = synthetic(7)
    e, gap = spectrum(h, eri, scalar, nelec)
    return dict(reference_ground_energy=e,
                reference_excitation_gap=gap,
                gap_convention="first_level",
                candidates=[
                    candidate(h, eri, scalar, nelec, count)
                    for count in (1, 2, 3)
                ],
                selected_candidate="explicit-3",
                minimality_scope="evaluated_candidates")


@pytest.fixture(scope="module")
def tradeoff_payload():
    h, eri, scalar, nelec = synthetic(31)
    e, gap = spectrum(h, eri, scalar, nelec)
    rows = [
        candidate(h, eri, scalar, nelec, count, method)
        for method in ("explicit", "optimized") for count in (2, 3)
    ]
    # An independent SciPy least-squares search supplies an actual admissible
    # optimized result; the oracle must not demand one preselected tensor.
    u, z = independent_compression(eri)
    rows[-1] = candidate(h,
                         eri,
                         scalar,
                         nelec,
                         3,
                         "optimized",
                         factor_data=(u, z))
    eligible = [
        r for r in rows
        if abs(r["ground_energy"] - e) < 1e-3 and abs(r["excitation_gap"] -
                                                      gap) < 1e-3
    ]
    return dict(
        reference_ground_energy=e,
        reference_excitation_gap=gap,
        gap_convention="first_level",
        candidates=rows,
        norm_convention="lcu",
        scalar_handling="excluded",
        selected_candidate=min(eligible, key=lambda r: r["one_norm"])["id"]
        if eligible else None)


def independent_compression(eri):
    from scipy.linalg import expm
    from scipy.optimize import least_squares
    initial_u, initial_z = factors(eri, 3)
    upper, triangle = np.triu_indices(4, 1), np.triu_indices(4)

    def unpack(values):
        rotations, cores = [], []
        for index, row in enumerate(values.reshape(3, 16)):
            skew = np.zeros((4, 4))
            skew[upper] = row[:6]
            rotations.append(initial_u[index] @ expm(skew - skew.T))
            core = np.zeros((4, 4))
            core[triangle] = row[6:]
            cores.append(core + np.triu(core, 1).T)
        return np.array(rotations), np.array(cores)

    def residual(values):
        u, z = unpack(values)
        return (np.einsum(
            "tpk,tqk,tkl,trl,tsl->pqrs", u, u, z, u, u, optimize=True) -
                eri).ravel()

    start = np.column_stack((np.zeros(
        (3, 6)), initial_z[:, triangle[0], triangle[1]])).ravel()
    result = least_squares(residual,
                           start,
                           max_nfev=200,
                           gtol=1e-11,
                           xtol=1e-11,
                           ftol=1e-11)
    assert result.success
    return unpack(result.x)


def test_leaf_budget_passes_both_explicit_families(leaf_payload):
    chemistry.check("science-double-factorization-leaf-budget", leaf_payload)
    p = copy.deepcopy(leaf_payload)
    h, eri, scalar, nelec = synthetic(7)
    p["candidates"] = [
        candidate(h, eri, scalar, nelec, count, family="cholesky")
        for count in (1, 2, 3)
    ]
    chemistry.check("science-double-factorization-leaf-budget", p)


def test_leaf_budget_accepts_permuted_retained_leaf_indices(leaf_payload):
    p = copy.deepcopy(leaf_payload)
    for row in p["candidates"]:
        row["retained_leaf_indices"] = list(reversed(range(row["num_leaves"])))
        row["rotations"].reverse()
        row["cores"].reverse()
    p["gap_convention"] = "first_distinct"
    chemistry.check("science-double-factorization-leaf-budget", p)


def test_gap_conventions_count_ground_multiplicity_explicitly():
    energies = np.array([-2., -2., -1., 0.])
    assert chemistry._gap(energies, "first_level", 1e-8) == 0.
    assert chemistry._gap(energies, "first_distinct", 1e-8) == 1.


def test_leaf_budget_rejects_claimed_spectra_and_truncated_selection(
        leaf_payload):
    for kind in ("ground", "gap", "selection", "eri", "rotation"):
        p = copy.deepcopy(leaf_payload)
        if kind == "ground":
            p["candidates"][0]["ground_energy"] = p["reference_ground_energy"]
        elif kind == "gap":
            p["candidates"][0]["excitation_gap"] += .01
        elif kind == "selection":
            p["selected_candidate"] = "explicit-1"
        elif kind == "eri":
            p["candidates"][2]["reconstructed_eri"][0][0][0][0] += .1
        else:
            p["candidates"][2]["rotations"][0][0][0] += .1
        with pytest.raises(CheckFailure):
            chemistry.check("science-double-factorization-leaf-budget", p)


def test_tradeoff_passes_and_checks_norm_evidence(tradeoff_payload):
    chemistry.check("science-double-factorization-optimized-tradeoff",
                    tradeoff_payload)
    for key in ("one_norm", "identity_coefficient"):
        p = copy.deepcopy(tradeoff_payload)
        p["candidates"][0][key] += .02
        with pytest.raises(CheckFailure):
            chemistry.check("science-double-factorization-optimized-tradeoff",
                            p)
    p = copy.deepcopy(tradeoff_payload)
    p["candidates"][0]["one_body_eigenvalues"][0] += .02
    with pytest.raises(CheckFailure):
        chemistry.check("science-double-factorization-optimized-tradeoff", p)


def test_tradeoff_no_qualifying_candidate_is_valid(tradeoff_payload):
    p = copy.deepcopy(tradeoff_payload)
    h, eri, scalar, nelec = synthetic(31)
    p["candidates"][-1] = candidate(h, eri, scalar, nelec, 3, "optimized")
    p["selected_candidate"] = None
    chemistry.check("science-double-factorization-optimized-tradeoff", p)
    p["selected_candidate"] = "optimized-3"
    with pytest.raises(CheckFailure):
        chemistry.check("science-double-factorization-optimized-tradeoff", p)


def test_tradeoff_selects_lowest_norm_among_admissible_computed_candidates(
        tradeoff_payload):
    p = copy.deepcopy(tradeoff_payload)
    row = p["candidates"][-1]
    u, z = np.array(row["rotations"]), np.array(row["cores"])
    z[0, 0, 0] += 1e-5
    h, eri, scalar, nelec = synthetic(31)
    alternate = candidate(h,
                          eri,
                          scalar,
                          nelec,
                          3,
                          "optimized",
                          factor_data=(u, z))
    alternate["id"] = "optimized-3-alternate"
    p["candidates"].append(alternate)
    choices = [row, alternate]
    assert abs(alternate["ground_energy"] -
               p["reference_ground_energy"]) < 1e-3
    assert abs(alternate["excitation_gap"] -
               p["reference_excitation_gap"]) < 1e-3
    assert abs(row["one_norm"] - alternate["one_norm"]) > 1e-7
    p["selected_candidate"] = min(choices, key=lambda r: r["one_norm"])["id"]
    chemistry.check("science-double-factorization-optimized-tradeoff", p)
    p["selected_candidate"] = max(choices, key=lambda r: r["one_norm"])["id"]
    with pytest.raises(CheckFailure, match="lowest-norm"):
        chemistry.check("science-double-factorization-optimized-tradeoff", p)


def test_tradeoff_burg_and_included_scalar(tradeoff_payload):
    p = copy.deepcopy(tradeoff_payload)
    p["norm_convention"] = "burg"
    p["scalar_handling"] = "included_absolute"
    for row in p["candidates"]:
        norm = sum(abs(x) for x in row["one_body_eigenvalues"])
        for core in row["cores"]:
            values, vectors = np.linalg.eigh(core)
            norm += .25 * sum(
                abs(values[i]) * sum(abs(x) for x in vectors[:, i])**2
                for i in range(4))
        row["one_norm"] = norm + abs(row["identity_coefficient"])
    chemistry.check("science-double-factorization-optimized-tradeoff", p)


def test_reference_spectrum_against_independent_fermionic_operators():
    from openfermion import FermionOperator, get_sparse_operator
    from itertools import product
    h, eri, scalar, nelec = synthetic(7)
    op = FermionOperator((), scalar)
    for p, q in product(range(4), repeat=2):
        for spin in (0, 1):
            op += FermionOperator(((2 * p + spin, 1), (2 * q + spin, 0)), h[p,
                                                                            q])
    for p, q, r, s in product(range(4), repeat=4):
        for sigma, tau in product((0, 1), repeat=2):
            op += FermionOperator(((2 * p + sigma, 1), (2 * r + tau, 1),
                                   (2 * s + tau, 0), (2 * q + sigma, 0)),
                                  .5 * eri[p, q, r, s])
    matrix = get_sparse_operator(op, n_qubits=8).toarray()
    # OpenFermion's dense convention uses q0 as the most significant bit.
    indices = [
        i for i in range(256)
        if sum((i >> q) & 1 for q in (0, 2, 4, 6)) == 1 and sum(
            (i >> q) & 1 for q in (1, 3, 5, 7)) == 1
    ]
    expected = np.linalg.eigvalsh(matrix[np.ix_(indices, indices)])
    actual, _ = chemistry._spectrum(h, eri, scalar, nelec)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)
    reversal = [int(f"{i:08b}"[::-1], 2) for i in range(256)]
    np.testing.assert_allclose(chemistry._full_hamiltonian(h, eri, scalar),
                               matrix[np.ix_(reversal, reversal)],
                               rtol=0,
                               atol=1e-11)
    identity = scalar + np.trace(h) + sum(
        .5 * eri[p, p, q, q] - .25 * eri[p, q, p, q] for p in range(4)
        for q in range(4))
    np.testing.assert_allclose(np.trace(matrix) / 256,
                               identity,
                               rtol=0,
                               atol=1e-11)


def test_custom_lcu_requires_actual_unitary_hamiltonian_evidence(
        tradeoff_payload):
    h, _, scalar, _ = synthetic(31)
    row = copy.deepcopy(tradeoff_payload["candidates"][-1])
    eri, cores = np.array(row["reconstructed_eri"]), np.array(row["cores"])
    identity = row["identity_coefficient"]
    matrix = chemistry._full_hamiltonian(h, eri,
                                         scalar) - identity * np.eye(256)
    values, vectors = np.linalg.eigh(matrix)
    norm = float(np.max(np.abs(values)))
    # H/||H|| is the mean of two explicitly unitary conjugate matrices.
    scaled = values / norm
    unitary = (
        vectors *
        (scaled + 1j * np.sqrt(np.maximum(0., 1 - scaled**2)))) @ vectors.T
    terms = np.stack([unitary, unitary.conj().T])
    row["unitary_lcu"] = {
        "coefficients": [norm / 2, norm / 2],
        "identity_shift": identity,
        "unitaries": np.stack([terms.real, terms.imag], axis=-1).tolist()
    }
    actual = chemistry._norm(row, h, eri, scalar, cores, "custom_lcu",
                             "excluded")
    assert abs(actual - norm) < 1e-10
    row["unitary_lcu"]["coefficients"][0] += .01
    with pytest.raises(CheckFailure, match="Hamiltonian"):
        chemistry._norm(row, h, eri, scalar, cores, "custom_lcu", "excluded")
    row["unitary_lcu"]["coefficients"][0] -= .01
    row["unitary_lcu"]["unitaries"][0][0][0][0] += .1
    with pytest.raises(CheckFailure, match="unitary"):
        chemistry._norm(row, h, eri, scalar, cores, "custom_lcu", "excluded")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "-7.8", True])
def test_lih_rejects_nonfinite_and_nonnumeric_evidence(lih_payload, bad):
    p = dict(lih_payload, ground_energy=bad)
    with pytest.raises(CheckFailure):
        chemistry.check("science-chemistry-reference-quality", p)
