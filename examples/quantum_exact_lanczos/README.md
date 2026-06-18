# Quantum Exact Lanczos Examples

These examples show Quantum Exact Lanczos as a workflow built from CUDA-Q
Algorithms primitives, not as a monolithic library API. The main teaching path is
H2, where the matrices and reference calculation are small enough to inspect.

## Recommended Path

Start with:

```bash
python3 quantum_exact_lanczos_h2.py --target qpp-cpu
```

This script walks through the calculation in stages:

1. load a precomputed Jordan-Wigner qubit Hamiltonian,
2. separate the scalar constant from the non-identity Pauli sum,
3. build a `PauliLCU` block encoding,
4. collect Chebyshev moments using qubitization,
5. build the Krylov Hamiltonian and overlap matrices,
6. filter the overlap matrix and solve the generalized eigenproblem, and
7. compare the final energy against dense exact diagonalization.

The walkthrough prints the intermediate moments, matrices, overlap eigenvalues,
and final energy reconstruction so the numerical path is visible.

## Additional Molecule Fixtures

The `data/` directory contains precomputed qubit Hamiltonians:

- `h2_sto3g_jw.json`
- `lih_sto3g_jw.json`
- `n2_active_space_jw.json`
- `benzene_active_space_jw.json`

These files are example fixtures. They are not tests and they are not a molecule
construction API. The intended workflow is that users generate molecular data
with a chemistry package, convert it to a qubit Hamiltonian, and then compose
CUDA-Q Algorithms primitives from that point forward.

To inspect a larger fixture without launching a QEL circuit:

```bash
python3 quantum_exact_lanczos_molecules.py --molecule lih --describe-only
python3 quantum_exact_lanczos_molecules.py --molecule n2 --describe-only
python3 quantum_exact_lanczos_molecules.py --molecule benzene --describe-only
```

To run the same workflow on one of the fixtures, omit `--describe-only`. Larger
molecules may be expensive and should be treated as opt-in example runs rather
than default unit tests.

## Numerical Notes

The block encoding is built for the non-identity Pauli sum. The scalar constant
is added back only after solving the scaled Krylov problem:

```text
E_physical = E_scaled * alpha + constant
```

The overlap matrix can become nearly singular when the Krylov basis loses
numerical independence. The filtering in these examples is local demonstration
code, included so users can see the classical postprocessing step explicitly.


## Reference

The QEL workflow in these examples follows the block-encoding Lanczos approach
from Kirby, Motta, and Mezzacapo, "Exact and efficient Lanczos method on a quantum computer," Quantum 7, 1018 (2023), arXiv:2208.00567.

https://arxiv.org/abs/2208.00567
