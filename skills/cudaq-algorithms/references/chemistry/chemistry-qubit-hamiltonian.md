# Chemistry qubit Hamiltonian bridge

Status: draft. Operation + object: **transform** **chemist-notation spatial
integrals into a Pauli Hamiltonian**.

## Identity and classification

- Public symbol: `cudaq_algorithms.chemistry.qubit_hamiltonian`.
- Source: `python/cudaq_algorithms/chemistry.py`.
- Tests: `test_df_qsvt_bridge.py` and fermion compiler tests.
- Kind/role/abstraction: classical transformation, driver, composite protocol.
- Layers: spin expansion followed by host fermion compilation.
- Output: `cudaq.SpinOperator`.

## Scientific contract

```python
qubit_hamiltonian(
    one_body,
    eri,
    scalar_offset=0.0,
    tolerance=1.0e-12,
    validate_symmetry=True,
)
```

The function calls [spin-orbital expansion](chemistry-spin-orbital-tensors.md)
and then the packaged `fermion.jordan_wigner` transform. The transform choice is
hard-wired: there is no Bravyi–Kitaev selector. `scalar_offset` becomes the
identity coefficient; `tolerance` prunes terms in the fermion compiler.

Use this bridge when the downstream algorithms expect the package's
Jordan–Wigner qubit and spin-orbital conventions. To choose Bravyi–Kitaev,
invoke spin expansion and [that transform](../fermion-transforms/bravyi-kitaev.md) explicitly; do not
pretend it is a `qubit_hamiltonian` option.

## Composition and limits

Consumes `cudaq-algorithms.chemistry-integrals.v1`. The output may feed
`PauliLCU`, `Trotter`, or another Pauli consumer only after its real-Hermitian
coefficient and register-extent requirements are checked. The bridge validates
ERI symmetry by default but does not check one-body Hermiticity; the underlying
fermion compiler also lets tolerance/cancellation produce an operator narrower
than the `2n` spin-orbital input, or an empty operator. Preserve the intended
width separately and handle scalar-only/empty results according to the chosen
consumer's contract. A source-grounded compressed chain reconstructs an
approximate ERI first, then calls this bridge; the factorization error and
pruning error remain visible.

The helper inherits real-orbital symmetry validation and dense spin-tensor
costs. It imports the fermion subpackage lazily. It does not return a quantum
kernel, state preparation, block encoding, optimizer result, or resource
estimate.

## Validation

Compare the resulting dense Pauli matrix against an independently assembled
second-quantized Hamiltonian, including the scalar identity term. Check the
same input through explicit spin expansion plus `jordan_wigner`, and include an
asymmetric-ERI failure. The runnable pointer is
`tests/python/test_df_qsvt_bridge.py`. Current public source and tests are
authoritative and must be rechecked at use time; this record was not freshly
executed. Eval coverage is
`chemistry-bridge-dependency-boundary` plus end-to-end application cases.
[source-provenance.md](../source-provenance.md) records historical last-review
audit context.
