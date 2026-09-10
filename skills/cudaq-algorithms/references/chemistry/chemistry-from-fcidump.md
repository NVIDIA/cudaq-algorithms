# FCIDUMP integral loader

Status: draft. Operation + object: **load** a **chemist-notation spatial
integral triple from FCIDUMP text**.

## Identity and classification

- Public symbol: `cudaq_algorithms.chemistry.from_fcidump`.
- Source: `python/cudaq_algorithms/chemistry.py`.
- Tests: `tests/python/test_fcidump.py`.
- Runnable usage: the FCIDUMP section of
  `docs/sphinx/guide/preprocessing.rst` and the function docstring.
- Kind/role/layer: classical transformation, computational leaf, host parser.
- Dependency: NumPy; PySCF and Psi4 are not required.
- Lifecycle/evidence: draft; current public source and tests are authoritative
  and must be rechecked at use time; not freshly executed.
  [source-provenance.md](../source-provenance.md) records historical last-review
  audit context.

## Input, rejection, and output

`from_fcidump(contents)` consumes the file **contents as a string**, not a
path; the caller owns file I/O. The first nonempty line must start an `&FCI` or
`$FCI` namelist, the header must terminate with `&END`, `$END`, or `/`, and
`NORB` must be a positive integer. A nonempty body is required.

Each integral record has exactly `value i j k l`; Fortran `D` exponents are
accepted and indices are 1-based, with zero used for one-body, orbital-energy,
or scalar records. The parser rejects malformed fields, out-of-range or
unsupported index patterns, unterminated/missing headers, and unrestricted
headers marked by nonzero `IUHF` or true `UHF`. Optional `value i 0 0 0`
orbital-energy records are ignored.

The result is `(one_body, eri, core_energy)`: contiguous real NumPy arrays of
shapes `(n,n)` and `(n,n,n,n)`, plus a Python `float`. One-body transpose
partners and all members of each real-orbital eightfold ERI symmetry orbit are
filled. The output provides
`cudaq-algorithms.chemistry-integrals.v1` and may feed
`chemistry.qubit_hamiltonian`, `chemistry.spin_orbital_tensors`, or double
factorization. Parsing performs none of those downstream operations.

## Resources and validation

There is no resource estimator or quantum circuit. Exact structural storage is
one dense `n^2` array plus one dense `n^4` array; runtime and peak memory are
unmeasured.

Validate against a known FCIDUMP fixture elementwise (`atol=1e-12` in the
cited tests), verify the generated eightfold orbit, and compare the downstream
qubit-Hamiltonian spectrum with the same integrals supplied directly
(`atol=1e-10`). Exercise every rejection above and the optional orbital-energy
path. `tests/python/test_fcidump.py` is the runnable usage test.

Eval coverage: authored `chemistry-end-to-end-composition`. Baseline and
with-skill arms have not been run.
