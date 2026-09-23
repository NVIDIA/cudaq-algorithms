# Numerical Comparison

## Precision, Hermiticity, and tolerances

- **Convention.** Establish dtype, target precision, and tolerance before a
  numerical comparison. Coefficient-pruning tolerances change the represented
  operator; comparison tolerances only judge outputs. A routine accepting
  non-Hermitian data must not be described as enforcing Hermiticity.
- **Rule.** Report global-phase-insensitive comparisons explicitly, and use
  invariant checks such as reconstructed operators, spectra, or expectation
  values when raw representations differ.
- **Source.** `docs/sphinx/conventions.rst`; fermion-transform, chemistry,
  QSVT, and validation family evidence linked from the catalog.

## Comparison discipline

Before comparing two implementations or sources:

1. identify both convention sets;
2. write the explicit translation;
3. select an invariant comparison when possible;
4. fix precision and tolerance;
5. label unresolved differences as unknown.

Prefer spectra, expectation values, reconstructed operators, known identities,
or other invariants over raw arrays when equivalent representations exist.
