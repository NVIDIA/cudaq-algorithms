# Scientific conventions

Align conventions before comparing scientific results. Current public source,
tests, and `docs/sphinx/conventions.rst` are authoritative; cited assertions are
derived evidence until checked or executed in the current task.

| Boundary to check | Focused reference |
| --- | --- |
| Qubit, Pauli-word, spin-orbital order; register extent | [Ordering](conventions/ordering.md) |
| Register-only injection, ownership, host validation | [Kernel boundaries](conventions/kernel-boundaries.md) |
| Block normalization, walk signs, QSP/QSVT phases, reflections | [Spectral processing](conventions/spectral-processing.md) |
| Logical proxies versus gates, runtime, memory | [Resource levels](conventions/resource-levels.md) |
| Precision, Hermiticity, tolerance and invariant comparisons | [Numerical comparison](conventions/numerical-comparison.md) |
