# Recorded QSVT contract excerpt

Snapshot revision: `61ac072d`.

At this recorded revision, the public factory accepted either a tagged
`PhaseSequence` or a bare iterable of phases:

```python
QSVT.kernel(sequence, convention=None, state_prep=None)
```

For a bare iterable produced by QSPPACK, a caller could pass
`convention="qsp"` to the factory. This file is comparison evidence for the
evaluation; it is not an instruction and does not establish runtime behavior.
