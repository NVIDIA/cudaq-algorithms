# State-preparation device kernels — family front door

Status: draft. These are independently callable, runtime-parameterized CUDA-Q
device kernels. They mutate a caller-owned `cudaq.qview`, return no host value,
and are not the one-argument injectable-provider seam unless their full
signature already has that shape or a caller supplies a wrapper.

This file is a routing front door and contains no primitive record.

| Operation + object | Public entry point | Focused record |
| --- | --- | --- |
| prepare / contiguous Hartree–Fock occupation | `stateprep.hartree_fock` | [canonical HF kernel](state-preparation-kernel-hartree-fock.md) |
| prepare / explicit occupation | `stateprep.hartree_fock_occupation` | [occupation HF kernel](state-preparation-kernel-hartree-fock-occupation.md) |
| apply / one UCCSD single excitation | `stateprep.single_excitation` | [single-excitation kernel](state-preparation-kernel-single-excitation.md) |
| apply / one UCCSD double excitation | `stateprep.double_excitation` | [double-excitation kernel](state-preparation-kernel-double-excitation.md) |
| apply / occupied-to-virtual UCCSD product | `stateprep.uccsd` | [UCCSD kernel](state-preparation-kernel-uccsd.md) |
| apply / generalized UCCGSD product | `stateprep.uccgsd` | [UCCGSD kernel](state-preparation-kernel-uccgsd.md) |
| apply / paired UpCCGSD product | `stateprep.upccgsd` | [UpCCGSD kernel](state-preparation-kernel-upccgsd.md) |
| apply / coupled-exchange product | `stateprep.ceo` | [CEO kernel](state-preparation-kernel-ceo.md) |
| apply / arbitrary grouped fixed-parameter UCC product | `stateprep.fixed_parameter_ucc` | [fixed-UCC kernel](state-preparation-kernel-fixed-parameter-ucc.md) |
| apply / adjacent real Givens rotation | `stateprep.givens_rotation` | [Givens rotation kernel](state-preparation-kernel-givens-rotation.md) |
| apply / adjacent phase-aware Givens rotation | `stateprep.phase_givens_rotation` | [phase Givens kernel](state-preparation-kernel-phase-givens-rotation.md) |
| prepare / real Slater determinant from flattened arrays | `stateprep.slater_determinant` | [real Slater kernel](state-preparation-kernel-slater-determinant.md) |
| prepare / complex Slater determinant from flattened arrays | `stateprep.complex_slater_determinant` | [complex Slater kernel](state-preparation-kernel-complex-slater-determinant.md) |

Eleven table entries are public exports in `stateprep.__all__`.
`single_excitation` and `double_excitation` are explicitly imported into the
public module and are addressable there, although they are omitted from
`__all__`; their focused records preserve that surface qualification.

Shared source is `python/cudaq_algorithms/stateprep/_kernels.py` and
`python/cudaq_algorithms/stateprep/_givens.py`; exports are in
`python/cudaq_algorithms/stateprep/__init__.py`. Device bodies provide no general
host validation or status channel. Use the host planners/validators named by a
focused record before launch. Claims are `derived` from source and committed
tests inspected during the historical last review recorded in
[Source provenance](../source-provenance.md). Current public source/tests are
authoritative and must be checked at use time; execution in the declared CUDA-Q
range and SkillEvaluator coverage remain unverified.
