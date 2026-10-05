# Qubitization — family front door

Qubitization exposes two independently selectable contracts:

## Selectable operation contracts

| Operation + object | Public entry point | Kind / layer | Capabilities | Record |
| --- | --- | --- | --- | --- |
| evolve / state by a qubitization walk | `Walk.kernel`, adjoint/controlled variants | quantum operation; kernel factory/device kernel | requires `cudaq-algorithms.block-encoding.zero-flagged.v1`; optional state preparation | [walk kernels](qubitization-walk.md) |
| measure / Chebyshev moment | `Walk.moment`, `Walk.moments` | measurement/readout; kernel + observable + `cudaq.observe` | requires `cudaq-algorithms.block-encoding.zero-flagged.v1`; odd orders also require `select_observable` | [moments](qubitization-moments.md) |

Both consume the
`cudaq-algorithms.block-encoding.zero-flagged.v1` capability. They are not
interchangeable: walk methods emit kernels, while moment methods call
`cudaq.observe` and return classical numbers.

Shared source is `python/cudaq_algorithms/qubitization.py`; shared tests are
`test_qubitization.py` and `test_walk_qsvt_orchestration.py`. Current public
source and tests are authoritative and must be rechecked at use time.
[Source lookup](../source-provenance.md) gives shared current-source paths.

## Workflow

Start with a normalized reference, a physical-sector definition and an encoding
of `H0` with normalization `alpha`; keep any separate scalar in `H = c*I + H0`.
The [moment interface](qubitization-moments.md) returns
`mu[k] = <T_k(H0/alpha)>`, despite the negative sign of the raw walk block.
Odd moments need the encoding's SELECT observable. State the moment orders and
budget, including how the analytically known zeroth moment is counted.
Constructing an encoding is not executing moments: `Walk.moments` evaluates
observables; a dense recurrence is an independent classical prediction.

For **return amplitudes**, expand the desired spectral function in the same
scaled variable. One option is
`G(t) = exp(-i*c*t) * [J_0(alpha*t)*mu[0] +
2*sum_{k>=1} (-i)^k J_k(alpha*t)*mu[k]]`.
Choose the truncation for the largest requested `abs(alpha*t)` and combine
the discarded-coefficient tail with moment uncertainty. For exact moments and
`||H0/alpha|| <= 1`, the absolute tail is bounded by
`2*sum_{k>degree} |J_k(alpha*t)|`. Compare complex amplitudes directly;
discarding their phase changes an interferometric prediction.

For **Krylov energies**, use the basis `T_j(H0/alpha)|reference>` and the
Chebyshev product identities to construct its Gram and projected-Hamiltonian
matrices. Solve on the numerically supported Gram subspace, restore `alpha`
and `c`, and relate convergence to the reference's spectral support. A missing
ground-state overlap cannot be repaired by requesting more redundant basis
vectors. Choose rank cutoffs against numerical/shot error and inspect
sensitivity near the cutoff; a small positive Gram eigenvalue need not be
reliably resolved. Classical post-processing is caller-owned, not a packaged
Krylov optimizer or a guarantee of ground-state recovery.

## Verification

These small-system checkpoints take a dense Hermitian `H0`, normalized complex
`psi`, and a matching `encoding` with `alpha > 0`. `count` is positive;
`mu` is the ordered array of moments beginning at zero. The first function
executes library moments and compares them with an independent recurrence.
The remaining functions are classical post-processing; `dimension` is positive
and `cutoff` is a caller-chosen positive Gram eigenvalue threshold.

```python
import numpy as np
from scipy.linalg import expm
from scipy.special import jv
from cudaq_algorithms import Walk

def moment_checkpoint(encoding, H0, psi, count):
    measured = np.asarray(Walk(encoding).moments(psi, count))
    scaled = H0 / encoding.alpha
    previous, current = np.asarray(psi), scaled @ psi
    reference = [np.vdot(psi, previous).real]
    if count > 1:
        reference.append(np.vdot(psi, current).real)
    for _ in range(2, count):
        previous, current = current, 2 * scaled @ current - previous
        reference.append(np.vdot(psi, current).real)
    return measured, float(np.max(np.abs(measured - reference)))

def return_checkpoint(mu, H0, psi, alpha, time, offset=0.0):
    orders = np.arange(len(mu))
    coefficients = (2.0 * (-1j)**orders) * jv(orders, alpha * time)
    coefficients[0] *= 0.5
    phase = np.exp(-1j * offset * time)
    estimate = phase * np.dot(coefficients, mu)
    exact = phase * np.vdot(psi, expm(-1j * time * H0) @ psi)
    return estimate, float(abs(estimate - exact))

def krylov_checkpoint(mu, alpha, dimension, cutoff, offset=0.0):
    mu = np.asarray(mu)
    if len(mu) < 2 * dimension:
        raise ValueError("need moments through order 2*dimension-1")
    i, j = np.indices((dimension, dimension))
    total, difference = i + j, np.abs(i - j)
    gram = 0.5 * (mu[total] + mu[difference])
    projected = 0.25 * (mu[total + 1] + mu[np.abs(total - 1)]
                        + mu[difference + 1] + mu[np.abs(difference - 1)])
    values, vectors = np.linalg.eigh(gram)
    keep = values > cutoff
    if not np.any(keep):
        raise ValueError("no numerically resolved Krylov direction")
    whitening = vectors[:, keep] / np.sqrt(values[keep])
    ritz = np.linalg.eigvalsh(whitening.conj().T @ projected @ whitening)
    return float(offset + alpha * ritz[0]), values, int(np.count_nonzero(keep))
```

Compare moment errors, complex return-amplitude errors and restored Ritz
energies with independent dense results; do not use a library path as its own
oracle. For Krylov checks, also compare the lowest eigenvalue with nonzero
reference support and the full physical-sector ground energy. Apparent
convergence and conservation laws alone do not certify either target.
Sources and runnable pointers: `python/cudaq_algorithms/qubitization.py`,
`tests/python/test_qubitization.py`, and
`docs/sphinx/examples/python/03_chemistry_to_ground_state.py` (its classical
Krylov helpers are example code). Numerical precision and sampling uncertainty
must be reported separately from approximation and rank-truncation errors.
