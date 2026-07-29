# cudaq-algorithms — getting started

A standalone onboarding bundle for the quantum algorithm engineering team.

- **`GETTING_STARTED.md`** — read this first: what the library is, install,
  the mental model, and the guided example order.
- **`0N_*.py`** — six runnable, self-verifying examples, meant to be read
  in order (1 → 6).

## Quick run

```bash
pip install cudaq_algorithms_cu12-*.whl   # pulls CUDA-Q, NumPy/SciPy, qsppack
pip install pyscf                          # only for example 3
python3 01_quickstart_block_encoding.py
```

Every example prints a narrated walkthrough and checks its numbers
against an independent reference, so a clean run is a self-test of your
install.

## The six examples

1. `01_quickstart_block_encoding.py` — block-encode a Hamiltonian, walk it, verify moments.
2. `02_hamiltonian_simulation.py` — `exp(-iHt)` via QSVT and via Trotter, cross-checked.
3. `03_chemistry_to_ground_state.py` — molecule → qubit Hamiltonian → quantum exact Lanczos vs FCI.
4. `04_double_factorization_and_the_protocol.py` — DF compression dial; one protocol, two encodings.
5. `05_state_prep_and_injection.py` — injectable `state_prep`; zero-argument synthesizable circuits.
6. `06_bring_your_own_encoding.py` — implement `BlockEncoding` from scratch; `Walk` consumes it unchanged.
