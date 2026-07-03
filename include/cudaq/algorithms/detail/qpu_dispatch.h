/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>

namespace cudaq::algorithms::detail {

inline constexpr std::size_t max_lcu_ancilla_qubits = 10;
inline constexpr std::size_t max_qpu_dispatch_qubits = max_lcu_ancilla_qubits;

} // namespace cudaq::algorithms::detail

// CUDA-Q currently needs explicit controlled-gate arities for these QPU helper
// kernels. Keep the arity ladder in one place so all block-encoding,
// qubitization, and QSVT kernels share the same limit.
//
// CONTRACT: COUNT must be in [0, max_qpu_dispatch_qubits]. These ladders are
// `if/else if` chains with NO terminal `else`, so a COUNT outside the handled
// range silently expands to a no-op (no gate, no error) -- because device
// (`__qpu__`) code cannot throw. Callers are responsible for keeping COUNT in
// range. In practice every register routed through these macros is an LCU
// ancilla / QSVT signal register whose size is `decompose_lcu`-validated
// against `max_lcu_ancilla_qubits` (see pauli_lcu.cpp), so the cap is enforced
// on the host before any kernel runs. A COUNT of 0 (single-term LCU, zero
// ancilla) is intentionally a no-op for the uncontrolled ladders at the
// reflection and signal-phase call sites, where the dropped phase is only an
// unobservable global phase. It is NOT safe where the phase is relative --
// block_encoding::select handles its 0-ancilla negative-coefficient case
// explicitly instead of relying on this ladder. If you add a new caller
// whose register is NOT the validated LCU ancilla, validate its size on the
// host first.
// clang-format off
#define CUDAQ_ALGORITHMS_QARGS_1(QREG) QREG[0]
#define CUDAQ_ALGORITHMS_QARGS_2(QREG) CUDAQ_ALGORITHMS_QARGS_1(QREG), QREG[1]
#define CUDAQ_ALGORITHMS_QARGS_3(QREG) CUDAQ_ALGORITHMS_QARGS_2(QREG), QREG[2]
#define CUDAQ_ALGORITHMS_QARGS_4(QREG) CUDAQ_ALGORITHMS_QARGS_3(QREG), QREG[3]
#define CUDAQ_ALGORITHMS_QARGS_5(QREG) CUDAQ_ALGORITHMS_QARGS_4(QREG), QREG[4]
#define CUDAQ_ALGORITHMS_QARGS_6(QREG) CUDAQ_ALGORITHMS_QARGS_5(QREG), QREG[5]
#define CUDAQ_ALGORITHMS_QARGS_7(QREG) CUDAQ_ALGORITHMS_QARGS_6(QREG), QREG[6]
#define CUDAQ_ALGORITHMS_QARGS_8(QREG) CUDAQ_ALGORITHMS_QARGS_7(QREG), QREG[7]
#define CUDAQ_ALGORITHMS_QARGS_9(QREG) CUDAQ_ALGORITHMS_QARGS_8(QREG), QREG[8]
#define CUDAQ_ALGORITHMS_QARGS_10(QREG) CUDAQ_ALGORITHMS_QARGS_9(QREG), QREG[9]
#define CUDAQ_ALGORITHMS_QARGS_11(QREG) CUDAQ_ALGORITHMS_QARGS_10(QREG), QREG[10]

#define CUDAQ_ALGORITHMS_APPLY_Z_BY_ARITY(QREG, COUNT) \
  { \
    if ((COUNT) == 1) { \
      z(CUDAQ_ALGORITHMS_QARGS_1(QREG)); \
    } else if ((COUNT) == 2) { \
      z<cudaq::ctrl>(CUDAQ_ALGORITHMS_QARGS_2(QREG)); \
    } else if ((COUNT) == 3) { \
      z<cudaq::ctrl>(CUDAQ_ALGORITHMS_QARGS_3(QREG)); \
    } else if ((COUNT) == 4) { \
      z<cudaq::ctrl>(CUDAQ_ALGORITHMS_QARGS_4(QREG)); \
    } else if ((COUNT) == 5) { \
      z<cudaq::ctrl>(CUDAQ_ALGORITHMS_QARGS_5(QREG)); \
    } else if ((COUNT) == 6) { \
      z<cudaq::ctrl>(CUDAQ_ALGORITHMS_QARGS_6(QREG)); \
    } else if ((COUNT) == 7) { \
      z<cudaq::ctrl>(CUDAQ_ALGORITHMS_QARGS_7(QREG)); \
    } else if ((COUNT) == 8) { \
      z<cudaq::ctrl>(CUDAQ_ALGORITHMS_QARGS_8(QREG)); \
    } else if ((COUNT) == 9) { \
      z<cudaq::ctrl>(CUDAQ_ALGORITHMS_QARGS_9(QREG)); \
    } else if ((COUNT) == 10) { \
      z<cudaq::ctrl>(CUDAQ_ALGORITHMS_QARGS_10(QREG)); \
    } \
  }

#define CUDAQ_ALGORITHMS_APPLY_CONTROLLED_Z_BY_ARITY(CONTROL, QREG, COUNT) \
  { \
    if ((COUNT) == 0) { \
      z(CONTROL); \
    } else if ((COUNT) == 1) { \
      z<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_1(QREG)); \
    } else if ((COUNT) == 2) { \
      z<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_2(QREG)); \
    } else if ((COUNT) == 3) { \
      z<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_3(QREG)); \
    } else if ((COUNT) == 4) { \
      z<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_4(QREG)); \
    } else if ((COUNT) == 5) { \
      z<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_5(QREG)); \
    } else if ((COUNT) == 6) { \
      z<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_6(QREG)); \
    } else if ((COUNT) == 7) { \
      z<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_7(QREG)); \
    } else if ((COUNT) == 8) { \
      z<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_8(QREG)); \
    } else if ((COUNT) == 9) { \
      z<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_9(QREG)); \
    } else if ((COUNT) == 10) { \
      z<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_10(QREG)); \
    } \
  }

#define CUDAQ_ALGORITHMS_APPLY_CONTROLLED_GATE_BY_ARITY(GATE, CONTROL, QREG, TARGET, COUNT) \
  { \
    if ((COUNT) == 0) { \
      GATE<cudaq::ctrl>(CONTROL, TARGET); \
    } else if ((COUNT) == 1) { \
      GATE<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_1(QREG), TARGET); \
    } else if ((COUNT) == 2) { \
      GATE<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_2(QREG), TARGET); \
    } else if ((COUNT) == 3) { \
      GATE<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_3(QREG), TARGET); \
    } else if ((COUNT) == 4) { \
      GATE<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_4(QREG), TARGET); \
    } else if ((COUNT) == 5) { \
      GATE<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_5(QREG), TARGET); \
    } else if ((COUNT) == 6) { \
      GATE<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_6(QREG), TARGET); \
    } else if ((COUNT) == 7) { \
      GATE<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_7(QREG), TARGET); \
    } else if ((COUNT) == 8) { \
      GATE<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_8(QREG), TARGET); \
    } else if ((COUNT) == 9) { \
      GATE<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_9(QREG), TARGET); \
    } else if ((COUNT) == 10) { \
      GATE<cudaq::ctrl>(CONTROL, CUDAQ_ALGORITHMS_QARGS_10(QREG), TARGET); \
    } \
  }

#define CUDAQ_ALGORITHMS_APPLY_R1_BY_ARITY(PHASE, QREG, COUNT) \
  { \
    if ((COUNT) == 1) { \
      r1(PHASE, CUDAQ_ALGORITHMS_QARGS_1(QREG)); \
    } else if ((COUNT) == 2) { \
      r1<cudaq::ctrl>(PHASE, CUDAQ_ALGORITHMS_QARGS_2(QREG)); \
    } else if ((COUNT) == 3) { \
      r1<cudaq::ctrl>(PHASE, CUDAQ_ALGORITHMS_QARGS_3(QREG)); \
    } else if ((COUNT) == 4) { \
      r1<cudaq::ctrl>(PHASE, CUDAQ_ALGORITHMS_QARGS_4(QREG)); \
    } else if ((COUNT) == 5) { \
      r1<cudaq::ctrl>(PHASE, CUDAQ_ALGORITHMS_QARGS_5(QREG)); \
    } else if ((COUNT) == 6) { \
      r1<cudaq::ctrl>(PHASE, CUDAQ_ALGORITHMS_QARGS_6(QREG)); \
    } else if ((COUNT) == 7) { \
      r1<cudaq::ctrl>(PHASE, CUDAQ_ALGORITHMS_QARGS_7(QREG)); \
    } else if ((COUNT) == 8) { \
      r1<cudaq::ctrl>(PHASE, CUDAQ_ALGORITHMS_QARGS_8(QREG)); \
    } else if ((COUNT) == 9) { \
      r1<cudaq::ctrl>(PHASE, CUDAQ_ALGORITHMS_QARGS_9(QREG)); \
    } else if ((COUNT) == 10) { \
      r1<cudaq::ctrl>(PHASE, CUDAQ_ALGORITHMS_QARGS_10(QREG)); \
    } \
  }

#define CUDAQ_ALGORITHMS_APPLY_CONTROLLED_R1_BY_ARITY(PHASE, CONTROL, QREG, COUNT) \
  { \
    if ((COUNT) == 1) { \
      r1<cudaq::ctrl>(PHASE, CONTROL, CUDAQ_ALGORITHMS_QARGS_1(QREG)); \
    } else if ((COUNT) == 2) { \
      r1<cudaq::ctrl>(PHASE, CONTROL, CUDAQ_ALGORITHMS_QARGS_2(QREG)); \
    } else if ((COUNT) == 3) { \
      r1<cudaq::ctrl>(PHASE, CONTROL, CUDAQ_ALGORITHMS_QARGS_3(QREG)); \
    } else if ((COUNT) == 4) { \
      r1<cudaq::ctrl>(PHASE, CONTROL, CUDAQ_ALGORITHMS_QARGS_4(QREG)); \
    } else if ((COUNT) == 5) { \
      r1<cudaq::ctrl>(PHASE, CONTROL, CUDAQ_ALGORITHMS_QARGS_5(QREG)); \
    } else if ((COUNT) == 6) { \
      r1<cudaq::ctrl>(PHASE, CONTROL, CUDAQ_ALGORITHMS_QARGS_6(QREG)); \
    } else if ((COUNT) == 7) { \
      r1<cudaq::ctrl>(PHASE, CONTROL, CUDAQ_ALGORITHMS_QARGS_7(QREG)); \
    } else if ((COUNT) == 8) { \
      r1<cudaq::ctrl>(PHASE, CONTROL, CUDAQ_ALGORITHMS_QARGS_8(QREG)); \
    } else if ((COUNT) == 9) { \
      r1<cudaq::ctrl>(PHASE, CONTROL, CUDAQ_ALGORITHMS_QARGS_9(QREG)); \
    } else if ((COUNT) == 10) { \
      r1<cudaq::ctrl>(PHASE, CONTROL, CUDAQ_ALGORITHMS_QARGS_10(QREG)); \
    } \
  }

// NOTE: unlike the COUNT-indexed ladders above (COUNT == total qubits), LAYER
// here is the control-layer index and uses LAYER+1 qubits: LAYER == N expands to
// QARGS_{N+1} (controls QREG[0..N-1] onto target QREG[N]). That is why QARGS_11
// exists. In prepare/unprepare, layer ranges 1..ancilla.size()-1, so with the
// max_qpu_dispatch_qubits cap the largest reachable LAYER is 9 (-> QARGS_10).
#define CUDAQ_ALGORITHMS_APPLY_CONTROLLED_RY_BY_ARITY(ANGLE, QREG, LAYER) \
  { \
    if ((LAYER) == 1) { \
      ry<cudaq::ctrl>(ANGLE, CUDAQ_ALGORITHMS_QARGS_2(QREG)); \
    } else if ((LAYER) == 2) { \
      ry<cudaq::ctrl>(ANGLE, CUDAQ_ALGORITHMS_QARGS_3(QREG)); \
    } else if ((LAYER) == 3) { \
      ry<cudaq::ctrl>(ANGLE, CUDAQ_ALGORITHMS_QARGS_4(QREG)); \
    } else if ((LAYER) == 4) { \
      ry<cudaq::ctrl>(ANGLE, CUDAQ_ALGORITHMS_QARGS_5(QREG)); \
    } else if ((LAYER) == 5) { \
      ry<cudaq::ctrl>(ANGLE, CUDAQ_ALGORITHMS_QARGS_6(QREG)); \
    } else if ((LAYER) == 6) { \
      ry<cudaq::ctrl>(ANGLE, CUDAQ_ALGORITHMS_QARGS_7(QREG)); \
    } else if ((LAYER) == 7) { \
      ry<cudaq::ctrl>(ANGLE, CUDAQ_ALGORITHMS_QARGS_8(QREG)); \
    } else if ((LAYER) == 8) { \
      ry<cudaq::ctrl>(ANGLE, CUDAQ_ALGORITHMS_QARGS_9(QREG)); \
    } else if ((LAYER) == 9) { \
      ry<cudaq::ctrl>(ANGLE, CUDAQ_ALGORITHMS_QARGS_10(QREG)); \
    } else if ((LAYER) == 10) { \
      ry<cudaq::ctrl>(ANGLE, CUDAQ_ALGORITHMS_QARGS_11(QREG)); \
    } \
  }
// clang-format on
