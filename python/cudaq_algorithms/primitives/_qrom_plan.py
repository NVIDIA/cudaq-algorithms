# SPDX-License-Identifier: Apache-2.0
"""Immutable backend-independent QROM plans and stage accounting."""
from dataclasses import dataclass
from ._unary_iteration_ir import Op


@dataclass(frozen=True)
class QROMStageResources:
    name: str
    toffoli_count: int
    tree_and_count: int = 0
    temporary_and_count: int = 0
    num_measurements: int = 0


@dataclass(frozen=True)
class QROMResources:
    stages: tuple[QROMStageResources, ...]
    num_clean: int
    num_dirty: int
    classical_bits: int = 0

    @property
    def toffoli_count(self) -> int:
        return sum(stage.toffoli_count for stage in self.stages)

    @property
    def tree_and_count(self) -> int:
        return sum(stage.tree_and_count for stage in self.stages)

    @property
    def temporary_and_count(self) -> int:
        return sum(stage.temporary_and_count for stage in self.stages)

    @property
    def num_measurements(self) -> int:
        return sum(stage.num_measurements for stage in self.stages)

    @property
    def num_ladder(self) -> int:
        return self.num_clean + self.num_dirty


@dataclass(frozen=True)
class QROMPlan:
    ops: tuple[Op, ...]
    parity_masks: tuple[tuple[int, ...], ...]
    num_results: int
    resources: QROMResources
