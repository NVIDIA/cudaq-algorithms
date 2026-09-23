# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small reproducer supplied by the application team."""


class StructuralBlockEncoding:
    """Expose only the public BlockEncoding protocol of an existing encoding."""

    def __init__(self, encoding):
        self.num_system = encoding.num_system
        self.num_ancilla = encoding.num_ancilla
        self.alpha = encoding.alpha
        self._prepare = encoding.prepare_kernel()
        self._unprepare = encoding.unprepare_kernel()
        self._apply = encoding.apply_kernel()
        self._controlled_apply = encoding.controlled_apply_kernel()
        self._walk = encoding.walk_step_kernel()
        self._adjoint_walk = encoding.adjoint_walk_step_kernel()
        self._controlled_walk = encoding.controlled_walk_step_kernel()
        self._controlled_adjoint_walk = encoding.controlled_adjoint_walk_step_kernel(
        )
        self._observable = encoding.select_observable()

    def prepare_kernel(self):
        return self._prepare

    def unprepare_kernel(self):
        return self._unprepare

    def apply_kernel(self):
        return self._apply

    def controlled_apply_kernel(self):
        return self._controlled_apply

    def walk_step_kernel(self):
        return self._walk

    def adjoint_walk_step_kernel(self):
        return self._adjoint_walk

    def controlled_walk_step_kernel(self):
        return self._controlled_walk

    def controlled_adjoint_walk_step_kernel(self):
        return self._controlled_adjoint_walk

    def select_observable(self):
        return self._observable
