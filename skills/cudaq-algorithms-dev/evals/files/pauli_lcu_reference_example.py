# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Representative packaged-API example; staged as read-only evidence."""

from cudaq_algorithms import PauliLCU

encoding = PauliLCU({"Z": 0.75, "X": -0.25})
assert encoding.num_system == 1
assert encoding.alpha == 1.0
