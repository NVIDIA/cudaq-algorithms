# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
assert stateprep.make_hartree_fock_occupation(8, 4, 0) == [0, 1, 2, 3]
assert stateprep.make_hartree_fock_occupation(8, 4, 2) == [0, 1, 2, 4]
with pytest.raises(ValueError, match="must be even when spin > 0"):
    stateprep.make_hartree_fock_occupation(8, 4, 1)
