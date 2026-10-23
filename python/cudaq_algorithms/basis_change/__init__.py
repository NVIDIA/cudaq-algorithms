# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fermionic basis changes with classical or coherently supplied angles."""

from ._encoding import AngleEncoding
from ._factory import FermionBasisChange, fermion_basis_change
from ._metrics import (FermionBasisChangeErrorEstimate,
                       FermionBasisChangeResourceEstimate,
                       estimate_fermion_basis_change_error,
                       estimate_fermion_basis_change_resources)
from ._plan import (GivensBasisChangeSchedule, GivensOperation,
                    GivensStructure, compile_givens_basis_change)

__all__ = [
    "AngleEncoding",
    "FermionBasisChange",
    "fermion_basis_change",
    "GivensStructure",
    "GivensOperation",
    "GivensBasisChangeSchedule",
    "compile_givens_basis_change",
    "FermionBasisChangeErrorEstimate",
    "FermionBasisChangeResourceEstimate",
    "estimate_fermion_basis_change_error",
    "estimate_fermion_basis_change_resources",
]
