# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classical encoding of complete Givens angles and exact number phases."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real


@dataclass(frozen=True)
class AngleEncoding:
    """Unsigned little-endian full-turn fixed point with nearest-even rounding.

    ``word[k]`` has radian weight ``2*pi * 2**(k-width)``. Widths from
    one through 52 are supported for the double-valued host parameters;
    rounding is evaluated in double precision (see :meth:`encode`).
    This encoding applies to complete Givens gates and number phases;
    it does not authorize wrapping a standalone Pauli rotation at 2*pi.
    Encoded values are Python integers and do not allocate quantum words.
    """

    #: Number of word bits, an integer from 1 through 52.
    width: int

    def __post_init__(self):
        if (isinstance(self.width, bool)
                or not isinstance(self.width, Integral)
                or not 1 <= self.width <= 52):
            raise ValueError(
                "angle_bits (width) must be an integer from 1 to 52")
        object.__setattr__(self, "width", int(self.width))

    def __reduce__(self):
        # Rebuild through the constructor, so unpickling validates again.
        return (type(self), (self.width, ))

    @property
    def bit_order(self):
        """Word bit order, always ``"little"`` (bit zero is least significant)."""
        return "little"

    @property
    def step(self):
        """Radians per code unit, ``2*pi / 2**width``."""
        return math.ldexp(math.tau, -self.width)

    @property
    def weights(self):
        """Radian weight of each word bit, least significant first."""
        return tuple(
            math.ldexp(math.tau, bit - self.width)
            for bit in range(self.width))

    def encode(self, angle):
        """Round a finite real radian angle to an unsigned full-turn code.

        The angle is first converted to an IEEE double, so exact integers
        beyond ``2**53``, ``Fraction`` and extended-precision values are
        rounded before reduction. The quotient by ``step`` is also rounded
        to a double before nearest-even rounding: from about 43 bits, and
        for some floating-point near-ties at smaller widths, a code can be
        one step from the exact nearest code, less than about 1e-15 rad
        beyond half a step. Exact half-step ties within one signed turn
        resolve to even; beyond one turn, ties follow the trigonometric
        argument reduction.
        """
        if isinstance(angle, bool) or not isinstance(angle, Real):
            raise ValueError("angle must be a finite real number")
        try:
            angle = float(angle)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("angle must be a finite real number") from exc
        if not math.isfinite(angle):
            raise ValueError("angle must be a finite real number")
        # Keep canonical inputs on their original floating-point grid: even
        # atan2(sin(pi/8), cos(pi/8)) can perturb an exact half-step tie. Outside
        # one signed full turn, use libm's trigonometric argument reduction.
        # Taking a remainder by floating-point tau accumulates its period error
        # for large angles and can select a completely different physical phase.
        reduced = (math.remainder(angle, math.tau) if abs(angle) <= math.tau
                   else math.atan2(math.sin(angle), math.cos(angle)))
        return round(reduced / self.step) % (1 << self.width)

    def decode(self, code):
        """Return the represented angle in [0, 2*pi), without measurement.

        The result is ``code * step`` rounded to a double.
        """
        if (isinstance(code, bool) or not isinstance(code, Integral)
                or not 0 <= code < (1 << self.width)):
            raise ValueError(
                "code must be an integer within the declared width")
        return int(code) * self.step
