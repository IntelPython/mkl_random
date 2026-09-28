# Copyright (c) 2026, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Benchmarks for randint across dtypes, ranges and bound shapes."""

import numpy as np

from ._utils import _SEED, _SIZES, _make_state

# case -> (dtype, low, high). The ranges select the integer fill paths:
#   small  narrow range, drawn with viRngUniform
#   wide   non-power-of-two range at or above INT_MAX (shifted 32-bit draw,
#          or masked 64-bit draw with rejection)
#   pow2   power-of-two range above INT_MAX (masked 64-bit draw, no rejection)
#   full   whole dtype range (raw bits for 32- and 64-bit dtypes)
_CASES = {
    "bool": ("bool", 0, 2),
    "int8_small": ("int8", 0, 100),
    "int8_full": ("int8", -(2**7), 2**7),
    "uint8_small": ("uint8", 0, 100),
    "uint8_full": ("uint8", 0, 2**8),
    "int16_small": ("int16", 0, 100),
    "int16_full": ("int16", -(2**15), 2**15),
    "uint16_small": ("uint16", 0, 100),
    "uint16_full": ("uint16", 0, 2**16),
    "int32_small": ("int32", 0, 100),
    "int32_wide": ("int32", -(2**30), 2**31),
    "int32_full": ("int32", -(2**31), 2**31),
    "uint32_small": ("uint32", 0, 100),
    "uint32_wide": ("uint32", 0, 3 * 2**30),
    "uint32_full": ("uint32", 0, 2**32),
    "int64_small": ("int64", 0, 100),
    "int64_wide": ("int64", 0, 3 * 2**32),
    "int64_pow2": ("int64", 0, 2**40),
    "int64_full": ("int64", -(2**63), 2**63),
    "uint64_small": ("uint64", 0, 100),
    "uint64_wide": ("uint64", 0, 3 * 2**32),
    "uint64_pow2": ("uint64", 0, 2**40),
    "uint64_full": ("uint64", 0, 2**64),
}

_BROADCAST_DTYPES = ["uint8", "int32", "int64"]


class Randint:
    """randint with scalar bounds."""

    params = [list(_CASES), _SIZES]
    param_names = ["case", "size"]

    def setup(self, case, size):
        self.rs = _make_state()
        self.dtype, self.low, self.high = _CASES[case]
        self.rs.randint(self.low, self.high, size, dtype=self.dtype)

    def time_randint(self, case, size):
        self.rs.randint(self.low, self.high, size, dtype=self.dtype)


class RandintBroadcast:
    """randint with an array of upper bounds (one bound per element)."""

    params = [_BROADCAST_DTYPES, _SIZES]
    param_names = ["dtype", "size"]

    def setup(self, dtype, size):
        self.rs = _make_state()
        rng = np.random.default_rng(_SEED)
        self.high = rng.integers(2, 100, size).astype(dtype)
        self.rs.randint(0, self.high, dtype=dtype)

    def time_randint(self, dtype, size):
        self.rs.randint(0, self.high, dtype=dtype)
