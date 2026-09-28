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

"""Benchmarks across the basic random number generators (``brng``)."""

import numpy as np

from ._utils import _ENGINES, _ENGINES_BITS, _make_state

_N = 1_000_000


class EngineFill:
    """Uniform, Gaussian and narrow-range integer fills for every engine."""

    params = [_ENGINES]
    param_names = ["brng"]

    def setup(self, brng):
        self.rs = _make_state(brng)
        self.rs.random_sample(_N)
        self.rs.standard_normal(_N)
        self.rs.randint(0, 1000, _N, dtype=np.int32)

    def time_random_sample(self, brng):
        self.rs.random_sample(_N)

    def time_standard_normal(self, brng):
        self.rs.standard_normal(_N)

    def time_randint_int32(self, brng):
        self.rs.randint(0, 1000, _N, dtype=np.int32)


class EngineBits:
    """Raw-bit fills, for the engines that implement viRngUniformBits."""

    params = [_ENGINES_BITS]
    param_names = ["brng"]

    def setup(self, brng):
        self.rs = _make_state(brng)
        self.rs.randint(0, 2**32, _N, dtype=np.uint32)
        self.rs.randint(0, 2**64, _N, dtype=np.uint64)
        self.rs.bytes(_N)

    def time_randint_uint32_full(self, brng):
        self.rs.randint(0, 2**32, _N, dtype=np.uint32)

    def time_randint_uint64_full(self, brng):
        self.rs.randint(0, 2**64, _N, dtype=np.uint64)

    def time_bytes(self, brng):
        self.rs.bytes(_N)
