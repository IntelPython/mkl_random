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

"""Benchmarks for array-valued distribution parameters.

One parameter value per output element takes a separate code path from
scalar parameters, so these are tracked apart from bench_continuous.py and
bench_discrete.py.
"""

import numpy as np

from ._utils import _SEED, _make_state

# Smaller than _utils._SIZES: the per-element path is much slower than a
# scalar fill.
_ARRAY_SIZES = [1_000, 100_000]


class ArrayParams:
    """Distributions called with one parameter value per output element."""

    params = [_ARRAY_SIZES]
    param_names = ["size"]

    def setup(self, size):
        self.rs = _make_state()
        rng = np.random.default_rng(_SEED)
        self.loc = rng.standard_normal(size)
        self.scale = rng.uniform(0.5, 2.0, size)
        self.high = self.loc + self.scale
        self.shape = rng.uniform(0.5, 5.0, size)
        self.lam = rng.uniform(1.0, 100.0, size)
        self.n = rng.integers(1, 100, size)
        self.p = rng.uniform(0.1, 0.9, size)
        self.rs.normal(self.loc, self.scale)
        self.rs.uniform(self.loc, self.high)
        self.rs.exponential(self.scale)
        self.rs.standard_gamma(self.shape)
        self.rs.poisson(self.lam)
        self.rs.binomial(self.n, self.p)

    def time_normal(self, size):
        self.rs.normal(self.loc, self.scale)

    def time_uniform(self, size):
        self.rs.uniform(self.loc, self.high)

    def time_exponential(self, size):
        self.rs.exponential(self.scale)

    def time_standard_gamma(self, size):
        self.rs.standard_gamma(self.shape)

    def time_poisson(self, size):
        self.rs.poisson(self.lam)

    def time_binomial(self, size):
        self.rs.binomial(self.n, self.p)
