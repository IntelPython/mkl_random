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

"""Benchmarks for multivariate distributions; *size* is the sample count."""

import numpy as np

from ._utils import _SEED, _SIZES, _make_state


class Multivariate:
    """multinomial (5 categories), multivariate_normal (3-D), dirichlet (4-D)"""

    params = [_SIZES]
    param_names = ["size"]

    def setup(self, size):
        self.rs = _make_state()
        rng = np.random.default_rng(_SEED)
        a = rng.standard_normal((3, 3))
        self.mean = rng.standard_normal(3)
        self.cov = a @ a.T + 3.0 * np.eye(3)
        self.pvals = np.full(5, 0.2)
        self.alpha = np.array([1.0, 2.0, 3.0, 4.0])
        self.rs.multinomial(20, self.pvals, size)
        self.rs.multivariate_normal(self.mean, self.cov, size)
        self.rs.dirichlet(self.alpha, size)

    def time_multinomial(self, size):
        self.rs.multinomial(20, self.pvals, size)

    def time_multivariate_normal(self, size):
        self.rs.multivariate_normal(self.mean, self.cov, size)

    def time_dirichlet(self, size):
        self.rs.dirichlet(self.alpha, size)
