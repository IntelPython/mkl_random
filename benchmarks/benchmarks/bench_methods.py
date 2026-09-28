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

"""Benchmarks for the ``method`` keyword of mkl_random.MKLRandomState.

Method names must be exact keys of mkl_random's alias tables: an unknown name
silently falls back to the default method.
"""

import numpy as np

from ._utils import _SEED, _SIZES, _make_state

_GAUSSIAN_METHODS = ["ICDF", "BoxMuller", "BoxMuller2"]
# lognormal accepts no BoxMuller2
_LOGNORMAL_METHODS = ["ICDF", "BoxMuller"]
_POISSON_METHODS = ["POISNORM", "PTPE"]
# PTPE uses table lookup below lam = 27 and acceptance/rejection above
_POISSON_LAMS = [10.0, 100.0]


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------


class GaussianMethods:
    """standard_normal / normal for each Gaussian method."""

    params = [_GAUSSIAN_METHODS, _SIZES]
    param_names = ["method", "size"]

    def setup(self, method, size):
        self.rs = _make_state()
        self.rs.standard_normal(size, method=method)
        self.rs.normal(1.0, 2.0, size, method=method)

    def time_standard_normal(self, method, size):
        self.rs.standard_normal(size, method=method)

    def time_normal(self, method, size):
        self.rs.normal(1.0, 2.0, size, method=method)


class LognormalMethods:
    """lognormal for each supported method."""

    params = [_LOGNORMAL_METHODS, _SIZES]
    param_names = ["method", "size"]

    def setup(self, method, size):
        self.rs = _make_state()
        self.rs.lognormal(0.0, 1.0, size, method=method)

    def time_lognormal(self, method, size):
        self.rs.lognormal(0.0, 1.0, size, method=method)


class MultinormalCholesky:
    """multinormal_cholesky (4-D) for each Gaussian method."""

    params = [_GAUSSIAN_METHODS, _SIZES]
    param_names = ["method", "size"]

    def setup(self, method, size):
        self.rs = _make_state()
        rng = np.random.default_rng(_SEED)
        a = rng.standard_normal((4, 4))
        self.mean = rng.standard_normal(4)
        self.ch = np.linalg.cholesky(a @ a.T + 4.0 * np.eye(4))
        self.rs.multinormal_cholesky(self.mean, self.ch, size, method=method)

    def time_multinormal_cholesky(self, method, size):
        self.rs.multinormal_cholesky(self.mean, self.ch, size, method=method)


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------


class PoissonMethods:
    """poisson for each method, below and above the PTPE switch point."""

    params = [_POISSON_METHODS, _POISSON_LAMS, _SIZES]
    param_names = ["method", "lam", "size"]

    def setup(self, method, lam, size):
        self.rs = _make_state()
        self.rs.poisson(lam, size, method=method)

    def time_poisson(self, method, lam, size):
        self.rs.poisson(lam, size, method=method)
