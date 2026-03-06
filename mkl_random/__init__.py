# Copyright (c) 2017, Intel Corporation
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

from __future__ import absolute_import, division, print_function

from . import _init_helper
from ._version import __version__
from .mklrand import (
    MKLRandomState,
    RandomState,
    beta,
    binomial,
    bytes,
    chisquare,
    choice,
    dirichlet,
    exponential,
    f,
    gamma,
    geometric,
    get_state,
    gumbel,
    hypergeometric,
    laplace,
    logistic,
    lognormal,
    logseries,
    multinomial,
    multinormal_cholesky,
    multivariate_normal,
    negative_binomial,
    noncentral_chisquare,
    noncentral_f,
    normal,
    pareto,
    permutation,
    poisson,
    power,
    rand,
    randint,
    randn,
    random_integers,
    random_sample,
    rayleigh,
    seed,
    set_state,
    shuffle,
    standard_cauchy,
    standard_exponential,
    standard_gamma,
    standard_normal,
    standard_t,
    triangular,
    uniform,
    vonmises,
    wald,
    weibull,
    zipf,
)

try:
    from numpy.testing.nosetester import _numpy_tester

    test = _numpy_tester().test
    bench = _numpy_tester().bench
    del _numpy_tester
except ModuleNotFoundError:
    # Pytest testing
    from numpy._pytesttester import PytestTester

    test = PytestTester(__name__)
    del PytestTester

from mkl_random import interfaces

from ._patch_numpy import (
    is_patched,
    mkl_random,
    patch_numpy_random,
    patched_names,
    restore_numpy_random,
)

__all__ = [
    "MKLRandomState",
    "RandomState",
    "seed",
    "get_state",
    "set_state",
    "random_sample",
    "choice",
    "randint",
    "bytes",
    "uniform",
    "rand",
    "randn",
    "random_integers",
    "standard_normal",
    "normal",
    "beta",
    "exponential",
    "standard_exponential",
    "standard_gamma",
    "gamma",
    "f",
    "noncentral_f",
    "chisquare",
    "noncentral_chisquare",
    "standard_cauchy",
    "standard_t",
    "vonmises",
    "pareto",
    "weibull",
    "power",
    "laplace",
    "gumbel",
    "logistic",
    "lognormal",
    "rayleigh",
    "wald",
    "triangular",
    "binomial",
    "negative_binomial",
    "poisson",
    "zipf",
    "geometric",
    "hypergeometric",
    "logseries",
    "multivariate_normal",
    "multinormal_cholesky",
    "multinomial",
    "dirichlet",
    "shuffle",
    "permutation",
    "interfaces",
    "patch_numpy_random",
    "restore_numpy_random",
    "is_patched",
    "patched_names",
]

del _init_helper
