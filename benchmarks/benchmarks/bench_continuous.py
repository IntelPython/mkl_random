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

"""Benchmarks for continuous distributions of mkl_random.MKLRandomState.

Each distribution gets its own benchmark class (Bench_<name>) so ASV renders a
separate grid tile per distribution. Params are size only. Arguments are
chosen so that every distinct sampling code path is covered.
"""

from ._utils import _SIZES, _make_state

# name -> (MKLRandomState method, positional arguments before ``size``)
_CONTINUOUS = {
    "uniform": ("uniform", (0.0, 1.0)),
    "random_sample": ("random_sample", ()),
    "standard_normal": ("standard_normal", ()),
    "normal": ("normal", (1.0, 2.0)),
    "standard_exponential": ("standard_exponential", ()),
    "exponential": ("exponential", (2.0,)),
    # MKL's GNORM gamma method may select different internal algorithms
    # around shape 1 and 0.6; these shapes exercise those regimes
    "standard_gamma": ("standard_gamma", (3.0,)),
    "standard_gamma_mid_shape": ("standard_gamma", (0.8,)),
    "standard_gamma_small_shape": ("standard_gamma", (0.5,)),
    # MKL's CJA beta method may select different internal algorithms (Cheng,
    # Johnk or Atkinson) depending on shapes; these shapes exercise those
    # regimes
    "beta": ("beta", (2.0, 5.0)),
    "beta_small_shape": ("beta", (0.5, 0.5)),
    "beta_mixed_shape": ("beta", (0.5, 2.0)),
    "chisquare": ("chisquare", (3.0,)),
    "standard_t": ("standard_t", (5.0,)),
    "standard_cauchy": ("standard_cauchy", ()),
    "lognormal": ("lognormal", (0.0, 1.0)),
    "laplace": ("laplace", (0.0, 1.0)),
    "gumbel": ("gumbel", (0.0, 1.0)),
    "logistic": ("logistic", (0.0, 1.0)),
    "rayleigh": ("rayleigh", (1.0,)),
    "wald": ("wald", (1.0, 2.0)),
    "weibull": ("weibull", (1.5,)),
    "pareto": ("pareto", (3.0,)),
    "power": ("power", (2.0,)),
    "triangular": ("triangular", (0.0, 0.5, 1.0)),
    # von Mises has separate samplers for kappa <= 1 and kappa > 1
    "vonmises": ("vonmises", (0.0, 4.0)),
    "vonmises_small_kappa": ("vonmises", (0.0, 0.5)),
    "f": ("f", (5.0, 10.0)),
    # noncentral chi-square has separate samplers for df > 1 and df < 1
    "noncentral_chisquare": ("noncentral_chisquare", (3.0, 2.0)),
    "noncentral_chisquare_small_df": ("noncentral_chisquare", (0.5, 2.0)),
    "noncentral_f": ("noncentral_f", (5.0, 10.0, 2.0)),
}


def _make_bench(name, method, args):
    def setup(self, size):
        self._draw = getattr(_make_state(), method)
        self._draw(*args, size=size)

    def time_sample(self, size):
        self._draw(*args, size=size)

    return type(
        f"Bench_{name}",
        (),
        {
            "params": [_SIZES],
            "param_names": ["size"],
            "setup": setup,
            "time_sample": time_sample,
        },
    )


globals().update(
    {
        f"Bench_{name}": _make_bench(name, method, args)
        for name, (method, args) in _CONTINUOUS.items()
    }
)
