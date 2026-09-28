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

"""Benchmarks for discrete distributions of mkl_random.MKLRandomState.

Each distribution gets its own benchmark class (Bench_<name>) so ASV renders a
separate grid tile per distribution. Params are size only. ``randint`` is
covered in bench_integers.py.
"""

from ._utils import _SIZES, _make_state

# name -> (MKLRandomState method, positional arguments before ``size``)
_DISCRETE = {
    # MKL's BTPE binomial method uses acceptance/rejection only when
    # n * min(p, 1 - p) >= 30
    "binomial": ("binomial", (10, 0.5)),
    "binomial_large_n": ("binomial", (1000, 0.3)),
    "negative_binomial": ("negative_binomial", (5, 0.5)),
    # default method (POISNORM); per-method timings are in bench_methods.py
    "poisson": ("poisson", (10.0,)),
    "geometric": ("geometric", (0.3,)),
    # MKL's H2PE hypergeometric method uses acceptance/rejection for a large
    # mode
    "hypergeometric": ("hypergeometric", (10, 20, 5)),
    "hypergeometric_large_mode": ("hypergeometric", (1000, 2000, 500)),
    "zipf": ("zipf", (2.0,)),
    "logseries": ("logseries", (0.9,)),
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
        for name, (method, args) in _DISCRETE.items()
    }
)
