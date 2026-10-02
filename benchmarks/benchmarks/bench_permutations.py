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

"""Benchmarks for shuffle, permutation and choice."""

import numpy as np

from ._utils import _SEED, _SIZES, _make_state

_ROW = 8  # elements per row of the 2-D inputs

# layout -> input builder. The layouts select the shuffle paths:
#   1d_int64  memcpy swaps, specialized for pointer-sized items
#   1d_int32  memcpy swaps, generic item size
#   2d_c      memcpy swaps of contiguous rows
#   2d_f      buffered swaps (rows are not contiguous)
#   list      untyped swaps of Python objects
_SHUFFLE_INPUTS = {
    "1d_int64": lambda n: np.arange(n, dtype=np.int64),
    "1d_int32": lambda n: np.arange(n, dtype=np.int32),
    "2d_c": lambda n: np.arange(n, dtype=np.float64).reshape(-1, _ROW),
    "2d_f": lambda n: np.asfortranarray(
        np.arange(n, dtype=np.float64).reshape(-1, _ROW)
    ),
    "list": lambda n: list(range(n)),
}

_CHOICE_POPULATION = 100_000
_CHOICE_SAMPLES = 10_000
# case -> (replace, weighted)
_CHOICE_CASES = {
    "replace": (True, False),
    "replace_p": (True, True),
    "no_replace": (False, False),
    "no_replace_p": (False, True),
}


class Shuffle:
    """shuffle in place; *size* is the total number of elements."""

    params = [list(_SHUFFLE_INPUTS), _SIZES]
    param_names = ["layout", "size"]

    def setup(self, layout, size):
        self.rs = _make_state()
        self.x = _SHUFFLE_INPUTS[layout](size)
        self.rs.shuffle(self.x)

    def time_shuffle(self, layout, size):
        self.rs.shuffle(self.x)


class Permutation:
    """permutation of a range, a 1-D array and the rows of a 2-D array."""

    params = [["int", "1d", "2d"], _SIZES]
    param_names = ["kind", "size"]

    def setup(self, kind, size):
        self.rs = _make_state()
        if kind == "int":
            self.x = size
        elif kind == "1d":
            self.x = np.arange(size, dtype=np.int64)
        else:
            self.x = np.arange(size, dtype=np.float64).reshape(-1, _ROW)
        self.rs.permutation(self.x)

    def time_permutation(self, kind, size):
        self.rs.permutation(self.x)


class Choice:
    """choice of 10k indices from 100k, with/without replacement and weights."""

    params = [list(_CHOICE_CASES)]
    param_names = ["case"]

    def setup(self, case):
        self.rs = _make_state()
        self.replace, weighted = _CHOICE_CASES[case]
        self.p = None
        if weighted:
            w = np.random.default_rng(_SEED).random(_CHOICE_POPULATION)
            self.p = w / w.sum()
        self.rs.choice(
            _CHOICE_POPULATION, _CHOICE_SAMPLES, replace=self.replace, p=self.p
        )

    def time_choice(self, case):
        self.rs.choice(
            _CHOICE_POPULATION, _CHOICE_SAMPLES, replace=self.replace, p=self.p
        )
