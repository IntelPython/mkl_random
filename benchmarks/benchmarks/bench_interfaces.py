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

"""Benchmarks for the NumPy drop-in paths.

``mkl_random.interfaces.numpy_random`` module functions, and ``numpy.random``
while patched by ``mkl_random.patch_numpy_random()``.
"""

import numpy as np

import mkl_random
from mkl_random.interfaces import numpy_random

from ._utils import _SEED, _SIZES


class NumpyRandomInterface:
    """Module functions of mkl_random.interfaces.numpy_random."""

    params = [_SIZES]
    param_names = ["size"]

    def setup(self, size):
        numpy_random.seed(_SEED)
        numpy_random.random_sample(size)
        numpy_random.standard_normal(size)
        numpy_random.normal(1.0, 2.0, size)
        numpy_random.randint(0, 1000, size)
        numpy_random.poisson(10.0, size)

    def time_random_sample(self, size):
        numpy_random.random_sample(size)

    def time_standard_normal(self, size):
        numpy_random.standard_normal(size)

    def time_normal(self, size):
        numpy_random.normal(1.0, 2.0, size)

    def time_randint(self, size):
        numpy_random.randint(0, 1000, size)

    def time_poisson(self, size):
        numpy_random.poisson(10.0, size)


class PatchedNumpyRandom:
    """numpy.random functions while patched by mkl_random.

    Fails instead of timing stock NumPy when the patch does not take effect.
    """

    params = [_SIZES]
    param_names = ["size"]

    def setup(self, size):
        mkl_random.patch_numpy_random()
        served_by = np.random.standard_normal.__module__ or ""
        if not (mkl_random.is_patched() and served_by.startswith("mkl_random")):
            mkl_random.restore_numpy_random()
            raise RuntimeError(
                "[mkl-patch] numpy.random is not served by mkl_random "
                f"after patch_numpy_random() (got {served_by!r})"
            )
        np.random.seed(_SEED)
        np.random.random_sample(size)
        np.random.standard_normal(size)
        np.random.randint(0, 1000, size)

    def teardown(self, size):
        mkl_random.restore_numpy_random()

    def time_random_sample(self, size):
        np.random.random_sample(size)

    def time_standard_normal(self, size):
        np.random.standard_normal(size)

    def time_randint(self, size):
        np.random.randint(0, 1000, size)
