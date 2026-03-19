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

import numpy as np
import pytest

import mkl_random
import mkl_random.interfaces.numpy_random as _nrand


def test_is_patched():
    """Test that is_patched() returns correct status."""
    assert not mkl_random.is_patched()
    try:
        mkl_random.patch_numpy_random()
        assert mkl_random.is_patched()
        mkl_random.restore_numpy_random()
        assert not mkl_random.is_patched()
    finally:
        while mkl_random.is_patched():
            mkl_random.restore_numpy_random()


def test_patch():
    old_module = np.random.normal.__module__
    assert not mkl_random.is_patched()

    try:
        mkl_random.patch_numpy_random()  # Enable mkl_random in NumPy
        assert mkl_random.is_patched()
        assert np.random.normal.__module__ == _nrand.normal.__module__

        mkl_random.restore_numpy_random()  # Disable mkl_random in NumPy
        assert not mkl_random.is_patched()
        assert np.random.normal.__module__ == old_module
    finally:
        while mkl_random.is_patched():
            mkl_random.restore_numpy_random()


def test_patch_redundant_patching():
    old_module = np.random.normal.__module__
    assert not mkl_random.is_patched()

    try:
        mkl_random.patch_numpy_random()
        mkl_random.patch_numpy_random()

        assert mkl_random.is_patched()
        assert np.random.normal.__module__ == _nrand.normal.__module__

        mkl_random.restore_numpy_random()
        assert mkl_random.is_patched()
        assert np.random.normal.__module__ == _nrand.normal.__module__

        mkl_random.restore_numpy_random()
        assert not mkl_random.is_patched()
        assert np.random.normal.__module__ == old_module
    finally:
        while mkl_random.is_patched():
            mkl_random.restore_numpy_random()


def test_patch_reentrant():
    old_module = np.random.normal.__module__
    assert not mkl_random.is_patched()

    try:
        with mkl_random.mkl_random():
            assert mkl_random.is_patched()
            assert np.random.normal.__module__ == _nrand.normal.__module__

            with mkl_random.mkl_random():
                assert mkl_random.is_patched()
                assert np.random.normal.__module__ == _nrand.normal.__module__

            assert mkl_random.is_patched()
            assert np.random.normal.__module__ == _nrand.normal.__module__

        assert not mkl_random.is_patched()
        assert np.random.normal.__module__ == old_module
    finally:
        while mkl_random.is_patched():
            mkl_random.restore_numpy_random()


def test_patch_warning():
    if mkl_random.is_patched():
        pytest.skip("This test should not be run with a pre-patched NumPy.")
    with pytest.warns(RuntimeWarning, match="restore_numpy_random*"):
        mkl_random.restore_numpy_random()
