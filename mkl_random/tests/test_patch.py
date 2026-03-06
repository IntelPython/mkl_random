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
    mkl_random.patch_numpy_random(np)
    assert mkl_random.is_patched()
    mkl_random.restore_numpy_random()
    assert not mkl_random.is_patched()


def test_patch_and_restore():
    """Test patch replacement and restore of original functions."""
    # Store original functions
    orig_normal = np.random.normal
    orig_randint = np.random.randint
    orig_RandomState = np.random.RandomState

    try:
        mkl_random.patch_numpy_random(np)

        # Check that functions are now different objects
        assert np.random.normal is not orig_normal
        assert np.random.randint is not orig_randint
        assert np.random.RandomState is not orig_RandomState

        # Check that they are from mkl_random interface module
        assert np.random.normal is _nrand.normal
        assert np.random.RandomState is _nrand.RandomState

    finally:
        mkl_random.restore_numpy_random()

    # Check that original functions are restored
    assert mkl_random.is_patched() is False
    assert np.random.normal is orig_normal
    assert np.random.randint is orig_randint
    assert np.random.RandomState is orig_RandomState


def test_context_manager():
    """Test context manager patching and automatic restoration."""
    orig_uniform = np.random.uniform
    assert not mkl_random.is_patched()

    with mkl_random.mkl_random(np):
        assert mkl_random.is_patched() is True
        assert np.random.uniform is not orig_uniform
        # Smoke test inside context
        arr = np.random.uniform(size=10)
        assert arr.shape == (10,)

    assert not mkl_random.is_patched()
    assert np.random.uniform is orig_uniform


def test_patched_functions_callable():
    """Smoke test that patched functions are callable without errors."""
    mkl_random.patch_numpy_random(np)
    try:
        # These calls should now be routed to mkl_random's implementations
        x = np.random.standard_normal(size=100)
        assert x.shape == (100,)

        y = np.random.randint(0, 100, size=50)
        assert y.shape == (50,)
        assert np.all(y >= 0) and np.all(y < 100)

        st = np.random.RandomState(12345)
        z = st.rand(10)
        assert z.shape == (10,)

    finally:
        mkl_random.restore_numpy_random()


def test_patched_names():
    """Test that patched_names() returns patched symbol names."""
    try:
        mkl_random.patch_numpy_random(np)
        names = mkl_random.patched_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "normal" in names
        assert "RandomState" in names
    finally:
        mkl_random.restore_numpy_random()


def test_patch_strict_raises_attribute_error():
    """Test strict mode raises AttributeError for missing patch names."""
    # Attempt to patch a clearly non-existent symbol in strict mode.
    with pytest.raises(AttributeError):
        mkl_random.patch_numpy_random(
            np,
            strict=True,
            names=["nonexistent_symbol"],
        )


def test_patch_redundant_patching():
    orig_normal = np.random.normal
    assert not mkl_random.is_patched()

    try:
        mkl_random.patch_numpy_random(np)
        mkl_random.patch_numpy_random(np)
        assert mkl_random.is_patched()
        assert np.random.normal is _nrand.normal
        mkl_random.restore_numpy_random()
        assert mkl_random.is_patched()
        assert np.random.normal is _nrand.normal
        mkl_random.restore_numpy_random()
        assert not mkl_random.is_patched()
        assert np.random.normal is orig_normal
    finally:
        while mkl_random.is_patched():
            mkl_random.restore_numpy_random()


def test_patch_reentrant():
    orig_uniform = np.random.uniform
    assert not mkl_random.is_patched()

    with mkl_random.mkl_random(np):
        assert mkl_random.is_patched()
        assert np.random.uniform is not orig_uniform

        with mkl_random.mkl_random(np):
            assert mkl_random.is_patched()
            assert np.random.uniform is not orig_uniform

        assert mkl_random.is_patched()
        assert np.random.uniform is not orig_uniform

    assert not mkl_random.is_patched()
    assert np.random.uniform is orig_uniform
