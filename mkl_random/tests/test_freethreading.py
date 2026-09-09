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

import os
import sys
import sysconfig
import threading

import numpy as np
import pytest

# Oversubscription: MKL spawns its own thread pool per calling thread, so
# generating from many Python threads concurrently can spawn far more OS
# threads than cores. Cap it before mkl_random/MKL initialize.
os.environ.setdefault("MKL_NUM_THREADS", "1")

import mkl_random  # noqa: E402

FREE_THREADED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def _run_on_threads(worker, n_threads):
    # Run worker(i) on n_threads and fail if any thread raised.
    errors = []

    def wrapped(i):
        try:
            worker(i)
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(exc)

    threads = [
        threading.Thread(target=wrapped, args=(i,)) for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


def test_concurrent_sampling_per_instance():
    # Each thread owns a private MKLRandomState seeded identically, so the
    # per-instance lock + `nogil` sampling must reproduce the single-threaded
    # result exactly regardless of concurrency.
    n_threads = 4
    size = 10**5 + 1  # large enough that per-thread nogil sampling overlaps
    seed = 1234
    expected = mkl_random.MKLRandomState(seed).normal(size=size)
    results = [None] * n_threads

    def worker(i):
        results[i] = mkl_random.MKLRandomState(seed).normal(size=size)

    _run_on_threads(worker, n_threads)

    for r in results:
        np.testing.assert_array_equal(r, expected)


def test_concurrent_shared_singleton():
    # Module-level functions share a single lock-guarded RandomState. Hammering
    # it from many threads must not corrupt state, crash, or return garbage.
    n_threads = 8
    size = 10**5 + 1
    results = [None] * n_threads

    def worker(i):
        results[i] = mkl_random.uniform(size=size)

    _run_on_threads(worker, n_threads)

    for r in results:
        assert r.shape == (size,)
        assert np.all(np.isfinite(r))


def test_concurrent_patch_restore():
    n_threads = 8
    n_iters = 20

    def worker(_i):
        for _ in range(n_iters):
            mkl_random.patch_numpy_random()
            mkl_random.restore_numpy_random()

    _run_on_threads(worker, n_threads)

    assert not mkl_random.is_patched()


def test_concurrent_multinormal_cholesky_shared():
    # A data race on the shared stream reuses values; assert few duplicates.
    n_threads = 8
    mean = np.zeros(1)
    ch = np.eye(1)
    rng = mkl_random.MKLRandomState(12345)
    chunks = [None] * n_threads

    def worker(i):
        parts = [
            rng.multinormal_cholesky(mean, ch, size=4000).ravel()
            for _ in range(25)
        ]
        chunks[i] = np.concatenate(parts)

    _run_on_threads(worker, n_threads)

    allvals = np.concatenate(chunks)
    assert np.all(np.isfinite(allvals))
    dup_frac = 1.0 - np.unique(allvals).size / allvals.size
    assert dup_frac < 0.01, f"shared stream corrupted: {dup_frac:.3%} dups"


def test_shuffle_reentrancy():
    # shuffle must not hold the lock across a user callback.
    rs = mkl_random.MKLRandomState(1)

    class ReentrantList(list):
        def __setitem__(self, i, v):
            rs.uniform(size=1)
            super().__setitem__(i, v)

    done = threading.Event()

    def run():
        rs.shuffle(ReentrantList(range(8)))
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(timeout=30), "shuffle deadlocked on re-entrant callback"


@pytest.mark.skipif(
    not FREE_THREADED, reason="requires a free-threaded CPython build"
)
def test_gil_not_reenabled_on_import():
    assert not sys._is_gil_enabled()  # pylint: disable=no-member
