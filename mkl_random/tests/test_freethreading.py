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
import subprocess
import sys
import sysconfig
import threading
import warnings
from collections import Counter

# Cap MKL threads before numpy (may init MKL).
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

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


def _sample_key(x):
    # Exact key: repr truncates float arrays, so use full-precision bytes.
    a = np.asarray(x)
    return (a.dtype.str, a.shape, a.tobytes())


def _draw_concurrently(rs, call, k):
    # k threads each draw once, released together by a barrier.
    out = [None] * k
    barrier = threading.Barrier(k)

    def body(i):
        barrier.wait()
        out[i] = _sample_key(call(rs))

    threads = [threading.Thread(target=body, args=(i,)) for i in range(k)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return Counter(out)


def test_concurrent_sampling_per_instance():
    # Independent instances: the same seed in each thread must reproduce the
    # single-threaded result regardless of concurrency.
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


def test_concurrent_patch_restore():
    n_threads = 8
    n_iters = 20

    def worker(_i):
        for _ in range(n_iters):
            mkl_random.patch_numpy_random()
            mkl_random.restore_numpy_random()

    _run_on_threads(worker, n_threads)

    assert not mkl_random.is_patched()


_MULTISET_CALLS = {
    "normal": lambda rs: rs.normal(),
    "poisson": lambda rs: rs.poisson(3.0),
    "randint": lambda rs: rs.randint(0, 2**30),
    "_rand_int32": lambda rs: rs._rand_int32(0, 2**30, None),
    "multinomial": lambda rs: rs.multinomial(8, [0.25] * 4),
    "mvn_cholesky": lambda rs: rs.multinormal_cholesky(np.zeros(3), np.eye(3)),
}


@pytest.mark.skipif(
    not FREE_THREADED, reason="race only manifests without the GIL"
)
@pytest.mark.parametrize(
    "call", _MULTISET_CALLS.values(), ids=list(_MULTISET_CALLS)
)
def test_shared_stream_multiset_invariant(call):
    # Concurrent draws must match the serial multiset; a mismatch = race.
    k, rounds, seed = 32, 20, 777
    rs = mkl_random.MKLRandomState(seed)
    rs.seed(seed)
    ref = Counter(_sample_key(call(rs)) for _ in range(k))
    for _ in range(rounds):
        rs.seed(seed)
        assert _draw_concurrently(rs, call, k) == ref


_GET_STATE_RACE = """
import threading
import mkl_random
rs = mkl_random.MKLRandomState(1, brng="MRG32K3A")

def flip():
    for _ in range(20000):
        rs.seed(1, brng="MRG32K3A")
        rs.seed(1, brng="SFMT19937")

def grab():
    for _ in range(20000):
        rs.get_state()

ts = [threading.Thread(target=grab) for _ in range(3)]
ts.append(threading.Thread(target=flip))
for t in ts:
    t.start()
for t in ts:
    t.join()
"""


@pytest.mark.skipif(
    not FREE_THREADED, reason="race only manifests without the GIL"
)
def test_get_state_race_no_heap_overflow():
    # get_state racing a BRNG change must not overflow the buffer (a crash).
    env = dict(os.environ, MKL_NUM_THREADS="1", PYTHONMALLOC="debug")
    proc = subprocess.run(
        [sys.executable, "-c", _GET_STATE_RACE],
        env=env,
        timeout=120,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


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


def test_patch_restore_reentrancy():
    # do_restore must warn outside the lock
    done = threading.Event()

    def run():
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = lambda *a, **k: mkl_random.is_patched()
            mkl_random.restore_numpy_random()  # imbalanced -> warns
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(timeout=30), "patch restore deadlocked in warn callback"


def test_seed_reentrancy():
    # A re-entrant __index__ on the seed must not deadlock
    rs = mkl_random.MKLRandomState(1)

    class ReSeed:
        def __index__(self):
            rs.uniform(size=1)
            return 42

    done = threading.Event()

    def run():
        rs.seed(ReSeed())
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(timeout=30), "seed deadlocked on re-entrant __index__"


def test_set_state_reentrancy():
    # A re-entrant __hash__ on the brng name must not deadlock.
    rs = mkl_random.MKLRandomState(1)
    st = rs.get_state()

    class ReStr(str):
        def __hash__(self):
            rs.uniform(size=1)
            return str.__hash__(self)

    done = threading.Event()

    def run():
        rs.set_state((ReStr(st[0]), st[1]))
        done.set()

    threading.Thread(target=run, daemon=True).start()
    assert done.wait(timeout=30), "set_state deadlocked on re-entrant __hash__"


_GIL_CHECK = "import sys, mkl_random; assert not sys._is_gil_enabled()"


@pytest.mark.skipif(
    not FREE_THREADED, reason="requires a free-threaded CPython build"
)
def test_import_does_not_reenable_gil():
    # Import in a clean subprocess (no forced PYTHON_GIL); GIL must stay off.
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    proc = subprocess.run(
        [sys.executable, "-c", _GIL_CHECK],
        env=env,
        timeout=60,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
