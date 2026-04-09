# This file includes tests from numpy.random module:
# https://github.com/numpy/numpy/blob/main/numpy/random/tests/test_randomstate_regression.py

import sys

import numpy as np
import pytest
from numpy.testing import assert_, assert_array_equal, assert_raises

import mkl_random.interfaces.numpy_random as mkl_random


class TestRegression:

    def test_VonMises_range(self):
        # Make sure generated random variables are in [-pi, pi].
        # Regression test for ticket #986.
        for mu in np.linspace(-7.0, 7.0, 5):
            r = mkl_random.vonmises(mu, 1, 50)
            assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    def test_hypergeometric_range(self):
        # Test for ticket #921
        assert_(np.all(mkl_random.hypergeometric(3, 18, 11, size=10) < 4))
        assert_(np.all(mkl_random.hypergeometric(18, 3, 11, size=10) > 0))

        arg = (2**20 - 2, 2**20 - 2, 2**20 - 2)
        # Test for ticket #5623
        # Only check for 32-bit systems due to MKL constraints
        assert_(mkl_random.hypergeometric(*arg) > 0)

    def test_logseries_convergence(self):
        # Test for ticket #923
        N = 1000
        mkl_random.seed(0)
        rvsn = mkl_random.logseries(0.8, size=N)
        # these two frequency counts should be close to theoretical
        # numbers with this large sample
        # theoretical large N result is 0.49706795
        freq = np.sum(rvsn == 1) / N
        msg = f"Frequency was {freq:f}, should be > 0.45"
        assert_(freq > 0.45, msg)
        # theoretical large N result is 0.19882718
        freq = np.sum(rvsn == 2) / N
        msg = f"Frequency was {freq:f}, should be < 0.23"
        assert_(freq < 0.23, msg)

    def test_shuffle_mixed_dimension(self):
        # Test for trac ticket #2074
        # only check that shuffle does not raise an error
        for t in [
            [1, 2, 3, None],
            [(1, 1), (2, 2), (3, 3), None],
            [1, (2, 2), (3, 3), None],
            [(1, 1), 2, 3, None],
        ]:
            shuffled = list(t)
            mkl_random.shuffle(shuffled)

    def test_call_within_randomstate(self):
        # Check that custom RandomState does not call into global state
        m = mkl_random.RandomState()
        res = np.array([2, 0, 4, 7, 0, 9, 1, 6, 3, 1])
        for i in range(3):
            mkl_random.seed(i)
            m.seed(4321)
            # If m.state is not honored, the result will change
            assert_array_equal(m.choice(10, size=10, p=np.ones(10) / 10.0), res)

    def test_multivariate_normal_size_types(self):
        # Test for multivariate_normal issue with 'size' argument.
        # Check that the multivariate_normal size argument can be a
        # numpy integer.
        mkl_random.multivariate_normal([0], [[0]], size=1)
        mkl_random.multivariate_normal([0], [[0]], size=np.int_(1))
        mkl_random.multivariate_normal([0], [[0]], size=np.int64(1))

    def test_beta_small_parameters(self):
        # Test that beta with small a and b parameters does not produce
        # NaNs due to roundoff errors causing 0 / 0, gh-5851
        mkl_random.seed(1234567890)
        x = mkl_random.beta(0.0001, 0.0001, size=100)
        assert_(not np.any(np.isnan(x)), "Nans in mkl_random.beta")

    def test_choice_sum_of_probs_tolerance(self):
        # The sum of probs should be 1.0 with some tolerance.
        # For low precision dtypes the tolerance was too tight.
        # See numpy github issue 6123.
        mkl_random.seed(1234)
        a = [1, 2, 3]
        counts = [4, 4, 2]
        for dt in np.float16, np.float32, np.float64:
            probs = np.array(counts, dtype=dt) / sum(counts)
            c = mkl_random.choice(a, p=probs)
            assert_(c in a)
            assert_raises(ValueError, mkl_random.choice, a, p=probs * 0.9)

    def test_shuffle_of_array_of_different_length_strings(self):
        # Test that permuting an array of different length strings
        # will not cause a segfault on garbage collection
        # Tests gh-7710
        mkl_random.seed(1234)

        a = np.array(["a", "a" * 1000])

        for _ in range(100):
            mkl_random.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc

        gc.collect()

    def test_shuffle_of_array_of_objects(self):
        # Test that permuting an array of objects will not cause
        # a segfault on garbage collection.
        # See gh-7719
        mkl_random.seed(1234)
        a = np.array([np.arange(1), np.arange(4)], dtype=object)

        for _ in range(1000):
            mkl_random.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc

        gc.collect()

    def test_permutation_subclass(self):
        class N(np.ndarray):
            pass

        rng = mkl_random.RandomState()
        orig = np.arange(3).view(N)
        rng.permutation(orig)
        assert_array_equal(orig, np.arange(3).view(N))

        class M:
            a = np.arange(5)

            def __array__(self, dtype=None, copy=None):
                return self.a

        m = M()
        rng.permutation(m)
        assert_array_equal(m.__array__(), np.arange(5))

    def test_warns_byteorder(self):
        # GH 13159
        other_byteord_dt = "<i4" if sys.byteorder == "big" else ">i4"
        with pytest.deprecated_call(match="non-native byteorder is not"):
            mkl_random.randint(0, 200, size=10, dtype=other_byteord_dt)

    def test_named_argument_initialization(self):
        # GH 13669
        rs1 = mkl_random.RandomState(123456789)
        rs2 = mkl_random.RandomState(seed=123456789)
        assert rs1.randint(0, 100) == rs2.randint(0, 100)

    def test_choice_return_dtype(self):
        # GH 9867, now long since the NumPy default changed.
        c = mkl_random.choice(10, p=[0.1] * 10, size=2)
        assert c.dtype == np.dtype(np.long)
        c = mkl_random.choice(10, p=[0.1] * 10, replace=False, size=2)
        assert c.dtype == np.dtype(np.long)
        c = mkl_random.choice(10, size=2)
        assert c.dtype == np.dtype(np.long)
        c = mkl_random.choice(10, replace=False, size=2)
        assert c.dtype == np.dtype(np.long)


def test_multinomial_empty():
    # gh-20483
    # Ensure that empty p-vals are correctly handled
    assert mkl_random.multinomial(10, []).shape == (0,)
    assert mkl_random.multinomial(3, [], size=(7, 5, 3)).shape == (7, 5, 3, 0)


def test_multinomial_1d_pval():
    # gh-20483
    with pytest.raises(TypeError, match="pvals must be a 1-d"):
        mkl_random.multinomial(10, 0.3)
