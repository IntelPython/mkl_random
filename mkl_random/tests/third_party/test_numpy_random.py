# This file includes tests from numpy.random module:
# https://github.com/numpy/numpy/blob/main/numpy/random/tests/test_random.py

import warnings

import numpy as np
import pytest
from numpy.testing import (
    assert_,
    assert_array_equal,
    assert_equal,
    assert_no_warnings,
    assert_raises,
)

import mkl_random.interfaces.numpy_random as mkl_random


class TestSeed:
    # MKL can't guarantee that seed will produce the same results for different
    # architectures or platforms, so we test that the seed is accepted
    # and produces a result
    def test_scalar(self):
        s = mkl_random.RandomState(0)
        assert isinstance(s.randint(1000), int)

    def test_array(self):
        s = mkl_random.RandomState(range(10))
        assert isinstance(s.randint(1000), int)
        s = mkl_random.RandomState(np.arange(10))
        assert isinstance(s.randint(1000), int)
        s = mkl_random.RandomState([0])
        assert isinstance(s.randint(1000), int)
        s = mkl_random.RandomState([4294967295])
        assert isinstance(s.randint(1000), int)

    def test_invalid_scalar(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, mkl_random.RandomState, -0.5)
        assert_raises(ValueError, mkl_random.RandomState, -1)

    def test_invalid_array(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, mkl_random.RandomState, [-0.5])
        assert_raises(ValueError, mkl_random.RandomState, [-1])
        assert_raises(ValueError, mkl_random.RandomState, [4294967296])
        assert_raises(ValueError, mkl_random.RandomState, [1, 2, 4294967296])
        assert_raises(ValueError, mkl_random.RandomState, [1, -2, 4294967296])

    def test_invalid_array_shape(self):
        # gh-9832
        assert_raises(
            ValueError, mkl_random.RandomState, np.array([], dtype=np.int64)
        )
        assert_raises(ValueError, mkl_random.RandomState, [[1, 2, 3]])
        assert_raises(
            ValueError, mkl_random.RandomState, [[1, 2, 3], [4, 5, 6]]
        )


class TestBinomial:
    def test_n_zero(self):
        # Tests the corner case of n == 0 for the binomial distribution.
        # binomial(0, p) should be zero for any p in [0, 1].
        # This test addresses issue #3480.
        zeros = np.zeros(2, dtype="int")
        for p in [0, 0.5, 1]:
            assert_(mkl_random.binomial(0, p) == 0)
            assert_array_equal(mkl_random.binomial(zeros, p), zeros)

    def test_p_is_nan(self):
        # Issue #4571.
        assert_raises(ValueError, mkl_random.binomial, 1, np.nan)


class TestMultinomial:
    def test_basic(self):
        mkl_random.multinomial(100, [0.2, 0.8])

    def test_zero_probability(self):
        mkl_random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def test_int_negative_interval(self):
        assert_(-5 <= mkl_random.randint(-5, -1) < -1)
        x = mkl_random.randint(-5, -1, 5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))

    def test_size(self):
        # gh-3173
        p = [0.5, 0.5]
        assert_equal(mkl_random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(mkl_random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(mkl_random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(mkl_random.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        assert_equal(mkl_random.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        assert_equal(
            mkl_random.multinomial(1, p, np.array((2, 2))).shape, (2, 2, 2)
        )

        assert_raises(TypeError, mkl_random.multinomial, 1, p, float(1))

    def test_multidimensional_pvals(self):
        assert_raises(ValueError, mkl_random.multinomial, 10, [[0, 1]])
        assert_raises(ValueError, mkl_random.multinomial, 10, [[0], [1]])
        assert_raises(
            ValueError, mkl_random.multinomial, 10, [[[0], [1]], [[1], [0]]]
        )
        assert_raises(
            ValueError, mkl_random.multinomial, 10, np.array([[0, 1], [1, 0]])
        )


class TestSetState:
    def _create_rng(self):
        seed = 1234567890
        prng = mkl_random.RandomState(seed)
        state = prng.get_state()
        return prng, state

    def test_basic(self):
        prng, state = self._create_rng()
        old = prng.tomaxint(16)
        prng.set_state(state)
        new = prng.tomaxint(16)
        assert_(np.all(old == new))

    def test_negative_binomial(self):
        # Ensure that the negative binomial results take floating point
        # arguments without truncation.
        prng, _ = self._create_rng()
        prng.negative_binomial(0.5, 0.5)

    def test_set_invalid_state(self):
        # gh-25402
        prng, _ = self._create_rng()
        with pytest.raises(IndexError):
            prng.set_state(())


class TestRandint:
    # valid integer/boolean types
    itype = [
        np.bool,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ]

    def test_unsupported_type(self):
        rng = mkl_random.RandomState()
        assert_raises(TypeError, rng.randint, 1, dtype=float)

    def test_bounds_checking(self):
        rng = mkl_random.RandomState()
        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1
            assert_raises(ValueError, rng.randint, lbnd - 1, ubnd, dtype=dt)
            assert_raises(ValueError, rng.randint, lbnd, ubnd + 1, dtype=dt)
            assert_raises(ValueError, rng.randint, ubnd, lbnd, dtype=dt)
            assert_raises(ValueError, rng.randint, 1, 0, dtype=dt)

    def test_rng_zero_and_extremes(self):
        rng = mkl_random.RandomState()
        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            tgt = ubnd - 1
            assert_equal(rng.randint(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(rng.randint(tgt, tgt + 1, size=1000, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(rng.randint(tgt, tgt + 1, size=1000, dtype=dt), tgt)

    def test_full_range(self):
        # Test for ticket #1690
        rng = mkl_random.RandomState()

        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            try:
                rng.randint(lbnd, ubnd, dtype=dt)
            except Exception as e:
                raise AssertionError(
                    "No error should have been raised, "
                    "but one was with the following "
                    f"message:\n\n{str(e)}"
                )

    def test_in_bounds_fuzz(self):
        rng = mkl_random.RandomState()

        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = rng.randint(2, ubnd, size=2**16, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)

        vals = rng.randint(0, 2, size=2**16, dtype=np.bool)

        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    def test_int64_uint64_corner_case(self):
        # When stored in Numpy arrays, `lbnd` is casted
        # as np.int64, and `ubnd` is casted as np.uint64.
        # Checking whether `lbnd` >= `ubnd` used to be
        # done solely via direct comparison, which is incorrect
        # because when Numpy tries to compare both numbers,
        # it casts both to np.float64 because there is
        # no integer superset of np.int64 and np.uint64. However,
        # `ubnd` is too large to be represented in np.float64,
        # causing it be round down to np.iinfo(np.int64).max,
        # leading to a ValueError because `lbnd` now equals
        # the new `ubnd`.

        dt = np.int64
        tgt = np.iinfo(np.int64).max
        lbnd = np.int64(np.iinfo(np.int64).max)
        ubnd = np.uint64(np.iinfo(np.int64).max + 1)

        # None of these function calls should
        # generate a ValueError now.
        actual = mkl_random.randint(lbnd, ubnd, dtype=dt)
        assert_equal(actual, tgt)

    def test_respect_dtype_singleton(self):
        # See gh-7203
        rng = mkl_random.RandomState()
        for dt in self.itype:
            lbnd = 0 if dt is np.bool else np.iinfo(dt).min
            ubnd = 2 if dt is np.bool else np.iinfo(dt).max + 1

            sample = rng.randint(lbnd, ubnd, dtype=dt)
            assert_equal(sample.dtype, np.dtype(dt))

        for dt in (bool, int):
            # The legacy rng uses "long" as the default integer:
            lbnd = 0 if dt is bool else np.iinfo("long").min
            ubnd = 2 if dt is bool else np.iinfo("long").max + 1

            # gh-7284: Ensure that we get Python data types
            sample = rng.randint(lbnd, ubnd, dtype=dt)
            assert_(not hasattr(sample, "dtype"))
            assert_equal(type(sample), dt)


class TestRandomDist:
    def test_random_integers_max_int(self):
        # Tests whether random_integers can generate the
        # maximum allowed Python int that can be converted
        # into a C long. Previous implementations of this
        # method have thrown an OverflowError when attempting
        # to generate this integer.
        with pytest.warns(DeprecationWarning):
            actual = mkl_random.random_integers(
                np.iinfo("l").max, np.iinfo("l").max
            )

        desired = np.iinfo("l").max
        assert_equal(actual, desired)

    def test_random_integers_deprecated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            # DeprecationWarning raised with high == None
            assert_raises(
                DeprecationWarning,
                mkl_random.random_integers,
                np.iinfo("l").max,
            )

            # DeprecationWarning raised with high != None
            assert_raises(
                DeprecationWarning,
                mkl_random.random_integers,
                np.iinfo("l").max,
                np.iinfo("l").max,
            )

    def test_choice_exceptions(self):
        sample = mkl_random.choice
        assert_raises(ValueError, sample, -1, 3)
        assert_raises(ValueError, sample, 3.0, 3)
        assert_raises(ValueError, sample, [[1, 2], [3, 4]], 3)
        assert_raises(ValueError, sample, [], 3)
        assert_raises(
            ValueError, sample, [1, 2, 3, 4], 3, p=[[0.25, 0.25], [0.25, 0.25]]
        )
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        # gh-13087
        assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
        assert_raises(
            ValueError, sample, [1, 2, 3], 2, replace=False, p=[1, 0, 0]
        )

    def test_choice_return_shape(self):
        p = [0.1, 0.9]
        # Check scalar
        assert_(np.isscalar(mkl_random.choice(2, replace=True)))
        assert_(np.isscalar(mkl_random.choice(2, replace=False)))
        assert_(np.isscalar(mkl_random.choice(2, replace=True, p=p)))
        assert_(np.isscalar(mkl_random.choice(2, replace=False, p=p)))
        assert_(np.isscalar(mkl_random.choice([1, 2], replace=True)))
        assert_(mkl_random.choice([None], replace=True) is None)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(mkl_random.choice(arr, replace=True) is a)

        # Check 0-d array
        s = ()
        assert_(not np.isscalar(mkl_random.choice(2, s, replace=True)))
        assert_(not np.isscalar(mkl_random.choice(2, s, replace=False)))
        assert_(not np.isscalar(mkl_random.choice(2, s, replace=True, p=p)))
        assert_(not np.isscalar(mkl_random.choice(2, s, replace=False, p=p)))
        assert_(not np.isscalar(mkl_random.choice([1, 2], s, replace=True)))
        assert_(mkl_random.choice([None], s, replace=True).ndim == 0)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(mkl_random.choice(arr, s, replace=True).item() is a)

        # Check multi dimensional array
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(mkl_random.choice(6, s, replace=True).shape, s)
        assert_equal(mkl_random.choice(6, s, replace=False).shape, s)
        assert_equal(mkl_random.choice(6, s, replace=True, p=p).shape, s)
        assert_equal(mkl_random.choice(6, s, replace=False, p=p).shape, s)
        assert_equal(mkl_random.choice(np.arange(6), s, replace=True).shape, s)

        # Check zero-size
        assert_equal(mkl_random.randint(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(mkl_random.randint(0, -10, size=0).shape, (0,))
        assert_equal(mkl_random.randint(10, 10, size=0).shape, (0,))
        assert_equal(mkl_random.choice(0, size=0).shape, (0,))
        assert_equal(mkl_random.choice([], size=(0,)).shape, (0,))
        assert_equal(
            mkl_random.choice(["a", "b"], size=(3, 0, 4)).shape, (3, 0, 4)
        )
        assert_raises(ValueError, mkl_random.choice, [], 10)

    def test_choice_nan_probabilities(self):
        a = np.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, mkl_random.choice, a, p=p)

    def test_shuffle(self):
        # Test lists, arrays (of various dtypes), and multidimensional versions
        # of both, c-contiguous or not:
        for conv in [
            lambda x: np.array([]),
            lambda x: x,
            lambda x: np.asarray(x).astype(np.int8),
            lambda x: np.asarray(x).astype(np.float32),
            lambda x: np.asarray(x).astype(np.complex64),
            lambda x: np.asarray(x).astype(object),
            lambda x: [(i, i) for i in x],
            lambda x: np.asarray([[i, i] for i in x]),
            lambda x: np.vstack([x, x]).T,
            # gh-11442
            lambda x: (
                np.asarray([(i, i) for i in x], [("a", int), ("b", int)]).view(
                    np.recarray
                )
            ),
            # gh-4270
            lambda x: np.asarray(
                [(i, i) for i in x], [("a", object), ("b", np.int32)]
            ),
        ]:
            rng = mkl_random.RandomState()
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            # Do not validate against expected results as we cannot guarantee
            # consistency across platforms or architectures.
            # This test is just to check that it runs on all types
            rng.shuffle(alist)

    @pytest.mark.parametrize("random", [mkl_random, mkl_random.RandomState()])
    def test_shuffle_untyped_warning(self, random):
        # Create a dict works like a sequence but isn't one
        values = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        with pytest.warns(
            UserWarning, match="you are shuffling a 'dict' object"
        ):
            random.shuffle(values)

    @pytest.mark.parametrize("random", [mkl_random, mkl_random.RandomState()])
    @pytest.mark.parametrize("use_array_like", [True, False])
    def test_shuffle_no_object_unpacking(self, random, use_array_like):
        class MyArr(np.ndarray):
            pass

        items = [
            None,
            np.array([3]),
            np.float64(3),
            np.array(10),
            np.float64(7),
        ]
        arr = np.array(items, dtype=object)
        item_ids = {id(i) for i in items}
        if use_array_like:
            arr = arr.view(MyArr)

        # The array was created fine, and did not modify any objects:
        assert all(id(i) in item_ids for i in arr)

        if use_array_like:
            with pytest.warns(
                UserWarning, match="Shuffling a one dimensional array.*"
            ):
                random.shuffle(arr)
        else:
            random.shuffle(arr)
            assert all(id(i) in item_ids for i in arr)

    def test_shuffle_memoryview(self):
        # gh-18273
        # allow graceful handling of memoryviews
        # (treat the same as arrays)
        rng = mkl_random.RandomState()
        a = np.arange(5).data
        rng.shuffle(a)

    def test_shuffle_not_writeable(self):
        a = np.zeros(3)
        a.flags.writeable = False
        with pytest.raises(ValueError, match="read-only"):
            mkl_random.shuffle(a)

    def test_dirichlet_size(self):
        # gh-3173
        p = np.array([51.72840233779265162, 39.74494232180943953])
        assert_equal(mkl_random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(mkl_random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(mkl_random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(mkl_random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(mkl_random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(mkl_random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))

        assert_raises(TypeError, mkl_random.dirichlet, p, float(1))

    def test_dirichlet_bad_alpha(self):
        # gh-2089
        alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, mkl_random.dirichlet, alpha)

        # gh-15876
        assert_raises(ValueError, mkl_random.dirichlet, [[5, 1]])
        assert_raises(ValueError, mkl_random.dirichlet, [[5], [1]])
        assert_raises(
            ValueError, mkl_random.dirichlet, [[[5], [1]], [[1], [5]]]
        )
        assert_raises(
            ValueError, mkl_random.dirichlet, np.array([[5, 1], [1, 5]])
        )

    def test_multivariate_normal_warnings(self):
        rng = mkl_random.RandomState()

        # Check that non positive-semidefinite covariance warns with
        # RuntimeWarning
        mean = [0, 0]
        cov = [[1, 2], [2, 1]]
        pytest.warns(RuntimeWarning, rng.multivariate_normal, mean, cov)

        # and that it doesn't warn with RuntimeWarning check_valid='ignore'
        assert_no_warnings(
            rng.multivariate_normal, mean, cov, check_valid="ignore"
        )

        # and that it raises with RuntimeWarning check_valid='raises'
        assert_raises(
            ValueError, rng.multivariate_normal, mean, cov, check_valid="raise"
        )

        cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rng.multivariate_normal(mean, cov)

    def test_poisson_exceptions(self):
        lambig = np.iinfo("l").max
        lamneg = -1
        assert_raises(ValueError, mkl_random.poisson, lamneg)
        assert_raises(ValueError, mkl_random.poisson, [lamneg] * 10)
        assert_raises(ValueError, mkl_random.poisson, lambig)
        assert_raises(ValueError, mkl_random.poisson, [lambig] * 10)

    # TODO: revisit test after experimenting with range calculation in uniform
    @pytest.mark.skip("Uniform does not overflow identically to NumPy")
    def test_uniform_range_bounds(self):
        fmin = np.finfo("float").min
        fmax = np.finfo("float").max

        func = mkl_random.uniform
        assert_raises(OverflowError, func, -np.inf, 0)
        assert_raises(OverflowError, func, 0, np.inf)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, [-np.inf], [0])
        assert_raises(OverflowError, func, [0], [np.inf])

        # (fmax / 1e17) - fmin is within range, so this should not throw
        # account for i386 extended precision DBL_MAX / 1e17 + DBL_MAX >
        # DBL_MAX by increasing fmin a bit
        mkl_random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e17)

    # TODO: revisit after changing conversion logic
    @pytest.mark.skip("mkl_random casts via NumPy instead of throwing")
    def test_scalar_exception_propagation(self):
        # Tests that exceptions are correctly propagated in distributions
        # when called with objects that throw exceptions when converted to
        # scalars.
        #
        # Regression test for gh: 8865

        class ThrowingFloat(np.ndarray):
            def __float__(self):
                raise TypeError

        throwing_float = np.array(1.0).view(ThrowingFloat)
        assert_raises(
            TypeError, mkl_random.uniform, throwing_float, throwing_float
        )

        class ThrowingInteger(np.ndarray):
            def __int__(self):
                raise TypeError

            __index__ = __int__

        throwing_int = np.array(1).view(ThrowingInteger)
        assert_raises(TypeError, mkl_random.hypergeometric, throwing_int, 1, 1)

    def test_vonmises_small(self):
        # check infinite loop, gh-4720
        mkl_random.seed()
        r = mkl_random.vonmises(mu=0.0, kappa=1.1e-8, size=10**6)
        np.testing.assert_(np.isfinite(r).all())


class TestBroadcast:
    def test_uniform(self):
        low = [0]
        high = [1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.uniform(low * 3, high).shape, desired_shape)
        assert_equal(rng.uniform(low, high * 3).shape, desired_shape)

    def test_normal(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.normal(loc * 3, scale).shape, desired_shape)
        assert_raises(ValueError, rng.normal, loc * 3, bad_scale)

        assert_equal(rng.normal(loc, scale * 3).shape, desired_shape)
        assert_raises(ValueError, rng.normal, loc, bad_scale * 3)

    def test_beta(self):
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.beta(a * 3, b).shape, desired_shape)
        assert_raises(ValueError, rng.beta, bad_a * 3, b)
        assert_raises(ValueError, rng.beta, a * 3, bad_b)

        assert_equal(rng.beta(a, b * 3).shape, desired_shape)
        assert_raises(ValueError, rng.beta, bad_a, b * 3)
        assert_raises(ValueError, rng.beta, a, bad_b * 3)

    def test_exponential(self):
        scale = [1]
        bad_scale = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.exponential(scale * 3).shape, desired_shape)
        assert_raises(ValueError, rng.exponential, bad_scale * 3)

    def test_standard_gamma(self):
        shape = [1]
        bad_shape = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.standard_gamma(shape * 3).shape, desired_shape)
        assert_raises(ValueError, rng.standard_gamma, bad_shape * 3)

    def test_gamma(self):
        shape = [1]
        scale = [2]
        bad_shape = [-1]
        bad_scale = [-2]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.gamma(shape * 3, scale).shape, desired_shape)
        assert_raises(ValueError, rng.gamma, bad_shape * 3, scale)
        assert_raises(ValueError, rng.gamma, shape * 3, bad_scale)

        assert_equal(rng.gamma(shape, scale * 3).shape, desired_shape)
        assert_raises(ValueError, rng.gamma, bad_shape, scale * 3)
        assert_raises(ValueError, rng.gamma, shape, bad_scale * 3)

    def test_f(self):
        dfnum = [1]
        dfden = [2]
        bad_dfnum = [-1]
        bad_dfden = [-2]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.f(dfnum * 3, dfden).shape, desired_shape)
        assert_raises(ValueError, rng.f, bad_dfnum * 3, dfden)
        assert_raises(ValueError, rng.f, dfnum * 3, bad_dfden)

        assert_equal(rng.f(dfnum, dfden * 3).shape, desired_shape)
        assert_raises(ValueError, rng.f, bad_dfnum, dfden * 3)
        assert_raises(ValueError, rng.f, dfnum, bad_dfden * 3)

    def test_noncentral_f(self):
        dfnum = [2]
        dfden = [3]
        nonc = [4]
        bad_dfnum = [0]
        bad_dfden = [-1]
        bad_nonc = [-2]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(
            rng.noncentral_f(dfnum * 3, dfden, nonc).shape, desired_shape
        )
        assert_raises(ValueError, rng.noncentral_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, rng.noncentral_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, rng.noncentral_f, dfnum * 3, dfden, bad_nonc)

        assert_equal(
            rng.noncentral_f(dfnum, dfden * 3, nonc).shape, desired_shape
        )
        assert_raises(ValueError, rng.noncentral_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, rng.noncentral_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, rng.noncentral_f, dfnum, dfden * 3, bad_nonc)

        assert_equal(
            rng.noncentral_f(dfnum, dfden, nonc * 3).shape, desired_shape
        )
        assert_raises(ValueError, rng.noncentral_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, rng.noncentral_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, rng.noncentral_f, dfnum, dfden, bad_nonc * 3)

    def test_chisquare(self):
        df = [1]
        bad_df = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.chisquare(df * 3).shape, desired_shape)
        assert_raises(ValueError, rng.chisquare, bad_df * 3)

    def test_noncentral_chisquare(self):
        df = [1]
        nonc = [2]
        bad_df = [-1]
        bad_nonc = [-2]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(
            rng.noncentral_chisquare(df * 3, nonc).shape, desired_shape
        )
        assert_raises(ValueError, rng.noncentral_chisquare, bad_df * 3, nonc)
        assert_raises(ValueError, rng.noncentral_chisquare, df * 3, bad_nonc)

        assert_equal(
            rng.noncentral_chisquare(df, nonc * 3).shape, desired_shape
        )
        assert_raises(ValueError, rng.noncentral_chisquare, bad_df, nonc * 3)
        assert_raises(ValueError, rng.noncentral_chisquare, df, bad_nonc * 3)

    def test_standard_t(self):
        df = [1]
        bad_df = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.standard_t(df * 3).shape, desired_shape)
        assert_raises(ValueError, rng.standard_t, bad_df * 3)

    def test_vonmises(self):
        mu = [2]
        kappa = [1]
        bad_kappa = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.vonmises(mu * 3, kappa).shape, desired_shape)
        assert_raises(ValueError, rng.vonmises, mu * 3, bad_kappa)

        assert_equal(rng.vonmises(mu, kappa * 3).shape, desired_shape)
        assert_raises(ValueError, rng.vonmises, mu, bad_kappa * 3)

    def test_pareto(self):
        a = [1]
        bad_a = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.pareto(a * 3).shape, desired_shape)
        assert_raises(ValueError, rng.pareto, bad_a * 3)

    def test_weibull(self):
        a = [1]
        bad_a = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.weibull(a * 3).shape, desired_shape)
        assert_raises(ValueError, rng.weibull, bad_a * 3)

    def test_power(self):
        a = [1]
        bad_a = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.power(a * 3).shape, desired_shape)
        assert_raises(ValueError, rng.power, bad_a * 3)

    def test_laplace(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.laplace(loc * 3, scale).shape, desired_shape)
        assert_raises(ValueError, rng.laplace, loc * 3, bad_scale)

        assert_equal(rng.laplace(loc, scale * 3).shape, desired_shape)
        assert_raises(ValueError, rng.laplace, loc, bad_scale * 3)

    def test_gumbel(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.gumbel(loc * 3, scale).shape, desired_shape)
        assert_raises(ValueError, rng.gumbel, loc * 3, bad_scale)

        assert_equal(rng.gumbel(loc, scale * 3).shape, desired_shape)
        assert_raises(ValueError, rng.gumbel, loc, bad_scale * 3)

    def test_logistic(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.logistic(loc * 3, scale).shape, desired_shape)
        assert_raises(ValueError, rng.logistic, loc * 3, bad_scale)

        assert_equal(rng.logistic(loc, scale * 3).shape, desired_shape)
        assert_raises(ValueError, rng.logistic, loc, bad_scale * 3)

    def test_lognormal(self):
        mean = [0]
        sigma = [1]
        bad_sigma = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.lognormal(mean * 3, sigma).shape, desired_shape)
        assert_raises(ValueError, rng.lognormal, mean * 3, bad_sigma)

        assert_equal(rng.lognormal(mean, sigma * 3).shape, desired_shape)
        assert_raises(ValueError, rng.lognormal, mean, bad_sigma * 3)

    def test_rayleigh(self):
        scale = [1]
        bad_scale = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.rayleigh(scale * 3).shape, desired_shape)
        assert_raises(ValueError, rng.rayleigh, bad_scale * 3)

    def test_wald(self):
        mean = [0.5]
        scale = [1]
        bad_mean = [0]
        bad_scale = [-2]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.wald(mean * 3, scale).shape, desired_shape)
        assert_raises(ValueError, rng.wald, bad_mean * 3, scale)
        assert_raises(ValueError, rng.wald, mean * 3, bad_scale)

        assert_equal(rng.wald(mean, scale * 3).shape, desired_shape)
        assert_raises(ValueError, rng.wald, bad_mean, scale * 3)
        assert_raises(ValueError, rng.wald, mean, bad_scale * 3)
        assert_raises(ValueError, rng.wald, 0.0, 1)
        assert_raises(ValueError, rng.wald, 0.5, 0.0)

    def test_triangular(self):
        left = [1]
        right = [3]
        mode = [2]
        bad_left_one = [3]
        bad_mode_one = [4]
        bad_left_two, bad_mode_two = right * 2

        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.triangular(left * 3, mode, right).shape, desired_shape)
        assert_raises(ValueError, rng.triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, rng.triangular, left * 3, bad_mode_one, right)
        assert_raises(
            ValueError, rng.triangular, bad_left_two * 3, bad_mode_two, right
        )

        assert_equal(rng.triangular(left, mode * 3, right).shape, desired_shape)
        assert_raises(ValueError, rng.triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, rng.triangular, left, bad_mode_one * 3, right)
        assert_raises(
            ValueError, rng.triangular, bad_left_two, bad_mode_two * 3, right
        )

        assert_equal(rng.triangular(left, mode, right * 3).shape, desired_shape)
        assert_raises(ValueError, rng.triangular, bad_left_one, mode, right * 3)
        assert_raises(ValueError, rng.triangular, left, bad_mode_one, right * 3)
        assert_raises(
            ValueError, rng.triangular, bad_left_two, bad_mode_two, right * 3
        )

    def test_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.binomial(n * 3, p).shape, desired_shape)
        assert_raises(ValueError, rng.binomial, bad_n * 3, p)
        assert_raises(ValueError, rng.binomial, n * 3, bad_p_one)
        assert_raises(ValueError, rng.binomial, n * 3, bad_p_two)

        assert_equal(rng.binomial(n, p * 3).shape, desired_shape)
        assert_raises(ValueError, rng.binomial, bad_n, p * 3)
        assert_raises(ValueError, rng.binomial, n, bad_p_one * 3)
        assert_raises(ValueError, rng.binomial, n, bad_p_two * 3)

    def test_negative_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.negative_binomial(n * 3, p).shape, desired_shape)
        assert_raises(ValueError, rng.negative_binomial, bad_n * 3, p)
        assert_raises(ValueError, rng.negative_binomial, n * 3, bad_p_one)
        assert_raises(ValueError, rng.negative_binomial, n * 3, bad_p_two)

        assert_equal(rng.negative_binomial(n, p * 3).shape, desired_shape)
        assert_raises(ValueError, rng.negative_binomial, bad_n, p * 3)
        assert_raises(ValueError, rng.negative_binomial, n, bad_p_one * 3)
        assert_raises(ValueError, rng.negative_binomial, n, bad_p_two * 3)

    def test_poisson(self):
        max_lam = mkl_random.RandomState()._poisson_lam_max

        lam = [1]
        bad_lam_one = [-1]
        bad_lam_two = [max_lam * 2]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.poisson(lam * 3).shape, desired_shape)
        assert_raises(ValueError, rng.poisson, bad_lam_one * 3)
        assert_raises(ValueError, rng.poisson, bad_lam_two * 3)

    def test_zipf(self):
        a = [2]
        bad_a = [0]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.zipf(a * 3).shape, desired_shape)
        assert_raises(ValueError, rng.zipf, bad_a * 3)
        with np.errstate(invalid="ignore"):
            assert_raises(ValueError, rng.zipf, np.nan)
            assert_raises(ValueError, rng.zipf, [0, 0, np.nan])

    def test_geometric(self):
        p = [0.5]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.geometric(p * 3).shape, desired_shape)
        assert_raises(ValueError, rng.geometric, bad_p_one * 3)
        assert_raises(ValueError, rng.geometric, bad_p_two * 3)

    def test_hypergeometric(self):
        ngood = [1]
        nbad = [2]
        nsample = [2]
        bad_ngood = [-1]
        bad_nbad = [-2]
        bad_nsample_one = [0]
        bad_nsample_two = [4]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(
            rng.hypergeometric(ngood * 3, nbad, nsample).shape, desired_shape
        )
        assert_raises(
            ValueError, rng.hypergeometric, bad_ngood * 3, nbad, nsample
        )
        assert_raises(
            ValueError, rng.hypergeometric, ngood * 3, bad_nbad, nsample
        )
        assert_raises(
            ValueError, rng.hypergeometric, ngood * 3, nbad, bad_nsample_one
        )
        assert_raises(
            ValueError, rng.hypergeometric, ngood * 3, nbad, bad_nsample_two
        )

        assert_equal(
            rng.hypergeometric(ngood, nbad * 3, nsample).shape, desired_shape
        )
        assert_raises(
            ValueError, rng.hypergeometric, bad_ngood, nbad * 3, nsample
        )
        assert_raises(
            ValueError, rng.hypergeometric, ngood, bad_nbad * 3, nsample
        )
        assert_raises(
            ValueError, rng.hypergeometric, ngood, nbad * 3, bad_nsample_one
        )
        assert_raises(
            ValueError, rng.hypergeometric, ngood, nbad * 3, bad_nsample_two
        )

        assert_equal(
            rng.hypergeometric(ngood, nbad, nsample * 3).shape, desired_shape
        )
        assert_raises(
            ValueError, rng.hypergeometric, bad_ngood, nbad, nsample * 3
        )
        assert_raises(
            ValueError, rng.hypergeometric, ngood, bad_nbad, nsample * 3
        )
        assert_raises(
            ValueError, rng.hypergeometric, ngood, nbad, bad_nsample_one * 3
        )
        assert_raises(
            ValueError, rng.hypergeometric, ngood, nbad, bad_nsample_two * 3
        )

    def test_logseries(self):
        p = [0.5]
        bad_p_one = [2]
        bad_p_two = [-1]
        desired_shape = (3,)

        rng = mkl_random.RandomState()
        assert_equal(rng.logseries(p * 3).shape, desired_shape)
        assert_raises(ValueError, rng.logseries, bad_p_one * 3)
        assert_raises(ValueError, rng.logseries, bad_p_two * 3)


# See Issue #4263
class TestSingleEltArrayInput:
    def _create_arrays(self):
        return np.array([2]), np.array([3]), np.array([4]), (1,)

    def test_one_arg_funcs(self):
        argOne, _, _, tgtShape = self._create_arrays()
        funcs = (
            mkl_random.exponential,
            mkl_random.standard_gamma,
            mkl_random.chisquare,
            mkl_random.standard_t,
            mkl_random.pareto,
            mkl_random.weibull,
            mkl_random.power,
            mkl_random.rayleigh,
            mkl_random.poisson,
            mkl_random.zipf,
            mkl_random.geometric,
            mkl_random.logseries,
        )

        probfuncs = (mkl_random.geometric, mkl_random.logseries)

        for func in funcs:
            if func in probfuncs:  # p < 1.0
                out = func(np.array([0.5]))

            else:
                out = func(argOne)

            assert_equal(out.shape, tgtShape)

    def test_two_arg_funcs(self):
        argOne, argTwo, _, tgtShape = self._create_arrays()
        funcs = (
            mkl_random.uniform,
            mkl_random.normal,
            mkl_random.beta,
            mkl_random.gamma,
            mkl_random.f,
            mkl_random.noncentral_chisquare,
            mkl_random.vonmises,
            mkl_random.laplace,
            mkl_random.gumbel,
            mkl_random.logistic,
            mkl_random.lognormal,
            mkl_random.wald,
            mkl_random.binomial,
            mkl_random.negative_binomial,
        )

        probfuncs = (mkl_random.binomial, mkl_random.negative_binomial)

        for func in funcs:
            if func in probfuncs:  # p <= 1
                argTwo = np.array([0.5])

            else:
                argTwo = argTwo

            out = func(argOne, argTwo)
            assert_equal(out.shape, tgtShape)

            out = func(argOne[0], argTwo)
            assert_equal(out.shape, tgtShape)

            out = func(argOne, argTwo[0])
            assert_equal(out.shape, tgtShape)

    # TODO: fix randint to handle single arrays correctly, remove skip
    @pytest.mark.skip("randint does not work with arrays")
    def test_randint(self):
        _, _, _, tgtShape = self._create_arrays()
        itype = [
            bool,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
        ]
        func = mkl_random.randint
        high = np.array([1])
        low = np.array([0])

        for dt in itype:
            out = func(low, high, dtype=dt)
            assert_equal(out.shape, tgtShape)

            out = func(low[0], high, dtype=dt)
            assert_equal(out.shape, tgtShape)

            out = func(low, high[0], dtype=dt)
            assert_equal(out.shape, tgtShape)

    def test_three_arg_funcs(self):
        argOne, argTwo, argThree, tgtShape = self._create_arrays()
        funcs = [
            mkl_random.noncentral_f,
            mkl_random.triangular,
            mkl_random.hypergeometric,
        ]

        for func in funcs:
            out = func(argOne, argTwo, argThree)
            assert_equal(out.shape, tgtShape)

            out = func(argOne[0], argTwo, argThree)
            assert_equal(out.shape, tgtShape)

            out = func(argOne, argTwo[0], argThree)
            assert_equal(out.shape, tgtShape)
