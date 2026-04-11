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

"""
An interface for the legacy RandomState interface of the NumPy random module
(`numpy.random`) that uses OneMKL RNG in the backend.
"""

import mkl_random


class RandomState(
    mkl_random.mklrand._MKLRandomState
):  # pylint: disable=maybe-no-member
    """
    RandomState(seed=None)

    Container for the Intel(R) MKL-powered Mersenne Twister pseudo-random
    number generator.

    For full documentation refer to `numpy.random.RandomState`.

    Notes
    -----
    While this class shares its API with the original `RandomState`, it has
    been rewritten to use MKL's vector statistics functionality, that
    provides efficient implementation of the MT19937.
    As a consequence, this version is NOT seed-compatible with the original
    `RandomState` and the result of `get_state` is NOT compatible with the
    original `RandomState`

    References
    -----
    MKL Documentation: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html  # noqa: E501,W505

    """

    def __init__(self, seed=None):
        super().__init__(seed=seed, brng="MT19937")

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

        For full documentation refer to `numpy.random.seed`.

        """
        return super().seed(seed=seed)

    def get_state(self, legacy=True):
        """
        get_state(legacy=True)

        Get the internal state of the generator.

        Parameters
        ----------
        legacy : bool, optional
            Flag indicating to return a legacy tuple state.

        Returns
        -------
        out : {tuple(str, bytes), dict}
            The returned tuple has the following items:

            1. a string specifying the basic pseudo-random number generation
               algorithm. It should always be `MT19937` for this class.
            2. a bytes object holding content of Intel MKL's stream for the
               generator.

            If `legacy` is False, a dictionary containing the state information
            is returned instead, with the following keys:

            1. `bit_generator`: a string specifying the basic pseudo-random
                number generation algorithm. It should always be `MT19937`
                for this class.
            2. `state`: a dictionary guaranteed to contain the key
                `mkl_stream`, whose value is a bytes object holding content of
                Intel MKL's stream for the generator.

            Compare with `numpy.random.get_state`.

            Notes
            -----
            As this class uses MKL in the backend, the state format is NOT
            compatible with the original `numpy.random.set_state`. The returned
            state represents the MKL stream state as a bytes object, which
            CANNOT be interpreted by NumPy's `RandomState`.
            The `legacy` argument is included for compatibility with the
            original `RandomState`.
        """
        return super().get_state(legacy=legacy)

    def set_state(self, state):
        """
        set_state(state)

        Set the internal state of the generator.

        For full documentation refer to `numpy.random.set_state`.

        Notes
        -----
        As this class uses MKL in the backend, the state of the generator
        is NOT deterministic with states returned from the original
        `numpy.random.get_state`.

        """
        return super().set_state(state=state)

    # pickling support
    def __reduce__(self):
        return (__NPRandomState_ctor, (), self.get_state())

    def random_sample(self, size=None):
        """
        random_sample(size=None)

        Return random floats in the half-open interval [0.0, 1.0).

        For full documentation refer to `numpy.random.random_sample`.

        """
        return super().random_sample(size=size)

    def random(self, size=None):
        """
        random(size=None)

        Alias for `random_sample`.

        For full documentation refer to `numpy.random.random_sample`.

        """
        return super().random_sample(size=size)

    def randint(self, low, high=None, size=None, dtype=int):
        """
        randint(low, high=None, size=None, dtype=int)

        Return random integers from `low` (inclusive) to `high` (exclusive).

        For full documentation refer to `numpy.random.randint`.

        """
        return super().randint(low=low, high=high, size=size, dtype=dtype)

    def bytes(self, length):
        """
        bytes(length)

        Return random bytes.

        For full documentation refer to `numpy.random.bytes`.

        """
        return super().bytes(length=length)

    def choice(self, a, size=None, replace=True, p=None):
        """
        choice(a, size=None, replace=True, p=None)

        Generates a random sample from a given 1-D array.

        For full documentation refer to `numpy.random.choice`.

        """
        return super().choice(a=a, size=size, replace=replace, p=p)

    def uniform(self, low=0.0, high=1.0, size=None):
        """
        uniform(low=0.0, high=1.0, size=None)

        Draw samples from a uniform distribution.

        For full documentation refer to `numpy.random.uniform`.

        """
        return super().uniform(low=low, high=high, size=size)

    def rand(self, *args):
        """
        rand(d0, d1, ..., dn)

        Random values in a given shape.

        For full documentation refer to `numpy.random.rand`.

        """
        return super().rand(*args)

    def randn(self, *args):
        """
        randn(d0, d1, ..., dn)

        Return a sample (or samples) from the "standard normal" distribution.

        For full documentation refer to `numpy.random.randn`.

        """
        return super().randn(*args)

    def random_integers(self, low, high=None, size=None):
        """
        random_integers(low, high=None, size=None)

        Return random integers from `low` (inclusive) to `high` (inclusive).

        For full documentation refer to `numpy.random.random_integers`.

        """
        return super().random_integers(low=low, high=high, size=size)

    def standard_normal(self, size=None):
        """
        standard_normal(size=None)

        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        For full documentation refer to `numpy.random.standard_normal`.

        """
        return super().standard_normal(size=size, method="ICDF")

    def normal(self, loc=0.0, scale=1.0, size=None):
        """
        normal(loc=0.0, scale=1.0, size=None, method='ICDF')

        Draw random samples from a normal (Gaussian) distribution.

        For full documentation refer to `numpy.random.normal`.

        """
        return super().normal(loc=loc, scale=scale, size=size, method="ICDF")

    def beta(self, a, b, size=None):
        """
        beta(a, b, size=None)

        Draw random samples from a Beta distribution.

        For full documentation refer to `numpy.random.beta`.

        """
        return super().beta(
            a=a,
            b=b,
            size=size,
        )

    def exponential(self, scale=1.0, size=None):
        """
        exponential(scale=1.0, size=None)

        Draw samples from an exponential distribution.

        For full documentation refer to `numpy.random.exponential`.

        """
        return super().exponential(scale=scale, size=size)

    def tomaxint(self, size=None):
        """
        tomaxint(size=None)

        Return a sample of uniformly distributed random integers in the
        interval [0, ``np.iinfo("long").max``].

        For full documentation refer to `numpy.random.RandomState.tomaxint`.

        """
        return super().tomaxint(size=size)

    def standard_exponential(self, size=None):
        """
        standard_exponential(size=None)

        Draw samples from the standard exponential distribution.

        For full documentation refer to `numpy.random.standard_exponential`.

        """
        return super().standard_exponential(size=size)

    def standard_gamma(self, shape, size=None):
        """
        standard_gamma(shape, size=None)

        Draw samples from the standard gamma distribution.

        For full documentation refer to `numpy.random.standard_gamma`.

        """
        return super().standard_gamma(shape=shape, size=size)

    def gamma(self, shape, scale=1.0, size=None):
        """
        gamma(shape, scale=1.0, size=None)

        Draw samples from a gamma distribution.

        For full documentation refer to `numpy.random.gamma`.

        """
        return super().gamma(shape=shape, scale=scale, size=size)

    def f(self, dfnum, dfden, size=None):
        """
        f(dfnum, dfden, size=None)

        Draw samples from an F distribution.

        For full documentation refer to `numpy.random.f`.

        """
        return super().f(dfnum=dfnum, dfden=dfden, size=size)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        """
        noncentral_f(dfnum, dfden, nonc, size=None)

        Draw samples from a non-central F distribution.

        For full documentation refer to `numpy.random.noncentral_f`.

        """
        return super().noncentral_f(
            dfnum=dfnum, dfden=dfden, nonc=nonc, size=size
        )

    def chisquare(self, df, size=None):
        """
        chisquare(df, size=None)

        Draw samples from a chi-square distribution.

        For full documentation refer to `numpy.random.chisquare`.

        """
        return super().chisquare(df=df, size=size)

    def noncentral_chisquare(self, df, nonc, size=None):
        """
        noncentral_chisquare(df, nonc, size=None)

        Draw samples from a non-central chi-square distribution.

        For full documentation refer to `numpy.random.noncentral_chisquare`.

        """
        return super().noncentral_chisquare(df=df, nonc=nonc, size=size)

    def standard_cauchy(self, size=None):
        """
        standard_cauchy(size=None)

        Draw samples from a standard Cauchy distribution.

        For full documentation refer to `numpy.random.standard_cauchy`.

        """
        return super().standard_cauchy(size=size)

    def standard_t(self, df, size=None):
        """
        standard_t(df, size=None)

        Draw samples from a standard Student's t distribution.

        For full documentation refer to `numpy.random.standard_t`.

        """
        return super().standard_t(df=df, size=size)

    def vonmises(self, mu, kappa, size=None):
        """
        vonmises(mu, kappa, size=None)

        Draw samples from a von Mises distribution.

        For full documentation refer to `numpy.random.vonmises`.

        """
        return super().vonmises(mu=mu, kappa=kappa, size=size)

    def pareto(self, a, size=None):
        """
        pareto(a, size=None)

        Draw samples from a Pareto II or Lomax distribution with a scale
        parameter of 1.

        For full documentation refer to `numpy.random.pareto`.

        """
        return super().pareto(a=a, size=size)

    def weibull(self, a, size=None):
        """
        weibull(a, size=None)

        Draw samples from a Weibull distribution.

        For full documentation refer to `numpy.random.weibull`.

        """
        return super().weibull(a=a, size=size)

    def power(self, a, size=None):
        """
        power(a, size=None)

        Draw samples from a power distribution.

        For full documentation refer to `numpy.random.power`.

        """
        return super().power(a=a, size=size)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        """
        laplace(loc=0.0, scale=1.0, size=None)

        Draw samples from the Laplace distribution.

        For full documentation refer to `numpy.random.laplace`.

        """
        return super().laplace(loc=loc, scale=scale, size=size)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        """
        gumbel(loc=0.0, scale=1.0, size=None)

        Draw samples from a Gumbel distribution.

        For full documentation refer to `numpy.random.gumbel`.

        """
        return super().gumbel(loc=loc, scale=scale, size=size)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        """
        logistic(loc=0.0, scale=1.0, size=None)

        Draw samples from a logistic distribution.

        For full documentation refer to `numpy.random.logistic`.

        """
        return super().logistic(loc=loc, scale=scale, size=size)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        """
        lognormal(mean=0.0, sigma=1.0, size=None)

        Draw random samples from a log-normal distribution.

        For full documentation refer to `numpy.random.lognormal`.

        """
        return super().lognormal(
            mean=mean, sigma=sigma, size=size, method="ICDF"
        )

    def rayleigh(self, scale=1.0, size=None):
        """
        rayleigh(scale=1.0, size=None)

        Draw samples from a Rayleigh distribution.

        For full documentation refer to `numpy.random.rayleigh`.

        """
        return super().rayleigh(scale=scale, size=size)

    def wald(self, mean, scale, size=None):
        """
        wald(mean, scale, size=None)

        Draw samples from a Wald distribution.

        For full documentation refer to `numpy.random.wald`.

        """
        return super().wald(mean=mean, scale=scale, size=size)

    def triangular(self, left, mode, right, size=None):
        """
        triangular(left, mode, right, size=None)

        Draw samples from a triangular distribution.

        For full documentation refer to `numpy.random.triangular`.

        """
        return super().triangular(left=left, mode=mode, right=right, size=size)

    def binomial(self, n, p, size=None):
        """
        binomial(n, p, size=None)

        Draw samples from a binomial distribution.

        For full documentation refer to `numpy.random.binomial`.

        """
        return super().binomial(n=n, p=p, size=size)

    def negative_binomial(self, n, p, size=None):
        """
        negative_binomial(n, p, size=None)

        Draw samples from a negative binomial distribution.

        For full documentation refer to `numpy.random.negative_binomial`.

        """
        return super().negative_binomial(n=n, p=p, size=size)

    def poisson(self, lam=1.0, size=None):
        """
        poisson(lam=1.0, size=None)

        Draw random samples from a Poisson distribution.

        For full documentation refer to `numpy.random.poisson`.

        """
        return super().poisson(lam=lam, size=size, method="POISNORM")

    def zipf(self, a, size=None):
        """
        zipf(a, size=None)

        Draw samples from a Zipf distribution.

        For full documentation refer to `numpy.random.zipf`.

        """
        return super().zipf(a=a, size=size)

    def geometric(self, p, size=None):
        """
        geometric(p, size=None)

        Draw samples from the geometric distribution.

        For full documentation refer to `numpy.random.geometric`.

        """
        return super().geometric(p=p, size=size)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        """
        hypergeometric(ngood, nbad, nsample, size=None)

        Draw samples from a hypergeometric distribution.

        For full documentation refer to `numpy.random.hypergeometric`.

        """
        return super().hypergeometric(
            ngood=ngood, nbad=nbad, nsample=nsample, size=size
        )

    def logseries(self, p, size=None):
        """
        logseries(p, size=None)

        Draw samples from a logarithmic series distribution.

        For full documentation refer to `numpy.random.logseries`.

        """
        return super().logseries(p=p, size=size)

    def multivariate_normal(
        self, mean, cov, size=None, check_valid="warn", tol=1e-8
    ):
        """
        multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)

        Draw random samples from a multivariate normal distribution.

        For full documentation refer to `numpy.random.multivariate_normal`.

        """
        return super().multivariate_normal(
            mean=mean, cov=cov, size=size, check_valid=check_valid, tol=tol
        )

    def multinomial(self, n, pvals, size=None):
        """
        multinomial(n, pvals, size=None)

        Draw samples from a multinomial distribution.

        For full documentation refer to `numpy.random.multinomial`.

        """
        return super().multinomial(n=n, pvals=pvals, size=size)

    def dirichlet(self, alpha, size=None):
        """
        dirichlet(alpha, size=None)

        Draw samples from the Dirichlet distribution.

        For full documentation refer to `numpy.random.dirichlet`.

        """
        return super().dirichlet(alpha=alpha, size=size)

    def shuffle(self, x):
        """
        shuffle(x)

        Modify a sequence in-place by shuffling its contents.

        For full documentation refer to `numpy.random.shuffle`.

        """
        return super().shuffle(x=x)

    def permutation(self, x):
        """
        permutation(x)

        Randomly permute a sequence, or return a permuted range.

        For full documentation refer to `numpy.random.permutation`.

        """
        return super().permutation(x=x)


def __NPRandomState_ctor():
    """
    Return a RandomState instance.
    This function exists solely to assist (un)pickling.
    Note that the state of the RandomState returned here is irrelevant, as this
    function's entire purpose is to return a newly allocated RandomState whose
    state pickle can set. Consequently the RandomState returned by this
    function is a freshly allocated copy with a seed=0.
    See https://github.com/numpy/numpy/issues/4763 for a detailed discussion
    """
    return RandomState(seed=0)


# instantiate a default RandomState object to be used by module-level functions
_rand = RandomState()


def sample(*args, **kwargs):
    """
    Alias of `random_sample`.

    For full documentation refer to `numpy.random.random_sample`.
    """
    return _rand.random_sample(*args, **kwargs)


def ranf(*args, **kwargs):
    """
    Alias of `random_sample`.

    For full documentation refer to `numpy.random.random_sample`.
    """
    return _rand.random_sample(*args, **kwargs)


# define module-level functions using methods of a default RandomState object
seed = _rand.seed
get_state = _rand.get_state
set_state = _rand.set_state
random_sample = _rand.random_sample
random = _rand.random
choice = _rand.choice
randint = _rand.randint
bytes = _rand.bytes
uniform = _rand.uniform
rand = _rand.rand
randn = _rand.randn
random_integers = _rand.random_integers
standard_normal = _rand.standard_normal
normal = _rand.normal
beta = _rand.beta
exponential = _rand.exponential
standard_exponential = _rand.standard_exponential
standard_gamma = _rand.standard_gamma
gamma = _rand.gamma
f = _rand.f
noncentral_f = _rand.noncentral_f
chisquare = _rand.chisquare
noncentral_chisquare = _rand.noncentral_chisquare
standard_cauchy = _rand.standard_cauchy
standard_t = _rand.standard_t
vonmises = _rand.vonmises
pareto = _rand.pareto
weibull = _rand.weibull
power = _rand.power
laplace = _rand.laplace
gumbel = _rand.gumbel
logistic = _rand.logistic
lognormal = _rand.lognormal
rayleigh = _rand.rayleigh
wald = _rand.wald
triangular = _rand.triangular

binomial = _rand.binomial
negative_binomial = _rand.negative_binomial
poisson = _rand.poisson
zipf = _rand.zipf
geometric = _rand.geometric
hypergeometric = _rand.hypergeometric
logseries = _rand.logseries

multivariate_normal = _rand.multivariate_normal
multinomial = _rand.multinomial
dirichlet = _rand.dirichlet

shuffle = _rand.shuffle
permutation = _rand.permutation
