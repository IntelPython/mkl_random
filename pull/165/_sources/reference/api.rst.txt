.. _fullapi:

MKL-powered RandomState
=======================

.. currentmodule:: mkl_random

.. autoclass:: MKLRandomState

.. note::

   ``mkl_random.RandomState`` is a deprecated alias of
   :class:`mkl_random.MKLRandomState`, kept for backward compatibility with
   legacy code.  A NumPy-compatible drop-in replacement for
   :class:`numpy.random.RandomState` is available in
   :mod:`mkl_random.interfaces.numpy_random`.

Seeding and state
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~MKLRandomState.seed
   ~MKLRandomState.get_state
   ~MKLRandomState.set_state

Simple random data
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~MKLRandomState.rand
   ~MKLRandomState.randn
   ~MKLRandomState.randint
   ~MKLRandomState.random_integers
   ~MKLRandomState.random_sample
   ~MKLRandomState.tomaxint
   ~MKLRandomState.choice
   ~MKLRandomState.bytes

Permutations
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~MKLRandomState.shuffle
   ~MKLRandomState.permutation

Distributions
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~MKLRandomState.beta
   ~MKLRandomState.binomial
   ~MKLRandomState.chisquare
   ~MKLRandomState.dirichlet
   ~MKLRandomState.exponential
   ~MKLRandomState.f
   ~MKLRandomState.gamma
   ~MKLRandomState.geometric
   ~MKLRandomState.gumbel
   ~MKLRandomState.hypergeometric
   ~MKLRandomState.laplace
   ~MKLRandomState.logistic
   ~MKLRandomState.lognormal
   ~MKLRandomState.logseries
   ~MKLRandomState.multinomial
   ~MKLRandomState.multivariate_normal
   ~MKLRandomState.negative_binomial
   ~MKLRandomState.noncentral_chisquare
   ~MKLRandomState.noncentral_f
   ~MKLRandomState.normal
   ~MKLRandomState.pareto
   ~MKLRandomState.poisson
   ~MKLRandomState.power
   ~MKLRandomState.rayleigh
   ~MKLRandomState.standard_cauchy
   ~MKLRandomState.standard_exponential
   ~MKLRandomState.standard_gamma
   ~MKLRandomState.standard_normal
   ~MKLRandomState.standard_t
   ~MKLRandomState.triangular
   ~MKLRandomState.uniform
   ~MKLRandomState.vonmises
   ~MKLRandomState.wald
   ~MKLRandomState.weibull
   ~MKLRandomState.zipf

MKL-specific methods
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~MKLRandomState.leapfrog
   ~MKLRandomState.skipahead
   ~MKLRandomState.randint_untyped
   ~MKLRandomState.multinormal_cholesky
