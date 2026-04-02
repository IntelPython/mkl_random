.. _interfaces:

:mod:`mkl_random.interfaces`
====================================================

:mod:`mkl_random.interfaces` provides drop-in replacements for supported random number generation
modules using :mod:`mkl_random` implementations. Currently, only a NumPy interface is provided,
but more may be added in the future.


.. _numpy_random_interface:

NumPy interface --- :mod:`mkl_random.interfaces.numpy_random`
-------------------------------------------------------------

:mod:`mkl_random.interfaces.numpy_random` is a drop-in replacement for the legacy portion of
`numpy.random <https://numpy.org/doc/stable/reference/random/legacy.html>`_.

.. note::
    While the API is the same, :mod:`mkl_random.interfaces.numpy_random` is **not** seed-compatible
    with `numpy.random <https://numpy.org/doc/stable/reference/random/legacy.html>`_. Given the same seed, the two modules
    will produce different sequences. The output of :func:`get_state` and accepted input to :func:`set_state` may also differ.
    It is not recommended to provide the output of :func:`get_state` from one module to :func:`set_state` of the other.
    There also may be differences in some edge cases, such as behavior of functions when given specific inputs.


RandomState class
^^^^^^^^^^^^^^^^^

.. autoclass:: mkl_random.interfaces.numpy_random.RandomState
    :members:
    :undoc-members:


Functions
^^^^^^^^^^^^^^^^^^^^^^

**Seeding and state functions:**

.. autosummary::

    mkl_random.interfaces.numpy_random.seed
    mkl_random.interfaces.numpy_random.get_state
    mkl_random.interfaces.numpy_random.set_state

**Simple random data:**

Similar to NumPy, the methods of :class:`RandomState` are exported as functions in the module.
Their usage is discouraged, as they are implemented from a global instance of :class:`RandomState`,
which means results may change across calls.

.. autosummary::

    mkl_random.interfaces.numpy_random.rand
    mkl_random.interfaces.numpy_random.randn
    mkl_random.interfaces.numpy_random.randint
    mkl_random.interfaces.numpy_random.random_integers
    mkl_random.interfaces.numpy_random.random_sample
    mkl_random.interfaces.numpy_random.random
    mkl_random.interfaces.numpy_random.ranf
    mkl_random.interfaces.numpy_random.choice
    mkl_random.interfaces.numpy_random.bytes
    mkl_random.interfaces.numpy_random.sample

**Permutations:**

.. autosummary::

    mkl_random.interfaces.numpy_random.shuffle
    mkl_random.interfaces.numpy_random.permutation

**Distributions:**

.. autosummary::

    mkl_random.interfaces.numpy_random.beta
    mkl_random.interfaces.numpy_random.binomial
    mkl_random.interfaces.numpy_random.chisquare
    mkl_random.interfaces.numpy_random.dirichlet
    mkl_random.interfaces.numpy_random.exponential
    mkl_random.interfaces.numpy_random.f
    mkl_random.interfaces.numpy_random.gamma
    mkl_random.interfaces.numpy_random.geometric
    mkl_random.interfaces.numpy_random.gumbel
    mkl_random.interfaces.numpy_random.hypergeometric
    mkl_random.interfaces.numpy_random.laplace
    mkl_random.interfaces.numpy_random.logistic
    mkl_random.interfaces.numpy_random.lognormal
    mkl_random.interfaces.numpy_random.logseries
    mkl_random.interfaces.numpy_random.multinomial
    mkl_random.interfaces.numpy_random.multivariate_normal
    mkl_random.interfaces.numpy_random.negative_binomial
    mkl_random.interfaces.numpy_random.noncentral_chisquare
    mkl_random.interfaces.numpy_random.noncentral_f
    mkl_random.interfaces.numpy_random.normal
    mkl_random.interfaces.numpy_random.pareto
    mkl_random.interfaces.numpy_random.poisson
    mkl_random.interfaces.numpy_random.power
    mkl_random.interfaces.numpy_random.rayleigh
    mkl_random.interfaces.numpy_random.standard_cauchy
    mkl_random.interfaces.numpy_random.standard_exponential
    mkl_random.interfaces.numpy_random.standard_gamma
    mkl_random.interfaces.numpy_random.standard_normal
    mkl_random.interfaces.numpy_random.standard_t
    mkl_random.interfaces.numpy_random.triangular
    mkl_random.interfaces.numpy_random.uniform
    mkl_random.interfaces.numpy_random.vonmises
    mkl_random.interfaces.numpy_random.wald
    mkl_random.interfaces.numpy_random.weibull
    mkl_random.interfaces.numpy_random.zipf
