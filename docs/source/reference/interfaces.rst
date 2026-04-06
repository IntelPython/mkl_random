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

.. autofunction:: mkl_random.interfaces.numpy_random.seed
.. autofunction:: mkl_random.interfaces.numpy_random.get_state
.. autofunction:: mkl_random.interfaces.numpy_random.set_state

**Simple random data:**

Similar to NumPy, the methods of :class:`RandomState` are exported as functions in the module.
Their usage is discouraged, as they are implemented from a global instance of :class:`RandomState`,
which means results may change across calls.

.. autofunction:: mkl_random.interfaces.numpy_random.rand
.. autofunction:: mkl_random.interfaces.numpy_random.randn
.. autofunction:: mkl_random.interfaces.numpy_random.randint
.. autofunction:: mkl_random.interfaces.numpy_random.random_integers
.. autofunction:: mkl_random.interfaces.numpy_random.random_sample
.. autofunction:: mkl_random.interfaces.numpy_random.random
.. autofunction:: mkl_random.interfaces.numpy_random.ranf
.. autofunction:: mkl_random.interfaces.numpy_random.choice
.. autofunction:: mkl_random.interfaces.numpy_random.bytes
.. autofunction:: mkl_random.interfaces.numpy_random.sample

**Permutations:**

.. autofunction:: mkl_random.interfaces.numpy_random.shuffle
.. autofunction:: mkl_random.interfaces.numpy_random.permutation

**Distributions:**

.. autofunction:: mkl_random.interfaces.numpy_random.beta
.. autofunction:: mkl_random.interfaces.numpy_random.binomial
.. autofunction:: mkl_random.interfaces.numpy_random.chisquare
.. autofunction:: mkl_random.interfaces.numpy_random.dirichlet
.. autofunction:: mkl_random.interfaces.numpy_random.exponential
.. autofunction:: mkl_random.interfaces.numpy_random.f
.. autofunction:: mkl_random.interfaces.numpy_random.gamma
.. autofunction:: mkl_random.interfaces.numpy_random.geometric
.. autofunction:: mkl_random.interfaces.numpy_random.gumbel
.. autofunction:: mkl_random.interfaces.numpy_random.hypergeometric
.. autofunction:: mkl_random.interfaces.numpy_random.laplace
.. autofunction:: mkl_random.interfaces.numpy_random.logistic
.. autofunction:: mkl_random.interfaces.numpy_random.lognormal
.. autofunction:: mkl_random.interfaces.numpy_random.logseries
.. autofunction:: mkl_random.interfaces.numpy_random.multinomial
.. autofunction:: mkl_random.interfaces.numpy_random.multivariate_normal
.. autofunction:: mkl_random.interfaces.numpy_random.negative_binomial
.. autofunction:: mkl_random.interfaces.numpy_random.noncentral_chisquare
.. autofunction:: mkl_random.interfaces.numpy_random.noncentral_f
.. autofunction:: mkl_random.interfaces.numpy_random.normal
.. autofunction:: mkl_random.interfaces.numpy_random.pareto
.. autofunction:: mkl_random.interfaces.numpy_random.poisson
.. autofunction:: mkl_random.interfaces.numpy_random.power
.. autofunction:: mkl_random.interfaces.numpy_random.rayleigh
.. autofunction:: mkl_random.interfaces.numpy_random.standard_cauchy
.. autofunction:: mkl_random.interfaces.numpy_random.standard_exponential
.. autofunction:: mkl_random.interfaces.numpy_random.standard_gamma
.. autofunction:: mkl_random.interfaces.numpy_random.standard_normal
.. autofunction:: mkl_random.interfaces.numpy_random.standard_t
.. autofunction:: mkl_random.interfaces.numpy_random.triangular
.. autofunction:: mkl_random.interfaces.numpy_random.uniform
.. autofunction:: mkl_random.interfaces.numpy_random.vonmises
.. autofunction:: mkl_random.interfaces.numpy_random.wald
.. autofunction:: mkl_random.interfaces.numpy_random.weibull
.. autofunction:: mkl_random.interfaces.numpy_random.zipf
