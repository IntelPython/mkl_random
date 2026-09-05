How-to Guides
=============

How to save and resume long computation
---------------------------------------

:class:`MKLRandomState` is pickleable. Pickling allows to save and restore
the internal state of the pseudo-random number generators.

.. code-block:: python
    :caption: Saving state of pseudo-random basic random number generators

        import numpy as np
        import mkl_random
        import pickle

        rs = mkl_random.MKLRandomState(seed=777, brng="r250")
        draw = rs.standard_normal(size=1357913)

        # pickle random state
        saved = pickle.dumps(rs)

        # draw some numbers as if computation were to continue
        post_draw = rs.gamma(5, 1, size=100)

        # restore random state, and continue from
        restored_rs = pickle.loads(saved)
        resumed_draw = restored_rs.gamma(5, 1, size=100)

        # sample from restored stated is the same as sample
        # from the original one
        assert np.array_equal(restored_rs, resumed_draw)


Stochastic computations in parallel with multiprocessing
--------------------------------------------------------

When performing stochastic computations in parallel, care is due to ensure
statistical independence of samples drawn in parallel.

Basic quasi-random number generators provide different means to accomplishing
this. Some support :meth:`skipahead` method or :meth:`leapfrog` method, while
others provide a fixed-size family of generators with nice property that generators
from such family, initialized equally, produce streams of randomness statistically
indistinguishable from independent.

.. py:method:: skipahead(nskips)
    :canonical: mkl_random.MKLRandomState.skipahead

    Advance the state of the generator using skip-ahead method, or raise :code:`ValueError`
    exception if not supported.

    The argument `nskips` must be a positive Python integer.

    The method is supported for :ref:`"philox4x32x10" <philox4x32x10_brng>`, :ref:`"mrg32k3a" <mrg32k3a_brng>`,
    :ref:`"mcg31m1" <mcg31m1_brng>`, :ref:`"mcg59" <mcg59_brng>`, :ref:`"wh" <wh_brng>`,
    :ref:`"mt19937" <mt19937_brng>`, :ref:`"sfmt19937" <sfmt19937_brng>`, and :ref:`"ars5" <ars5_brng>`
    basic random number generators.

.. note::
    When using :meth:`skipahead`, it is important to ensure that a parallel task does not consume more than
    :code:`nskips` states, otherwise streams of randomness begin to overlap and the assumption of statistical
    independence breaks down.

.. py:method:: leapfrog(k, nstreams)
    :canonical: mkl_random.MKLRandomState.leapfrog

    Initialize the state of the generator using leap-frog method, or raise :code:`ValueError`
    exception if not supported.

    The leap-frog method partitions state trajectory into :code:`nstream` interleaved non-overlapping
    sub-sequences, and argument :code:`k` identifies the subsequence.

    The method is supported for :ref:`"mcg31m1" <mcg31m1_brng>`, :ref:`"mcg59" <mcg59_brng>`, and :ref:`"wh" <wh_brng>`
    basic pseudo-random number generators.

.. note::
    When using :meth:`leapfrog` or :meth:`skipahead` methods one must remember that parallel tasks partition
    generators period and choose a generator with sufficiently long period to avoid cycling over the period
    more than once, as doing so also breaks the assumption of statistical independence and may compromise
    correctness of the simulation.

:mod:`mkl_random` also provides two families of basic pseudo-random number generators, :ref:`"mt2203" <mt2203_brng>` and
:ref:`"wh" <wh_brng>`, with property that members from particular family, initialized equally, produce streams of
randomness stasistically indistunguishable from independent. To use such families in parallel computation, assign
difference family generators to different parallel workers and sample those assigned generators in each parallel worker.
Please refer to "examples/" folder in the `GitHub repo <https://github.com/IntelPython/mkl_random>`_ for more details.


Using :mod:`mkl_random` as a drop-in replacement for `numpy.random <https://numpy.org/doc/stable/reference/random/legacy.html>`_
-----------------------------------------------------------------

The :mod:`mkl_random.interfaces.numpy_random` module is aligned to the legacy
portion of the `numpy.random <https://numpy.org/doc/stable/reference/random/legacy.html>`_ legacy API.
You can import it in place of `numpy.random <https://numpy.org/doc/stable/reference/random/legacy.html>`_
without changing the rest of your code:

.. code-block:: python
    :caption: Drop-in replacement for numpy.random

        from mkl_random.interfaces import numpy_random as rng

        rng.seed(1234)
        x = rng.standard_normal(size=100)
        y = rng.uniform(0, 1, size=100)

See :ref:`interfaces` for a full list of available functions.

.. note::
    While the API is the same, :mod:`mkl_random.interfaces.numpy_random` is **not** seed-compatible
    with `numpy.random <https://numpy.org/doc/stable/reference/random/legacy.html>`_. Given the same seed,
    the two modules will produce different sequences. There also may be differences in some edge cases, such as
    behavior of functions when given specific inputs.


How to patch `numpy.random <https://numpy.org/doc/stable/reference/random/legacy.html>`_ with :mod:`mkl_random`
-------------------------------------------------------------

Existing code that calls `numpy.random <https://numpy.org/doc/stable/reference/random/legacy.html>`_
directly can be patched to use :mod:`mkl_random.interfaces.numpy_random` at runtime.

The recommended approach is to use the :class:`mkl_random.mkl_random` context manager:

.. code-block:: python
    :caption: Temporarily patch numpy.random using context manager

        import numpy as np
        import mkl_random

        with mkl_random.mkl_random():
            x = np.random.standard_normal(100)   # uses mkl_random
            y = np.random.uniform(0, 1, size=100) # uses mkl_random

:mod:`mkl_random` also exposes the explicit patching functions:

.. code-block:: python
    :caption: Patch numpy.random for the duration of a script

        import mkl_random
        mkl_random.patch_numpy_random() # subsequent numpy.random calls use mkl_random

        import numpy as np
        data = np.random.normal(0, 1, size=100)

.. note::
    The patching functions are provided for users' convenience, but they are not recommended
    for new code. It is instead recommended to use :mod:`mkl_random` directly for new code.
    For existing code where patching may be desirable, it is also suggested to prefer the
    context manager, as it scopes the patch to blocks and thus, prevents user error of
    forgetting to restore the original state, calling the patch multiple times, or
    creating undefined behavior when patching in a multi-threaded program.

You can also use :class:`mkl_random.mkl_random` as a decorator:

.. code-block:: python
    :caption: Patch numpy.random as a decorator

        import numpy as np
        import mkl_random

        @mkl_random.mkl_random()
        def get_data():
            return np.random.standard_normal(100)

See :ref:`patching` for details.
