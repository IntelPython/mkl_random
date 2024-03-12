How-to Guides
=============

How to save and resume long computation
---------------------------------------

:class:`RandomState` is pickleable. Pickling allows to save and restore
the internal state of the pseudo-random number generators.

.. code-block:: python
    :caption: Saving state of pseudo-random basic random number generators

        import numpy as np
        import mkl_random
        import pickle

        rs = mkl_random.RandomState(seed=777, brng="r250")
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
    :canonical: mkl_random.RandomState.skipahead

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
    :canonical: mkl_random.RandomState.leapfrog

    Initialize the state of the generator using leap-frog method, or raise :code:`ValueError`
    exception if not supported.

    The leap-frog method partitions state tragectory into :code:`nstream` interleaved non-overlapping 
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