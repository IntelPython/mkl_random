.. _philox4x32x10_brng:

Philox4x32x10 brng
==================

The Philox 4x32x10 counter-based pseudo-random number generator can be 
initialized with  either an integral seed, a list of integral seeds, or automatically.

.. code-block:: python
    :caption: Construction for Philox4x32x10 basic pseudo-random number generator with scalar seed

        import mkl_random
        rs = mkl_random.RandomState(1234, brng="philox4x32x10")

        # Use random state instance to generate 1000 random numbers from
        # Uniform(0, 1) distribution
        esample = rs.uniform(0, 1, size=1000)

.. code-block:: python
    :caption: Construction for Philox4x32x10 basic pseudo-random number generator with vector seed

        import mkl_random
        rs_vec = mkl_random.RandomState([1234, 567, 89, 0], brng="philox4x32x10")

        # Use random state instance to generate 1000 random numbers from
        # Gamma(3, 1) distibution
        gsample = rs_vec.gamma(3, 1, size=1000)

When seed is not specified, the generator is initialized using system clock, e.g.:

.. code-block:: python
    :caption: Construction for Philox4x32x10 basic pseudo random number generator with automatic seed

        import mkl_random
        rs_def = mkl_random.RandomState(brng="philox4x32x10")

        # Use random state instance to generate 1000 random numbers
        # from discrete uniform distribution [1, 6]
        isample = rs_def.randint(1, 6 + 1, size=1000)
