Wichmann-Hill brng
==================

Each generator from the set of 273 Wichmann-Hill’s combined multiplicative congruential 
generators can be initialized with  either an integral seed, a list of integral seeds, 
or automatically.

An individual member of the set can be addressed by using a tuple to specify the generator as
:code:`brng=("WH", set_id)` where :math:`0 \leq \text{set_id} \le 273`.

.. code-block:: python
    :caption: Construction for WH basic random number generator with scalar seed

        import mkl_random
        seed = 777
        # initialize representative generator from the set
        rs0 = mkl_random.RandomState(seed, brng="WH")

        # initialize 0-th member of the set
        rs0 = mkl_random.RandomState(seed, brng=("WH", 0))

        # initialize 5-th member of the set
        rs5 = mkl_random.RandomState(seed, brng=("WH", 5))

        sample = rs5.uniform(0, 1, size=1_000_000)

.. code-block:: python
    :caption: Construction for WH basic random number generator with vector seed

        import mkl_random
        rs = mkl_random.RandomState([1234, 567, 89, 0], brng=("WH", 200))

        # Use random state instance to generate 1000 random numbers from
        # Gamma(3, 1) distibution
        gsample = rs_vec.gamma(3, 1, size=1000)

When seed is not specified, the generator is initialized using system clock, e.g.:

.. code-block:: python
    :caption: Construction for WH basic random number generator automatic seed

        import mkl_random
        rs_def = mkl_random.RandomState(brng="WH")

        # Use random state instance to generate 1000 random numbers
        # from discrete uniform distribution [1, 6]
        isample = rs_def.randint(1, 6 + 1, size=1000)