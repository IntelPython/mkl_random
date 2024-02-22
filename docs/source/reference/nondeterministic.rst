.. _nondeterm_brng:

Nondeterm brng
==============

The generator with non-deterministic source of randomness, such as a hardware device, requires no seeding.
This basic random number generator should not be used if reproducibility of stochastic simulation is required.

.. code-block:: python
    :caption: Construction for non-deterministic basic random number generator

        import mkl_random
        rs = mkl_random.RandomState(brng="nondeterm")

        # Use random state instance to generate 1000 random numbers from
        # Uniform(0, 1) distribution
        esample = rs.uniform(0, 1, size=1000)

Seed parameter provided to the constructor of :class:`mkl_random.RandomState`,
or :meth:`mkl_random.RandomState.seed` is ignored.