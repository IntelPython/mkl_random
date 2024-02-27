:mod:`mkl_random` APIs
======================

The class :doc:`mkl_random.RandomState <./api>` exposes sampling from probability distributions while supporting
different streams of randomness, also known as basic random number generators.

The basic random number generator is chosen by specifying :code:`brng` keyword argument to the constructor of :code:`mkl.RandomState` class.

The list of supported basic random number generators is as follows (also see `oneMKL Engines <oneMKLBRNG_>`_):

* :ref:`'MT19937' <mt19937_brng>` - the Mersenne Twister pseudo-random number generator (default)
* :ref:`'SFMT19937' <sfmt19937_brng>` - the SIMD-oriented Mersenne Twister pseudo-random number generator
* :ref:`'MT2203' <mt2203_brng>` -  the set of 6024 Mersenne Twister pseudo-random number generators
* :ref:`'R250' <r250_brng>` - the 32-bit generalized feedback shift register pseudo-random number generator GFSR(250,103)
* :ref:`'WH' <wh_brng>` -  the set of 273 Wichmann-Hill’s combined multiplicative congruential pseudo-random number generators
* :ref:`'MCG31' <mcg31m1_brng>` - the 31-bit multiplicative congruential pseudo-random number generator
* :ref:`'MCG59' <mcg59_brng>` - the 59-bit multiplicative congruential pseudo-random number generator
* :ref:`'MRG32K3A' <mrg32k3a_brng>` - the combined multiple recursive pseudo-random number generator MRG32k3a
* :ref:`'PHILOX4X32X10' <philox4x32x10_brng>` - the Philox4x32x10 counter-based pseudo-random number generator
* :ref:`'NONDETERM' <nondeterm_brng>` - the generator with non-deterministic source of randomness (for example, a hardware device)
* :ref:`'ARS5' <ars5_brng>` - the ARS5 counter-based pseudo-random number generator based on AES encryption algorithm

.. _oneMKLBRNG: https://spec.oneapi.io/versions/1.0-rev-2/elements/oneMKL/source/domains/rng/engines-basic-random-number-generators.html

.. toctree::
    :hidden:

    api
    mt19937
    sfmt19937
    r250
    mt2203
    wichmann_hill
    mcg31
    mcg59
    mrg32k3a
    philox4x32x10
    nondeterministic
    ars5
