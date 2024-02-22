:mod:`mkl_random` APIs
======================

The class :doc:`mkl_random.RandomState <./api>` exposes sampling from probability distributions while supporting
different streams of randomness, also known as basic random number generators.

The basic random number generator is chosen by specifying :code:`brng` keyword argument to the constructor of :code:`mkl.RandomState` class.

The list of supported basic random number generators is as follows (also see `oneMKL Engines <oneMKLBRNG_>`_):

* :code:`'MT19937'` - the Mersenne Twister pseudo-random number generator (default), :doc:`example <mt19937>`
* :code:`'SFMT19937'` - the SIMD-oriented Mersenne Twister pseudo-random number generator, :doc:`example <sfmt19937>`
* :code:`'MT2203'` -  the set of 6024 Mersenne Twister pseudorandom number generators, :doc:`example <mt2203>`
* :code:`'R250'` - the 32-bit generalized feedback shift register pseudorandom number generator GFSR(250,103), :doc:`example <r250>`
* :code:`'WH'` -  the set of 273 Wichmann-Hill’s combined multiplicative congruential generators, :doc:`example <wichmann_hill>`
* :code:`'MCG31'` - the 31-bit multiplicative congruential pseudorandom number generator, :doc:`example <mcg31>`
* :code:`'MCG59'` - the 59-bit multiplicative congruential pseudorandom number generator, :doc:`example <mcg59>`
* :code:`'MRG32K3A'` - the combined multiple recursive pseudorandom number generator MRG32k3a, :doc:`example <mrg32k3a>`
* :code:`'PHILOX4X32X10'` - the Philox4x32x10 counter-based pseudorandom number generator, :doc:`example <philox4x32x10>`
* :code:`'NONDETERM'` - the generator with non-deterministic source of randomness (for example, a hardware device), :doc:`example <nondeterministic>`
* :code:`'ARS5'` - the ARS5 counter-based pseudorandom number generator based on AES encryption algorithm, :doc:`example <ars5>`

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
