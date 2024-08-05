Beginner's guide
================

Installation
------------

The package :mod:`mkl_random` is available in `conda <https://docs.conda.io/en/latest/>`_ ecosystem on "conda-forge", "main", and 
"intel" `channels <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/channels.html>`_ (i.e. locations).

.. code-block:: bash
    :caption: Install mkl_random from conda-forge channel

        $ conda install -c conda-forge mkl_random

.. code-block:: bash
    :caption: Install mkl_random from intel channel

        $ conda install -c https://software.repos.intel.com/python/conda mkl_random

.. code-block:: bash
    :caption: Install mkl_random from default channel main

        $ conda install mkl_random

The package can also be installed via :code:`pip` package manager, either from central Python package index (PyPI) repository, or from index maintained by Intel(R):

.. code-block:: bash
    :caption: Install mkl_random using pip from intel channel on Anaconda

    $ pip install -i https://pypi.anaconda.org/intel/simple mkl_random

.. code-block:: bash
    :caption: Install mkl_random using pip from PyPI

    $ pip install mkl_random

The :mod:`mkl_random` is also distributed as part of `Intel(R) Distribution for Python* <https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html>`_.

First steps
-----------

The :mod:`mkl_random` package has followed the design of :class:`numpy.random` package to 
make :mod:`mkl_random` easy to use for those already familiar with the :mod:`numpy.random` module.

.. note::
    Since the first release of :mod:`mkl_random`, NumPy introduced new classes :class:`numpy.random.Generator` and 
    :class:`numpy.random.BitGenerator`, while also retaining :class:`numpy.random.RandomState` for backwards
    compatibility. :mod:`mkl_random`, at present, does not provide classes mirroring :class:`Generator` or 
    :class:`BitGenerators`.

The state of pseudo-random number generator is stored in :class:`mkl_random.RandomState` class, 
so using :mod:`mkl_random` begins with creating an instance of this class:

.. code-block:: python
    :caption: Construct random number generator

        import mkl_random
        rs = mkl_random.RandomState(seed=1234)

Sampling from difference probability distribution is done by calling the class methods on the constructed instance:

.. code-block:: python
    :caption: Generate one million variates from standard continuous uniform distribution

        s = rs.uniform(0, 1, size=1_000_000)

Drawing samples updates the state of pseudo-random number generator so that next sample is statistically 
independent from the previous one (with caveats of using pseudo-random generators implied). 

Here is an example of estimating value of :math:`\pi` by using Monte-Carlo method:

.. code-block:: python
    :caption: Using Monte-Carlo method to estimate value of pi

        import numpy as np
        import mkl_random
 
        rs = mkl_random.RandomState(seed=1234)

        sample_size = 10**8
        batch_size = 10**6
        accepted = 0
        sampled = 0
        while sampled < sample_size:
            sampled += batch_size
            x = rs.uniform(0, 1, size=batch_size)
            y = rs.uniform(0, 1, size=batch_size)
            accepted += np.sum(x*x + y*y < 1.0)
        
        print("Pi estimate: ", 4. * (accepted / sample_size))

Sample output of running such an example:

.. code-block:: bash
    :caption: Sample output after executing above script

        $ python pi.py
        Pi estimate:  3.14167732


Pseudo-random vs. non-deterministic generators
----------------------------------------------

.. _pseudorandom_vs_truerandom:

Stochastic computations often need to work with *independent* samples 
from either the same probability distribution, or a set of probability 
distributions of interest.

`True random generator <https://en.wikipedia.org/wiki/Hardware_random_number_generator>`_ relies on 
laws of physics to provide those, leveraging dedicated hardware providing a source of entropy.

`Psuedo-random generator <https://en.wikipedia.org/wiki/Pseudorandom_number_generator>`_ is an algorithm that outputs a sequence that emulates true randomness.
The quality of emulation is tested statistically through a battery of test, e.g. `Diehard tests <https://en.wikipedia.org/wiki/Diehard_tests>`_.
These tests check if various statistical tests can separate the pseudo-random sequence from a true random one.

Pseudo-random generators usually have an internal state and require its initialization, also sometimes known as seeding.
States initialization algorithms take user provided _seed_ value, usually an integer or a finite seqeuence of integers, and scramble it 
to populate the internal state of the pseudo-random generator. 

The sequence from the pseudo-random generator, unlike from true random generator, is repeatable, provided the internal state can be
saved and restored, or initialized to the same state.
