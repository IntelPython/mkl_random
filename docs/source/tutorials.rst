Beginner's guide
================

Installation
------------

Package :mod:`mkl_random` is available in conda ecosystem on "conda-forge", default, and "intel" channels.

.. code-block:: bash
    :caption: Install mkl_random from conda-forge channel

        $ conda install -c conda-forge mkl_random

.. code-block:: bash
    :caption: Install mkl_random from intel channel

        $ conda install -c intel mkl_random

.. code-block:: bash
    :caption: Install mkl_random from default

        $ conda install mkl_random

The package can also be installed via :code:`pip`, either from PyPI, or from 

.. code-block:: bash
    :caption: Install mkl_random using pip from intel channel on Anaconda

    $ pip install -i https://pypi.anaconda.org/intel/simple mkl_random

.. code-block:: bash
    :caption: Install mkl_random using pip from PyPI

    $ pip install mkl_random

The :mod:`mkl_random` is also distributed as part of `Intel(R) Distribution for Python* <https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html>`_.

First steps
-----------

The :mod:`mkl_random` package was designed following :class:`numpy.random.RandomState` class to 
make use of :mod:`mkl_random` easy for current uses of :mod:`numpy.random` module.

.. note::
    Since the first release of `mkl_random`, NumPy introduced new classes :class:`numpy.random.Generator` and 
    :class:`numpy.random.BitGenerator`, while also retaining :class:`numpy.random.RandomState` for backwards
    compatibility.

The state of pseudo-random number generator is stored in :class:`mkl_random.RandomState` class, 
so using :mod:`mkl_random` begins with creating an instance of this class:

.. code-block:: python
    :caption: Construct random number generator

        import mkl_random
        rs = mkl_random.RandomState(seed=1234)

Sampling from difference probability distribution is done by calling class methods on the constructed instance:

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

        n = 10**8
        batch_size = 10**6
        accepted = 0
        sampled = 0
        while sampled < n:
            sampled += batch_size
            x = rs.uniform(0, 1, size=batch_size)
            y = rs.uniform(0, 1, size=batch_size)
            accepted += np.sum(x*x + y*y < 1.0)
        
        print("Pi estimate: ", 4. * (accepted / n))

Sample output of running such an example:

.. code-block:: bash
    :caption: Sample output after executing above script

        $ python pi.py
        Pi estimate:  3.14167732