Contributing
============

:mod:`mkl_random` is an free and open source project. 
We welcome and appreciate your contributions.

To contribute, fork the repo https://github.com/IntelPython/mkl_random.git,
clone it:

.. code-block::
    :caption: How to clone the repo

         git clone https://github.com/<fork-org>/mkl_random.git


A working compiler is needed build :mod:`mkl_random`. 
Both Gnu :code:`g++` and Intel LLVM :code:`icpx` are supported.

Make sure to install Python packages required to build :mod:`mkl_random`:

* :mod:`python`
* :mod:`numpy`
* :mod:`cython`
* :mod:`setuptools`

You would also need Intel(R) MKL library and its headers. Set :code:`MKLROOT` environment
variable so that :code:`${MKLROOT}/include/mkl.h` and :code:`${MKLROOT}/lib/libmkl_rt.so`
can be found.

.. code-block:: bash
   :caption: Building mkl_random

      $ export MKLROOT=</path/to/mkl>
      python -m pip install .

To run test suite, install :mod:`pytest`, and run

.. code-block:: bash
   :caption: Running mkl_random test suite

      python -m pytest mkl_random/tests

To build documentation, install dependencies and running

.. code-block:: bash
   :caption: Building mkl_random documentation

      $ sphinx-build -M html docs/source docs/build

Rendered documentation can be found in "docs/build/html".