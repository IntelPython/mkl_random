Contributing
============

:mod:`mkl_random` is an free and open source project.
We welcome and appreciate your contributions.

To contribute, fork the repo https://github.com/IntelPython/mkl_random.git,
clone it:

.. code-block::
    :caption: How to clone the repo

         git clone https://github.com/<fork-org>/mkl_random.git


A working compiler is needed to build :mod:`mkl_random`.
Gnu :code:`g++`, standard :code:`clang++`, and Intel LLVM :code:`icpx` are supported.

Make sure to install Python packages required to build :mod:`mkl_random`:

* :mod:`python`
* :mod:`numpy`
* :mod:`cython`
* :mod:`meson-python`
* :mod:`ninja`
* :mod:`cmake`
* :mod:`mkl-devel`

The Intel(R) MKL library and its headers are provided by the :mod:`mkl-devel`
package and located through CMake, so no environment variable needs to be set.

.. code-block:: bash
   :caption: Building mkl_random

      python -m pip install . --no-build-isolation --no-deps

To run test suite, install :mod:`pytest`, and run

.. code-block:: bash
   :caption: Running mkl_random test suite

      python -m pytest mkl_random/tests

To build documentation, install dependencies and running

.. code-block:: bash
   :caption: Building mkl_random documentation

      $ sphinx-build -M html docs/source docs/build

Rendered documentation can be found in "docs/build/html".
