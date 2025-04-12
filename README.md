## `mkl_random` -- a NumPy-based Python interface to Intel® oneAPI Math Kernel Library (OneMKL) Random Number Generation functionality
[![Conda package using conda-forge](https://github.com/IntelPython/mkl_random/actions/workflows/conda-package-cf.yml/badge.svg)](https://github.com/IntelPython/mkl_random/actions/workflows/conda-package-cf.yml)

`mkl_random` started as a part of Intel® Distribution for Python optimizations to NumPy.

Per NumPy's community suggestions, voiced in https://github.com/numpy/numpy/pull/8209, it is being released as a
stand-alone package.

Prebuilt `mkl_random` can be installed into conda environment from Intel's channel using:

```
  conda install -c https://software.repos.intel.com/python/conda mkl_random
```

or from conda forge channel:

```
   conda install -c conda-forge mkl_random
```

---

To install `mkl_random` PyPI package please use following command:

```
   python -m pip install -i https://software.repos.intel.com/python/pypi --extra-index-url https://pypi.org/simple mkl_random
```

If command above installs NumPy package from the Pypi, please use following command to install Intel optimized NumPy wheel package from Intel Pypi Cloud:

```
   python -m pip install -i https://software.repos.intel.com/python/pypi --extra-index-url https://pypi.org/simple mkl_random numpy==<numpy_version>
```

Where `<numpy_version>` should be the latest version from https://software.repos.intel.com/python/conda/

---

`mkl_random` is not fixed-seed backward compatible drop-in replacement for `numpy.random`, meaning that it implements sampling from the same distributions as `numpy.random`.

For distributions directly supported in Intel® OneMKL, `method` keyword is supported:

```
   mkl_random.standard_normal(size=(10**5, 10**3), method='BoxMuller')
```

Additionally, `mkl_random` exposes different basic random number generation algorithms available in MKL. For example to use `SFMT19937` use

```
   mkl_random.RandomState(77777, brng='SFMT19937')
```

For generator families, such that `MT2203` and Wichmann-Hill, a particular member of the family can be chosen by specifying ``brng=('WH', 3)``, etc.

The list of supported by `mkl_random.RandomState` constructor `brng` keywords is as follows:

  * 'MT19937'
  * 'SFMT19937'
  * 'WH' or ('WH', id)
  * 'MT2203' or ('MT2203', id)
  * 'MCG31'
  * 'R250'
  * 'MRG32K3A'
  * 'MCG59'
  * 'PHILOX4X32X10'
  * 'NONDETERM'
  * 'ARS5'
