## ``mkl_random`` -- a NumPy-based Python interface to Intel (R) MKL Random Number Generation functionality
[![Build Status](https://travis-ci.com/IntelPython/mkl_random.svg?branch=master)](https://travis-ci.com/IntelPython/mkl_random)

`mkl_random` has started as Intel (R) Distribution for Python optimizations for NumPy.

Per NumPy's community suggestions, voiced in https://github.com/numpy/numpy/pull/8209, it is being released as a
stand-alone package.

Prebuilt `mkl_random` can be installed into conda environment from Intel's channel on Anaconda cloud:

```
  conda install -c intel mkl_random
```

`mkl_random` is not fixed-seed backward compatible drop-in replacement for `numpy.random`, meaning that it implements sampling from the same distributions as `numpy.random`.

For distributions directly supported in Intel (R) Math Kernel Library (MKL), `method` keyword is supported:

```
   mkl_random.standard_normal(size=(10**5, 10**3), method='BoxMuller')
```

Additionally, `mkl_random` exposes different basic random number generation algorithms available in MKL. For example to use `SFMT19937` use

```
   mkl_random.RandomState(77777, brng='SFMT19937')
```

For generator families, such that `MT2203` and Wichmann-Hill, a particular member of the family can be chosen by specifying ``brng=('WH', 3)``, etc.

See MKL reference guide for more details:
   https://software.intel.com/en-us/mkl-developer-reference-c-random-number-generators

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
