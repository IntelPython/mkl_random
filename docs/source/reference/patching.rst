.. _patching:

Patching :mod:`numpy.random`
============================

:mod:`mkl_random` can temporarily replace functions and classes in :mod:`numpy.random` with
:mod:`mkl_random`implementations from the :ref:`numpy interface <numpy_random_interface>`.


Functions
---------

.. autofunction:: mkl_random.patch_numpy_random

.. autofunction:: mkl_random.restore_numpy_random

.. autofunction:: mkl_random.is_patched


Context manager
---------------

.. autoclass:: mkl_random.mkl_random
    :members:

:class:`mkl_random.mkl_random` is both a context manager and a decorator, making it possible to
scope the patch to a block of code or a function.
