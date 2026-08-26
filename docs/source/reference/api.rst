.. _fullapi:

MKL-powered RandomState
=======================

.. autoclass:: mkl_random.MKLRandomState
    :members:
    :inherited-members:

.. note::

   ``mkl_random.RandomState`` is a deprecated alias of
   :class:`mkl_random.MKLRandomState`, kept for backward compatibility with
   legacy code.  A NumPy-compatible drop-in replacement for
   :class:`numpy.random.RandomState` is available in
   :mod:`mkl_random.interfaces.numpy_random`.
