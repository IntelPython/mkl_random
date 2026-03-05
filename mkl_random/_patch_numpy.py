# Copyright (c) 2019, Intel Corporation
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Define functions for patching NumPy with MKL-based NumPy interface."""

import warnings
from contextlib import ContextDecorator
from threading import Lock, local

import numpy as _np

import mkl_random.interfaces.numpy_random as _nrand

_DEFAULT_NAMES = tuple(_nrand.__all__)


class _GlobalPatch:
    def __init__(self):
        self._lock = Lock()
        self._patch_count = 0
        self._restore_dict = {}
        self._patched_functions = tuple(_DEFAULT_NAMES)
        self._numpy_module = None
        self._requested_names = None
        self._active_names = ()
        self._patched = ()
        self._tls = local()

    def _normalize_names(self, names):
        if names is None:
            names = _DEFAULT_NAMES
        return tuple(names)

    def _validate_module(self, numpy_module):
        if not hasattr(numpy_module, "random"):
            raise TypeError(
                "Expected a numpy-like module with a `.random` attribute."
            )

    def _register_func(self, name, func):
        if name not in self._patched_functions:
            raise ValueError(f"{name} not an mkl_random function.")
        np_random = self._numpy_module.random
        if name not in self._restore_dict:
            self._restore_dict[name] = getattr(np_random, name)
        setattr(np_random, name, func)

    def _restore_func(self, name, verbose=False):
        if name not in self._patched_functions:
            raise ValueError(f"{name} not an mkl_random function.")
        try:
            val = self._restore_dict[name]
        except KeyError:
            if verbose:
                print(f"failed to restore {name}")
            return
        else:
            if verbose:
                print(f"found and restoring {name}...")
            np_random = self._numpy_module.random
            setattr(np_random, name, val)

    def _initialize_patch(self, numpy_module, names, strict):
        self._validate_module(numpy_module)
        np_random = numpy_module.random
        missing = []
        patchable = []
        for name in names:
            if name not in self._patched_functions:
                missing.append(name)
                continue
            if not hasattr(np_random, name) or not hasattr(_nrand, name):
                missing.append(name)
                continue
            patchable.append(name)

        if strict and missing:
            raise AttributeError(
                "Could not patch these names (missing on numpy.random or "
                "mkl_random.interfaces.numpy_random): "
                + ", ".join(str(x) for x in missing)
            )

        self._numpy_module = numpy_module
        self._requested_names = names
        self._active_names = tuple(patchable)
        self._patched = tuple(patchable)

    def do_patch(
        self,
        numpy_module=None,
        names=None,
        strict=False,
        verbose=False,
    ):
        if numpy_module is None:
            numpy_module = _np
        names = self._normalize_names(names)
        strict = bool(strict)

        with self._lock:
            local_count = getattr(self._tls, "local_count", 0)
            if self._patch_count == 0:
                self._initialize_patch(numpy_module, names, strict)
                if verbose:
                    print(
                        "Now patching NumPy random submodule with mkl_random "
                        "NumPy interface."
                    )
                    print(
                        "Please direct bug reports to "
                        "https://github.com/IntelPython/mkl_random"
                    )
                for name in self._active_names:
                    self._register_func(name, getattr(_nrand, name))
            else:
                if self._numpy_module is not numpy_module:
                    raise RuntimeError(
                        "Already patched a different numpy module; "
                        "call restore() first."
                    )
                if names != self._requested_names:
                    raise RuntimeError(
                        "Already patched with a different names set; "
                        "call restore() first."
                    )
            self._patch_count += 1
            self._tls.local_count = local_count + 1

    def do_restore(self, verbose=False):
        with self._lock:
            local_count = getattr(self._tls, "local_count", 0)
            if local_count <= 0:
                if verbose:
                    warnings.warn(
                        "Warning: restore_numpy_random called more times than "
                        "patch_numpy_random in this thread.",
                        stacklevel=2,
                    )
                return

            self._tls.local_count = local_count - 1
            self._patch_count -= 1
            if self._patch_count == 0:
                if verbose:
                    print("Now restoring original NumPy random submodule.")
                for name in tuple(self._restore_dict):
                    self._restore_func(name, verbose=verbose)
                self._restore_dict.clear()
                self._numpy_module = None
                self._requested_names = None
                self._active_names = ()
                self._patched = ()

    def is_patched(self):
        with self._lock:
            return self._patch_count > 0

    def patched_names(self):
        with self._lock:
            return list(self._patched)


_patch = _GlobalPatch()


def patch_numpy_random(
    numpy_module=None,
    names=None,
    strict=False,
    verbose=False,
):
    """
    Patch NumPy's random submodule with mkl_random's NumPy interface.

    Parameters
    ----------
    numpy_module : module, optional
        NumPy-like module to patch. Defaults to imported NumPy.
    names : iterable[str], optional
        Attributes under `numpy_module.random` to patch.
    strict : bool, optional
        Raise if any requested symbol cannot be patched.
    verbose : bool, optional
        Print messages when starting the patching process.

    Examples
    --------
    >>> import numpy as np
    >>> import mkl_random
    >>> mkl_random.is_patched()
    False
    >>> mkl_random.patch_numpy_random(np)
    >>> mkl_random.is_patched()
    True
    >>> mkl_random.restore()
    >>> mkl_random.is_patched()
    False
    """
    _patch.do_patch(
        numpy_module=numpy_module,
        names=names,
        strict=bool(strict),
        verbose=bool(verbose),
    )


def restore_numpy_random(verbose=False):
    """
    Restore NumPy's random submodule to its original implementations.

    Parameters
    ----------
    verbose : bool, optional
        Print message when starting restoration process.
    """
    _patch.do_restore(verbose=bool(verbose))


def is_patched():
    """Return whether NumPy has been patched with mkl_random."""
    return _patch.is_patched()


def patched_names():
    """Return names actually patched in `numpy.random`."""
    return _patch.patched_names()


class mkl_random(ContextDecorator):
    """
    Context manager and decorator to temporarily patch NumPy random submodule
    with MKL-based implementations.

    Examples
    --------
    >>> import numpy as np
    >>> import mkl_random
    >>> with mkl_random.mkl_random(np):
    ...     x = np.random.normal(size=10)
    """

    def __init__(self, numpy_module=None, names=None, strict=False):
        self._numpy_module = numpy_module
        self._names = names
        self._strict = strict

    def __enter__(self):
        patch_numpy_random(
            numpy_module=self._numpy_module,
            names=self._names,
            strict=self._strict,
        )
        return self

    def __exit__(self, *exc):
        restore_numpy_random()
        return False
