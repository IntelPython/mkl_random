# Copyright (c) 2017, Intel Corporation
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

import os
import sys
from os.path import join
import Cython.Build
from setuptools import setup, Extension
import numpy as np


def extensions():
    mkl_root = os.environ.get('MKLROOT', None)
    if mkl_root:
        mkl_info = {
            'include_dirs': [join(mkl_root, 'include')],
            'library_dirs': [join(mkl_root, 'lib'), join(mkl_root, 'lib', 'intel64')],
            'libraries': ['mkl_rt']
        }
    else:
        raise ValueError("MKLROOT environment variable not set.")

    mkl_include_dirs = mkl_info.get('include_dirs', [])
    mkl_library_dirs = mkl_info.get('library_dirs', [])
    mkl_libraries = mkl_info.get('libraries', ['mkl_rt'])

    libs = mkl_libraries
    lib_dirs = mkl_library_dirs

    if sys.platform == 'win32':
        libs.append('Advapi32')

    Q = '/Q' if sys.platform.startswith('win') or sys.platform == 'cygwin' else '-'
    eca = [Q + "std=c++11"]
    if sys.platform == "linux":
        eca.extend(["-Wno-unused-but-set-variable", "-Wno-unused-function"])

    defs = [('_FILE_OFFSET_BITS', '64'),
            ('_LARGEFILE_SOURCE', '1'),
            ('_LARGEFILE64_SOURCE', '1'),
            ("PY_ARRAY_UNIQUE_SYMBOL", "mkl_random_ext")]

    exts = [
        Extension(
            "mkl_random.mklrand",
            sources = [
                join("mkl_random", "mklrand.pyx"),
                join("mkl_random", "src", "mkl_distributions.cpp"),
                join("mkl_random", "src", "randomkit.cpp"),
            ],
            depends = [
                join("mkl_random", "src", "mkl_distributions.hpp"),
                join("mkl_random", "src", "randomkit.h"),
                join("mkl_random", "src", "numpy_multiiter_workaround.h")
            ],
            include_dirs = [join("mkl_random", "src"), np.get_include()] + mkl_include_dirs,
            libraries = libs,
            library_dirs = lib_dirs,
            extra_compile_args = eca,
            define_macros=defs + [("NDEBUG", None)],
            language="c++"
        )
    ]

    return exts


setup(
    cmdclass={'build_ext': Cython.Build.build_ext},
    ext_modules=extensions(),
    zip_safe=False,
)
