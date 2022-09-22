#!/usr/bin/env python
# Copyright (c) 2017-2022, Intel Corporation
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
import io
import re
from os.path import join
import Cython.Build
from setuptools import setup, Extension
import numpy as np


with io.open('mkl_random/_version.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

VERSION = version

CLASSIFIERS = CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""


def extensions():
    mkl_root = os.environ.get('MKLROOT', None)
    if mkl_root:
        mkl_info = {
            'include_dirs': [join(mkl_root, 'include')],
            'library_dirs': [join(mkl_root, 'lib'), join(mkl_root, 'lib', 'intel64')],
            'libraries': ['mkl_rt']
        }
    else:
        try:
            mkl_info = get_info('mkl')
        except:
            mkl_info = dict()

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
            ('_LARGEFILE64_SOURCE', '1')]

    exts = [
        Extension(
            "mkl_random.mklrand",
            [
                os.path.join("mkl_random", "mklrand.pyx"),
                os.path.join("mkl_random", "src", "mkl_distributions.cpp"),
                os.path.join("mkl_random", "src", "randomkit.c"),
            ],
            depends = [
                os.path.join("mkl_random", "src", "mkl_distributions.hpp"),
                os.path.join("mkl_random", "src", "randomkit.h"),
                os.path.join("mkl_random", "src", "numpy.pxd")
            ],
            include_dirs = [os.path.join("mkl_random", "src"), np.get_include()] + mkl_include_dirs,
            libraries = libs,
            library_dirs = lib_dirs,
            extra_compile_args = eca + [
                # "-ggdb", "-O0", "-Wall", "-Wextra",
            ],
            define_macros=defs + [("NDEBUG",None),], # [("DEBUG", None),]
            language="c++"
        )
    ]

    return exts


setup(
    name = "mkl_random",
    maintainer = "Intel Corp.",
    maintainer_email = "scripting@intel.com",
    description = "NumPy-based Python interface to Intel (R) MKL Random Number Generation functionality",
    version = version,
    include_package_data=True,
    ext_modules=extensions(),
    cmdclass={'build_ext': Cython.Build.build_ext},
    zip_safe=False,
    long_description = long_description,
    long_description_content_type="text/markdown",
    url = "http://github.com/IntelPython/mkl_random",
    author = "Intel Corporation",
    download_url = "http://github.com/IntelPython/mkl_random",
    license = "BSD",
    classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms = ["Windows", "Linux", "Mac OS-X"],
    test_suite = "pytest",
    python_requires = '>=3.7',
    setup_requires=["Cython",],
    install_requires = ["numpy >=1.16"],
    keywords=["MKL", "VSL", "true randomness", "pseudorandomness",
              "Philox", "MT-19937", "SFMT-19937", "MT-2203", "ARS-5",
              "R-250", "MCG-31",],
)
