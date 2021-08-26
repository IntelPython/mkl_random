#!/usr/bin/env python
# Copyright (c) 2017-2019, Intel Corporation
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

from __future__ import division, print_function, absolute_import

import os
import sys
import io
import re

with io.open('mkl_random/_version.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)

with open("README.md", "r", encoding="utf-8") as file:
    long_description = file.read()

VERSION = version

CLASSIFIERS = ""

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=False)

    config.add_subpackage('mkl_random')

    config.version = VERSION

    return config


from distutils.command.sdist import sdist
def setup_package():
    src_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    old_path = os.getcwd()
    os.chdir(src_path)
    sys.path.insert(0, src_path)

    from setuptools import setup
    from numpy.distutils.core import setup
    metadata = dict(
        name = 'mkl_random',
        maintainer = "Intel Corp.",
        maintainer_email = "scripting@intel.com",
        description = "NumPy-based Python interface to Intel (R) MKL Random Number Generation functionality",
        long_description = long_description,
        long_description_content_type="text/markdown",
        url = "http://github.com/IntelPython/mkl_random",
        author = "Intel Corporation",
        download_url = "http://github.com/IntelPython/mkl_random",
        license = 'BSD',
        classifiers = [_f for _f in CLASSIFIERS.split('\n') if _f],
        platforms = ["Windows", "Linux", "Mac OS-X"],
        test_suite = 'nose.collector',
        python_requires = '>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*',
        install_requires=['numpy'],
        configuration = configuration
    )

    try:
        setup(**metadata)
    finally:
        del sys.path[0]
        os.chdir(old_path)

    return None

if __name__ == '__main__':
    setup_package()
