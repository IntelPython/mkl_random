#!/bin/bash -x

export CFLAGS="-I$PREFIX/include $CFLAGS"
MKLROOT=$CONDA_PREFIX $PYTHON setup.py install --old-and-unmanageable
