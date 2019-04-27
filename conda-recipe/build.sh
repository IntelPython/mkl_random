#!/bin/bash -x

export CFLAGS="-I$PREFIX/include $CFLAGS"
$PYTHON setup.py install --old-and-unmanageable
