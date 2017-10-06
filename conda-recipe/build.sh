#!/bin/bash -x

if [ `uname` == Darwin ]; then
    export MACOSX_DEPLOYMENT_TARGET=10.10
fi

export CFLAGS="-I$PREFIX/include $CFLAGS"
export CC=icc
export LDSHARED="icc -shared"

$PYTHON setup.py config --compiler=intelem --fcompiler=intelem build install --old-and-unmanageable
