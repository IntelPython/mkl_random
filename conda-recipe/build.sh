#!/bin/bash -x

export CFLAGS="-I$PREFIX/include $CFLAGS"
export MKLROOT=$CONDA_PREFIX
$PYTHON -m pip install . --no-build-isolation --no-deps
