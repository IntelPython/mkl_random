#!/bin/bash -ex

# meson locates MKL via find_package(MKL CONFIG) -> $PREFIX/lib/cmake/mkl,
# which is on CMAKE_PREFIX_PATH thanks to the compiler activation scripts.
# Do not export MKLROOT: MKLConfig.cmake then searches $MKLROOT/include with
# NO_DEFAULT_PATH and hard-errors if mkl_version.h is not there.
$PYTHON -m pip install --no-build-isolation --no-deps .
