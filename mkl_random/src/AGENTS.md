# AGENTS.md — mkl_random/src/

Hand-written C++11 kernels compiled into the `mklrand` extension.

## Key files
- `randomkit.cpp`, `randomkit.h` — stream state and basic-generator setup
  (`brng_list`)
- `mkl_distributions.cpp`, `mkl_distributions.h` — distribution kernels on MKL
  VS
- `mklrand_py_helper.h` — Python/NumPy helpers for the extension
- `numpy_multiiter_workaround.h` — needed only to build against NumPy < 2.0
  (numpy/numpy#26990)

## Guardrails
- Coordinate kernel changes with their callers in `mklrand.pyx`.
- A kernel change that alters what a fixed seed produces is user-visible; call
  it out in the CHANGELOG.
- Keep code C++11; `meson.build` sets the standard.
