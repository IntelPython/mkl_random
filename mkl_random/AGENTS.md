# AGENTS.md — mkl_random/

Package sources: the Cython extension, public API, and patching entry points.

## Key files
- `mklrand.pyx` — the `mklrand` extension: `MKLRandomState`, the
  distributions, and the basic-generator registry
- `__init__.py` — public API surface
- `_init_helper.py` — imported first by `__init__.py`; adds `Library/bin` to
  the DLL search path in a Windows venv
- `_patch_numpy.py` — `patch_numpy_random()`, `restore_numpy_random()`,
  `is_patched()`, and the `mkl_random()` context manager
- `patch.py`, `_patch_startup.py`, `with_patch.py`, `__main__.py` — persistent
  (`.pth`) and one-shot patching behind `python -m mkl_random`
- `_version.py` — the version; `meson.build` reads it

## Guardrails
- Use `MKLRandomState` for the MKL-specific API and
  `interfaces.numpy_random.RandomState` for the NumPy drop-in in new code and
  docs; `mkl_random.RandomState` is deprecated. The existing
  `docs/source/reference/<brng>.rst` pages still use it, so don't copy that.
- Edit `mklrand.pyx`; Cython generates the C++ into the build directory.
- Append a new basic generator at the end of the enum: its value is the
  integer `brng=` accepts (`0 <= brng < BRNG_KINDS`) and indexes `brng_list`.
  Then update every registration point:
  - the enum and `BRNG_KINDS` in `src/randomkit.h`, and the enum's mirror in
    `mklrand.pyx`
  - `brng_list` in `src/randomkit.cpp`
  - `_brng_dict`, `_brng_dict_stream_max`, and the `brng` docstrings in
    `mklrand.pyx`
  - in `tests/test_random.py`: `_ALL_BRNGS`, the out-of-range id (`11`) in
    `test_out_of_range_integer_brng_falls_back`, and the upper boundary in
    `test_boundary_integer_brng_accepted`
  - the generator list in the root `README.md`
  - a new page under `docs/source/reference/`, plus the bullet list and
    `toctree` in its `index.rst`
  - the skip-ahead and leap-frog lists in `docs/source/how_to.rst`, if the
    generator supports either
  - `_ENGINES` in `benchmarks/benchmarks/_utils.py`, and `_ENGINES_BITS` if
    the generator implements `viRngUniformBits32/64`
- Keep `freethreading_compatible=True` in `mklrand.pyx`.
- New `mkl_random/*.py` and `interfaces/*.py` modules must be added to the
  matching `py.install_sources` call in `meson.build`, or they are not
  installed. `tests/` is installed by `install_subdir`; C++ files go in the
  extension's `sources`.
