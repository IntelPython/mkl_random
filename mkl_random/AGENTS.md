# AGENTS.md — mkl_random/

Package sources: the Cython extension, public API, and patching entry points.

## Key files
- `mklrand.pyx` — the `mklrand` extension: `MKLRandomState`, the
  distributions, and the basic-generator registry
- `__init__.py` — public API surface
- `_patch_numpy.py` — `patch_numpy_random()`, `restore_numpy_random()`,
  `is_patched()`, and the `mkl_random()` context manager
- `patch.py`, `_patch_startup.py`, `with_patch.py`, `__main__.py` — persistent
  (`.pth`) and one-shot patching behind `python -m mkl_random`
- `_version.py` — the version; `meson.build` reads it

## Guardrails
- Use `MKLRandomState` for the MKL-specific API and
  `interfaces.numpy_random.RandomState` for the NumPy drop-in in new code and
  docs; `mkl_random.RandomState` is deprecated.
- Edit `mklrand.pyx`; Cython generates the C++ into the build directory.
- Adding a basic generator touches every registration point: the enum in
  `src/randomkit.h` and its mirror in `mklrand.pyx`, `brng_list` in
  `src/randomkit.cpp`, `_brng_dict` and `_brng_dict_stream_max` in
  `mklrand.pyx`, the `brng` docstrings, the list in `README.md`, and a page
  under `docs/source/reference/`. The enum value indexes `brng_list`, so keep
  the order in sync.
- Keep `freethreading_compatible=True` in `mklrand.pyx`.
- New modules must be listed in `py.install_sources` in `meson.build`, or they
  are not installed.
