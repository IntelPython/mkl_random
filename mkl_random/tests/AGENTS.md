# AGENTS.md — mkl_random/tests/

Test suite for the generators, the NumPy interface, and patching.
`meson.build` installs it with the package.

## Files
- `test_random.py` — distributions and generators
- `test_patch.py` — patch and restore state
- `test_cli.py` — persistent patch install, uninstall, and status
- `test_freethreading.py` — concurrent use; the race tests run only on a
  free-threaded build
- `third_party/` — tests adapted from NumPy's `numpy/random/tests`

## Expectations
- Behavior changes include test updates in the same PR; bug fixes include a
  regression test.
- Seed every generator a test uses, and keep tests free of timing assertions.

## Entry points
- `pytest mkl_random/tests` from a checkout with an editable install
- `pytest --pyargs mkl_random` against an installed package
