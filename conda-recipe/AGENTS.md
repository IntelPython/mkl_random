# AGENTS.md — conda-recipe/

The conda recipe, built against conda-forge only.

## Files
- `meta.yaml` — package metadata, dependencies, and the package test
  (`pip check` and `pytest --pyargs mkl_random`)
- `build.sh` / `bld.bat` — `pip install` of the source tree; neither sets
  `MKLROOT` (see the comments in both)
- `conda_build_config.yaml` — the Python versions CI builds, NumPy 2 and the
  compiler and C library pins for Linux, macOS and Windows. `ci/matrix.py`
  reads its `python:` list: one version per line, no selectors

## Guardrails
- Treat recipe files as canonical for packaging intent and dependency pins.
- `meta.yaml` pins `mkl` at run time with `pin_compatible`.
- The version comes from the git tag (`GIT_DESCRIBE_TAG`).
- `intel-ci.yml` builds this recipe with `ci/build.py`, which runs the package
  test in `meta.yaml`, and then tests the built package in a separate job.
- Wheels don't come from this recipe: `ci/build.py --system wheel` builds them
  from `pyproject.toml`.
