# AGENTS.md — conda-recipe/

Intel-channel conda packaging.

## Files
- `meta.yaml` — package metadata, dependencies, and the package test
  (`pytest --pyargs mkl_random`)
- `build.sh` / `bld.bat` — build a wheel with `python -m build`, then install
  it; `build.sh` also retags the wheel's platform
- `conda_build_config.yaml` — compiler and C library pins

## Guardrails
- Treat recipe files as canonical for packaging intent and dependency pins.
- Keep recipe changes in step with `.github/workflows/conda-package.yml`, which
  builds and tests this recipe.
