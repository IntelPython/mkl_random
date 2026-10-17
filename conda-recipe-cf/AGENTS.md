# AGENTS.md — conda-recipe-cf/

conda-forge variant of the conda recipe.

## Differences from `conda-recipe/`
- Installs with `pip install` directly; no wheel build or retag
- Pins NumPy with `pin_compatible` at run time and runs `pip check` in the
  package test
- Adds macOS compiler entries to `conda_build_config.yaml`

## Guardrails
- Keep conda-forge recipe semantics separate from the Intel-channel recipe.
- Keep changes in step with `.github/workflows/conda-package-cf.yml`, which
  builds and tests this recipe.
