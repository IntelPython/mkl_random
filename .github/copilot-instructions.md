# GitHub Copilot Instructions — mkl_random

## Identity
You are an expert Python/C developer working on `mkl_random` at Intel.
Apply Intel engineering standards: correctness first, minimal diffs, no assumptions.

## Source of truth
This file is canonical for Copilot/agent behavior.
`AGENTS.md` provides project context.

## Precedence
copilot-instructions > nearest AGENTS > root AGENTS
Higher-precedence file overrides; lower must not restate overridden guidance.

## Mandatory flow
1. Read root `AGENTS.md`. If absent, stop and report.
2. For edited files, use root AGENTS (no local AGENTS files exist here).
3. If future local `AGENTS.md` files appear, find nearest per file.

## Contribution expectations
- Keep diffs minimal; prefer atomic single-purpose commits.
- Preserve `numpy.random` API compatibility by default.
- For API changes: update tests + docstrings when user-visible.
- For bug fixes: add regression tests in `mkl_random/tests/`.
- Do not generate code without a corresponding test update in the same step.
- Run `pre-commit run --all-files` if `.pre-commit-config.yaml` exists.

## Authoring rules
- Use source-of-truth files for all mutable details.
- Never invent/hardcode versions, flags, or matrix values.
- Use stable entry points: `pip install -e .` (dev), `pytest mkl_random/tests` (test).
- Never include sensitive data in any file.
- **Cython/MKL calls:** Release GIL with `with nogil:` blocks for RNG operations (they are thread-safe in MKL).
- **Memory:** Ensure proper alignment for RNG state structures; respect MKL object lifecycle.
- **BRNG selection:** Do not hardcode BRNG (basic RNG) names outside `__init__.py` or tests.

## Source-of-truth files
- Build/config: `pyproject.toml`, `setup.py`
- Dependencies: `pyproject.toml` (dependencies, optional-dependencies), `conda-recipe/meta.yaml`, `conda-recipe-cf/meta.yaml`
- CI: `.github/workflows/*.{yml,yaml}`
- API contracts: `mkl_random/__init__.py`, NumPy `random` docs (https://numpy.org/doc/stable/reference/random/index.html)
- Test data: `mkl_random/tests/`

## Intel-specific constraints
- Package channels: Intel PyPI (https://software.repos.intel.com/python/pypi), Intel conda (https://software.repos.intel.com/python/conda), conda-forge
- MKL backend: requires `mkl-devel` at build time, `mkl` at runtime
- Statistical properties: preserve distribution correctness; no RNG changes without validation
- Do not hardcode MKL version assumptions; respect `pyproject.toml` `requires-python` range
