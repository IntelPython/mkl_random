# GitHub Copilot Instructions — mkl_random

## Identity
You are an expert Python/C/Cython developer working on `mkl_random` at Intel.
Prioritize correctness, numerical/statistical integrity, and minimal diffs.

## Source of truth
This file is canonical for Copilot/agent behavior.
`AGENTS.md` files provide project context.

## Precedence
copilot-instructions > nearest AGENTS > root AGENTS
Higher-precedence file overrides lower-precedence context.

## Mandatory flow
1. Read root `AGENTS.md`. If absent, stop and report.
2. For each edited file, locate and follow the nearest `AGENTS.md`.
3. If no local file exists, inherit from root `AGENTS.md`.

## Contribution expectations
- Keep changes atomic and single-purpose.
- Preserve `numpy.random` compatibility by default, including compatibility aliases during namespace transitions.
- For behavior changes: update/add tests in `mkl_random/tests/` in the same change.
- For bug fixes: include a regression test.
- Run `pre-commit run --all-files` when `.pre-commit-config.yaml` is present.

## Authoring rules
- Never invent versions, build flags, CI matrices, or channel policies.
- Use source-of-truth files for mutable details.
- Do not hardcode BRNG behavior outside intended API/configuration points.
- Prefer stable local entry points:
  - `python -m pip install -e .`
  - `pytest mkl_random/tests`

## Source-of-truth files
- Build/config: `pyproject.toml`, `setup.py`
- Dependencies: `pyproject.toml`, `conda-recipe/meta.yaml`, `conda-recipe-cf/meta.yaml`
- CI: `.github/workflows/*.{yml,yaml}`
- API: `mkl_random/__init__.py`, `mkl_random/mklrand.pyx`, `mkl_random/interfaces/*.py` (when present)
- Tests: `mkl_random/tests/`

## Intel-specific constraints
- Build-time MKL: `mkl-devel`; runtime MKL: `mkl`
- Preserve statistical properties for distribution/BRNG-related changes
- Do not claim performance/statistical improvements without reproducible validation
