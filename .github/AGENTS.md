# AGENTS.md — .github/

CI/CD workflows and repo automation.

## Workflows (source of truth)
- `intel-ci.yml` — builds and tests the conda packages of `conda-recipe/` and
  the wheels, for the platforms and Python versions the recipe and
  `pyproject.toml` declare (`ci/matrix.py`); see `ci/README.md`
- `build_pip.yml` — editable pip build and test, including pre-release NumPy
- `build-with-clang.yml` — build and test with `icx`/`icpx` from the oneAPI apt
  repository
- `build-with-standard-clang.yml` — build and test with Ubuntu's `clang`
  package
- `build-docs.yml` — Sphinx build; publishes to `gh-pages` from `master`
- `pre-commit.yml` — lint/format checks
- `coverity.yml` — Coverity static analysis (see `coverity/README.md`)
- `openssf-scorecard.yml` — OpenSSF Scorecard
- `zizmor.yml` — GitHub Actions security lint

The three pip workflows test the editable install: `build_pip.yml` runs
`pytest mkl_random/tests`, the clang workflows `pytest --pyargs mkl_random`.
`intel-ci.yml` installs each built conda package and wheel in a test job of its
own and runs the suite with `ci/test.py`.

## Templates and config
- `copilot-instructions.md` — GitHub Copilot behavior; points back to the
  `AGENTS.md` files
- `pull_request_template.md` — PR description, verification, and checklist
- `ISSUE_TEMPLATE/` — bug report and feature request forms; `config.yml` adds
  contact links that send security reports to Intel's vulnerability process
  and usage questions to the docs
- `dependabot.yml` — update schedule for GitHub Actions and pre-commit hooks
- `CODEOWNERS` — default reviewers

## Policy
- Treat workflow YAML as canonical for platform/Python matrices, except
  `intel-ci.yml`, whose conda matrix is `conda-recipe/conda_build_config.yaml`
  and whose wheel matrix is the classifiers in `pyproject.toml`.
- Avoid doc claims about CI coverage unless present in workflow config.
