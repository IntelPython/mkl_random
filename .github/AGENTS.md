# AGENTS.md — .github/

CI/CD workflows and repo automation.

## Workflows (source of truth)
- `conda-package.yml` — Intel-channel conda build and test
- `conda-package-cf.yml` — conda-forge conda build and test
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
The conda workflows build with `conda build --no-test` and run
`pytest --pyargs mkl_random` in separate test jobs on the built package.

## Templates and config
- `pull_request_template.md` — PR description, verification, and checklist
- `ISSUE_TEMPLATE/` — bug report and feature request forms; `config.yml` adds
  contact links that send security reports to Intel's vulnerability process
  and usage questions to the docs
- `dependabot.yml` — update schedule for GitHub Actions and pre-commit hooks
- `CODEOWNERS` — default reviewers

## Policy
- Treat workflow YAML as canonical for platform/Python matrices.
- Avoid doc claims about CI coverage unless present in workflow config.
