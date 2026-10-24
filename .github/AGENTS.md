# AGENTS.md — .github/

CI/CD workflows and repo automation.

## Workflows (source of truth)
- `conda-package.yml` — Intel-channel conda build and test
- `conda-package-cf.yml` — conda-forge conda build and test
- `build_pip.yml` — editable pip build, including pre-release NumPy
- `build-with-clang.yml` — build with `icx`/`icpx` from the oneAPI apt repository
- `build-with-standard-clang.yml` — build with upstream clang
- `build-docs.yml` — Sphinx build; publishes to `gh-pages` from `master`
- `pre-commit.yml` — lint/format checks
- `coverity.yml` — Coverity static analysis (see `coverity/README.md`)
- `openssf-scorecard.yml` — OpenSSF Scorecard
- `zizmor.yml` — GitHub Actions security lint

## Policy
- Treat workflow YAML as canonical for platform/Python matrices.
- Avoid doc claims about CI coverage unless present in workflow config.
