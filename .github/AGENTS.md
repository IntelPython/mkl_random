# AGENTS.md — .github/

CI/CD workflows and repo automation.

## Workflows (source of truth)
- `conda-package.yml` — Intel channel conda build/test pipeline
- `conda-package-cf.yml` — conda-forge-oriented build/test pipeline
- `build-with-clang.yml` — clang compatibility checks
- `build-docs.yml` — docs build pipeline

## Policy
- Treat workflow YAML as canonical for platform/Python matrices.
- Keep artifact naming and channel usage aligned with workflow files.
- Avoid doc claims about CI coverage unless present in workflow config.
