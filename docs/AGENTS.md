# AGENTS.md — docs/

Sphinx sources for the published documentation.

## Scope
- `source/` — pages; `source/conf.py` — Sphinx configuration
- `source/reference/` — API reference, with one page per basic generator
- `source/maintenance/index.rst` — the contributor page; keep its commands in
  step with `CONTRIBUTING.md`

## Guardrails
- `release` in `source/conf.py` is set by hand; keep it equal to
  `mkl_random/_version.py`.
- Build locally with `sphinx-build -M html docs/source docs/build` before
  changing `conf.py` or adding extensions.
- `.github/workflows/build-docs.yml` builds the docs on pull requests (as an
  artifact) and publishes them to `gh-pages` on pushes to `master`.
