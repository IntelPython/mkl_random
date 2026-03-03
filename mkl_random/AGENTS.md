# AGENTS.md — mkl_random/

Core package implementation for MKL-backed random functionality.

## Key files
- `__init__.py` — public package exports/API wiring
- `mklrand.pyx` — Cython bindings to MKL RNG support
- `_init_helper.py` — platform/runtime loading helpers
- `src/` — C-level support and generation assets
- `tests/` — package tests (see local AGENTS in tests)

## Guardrails
- Preserve `numpy.random`-compatible behavior by default.
- RNG algorithm/distribution changes must preserve statistical correctness.
- Keep BRNG/distribution behavior explicit and test-covered.
- Prefer minimal, isolated edits around touched API paths.
