# AGENTS.md — mkl_random/tests/

Test suite for RNG behavior, API compatibility, and regressions.

## Expectations
- Behavior changes in package code should include test updates in the same PR.
- Add regression tests for bug fixes.
- Keep tests deterministic when possible (fixed seeds / stable assertions).
- For statistical assertions, use robust criteria and document rationale.

## Entry point
- `pytest mkl_random/tests`
