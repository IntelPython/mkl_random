# AGENTS.md — benchmarks/

ASV performance suite for `mkl_random`.

## Scope
- `asv.conf.json` — ASV configuration, channels, and regression thresholds
- `benchmarks/` — benchmark modules plus shared helpers in `_utils.py`
- `README.md` — coverage table, threading default, measurement notes, and run
  commands

## Guardrails
- Treat `asv.conf.json` as canonical for ASV settings; treat `README.md` as
  canonical for what each module covers and how measurements are taken.
- Comparability across machines depends on the thread default in
  `benchmarks/__init__.py`, the fixed seeds, and the warmup calls in the timing
  benchmarks' `setup`. Changing any of them invalidates comparison against
  existing results — call it out explicitly.
- Follow the engine and `method=` restrictions in `README.md` when adding
  benchmarks.
- Report performance numbers with reproducible context: hardware, thread count,
  versions, and the command used.
- Results under `benchmarks/.asv/` are local artifacts.
