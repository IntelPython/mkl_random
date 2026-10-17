# AGENTS.md — mkl_random

Entry point for agent context in this repo.

## What this project is
`mkl_random` is a NumPy-based interface to Intel® oneMKL random number
generation. `MKLRandomState` exposes the distributions of legacy
`numpy.random.RandomState` with a choice of MKL basic generators (`brng`).
`mkl_random.interfaces.numpy_random` is a drop-in replacement for legacy
`numpy.random`, and runtime patching can install it in NumPy's place.

## Key components
- **Package and public API:** `mkl_random/`, `mkl_random/__init__.py`
- **Cython extension:** `mkl_random/mklrand.pyx`
- **C++ kernels:** `mkl_random/src/`
- **NumPy interface:** `mkl_random/interfaces/`
- **Patching:** `_patch_numpy.py`, plus persistent and one-shot patching in
  `patch.py`, `with_patch.py`, `_patch_startup.py`, and the `__main__.py` CLI
- **Tests:** `mkl_random/tests/`
- **Docs:** `docs/` (Sphinx)
- **Packaging:** `conda-recipe/`, `conda-recipe-cf/`
- **Benchmarks:** `benchmarks/`

## Build/runtime basics
- Build system: `pyproject.toml` + `meson.build`
- Build deps: `mkl-devel`, `numpy`, `meson-python`, `cmake`, `ninja`, `cython`,
  and a C++ compiler
- Runtime deps: `numpy`; the conda recipes add MKL
- Setup, checks, and style: `CONTRIBUTING.md`
- Single test: `pytest mkl_random/tests/<file>::<test>`
- Single-file lint: `pre-commit run --files <path>`

## Development guardrails
- Keep signatures and accepted arguments compatible with legacy `numpy.random`.
- Seeded output is part of the contract: a change to what a fixed seed
  produces needs a CHANGELOG entry naming the affected streams.
- Edit `mklrand.pyx` and `src/`, not the Cython-generated C++.
- Keep patching reversible, with `is_patched()` reporting the truth.
- Keep the extension free-threading compatible.
- Pair behavior changes with tests and keep diffs minimal.
- Avoid hardcoding mutable versions/matrices/channels in docs.

## Where truth lives
- Build/config: `pyproject.toml`, `meson.build`
- Dependencies: `pyproject.toml`, `conda-recipe*/meta.yaml`
- CI/workflows: `.github/workflows/*.yml`
- Public API: `mkl_random/__init__.py`, `mkl_random/interfaces/numpy_random.py`
- Tests: `mkl_random/tests/`

## Directory map
Use nearest local `AGENTS.md` when present:
- `.github/AGENTS.md` — CI workflows and automation policy
- `mkl_random/AGENTS.md` — package modules, the extension, and generators
- `mkl_random/src/AGENTS.md` — C++ kernels
- `mkl_random/interfaces/AGENTS.md` — `numpy.random` drop-in interface
- `mkl_random/tests/AGENTS.md` — test scope and conventions
- `docs/AGENTS.md` — Sphinx documentation
- `conda-recipe/AGENTS.md` — Intel-channel conda packaging
- `conda-recipe-cf/AGENTS.md` — conda-forge recipe
- `benchmarks/AGENTS.md` — ASV performance suite
