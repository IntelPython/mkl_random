# AGENTS.md — mkl_random

## What this project is
NumPy-based Python interface to Intel® oneMKL Random Number Generation (RNG) functionality. Provides MKL-accelerated random sampling from distributions compatible with `numpy.random`. Part of Intel® Distribution for Python. Archetype: **python** (Cython + C extensions).

Layers: Python interface, Cython bindings (`mklrand.pyx`), C backend (`src/`).

## How it's structured
- `mkl_random/` — main package
  - `mklrand.pyx` — Cython RNG interface
  - `src/` — C code generation scripts
  - `tests/` — pytest suite
- `conda-recipe/`, `conda-recipe-cf/` — Intel/conda-forge builds
- `examples/` — parallel MC, random states demos

Build: `pyproject.toml` + `setup.py`. Runtime: `mkl`, `numpy>=1.26.4`.

## How to work in it
- Keep changes atomic and single-purpose.
- Preserve `numpy.random` API compatibility; document divergence in commit message.
- Pair changes with tests and docstrings.
- Never assume MKL or NumPy versions; use source-of-truth files.
- **RNG specifics:** Changes to BRNG (basic RNG) selection or distribution methods must preserve statistical properties.
- **Local dev:** `conda create -n dev python numpy cython mkl-devel pytest && pip install -e .`

For agent policy: `.github/copilot-instructions.md`

## Where truth lives
- Build/config: `pyproject.toml`, `setup.py`
- Dependencies: `pyproject.toml` (`dependencies`, `optional-dependencies`), `conda-recipe/meta.yaml`, `conda-recipe-cf/meta.yaml`
- CI: `.github/workflows/`
- API/contracts: `mkl_random/__init__.py`, NumPy `random` docs
- Stable entry points: `python -m pip install .`, `pytest mkl_random/tests`

## Directory map
No local AGENTS files — project is small enough for root-level guidance only.
