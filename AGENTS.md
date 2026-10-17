# AGENTS.md — mkl_random

Entry point for agent context in this repo.

## What this project is
`mkl_random` is a NumPy-compatible random module backed by Intel® oneMKL RNG.
It provides accelerated random sampling with API compatibility goals relative to `numpy.random`.

## Key components
- **Python package:** `mkl_random/`
- **Cython layer:** `mkl_random/mklrand.pyx`
- **C/C++ backend sources:** `mkl_random/src/` (`*.cpp`, `*.h`, support scripts)
- **Tests:** `mkl_random/tests/`
- **Packaging:** `conda-recipe/`, `conda-recipe-cf/`
- **Examples:** `examples/`

## Build/runtime basics
- Build system: `pyproject.toml` + `setup.py`
- Build deps: `cython`, `numpy`, `mkl-devel`
- Runtime deps: `numpy`, `mkl`

## Development guardrails
- Preserve `numpy.random` API compatibility unless change is explicitly requested.
- Keep namespace/API transitions explicit (e.g., `mkl_random.MKLRandomState` with compatibility aliases where applicable).
- RNG changes must preserve statistical correctness and reproducibility expectations.
- If patching/integration entry points are touched, keep patch semantics and rollback behavior explicit.
- Keep diffs minimal and pair behavior changes with tests.
- Avoid hardcoding mutable versions/matrices/channels in docs.

## Where truth lives
- Build/config: `pyproject.toml`, `setup.py`
- Dependencies: `pyproject.toml`, `conda-recipe*/meta.yaml`
- CI matrices/workflows: `.github/workflows/*.{yml,yaml}`
- Public API: `mkl_random/__init__.py`, `mkl_random/interfaces/*.py` (when present)
- Tests: `mkl_random/tests/`

For behavior policy, see `.github/copilot-instructions.md`.

## Directory map
Use nearest local `AGENTS.md` when present:
- `.github/AGENTS.md` — CI workflows and automation policy
- `mkl_random/AGENTS.md` — package-level implementation context
- `mkl_random/tests/AGENTS.md` — testing scope and conventions
- `conda-recipe/AGENTS.md` — Intel-channel conda packaging
- `conda-recipe-cf/AGENTS.md` — conda-forge recipe context
- `examples/AGENTS.md` — runnable examples and expected behavior
