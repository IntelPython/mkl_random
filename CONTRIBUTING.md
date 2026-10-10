# Contributing to `mkl_random`

See [README.md](README.md) for usage, [AGENTS.md](AGENTS.md) for a map of the
source tree, and [SECURITY.md](SECURITY.md) to report a vulnerability.

## Setup

Building needs a C++ compiler (`g++`, `clang++`, or `icpx`).

```sh
conda create -n mkl_random-dev -c conda-forge python pip mkl-devel numpy \
    meson-python ninja cmake "cython>=3.1.0" pytest
conda activate mkl_random-dev
pip install -e . --no-build-isolation --no-deps
```

To build the docs, install `sphinx`, `furo`, `sphinx-design`, and
`sphinxcontrib-programoutput`, then run
`sphinx-build -M html docs/source docs/build`.

## Checks

```sh
pytest mkl_random/tests
pre-commit run --all-files
```

The pre-commit hooks enforce formatting; otherwise, match the surrounding code.

## Guidelines

- Keep changes small and focused.
- Keep signatures compatible with legacy `numpy.random`.
- If a change alters what a fixed seed produces, say which streams changed in
  the `CHANGELOG.md` entry.
- Add tests with behavior changes, and a regression test with bug fixes. Seed
  every generator a test uses.
- Edit `mklrand.pyx` and `src/`, not generated C++.
- Keep patching reversible.

## Pull requests

Work on a branch and fill in the PR template. For user-visible changes, add a
`CHANGELOG.md` entry under `[dev]` with a `gh-NNN` link.

Contributions are licensed under the terms in [LICENSE.txt](LICENSE.txt).
