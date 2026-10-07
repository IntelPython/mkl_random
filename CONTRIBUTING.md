# Contributing to `mkl_random`

This document covers the development workflow: how to get a working build, how
to run the checks, and what to include in a pull request.

For end-user installation and usage, see [README.md](README.md) and the
[documentation](https://intelpython.github.io/mkl_random/). For a map of the
source tree, see [`AGENTS.md`](AGENTS.md), which links to the local `AGENTS.md`
files in directories that have their own rules. Security vulnerabilities go
through the process in [SECURITY.md](SECURITY.md).

---

## Development setup

Building requires a C++ compiler, oneMKL headers and libraries (`mkl-devel`),
and NumPy. CI builds with `g++`, upstream `clang++`, and Intel `icpx`. A conda
environment is the least surprising way to get the rest:

```sh
# add python=X.Y to target a specific interpreter
conda create -n mkl_random-dev -c conda-forge --override-channels python pip \
    mkl-devel numpy meson-python ninja cmake "cython>=3.1.0" pytest
conda activate mkl_random-dev
```

Then build in place, which reuses the environment's MKL and NumPy:

```sh
pip install -e . --no-build-isolation --no-deps --verbose
```

`pyproject.toml` defines the supported Python range, and `.github/workflows/` is
canonical for the versions CI covers. `README.md` documents the non-editable
install paths.

### Rebuilding

`meson-python` rebuilds the extension on import for editable installs, so
editing `mklrand.pyx`, `src/`, or `meson.build` and rerunning `pytest` is
usually enough. The Cython-generated C++ and the compiled extension live under
`build/<tag>/` rather than in the source tree. If a build gets into a bad state,
`rm -rf build` and reinstall.

## Running the checks

```sh
pytest mkl_random/tests         # test suite
pre-commit run --all-files      # lint and format hooks
```

To run a single test, use `pytest mkl_random/tests/<file>::<test>`; to lint one
file, `pre-commit run --files <path>`.

Install the hooks once with `pre-commit install` and they run on each commit.
`.pre-commit-config.yaml` is the source of truth for the tooling.

To build the documentation, install `sphinx`, `furo`, `sphinx-design`, and
`sphinxcontrib-programoutput`, then run
`sphinx-build -M html docs/source docs/build`.

Opening a pull request also runs CI, which builds and tests the package across
platforms and Python versions, builds the documentation, and runs various lint
and static-analysis checks.

## Code style

Style is loose, and the pre-commit hooks enforce most of it:

- Python is formatted with `black` and `isort`, with a line length of 80.
- Cython is not touched by `black`. `isort` sorts its imports, `cython-lint`
  checks it against the same 80-column limit, and string literals use double
  quotes.
- C++ sources follow the repository's `.clang-format`.
- Otherwise, match the surrounding code.

## Dos and don'ts

**Do**

- Keep changes atomic and single-purpose.
- Keep signatures and accepted arguments compatible with legacy `numpy.random`.
  `mkl_random.interfaces.numpy_random` and patching make this package a drop-in
  replacement. Call out an intentional break in the PR.
- Treat seeded output as part of the API. If a change alters what a fixed seed
  produces, say which streams changed in the `CHANGELOG.md` entry.
- Add tests in `mkl_random/tests/` alongside behavior changes, and a regression
  test with every bug fix.
- Seed every generator a test uses, so tests stay deterministic.
- Keep patching reversible and observable: anything installed can be
  uninstalled, and `is_patched()` reports the truth.
- Keep the extension free-threading compatible.
- Cite the source-of-truth file for mutable details: `pyproject.toml`,
  `meson.build`, `conda-recipe*/meta.yaml`, `.github/workflows/`.
- Give benchmark numbers reproducible context — hardware, versions, and the
  command you ran.

**Don't**

- Commit generated artifacts, or hand-edit Cython-generated C++.
- Hardcode versions, build flags, CI matrices, or channel URLs in documentation.
- Assert on timing or throughput in the test suite.
- Introduce ISA-specific assumptions outside explicit build configuration.

## Submitting a change

Work on a branch: the `no-commit-to-branch` hook blocks direct commits to
`master` and `maintenance/*`.

If the change is user-visible — behavior, API, seeded output, packaging, or
build output — add a `CHANGELOG.md` entry under the `[dev]` heading in the
matching section, with a
`[gh-NNN](https://github.com/IntelPython/mkl_random/pull/NNN)` link. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Docs,
tooling, and CI-only changes are usually left out.

Then open the PR and fill in the template, including what you verified locally
and what you left to CI.

By contributing you agree that your contributions are licensed under the
BSD-3-Clause terms in [LICENSE.txt](LICENSE.txt).
