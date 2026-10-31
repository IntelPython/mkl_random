# Contributing to `mkl_random`

See [README.md](README.md) for usage, [AGENTS.md](AGENTS.md) for a map of the
source tree, and [SECURITY.md](SECURITY.md) to report a vulnerability.

## Setup

Building needs C and C++ compilers (`gcc`/`g++`, `clang`/`clang++`, or
`icx`/`icpx`; MSVC on Windows).

```sh
conda create -n mkl_random-dev --override-channels -c conda-forge python pip \
    mkl-devel numpy meson-python ninja cmake "cython>=3.1.0" pytest \
    pre-commit pylint shellcheck
conda activate mkl_random-dev
pip install -e . --no-build-isolation --no-deps
```

`--override-channels` keeps MKL from mixing with your configured channels.

To build the docs, install `sphinx`, `furo`, `sphinx-design`, and
`sphinxcontrib-programoutput`, then run
`sphinx-build -M html docs/source docs/build`.

## Checks

Work on a branch: on `master` or a `maintenance/*` branch, the
`no-commit-to-branch` hook makes `pre-commit run` fail. From the editable
install above:

```sh
pytest mkl_random/tests
pre-commit run --all-files
```

The editable install rebuilds the extension on import, and the `pylint` and
`shellcheck` hooks run the copies installed in the environment, so activate
`mkl_random-dev` first. The first run needs network access: pre-commit fetches
each hook, and builds `gitleaks` and `actionlint` from source with Go,
downloading a Go toolchain if none is on `PATH`. Run `pre-commit install` once
to run the hooks on each commit.

The pre-commit hooks enforce formatting; otherwise, match the surrounding code.

## Guidelines

- Keep changes small and focused.
- Keep signatures compatible with legacy `numpy.random`.
- If a change alters what a fixed seed produces, say which streams changed in
  the `CHANGELOG.md` entry.
- Add tests with behavior changes, and a regression test with bug fixes. Seed
  every generator a test uses.
- Edit `mkl_random/mklrand.pyx` and `mkl_random/src/`, not generated C++.
- Keep patching reversible.

## Pull requests

Fill in the PR template. For user-visible changes, add a `CHANGELOG.md` entry
under `## [dev]`, in the matching section, with a
`[gh-NNN](https://github.com/IntelPython/mkl_random/pull/NNN)` link.

Contributions are licensed under the terms in [LICENSE.txt](LICENSE.txt).
