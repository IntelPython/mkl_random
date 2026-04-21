# changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [dev] (MM/DD/YYYY)

### Added

### Changed

### Fixed

## [1.4.0] (04/16/2026)

### Added
* Added `mkl_random` patching for NumPy, with `mkl_random` context manager, `is_patched` query, and `patch_numpy_random` and `restore_numpy_random` calls to replace `numpy.random` calls with calls from `mkl_random.interfaces.numpy_random` [gh-90](https://github.com/IntelPython/mkl_random/pull/90)

* Added `mkl_random.interfaces` with `mkl_random.interfaces.numpy_random` interface, which aliases `mkl_random` functionality to more strictly adhere to NumPy's API (i.e., drops arguments and functions which are not part of standard NumPy) [gh-92](https://github.com/IntelPython/mkl_random/pull/92)

* Added third-party tests from `numpy.random` which tests the `mkl_random.interfaces.numpy_random` interface [gh-103](https://github.com/IntelPython/mkl_random/pull/103)

### Changed
* Updates to `mkl_random` implementations to better align with newer versions of `numpy.random` [gh-103](https://github.com/IntelPython/mkl_random/pull/103), [gh-115](https://github.com/IntelPython/mkl_random/pull/115)
* Made conda recipe dependency on numpy configurable through `USE_NUMPY_BASE` environment variable [gh-105](https://github.com/IntelPython/mkl_random/pull/105)

### Fixed
* Various bugfixes including a hang in `zipf` when called with `np.nan` and size-1 1D arrays being cast to scalars [gh-103](https://github.com/IntelPython/mkl_random/pull/103), [gh-115](https://github.com/IntelPython/mkl_random/pull/115)

### Removed
* Dropped support for Python 3.9 [gh-81](https://github.com/IntelPython/mkl_random/pull/81)

### Deprecated
* Deprecated `mkl_random.RandomState` in favor of `mkl_random.MKLRandomState`, delegating `mkl_random.RandomState` to the `mkl_random.interfaces.numpy_random` interface [gh-92](https://github.com/IntelPython/mkl_random/pull/92)

## [1.3.1] (12/02/2025)

### Added
* Enabled support of Python 3.14 [gh-79](https://github.com/IntelPython/mkl_random/pull/79)

## [1.3.0] (10/06/2025)

### Changed
* Used `GIT_DESCRIBE_TAG` and `GIT_DESCRIBE_NUMBER` in `meta.yaml` instead of manual stepping the numbers [gh-75](https://github.com/IntelPython/mkl_random/pull/75)
* Extended conda build scripts with the use of `WHEELS_OUTPUT_FOLDER` variable to build wheel packages [gh-74](https://github.com/IntelPython/mkl_random/pull/74)
* Updated `meta.yaml` to have a run dependency on `numpy-base` package [gh-73](https://github.com/IntelPython/mkl_random/pull/73)
