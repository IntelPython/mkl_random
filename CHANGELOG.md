# changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [dev] (MM/DD/YYYY)

### Added
* Enabled support of Python 3.14 [gh-79](https://github.com/IntelPython/mkl_random/pull/79)

## [1.3.0] (10/06/2025)

### Changed
* Used `GIT_DESCRIBE_TAG` and `GIT_DESCRIBE_NUMBER` in `meta.yaml` instead of manual stepping the numbers [gh-75](https://github.com/IntelPython/mkl_random/pull/75)
* Extended conda build scripts with the use of `WHEELS_OUTPUT_FOLDER` variable to build wheel packages [gh-74](https://github.com/IntelPython/mkl_random/pull/74)
* Updated `meta.yaml` to have a run dependency on `numpy-base` package [gh-73](https://github.com/IntelPython/mkl_random/pull/73)
