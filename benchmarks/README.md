# mkl_random ASV Benchmarks

Performance benchmarks for [mkl_random](https://github.com/IntelPython/mkl_random) using
[Airspeed Velocity (ASV)](https://asv.readthedocs.io/en/stable/).

### Coverage

| File | API | Cases | Engines | Sizes |
|------|-----|-------|---------|-------|
| `bench_continuous.py` | `MKLRandomState` | Every continuous distribution; shape parameters chosen to hit each oneMKL method branch (e.g. gamma a>1, 0.6<a<1, a<0.6; beta Cheng/Johnk/Atkinson) | MT19937 | 1k, 1M |
| `bench_discrete.py` | `MKLRandomState` | Every discrete distribution; binomial and hypergeometric in both table and acceptance/rejection regimes | MT19937 | 1k, 1M |
| `bench_methods.py` | `MKLRandomState` | `method=` variants: gaussian, lognormal, multinormal Cholesky, poisson (POISNORM, PTPE at λ=10 and λ=100) | MT19937 | 1k, 1M |
| `bench_engines.py` | `MKLRandomState(brng=...)` | uniform, normal and bounded `randint` per engine; full-range bits and `bytes` on engines that support UniformBits | all deterministic engines | 1M |
| `bench_integers.py` | `randint` | Every dtype, each C code path (small range, wide range, power-of-two, full range), and array-valued bounds | MT19937 | 1k, 1M |
| `bench_permutations.py` | `shuffle`, `permutation`, `choice` | 1-D, 2-D C/F order, lists; `choice` with and without replacement and `p` | MT19937 | 1k, 1M |
| `bench_multivariate.py` | `multinomial`, `multivariate_normal`, `dirichlet` | small fixed dimensions | MT19937 | 1k, 1M |
| `bench_array_params.py` | `MKLRandomState` | Distributions with one parameter value per output element | MT19937 | 1k, 100k |
| `bench_interfaces.py` | `mkl_random.interfaces.numpy_random`, patched `numpy.random` | Common NumPy calls through the drop-in paths | MT19937 | 1k, 1M |
| `bench_memory.py` | `MKLRandomState` | Peak RSS of 10M-element fills and of 10k repeated small calls | MT19937 | 10M |

## Threading

Set `MKL_NUM_THREADS` in the environment before running ASV to control the
thread count used by MKL:

```bash
MKL_NUM_THREADS=8 asv run --python=same --quick HEAD^!
```

If `MKL_NUM_THREADS` is not set, `__init__.py` applies a default: **4** threads
when the machine has 4 or more physical cores, or **1** (single-threaded)
otherwise. This keeps results comparable across CI machines in the shared pool
regardless of their total core count. Physical cores are detected via
`psutil.cpu_count(logical=False)` — hyperthreads are excluded per MKL
recommendation.

## Notes on Measurement

### Seeding and warmup

Every benchmark seeds its state with a fixed seed in `setup`. Timing
benchmarks also make one untimed call per timed method, so stream
initialization and first-call costs are not charged to the first measured
iteration. Peak-memory benchmarks skip this, because ASV counts memory
allocated in `setup` towards the peak.

### Engine restrictions

oneMKL does not implement `UniformBits32`/`UniformBits64` for `WH`, `MCG31`,
`R250` and `MRG32K3A`, and on those engines mkl_random currently returns
uninitialized memory instead of raising. Full-range integer draws and
`bytes()` are therefore benchmarked only on the engines listed in
`_utils._ENGINES_BITS`. The
non-deterministic engine is excluded everywhere because its output, and its
availability, depend on the hardware.

### Method names

An unrecognized `method=` string silently falls back to the default method.
When adding a method benchmark, use a name listed in the docstring of the
corresponding `MKLRandomState` method.

### Patched NumPy

`PatchedNumpyRandom` raises in `setup` if `numpy.random` is not served by
mkl_random after `patch_numpy_random()`, so a broken patch shows up as a
failed benchmark rather than as stock NumPy timings.

## Running Benchmarks

Prerequisites:

```bash
pip install ".[benchmark]"
```

Check that every benchmark imports and sets up:

```bash
asv check --python=same
```

Run benchmarks against the current environment:

```bash
asv run --python=same --quick HEAD^!
```

Compare two commits:

```bash
asv continuous --python=same HEAD~1 HEAD
```

View results in a browser:

```bash
asv publish
asv preview
```

## CI

The benchmark pipeline installs `requirements.txt` with conda, so entries must
be conda package names; keep it in step with the `benchmark` extra in
`pyproject.toml`. Results are recorded only for branches listed under
`branches` in `asv.conf.json`.
