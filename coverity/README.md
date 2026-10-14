# Triaging Coverity Scan findings

Static analysis runs on [Coverity Scan](https://scan.coverity.com) via
[`.github/workflows/coverity.yml`](../.github/workflows/coverity.yml) (weekly + on
demand). Analysis runs on Black Duck's servers; triage is done in the Scan web UI.

Most findings are false positives in Cython-**generated** `mklrand.cpp`, not in
code we maintain. This guide records why, and how to keep triage from resetting.

## Where findings come from

- **Generated** `mklrand.cpp` (from `mklrand.pyx`): `__Pyx_*` / `__pyx_pw_*` /
  `__pyx_tp_*` helpers and wrappers are boilerplate — findings here are ~always
  false positives (see the table below). `__pyx_pf_*` functions are the C
  translation of our `.pyx` bodies; a real `.pyx` bug could surface here, though
  so far all have been false positives too (Coverity can't see Python-level
  invariants like non-negative dtype sizes or fixed-length tuples).
- **Hand-written** C++ under `mkl_random/src/`. Most likely place for a real bug.

## Keeping triage durable: the Cython pin

A Cython *version* bump regenerates `mklrand.cpp` wholesale, which churns the
Coverity CIDs and silently drops their triage — the same boilerplate then returns
under new CIDs. So **Cython is pinned in `coverity.yml`** (not `pyproject.toml`,
so shipped wheels are unaffected). The pin works only because the build runs with
`--no-build-isolation`; bumping it means re-triaging the boilerplate.

## Reducing the noise: a Project Component

**Project Settings → Components** buckets defects by a path regex. Define one to
group (not hide) the generated file so it can be filtered out of view — path-based,
so it survives Cython bumps:

- **Name:** `Cython-generated`  **Path regex:** `.*/mklrand\.cpython.*`

This matches only the generated file (hand-written sources are under
`mkl_random/src/`, no `mklrand` substring). Group only — do **not** mark it
*ignored*, as that also drops the `__pyx_pf_*` bodies (see [declined](#evaluated-and-declined)).

## Review checklist

Don't blanket-ignore the generated file — prioritise instead:

1. **Findings under `mkl_random/src/`** — review every one.
2. **High/Medium findings in `__pyx_pf_*`** — verify against the `.pyx`; if it's a
   Python-level invariant Coverity can't see, mark `False Positive` with a reason.
3. **Boilerplate families below** — bulk-triage `False Positive` / `Ignore`.

## Known false-positive families

Match on **checker + mechanism**, not CID (CIDs reset on a Cython bump or engine
upgrade). Helper names below are from Cython 3.3.0 and vary between versions.

| Family | Checker | Why it's a false positive |
| --- | --- | --- |
| `__pyx_tp_traverse_*`, `__Pyx_CyFunction_traverse` | DEADCODE | `__Pyx_call_type_traverse` is a constant-`0` macro on standard CPython, so `if (e) return e;` is dead; live only in the Limited-API build. |
| `__Pyx_PyCode_New`, `__Pyx_CallSlotAsVectorcallUnpackDict` | DEADCODE | `__Pyx_PyTuple_SET_ITEM` expands to void `PyTuple_SET_ITEM` yielding `0`, so `if (... != 0)` is dead; live only in the Limited-API build. |
| `__Pyx_AddTraceback` | DEADCODE | `c_line` is `0` unless the optional `CYTHON_CLINE_IN_TRACEBACK` feature is on, so the `-c_line` branch is dead by default. |
| `__Pyx_ParseKeywordDict` | DEADCODE | In the CPython < 3.13 branch `found` is only 0/1, so `if (found < 0)` is dead; that guard serves the ≥ 3.13 `PyDict_GetItemRef` path. |
| `__Pyx_PyLong_As_*` (per C type) | DEADCODE | Integer-conversion helper templates; dead branches are compile-time-selected version/overflow guards. |
| `__pyx_pf_*` temp cleanup (e.g. `choice`, `multivariate_normal`) | UNUSED_VALUE | `__pyx_t_N = 0;` nulls a temporary after its ref is transferred, guarding the error path against a double-DECREF; dead only on the straight-line path. |
| `__pyx_pw_*` keyword wrappers (`rand`, `randn`, …) | CHECKED_RETURN | `PyDict_Size` result is checked on the *next* line (`if (... < 0) __PYX_ERR(...)`); the statistical heuristic misfires because it's one line from the call. |

`sanity_check_for_cython` (DEADCODE) is the same disposition but comes from
meson's compiler-probe unit, not `mklrand.cpp`.

**Verified individually — OUT_OF_BOUNDS in `_seed_impl` (High):** the unpack
`brng_token, stream_id = _parse_brng_argument(brng)` (`mklrand.pyx:1571`) generates
`PyTuple_GET_ITEM(sequence, 1)`, guarded by `if (unlikely(size != 2)) __PYX_ERR(...)`,
and `_parse_brng_argument` always returns a fixed 2-tuple (`mklrand.pyx:1533`).
Coverity just doesn't tie the index back to the guard. No overflow possible.

## Evaluated and declined

- **Modeling files** correct the behavior of *called* functions; our FPs are
  intraprocedural (dead branches, macros, `#if`), which models can't reach. The
  only fit is the `PyDict_Size` CHECKED_RETURN case — not worth the maintenance.
- **Dropping the generated unit** (hard exclude), e.g. after `cov-build`:
  ```bash
  cov-manage-emit --dir cov-int --tu-pattern "file('.*mklrand.*\\.cpp')" delete
  ```
  Also drops the `__pyx_pf_*` bodies, so it's disabled in favour of the checklist.
