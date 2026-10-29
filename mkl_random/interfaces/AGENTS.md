# AGENTS.md — mkl_random/interfaces/

Drop-in replacement for the legacy part of `numpy.random`.

## Scope
- `numpy_random.py` — public module; its `__all__` is the covered set, and
  exactly what patching replaces (`_patch_numpy.py`)
- `_numpy_random.py` — implementation
- `README.md` — human-readable list of the covered classes and functions

## Guardrails
- Legacy `numpy.random` signatures and semantics are the contract here.
- Keep `README.md` in sync with `numpy_random.__all__`.
- Patching installs this interface in NumPy's place, so changes here reach
  patched NumPy users too; cover them with tests in `mkl_random/tests/`.
