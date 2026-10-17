# AGENTS.md — mkl_random/interfaces/

Drop-in replacement for the legacy part of `numpy.random`.

## Scope
- `numpy_random.py` — public module; `_numpy_random.py` — implementation
- `README.md` — the list of classes and functions the interface covers

## Guardrails
- Legacy `numpy.random` signatures and semantics are the contract here.
- Keep `README.md` in sync when the covered set changes.
- Patching installs this interface in NumPy's place, so changes here reach
  patched NumPy users too; cover them with tests in `mkl_random/tests/`.
