# AGENTS.md — conda-recipe/

Intel-channel conda packaging context.

## Scope
- `meta.yaml` — package metadata and dependency pins
- `build.sh` / `bld.bat` — platform build scripts

## Guardrails
- Treat recipe files as canonical for build/runtime dependency intent.
- Keep recipe updates synchronized with CI workflow expectations.
- Do not infer platform/Python matrix from docs; read workflow YAML.
