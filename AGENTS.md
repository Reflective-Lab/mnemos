# mnemos Agent Guide

This is the canonical agent entrypoint for `mnemos`.

`mnemos` is a Converge extension for knowledge bases, recall, retrieval,
storage, and agentic memory.

## Start Here

1. Read `README.md`.
2. Read `/Users/kpernyer/dev/extensions/kb/Modules/Mnemos.md`.
3. Check `Cargo.toml` feature flags for CLI, gRPC, and memory-only modes.
4. Use `just --list` for local commands.

## Commands

```bash
just check
just check-memory
just test
just lint
just doc
```

## Boundaries

- Converge owns the proposal and fact contract.
- `mnemos` owns retrieval, storage, memory, learning, ingestion, and recall
  suggestors.
- Products decide whether recall runs embedded, through gRPC, or not at all.

## Rules

- Preserve `unsafe_code = "forbid"`.
- Do not let recall bypass Converge promotion.
- Keep storage and ingestion behavior explicit and testable.
- Update `README.md`, `CHANGELOG.md`, and the extensions KB when public memory
  or recall behavior changes.
