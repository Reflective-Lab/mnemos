# Contributing to mnemos

mnemos is a Converge extension. Contributions follow the same conventions as the Converge foundation.

## Development

```sh
cargo check --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

While `converge-pack` is unreleased, the workspace patches it to the local Converge checkout via `[patch.crates-io]`. You need both repos checked out side by side:

```
~/dev/
├── work/converge/
└── extensions/mnemos/
```

## Boundaries

mnemos implements the **Suggestor layer** (`KnowledgeRetrievalSuggestor`, `KnowledgeStoreSuggestor`) on top of the **Backend layer** (vector storage, embedding clients). See the foundation's [Plug Boundary](https://github.com/Reflective-Lab/converge/blob/main/kb/Architecture/Plug%20Boundary.md) for the layering rule.

When adding capabilities, ask:

- Is this a new Suggestor (purposeful, agency-aware)? Add it under `src/suggestor.rs` or alongside.
- Is this a new Backend (operational adapter)? Add it under `src/storage/`, `src/embedding/`, or a new module.
- Does it cross the layer line? It probably shouldn't — split it.

## No `unsafe`

The workspace forbids `unsafe`. If you genuinely need it, open an issue first.

## Pull Requests

- Keep PRs small and focused.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Run `cargo check --workspace` before pushing.
- Reference the relevant Converge ADR if your change crosses a contract boundary.

## License

By contributing, you agree your contributions are licensed under MIT.
