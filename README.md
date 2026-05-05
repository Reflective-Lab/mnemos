# mnemos

A self-learning knowledgebase that gets smarter the more you use it. Implements
Converge recall and storage suggestors on top of vector storage, agentic
memory, and continual learning.

`mnemos` is a Converge **extension** — it depends on Converge's stable
contracts (`converge-pack`) and lives outside the Converge foundation
repository. See the foundation's
[Plug Boundary](https://github.com/Reflective-Lab/converge/blob/main/kb/Architecture/Plug%20Boundary.md)
for why.

## Layout

- `crates/mnemos` — library + CLI + gRPC server. Implements
  `KnowledgeRetrievalSuggestor` and `KnowledgeStoreSuggestor` against
  `converge-pack`, plus vector storage, embedding, agentic memory, and
  ingestion.

## Status

Extracted from `converge/crates/knowledge` on 2026-05-05 as part of the
v3.8 foundation extraction (ADR-008). Pre-1.0 — no published versions yet.

## Build

```sh
cargo check
cargo build --release
```

While `converge-pack` is unreleased, this workspace patches it to the local
checkout at `../../work/converge/crates/pack` via `[patch.crates-io]`.

## License

MIT — see [LICENSE](LICENSE).
