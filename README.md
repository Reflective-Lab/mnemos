# mnemos

[![CI](https://github.com/Reflective-Lab/mnemos-knowledge/actions/workflows/ci.yml/badge.svg)](https://github.com/Reflective-Lab/mnemos-knowledge/actions/workflows/ci.yml)
[![Coverage](https://github.com/Reflective-Lab/mnemos-knowledge/actions/workflows/coverage.yml/badge.svg)](https://github.com/Reflective-Lab/mnemos-knowledge/actions/workflows/coverage.yml)
[![Security](https://github.com/Reflective-Lab/mnemos-knowledge/actions/workflows/security.yml/badge.svg)](https://github.com/Reflective-Lab/mnemos-knowledge/actions/workflows/security.yml)
[![Stability](https://github.com/Reflective-Lab/mnemos-knowledge/actions/workflows/stability.yml/badge.svg)](https://github.com/Reflective-Lab/mnemos-knowledge/actions/workflows/stability.yml)
[![Crates.io](https://img.shields.io/crates/v/converge-mnemos-knowledge.svg)](https://crates.io/crates/converge-mnemos-knowledge)
[![docs.rs](https://docs.rs/converge-mnemos-knowledge/badge.svg)](https://docs.rs/converge-mnemos-knowledge)
[![dependency status](https://deps.rs/repo/github/Reflective-Lab/mnemos-knowledge/status.svg)](https://deps.rs/repo/github/Reflective-Lab/mnemos-knowledge)
![MSRV](https://img.shields.io/badge/MSRV-1.94.0-blue)
<img alt="gitleaks badge" src="https://img.shields.io/badge/protected%20by-gitleaks-blue">
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Knowledge, recall, retrieval, and memory for Converge formations.

`mnemos` is a Converge extension. It implements knowledge-base storage,
retrieval, ingestion, embeddings, feedback learning, and Converge suggestors
without putting those mechanisms inside the Converge foundation.

Cargo package: `converge-mnemos-knowledge`. Rust library and binary names
remain `mnemos` and `mnemos-server`.

## Why It Exists

Converge owns governed proposal promotion. Mnemos owns memory. A formation can
ask Mnemos for relevant knowledge, store durable observations, or learn from
feedback while Converge still decides what becomes fact.

## What Mnemos Owns

- `KnowledgeBase`, entries, search options, and search results.
- Local storage and vector-style retrieval.
- Embedding support, including OpenAI embeddings.
- Markdown and rich-media ingestion.
- Agentic memory: causal, temporal, reflexion, skill, session, online, and
  meta-learning modules.
- Feedback collection, replay, batch learning, and insight jobs.
- CLI and gRPC server surfaces.
- `KnowledgeRetrievalSuggestor` and `KnowledgeStoreSuggestor`.
- Typed proposal provenance through `ProvenanceSource` / `MNEMOS_PROVENANCE`.
- Suggestor-boundary tracing through `mnemos.suggestor.execute` spans.

## Boundary

| Layer | Responsibility |
|---|---|
| Converge | Context, proposals, facts, promotion, and suggestor contract. |
| Mnemos | Knowledge storage, recall, ingestion, memory, learning, and recall suggestors. |
| Products | Which knowledge stores to use, tenancy, credentials, retention, and deployment mode. |

## Repository Layout

```text
crates/mnemos/
  proto/knowledge.proto
  src/core/       KnowledgeBase, entries, search
  src/embedding/  Hash and OpenAI embedding support
  src/ingest/     Markdown, rich media, routing
  src/agentic/    Causal, temporal, reflexion, skills, sessions
  src/learning/   Feedback, replay, batch jobs
  src/grpc/       gRPC server and client
  src/suggestor.rs
```

## Usage

```rust
use mnemos::{KnowledgeBase, KnowledgeEntry};

let kb = KnowledgeBase::open("./knowledge.db").await?;
kb.add_entry(KnowledgeEntry::new(
    "Rust ownership",
    "Ownership and borrowing keep memory safe without a GC.",
))
.await?;

let results = kb.search_simple("memory safety", 5).await?;
```

## Feature Flags

- Default: `cli`, `grpc`.
- `cli`: enables the `mnemos` binary.
- `grpc`: enables the `mnemos-server` binary.
- `memory-only`: checks memory-only operation.

## Development

```sh
just check
just check-memory
just test
just lint
just doc
```

Converge platform dependencies resolve from crates.io.

## Project Files

- [AGENTS.md](AGENTS.md) - agent entrypoint and boundary rules.
- [CHANGELOG.md](CHANGELOG.md) - release notes.
- [CONTRIBUTING.md](CONTRIBUTING.md) - contribution guide.
- [SECURITY.md](SECURITY.md) - vulnerability reporting and operator notes.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - community expectations.

## Status

Extracted from `converge/crates/knowledge` on 2026-05-05 as part of the v3.8
foundation extraction.

## License

MIT - see [LICENSE](LICENSE).
