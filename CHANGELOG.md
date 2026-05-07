# Changelog

All notable changes to mnemos will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-05-07

### Changed

- Cargo package renamed from `mnemos` to `converge-mnemos-knowledge`; Rust
  library and binary names remain `mnemos` and `mnemos-server`.
- `Justfile` `security-audit` now passes the same `--ignore RUSTSEC-*`
  flags `cargo-deny` already ignores in `deny.toml`, with a comment to
  keep them in lockstep.
- Coverage recipe excludes binary entry points and gRPC transport stubs
  (`main.rs`, `suggestor.rs`, `bin/`, `grpc/`) — these are exercised by
  integration tests, not unit tests, so including them depressed the
  reported coverage without reflecting actual unit-test reach.

### Fixed

- `deny.toml` `[advisories].ignore` now carries documented entries for
  `RUSTSEC-2025-0134` (rustls-pemfile via tonic 0.12) and
  `RUSTSEC-2025-0141` (bincode 1.3.3 — direct dep, trusted local
  cache files). Mirrors the foundation baseline.

## [1.0.0] - 2026-05-05

### Added

Initial release. Extracted from `converge/crates/knowledge` as a Converge extension per [ADR-008](https://github.com/Reflective-Lab/converge/blob/main/kb/Architecture/ADRs/ADR-008-extension-crate-boundaries.md).

- Self-learning knowledgebase with HNSW-style vector indexing
- `KnowledgeRetrievalSuggestor` and `KnowledgeStoreSuggestor` implementations of `converge_pack::Suggestor`
- Agentic memory primitives: causal, temporal, reflexion, skill library, online and meta-learning
- Embedding engine with hash-based and OpenAI backends
- Markdown ingestion with routing
- gRPC service (`mnemos-server`) and CLI (`mnemos`)
- Bincode-based local storage backend
- Proto package: `mnemos.v1`

### Changed

- Crate renamed from `converge-knowledge` to `mnemos`
- Proto package renamed from `converge.knowledge.v1` to `mnemos.v1`
- Binaries renamed: `converge-knowledge` → `mnemos`, `converge-knowledge-server` → `mnemos-server`
