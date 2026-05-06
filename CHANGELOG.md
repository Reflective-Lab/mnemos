# Changelog

All notable changes to mnemos will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Cargo package renamed from `mnemos` to `converge-mnemos-knowledge`; Rust
  library and binary names remain `mnemos` and `mnemos-server`.

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
