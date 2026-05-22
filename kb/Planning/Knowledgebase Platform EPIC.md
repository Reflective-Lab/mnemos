---
tags: [planning, epic, knowledgebase, deployment, tenancy, meta-knowledge]
source: llm
---
# Knowledgebase Platform EPIC

## Context

Mnemos already has the core knowledge-base shape: entries, storage, retrieval,
embeddings, ingestion, learning, gRPC, CLI, and Converge suggestors. Current
retrieval work is improving the candidate-generation layer with vector search,
BM25, RRF, graph relationships, and learning.

This epic is the platform track for making the knowledge base deployable,
tenant-safe, and introspective. It deliberately sits beside retrieval work
rather than replacing it.

## Goals

- Run Mnemos as a reliable embedded or remote knowledge service.
- Support multiple tenants, workspaces, products, or memory scopes without
  cross-scope recall.
- Store and retrieve meta-knowledge about the knowledge base itself: corpus
  state, provenance, schemas, freshness, quality, policies, and retrieval
  traces.
- Keep recall governed by Converge promotion; no deployment or meta-knowledge
  path may turn retrieved content directly into facts.

## Non-Goals

- Replacing vector/BM25/RRF/graph retrieval.
- Building a hosted SaaS control plane.
- Moving product tenancy decisions into Converge foundation.
- Adding Prism fuzzy inference as a Mnemos core dependency. Prism can compose
  above Mnemos for product-specific soft reranking or admission rules.

## Workstreams

### 1. Deployment Surface

Make Mnemos straightforward to run as a service.

Requirements:

- Add a typed runtime config that can be loaded from file and environment.
- Document supported modes: embedded library, CLI, gRPC server, and
  `memory-only`.
- Add server readiness and liveness semantics beyond the current health call.
- Define storage path layout, backup/restore expectations, and flush behavior.
- Add graceful shutdown behavior for the gRPC server.
- Document operator-visible logs and tracing fields for startup, search, ingest,
  storage, and errors.

### 2. Multi-Tenant Knowledge Stores

Make tenancy explicit at storage, API, and retrieval boundaries.

Requirements:

- Introduce a typed tenant or namespace identifier.
- Ensure add, update, delete, get, search, feedback, related-entry, and stats
  operations execute inside one tenant scope.
- Apply tenant scope before vector, lexical, graph, or learning-based scoring.
- Prevent related-entry links across tenants unless an explicit shared-scope
  policy exists.
- Define storage isolation: separate files/directories or a tenant-keyed storage
  index.
- Provide a migration path for existing single-tenant stores.
- Add CLI and gRPC support for passing tenant scope.

### 3. Meta-Knowledge

Let the knowledge base know useful facts about its own corpus and operation.

Requirements:

- Define meta-knowledge entry types for source lineage, corpus summary,
  schema/version, freshness, quality signals, ingestion runs, retrieval traces,
  and policy notes.
- Keep meta-knowledge distinguishable from user/domain knowledge by category,
  metadata, or a typed field.
- Add APIs or helpers to write meta-knowledge during ingestion, retrieval, and
  maintenance jobs.
- Allow meta-knowledge to be searched when explicitly requested, while keeping it
  out of ordinary user recall by default.
- Record retrieval metadata sufficient to explain why a candidate was surfaced:
  retriever(s), rank, score, filters, and tenant scope.
- Keep Converge promotion as the only path from retrieved meta-knowledge to
  accepted facts.

## Acceptance Criteria

- A developer can start a local gRPC Mnemos service from documented config and
  verify readiness.
- Two tenants can store similarly named entries and searches return only the
  caller's tenant entries.
- Tenant scope is enforced before all scoring and fusion paths.
- Single-tenant stores still open through the existing `KnowledgeBase::open`
  path or through a documented migration.
- Meta-knowledge entries can be created, filtered, and explicitly searched.
- Ordinary search excludes meta-knowledge unless requested.
- Retrieval results can expose or persist enough trace data to audit vector,
  BM25, RRF, graph, and learning contributions.
- Public behavior changes are reflected in `README.md`, `CHANGELOG.md`, and the
  extensions KB module.

## Where To Start

1. Model the boundary in `crates/mnemos/src/core/`: tenant ID, scoped search
   options, and meta-knowledge type markers.
2. Extend persistence in `crates/mnemos/src/storage/mod.rs` without breaking
   existing stores.
3. Thread scope through `crates/mnemos/proto/knowledge.proto`,
   `crates/mnemos/src/grpc/`, and `crates/mnemos/src/main.rs`.
4. Add deployment config to `crates/mnemos/src/bin/server.rs` and document it in
   `README.md` plus `kb/Building/Getting Started.md`.
5. Add retrieval traces near `crates/mnemos/src/core/search.rs` and
   `crates/mnemos/src/core/knowledge_base.rs`.

## Key Files

- `crates/mnemos/src/core/entry.rs`
- `crates/mnemos/src/core/search.rs`
- `crates/mnemos/src/core/knowledge_base.rs`
- `crates/mnemos/src/storage/mod.rs`
- `crates/mnemos/src/grpc/server.rs`
- `crates/mnemos/src/grpc/client.rs`
- `crates/mnemos/src/main.rs`
- `crates/mnemos/src/bin/server.rs`
- `crates/mnemos/proto/knowledge.proto`
- `crates/mnemos/src/ingest/`
- `crates/mnemos/src/learning/`
- `README.md`
- `CHANGELOG.md`
- `kb/Architecture/Retrieval.md`
- `/Users/kpernyer/dev/reflective/stack/mosaic-extensions/kb/Modules/Mnemos.md`

## Suggested Child Tickets

1. **Deployment config and readiness**
   - Typed server config.
   - Env/file loading.
   - Readiness/liveness semantics.
   - Operator docs.

2. **Tenant-scoped core API**
   - Tenant ID type.
   - Scoped add/get/search/update/delete/stats.
   - Search scope applied before scoring.
   - Unit tests for tenant isolation.

3. **Tenant-aware storage and migration**
   - Storage layout decision.
   - Backward-compatible single-tenant load.
   - Migration tests.
   - Backup/restore notes.

4. **Tenant-aware gRPC and CLI**
   - Proto fields.
   - Server/client propagation.
   - CLI flags.
   - Integration tests for tenant isolation.

5. **Meta-knowledge model**
   - Type markers and metadata conventions.
   - Helpers for ingestion/retrieval/maintenance meta entries.
   - Default exclusion from ordinary recall.
   - Explicit meta search.

6. **Retrieval trace payload**
   - Candidate trace structure.
   - Vector/BM25/RRF/graph/learning contribution recording.
   - Tests and docs.

## Test Plan

- Unit tests for tenant ID validation and scoped filters.
- Storage round-trip tests for multiple tenants.
- Migration test from a single-tenant store.
- gRPC tests that add/search the same title in two tenants and prove isolation.
- CLI smoke test for `--tenant`.
- Search tests proving tenant and meta filters run before vector, BM25, RRF,
  graph, and learning paths.
- Doc checks through `just doc`.
- Standard repo gates: `just check`, `just check-memory`, `just test`,
  `just lint`.

## Size

Large. Split into the child tickets above before implementation.
