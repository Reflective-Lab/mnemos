---
tags: [architecture, surface]
source: mixed
---
# Surface

`mnemos` exposes one canonical published crate (`converge-mnemos-knowledge`)
whose Rust library name is `mnemos`.

## Public surface

- `mnemos` — knowledge storage, retrieval, ingestion, learning, and Converge
  recall suggestors.
- `ProvenanceSource` and `MNEMOS_PROVENANCE` for typed proposal provenance
  before crossing into `converge-pack::ProposedFact`.
- `mnemos.suggestor.execute` tracing spans on knowledge suggestor execution.

## Contract dependencies

- `converge-pack` — `Pack`, `ProposedFact`, `ProposedPlan`, `ProblemSpec`
- `converge-model` — semantic types
- `converge-provider` — capability identity (when applicable)

## Forbidden imports

Per [Extension Release Checklist §1](https://github.com/Reflective-Lab/converge/blob/main/kb/Standards/Extension%20Release%20Checklist.md):

- No imports of `converge-core` internals.
- No imports of foundation `runtime`, `provider`, or transport crates.
- No re-exports of foundation types except those promised stable.
