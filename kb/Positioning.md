---
tags: [positioning, pitch, memory, retrieval]
source: llm
date: 2026-06-12
---
# Positioning

Why Mnemos exists, why it plays well with LLMs, and the full capability
catalog. Companion pitches live in the Ferrox, Arbiter, Soter, Prism, and
Crucible knowledge bases; this note is the Mnemos chapter of the same story.

## Elevator Pitch

Mnemos is the **memory of the Converge platform** — named for Mnemosyne, and
shaped by one discipline: *recall is not truth*. It owns knowledge storage,
hybrid retrieval, ingestion, embeddings, and agentic memory (causal,
temporal, reflexion, skill, session, online, and meta-learning modules),
while Converge alone decides what becomes fact. A formation can ask Mnemos
what is already known, store durable observations, and learn from feedback —
and every recall lands as a typed proposal with `MNEMOS_PROVENANCE`, never as
direct promotion.

The retrieval core is deliberately engineered rather than fashionable:
vector search catches paraphrase, BM25 catches acronyms and exact
identifiers, and Reciprocal Rank Fusion (`1 / (60 + rank)`) merges the two
rankings without ever pretending a cosine score and a BM25 score are
comparable numbers. Scope filters apply *before* ranking, so recall never
leaks across category or tenant boundaries.

## Why It Plays Well With LLMs

An LLM is the famous amnesiac genius: brilliant in the moment, frozen at
training time, and unable to remember yesterday. Mnemos is the missing
hippocampus:

- **Durable recall** — what the agent learned, observed, or was told
  persists across sessions and surfaces (`search_simple` to full hybrid
  `SearchOptions`), so context windows stop being the only memory.
- **Hybrid retrieval matters specifically for agents**: an LLM's query may
  paraphrase ("memory safety") or be brutally exact ("BM25 RRF acronym
  recall") — vector-only stacks fail the second case; Mnemos is built for
  both.
- **Agentic memory is more than RAG**: reflexion stores what went wrong and
  why; skill memory accumulates what worked; causal and temporal modules
  keep *when* and *because* queryable — the raw material of an agent that
  improves rather than repeats itself.
- **Governance is the differentiator**: retrieved content arrives as
  evidence with provenance, not as injected gospel — the structural antidote
  to "the vector store said so," with Converge promotion as the gate.

The LLM thinks; Mnemos remembers; Converge decides what is true.

## What It Solves Better Than Anything Else

Mnemos's niche is **governed memory inside the loop, in pure Rust**. Not a
bolt-on vector database: a knowledge base that runs embedded or over gRPC,
with hybrid lexical+semantic recall, scoped filtering ahead of ranking,
feedback-driven relevance learning, and typed Suggestor surfaces
(`KnowledgeRetrievalSuggestor`, `KnowledgeStoreSuggestor`) that plug recall
directly into the Converge proposal path. The whole memory stack carries the
same supply-chain, provenance, and audit discipline as every other
extension — which RAG-stack-of-the-week pipelines do not.

## Capability Catalog

### Retrieval

| Capability | Tagline |
|---|---|
| Vector search (cosine over normalized embeddings) | Finds what you meant, not just what you typed. |
| BM25 lexical ranking | Acronyms, identifiers, and rare terms — the words that must match exactly. |
| Reciprocal Rank Fusion | `1 / (60 + rank)`: merge rankings, never compare raw scores. |
| Scoped category/tag filtering | Applied before ranking — recall cannot leak across boundaries. |
| Learned relevance | Feedback shapes tomorrow's `SearchResult::score`. |

### Embeddings

| Capability | Tagline |
|---|---|
| Hash embeddings | Local, deterministic, key-less — memory with no external dependency. |
| OpenAI embeddings | Higher-fidelity semantics when a product opts in. |

### Agentic memory (`mnemos::agentic`)

| Module | Tagline |
|---|---|
| Causal | Remember *because*, not just *that*. |
| Temporal | When it happened, and what held true at the time. |
| Reflexion | Failures, post-mortems, and the lessons distilled from them. |
| Skills | What worked, accumulated into reusable competence. |
| Sessions | Working memory with a lifetime — context that knows it expires. |
| Online | Learn during the episode, not after it. |
| Meta | Memory about the memory — what is worth remembering at all. |
| Policy | The rules for what gets remembered, recalled, and forgotten. |

### Learning (`mnemos::learning`)

| Capability | Tagline |
|---|---|
| Feedback collection | Every thumbs-up and correction becomes training signal. |
| Replay | Re-run past episodes against the present knowledge base. |
| Batch learning + insight jobs | Offline consolidation — sleep for knowledge bases. |
| GNN module | Graph-shaped learning over how knowledge connects. |

### Ingestion and surfaces

| Capability | Tagline |
|---|---|
| Markdown + rich-media ingestion with routing | From documents to entries without a side pipeline. |
| `KnowledgeRetrievalSuggestor` / `KnowledgeStoreSuggestor` | Recall and remembrance as governed proposals. |
| CLI (`mnemos`) and gRPC (`mnemos-server`) | Embedded library, command line, or network service — products choose. |
| `memory-only` mode | The whole stack, no disk required. |

## Boundaries (One-Line Reminders)

- Mnemos answers: *what do we already know, and how confident is the
  recall?* (evidence with provenance — never fact by itself)
- Prism answers: *what does the data say, closed-form and auditable?*
  (`Observed` / `Argued`)
- Crucible answers: *what does a model fitted to our data predict?* (trained
  opinion with provenance)
- Arbiter answers: *should this concrete request be allowed now?* (`Decided`)
- Ferrox answers: *what is the best feasible plan?* (`Searched`, optimization)
- Soter answers: *can any modeled request violate this invariant?*
  (`Searched`, symbolic)
- Hybrid retrieval is only recall — it must not bypass Converge promotion.
  See [[Architecture/Retrieval]].
