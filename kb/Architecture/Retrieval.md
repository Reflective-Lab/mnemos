---
tags: [architecture, retrieval, memory]
source: llm
---
# Retrieval

Mnemos supports vector retrieval and opt-in hybrid retrieval.

## Hybrid Baseline

Hybrid retrieval is the baseline to use when exact terms and semantic similarity
both matter:

- Vector search catches paraphrase and semantic similarity.
- BM25 catches acronyms, exact identifiers, rare terms, and keyword-heavy
  queries.
- Reciprocal Rank Fusion merges ranks without comparing raw BM25 and vector
  scores.

The fusion score is:

```text
score = sum(1 / (60 + rank))
```

Ranks are 1-based within each retriever. A document that ranks high in either
BM25 or vector retrieval receives a strong fused score; a document that ranks
high in both receives both contributions.

## Current Implementation

- `SearchOptions::hybrid(...)` enables hybrid retrieval.
- BM25 is computed over filtered in-memory entries using `KnowledgeEntry`
  embedding text.
- Vector ranks still use normalized embeddings and cosine distance.
- RRF orders the final candidate set; learned relevance can still affect the
  returned `SearchResult::score`.
- Category and tag filters are applied before ranking so scoped recall does not
  leak across search boundaries.

## Boundaries

Hybrid retrieval is only recall. It must not bypass Converge promotion or turn
retrieved content into facts directly. Products still decide whether Mnemos runs
embedded, through gRPC, or not at all.

## Prism Fuzzy Scoring

Prism's fuzzy inference capability could help above the retrieval baseline, but
it should not replace BM25, vector ranking, or RRF.

Good uses:

- Explainable reranking over already-retrieved candidates.
- Soft admission rules over signals such as semantic similarity, BM25 rank, RRF
  rank, source trust, recency, learned relevance, and scope match.
- Product-specific recall policy where a rule trace such as "exact term high
  and source trust medium" is useful to inspect.

Poor uses:

- First-pass candidate generation.
- Replacing metadata filters or hard authorization boundaries.
- Adding Prism as a core Mnemos dependency only to tune search weights.

Keep the dependency direction clean: Mnemos owns recall signals and filtered
candidate sets. Prism owns fuzzy inference. A product or formation can compose
both when it needs explainable soft scoring.
