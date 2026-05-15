//! Knowledge Suggestors — make the knowledge base participate in convergence.
//!
//! Two agents:
//! - [`KnowledgeRetrievalSuggestor`] — searches KB, proposes relevant knowledge
//! - [`KnowledgeStoreSuggestor`] — stores convergence results in KB for future use

use std::sync::Arc;

use async_trait::async_trait;
use converge_pack::{
    AgentEffect, Context, ContextFact, ContextKey, FactPayload, ProvenanceSource, Suggestor,
    TextPayload,
};
use serde::{Deserialize, Serialize};
use tracing::Instrument;

use crate::core::{KnowledgeBase, KnowledgeEntry, SearchOptions};
use crate::provenance::{MNEMOS_PROVENANCE, suggestor_span};

/// A typed knowledge-search hit proposed by Mnemos retrieval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeHitPayload {
    /// Source system that produced the hit.
    pub source: String,
    /// Query text used for retrieval.
    pub query: String,
    /// Knowledge entry title.
    pub title: String,
    /// Knowledge entry body.
    pub content: String,
    /// Retrieval score in the search backend's normalized range.
    pub score: f32,
}

impl FactPayload for KnowledgeHitPayload {
    const FAMILY: &'static str = "mnemos.knowledge.hit";
    const VERSION: u16 = 1;
}

/// Searches the knowledge base for information relevant to the current context.
///
/// Reads queries from Seeds, searches the KB, and proposes relevant
/// knowledge as Hypotheses for other agents to build on.
pub struct KnowledgeRetrievalSuggestor {
    kb: Arc<KnowledgeBase>,
    max_results: usize,
}

impl KnowledgeRetrievalSuggestor {
    /// Create a retrieval suggestor backed by the given knowledge base.
    pub fn new(kb: Arc<KnowledgeBase>) -> Self {
        Self { kb, max_results: 5 }
    }

    /// Override the maximum number of search hits proposed per seed fact.
    pub fn with_max_results(mut self, n: usize) -> Self {
        self.max_results = n;
        self
    }
}

#[async_trait]
impl Suggestor for KnowledgeRetrievalSuggestor {
    fn name(&self) -> &'static str {
        "knowledge-retrieval"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Seeds]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        ctx.has(ContextKey::Seeds) && !ctx.has(ContextKey::Hypotheses)
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        let span = suggestor_span(
            "knowledge-retrieval",
            ContextKey::Seeds,
            ContextKey::Hypotheses,
            ctx.count(ContextKey::Seeds),
        );

        async move {
            let seeds = ctx.get(ContextKey::Seeds);
            let mut proposals = Vec::new();

            for seed in seeds {
                let Some(query) = seed.text() else {
                    continue;
                };
                let options = SearchOptions {
                    limit: self.max_results,
                    ..SearchOptions::default()
                };

                if let Ok(results) = self.kb.search(query, options).await {
                    for (i, result) in results.into_iter().enumerate() {
                        let payload = KnowledgeHitPayload {
                            source: "knowledge-base".to_string(),
                            query: query.to_string(),
                            title: result.entry.title,
                            content: result.entry.content,
                            score: result.score,
                        };
                        proposals.push(
                            MNEMOS_PROVENANCE
                                .proposed_fact(
                                    ContextKey::Hypotheses,
                                    format!("kb-{}-{}", seed.id(), i),
                                    payload,
                                )
                                .with_confidence(f64::from(result.score)),
                        );
                    }
                }
            }

            AgentEffect::with_proposals(proposals)
        }
        .instrument(span)
        .await
    }
}

/// Stores convergence results in the knowledge base for future retrieval.
///
/// Reads promoted strategies and evaluations, stores them as knowledge
/// entries so future formations can benefit from past convergence.
pub struct KnowledgeStoreSuggestor {
    kb: Arc<KnowledgeBase>,
}

impl KnowledgeStoreSuggestor {
    /// Create a storage suggestor backed by the given knowledge base.
    pub fn new(kb: Arc<KnowledgeBase>) -> Self {
        Self { kb }
    }
}

#[async_trait]
impl Suggestor for KnowledgeStoreSuggestor {
    fn name(&self) -> &'static str {
        "knowledge-store"
    }

    fn dependencies(&self) -> &[ContextKey] {
        &[ContextKey::Evaluations]
    }

    fn accepts(&self, ctx: &dyn Context) -> bool {
        ctx.has(ContextKey::Evaluations)
            && !ctx
                .get(ContextKey::Seeds)
                .iter()
                .any(|f| f.id().starts_with("stored-"))
    }

    async fn execute(&self, ctx: &dyn Context) -> AgentEffect {
        let span = suggestor_span(
            "knowledge-store",
            ContextKey::Evaluations,
            ContextKey::Seeds,
            ctx.count(ContextKey::Evaluations),
        );

        async move {
            let evaluations = ctx.get(ContextKey::Evaluations);
            let mut proposals = Vec::new();

            for eval in evaluations {
                let Some(content) = fact_content_for_store(eval) else {
                    continue;
                };
                let entry = KnowledgeEntry::new(eval.id().as_str(), content)
                    .with_category("convergence-result")
                    .with_tags(vec!["auto-stored", "formation-output"]);

                if self.kb.add_entry(entry).await.is_ok() {
                    proposals.push(MNEMOS_PROVENANCE.proposed_fact(
                        ContextKey::Seeds,
                        format!("stored-{}", eval.id()),
                        TextPayload::new(format!(
                            "stored evaluation {} in knowledge base",
                            eval.id()
                        )),
                    ));
                }
            }

            AgentEffect::with_proposals(proposals)
        }
        .instrument(span)
        .await
    }
}

fn fact_content_for_store(fact: &ContextFact) -> Option<String> {
    fact.text().map(str::to_string).or_else(|| {
        fact.to_wire()
            .ok()
            .map(|wire| wire.payload.payload.to_string())
    })
}
