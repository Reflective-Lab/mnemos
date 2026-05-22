//! Knowledge base implementation using ruvector.

use super::{KnowledgeEntry, SearchOptions, SearchResult};
use crate::embedding::EmbeddingEngine;
use crate::error::{Error, Result};
use crate::learning::LearningEngine;
use crate::math::cosine_similarity;
use crate::storage::StorageBackend;

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, instrument};
use uuid::Uuid;

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;
const RRF_K: f32 = 60.0;

/// Configuration for the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseConfig {
    /// Embedding dimension size.
    pub dimensions: usize,

    /// Path to storage file.
    pub storage_path: String,

    /// Enable self-learning features.
    pub learning_enabled: bool,

    /// Learning rate for GNN updates.
    pub learning_rate: f32,

    /// Number of HNSW neighbors (M parameter).
    pub hnsw_m: usize,

    /// HNSW ef_construction parameter.
    pub hnsw_ef_construction: usize,

    /// HNSW ef_search parameter.
    pub hnsw_ef_search: usize,

    /// Batch size for bulk operations.
    pub batch_size: usize,
}

impl Default for KnowledgeBaseConfig {
    fn default() -> Self {
        Self {
            dimensions: 384,
            storage_path: "./knowledge.db".to_string(),
            learning_enabled: true,
            learning_rate: 0.01,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 100,
            batch_size: 1000,
        }
    }
}

impl KnowledgeBaseConfig {
    /// Create config with custom storage path.
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.storage_path = path.into();
        self
    }

    /// Set embedding dimensions.
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dimensions = dims;
        self
    }

    /// Disable learning features.
    pub fn without_learning(mut self) -> Self {
        self.learning_enabled = false;
        self
    }
}

/// A self-learning knowledge base powered by ruvector.
pub struct KnowledgeBase {
    /// Configuration.
    config: KnowledgeBaseConfig,

    /// Storage backend for persistence.
    storage: Arc<StorageBackend>,

    /// Embedding engine for text vectorization.
    embeddings: Arc<EmbeddingEngine>,

    /// Learning engine for self-improvement.
    learning: Option<Arc<RwLock<LearningEngine>>>,

    /// In-memory entry cache (id -> entry).
    entries: DashMap<Uuid, KnowledgeEntry>,

    /// Vector index (id -> embedding).
    vectors: DashMap<Uuid, Vec<f32>>,

    /// Entry count.
    count: Arc<RwLock<usize>>,
}

impl KnowledgeBase {
    /// Open or create a knowledge base at the given path.
    #[instrument(skip_all)]
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let config = KnowledgeBaseConfig::default().with_path(path.as_ref().to_string_lossy());
        Self::with_config(config).await
    }

    /// Create a knowledge base with custom configuration.
    #[instrument(skip_all, fields(path = %config.storage_path))]
    pub async fn with_config(config: KnowledgeBaseConfig) -> Result<Self> {
        info!("Initializing knowledge base at {}", config.storage_path);

        let storage = Arc::new(StorageBackend::open(&config.storage_path).await?);
        let embeddings = Arc::new(EmbeddingEngine::new(config.dimensions));

        let learning = if config.learning_enabled {
            Some(Arc::new(RwLock::new(LearningEngine::new(
                config.dimensions,
                config.learning_rate,
            ))))
        } else {
            None
        };

        let kb = Self {
            config,
            storage,
            embeddings,
            learning,
            entries: DashMap::new(),
            vectors: DashMap::new(),
            count: Arc::new(RwLock::new(0)),
        };

        // Load existing entries from storage
        kb.load_entries().await?;

        info!("Knowledge base initialized with {} entries", kb.len());
        Ok(kb)
    }

    /// Load entries from storage.
    async fn load_entries(&self) -> Result<()> {
        let stored = self.storage.load_all().await?;

        for (entry, embedding) in stored {
            self.entries.insert(entry.id, entry.clone());
            self.vectors.insert(entry.id, embedding);
        }

        *self.count.write() = self.entries.len();
        Ok(())
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        *self.count.read()
    }

    /// Check if the knowledge base is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get configuration.
    pub fn config(&self) -> &KnowledgeBaseConfig {
        &self.config
    }

    /// Add a new knowledge entry.
    #[instrument(skip(self, entry), fields(title = %entry.title))]
    pub async fn add_entry(&self, entry: KnowledgeEntry) -> Result<Uuid> {
        let id = entry.id;

        // Generate embedding from content
        let text = entry.embedding_text();
        let embedding = self.embeddings.embed(&text).await?;

        // Store in memory
        self.entries.insert(id, entry.clone());
        self.vectors.insert(id, embedding.clone());

        // Persist to storage
        self.storage.save_entry(&entry, &embedding).await?;

        *self.count.write() += 1;
        debug!("Added entry {}", id);

        Ok(id)
    }

    /// Add multiple entries in batch.
    #[instrument(skip(self, entries), fields(count = entries.len()))]
    pub async fn add_entries(&self, entries: Vec<KnowledgeEntry>) -> Result<Vec<Uuid>> {
        let mut ids = Vec::with_capacity(entries.len());

        for chunk in entries.chunks(self.config.batch_size) {
            let mut batch = Vec::with_capacity(chunk.len());
            for entry in chunk {
                let text = entry.embedding_text();
                let embedding = self.embeddings.embed(&text).await?;
                batch.push((entry.clone(), embedding));
            }

            for (entry, embedding) in &batch {
                self.entries.insert(entry.id, entry.clone());
                self.vectors.insert(entry.id, embedding.clone());
                ids.push(entry.id);
            }

            self.storage.save_batch(&batch).await?;
        }

        *self.count.write() += ids.len();
        info!("Added {} entries in batch", ids.len());

        Ok(ids)
    }

    /// Get an entry by ID.
    pub fn get(&self, id: Uuid) -> Option<KnowledgeEntry> {
        self.entries.get(&id).map(|e| e.clone())
    }

    /// Update an existing entry.
    #[instrument(skip(self, entry), fields(id = %entry.id))]
    pub async fn update_entry(&self, entry: KnowledgeEntry) -> Result<()> {
        let id = entry.id;

        if !self.entries.contains_key(&id) {
            return Err(Error::not_found(id.to_string()));
        }

        // Regenerate embedding
        let text = entry.embedding_text();
        let embedding = self.embeddings.embed(&text).await?;

        // Update in memory
        self.entries.insert(id, entry.clone());
        self.vectors.insert(id, embedding.clone());

        // Persist
        self.storage.save_entry(&entry, &embedding).await?;

        debug!("Updated entry {}", id);
        Ok(())
    }

    /// Delete an entry.
    #[instrument(skip(self), fields(id = %id))]
    pub async fn delete_entry(&self, id: Uuid) -> Result<()> {
        if self.entries.remove(&id).is_none() {
            return Err(Error::not_found(id.to_string()));
        }

        self.vectors.remove(&id);
        self.storage.delete_entry(id).await?;

        *self.count.write() -= 1;
        debug!("Deleted entry {}", id);

        Ok(())
    }

    /// Search the knowledge base.
    #[instrument(skip(self), fields(k = options.limit))]
    pub async fn search(&self, query: &str, options: SearchOptions) -> Result<Vec<SearchResult>> {
        if options.limit == 0 {
            return Ok(Vec::new());
        }

        // Generate query embedding
        let query_embedding = self.embeddings.embed(query).await?;
        let scoped_entries = self.filtered_entries(&options);

        // Find similar vectors using brute force for now, after scope filters.
        // (ruvector HNSW would be used in production)
        let mut candidates = self.vector_candidates(&query_embedding, &scoped_entries);

        // Sort by distance (ascending)
        sort_candidates_by_distance(&mut candidates);

        // Apply learning-based re-ranking if enabled
        if options.use_learning
            && let Some(learning) = &self.learning
        {
            let learning = learning.read();
            candidates = learning.rerank(&query_embedding, candidates, &self.vectors);
            sort_candidates_by_distance(&mut candidates);
        }

        let mut results = if options.hybrid {
            self.hybrid_results(query, &scoped_entries, &candidates, &options)
        } else {
            self.vector_results(&candidates, &options)
        };

        // Apply MMR diversity if requested
        if options.diversity > 0.0 {
            results = apply_mmr(results, options.diversity);
        }

        // Record query for learning
        if let Some(learning) = &self.learning {
            let mut learning = learning.write();
            learning.record_query(&query_embedding, &results);
        }

        debug!("Search returned {} results", results.len());
        Ok(results)
    }

    fn filtered_entries(&self, options: &SearchOptions) -> Vec<(Uuid, KnowledgeEntry)> {
        self.entries
            .iter()
            .filter_map(|entry| {
                if entry_matches_options(entry.value(), options) {
                    Some((*entry.key(), entry.value().clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    fn vector_candidates(
        &self,
        query_embedding: &[f32],
        entries: &[(Uuid, KnowledgeEntry)],
    ) -> Vec<(Uuid, f32)> {
        entries
            .iter()
            .filter_map(|(id, _)| {
                self.vectors
                    .get(id)
                    .map(|vector| (*id, cosine_distance(query_embedding, &vector)))
            })
            .collect()
    }

    fn vector_results(
        &self,
        candidates: &[(Uuid, f32)],
        options: &SearchOptions,
    ) -> Vec<SearchResult> {
        let mut results = Vec::new();

        for (id, distance) in candidates {
            let similarity = 1.0 - distance;
            if similarity < options.min_similarity {
                continue;
            }

            if let Some(entry) = self.entries.get(id) {
                results.push(SearchResult::new(entry.clone(), similarity, *distance));
            }

            if results.len() >= options.limit {
                break;
            }
        }

        results
    }

    fn hybrid_results(
        &self,
        query: &str,
        entries: &[(Uuid, KnowledgeEntry)],
        candidates: &[(Uuid, f32)],
        options: &SearchOptions,
    ) -> Vec<SearchResult> {
        let vector_rank: Vec<Uuid> = candidates.iter().map(|(id, _)| *id).collect();
        let lexical_docs: Vec<(Uuid, String)> = entries
            .iter()
            .filter(|(id, _)| self.vectors.contains_key(id))
            .map(|(id, entry)| (*id, entry.embedding_text()))
            .collect();
        let lexical_rank: Vec<Uuid> = bm25_rank(query, &lexical_docs)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let fused = reciprocal_rank_fusion(&[vector_rank, lexical_rank], RRF_K);
        let distances: HashMap<Uuid, f32> = candidates.iter().copied().collect();
        let mut results = Vec::new();

        for (id, rrf_score) in fused {
            let Some(distance) = distances.get(&id).copied() else {
                continue;
            };
            let similarity = 1.0 - distance;
            if similarity < options.min_similarity {
                continue;
            }

            if let Some(entry) = self.entries.get(&id) {
                results.push(SearchResult::with_rank_score(
                    entry.clone(),
                    similarity,
                    distance,
                    rrf_score,
                ));
            }

            if results.len() >= options.limit {
                break;
            }
        }

        results
    }

    /// Simple search with default options.
    pub async fn search_simple(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search(query, SearchOptions::new(limit)).await
    }

    /// Record user feedback on a search result.
    #[instrument(skip(self))]
    pub async fn record_feedback(&self, entry_id: Uuid, positive: bool) -> Result<()> {
        if let Some(mut entry) = self.entries.get_mut(&entry_id) {
            let boost = if positive { 0.1 } else { -0.05 };
            entry.record_access(1.0 + boost);

            // Update learning engine
            if let Some(learning) = &self.learning {
                let mut learning = learning.write();
                if let Some(embedding) = self.vectors.get(&entry_id) {
                    learning.record_feedback(&embedding, positive);
                }
            }

            // Persist updated entry
            let entry = entry.clone();
            if let Some(embedding) = self.vectors.get(&entry_id) {
                self.storage.save_entry(&entry, &embedding).await?;
            }
        }

        Ok(())
    }

    /// Get entries related to a given entry.
    pub fn get_related(&self, id: Uuid, limit: usize) -> Vec<KnowledgeEntry> {
        if let Some(entry) = self.entries.get(&id) {
            entry
                .related_entries
                .iter()
                .take(limit)
                .filter_map(|rel_id| self.entries.get(rel_id).map(|e| e.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Link two entries as related.
    #[allow(clippy::unused_async)]
    pub async fn link_entries(&self, id1: Uuid, id2: Uuid) -> Result<()> {
        if let Some(mut entry1) = self.entries.get_mut(&id1) {
            if !entry1.related_entries.contains(&id2) {
                entry1.related_entries.push(id2);
            }
        } else {
            return Err(Error::not_found(id1.to_string()));
        }

        if let Some(mut entry2) = self.entries.get_mut(&id2)
            && !entry2.related_entries.contains(&id1)
        {
            entry2.related_entries.push(id1);
        }

        Ok(())
    }

    /// Get all entries (for export/backup).
    pub fn all_entries(&self) -> Vec<KnowledgeEntry> {
        self.entries.iter().map(|e| e.value().clone()).collect()
    }

    /// Get statistics about the knowledge base.
    pub fn stats(&self) -> KnowledgeBaseStats {
        let total = self.len();
        let categories: std::collections::HashSet<_> = self
            .entries
            .iter()
            .filter_map(|e| e.category.clone())
            .collect();

        let tags: std::collections::HashSet<_> =
            self.entries.iter().flat_map(|e| e.tags.clone()).collect();

        let total_access: u64 = self.entries.iter().map(|e| e.access_count).sum();

        KnowledgeBaseStats {
            total_entries: total,
            unique_categories: categories.len(),
            unique_tags: tags.len(),
            total_access_count: total_access,
            dimensions: self.config.dimensions,
            learning_enabled: self.config.learning_enabled,
        }
    }

    /// Flush all pending writes to storage.
    pub async fn flush(&self) -> Result<()> {
        self.storage.flush().await
    }
}

/// Statistics about the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseStats {
    pub total_entries: usize,
    pub unique_categories: usize,
    pub unique_tags: usize,
    pub total_access_count: u64,
    pub dimensions: usize,
    pub learning_enabled: bool,
}

/// Calculate cosine distance between two vectors.
///
/// Returns `1.0 - cosine_similarity(a, b)`.  A zero-norm vector yields a
/// distance of `1.0` (maximally distant), consistent with the previous
/// behaviour.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

fn entry_matches_options(entry: &KnowledgeEntry, options: &SearchOptions) -> bool {
    if let Some(ref category) = options.category
        && entry.category.as_ref() != Some(category)
    {
        return false;
    }

    if !options.tags.is_empty()
        && !options
            .tags
            .iter()
            .any(|tag| entry.tags.iter().any(|entry_tag| entry_tag == tag))
    {
        return false;
    }

    true
}

fn sort_candidates_by_distance(candidates: &mut [(Uuid, f32)]) {
    candidates.sort_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.0.as_bytes().cmp(b.0.as_bytes()))
    });
}

fn lexical_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() || matches!(ch, '_' | '-' | '+' | '#') {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn bm25_rank(query: &str, docs: &[(Uuid, String)]) -> Vec<(Uuid, f32)> {
    let query_terms: HashSet<String> = lexical_tokens(query).into_iter().collect();
    if query_terms.is_empty() || docs.is_empty() {
        return Vec::new();
    }

    let doc_tokens: Vec<Vec<String>> = docs.iter().map(|(_, text)| lexical_tokens(text)).collect();
    let total_len: usize = doc_tokens.iter().map(Vec::len).sum();
    if total_len == 0 {
        return Vec::new();
    }

    let doc_count = docs.len() as f32;
    let avg_doc_len = total_len as f32 / doc_count;
    let mut doc_frequency: HashMap<String, usize> = HashMap::new();

    for tokens in &doc_tokens {
        let mut seen = HashSet::new();
        for token in tokens {
            if seen.insert(token) {
                *doc_frequency.entry(token.clone()).or_default() += 1;
            }
        }
    }

    let mut ranked = Vec::new();

    for ((id, _), tokens) in docs.iter().zip(doc_tokens.iter()) {
        if tokens.is_empty() {
            continue;
        }

        let mut term_frequency: HashMap<&str, usize> = HashMap::new();
        for token in tokens {
            *term_frequency.entry(token.as_str()).or_default() += 1;
        }

        let doc_len = tokens.len() as f32;
        let mut score = 0.0f32;

        for term in &query_terms {
            let Some(df) = doc_frequency.get(term) else {
                continue;
            };
            let tf = term_frequency.get(term.as_str()).copied().unwrap_or(0) as f32;
            if tf == 0.0 {
                continue;
            }

            let df = *df as f32;
            let idf = ((doc_count - df + 0.5) / (df + 0.5) + 1.0).ln();
            let denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * doc_len / avg_doc_len);
            score += idf * (tf * (BM25_K1 + 1.0)) / denominator;
        }

        if score > 0.0 {
            ranked.push((*id, score));
        }
    }

    ranked.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| a.0.as_bytes().cmp(b.0.as_bytes()))
    });
    ranked
}

fn reciprocal_rank_fusion(rankings: &[Vec<Uuid>], rrf_k: f32) -> Vec<(Uuid, f32)> {
    let mut scores: HashMap<Uuid, f32> = HashMap::new();

    for ranking in rankings {
        for (rank_index, id) in ranking.iter().enumerate() {
            let rank = rank_index as f32 + 1.0;
            *scores.entry(*id).or_default() += 1.0 / (rrf_k + rank);
        }
    }

    let mut fused: Vec<_> = scores.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| a.0.as_bytes().cmp(b.0.as_bytes()))
    });
    fused
}

/// Apply Maximal Marginal Relevance for diversity.
fn apply_mmr(mut results: Vec<SearchResult>, lambda: f32) -> Vec<SearchResult> {
    if results.len() <= 1 {
        return results;
    }

    let mut selected = vec![results.remove(0)];

    while !results.is_empty() && selected.len() < results.len() + selected.len() {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;

        for (i, candidate) in results.iter().enumerate() {
            // Relevance term
            let relevance = candidate.similarity;

            // Diversity term: max similarity to already selected
            let max_sim = selected
                .iter()
                .map(|s| {
                    // Simplified: use score similarity
                    1.0 - (s.score - candidate.score).abs()
                })
                .max_by(f32::total_cmp)
                .unwrap_or(0.0);

            // MMR score
            let mmr = lambda * relevance - (1.0 - lambda) * max_sim;

            if mmr > best_score {
                best_score = mmr;
                best_idx = i;
            }
        }

        selected.push(results.remove(best_idx));
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::KnowledgeEntry;
    use tempfile::tempdir;

    fn small_config(path: &Path) -> KnowledgeBaseConfig {
        KnowledgeBaseConfig::default()
            .with_path(path.to_string_lossy())
            .with_dimensions(32)
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_distance(&a, &c) - 1.0).abs() < 1e-6);

        // Zero-norm path returns 1.0 (max distance).
        let z = vec![0.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &z) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn lexical_tokens_preserve_exact_retrieval_terms() {
        let tokens = lexical_tokens("BM25, RRF, C++, C#, memory-only, and IDF.");

        assert!(tokens.iter().any(|token| token == "bm25"));
        assert!(tokens.iter().any(|token| token == "rrf"));
        assert!(tokens.iter().any(|token| token == "c++"));
        assert!(tokens.iter().any(|token| token == "c#"));
        assert!(tokens.iter().any(|token| token == "memory-only"));
        assert!(tokens.iter().any(|token| token == "idf"));
    }

    #[test]
    fn bm25_rewards_rare_exact_terms() {
        let exact = Uuid::new_v4();
        let semantic = Uuid::new_v4();
        let hybrid = Uuid::new_v4();
        let docs = vec![
            (
                exact,
                "BM25 scores documents with term frequency and IDF.".to_string(),
            ),
            (
                semantic,
                "Vector embeddings capture broad semantic similarity.".to_string(),
            ),
            (
                hybrid,
                "Hybrid recall combines several retrieval ranks.".to_string(),
            ),
        ];

        let ranked = bm25_rank("BM25 IDF", &docs);

        assert_eq!(ranked.first().map(|(id, _)| *id), Some(exact));
        assert!(ranked.first().is_some_and(|(_, score)| *score > 0.0));
    }

    #[test]
    fn reciprocal_rank_fusion_rewards_shared_high_rank() {
        let shared = Uuid::new_v4();
        let lexical_only = Uuid::new_v4();
        let vector_only = Uuid::new_v4();

        let fused = reciprocal_rank_fusion(
            &[vec![shared, vector_only], vec![lexical_only, shared]],
            RRF_K,
        );

        let shared_score = fused
            .iter()
            .find(|(id, _)| *id == shared)
            .map(|(_, score)| *score)
            .unwrap();
        let vector_only_score = fused
            .iter()
            .find(|(id, _)| *id == vector_only)
            .map(|(_, score)| *score)
            .unwrap();

        assert_eq!(fused.first().map(|(id, _)| *id), Some(shared));
        assert!(shared_score > vector_only_score);
    }

    #[test]
    fn config_builder_sets_fields() {
        let cfg = KnowledgeBaseConfig::default()
            .with_path("/tmp/x.db")
            .with_dimensions(64)
            .without_learning();
        assert_eq!(cfg.storage_path, "/tmp/x.db");
        assert_eq!(cfg.dimensions, 64);
        assert!(!cfg.learning_enabled);
    }

    #[tokio::test]
    async fn open_creates_empty_kb() {
        let dir = tempdir().unwrap();
        let kb = KnowledgeBase::open(dir.path().join("kb.db")).await.unwrap();
        assert_eq!(kb.len(), 0);
        assert!(kb.is_empty());
        assert_eq!(kb.config().dimensions, 384);
    }

    #[tokio::test]
    async fn add_get_update_delete_roundtrip() {
        let dir = tempdir().unwrap();
        let kb = KnowledgeBase::with_config(small_config(&dir.path().join("kb.db")))
            .await
            .unwrap();

        let entry = KnowledgeEntry::new("Title", "body text").with_category("docs");
        let id = kb.add_entry(entry.clone()).await.unwrap();
        assert_eq!(kb.len(), 1);
        assert!(!kb.is_empty());

        let fetched = kb.get(id).expect("entry should exist");
        assert_eq!(fetched.title, "Title");

        let mut updated = fetched;
        updated.content = "new body".into();
        kb.update_entry(updated.clone()).await.unwrap();
        assert_eq!(kb.get(id).unwrap().content, "new body");

        kb.delete_entry(id).await.unwrap();
        assert_eq!(kb.len(), 0);
        assert!(kb.get(id).is_none());
    }

    #[tokio::test]
    async fn update_missing_entry_errors() {
        let dir = tempdir().unwrap();
        let kb = KnowledgeBase::with_config(small_config(&dir.path().join("kb.db")))
            .await
            .unwrap();
        let stranger = KnowledgeEntry::new("ghost", "body");
        let err = kb.update_entry(stranger).await.unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));
    }

    #[tokio::test]
    async fn delete_missing_entry_errors() {
        let dir = tempdir().unwrap();
        let kb = KnowledgeBase::with_config(small_config(&dir.path().join("kb.db")))
            .await
            .unwrap();
        let err = kb.delete_entry(Uuid::new_v4()).await.unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));
    }

    #[tokio::test]
    async fn add_entries_batch_persists() {
        let dir = tempdir().unwrap();
        let kb = KnowledgeBase::with_config(small_config(&dir.path().join("kb.db")))
            .await
            .unwrap();
        let batch: Vec<_> = (0..5)
            .map(|i| KnowledgeEntry::new(format!("t{i}"), format!("body {i}")))
            .collect();
        let ids = kb.add_entries(batch).await.unwrap();
        assert_eq!(ids.len(), 5);
        assert_eq!(kb.len(), 5);
        kb.flush().await.unwrap();
    }

    #[tokio::test]
    async fn search_filters_and_results() {
        let dir = tempdir().unwrap();
        // Larger dims so hash-embedder collisions don't make small-corpus
        // searches flaky.
        let cfg = KnowledgeBaseConfig::default()
            .with_path(dir.path().join("kb.db").to_string_lossy())
            .with_dimensions(128);
        let kb = KnowledgeBase::with_config(cfg).await.unwrap();
        kb.add_entry(
            KnowledgeEntry::new("rust ownership", "borrow checker introduction")
                .with_category("rust")
                .with_tags(["ownership"]),
        )
        .await
        .unwrap();
        kb.add_entry(
            KnowledgeEntry::new("python decorators", "functions wrapping functions")
                .with_category("python")
                .with_tags(["meta"]),
        )
        .await
        .unwrap();

        // search_simple returns Ok (results may be empty if hash embedding
        // has no positive overlap; we only assert the call path succeeds).
        let _ = kb.search_simple("borrow", 10).await.unwrap();

        // Category filter — only rust-categorised entries (or none).
        let only_rust = kb
            .search(
                "wrapping",
                SearchOptions::new(10)
                    .with_category("rust")
                    .without_learning(),
            )
            .await
            .unwrap();
        for r in &only_rust {
            assert_eq!(r.entry.category.as_deref(), Some("rust"));
        }

        // Tag filter — every result must carry the requested tag.
        let by_tag = kb
            .search("anything", SearchOptions::new(10).with_tags(["ownership"]))
            .await
            .unwrap();
        for r in &by_tag {
            assert!(r.entry.tags.iter().any(|t| t == "ownership"));
        }

        // Diversity branch — exercises apply_mmr.
        let _ = kb
            .search("functions", SearchOptions::new(5).with_diversity(0.5))
            .await
            .unwrap();

        // min_similarity above the achievable maximum filters everything out.
        let none = kb
            .search("borrow", SearchOptions::new(10).with_min_similarity(1.0))
            .await
            .unwrap();
        assert!(none.is_empty());
    }

    #[tokio::test]
    async fn search_zero_limit_returns_empty() {
        let dir = tempdir().unwrap();
        let kb = KnowledgeBase::with_config(small_config(&dir.path().join("kb.db")))
            .await
            .unwrap();
        kb.add_entry(KnowledgeEntry::new("BM25", "RRF hybrid retrieval"))
            .await
            .unwrap();

        let vector = kb
            .search("BM25", SearchOptions::new(0).without_learning())
            .await
            .unwrap();
        let hybrid = kb
            .search("BM25", SearchOptions::new(0).hybrid(0.3).without_learning())
            .await
            .unwrap();

        assert!(vector.is_empty());
        assert!(hybrid.is_empty());
    }

    #[tokio::test]
    async fn hybrid_search_uses_bm25_and_rrf() {
        let dir = tempdir().unwrap();
        let cfg = KnowledgeBaseConfig::default()
            .with_path(dir.path().join("kb.db").to_string_lossy())
            .with_dimensions(128)
            .without_learning();
        let kb = KnowledgeBase::with_config(cfg).await.unwrap();

        kb.add_entry(KnowledgeEntry::new(
            "semantic search",
            "Vector embeddings capture conceptual similarity and paraphrase recall.",
        ))
        .await
        .unwrap();
        kb.add_entry(KnowledgeEntry::new(
            "reciprocal rank fusion",
            "RRF combines BM25 lexical ranks with vector semantic ranks.",
        ))
        .await
        .unwrap();
        kb.add_entry(KnowledgeEntry::new(
            "storage backend",
            "Persistence keeps knowledge entries durable across restarts.",
        ))
        .await
        .unwrap();

        let results = kb
            .search("RRF", SearchOptions::new(3).hybrid(0.3).without_learning())
            .await
            .unwrap();

        assert_eq!(
            results.first().map(|result| result.entry.title.as_str()),
            Some("reciprocal rank fusion")
        );
        assert!(results.first().is_some_and(|result| result.score > 0.0));
    }

    #[tokio::test]
    async fn record_feedback_and_stats() {
        let dir = tempdir().unwrap();
        let kb = KnowledgeBase::with_config(small_config(&dir.path().join("kb.db")))
            .await
            .unwrap();
        let id = kb
            .add_entry(
                KnowledgeEntry::new("a", "alpha")
                    .with_category("c")
                    .with_tags(["t"]),
            )
            .await
            .unwrap();
        kb.record_feedback(id, true).await.unwrap();
        kb.record_feedback(id, false).await.unwrap();
        kb.record_feedback(Uuid::new_v4(), true).await.unwrap(); // unknown id is ok

        let stats = kb.stats();
        assert_eq!(stats.total_entries, 1);
        assert_eq!(stats.unique_categories, 1);
        assert_eq!(stats.unique_tags, 1);
        assert!(stats.learning_enabled);
        assert_eq!(stats.dimensions, 32);
        assert!(stats.total_access_count >= 2);
    }

    #[tokio::test]
    async fn linking_and_related() {
        let dir = tempdir().unwrap();
        let kb = KnowledgeBase::with_config(small_config(&dir.path().join("kb.db")))
            .await
            .unwrap();
        let a = kb.add_entry(KnowledgeEntry::new("a", "x")).await.unwrap();
        let b = kb.add_entry(KnowledgeEntry::new("b", "y")).await.unwrap();

        kb.link_entries(a, b).await.unwrap();
        // idempotent
        kb.link_entries(a, b).await.unwrap();

        let related = kb.get_related(a, 5);
        assert_eq!(related.len(), 1);
        assert_eq!(related[0].id, b);

        // Unknown source id errors.
        let err = kb.link_entries(Uuid::new_v4(), b).await.unwrap_err();
        assert!(matches!(err, Error::NotFound(_)));

        // get_related on unknown id returns empty.
        assert!(kb.get_related(Uuid::new_v4(), 5).is_empty());

        // all_entries surfaces every entry.
        assert_eq!(kb.all_entries().len(), 2);
    }

    #[tokio::test]
    async fn reopens_with_existing_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("kb.db");
        let kb = KnowledgeBase::with_config(small_config(&path))
            .await
            .unwrap();
        kb.add_entry(KnowledgeEntry::new("persist", "me"))
            .await
            .unwrap();
        kb.flush().await.unwrap();
        drop(kb);

        let kb2 = KnowledgeBase::with_config(small_config(&path))
            .await
            .unwrap();
        assert_eq!(kb2.len(), 1);
        assert_eq!(kb2.all_entries()[0].title, "persist");
    }

    #[tokio::test]
    async fn learning_disabled_skips_engine() {
        let dir = tempdir().unwrap();
        let cfg = small_config(&dir.path().join("kb.db")).without_learning();
        let kb = KnowledgeBase::with_config(cfg).await.unwrap();
        let id = kb.add_entry(KnowledgeEntry::new("t", "c")).await.unwrap();
        // Search and feedback both no-op the learning branch.
        let _ = kb.search_simple("t", 5).await.unwrap();
        kb.record_feedback(id, true).await.unwrap();
        assert!(!kb.stats().learning_enabled);
    }

    #[test]
    fn mmr_short_circuits_short_lists() {
        let entry = KnowledgeEntry::new("t", "c");
        let r = SearchResult::new(entry, 0.5, 0.5);
        let one = apply_mmr(vec![r.clone()], 0.5);
        assert_eq!(one.len(), 1);
        let empty: Vec<SearchResult> = apply_mmr(Vec::new(), 0.5);
        assert!(empty.is_empty());

        // Multiple results pass through MMR selection loop.
        let mut many = Vec::new();
        for i in 0..3 {
            let e = KnowledgeEntry::new(format!("t{i}"), "c");
            many.push(SearchResult::new(e, 0.9 - i as f32 * 0.1, 0.1 * i as f32));
        }
        let picked = apply_mmr(many, 0.7);
        assert!(!picked.is_empty());
    }
}
