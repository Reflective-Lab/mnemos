//! Knowledge base implementation using ruvector.

use super::{KnowledgeEntry, SearchOptions, SearchResult};
use crate::embedding::EmbeddingEngine;
use crate::error::{Error, Result};
use crate::learning::LearningEngine;
use crate::storage::StorageBackend;

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info, instrument};
use uuid::Uuid;

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
        // Generate query embedding
        let query_embedding = self.embeddings.embed(query).await?;

        // Find similar vectors using brute force for now
        // (ruvector HNSW would be used in production)
        let mut candidates: Vec<(Uuid, f32)> = self
            .vectors
            .iter()
            .map(|entry| {
                let id = *entry.key();
                let distance = cosine_distance(&query_embedding, entry.value());
                (id, distance)
            })
            .collect();

        // Sort by distance (ascending)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Apply learning-based re-ranking if enabled
        if options.use_learning {
            if let Some(learning) = &self.learning {
                let learning = learning.read();
                candidates = learning.rerank(&query_embedding, candidates, &self.vectors);
            }
        }

        // Build results
        let mut results = Vec::new();

        for (id, distance) in candidates.into_iter().take(options.limit * 2) {
            if let Some(entry) = self.entries.get(&id) {
                let entry = entry.clone();

                // Apply filters
                if let Some(ref cat) = options.category {
                    if entry.category.as_ref() != Some(cat) {
                        continue;
                    }
                }

                if !options.tags.is_empty()
                    && !options
                        .tags
                        .iter()
                        .any(|t| entry.tags.iter().any(|et| et == t))
                {
                    continue;
                }

                let similarity = 1.0 - distance;
                if similarity < options.min_similarity {
                    continue;
                }

                results.push(SearchResult::new(entry, similarity, distance));

                if results.len() >= options.limit {
                    break;
                }
            }
        }

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
    pub async fn link_entries(&self, id1: Uuid, id2: Uuid) -> Result<()> {
        if let Some(mut entry1) = self.entries.get_mut(&id1) {
            if !entry1.related_entries.contains(&id2) {
                entry1.related_entries.push(id2);
            }
        } else {
            return Err(Error::not_found(id1.to_string()));
        }

        if let Some(mut entry2) = self.entries.get_mut(&id2) {
            if !entry2.related_entries.contains(&id1) {
                entry2.related_entries.push(id1);
            }
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
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
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
                .max_by(|a, b| a.partial_cmp(b).unwrap())
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

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_distance(&a, &c) - 1.0).abs() < 1e-6);
    }
}
