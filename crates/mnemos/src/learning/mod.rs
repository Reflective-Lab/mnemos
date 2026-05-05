//! Self-learning engine using GNN-inspired approaches.
//!
//! This module implements adaptive learning mechanisms inspired by Graph Neural Networks
//! to improve search results over time based on user interactions.
//!
//! # Components
//!
//! - **LearningEngine**: Core learning with GNN message passing
//! - **FeedbackCollector**: Implicit signal capture from user interactions
//! - **BatchScheduler**: Background jobs for pattern detection and enrichment
//! - **InsightStore**: Storage for discovered patterns and relationships
//!
//! # Learning Flow
//!
//! ```text
//! User Interactions
//!        │
//!        ▼
//! ┌─────────────────┐
//! │FeedbackCollector│ ──► Implicit signals (view, select, dwell)
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │FeedbackProcessor│ ──► ProcessedFeedback (relevance deltas)
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ LearningEngine  │ ──► Update weights, GNN propagation
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │ BatchScheduler  │ ──► Patterns, gaps, classifications
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  InsightStore   │ ──► Published insights for retrieval
//! └─────────────────┘
//! ```

pub mod batch;
pub mod feedback;
mod gnn;
mod replay;

pub use batch::{
    BatchInput, BatchJob, BatchScheduler, EntryMetadata, Insight, InsightStore, JobRun, JobStatus,
    JobType, KnowledgeClass, RelationshipType, Trend,
};
pub use feedback::{
    FeedbackCollector, FeedbackConfig, FeedbackProcessor, FeedbackSignal, ProcessedFeedback,
    QueryId, SessionId, SignalType,
};
pub use gnn::GnnLayer;
pub use replay::ReplayBuffer;

use crate::core::SearchResult;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

/// Learning engine that improves search quality over time.
///
/// The engine uses several mechanisms inspired by the ruvector-gnn crate:
/// - Experience replay for stable learning
/// - GNN-style message passing for relevance propagation
/// - Elastic weight consolidation to prevent catastrophic forgetting
pub struct LearningEngine {
    /// Embedding dimensions.
    dimensions: usize,

    /// Learning rate.
    learning_rate: f32,

    /// Query-result relevance weights.
    relevance_weights: Vec<f32>,

    /// GNN layer for embedding transformation.
    gnn_layer: GnnLayer,

    /// Experience replay buffer.
    replay_buffer: ReplayBuffer<Experience>,

    /// Query patterns for learning (reserved for future pattern-based learning).
    #[allow(dead_code)]
    query_patterns: VecDeque<QueryPattern>,

    /// Entry relevance scores learned from feedback.
    entry_scores: DashMap<Uuid, f32>,

    /// Fisher information for EWC.
    fisher_diagonal: Vec<f32>,

    /// Total queries processed.
    query_count: u64,
}

/// A recorded query pattern for learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QueryPattern {
    /// Query embedding.
    query_embedding: Vec<f32>,

    /// Result embeddings with feedback.
    result_embeddings: Vec<(Vec<f32>, f32)>,

    /// Timestamp.
    timestamp: u64,
}

impl LearningEngine {
    /// Create a new learning engine.
    pub fn new(dimensions: usize, learning_rate: f32) -> Self {
        Self {
            dimensions,
            learning_rate,
            relevance_weights: vec![1.0; dimensions],
            gnn_layer: GnnLayer::new(dimensions, dimensions * 2, 4),
            replay_buffer: ReplayBuffer::new(10000),
            query_patterns: VecDeque::with_capacity(1000),
            entry_scores: DashMap::new(),
            fisher_diagonal: vec![0.0; dimensions],
            query_count: 0,
        }
    }

    /// Re-rank candidates based on learned patterns.
    pub fn rerank(
        &self,
        query_embedding: &[f32],
        mut candidates: Vec<(Uuid, f32)>,
        vectors: &DashMap<Uuid, Vec<f32>>,
    ) -> Vec<(Uuid, f32)> {
        // Transform query through GNN layer
        let neighbors: Vec<Vec<f32>> = candidates
            .iter()
            .take(10)
            .filter_map(|(id, _)| vectors.get(id).map(|v| v.clone()))
            .collect();

        let edge_weights: Vec<f32> = candidates
            .iter()
            .take(10)
            .map(|(_, d)| 1.0 - d.min(1.0))
            .collect();

        let transformed_query = self
            .gnn_layer
            .forward(query_embedding, &neighbors, &edge_weights);

        // Re-compute distances with transformed query
        for (id, distance) in candidates.iter_mut() {
            if let Some(vector) = vectors.get(id) {
                // Apply learned relevance weights
                let weighted_distance = self.weighted_distance(&transformed_query, &vector);

                // Apply entry-specific learned score
                let entry_boost = self.entry_scores.get(id).map(|s| *s).unwrap_or(1.0);

                *distance = weighted_distance / entry_boost;
            }
        }

        // Re-sort
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        candidates
    }

    /// Compute weighted distance using learned weights.
    fn weighted_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut weighted_dot = 0.0f32;
        let mut weighted_norm_a = 0.0f32;
        let mut weighted_norm_b = 0.0f32;

        for i in 0..a.len().min(b.len()).min(self.dimensions) {
            let w = self.relevance_weights[i];
            weighted_dot += a[i] * b[i] * w;
            weighted_norm_a += a[i] * a[i] * w;
            weighted_norm_b += b[i] * b[i] * w;
        }

        let norm = (weighted_norm_a * weighted_norm_b).sqrt();
        if norm > 0.0 {
            1.0 - (weighted_dot / norm)
        } else {
            1.0
        }
    }

    /// Record a query and its results for learning.
    pub fn record_query(&mut self, query_embedding: &[f32], results: &[SearchResult]) {
        if results.is_empty() {
            return;
        }

        self.query_count += 1;

        // Store in replay buffer
        for (rank, result) in results.iter().enumerate() {
            self.replay_buffer.add(Experience {
                query: query_embedding.to_vec(),
                result_id: result.entry.id,
                rank: rank as u32,
                score: result.score,
            });
        }

        // Periodic learning from replay buffer
        if self.query_count % 10 == 0 {
            self.learn_from_replay();
        }
    }

    /// Record user feedback on a search result.
    pub fn record_feedback(&mut self, result_embedding: &[f32], positive: bool) {
        let adjustment = if positive {
            self.learning_rate
        } else {
            -self.learning_rate * 0.5
        };

        // Adjust relevance weights based on feedback
        for (i, &val) in result_embedding.iter().enumerate() {
            if i < self.dimensions {
                // Apply adjustment proportional to embedding value
                let delta = adjustment * val.abs();

                // EWC regularization: smaller updates for important weights
                let ewc_factor = 1.0 / (1.0 + self.fisher_diagonal[i]);

                self.relevance_weights[i] =
                    (self.relevance_weights[i] + delta * ewc_factor).clamp(0.1, 10.0);
            }
        }

        // Update Fisher information estimate
        self.update_fisher(result_embedding);
    }

    /// Update Fisher information diagonal for EWC.
    fn update_fisher(&mut self, embedding: &[f32]) {
        for (i, &val) in embedding.iter().enumerate() {
            if i < self.dimensions {
                // Exponential moving average of squared gradients
                self.fisher_diagonal[i] = 0.99 * self.fisher_diagonal[i] + 0.01 * val * val;
            }
        }
    }

    /// Learn from replay buffer samples.
    fn learn_from_replay(&mut self) {
        let samples = self.replay_buffer.sample(32);

        for experience in samples {
            // Higher-ranked results should have boosted scores
            let target_boost = 1.0 + (1.0 / (1.0 + experience.rank as f32));

            // Update entry score
            self.entry_scores
                .entry(experience.result_id)
                .and_modify(|s| {
                    *s = (*s + target_boost) / 2.0;
                })
                .or_insert(target_boost);

            // Update GNN layer weights (simplified)
            self.gnn_layer
                .update(&experience.query, target_boost, self.learning_rate);
        }
    }

    /// Get the current query count.
    pub fn query_count(&self) -> u64 {
        self.query_count
    }

    /// Get learning statistics.
    pub fn stats(&self) -> LearningStats {
        let avg_weight: f32 = self.relevance_weights.iter().sum::<f32>() / self.dimensions as f32;
        let weight_variance: f32 = self
            .relevance_weights
            .iter()
            .map(|w| (w - avg_weight).powi(2))
            .sum::<f32>()
            / self.dimensions as f32;

        LearningStats {
            query_count: self.query_count,
            replay_buffer_size: self.replay_buffer.len(),
            learned_entries: self.entry_scores.len(),
            avg_relevance_weight: avg_weight,
            weight_variance,
        }
    }
}

/// Experience for replay buffer.
#[derive(Debug, Clone)]
struct Experience {
    query: Vec<f32>,
    result_id: Uuid,
    rank: u32,
    /// Raw score at record time (reserved for future reward-weighted learning).
    #[allow(dead_code)]
    score: f32,
}

/// Learning statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningStats {
    /// Total number of queries processed.
    pub query_count: u64,
    /// Current number of experiences in the replay buffer.
    pub replay_buffer_size: usize,
    /// Number of entries with learned relevance scores.
    pub learned_entries: usize,
    /// Mean relevance weight across all embedding dimensions.
    pub avg_relevance_weight: f32,
    /// Variance of relevance weights (measures how specialised learning has become).
    pub weight_variance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_engine_creation() {
        let engine = LearningEngine::new(128, 0.01);
        assert_eq!(engine.dimensions, 128);
        assert_eq!(engine.query_count, 0);
    }

    #[test]
    fn test_feedback_updates_weights() {
        let mut engine = LearningEngine::new(64, 0.1);
        let initial_weights = engine.relevance_weights.clone();

        let embedding = vec![0.5; 64];
        engine.record_feedback(&embedding, true);

        // Weights should have changed
        assert_ne!(engine.relevance_weights, initial_weights);
    }
}
