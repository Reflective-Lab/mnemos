//! Batch Learning Jobs
//!
//! Background jobs that analyze patterns, identify gaps, classify knowledge,
//! and enrich the knowledge base over time.
//!
//! # Job Types
//!
//! | Job | Frequency | Purpose |
//! |-----|-----------|---------|
//! | PatternDetector | Hourly | Cluster queries, find hot topics |
//! | GapIdentifier | Daily | Find missing knowledge areas |
//! | KnowledgeClassifier | Daily | Core vs Derived classification |
//! | RelationshipMiner | Weekly | Discover hidden connections |
//! | ModelConsolidator | Weekly | Update EWC weights, prune old data |
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │                  BatchScheduler                      │
//! │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
//! │  │ Pattern │ │   Gap   │ │Classify │ │ Miner   │   │
//! │  │Detector │ │Identifier│ │  Job   │ │   Job   │   │
//! │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
//! │       │           │           │           │         │
//! │       └───────────┴───────────┴───────────┘         │
//! │                       │                             │
//! │                       ▼                             │
//! │              ┌─────────────────┐                    │
//! │              │  InsightStore   │                    │
//! │              │  (publishes to  │                    │
//! │              │  knowledge base)│                    │
//! │              └─────────────────┘                    │
//! └─────────────────────────────────────────────────────┘
//! ```

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::sync::{RwLock, mpsc};
use tokio::time::interval;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::feedback::{FeedbackSignal, ProcessedFeedback, SignalType};

/// Type of batch job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JobType {
    /// Detect query patterns and hot topics.
    PatternDetector,
    /// Identify gaps in knowledge coverage.
    GapIdentifier,
    /// Classify knowledge as core/derived/contextual.
    KnowledgeClassifier,
    /// Mine relationships between entries.
    RelationshipMiner,
    /// Consolidate learning models (EWC, pruning).
    ModelConsolidator,
}

impl JobType {
    /// Default interval for this job type.
    pub fn default_interval(&self) -> Duration {
        match self {
            JobType::PatternDetector => Duration::hours(1),
            JobType::GapIdentifier => Duration::days(1),
            JobType::KnowledgeClassifier => Duration::days(1),
            JobType::RelationshipMiner => Duration::weeks(1),
            JobType::ModelConsolidator => Duration::weeks(1),
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            JobType::PatternDetector => "Pattern Detector",
            JobType::GapIdentifier => "Gap Identifier",
            JobType::KnowledgeClassifier => "Knowledge Classifier",
            JobType::RelationshipMiner => "Relationship Miner",
            JobType::ModelConsolidator => "Model Consolidator",
        }
    }
}

/// Status of a job run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobStatus {
    /// Job is scheduled but not started.
    Pending,
    /// Job is currently running.
    Running {
        /// When it started.
        started_at: DateTime<Utc>,
        /// Progress percentage (0-100).
        progress: u8,
    },
    /// Job completed successfully.
    Completed {
        /// When it finished.
        finished_at: DateTime<Utc>,
        /// Duration in seconds.
        duration_secs: u64,
        /// Items processed.
        items_processed: usize,
    },
    /// Job failed.
    Failed {
        /// When it failed.
        failed_at: DateTime<Utc>,
        /// Error message.
        error: String,
    },
}

/// Record of a job execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRun {
    /// Unique run ID.
    pub id: Uuid,
    /// Job type.
    pub job_type: JobType,
    /// Status.
    pub status: JobStatus,
    /// When this run was scheduled.
    pub scheduled_at: DateTime<Utc>,
}

/// Insight produced by a batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Insight {
    /// A pattern detected in queries.
    QueryPattern {
        /// Pattern description.
        description: String,
        /// Representative query texts.
        example_queries: Vec<String>,
        /// Frequency of this pattern.
        frequency: f32,
        /// Associated entry IDs.
        related_entries: Vec<Uuid>,
    },

    /// A gap in knowledge coverage.
    KnowledgeGap {
        /// Topic or area with gap.
        topic: String,
        /// Queries that couldn't be answered well.
        unresolved_queries: Vec<String>,
        /// Suggested content to add.
        suggestions: Vec<String>,
        /// Severity (0.0 to 1.0).
        severity: f32,
    },

    /// Classification of an entry.
    Classification {
        /// Entry ID.
        entry_id: Uuid,
        /// Knowledge class.
        class: KnowledgeClass,
        /// Confidence (0.0 to 1.0).
        confidence: f32,
        /// Reasoning.
        reason: String,
    },

    /// A discovered relationship.
    Relationship {
        /// Source entry.
        source_id: Uuid,
        /// Target entry.
        target_id: Uuid,
        /// Relationship type.
        relationship: RelationshipType,
        /// Strength (0.0 to 1.0).
        strength: f32,
    },

    /// Hot topic detection.
    HotTopic {
        /// Topic name.
        topic: String,
        /// Related entry IDs.
        entry_ids: Vec<Uuid>,
        /// Current interest score.
        interest_score: f32,
        /// Trend direction.
        trend: Trend,
    },
}

/// Knowledge classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeClass {
    /// Fundamental knowledge that rarely changes.
    Core,
    /// Knowledge derived from core knowledge.
    Derived,
    /// Context-specific knowledge.
    Contextual,
    /// Temporary or soon-to-expire knowledge.
    Ephemeral,
}

/// Relationship type between entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// One entry extends another.
    Extends,
    /// One entry contradicts another.
    Contradicts,
    /// One entry is a prerequisite for another.
    Prerequisite,
    /// Entries are related but distinct.
    Related,
    /// One entry supersedes another.
    Supersedes,
    /// Entries are frequently accessed together.
    CoAccessed,
}

/// Trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    /// Interest is increasing.
    Rising,
    /// Interest is stable.
    Stable,
    /// Interest is decreasing.
    Falling,
}

/// Store for insights produced by batch jobs.
pub struct InsightStore {
    /// All insights.
    insights: RwLock<Vec<Insight>>,
    /// Insights by type for quick lookup.
    by_entry: RwLock<HashMap<Uuid, Vec<usize>>>,
    /// When last updated.
    last_updated: RwLock<DateTime<Utc>>,
}

impl InsightStore {
    /// Create a new insight store.
    pub fn new() -> Self {
        Self {
            insights: RwLock::new(Vec::new()),
            by_entry: RwLock::new(HashMap::new()),
            last_updated: RwLock::new(Utc::now()),
        }
    }

    /// Add an insight.
    pub async fn add(&self, insight: Insight) {
        let mut insights = self.insights.write().await;
        let idx = insights.len();

        // Index by entry ID
        let entry_ids = Self::extract_entry_ids(&insight);
        {
            let mut by_entry = self.by_entry.write().await;
            for id in entry_ids {
                by_entry.entry(id).or_default().push(idx);
            }
        }

        insights.push(insight);
        *self.last_updated.write().await = Utc::now();
    }

    /// Get insights for an entry.
    pub async fn for_entry(&self, entry_id: Uuid) -> Vec<Insight> {
        let by_entry = self.by_entry.read().await;
        let insights = self.insights.read().await;

        by_entry
            .get(&entry_id)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&i| insights.get(i).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all gaps.
    pub async fn gaps(&self) -> Vec<Insight> {
        self.insights
            .read()
            .await
            .iter()
            .filter(|i| matches!(i, Insight::KnowledgeGap { .. }))
            .cloned()
            .collect()
    }

    /// Get all hot topics.
    pub async fn hot_topics(&self) -> Vec<Insight> {
        self.insights
            .read()
            .await
            .iter()
            .filter(|i| matches!(i, Insight::HotTopic { .. }))
            .cloned()
            .collect()
    }

    /// Get classification for an entry.
    pub async fn classification(&self, entry_id: Uuid) -> Option<KnowledgeClass> {
        self.for_entry(entry_id).await.into_iter().find_map(|i| {
            if let Insight::Classification { class, .. } = i {
                Some(class)
            } else {
                None
            }
        })
    }

    /// Clear old insights.
    pub async fn clear(&self) {
        self.insights.write().await.clear();
        self.by_entry.write().await.clear();
    }

    fn extract_entry_ids(insight: &Insight) -> Vec<Uuid> {
        match insight {
            Insight::QueryPattern {
                related_entries, ..
            } => related_entries.clone(),
            Insight::KnowledgeGap { .. } => vec![],
            Insight::Classification { entry_id, .. } => vec![*entry_id],
            Insight::Relationship {
                source_id,
                target_id,
                ..
            } => vec![*source_id, *target_id],
            Insight::HotTopic { entry_ids, .. } => entry_ids.clone(),
        }
    }
}

impl Default for InsightStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Input data for batch jobs.
pub struct BatchInput {
    /// Recent feedback signals.
    pub signals: Vec<FeedbackSignal>,
    /// Processed feedback.
    pub processed_feedback: Vec<ProcessedFeedback>,
    /// Entry metadata (id -> metadata).
    pub entry_metadata: HashMap<Uuid, EntryMetadata>,
    /// Entry embeddings (id -> embedding).
    pub entry_embeddings: HashMap<Uuid, Vec<f32>>,
    /// Current relationships.
    pub relationships: Vec<(Uuid, Uuid, f32)>,
}

/// Metadata about an entry for batch processing.
#[derive(Debug, Clone)]
pub struct EntryMetadata {
    /// Entry ID.
    pub id: Uuid,
    /// When created.
    pub created_at: DateTime<Utc>,
    /// Last accessed.
    pub last_accessed: DateTime<Utc>,
    /// Access count.
    pub access_count: u64,
    /// Current relevance score.
    pub relevance_score: f32,
    /// Tags.
    pub tags: Vec<String>,
    /// Category.
    pub category: Option<String>,
}

/// Trait for batch job implementations.
#[async_trait::async_trait]
pub trait BatchJob: Send + Sync {
    /// Job type.
    fn job_type(&self) -> JobType;

    /// Run the job with given input.
    async fn run(&self, input: &BatchInput) -> Result<Vec<Insight>, String>;

    /// Estimated duration in seconds.
    fn estimated_duration_secs(&self) -> u64 {
        60
    }
}

/// Pattern detection job.
pub struct PatternDetectorJob {
    /// Minimum query frequency to be a pattern.
    pub min_frequency: f32,
    /// Maximum number of patterns to report.
    pub max_patterns: usize,
}

impl Default for PatternDetectorJob {
    fn default() -> Self {
        Self {
            min_frequency: 0.05,
            max_patterns: 20,
        }
    }
}

#[async_trait::async_trait]
impl BatchJob for PatternDetectorJob {
    fn job_type(&self) -> JobType {
        JobType::PatternDetector
    }

    async fn run(&self, input: &BatchInput) -> Result<Vec<Insight>, String> {
        let mut insights = Vec::new();

        // Extract query texts and cluster them
        let queries: Vec<String> = input
            .signals
            .iter()
            .filter_map(|s| match &s.signal {
                SignalType::Query { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect();

        if queries.is_empty() {
            return Ok(insights);
        }

        // Simple clustering by common terms
        let mut term_counts: HashMap<String, Vec<String>> = HashMap::new();
        for query in &queries {
            for word in query.to_lowercase().split_whitespace() {
                if word.len() > 3 {
                    term_counts
                        .entry(word.to_string())
                        .or_default()
                        .push(query.clone());
                }
            }
        }

        // Find patterns (terms that appear frequently)
        let total_queries = queries.len() as f32;
        let mut patterns: Vec<_> = term_counts
            .into_iter()
            .filter(|(_, qs)| qs.len() as f32 / total_queries >= self.min_frequency)
            .map(|(term, qs)| (term, qs.len() as f32 / total_queries, qs))
            .collect();

        patterns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        patterns.truncate(self.max_patterns);

        for (term, freq, example_queries) in patterns {
            // Find related entries (those that were viewed/selected for these queries)
            let related_entries: Vec<Uuid> = input
                .processed_feedback
                .iter()
                .filter(|fb| fb.relevance_delta > 0.0)
                .map(|fb| fb.entry_id)
                .collect();

            insights.push(Insight::QueryPattern {
                description: format!("Queries about '{}'", term),
                example_queries: example_queries.into_iter().take(5).collect(),
                frequency: freq,
                related_entries,
            });
        }

        Ok(insights)
    }
}

/// Gap identification job.
pub struct GapIdentifierJob {
    /// Minimum unresolved queries to report a gap.
    pub min_unresolved: usize,
    /// Maximum gaps to report.
    pub max_gaps: usize,
}

impl Default for GapIdentifierJob {
    fn default() -> Self {
        Self {
            min_unresolved: 3,
            max_gaps: 10,
        }
    }
}

#[async_trait::async_trait]
impl BatchJob for GapIdentifierJob {
    fn job_type(&self) -> JobType {
        JobType::GapIdentifier
    }

    async fn run(&self, input: &BatchInput) -> Result<Vec<Insight>, String> {
        let mut insights = Vec::new();

        // Find queries with low or no positive feedback
        let mut query_results: HashMap<String, (usize, usize)> = HashMap::new(); // (total, positive)

        for signal in &input.signals {
            if let SignalType::Query {
                text, result_ids, ..
            } = &signal.signal
            {
                let entry = query_results.entry(text.clone()).or_insert((0, 0));
                entry.0 += 1;

                // Check if any result got positive feedback
                let has_positive = input
                    .processed_feedback
                    .iter()
                    .any(|fb| result_ids.contains(&fb.entry_id) && fb.relevance_delta > 0.0);
                if has_positive {
                    entry.1 += 1;
                }
            }
        }

        // Group low-success queries by topic
        let low_success: Vec<_> = query_results
            .into_iter()
            .filter(|(_, (total, positive))| {
                *total >= self.min_unresolved && (*positive as f32 / *total as f32) < 0.3
            })
            .collect();

        if !low_success.is_empty() {
            // Simple topic extraction (first significant word)
            let mut by_topic: HashMap<String, Vec<String>> = HashMap::new();
            for (query, _) in low_success {
                let topic = query
                    .split_whitespace()
                    .find(|w| w.len() > 3)
                    .unwrap_or("general")
                    .to_string();
                by_topic.entry(topic).or_default().push(query);
            }

            for (topic, queries) in by_topic.into_iter().take(self.max_gaps) {
                let severity = (queries.len() as f32 / 10.0).min(1.0);
                insights.push(Insight::KnowledgeGap {
                    topic: topic.clone(),
                    unresolved_queries: queries.clone(),
                    suggestions: vec![format!("Add documentation about {}", topic)],
                    severity,
                });
            }
        }

        Ok(insights)
    }
}

/// Knowledge classification job.
pub struct KnowledgeClassifierJob {
    /// Core knowledge access threshold (accesses per day).
    pub core_access_threshold: f32,
    /// Age threshold for ephemeral (days).
    pub ephemeral_age_days: i64,
}

impl Default for KnowledgeClassifierJob {
    fn default() -> Self {
        Self {
            core_access_threshold: 1.0,
            ephemeral_age_days: 7,
        }
    }
}

#[async_trait::async_trait]
impl BatchJob for KnowledgeClassifierJob {
    fn job_type(&self) -> JobType {
        JobType::KnowledgeClassifier
    }

    async fn run(&self, input: &BatchInput) -> Result<Vec<Insight>, String> {
        let mut insights = Vec::new();
        let now = Utc::now();

        for (id, meta) in &input.entry_metadata {
            let age_days = (now - meta.created_at).num_days().max(1);
            let access_rate = meta.access_count as f32 / age_days as f32;

            let (class, confidence, reason) =
                if access_rate >= self.core_access_threshold && age_days > 30 {
                    (
                        KnowledgeClass::Core,
                        0.8,
                        "High access rate over extended period".to_string(),
                    )
                } else if age_days <= self.ephemeral_age_days && meta.access_count <= 2 {
                    (
                        KnowledgeClass::Ephemeral,
                        0.6,
                        "New entry with limited access".to_string(),
                    )
                } else if meta
                    .tags
                    .iter()
                    .any(|t| t.contains("derived") || t.contains("computed"))
                {
                    (
                        KnowledgeClass::Derived,
                        0.7,
                        "Tagged as derived knowledge".to_string(),
                    )
                } else if meta
                    .category
                    .as_ref()
                    .is_some_and(|c| c.contains("project"))
                {
                    (
                        KnowledgeClass::Contextual,
                        0.7,
                        "Project-specific content".to_string(),
                    )
                } else {
                    (
                        KnowledgeClass::Derived,
                        0.4,
                        "Default classification".to_string(),
                    )
                };

            insights.push(Insight::Classification {
                entry_id: *id,
                class,
                confidence,
                reason,
            });
        }

        Ok(insights)
    }
}

/// Relationship mining job.
pub struct RelationshipMinerJob {
    /// Minimum co-access count to create relationship.
    pub min_co_access: usize,
    /// Minimum embedding similarity for related.
    pub min_similarity: f32,
}

impl Default for RelationshipMinerJob {
    fn default() -> Self {
        Self {
            min_co_access: 3,
            min_similarity: 0.7,
        }
    }
}

#[async_trait::async_trait]
impl BatchJob for RelationshipMinerJob {
    fn job_type(&self) -> JobType {
        JobType::RelationshipMiner
    }

    async fn run(&self, input: &BatchInput) -> Result<Vec<Insight>, String> {
        let mut insights = Vec::new();

        // Count co-accesses from signals
        let mut co_access_counts: HashMap<(Uuid, Uuid), usize> = HashMap::new();

        for signal in &input.signals {
            if let SignalType::CoAccess { entry_ids } = &signal.signal {
                for i in 0..entry_ids.len() {
                    for j in (i + 1)..entry_ids.len() {
                        let pair = if entry_ids[i] < entry_ids[j] {
                            (entry_ids[i], entry_ids[j])
                        } else {
                            (entry_ids[j], entry_ids[i])
                        };
                        *co_access_counts.entry(pair).or_insert(0) += 1;
                    }
                }
            }
        }

        // Create co-access relationships
        for ((id1, id2), count) in co_access_counts {
            if count >= self.min_co_access {
                let strength = (count as f32 / 10.0).min(1.0);
                insights.push(Insight::Relationship {
                    source_id: id1,
                    target_id: id2,
                    relationship: RelationshipType::CoAccessed,
                    strength,
                });
            }
        }

        // Find similar entries by embedding
        let entries: Vec<_> = input.entry_embeddings.iter().collect();
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let (id1, emb1) = entries[i];
                let (id2, emb2) = entries[j];

                let similarity = cosine_similarity(emb1, emb2);
                if similarity >= self.min_similarity {
                    insights.push(Insight::Relationship {
                        source_id: *id1,
                        target_id: *id2,
                        relationship: RelationshipType::Related,
                        strength: similarity,
                    });
                }
            }
        }

        Ok(insights)
    }

    fn estimated_duration_secs(&self) -> u64 {
        300 // 5 minutes for relationship mining
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Scheduler for batch jobs.
pub struct BatchScheduler {
    /// Available jobs.
    jobs: Vec<Arc<dyn BatchJob>>,
    /// Job history.
    history: RwLock<Vec<JobRun>>,
    /// Insight store.
    insight_store: Arc<InsightStore>,
    /// Whether scheduler is running.
    running: AtomicBool,
    /// Total jobs executed.
    total_runs: AtomicU64,
}

impl BatchScheduler {
    /// Create a new scheduler with default jobs.
    pub fn new(insight_store: Arc<InsightStore>) -> Self {
        Self {
            jobs: vec![
                Arc::new(PatternDetectorJob::default()),
                Arc::new(GapIdentifierJob::default()),
                Arc::new(KnowledgeClassifierJob::default()),
                Arc::new(RelationshipMinerJob::default()),
            ],
            history: RwLock::new(Vec::new()),
            insight_store,
            running: AtomicBool::new(false),
            total_runs: AtomicU64::new(0),
        }
    }

    /// Add a custom job.
    pub fn add_job(&mut self, job: Arc<dyn BatchJob>) {
        self.jobs.push(job);
    }

    /// Run a specific job type immediately.
    pub async fn run_job(&self, job_type: JobType, input: &BatchInput) -> Result<JobRun, String> {
        let job = self
            .jobs
            .iter()
            .find(|j| j.job_type() == job_type)
            .ok_or_else(|| format!("Job type {:?} not found", job_type))?;

        let run_id = Uuid::new_v4();
        let started_at = Utc::now();

        info!(job = %job_type.name(), "Starting batch job");

        let run = JobRun {
            id: run_id,
            job_type,
            status: JobStatus::Running {
                started_at,
                progress: 0,
            },
            scheduled_at: started_at,
        };

        // Record start
        self.history.write().await.push(run.clone());

        // Execute
        let result = job.run(input).await;

        let finished_at = Utc::now();
        let duration_secs = (finished_at - started_at).num_seconds() as u64;

        let final_status = match result {
            Ok(insights) => {
                let count = insights.len();
                // Publish insights
                for insight in insights {
                    self.insight_store.add(insight).await;
                }

                info!(job = %job_type.name(), insights = count, duration_secs, "Batch job completed");

                JobStatus::Completed {
                    finished_at,
                    duration_secs,
                    items_processed: count,
                }
            }
            Err(e) => {
                warn!(job = %job_type.name(), error = %e, "Batch job failed");
                JobStatus::Failed {
                    failed_at: finished_at,
                    error: e,
                }
            }
        };

        // Update history
        {
            let mut history = self.history.write().await;
            if let Some(run) = history.iter_mut().find(|r| r.id == run_id) {
                run.status = final_status.clone();
            }
        }

        self.total_runs.fetch_add(1, Ordering::Relaxed);

        Ok(JobRun {
            id: run_id,
            job_type,
            status: final_status,
            scheduled_at: started_at,
        })
    }

    /// Run all jobs.
    pub async fn run_all(&self, input: &BatchInput) -> Vec<JobRun> {
        let mut runs = Vec::new();
        for job in &self.jobs {
            if let Ok(run) = self.run_job(job.job_type(), input).await {
                runs.push(run);
            }
        }
        runs
    }

    /// Get job history.
    pub async fn history(&self) -> Vec<JobRun> {
        self.history.read().await.clone()
    }

    /// Get last run for a job type.
    pub async fn last_run(&self, job_type: JobType) -> Option<JobRun> {
        self.history
            .read()
            .await
            .iter()
            .rfind(|r| r.job_type == job_type)
            .cloned()
    }

    /// Total runs executed.
    pub fn total_runs(&self) -> u64 {
        self.total_runs.load(Ordering::Relaxed)
    }

    /// Start background scheduling (non-blocking).
    pub fn start_background(
        self: Arc<Self>,
        mut input_receiver: mpsc::Receiver<BatchInput>,
    ) -> tokio::task::JoinHandle<()> {
        self.running.store(true, Ordering::SeqCst);

        tokio::spawn(async move {
            let mut check_interval = interval(std::time::Duration::from_secs(60));

            while self.running.load(Ordering::SeqCst) {
                tokio::select! {
                    Some(input) = input_receiver.recv() => {
                        debug!("Received batch input, running all jobs");
                        self.run_all(&input).await;
                    }
                    _ = check_interval.tick() => {
                        // Periodic maintenance could go here
                    }
                }
            }
        })
    }

    /// Stop background scheduling.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_type_intervals() {
        assert_eq!(
            JobType::PatternDetector.default_interval(),
            Duration::hours(1)
        );
        assert_eq!(JobType::GapIdentifier.default_interval(), Duration::days(1));
    }

    #[tokio::test]
    async fn test_insight_store() {
        let store = InsightStore::new();

        let entry_id = Uuid::new_v4();
        store
            .add(Insight::Classification {
                entry_id,
                class: KnowledgeClass::Core,
                confidence: 0.9,
                reason: "Test".to_string(),
            })
            .await;

        let class = store.classification(entry_id).await;
        assert_eq!(class, Some(KnowledgeClass::Core));
    }

    #[tokio::test]
    async fn test_pattern_detector() {
        let job = PatternDetectorJob::default();

        let signals = vec![
            FeedbackSignal::new(
                super::super::feedback::SessionId::new(),
                None,
                SignalType::Query {
                    text: "rust async programming".to_string(),
                    embedding: None,
                    result_ids: vec![],
                },
            ),
            FeedbackSignal::new(
                super::super::feedback::SessionId::new(),
                None,
                SignalType::Query {
                    text: "async rust patterns".to_string(),
                    embedding: None,
                    result_ids: vec![],
                },
            ),
            FeedbackSignal::new(
                super::super::feedback::SessionId::new(),
                None,
                SignalType::Query {
                    text: "rust async await".to_string(),
                    embedding: None,
                    result_ids: vec![],
                },
            ),
        ];

        let input = BatchInput {
            signals,
            processed_feedback: vec![],
            entry_metadata: HashMap::new(),
            entry_embeddings: HashMap::new(),
            relationships: vec![],
        };

        let insights = job.run(&input).await.unwrap();

        // Should detect "async" and "rust" as patterns
        assert!(!insights.is_empty());
    }

    #[tokio::test]
    async fn test_scheduler_run_job() {
        let store = Arc::new(InsightStore::new());
        let scheduler = BatchScheduler::new(store.clone());

        let input = BatchInput {
            signals: vec![],
            processed_feedback: vec![],
            entry_metadata: HashMap::new(),
            entry_embeddings: HashMap::new(),
            relationships: vec![],
        };

        let run = scheduler.run_job(JobType::PatternDetector, &input).await;
        assert!(run.is_ok());

        let run = run.unwrap();
        assert!(matches!(run.status, JobStatus::Completed { .. }));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }
}
