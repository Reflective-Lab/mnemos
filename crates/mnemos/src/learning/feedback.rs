//! Implicit Feedback Collection
//!
//! Automatically captures user interaction signals to improve search quality.
//! Unlike explicit feedback ("this was helpful"), implicit feedback is derived
//! from natural usage patterns.
//!
//! # Signal Types
//!
//! | Signal | What It Means | Learning Impact |
//! |--------|---------------|-----------------|
//! | Query | User searched for something | Records query patterns |
//! | View | User looked at a result | Weak positive signal |
//! | Select | User chose this result | Strong positive signal |
//! | Ignore | Result shown but not viewed | Weak negative signal |
//! | Dwell | Time spent on result | Strength of interest |
//! | FollowUp | Query after viewing result | Indicates gap or relation |
//! | CoAccess | Items accessed together | Hidden relationship |
//!
//! # Example
//!
//! ```ignore
//! use mnemos::learning::FeedbackCollector;
//!
//! let collector = FeedbackCollector::new();
//!
//! // Start a session
//! let session = collector.start_session();
//!
//! // Track a search
//! let query_id = collector.record_query(&session, "rust async patterns", &results);
//!
//! // User views result #2
//! collector.record_view(&session, query_id, results[1].entry_id);
//!
//! // User selects (clicks, copies, etc.) result #2
//! collector.record_select(&session, query_id, results[1].entry_id);
//!
//! // Later, process accumulated feedback
//! let signals = collector.drain_signals();
//! learning_engine.apply_implicit_feedback(signals);
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Session identifier for grouping related interactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(Uuid);

impl SessionId {
    /// Create a new session ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Query identifier for tracking result interactions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QueryId(Uuid);

impl QueryId {
    /// Create a new query ID.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for QueryId {
    fn default() -> Self {
        Self::new()
    }
}

/// Type of implicit signal captured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    /// User issued a query.
    Query {
        /// The query text.
        text: String,
        /// Embedding of the query (if available).
        embedding: Option<Vec<f32>>,
        /// IDs of results returned, in order.
        result_ids: Vec<Uuid>,
    },

    /// User viewed a result (e.g., expanded, hovered).
    View {
        /// The entry that was viewed.
        entry_id: Uuid,
        /// Position in the result list (0-indexed).
        position: usize,
    },

    /// User selected/used a result (e.g., clicked, copied).
    Select {
        /// The entry that was selected.
        entry_id: Uuid,
        /// Position in the result list.
        position: usize,
    },

    /// User explicitly dismissed a result.
    Dismiss {
        /// The entry that was dismissed.
        entry_id: Uuid,
        /// Position in the result list.
        position: usize,
    },

    /// Time spent on a result (dwell time).
    Dwell {
        /// The entry.
        entry_id: Uuid,
        /// Time spent in milliseconds.
        duration_ms: u64,
    },

    /// User made a follow-up query after viewing results.
    FollowUp {
        /// Previous query ID.
        previous_query: QueryId,
        /// Entry IDs that were viewed before the follow-up.
        viewed_entries: Vec<Uuid>,
    },

    /// Multiple entries accessed in the same session.
    CoAccess {
        /// Entries accessed together.
        entry_ids: Vec<Uuid>,
    },

    /// Session ended - compute final signals.
    SessionEnd {
        /// Total session duration in seconds.
        duration_secs: u64,
        /// Number of queries in session.
        query_count: usize,
    },
}

/// A captured feedback signal with context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSignal {
    /// Unique signal ID.
    pub id: Uuid,
    /// Session this signal belongs to.
    pub session_id: SessionId,
    /// Query this signal relates to (if any).
    pub query_id: Option<QueryId>,
    /// The signal type and data.
    pub signal: SignalType,
    /// When this signal was captured.
    pub timestamp: DateTime<Utc>,
}

impl FeedbackSignal {
    /// Create a new feedback signal.
    pub fn new(session_id: SessionId, query_id: Option<QueryId>, signal: SignalType) -> Self {
        Self {
            id: Uuid::new_v4(),
            session_id,
            query_id,
            signal,
            timestamp: Utc::now(),
        }
    }
}

/// Active session state.
struct ActiveSession {
    /// Session ID.
    id: SessionId,
    /// When session started.
    started_at: Instant,
    /// Queries in this session.
    queries: Vec<QueryId>,
    /// Entries viewed in this session.
    viewed_entries: Vec<Uuid>,
    /// Entries selected in this session.
    selected_entries: Vec<Uuid>,
    /// Last activity time.
    last_activity: Instant,
    /// Current query context.
    current_query: Option<QueryContext>,
}

/// Context for the current query.
#[allow(dead_code)]
struct QueryContext {
    /// Query ID.
    id: QueryId,
    /// Query text (reserved for future query-text analytics).
    text: String,
    /// Result IDs returned (reserved for future result-tracking analytics).
    result_ids: Vec<Uuid>,
    /// When query was issued (reserved for future temporal analytics).
    timestamp: Instant,
    /// Entries viewed for this query.
    viewed: Vec<Uuid>,
}

impl ActiveSession {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            id: SessionId::new(),
            started_at: now,
            queries: Vec::new(),
            viewed_entries: Vec::new(),
            selected_entries: Vec::new(),
            last_activity: now,
            current_query: None,
        }
    }

    fn touch(&mut self) {
        self.last_activity = Instant::now();
    }

    fn is_expired(&self, timeout: Duration) -> bool {
        self.last_activity.elapsed() > timeout
    }
}

/// Configuration for the feedback collector.
#[derive(Debug, Clone)]
pub struct FeedbackConfig {
    /// Session timeout (inactive duration before session ends).
    pub session_timeout: Duration,
    /// Maximum signals to buffer before forcing drain.
    pub max_buffer_size: usize,
    /// Minimum dwell time to record (filters accidental views).
    pub min_dwell_ms: u64,
    /// Whether to automatically compute co-access signals.
    pub compute_co_access: bool,
    /// Minimum views for co-access signal.
    pub co_access_min_views: usize,
}

impl Default for FeedbackConfig {
    fn default() -> Self {
        Self {
            session_timeout: Duration::from_mins(30),
            max_buffer_size: 10_000,
            min_dwell_ms: 500, // Half second minimum
            compute_co_access: true,
            co_access_min_views: 2,
        }
    }
}

/// Collector for implicit feedback signals.
///
/// Thread-safe and designed for concurrent access from multiple queries.
pub struct FeedbackCollector {
    config: FeedbackConfig,
    /// Active sessions by ID.
    sessions: Arc<RwLock<HashMap<SessionId, ActiveSession>>>,
    /// Buffered signals waiting to be processed.
    signals: Arc<RwLock<VecDeque<FeedbackSignal>>>,
    /// Total signals collected.
    total_signals: AtomicU64,
    /// Total sessions created.
    total_sessions: AtomicU64,
}

impl FeedbackCollector {
    /// Create a new feedback collector with default config.
    pub fn new() -> Self {
        Self::with_config(FeedbackConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: FeedbackConfig) -> Self {
        Self {
            config,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            signals: Arc::new(RwLock::new(VecDeque::new())),
            total_signals: AtomicU64::new(0),
            total_sessions: AtomicU64::new(0),
        }
    }

    /// Start a new session.
    pub async fn start_session(&self) -> SessionId {
        let session = ActiveSession::new();
        let id = session.id;

        let mut sessions = self.sessions.write().await;
        sessions.insert(id, session);
        self.total_sessions.fetch_add(1, Ordering::Relaxed);

        id
    }

    /// Get or create a session (for stateless APIs).
    pub async fn get_or_create_session(&self, session_id: Option<SessionId>) -> SessionId {
        if let Some(id) = session_id {
            let sessions = self.sessions.read().await;
            if sessions.contains_key(&id) {
                return id;
            }
        }
        self.start_session().await
    }

    /// Record a query and its results.
    pub async fn record_query(
        &self,
        session_id: SessionId,
        query_text: &str,
        result_ids: Vec<Uuid>,
        query_embedding: Option<Vec<f32>>,
    ) -> QueryId {
        let query_id = QueryId::new();

        // Update session state
        {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                // Check if this is a follow-up query
                if let Some(prev_query) = &session.current_query
                    && !prev_query.viewed.is_empty()
                {
                    self.emit_signal(FeedbackSignal::new(
                        session_id,
                        Some(query_id),
                        SignalType::FollowUp {
                            previous_query: prev_query.id,
                            viewed_entries: prev_query.viewed.clone(),
                        },
                    ))
                    .await;
                }

                // Set new query context
                session.current_query = Some(QueryContext {
                    id: query_id,
                    text: query_text.to_string(),
                    result_ids: result_ids.clone(),
                    timestamp: Instant::now(),
                    viewed: Vec::new(),
                });
                session.queries.push(query_id);
                session.touch();
            }
        }

        // Emit query signal
        self.emit_signal(FeedbackSignal::new(
            session_id,
            Some(query_id),
            SignalType::Query {
                text: query_text.to_string(),
                embedding: query_embedding,
                result_ids,
            },
        ))
        .await;

        query_id
    }

    /// Record that a user viewed a result.
    pub async fn record_view(&self, session_id: SessionId, entry_id: Uuid, position: usize) {
        let query_id = {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.viewed_entries.push(entry_id);
                session.touch();

                if let Some(query) = &mut session.current_query {
                    query.viewed.push(entry_id);
                    Some(query.id)
                } else {
                    None
                }
            } else {
                None
            }
        };

        self.emit_signal(FeedbackSignal::new(
            session_id,
            query_id,
            SignalType::View { entry_id, position },
        ))
        .await;
    }

    /// Record that a user selected/used a result.
    pub async fn record_select(&self, session_id: SessionId, entry_id: Uuid, position: usize) {
        let query_id = {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(&session_id) {
                session.selected_entries.push(entry_id);
                session.touch();
                session.current_query.as_ref().map(|q| q.id)
            } else {
                None
            }
        };

        self.emit_signal(FeedbackSignal::new(
            session_id,
            query_id,
            SignalType::Select { entry_id, position },
        ))
        .await;
    }

    /// Record that a user dismissed a result.
    pub async fn record_dismiss(&self, session_id: SessionId, entry_id: Uuid, position: usize) {
        let query_id = {
            let sessions = self.sessions.read().await;
            sessions
                .get(&session_id)
                .and_then(|s| s.current_query.as_ref().map(|q| q.id))
        };

        self.emit_signal(FeedbackSignal::new(
            session_id,
            query_id,
            SignalType::Dismiss { entry_id, position },
        ))
        .await;
    }

    /// Record dwell time on a result.
    pub async fn record_dwell(&self, session_id: SessionId, entry_id: Uuid, duration_ms: u64) {
        // Filter out accidental/brief views
        if duration_ms < self.config.min_dwell_ms {
            return;
        }

        let query_id = {
            let sessions = self.sessions.read().await;
            sessions
                .get(&session_id)
                .and_then(|s| s.current_query.as_ref().map(|q| q.id))
        };

        self.emit_signal(FeedbackSignal::new(
            session_id,
            query_id,
            SignalType::Dwell {
                entry_id,
                duration_ms,
            },
        ))
        .await;
    }

    /// End a session explicitly.
    pub async fn end_session(&self, session_id: SessionId) {
        let session_data = {
            let mut sessions = self.sessions.write().await;
            sessions.remove(&session_id)
        };

        if let Some(session) = session_data {
            // Emit session end signal
            self.emit_signal(FeedbackSignal::new(
                session_id,
                None,
                SignalType::SessionEnd {
                    duration_secs: session.started_at.elapsed().as_secs(),
                    query_count: session.queries.len(),
                },
            ))
            .await;

            // Compute co-access if enabled
            if self.config.compute_co_access
                && session.viewed_entries.len() >= self.config.co_access_min_views
            {
                self.emit_signal(FeedbackSignal::new(
                    session_id,
                    None,
                    SignalType::CoAccess {
                        entry_ids: session.viewed_entries,
                    },
                ))
                .await;
            }
        }
    }

    /// Clean up expired sessions.
    pub async fn cleanup_expired_sessions(&self) {
        let expired: Vec<SessionId> = {
            let sessions = self.sessions.read().await;
            sessions
                .iter()
                .filter(|(_, s)| s.is_expired(self.config.session_timeout))
                .map(|(id, _)| *id)
                .collect()
        };

        for session_id in expired {
            self.end_session(session_id).await;
        }
    }

    /// Drain all buffered signals for processing.
    pub async fn drain_signals(&self) -> Vec<FeedbackSignal> {
        let mut signals = self.signals.write().await;
        signals.drain(..).collect()
    }

    /// Get number of buffered signals.
    pub async fn pending_signals(&self) -> usize {
        self.signals.read().await.len()
    }

    /// Get total signals collected.
    pub fn total_signals(&self) -> u64 {
        self.total_signals.load(Ordering::Relaxed)
    }

    /// Get total sessions created.
    pub fn total_sessions(&self) -> u64 {
        self.total_sessions.load(Ordering::Relaxed)
    }

    /// Get number of active sessions.
    pub async fn active_sessions(&self) -> usize {
        self.sessions.read().await.len()
    }

    /// Internal: emit a signal to the buffer.
    async fn emit_signal(&self, signal: FeedbackSignal) {
        let mut signals = self.signals.write().await;
        signals.push_back(signal);
        self.total_signals.fetch_add(1, Ordering::Relaxed);

        // Auto-cleanup if buffer is too large
        if signals.len() > self.config.max_buffer_size {
            // Remove oldest 10%
            let to_remove = self.config.max_buffer_size / 10;
            for _ in 0..to_remove {
                signals.pop_front();
            }
        }
    }
}

impl Default for FeedbackCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Processed feedback ready for the learning engine.
#[derive(Debug, Clone)]
pub struct ProcessedFeedback {
    /// Entry ID.
    pub entry_id: Uuid,
    /// Relevance delta (-1.0 to 1.0).
    pub relevance_delta: f32,
    /// Query embedding this feedback relates to.
    pub query_embedding: Option<Vec<f32>>,
    /// Confidence in this signal (0.0 to 1.0).
    pub confidence: f32,
}

/// Process raw signals into learning updates.
pub struct FeedbackProcessor {
    /// Weight for view signals.
    pub view_weight: f32,
    /// Weight for select signals.
    pub select_weight: f32,
    /// Weight for dismiss signals.
    pub dismiss_weight: f32,
    /// Position decay factor (earlier positions worth more).
    pub position_decay: f32,
    /// Dwell time normalization (milliseconds for full weight).
    pub dwell_normalization_ms: f32,
}

impl Default for FeedbackProcessor {
    fn default() -> Self {
        Self {
            view_weight: 0.1,
            select_weight: 0.5,
            dismiss_weight: -0.3,
            position_decay: 0.9,
            dwell_normalization_ms: 30_000.0, // 30 seconds
        }
    }
}

impl FeedbackProcessor {
    /// Process signals into learning updates.
    pub fn process(&self, signals: Vec<FeedbackSignal>) -> Vec<ProcessedFeedback> {
        let mut feedback = Vec::new();
        let mut query_embeddings: HashMap<QueryId, Option<Vec<f32>>> = HashMap::new();

        // First pass: collect query embeddings
        for signal in &signals {
            if let SignalType::Query {
                embedding,
                result_ids: _,
                text: _,
            } = &signal.signal
                && let Some(query_id) = signal.query_id
            {
                query_embeddings.insert(query_id, embedding.clone());
            }
        }

        // Second pass: process interaction signals
        for signal in signals {
            let query_embedding = signal
                .query_id
                .and_then(|qid| query_embeddings.get(&qid).cloned())
                .flatten();

            match signal.signal {
                SignalType::View { entry_id, position } => {
                    let position_factor = self.position_decay.powi(position as i32);
                    feedback.push(ProcessedFeedback {
                        entry_id,
                        relevance_delta: self.view_weight * position_factor,
                        query_embedding,
                        confidence: 0.3 * position_factor,
                    });
                }

                SignalType::Select { entry_id, position } => {
                    let position_factor = self.position_decay.powi(position as i32);
                    feedback.push(ProcessedFeedback {
                        entry_id,
                        relevance_delta: self.select_weight * position_factor,
                        query_embedding,
                        confidence: 0.8,
                    });
                }

                SignalType::Dismiss { entry_id, position } => {
                    let position_factor = self.position_decay.powi(position as i32);
                    feedback.push(ProcessedFeedback {
                        entry_id,
                        relevance_delta: self.dismiss_weight * position_factor,
                        query_embedding,
                        confidence: 0.6,
                    });
                }

                SignalType::Dwell {
                    entry_id,
                    duration_ms,
                } => {
                    let dwell_factor = (duration_ms as f32 / self.dwell_normalization_ms).min(1.0);
                    feedback.push(ProcessedFeedback {
                        entry_id,
                        relevance_delta: self.select_weight * dwell_factor,
                        query_embedding,
                        confidence: 0.5 * dwell_factor,
                    });
                }

                SignalType::CoAccess { entry_ids } => {
                    // For co-access, create pairwise positive signals
                    // (handled separately by the learning engine)
                    for i in 0..entry_ids.len() {
                        for j in (i + 1)..entry_ids.len() {
                            feedback.push(ProcessedFeedback {
                                entry_id: entry_ids[i],
                                relevance_delta: 0.2, // Moderate co-access boost
                                query_embedding: None,
                                confidence: 0.4,
                            });
                            feedback.push(ProcessedFeedback {
                                entry_id: entry_ids[j],
                                relevance_delta: 0.2,
                                query_embedding: None,
                                confidence: 0.4,
                            });
                        }
                    }
                }

                // Query, FollowUp, SessionEnd don't directly produce ProcessedFeedback
                _ => {}
            }
        }

        // Aggregate feedback for same entry
        self.aggregate_feedback(feedback)
    }

    /// Aggregate multiple feedback signals for the same entry.
    fn aggregate_feedback(&self, feedback: Vec<ProcessedFeedback>) -> Vec<ProcessedFeedback> {
        let mut by_entry: HashMap<Uuid, Vec<ProcessedFeedback>> = HashMap::new();

        for fb in feedback {
            by_entry.entry(fb.entry_id).or_default().push(fb);
        }

        by_entry
            .into_iter()
            .map(|(entry_id, signals)| {
                let total_weight: f32 = signals.iter().map(|s| s.confidence).sum();
                let weighted_delta: f32 = signals
                    .iter()
                    .map(|s| s.relevance_delta * s.confidence)
                    .sum::<f32>()
                    / total_weight.max(0.001);

                let avg_confidence = total_weight / signals.len() as f32;

                // Use the last query embedding (most recent context)
                let query_embedding = signals.into_iter().rev().find_map(|s| s.query_embedding);

                ProcessedFeedback {
                    entry_id,
                    relevance_delta: weighted_delta,
                    query_embedding,
                    confidence: avg_confidence,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_lifecycle() {
        let collector = FeedbackCollector::new();

        let session = collector.start_session().await;
        assert_eq!(collector.active_sessions().await, 1);

        collector.end_session(session).await;
        assert_eq!(collector.active_sessions().await, 0);
    }

    #[tokio::test]
    async fn test_query_recording() {
        let collector = FeedbackCollector::new();
        let session = collector.start_session().await;

        let entry1 = Uuid::new_v4();
        let entry2 = Uuid::new_v4();

        let _query_id = collector
            .record_query(session, "test query", vec![entry1, entry2], None)
            .await;

        let signals = collector.drain_signals().await;
        assert_eq!(signals.len(), 1);

        match &signals[0].signal {
            SignalType::Query {
                text, result_ids, ..
            } => {
                assert_eq!(text, "test query");
                assert_eq!(result_ids.len(), 2);
            }
            _ => panic!("Expected Query signal"),
        }
    }

    #[tokio::test]
    async fn test_view_and_select() {
        let collector = FeedbackCollector::new();
        let session = collector.start_session().await;

        let entry1 = Uuid::new_v4();
        let entry2 = Uuid::new_v4();

        collector
            .record_query(session, "test", vec![entry1, entry2], None)
            .await;
        collector.record_view(session, entry1, 0).await;
        collector.record_select(session, entry1, 0).await;

        let signals = collector.drain_signals().await;
        assert_eq!(signals.len(), 3); // query + view + select
    }

    #[tokio::test]
    async fn test_follow_up_detection() {
        let collector = FeedbackCollector::new();
        let session = collector.start_session().await;

        let entry1 = Uuid::new_v4();

        // First query
        collector
            .record_query(session, "first query", vec![entry1], None)
            .await;
        collector.record_view(session, entry1, 0).await;

        // Follow-up query
        collector
            .record_query(session, "follow up query", vec![], None)
            .await;

        let signals = collector.drain_signals().await;

        // Should have: query1, view, follow_up, query2
        let follow_up = signals
            .iter()
            .find(|s| matches!(s.signal, SignalType::FollowUp { .. }));
        assert!(follow_up.is_some());
    }

    #[tokio::test]
    async fn test_co_access_on_session_end() {
        let config = FeedbackConfig {
            co_access_min_views: 2,
            ..FeedbackConfig::default()
        };
        let collector = FeedbackCollector::with_config(config);

        let session = collector.start_session().await;

        let entry1 = Uuid::new_v4();
        let entry2 = Uuid::new_v4();

        collector
            .record_query(session, "test", vec![entry1, entry2], None)
            .await;
        collector.record_view(session, entry1, 0).await;
        collector.record_view(session, entry2, 1).await;

        collector.end_session(session).await;

        let signals = collector.drain_signals().await;

        let co_access = signals
            .iter()
            .find(|s| matches!(s.signal, SignalType::CoAccess { .. }));
        assert!(co_access.is_some());
    }

    #[tokio::test]
    async fn test_dwell_filtering() {
        let config = FeedbackConfig {
            min_dwell_ms: 500,
            ..FeedbackConfig::default()
        };
        let collector = FeedbackCollector::with_config(config);

        let session = collector.start_session().await;
        let entry = Uuid::new_v4();

        // Too short - should be filtered
        collector.record_dwell(session, entry, 100).await;
        assert_eq!(collector.pending_signals().await, 0);

        // Long enough - should be recorded
        collector.record_dwell(session, entry, 1000).await;
        assert_eq!(collector.pending_signals().await, 1);
    }

    #[test]
    fn test_feedback_processing() {
        let processor = FeedbackProcessor::default();

        let entry = Uuid::new_v4();
        let signals = vec![
            FeedbackSignal::new(
                SessionId::new(),
                Some(QueryId::new()),
                SignalType::View {
                    entry_id: entry,
                    position: 0,
                },
            ),
            FeedbackSignal::new(
                SessionId::new(),
                Some(QueryId::new()),
                SignalType::Select {
                    entry_id: entry,
                    position: 0,
                },
            ),
        ];

        let processed = processor.process(signals);
        assert_eq!(processed.len(), 1); // Aggregated

        // Should be positive (view + select)
        assert!(processed[0].relevance_delta > 0.0);
    }

    #[test]
    fn test_position_decay() {
        let processor = FeedbackProcessor::default();

        let entry1 = Uuid::new_v4();
        let entry2 = Uuid::new_v4();

        let signals = vec![
            FeedbackSignal::new(
                SessionId::new(),
                Some(QueryId::new()),
                SignalType::Select {
                    entry_id: entry1,
                    position: 0,
                },
            ),
            FeedbackSignal::new(
                SessionId::new(),
                Some(QueryId::new()),
                SignalType::Select {
                    entry_id: entry2,
                    position: 5,
                },
            ),
        ];

        let processed = processor.process(signals);
        assert_eq!(processed.len(), 2);

        let fb1 = processed.iter().find(|p| p.entry_id == entry1).unwrap();
        let fb2 = processed.iter().find(|p| p.entry_id == entry2).unwrap();

        // Position 0 should have higher delta than position 5
        assert!(fb1.relevance_delta > fb2.relevance_delta);
    }
}
