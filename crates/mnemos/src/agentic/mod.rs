//! AgenticDB: Suggestor Memory System
//!
//! Implements the 5-table architecture from ruvector for AI agent memory:
//!
//! 1. **Reflexion Episodes** - Self-critique memory for learning from mistakes
//! 2. **Skill Library** - Consolidated successful patterns
//! 3. **Causal Memory** - Hypergraph relationships between concepts
//! 4. **Learning Sessions** - RL training data with rewards
//! 5. **Vector Store** - Core embeddings (handled by main KnowledgeBase)
//!
//! Plus advanced learning mechanisms:
//!
//! - **Temporal Patterns** - Time crystals for periodic behavior detection
//! - **Online Learning** - Continual adaptation with EWC forgetting prevention
//! - **Meta-Learning** - Learning to learn with MAML/Reptile-style adaptation

mod causal;
mod meta;
mod online;
mod reflexion;
mod sessions;
mod skills;
mod temporal;

pub use causal::{CausalEdge, CausalMemory, CausalNode, Hyperedge};
pub use meta::{FewShotLearner, LearningStrategy, MetaLearner, TaskFeatures};
pub use online::{DriftDetector, Experience, ExperienceWindow, OnlineLearner, ParameterSnapshot};
pub use reflexion::{Critique, CritiqueType, ReflexionEpisode, ReflexionMemory};
pub use sessions::{LearningSession, Reward, SessionTurn};
pub use skills::{Skill, SkillLibrary, SkillPattern};
pub use temporal::{TemporalMemory, TemporalOccurrence, TemporalPeriod, TimeCrystal};

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// AgenticDB: Complete agent memory system.
///
/// Provides persistent memory for AI agents with:
/// - Self-reflection and learning from mistakes
/// - Skill consolidation and reuse
/// - Causal reasoning over relationships
/// - Reinforcement learning from sessions
/// - Temporal pattern detection (time crystals)
/// - Online/continual learning with forgetting prevention
/// - Meta-learning for quick adaptation
pub struct AgenticDB {
    /// Reflexion episodes for self-critique.
    pub reflexion: Arc<RwLock<ReflexionMemory>>,

    /// Library of learned skills.
    pub skills: Arc<RwLock<SkillLibrary>>,

    /// Causal relationships between concepts.
    pub causal: Arc<RwLock<CausalMemory>>,

    /// Learning sessions with rewards.
    pub sessions: Arc<RwLock<Vec<LearningSession>>>,

    /// Temporal patterns (time crystals).
    pub temporal: Arc<RwLock<TemporalMemory>>,

    /// Meta-learner for quick task adaptation.
    pub meta: Arc<RwLock<MetaLearner>>,

    /// Experience window for continual learning.
    pub experiences: Arc<RwLock<ExperienceWindow>>,

    /// Distribution drift detector.
    pub drift_detector: Arc<RwLock<DriftDetector>>,

    /// Storage path.
    #[allow(dead_code)]
    path: Option<String>,
}

impl AgenticDB {
    /// Create a new in-memory AgenticDB.
    ///
    /// # Arguments
    /// * `meta_param_dim` - Dimension of meta-learning parameters (default 64)
    /// * `drift_feature_dim` - Dimension of features for drift detection (default 64)
    pub fn new() -> Self {
        Self::with_dimensions(64, 64)
    }

    /// Create with specific dimensions for meta-learning.
    pub fn with_dimensions(meta_param_dim: usize, drift_feature_dim: usize) -> Self {
        Self {
            reflexion: Arc::new(RwLock::new(ReflexionMemory::new())),
            skills: Arc::new(RwLock::new(SkillLibrary::new())),
            causal: Arc::new(RwLock::new(CausalMemory::new())),
            sessions: Arc::new(RwLock::new(Vec::new())),
            temporal: Arc::new(RwLock::new(TemporalMemory::new())),
            meta: Arc::new(RwLock::new(MetaLearner::new("agent_meta", meta_param_dim))),
            experiences: Arc::new(RwLock::new(ExperienceWindow::new(1000))),
            drift_detector: Arc::new(RwLock::new(DriftDetector::new(drift_feature_dim))),
            path: None,
        }
    }

    /// Create with persistence.
    pub fn with_path(path: impl Into<String>) -> Self {
        Self {
            reflexion: Arc::new(RwLock::new(ReflexionMemory::new())),
            skills: Arc::new(RwLock::new(SkillLibrary::new())),
            causal: Arc::new(RwLock::new(CausalMemory::new())),
            sessions: Arc::new(RwLock::new(Vec::new())),
            temporal: Arc::new(RwLock::new(TemporalMemory::new())),
            meta: Arc::new(RwLock::new(MetaLearner::new("agent_meta", 64))),
            experiences: Arc::new(RwLock::new(ExperienceWindow::new(1000))),
            drift_detector: Arc::new(RwLock::new(DriftDetector::new(64))),
            path: Some(path.into()),
        }
    }

    /// Record a reflexion episode (self-critique).
    ///
    /// # Example
    /// ```rust,no_run
    /// use mnemos::agentic::{AgenticDB, ReflexionEpisode, Critique, CritiqueType};
    ///
    /// # async fn example() {
    /// let db = AgenticDB::new();
    ///
    /// // Suggestor attempted something and failed
    /// let episode = ReflexionEpisode::new(
    ///     "write_code",
    ///     "Write a function to parse JSON",
    ///     "fn parse() { /* incomplete */ }",
    ///     false, // did not succeed
    /// )
    /// .with_critique(Critique::new(
    ///     CritiqueType::MissingStep,
    ///     "Did not handle error cases",
    ///     "Add Result return type and error handling",
    /// ));
    ///
    /// db.add_reflexion(episode).await;
    /// # }
    /// ```
    pub async fn add_reflexion(&self, episode: ReflexionEpisode) {
        let mut reflexion = self.reflexion.write().await;
        reflexion.add_episode(episode);
    }

    /// Query similar past failures to avoid repeating mistakes.
    pub async fn query_similar_failures(&self, task: &str, limit: usize) -> Vec<ReflexionEpisode> {
        let reflexion = self.reflexion.read().await;
        reflexion.find_similar_failures(task, limit)
    }

    /// Register a successful skill pattern.
    pub async fn register_skill(&self, skill: Skill) {
        let mut skills = self.skills.write().await;
        skills.add_skill(skill);
    }

    /// Find applicable skills for a task.
    pub async fn find_skills(&self, _task_description: &str) -> Vec<&Skill> {
        // Note: This is a simplified version
        Vec::new()
    }

    /// Add a causal relationship.
    pub async fn add_causal_link(
        &self,
        cause: Uuid,
        effect: Uuid,
        relationship: impl Into<String>,
        strength: f32,
    ) {
        let mut causal = self.causal.write().await;
        causal.add_edge(CausalEdge {
            id: Uuid::new_v4(),
            cause,
            effect,
            relationship: relationship.into(),
            strength,
            evidence_count: 1,
        });
    }

    /// Start a new learning session.
    pub async fn start_session(&self, goal: impl Into<String>) -> Uuid {
        let session = LearningSession::new(goal);
        let id = session.id;
        let mut sessions = self.sessions.write().await;
        sessions.push(session);
        id
    }

    /// Record a turn in a session.
    pub async fn record_turn(
        &self,
        session_id: Uuid,
        action: impl Into<String>,
        observation: impl Into<String>,
        reward: Reward,
    ) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.iter_mut().find(|s| s.id == session_id) {
            session.add_turn(SessionTurn {
                action: action.into(),
                observation: observation.into(),
                reward,
                timestamp: chrono::Utc::now(),
            });
        }
    }

    /// Get statistics about the agent memory.
    pub async fn stats(&self) -> AgenticStats {
        let reflexion = self.reflexion.read().await;
        let skills = self.skills.read().await;
        let causal = self.causal.read().await;
        let sessions = self.sessions.read().await;
        let temporal = self.temporal.read().await;
        let meta = self.meta.read().await;
        let experiences = self.experiences.read().await;

        AgenticStats {
            reflexion_episodes: reflexion.len(),
            failed_episodes: reflexion.failure_count(),
            skills_count: skills.len(),
            causal_nodes: causal.node_count(),
            causal_edges: causal.edge_count(),
            total_sessions: sessions.len(),
            total_turns: sessions.iter().map(|s| s.turns.len()).sum(),
            temporal_patterns: temporal.len(),
            meta_tasks_learned: meta.num_tasks() as usize,
            learning_strategies: meta.num_strategies(),
            experience_buffer_size: experiences.len(),
        }
    }

    /// Record a temporal pattern (time crystal).
    ///
    /// Use this to track periodic behavior patterns like:
    /// - Daily coding activity
    /// - Weekly deployment schedules
    /// - Monthly review cycles
    pub async fn record_temporal(&self, pattern_name: &str, period: TemporalPeriod, value: f32) {
        let mut temporal = self.temporal.write().await;
        temporal.record(pattern_name, period, value);
    }

    /// Predict activity for a temporal pattern.
    pub async fn predict_temporal(&self, pattern_name: &str) -> Option<f32> {
        let temporal = self.temporal.read().await;
        temporal.predict(pattern_name)
    }

    /// Add experience for continual learning.
    pub async fn add_experience(
        &mut self,
        features: Vec<f32>,
        target: f32,
        task_id: Option<String>,
    ) {
        let mut experiences = self.experiences.write().await;
        experiences.add(features, target, task_id);
    }

    /// Check for distribution drift.
    ///
    /// Returns true if the current features indicate a distribution shift
    /// that may require model adaptation.
    pub async fn check_drift(&self, features: &[f32]) -> bool {
        let mut detector = self.drift_detector.write().await;
        detector.update(features)
    }

    /// Update meta-learner after completing a task.
    pub async fn meta_update(
        &self,
        task_id: &str,
        final_params: &[f32],
        task_embedding: Option<Vec<f32>>,
    ) {
        let mut meta = self.meta.write().await;
        meta.meta_update(task_id, final_params, task_embedding);
    }

    /// Get initialization parameters for a new task.
    pub async fn get_task_initialization(&self, task_embedding: Option<&[f32]>) -> Vec<f32> {
        let meta = self.meta.read().await;
        meta.initialize_for_task(task_embedding)
    }

    /// Register a learning strategy.
    pub async fn register_strategy(&self, strategy: LearningStrategy) {
        let mut meta = self.meta.write().await;
        meta.register_strategy(strategy);
    }

    /// Select best strategy for a task.
    pub async fn select_strategy(&self, task_features: &TaskFeatures) -> Option<String> {
        let meta = self.meta.read().await;
        meta.select_strategy(task_features).map(|s| s.name.clone())
    }
}

impl Default for AgenticDB {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the agent memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgenticStats {
    /// Total reflexion episodes recorded.
    pub reflexion_episodes: usize,
    /// Number of failed episodes.
    pub failed_episodes: usize,
    /// Number of skills in library.
    pub skills_count: usize,
    /// Number of causal nodes.
    pub causal_nodes: usize,
    /// Number of causal edges.
    pub causal_edges: usize,
    /// Total learning sessions.
    pub total_sessions: usize,
    /// Total turns across all sessions.
    pub total_turns: usize,
    /// Number of temporal patterns (time crystals).
    pub temporal_patterns: usize,
    /// Number of tasks the meta-learner has learned.
    pub meta_tasks_learned: usize,
    /// Number of learning strategies discovered.
    pub learning_strategies: usize,
    /// Current experience buffer size.
    pub experience_buffer_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: Creating an AgenticDB and recording a reflexion episode.
    ///
    /// This demonstrates how an AI agent can:
    /// 1. Attempt a task
    /// 2. Fail and record what went wrong
    /// 3. Store a self-critique for future reference
    ///
    /// The agent can later query these failures to avoid repeating mistakes.
    #[tokio::test]
    async fn test_reflexion_workflow() {
        // Create a new agent memory database
        let db = AgenticDB::new();

        // Scenario: Suggestor tried to write code but made an error
        let episode = ReflexionEpisode::new(
            "code_generation",                                    // task type
            "Write a function to calculate factorial",            // original goal
            "fn factorial(n: i32) -> i32 { n * factorial(n-1) }", // what was attempted
            false,                                                // did it succeed? No!
        )
        .with_critique(Critique::new(
            CritiqueType::LogicError,
            "Missing base case causes infinite recursion", // what went wrong
            "Add: if n <= 1 { return 1; }",                // how to fix
        ))
        .with_critique(Critique::new(
            CritiqueType::MissingStep,
            "No handling for negative numbers",
            "Add input validation or use unsigned type",
        ));

        // Store the episode
        db.add_reflexion(episode).await;

        // Verify it was stored
        let stats = db.stats().await;
        assert_eq!(stats.reflexion_episodes, 1);
        assert_eq!(stats.failed_episodes, 1);

        // Later, when attempting a similar task, query for past failures
        let similar = db.query_similar_failures("factorial function", 5).await;
        assert_eq!(similar.len(), 1);

        // The agent can now learn from past mistakes!
        let past_mistake = &similar[0];
        assert!(!past_mistake.succeeded);
        assert_eq!(past_mistake.critiques.len(), 2);
    }

    /// Test: Building a skill library from successful patterns.
    ///
    /// When an agent successfully completes a task, it can:
    /// 1. Extract the successful pattern
    /// 2. Store it in the skill library
    /// 3. Reuse it for similar future tasks
    #[tokio::test]
    async fn test_skill_library() {
        let db = AgenticDB::new();

        // Suggestor successfully completed a task - consolidate as a skill
        let skill = Skill::new(
            "error_handling",
            "Rust Error Handling Pattern",
            vec![
                SkillPattern::new("result_type", "fn do_thing() -> Result<T, Error> { ... }"),
                SkillPattern::new("question_mark", "let value = risky_op()?;"),
            ],
        )
        .with_success_rate(0.95)
        .with_usage_count(42);

        db.register_skill(skill).await;

        let stats = db.stats().await;
        assert_eq!(stats.skills_count, 1);
    }

    /// Test: Recording causal relationships.
    ///
    /// The agent can build a knowledge graph of cause-effect relationships:
    /// - "Using unwrap() causes panic on None"
    /// - "Adding tests causes higher code quality"
    #[tokio::test]
    async fn test_causal_memory() {
        let db = AgenticDB::new();

        let cause_id = Uuid::new_v4();
        let effect_id = Uuid::new_v4();

        // Record: "Using unwrap() → can cause panic"
        db.add_causal_link(
            cause_id, effect_id, "causes", 0.8, // 80% confidence
        )
        .await;

        let stats = db.stats().await;
        assert_eq!(stats.causal_edges, 1);
    }

    /// Test: Recording a learning session with rewards.
    ///
    /// This simulates reinforcement learning:
    /// 1. Suggestor starts a session with a goal
    /// 2. Takes actions and observes results
    /// 3. Receives rewards (positive or negative)
    /// 4. Learns from the trajectory
    #[tokio::test]
    async fn test_learning_session() {
        let db = AgenticDB::new();

        // Start a session
        let session_id = db.start_session("Fix the bug in auth module").await;

        // Turn 1: Suggestor reads the code
        db.record_turn(
            session_id,
            "read_file auth.rs",
            "Found potential null check issue on line 42",
            Reward::Neutral,
        )
        .await;

        // Turn 2: Suggestor makes a fix
        db.record_turn(
            session_id,
            "edit auth.rs: add None check",
            "File updated successfully",
            Reward::Positive(0.5),
        )
        .await;

        // Turn 3: Suggestor runs tests
        db.record_turn(
            session_id,
            "run tests",
            "All 15 tests passing",
            Reward::Positive(1.0), // High reward for success!
        )
        .await;

        let stats = db.stats().await;
        assert_eq!(stats.total_sessions, 1);
        assert_eq!(stats.total_turns, 3);
    }

    /// Test: Complete workflow demonstrating all 5 tables working together.
    ///
    /// Scenario: Suggestor is asked to implement a feature.
    /// 1. Check past failures (Reflexion)
    /// 2. Find applicable skills (Skill Library)
    /// 3. Understand related concepts (Causal Memory)
    /// 4. Execute and learn (Learning Session)
    /// 5. Store embeddings (Vector Store - via main KB)
    #[tokio::test]
    async fn test_integrated_workflow() {
        let db = AgenticDB::new();

        // Step 1: Record a past failure to learn from
        let past_failure = ReflexionEpisode::new(
            "api_design",
            "Design REST API endpoint",
            "POST /users with no validation",
            false,
        )
        .with_critique(Critique::new(
            CritiqueType::DesignFlaw,
            "No input validation leads to security issues",
            "Always validate and sanitize inputs",
        ));
        db.add_reflexion(past_failure).await;

        // Step 2: Register a successful skill
        let skill = Skill::new(
            "api_validation",
            "Input Validation Pattern",
            vec![SkillPattern::new("validate_first", "validate(&input)?;")],
        );
        db.register_skill(skill).await;

        // Step 3: Add causal knowledge
        let validation_id = Uuid::new_v4();
        let security_id = Uuid::new_v4();
        db.add_causal_link(validation_id, security_id, "improves", 0.9)
            .await;

        // Step 4: Start a new session for the current task
        let session_id = db
            .start_session("Implement user registration endpoint")
            .await;

        // Suggestor checks past failures first
        let failures = db.query_similar_failures("api endpoint", 5).await;
        assert!(!failures.is_empty(), "Should find past API failure");

        // Suggestor uses skill and records turn
        db.record_turn(
            session_id,
            "apply validation skill",
            "Added input validation with proper error handling",
            Reward::Positive(0.8),
        )
        .await;

        // Final stats
        let stats = db.stats().await;
        assert_eq!(stats.reflexion_episodes, 1);
        assert_eq!(stats.skills_count, 1);
        assert_eq!(stats.causal_edges, 1);
        assert_eq!(stats.total_sessions, 1);
        assert_eq!(stats.total_turns, 1);

        println!("AgenticDB Stats: {:?}", stats);
    }
}
