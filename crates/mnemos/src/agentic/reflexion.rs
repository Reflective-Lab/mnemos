//! Reflexion Episodes - Self-Critique Memory
//!
//! Implements the Reflexion pattern from "Reflexion: Language Agents with Verbal Reinforcement Learning"
//! (Shinn et al., 2023). This allows agents to:
//!
//! 1. Attempt a task
//! 2. If failed, generate a critique of what went wrong
//! 3. Store the critique for future reference
//! 4. When attempting similar tasks, retrieve past critiques to avoid mistakes
//!
//! # Example
//!
//! ```rust,no_run
//! use mnemos::agentic::{ReflexionEpisode, Critique, CritiqueType};
//!
//! // Suggestor tried to implement sorting but made an error
//! let episode = ReflexionEpisode::new(
//!     "algorithm_implementation",
//!     "Implement quicksort",
//!     "fn quicksort(arr: &mut [i32]) { /* buggy code */ }",
//!     false, // failed
//! )
//! .with_critique(Critique::new(
//!     CritiqueType::LogicError,
//!     "Partition function doesn't handle equal elements",
//!     "Use <= instead of < in comparison",
//! ));
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A reflexion episode capturing a task attempt and self-critique.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflexionEpisode {
    /// Unique identifier.
    pub id: Uuid,

    /// Type of task attempted.
    pub task_type: String,

    /// Original goal or instruction.
    pub goal: String,

    /// What the agent actually did/produced.
    pub attempt: String,

    /// Whether the attempt succeeded.
    pub succeeded: bool,

    /// Self-critiques identifying what went wrong.
    pub critiques: Vec<Critique>,

    /// Retry count (how many times this was attempted).
    pub retry_count: u32,

    /// When this episode occurred.
    pub timestamp: DateTime<Utc>,

    /// Embedding of the goal for similarity search.
    #[serde(skip)]
    pub goal_embedding: Option<Vec<f32>>,
}

impl ReflexionEpisode {
    /// Create a new reflexion episode.
    pub fn new(
        task_type: impl Into<String>,
        goal: impl Into<String>,
        attempt: impl Into<String>,
        succeeded: bool,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            task_type: task_type.into(),
            goal: goal.into(),
            attempt: attempt.into(),
            succeeded,
            critiques: Vec::new(),
            retry_count: 0,
            timestamp: Utc::now(),
            goal_embedding: None,
        }
    }

    /// Add a self-critique.
    pub fn with_critique(mut self, critique: Critique) -> Self {
        self.critiques.push(critique);
        self
    }

    /// Set retry count.
    pub fn with_retry_count(mut self, count: u32) -> Self {
        self.retry_count = count;
        self
    }

    /// Set the goal embedding for similarity search.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.goal_embedding = Some(embedding);
        self
    }

    /// Get a summary of all critiques.
    pub fn critique_summary(&self) -> String {
        self.critiques
            .iter()
            .map(|c| format!("[{}] {}: {}", c.critique_type, c.issue, c.suggestion))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A self-critique identifying a specific issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Critique {
    /// Type of issue identified.
    pub critique_type: CritiqueType,

    /// Description of what went wrong.
    pub issue: String,

    /// Suggested fix or improvement.
    pub suggestion: String,

    /// Confidence in this critique (0.0 to 1.0).
    pub confidence: f32,
}

impl Critique {
    /// Create a new critique.
    pub fn new(
        critique_type: CritiqueType,
        issue: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self {
            critique_type,
            issue: issue.into(),
            suggestion: suggestion.into(),
            confidence: 1.0,
        }
    }

    /// Set confidence level.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Types of critiques that can be identified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CritiqueType {
    /// Logical error in reasoning or code.
    LogicError,

    /// Missing step in the process.
    MissingStep,

    /// Syntax or formatting error.
    SyntaxError,

    /// Design or architectural flaw.
    DesignFlaw,

    /// Edge case not handled.
    EdgeCase,

    /// Performance issue.
    Performance,

    /// Security vulnerability.
    Security,

    /// Misunderstood requirements.
    Misunderstanding,

    /// Wrong tool or approach used.
    WrongApproach,

    /// Other issue type.
    Other,
}

impl std::fmt::Display for CritiqueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CritiqueType::LogicError => write!(f, "LogicError"),
            CritiqueType::MissingStep => write!(f, "MissingStep"),
            CritiqueType::SyntaxError => write!(f, "SyntaxError"),
            CritiqueType::DesignFlaw => write!(f, "DesignFlaw"),
            CritiqueType::EdgeCase => write!(f, "EdgeCase"),
            CritiqueType::Performance => write!(f, "Performance"),
            CritiqueType::Security => write!(f, "Security"),
            CritiqueType::Misunderstanding => write!(f, "Misunderstanding"),
            CritiqueType::WrongApproach => write!(f, "WrongApproach"),
            CritiqueType::Other => write!(f, "Other"),
        }
    }
}

/// Memory store for reflexion episodes.
pub struct ReflexionMemory {
    episodes: Vec<ReflexionEpisode>,
}

impl ReflexionMemory {
    /// Create a new reflexion memory.
    pub fn new() -> Self {
        Self {
            episodes: Vec::new(),
        }
    }

    /// Add an episode to memory.
    pub fn add_episode(&mut self, episode: ReflexionEpisode) {
        self.episodes.push(episode);
    }

    /// Find similar past failures for a given task.
    ///
    /// This uses simple keyword matching. In production, use embedding similarity.
    pub fn find_similar_failures(&self, task: &str, limit: usize) -> Vec<ReflexionEpisode> {
        let task_lower = task.to_lowercase();
        let keywords: Vec<&str> = task_lower.split_whitespace().collect();

        let mut scored: Vec<(f32, &ReflexionEpisode)> = self
            .episodes
            .iter()
            .filter(|e| !e.succeeded) // Only failures
            .map(|e| {
                let goal_lower = e.goal.to_lowercase();
                let type_lower = e.task_type.to_lowercase();

                // Simple keyword matching score
                let score: f32 = keywords
                    .iter()
                    .map(|k| {
                        if goal_lower.contains(k) || type_lower.contains(k) {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum();

                (score, e)
            })
            .filter(|(score, _)| *score > 0.0)
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));

        scored
            .into_iter()
            .take(limit)
            .map(|(_, e)| e.clone())
            .collect()
    }

    /// Get all episodes of a specific type.
    pub fn get_by_type(&self, task_type: &str) -> Vec<&ReflexionEpisode> {
        self.episodes
            .iter()
            .filter(|e| e.task_type == task_type)
            .collect()
    }

    /// Get recent failures.
    pub fn recent_failures(&self, limit: usize) -> Vec<&ReflexionEpisode> {
        let mut failures: Vec<_> = self.episodes.iter().filter(|e| !e.succeeded).collect();
        failures.sort_by_key(|f| std::cmp::Reverse(f.timestamp));
        failures.into_iter().take(limit).collect()
    }

    /// Total episode count.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Count of failed episodes.
    pub fn failure_count(&self) -> usize {
        self.episodes.iter().filter(|e| !e.succeeded).count()
    }

    /// Success rate.
    pub fn success_rate(&self) -> f32 {
        if self.episodes.is_empty() {
            return 0.0;
        }
        let successes = self.episodes.iter().filter(|e| e.succeeded).count();
        successes as f32 / self.episodes.len() as f32
    }
}

impl Default for ReflexionMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: Basic reflexion episode creation.
    ///
    /// What happens:
    /// 1. Create an episode describing a failed attempt
    /// 2. Add critiques explaining what went wrong
    /// 3. The episode captures the full context for future learning
    #[test]
    fn test_episode_creation() {
        let episode = ReflexionEpisode::new(
            "code_review",
            "Review pull request #123",
            "Approved without noticing the SQL injection",
            false,
        )
        .with_critique(Critique::new(
            CritiqueType::Security,
            "Missed SQL injection vulnerability in user input",
            "Always check for unsanitized inputs in database queries",
        ))
        .with_retry_count(2);

        assert_eq!(episode.task_type, "code_review");
        assert!(!episode.succeeded);
        assert_eq!(episode.critiques.len(), 1);
        assert_eq!(episode.retry_count, 2);
    }

    /// Test: Finding similar failures.
    ///
    /// What happens:
    /// 1. Store multiple failed episodes
    /// 2. Search for episodes similar to a new task
    /// 3. Return episodes that match keywords
    /// 4. Suggestor can learn from past mistakes
    #[test]
    fn test_find_similar_failures() {
        let mut memory = ReflexionMemory::new();

        // Add some failures
        memory.add_episode(
            ReflexionEpisode::new(
                "sql_query",
                "Write SQL query for user search",
                "SELECT * FROM users WHERE name = '{input}'",
                false,
            )
            .with_critique(Critique::new(
                CritiqueType::Security,
                "SQL injection possible",
                "Use parameterized queries",
            )),
        );

        memory.add_episode(
            ReflexionEpisode::new(
                "api_design",
                "Design REST API for payments",
                "POST /pay without authentication",
                false,
            )
            .with_critique(Critique::new(
                CritiqueType::Security,
                "No auth on sensitive endpoint",
                "Add authentication middleware",
            )),
        );

        // Search for SQL-related failures
        let similar = memory.find_similar_failures("SQL query for orders", 5);
        assert!(!similar.is_empty());
        // SQL query should be the top result
        assert!(similar.iter().any(|e| e.task_type == "sql_query"));

        // Search for API-related failures
        let similar = memory.find_similar_failures("REST API endpoint", 5);
        assert!(!similar.is_empty());
        // API design should be in the results
        assert!(similar.iter().any(|e| e.task_type == "api_design"));
    }

    /// Test: Success rate tracking.
    ///
    /// What happens:
    /// 1. Track both successes and failures
    /// 2. Calculate success rate
    /// 3. Suggestor can see improvement over time
    #[test]
    fn test_success_rate() {
        let mut memory = ReflexionMemory::new();

        // Add 3 failures
        for _ in 0..3 {
            memory.add_episode(ReflexionEpisode::new("test", "goal", "attempt", false));
        }

        // Add 1 success
        memory.add_episode(ReflexionEpisode::new("test", "goal", "attempt", true));

        assert_eq!(memory.len(), 4);
        assert_eq!(memory.failure_count(), 3);
        assert!((memory.success_rate() - 0.25).abs() < 0.01);
    }
}
