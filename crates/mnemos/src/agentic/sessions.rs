//! Learning Sessions - RL Training Data
//!
//! Implements learning session tracking for reinforcement learning:
//! 1. Track agent actions and observations
//! 2. Record rewards (positive/negative/neutral)
//! 3. Build trajectories for training
//! 4. Support offline RL and online learning

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A learning session capturing agent interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningSession {
    /// Unique identifier.
    pub id: Uuid,

    /// Session goal or task description.
    pub goal: String,

    /// Session turns (action-observation-reward tuples).
    pub turns: Vec<SessionTurn>,

    /// Session outcome.
    pub outcome: SessionOutcome,

    /// Session start time.
    pub started_at: DateTime<Utc>,

    /// Session end time.
    pub ended_at: Option<DateTime<Utc>>,

    /// Total reward accumulated.
    pub total_reward: f32,

    /// Session metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

impl LearningSession {
    /// Create a new learning session.
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            goal: goal.into(),
            turns: Vec::new(),
            outcome: SessionOutcome::InProgress,
            started_at: Utc::now(),
            ended_at: None,
            total_reward: 0.0,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add a turn to the session.
    pub fn add_turn(&mut self, turn: SessionTurn) {
        self.total_reward += turn.reward.value();
        self.turns.push(turn);
    }

    /// Complete the session.
    pub fn complete(&mut self, success: bool) {
        self.ended_at = Some(Utc::now());
        self.outcome = if success {
            SessionOutcome::Success
        } else {
            SessionOutcome::Failure
        };
    }

    /// Abort the session.
    pub fn abort(&mut self, reason: impl Into<String>) {
        self.ended_at = Some(Utc::now());
        self.outcome = SessionOutcome::Aborted(reason.into());
    }

    /// Get session duration.
    pub fn duration(&self) -> chrono::Duration {
        let end = self.ended_at.unwrap_or_else(Utc::now);
        end - self.started_at
    }

    /// Get trajectory for RL training.
    pub fn to_trajectory(&self) -> Vec<(String, String, f32)> {
        self.turns
            .iter()
            .map(|t| (t.action.clone(), t.observation.clone(), t.reward.value()))
            .collect()
    }

    /// Calculate discounted return (for RL).
    pub fn discounted_return(&self, gamma: f32) -> f32 {
        let mut total = 0.0;

        for turn in self.turns.iter().rev() {
            total = turn.reward.value() + gamma * total;
        }

        total
    }
}

/// A single turn in a learning session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTurn {
    /// Action taken by the agent.
    pub action: String,

    /// Observation/result of the action.
    pub observation: String,

    /// Reward received.
    pub reward: Reward,

    /// Turn timestamp.
    pub timestamp: DateTime<Utc>,
}

impl SessionTurn {
    /// Create a new turn.
    pub fn new(action: impl Into<String>, observation: impl Into<String>, reward: Reward) -> Self {
        Self {
            action: action.into(),
            observation: observation.into(),
            reward,
            timestamp: Utc::now(),
        }
    }
}

/// Reward signal.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Reward {
    /// Positive reward.
    Positive(f32),

    /// Negative reward (penalty).
    Negative(f32),

    /// No reward.
    Neutral,

    /// Sparse reward at end of episode.
    Terminal(f32),
}

impl Reward {
    /// Get numeric value.
    pub fn value(&self) -> f32 {
        match self {
            Reward::Positive(v) => *v,
            Reward::Negative(v) => -*v,
            Reward::Neutral => 0.0,
            Reward::Terminal(v) => *v,
        }
    }

    /// Check if positive.
    pub fn is_positive(&self) -> bool {
        self.value() > 0.0
    }

    /// Check if negative.
    pub fn is_negative(&self) -> bool {
        self.value() < 0.0
    }
}

/// Session outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionOutcome {
    /// Session still in progress.
    InProgress,

    /// Session completed successfully.
    Success,

    /// Session failed.
    Failure,

    /// Session aborted with reason.
    Aborted(String),
}

impl SessionOutcome {
    /// Check if completed (success or failure).
    pub fn is_completed(&self) -> bool {
        matches!(self, SessionOutcome::Success | SessionOutcome::Failure)
    }

    /// Check if successful.
    pub fn is_success(&self) -> bool {
        matches!(self, SessionOutcome::Success)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: Recording a learning session.
    ///
    /// What happens:
    /// 1. Start a session with a goal
    /// 2. Record turns (action → observation → reward)
    /// 3. Track cumulative reward
    /// 4. Complete session with outcome
    #[test]
    fn test_session_recording() {
        let mut session = LearningSession::new("Find and fix the bug");

        // Turn 1: Read code
        session.add_turn(SessionTurn::new(
            "read_file main.rs",
            "Found suspicious null check on line 42",
            Reward::Positive(0.1), // Small reward for progress
        ));

        // Turn 2: Make a wrong change
        session.add_turn(SessionTurn::new(
            "edit main.rs: remove null check",
            "Compilation error: cannot assign to immutable",
            Reward::Negative(0.2), // Penalty for error
        ));

        // Turn 3: Fix the fix
        session.add_turn(SessionTurn::new(
            "edit main.rs: add mut keyword",
            "File saved successfully",
            Reward::Neutral,
        ));

        // Turn 4: Test
        session.add_turn(SessionTurn::new(
            "run_tests",
            "All 15 tests passing",
            Reward::Terminal(1.0), // Big reward for success!
        ));

        session.complete(true);

        assert_eq!(session.turns.len(), 4);
        assert!(session.outcome.is_success());

        // Check total reward: 0.1 - 0.2 + 0 + 1.0 = 0.9
        assert!((session.total_reward - 0.9).abs() < 0.01);
    }

    /// Test: Discounted return calculation.
    ///
    /// What happens:
    /// 1. Record rewards over time
    /// 2. Calculate discounted return with gamma
    /// 3. Later rewards contribute less (temporal discounting)
    #[test]
    fn test_discounted_return() {
        let mut session = LearningSession::new("Test gamma");

        session.add_turn(SessionTurn::new("a1", "o1", Reward::Positive(1.0)));
        session.add_turn(SessionTurn::new("a2", "o2", Reward::Positive(1.0)));
        session.add_turn(SessionTurn::new("a3", "o3", Reward::Positive(1.0)));

        // With gamma=0.9:
        // G = r1 + 0.9*r2 + 0.81*r3
        // G = 1.0 + 0.9 + 0.81 = 2.71
        let g = session.discounted_return(0.9);
        assert!((g - 2.71).abs() < 0.01);

        // With gamma=0.5 (more discounting):
        // G = 1.0 + 0.5 + 0.25 = 1.75
        let g = session.discounted_return(0.5);
        assert!((g - 1.75).abs() < 0.01);
    }

    /// Test: Trajectory extraction for RL.
    ///
    /// What happens:
    /// 1. Session is converted to trajectory
    /// 2. Trajectory is list of (action, observation, reward) tuples
    /// 3. Can be used for offline RL training
    #[test]
    fn test_trajectory() {
        let mut session = LearningSession::new("Demo");

        session.add_turn(SessionTurn::new("step1", "result1", Reward::Positive(0.5)));
        session.add_turn(SessionTurn::new("step2", "result2", Reward::Negative(0.1)));

        let trajectory = session.to_trajectory();

        assert_eq!(trajectory.len(), 2);
        assert_eq!(trajectory[0].0, "step1");
        assert!((trajectory[0].2 - 0.5).abs() < 0.01);
        assert!((trajectory[1].2 - (-0.1)).abs() < 0.01);
    }
}
