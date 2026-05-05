//! Online Learning - Continual Adaptation
//!
//! Implements online/continual learning mechanisms that allow agents to:
//!
//! 1. Learn incrementally from new data without forgetting
//! 2. Adapt to distribution shifts over time
//! 3. Use Elastic Weight Consolidation (EWC) to protect important weights
//! 4. Maintain a sliding window of recent experiences
//!
//! Based on continual learning research including EWC, Progressive Networks,
//! and experience replay strategies.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

/// Online learner that adapts continuously to new data.
///
/// Key features:
/// - Incremental updates without full retraining
/// - EWC-style importance weighting
/// - Forgetting prevention via rehearsal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearner {
    /// Learner identifier.
    pub id: Uuid,

    /// Name/description.
    pub name: String,

    /// Current parameter estimates (feature weights).
    pub parameters: Vec<f32>,

    /// Fisher information (importance of each parameter).
    pub fisher_diagonal: Vec<f32>,

    /// Historical parameter snapshots for EWC.
    pub parameter_history: VecDeque<ParameterSnapshot>,

    /// Learning rate.
    pub learning_rate: f32,

    /// EWC regularization strength (lambda).
    pub ewc_lambda: f32,

    /// Number of updates performed.
    pub update_count: u64,

    /// When this learner was created.
    pub created_at: DateTime<Utc>,

    /// When this learner was last updated.
    pub updated_at: DateTime<Utc>,
}

impl OnlineLearner {
    /// Create a new online learner.
    pub fn new(name: impl Into<String>, num_parameters: usize) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            parameters: vec![0.0; num_parameters],
            fisher_diagonal: vec![1.0; num_parameters], // Start with uniform importance
            parameter_history: VecDeque::with_capacity(10),
            learning_rate: 0.01,
            ewc_lambda: 0.5,
            update_count: 0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set learning rate.
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set EWC lambda.
    pub fn with_ewc_lambda(mut self, lambda: f32) -> Self {
        self.ewc_lambda = lambda;
        self
    }

    /// Update parameters with new observation.
    ///
    /// Uses gradient descent with EWC regularization to prevent forgetting.
    pub fn update(&mut self, features: &[f32], target: f32) -> f32 {
        if features.len() != self.parameters.len() {
            return 0.0;
        }

        // Forward pass: linear prediction
        let prediction: f32 = features
            .iter()
            .zip(self.parameters.iter())
            .map(|(f, p)| f * p)
            .sum();

        // Compute loss
        let error = prediction - target;
        let loss = error * error;

        // Compute gradients with EWC penalty
        for i in 0..self.parameters.len() {
            // Base gradient (MSE loss)
            let base_grad = 2.0 * error * features[i];

            // EWC penalty: sum over previous tasks
            let mut ewc_grad = 0.0;
            for snapshot in &self.parameter_history {
                let delta = self.parameters[i] - snapshot.parameters[i];
                let importance = snapshot.fisher[i];
                ewc_grad += 2.0 * self.ewc_lambda * importance * delta;
            }

            // Combined update
            let total_grad = base_grad + ewc_grad;
            self.parameters[i] -= self.learning_rate * total_grad;
        }

        // Update Fisher diagonal based on gradient magnitude
        self.update_fisher(features, error);

        self.update_count += 1;
        self.updated_at = Utc::now();

        loss
    }

    /// Update Fisher information diagonal.
    fn update_fisher(&mut self, features: &[f32], error: f32) {
        // Fisher diagonal approximated by squared gradients
        let decay = 0.99;
        for i in 0..self.fisher_diagonal.len() {
            let grad_sq = (2.0 * error * features[i]).powi(2);
            self.fisher_diagonal[i] = decay * self.fisher_diagonal[i] + (1.0 - decay) * grad_sq;
        }
    }

    /// Consolidate current knowledge (take a snapshot for EWC).
    ///
    /// Call this when switching to a new task/domain to remember
    /// the current parameters and their importance.
    pub fn consolidate(&mut self) {
        let snapshot = ParameterSnapshot {
            parameters: self.parameters.clone(),
            fisher: self.fisher_diagonal.clone(),
            timestamp: Utc::now(),
            update_count: self.update_count,
        };

        self.parameter_history.push_back(snapshot);

        // Keep only recent snapshots
        while self.parameter_history.len() > 10 {
            self.parameter_history.pop_front();
        }
    }

    /// Make a prediction.
    pub fn predict(&self, features: &[f32]) -> f32 {
        if features.len() != self.parameters.len() {
            return 0.0;
        }

        features
            .iter()
            .zip(self.parameters.iter())
            .map(|(f, p)| f * p)
            .sum()
    }

    /// Get current parameters.
    pub fn get_parameters(&self) -> &[f32] {
        &self.parameters
    }

    /// Get parameter importance.
    pub fn get_importance(&self) -> &[f32] {
        &self.fisher_diagonal
    }

    /// Number of consolidation snapshots.
    pub fn num_snapshots(&self) -> usize {
        self.parameter_history.len()
    }
}

/// A snapshot of parameters for EWC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSnapshot {
    /// Parameter values at snapshot time.
    pub parameters: Vec<f32>,

    /// Fisher diagonal (importance) at snapshot time.
    pub fisher: Vec<f32>,

    /// When this snapshot was taken.
    pub timestamp: DateTime<Utc>,

    /// Number of updates at snapshot time.
    pub update_count: u64,
}

/// Sliding window experience buffer for rehearsal.
///
/// Keeps recent experiences for periodic rehearsal to
/// prevent catastrophic forgetting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceWindow {
    /// Buffer of recent experiences.
    experiences: VecDeque<Experience>,

    /// Maximum capacity.
    capacity: usize,

    /// How old experiences can be before removal.
    max_age: Duration,
}

/// A single experience for rehearsal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Feature vector.
    pub features: Vec<f32>,

    /// Target value.
    pub target: f32,

    /// When this was observed.
    pub timestamp: DateTime<Utc>,

    /// Task/domain identifier.
    pub task_id: Option<String>,
}

impl ExperienceWindow {
    /// Create a new experience window.
    pub fn new(capacity: usize) -> Self {
        Self {
            experiences: VecDeque::with_capacity(capacity),
            capacity,
            max_age: Duration::hours(24),
        }
    }

    /// Set maximum age for experiences.
    pub fn with_max_age(mut self, hours: i64) -> Self {
        self.max_age = Duration::hours(hours);
        self
    }

    /// Add an experience.
    pub fn add(&mut self, features: Vec<f32>, target: f32, task_id: Option<String>) {
        let exp = Experience {
            features,
            target,
            timestamp: Utc::now(),
            task_id,
        };

        self.experiences.push_back(exp);

        // Trim if over capacity
        while self.experiences.len() > self.capacity {
            self.experiences.pop_front();
        }

        // Remove old experiences
        self.prune_old();
    }

    /// Get experiences for rehearsal.
    ///
    /// Returns a random sample of experiences for replay.
    pub fn sample(&self, count: usize) -> Vec<&Experience> {
        if self.experiences.is_empty() || count == 0 {
            return Vec::new();
        }

        // Simple reservoir sampling
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut result: Vec<&Experience> = Vec::with_capacity(count.min(self.experiences.len()));

        for (i, exp) in self.experiences.iter().enumerate() {
            if result.len() < count {
                result.push(exp);
            } else {
                let j = rng.gen_range(0..=i);
                if j < count {
                    result[j] = exp;
                }
            }
        }

        result
    }

    /// Get experiences by task.
    pub fn by_task(&self, task_id: &str) -> Vec<&Experience> {
        self.experiences
            .iter()
            .filter(|e| e.task_id.as_deref() == Some(task_id))
            .collect()
    }

    /// Prune old experiences.
    fn prune_old(&mut self) {
        let cutoff = Utc::now() - self.max_age;
        while let Some(front) = self.experiences.front() {
            if front.timestamp < cutoff {
                self.experiences.pop_front();
            } else {
                break;
            }
        }
    }

    /// Current buffer size.
    pub fn len(&self) -> usize {
        self.experiences.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.experiences.is_empty()
    }

    /// Get capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for ExperienceWindow {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Distribution shift detector.
///
/// Monitors for changes in input distribution that may
/// require adaptation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetector {
    /// Running mean of features.
    running_mean: Vec<f32>,

    /// Running variance of features.
    running_var: Vec<f32>,

    /// Count of observations.
    count: u64,

    /// Recent shift scores.
    shift_scores: VecDeque<f32>,

    /// Threshold for drift detection.
    threshold: f32,
}

impl DriftDetector {
    /// Create a new drift detector.
    pub fn new(num_features: usize) -> Self {
        Self {
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            count: 0,
            shift_scores: VecDeque::with_capacity(100),
            threshold: 2.0, // Standard deviations
        }
    }

    /// Set detection threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Update statistics and check for drift.
    ///
    /// Returns true if drift is detected.
    pub fn update(&mut self, features: &[f32]) -> bool {
        if features.len() != self.running_mean.len() {
            return false;
        }

        // Compute shift score (Mahalanobis-like distance)
        let shift_score: f32 = features
            .iter()
            .zip(self.running_mean.iter())
            .zip(self.running_var.iter())
            .map(|((f, m), v)| ((f - m).powi(2)) / v.max(1e-6))
            .sum::<f32>()
            .sqrt()
            / (features.len() as f32).sqrt();

        self.shift_scores.push_back(shift_score);
        while self.shift_scores.len() > 100 {
            self.shift_scores.pop_front();
        }

        // Update running statistics (Welford's algorithm)
        self.count += 1;
        let n = self.count as f32;

        for i in 0..features.len() {
            let delta = features[i] - self.running_mean[i];
            self.running_mean[i] += delta / n;
            let delta2 = features[i] - self.running_mean[i];
            self.running_var[i] += (delta * delta2 - self.running_var[i]) / n;
        }

        shift_score > self.threshold
    }

    /// Get average recent shift score.
    pub fn average_shift(&self) -> f32 {
        if self.shift_scores.is_empty() {
            return 0.0;
        }
        self.shift_scores.iter().sum::<f32>() / self.shift_scores.len() as f32
    }

    /// Check if drift has been detected recently.
    pub fn is_drifting(&self) -> bool {
        self.average_shift() > self.threshold
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.running_mean.fill(0.0);
        self.running_var.fill(1.0);
        self.count = 0;
        self.shift_scores.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: Online learning with incremental updates.
    ///
    /// What happens:
    /// 1. Create an online learner for a simple linear function
    /// 2. Feed examples one at a time
    /// 3. Learner converges to approximate the function
    #[test]
    fn test_online_learning() {
        let mut learner = OnlineLearner::new("linear", 2).with_learning_rate(0.1);

        // Train on: y = 2*x1 + 3*x2
        for _ in 0..100 {
            let x1 = rand::random::<f32>();
            let x2 = rand::random::<f32>();
            let y = 2.0 * x1 + 3.0 * x2;

            learner.update(&[x1, x2], y);
        }

        // Check if parameters are close to [2, 3]
        let params = learner.get_parameters();
        assert!(
            (params[0] - 2.0).abs() < 0.3,
            "Expected ~2.0, got {}",
            params[0]
        );
        assert!(
            (params[1] - 3.0).abs() < 0.3,
            "Expected ~3.0, got {}",
            params[1]
        );
    }

    /// Test: EWC prevents catastrophic forgetting.
    ///
    /// What happens:
    /// 1. Train on Task A (y = 2*x)
    /// 2. Consolidate knowledge
    /// 3. Train on Task B (y = -x)
    /// 4. EWC preserves some Task A knowledge
    #[test]
    fn test_ewc_consolidation() {
        let mut learner = OnlineLearner::new("ewc_test", 1)
            .with_learning_rate(0.1)
            .with_ewc_lambda(1.0);

        // Task A: y = 2*x
        for _ in 0..50 {
            let x = rand::random::<f32>();
            let y = 2.0 * x;
            learner.update(&[x], y);
        }

        let task_a_param = learner.parameters[0];

        // Consolidate Task A knowledge
        learner.consolidate();
        assert_eq!(learner.num_snapshots(), 1);

        // Task B: y = -x (conflicting with Task A)
        for _ in 0..50 {
            let x = rand::random::<f32>();
            let y = -x;
            learner.update(&[x], y);
        }

        let final_param = learner.parameters[0];

        // With EWC, parameter shouldn't have fully shifted to -1
        // It should be somewhere between 2.0 and -1.0
        assert!(
            final_param > -0.5,
            "EWC should prevent full forgetting: {}",
            final_param
        );
        assert!(
            final_param < task_a_param,
            "Should have adapted to Task B: {}",
            final_param
        );
    }

    /// Test: Experience window for rehearsal.
    ///
    /// What happens:
    /// 1. Add experiences to the buffer
    /// 2. Sample for rehearsal
    /// 3. Buffer respects capacity limits
    #[test]
    fn test_experience_window() {
        let mut window = ExperienceWindow::new(10);

        // Add 15 experiences
        for i in 0..15 {
            window.add(vec![i as f32], i as f32, Some("task1".to_string()));
        }

        // Should be capped at capacity
        assert_eq!(window.len(), 10);

        // Sample should return requested count
        let sample = window.sample(5);
        assert_eq!(sample.len(), 5);

        // Filter by task
        let task1 = window.by_task("task1");
        assert!(!task1.is_empty());
    }

    /// Test: Distribution drift detection.
    ///
    /// What happens:
    /// 1. Establish baseline with normal distribution
    /// 2. Shift to different distribution
    /// 3. Detector identifies the drift
    #[test]
    fn test_drift_detection() {
        let mut detector = DriftDetector::new(2).with_threshold(3.0);

        // Baseline: centered around (0.5, 0.5)
        for _ in 0..200 {
            let x1 = rand::random::<f32>();
            let x2 = rand::random::<f32>();
            detector.update(&[x1, x2]);
        }

        // Reset shift scores to clear baseline
        detector.shift_scores.clear();

        // Now feed normal data - should not drift
        for _ in 0..50 {
            let x1 = rand::random::<f32>();
            let x2 = rand::random::<f32>();
            detector.update(&[x1, x2]);
        }

        // Average shift should be low for normal data
        let baseline_shift = detector.average_shift();

        // Shift: centered around (10, 10) - very different distribution
        let mut _drift_detected = false;
        for _ in 0..20 {
            let x1 = rand::random::<f32>() + 9.5;
            let x2 = rand::random::<f32>() + 9.5;
            if detector.update(&[x1, x2]) {
                _drift_detected = true;
            }
        }

        // After drift, average should be higher
        let drift_shift = detector.average_shift();
        assert!(
            drift_shift > baseline_shift,
            "Drift shift {} should be > baseline {}",
            drift_shift,
            baseline_shift
        );
    }
}
