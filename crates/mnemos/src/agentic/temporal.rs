//! Time Crystals - Periodic Behavior Patterns
//!
//! Implements temporal pattern detection for agents, inspired by "time crystals"
//! that exhibit periodic behavior. This module helps agents:
//!
//! 1. Detect recurring patterns in their behavior over time
//! 2. Learn optimal timing for actions (when to do what)
//! 3. Build temporal context for decision making
//! 4. Identify anomalies in timing patterns

use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A time crystal capturing periodic behavior patterns.
///
/// Named after the physics concept of time crystals that exhibit
/// periodic structure in time rather than space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeCrystal {
    /// Unique identifier.
    pub id: Uuid,

    /// Pattern name/description.
    pub name: String,

    /// The periodic interval of this pattern.
    pub period: TemporalPeriod,

    /// Observed occurrences within the period.
    pub occurrences: Vec<TemporalOccurrence>,

    /// Computed distribution over the period.
    pub distribution: Vec<f32>,

    /// Confidence in this pattern (0.0 to 1.0).
    pub confidence: f32,

    /// When this crystal was created.
    pub created_at: DateTime<Utc>,

    /// When this crystal was last updated.
    pub updated_at: DateTime<Utc>,
}

impl TimeCrystal {
    /// Create a new time crystal for a given period.
    pub fn new(name: impl Into<String>, period: TemporalPeriod) -> Self {
        let bins = period.bin_count();
        let now = Utc::now();

        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            period,
            occurrences: Vec::new(),
            distribution: vec![0.0; bins],
            confidence: 0.0,
            created_at: now,
            updated_at: now,
        }
    }

    /// Record an occurrence at a specific time.
    pub fn record(&mut self, timestamp: DateTime<Utc>, value: f32) {
        let bin = self.period.time_to_bin(&timestamp);
        self.occurrences.push(TemporalOccurrence {
            timestamp,
            bin,
            value,
        });
        self.updated_at = Utc::now();
        self.recompute_distribution();
    }

    /// Record an occurrence now.
    pub fn record_now(&mut self, value: f32) {
        self.record(Utc::now(), value);
    }

    /// Get the expected value at a given time.
    pub fn predict(&self, timestamp: &DateTime<Utc>) -> f32 {
        let bin = self.period.time_to_bin(timestamp);
        self.distribution.get(bin).copied().unwrap_or(0.0)
    }

    /// Get the expected value now.
    pub fn predict_now(&self) -> f32 {
        self.predict(&Utc::now())
    }

    /// Find the best time within the period for high values.
    pub fn best_time(&self) -> usize {
        self.distribution
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Check if current time is anomalous compared to pattern.
    pub fn is_anomalous(&self, timestamp: &DateTime<Utc>, value: f32, threshold: f32) -> bool {
        let expected = self.predict(timestamp);
        (value - expected).abs() > threshold
    }

    /// Recompute distribution from occurrences.
    fn recompute_distribution(&mut self) {
        let bins = self.period.bin_count();
        let mut counts = vec![0.0f32; bins];
        let mut totals = vec![0.0f32; bins];

        for occ in &self.occurrences {
            if occ.bin < bins {
                counts[occ.bin] += 1.0;
                totals[occ.bin] += occ.value;
            }
        }

        // Compute averages
        for i in 0..bins {
            self.distribution[i] = if counts[i] > 0.0 {
                totals[i] / counts[i]
            } else {
                0.0
            };
        }

        // Update confidence based on sample size
        let total_samples: f32 = counts.iter().sum();
        self.confidence = (total_samples / (bins as f32 * 3.0)).min(1.0);
    }

    /// Get the period description.
    pub fn period_description(&self) -> String {
        match &self.period {
            TemporalPeriod::Hourly => "hourly (by minute)".to_string(),
            TemporalPeriod::Daily => "daily (by hour)".to_string(),
            TemporalPeriod::Weekly => "weekly (by day)".to_string(),
            TemporalPeriod::Monthly => "monthly (by day)".to_string(),
            TemporalPeriod::Custom { name, .. } => format!("custom: {}", name),
        }
    }
}

/// An occurrence within a time crystal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalOccurrence {
    /// When this occurred.
    pub timestamp: DateTime<Utc>,

    /// Which bin this falls into.
    pub bin: usize,

    /// The value/intensity of this occurrence.
    pub value: f32,
}

/// Temporal periods for pattern detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalPeriod {
    /// Hourly pattern (60 bins for minutes).
    Hourly,

    /// Daily pattern (24 bins for hours).
    Daily,

    /// Weekly pattern (7 bins for days).
    Weekly,

    /// Monthly pattern (31 bins for days).
    Monthly,

    /// Custom period with named bins.
    Custom {
        /// Period name.
        name: String,
        /// Number of bins.
        bins: usize,
        /// Duration of the period.
        duration_minutes: u64,
    },
}

impl TemporalPeriod {
    /// Get the number of bins for this period.
    pub fn bin_count(&self) -> usize {
        match self {
            TemporalPeriod::Hourly => 60,
            TemporalPeriod::Daily => 24,
            TemporalPeriod::Weekly => 7,
            TemporalPeriod::Monthly => 31,
            TemporalPeriod::Custom { bins, .. } => *bins,
        }
    }

    /// Convert a timestamp to a bin index.
    pub fn time_to_bin(&self, timestamp: &DateTime<Utc>) -> usize {
        match self {
            TemporalPeriod::Hourly => timestamp.minute() as usize,
            TemporalPeriod::Daily => timestamp.hour() as usize,
            TemporalPeriod::Weekly => timestamp.weekday().num_days_from_monday() as usize,
            TemporalPeriod::Monthly => (timestamp.day() - 1) as usize,
            TemporalPeriod::Custom {
                duration_minutes,
                bins,
                ..
            } => {
                let minutes_since_epoch = timestamp.timestamp() as u64 / 60;
                ((minutes_since_epoch % duration_minutes) * (*bins as u64) / duration_minutes)
                    as usize
            }
        }
    }

    /// Get a label for a bin.
    pub fn bin_label(&self, bin: usize) -> String {
        match self {
            TemporalPeriod::Hourly => format!(":{:02}", bin),
            TemporalPeriod::Daily => format!("{:02}:00", bin),
            TemporalPeriod::Weekly => {
                let day = match bin {
                    0 => "Monday",
                    1 => "Tuesday",
                    2 => "Wednesday",
                    3 => "Thursday",
                    4 => "Friday",
                    5 => "Saturday",
                    6 => "Sunday",
                    _ => "Unknown",
                };
                day.to_string()
            }
            TemporalPeriod::Monthly => format!("Day {}", bin + 1),
            TemporalPeriod::Custom { name, .. } => format!("{} bin {}", name, bin),
        }
    }
}

/// Store for time crystals.
pub struct TemporalMemory {
    crystals: HashMap<String, TimeCrystal>,
}

impl TemporalMemory {
    /// Create a new temporal memory.
    pub fn new() -> Self {
        Self {
            crystals: HashMap::new(),
        }
    }

    /// Get or create a time crystal.
    pub fn get_or_create(&mut self, name: &str, period: TemporalPeriod) -> &mut TimeCrystal {
        self.crystals
            .entry(name.to_string())
            .or_insert_with(|| TimeCrystal::new(name, period))
    }

    /// Record an event for a pattern.
    pub fn record(&mut self, pattern_name: &str, period: TemporalPeriod, value: f32) {
        let crystal = self.get_or_create(pattern_name, period);
        crystal.record_now(value);
    }

    /// Get prediction for a pattern.
    pub fn predict(&self, pattern_name: &str) -> Option<f32> {
        self.crystals.get(pattern_name).map(|c| c.predict_now())
    }

    /// List all patterns.
    pub fn list_patterns(&self) -> Vec<&str> {
        self.crystals.keys().map(|s| s.as_str()).collect()
    }

    /// Get a crystal by name.
    pub fn get(&self, name: &str) -> Option<&TimeCrystal> {
        self.crystals.get(name)
    }

    /// Total crystal count.
    pub fn len(&self) -> usize {
        self.crystals.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.crystals.is_empty()
    }
}

impl Default for TemporalMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    /// Test: Daily usage pattern tracking.
    ///
    /// What happens:
    /// 1. Create a time crystal for daily patterns
    /// 2. Record high activity during work hours (9-17)
    /// 3. Record low activity at night
    /// 4. Crystal learns the pattern and can predict
    #[test]
    fn test_daily_pattern() {
        let mut crystal = TimeCrystal::new("coding_activity", TemporalPeriod::Daily);

        // Simulate activity pattern: high during work hours
        for hour in 0..24 {
            let value = if hour >= 9 && hour < 17 {
                1.0 // High activity during work hours
            } else if hour >= 22 || hour < 6 {
                0.0 // No activity at night
            } else {
                0.3 // Low activity morning/evening
            };

            // Create timestamp for this hour
            let timestamp = Utc::now().with_hour(hour).unwrap().with_minute(0).unwrap();
            crystal.record(timestamp, value);
        }

        // Verify pattern learned
        let work_hour = crystal.predict(&Utc::now().with_hour(10).unwrap().with_minute(0).unwrap());
        let night_hour = crystal.predict(&Utc::now().with_hour(2).unwrap().with_minute(0).unwrap());

        assert!(work_hour > night_hour);
        assert!(work_hour >= 0.9); // Should be close to 1.0

        // Best time should be a work hour
        let best = crystal.best_time();
        assert!(best >= 9 && best < 17);
    }

    /// Test: Weekly pattern for recurring tasks.
    ///
    /// What happens:
    /// 1. Create a weekly pattern for "deploy" activity
    /// 2. Record deploys on Tuesday and Thursday
    /// 3. Crystal predicts high likelihood on those days
    #[test]
    fn test_weekly_pattern() {
        let mut crystal = TimeCrystal::new("deploy_activity", TemporalPeriod::Weekly);

        // Simulate: deploys happen on Tuesday (1) and Thursday (3)
        for day in 0..7 {
            let value = if day == 1 || day == 3 { 1.0 } else { 0.0 };
            // Create timestamp for each day
            let days_offset = day as i64 - Utc::now().weekday().num_days_from_monday() as i64;
            let timestamp = Utc::now() + Duration::days(days_offset);
            crystal.record(timestamp, value);
        }

        // Verify Tuesday and Thursday have high values
        assert!(crystal.distribution[1] > 0.5); // Tuesday
        assert!(crystal.distribution[3] > 0.5); // Thursday
        assert!(crystal.distribution[0] < 0.1); // Monday
        assert!(crystal.distribution[5] < 0.1); // Saturday
    }

    /// Test: Anomaly detection in patterns.
    ///
    /// What happens:
    /// 1. Establish a pattern (low activity at night)
    /// 2. Check if unusual activity is flagged
    /// 3. Suggestor can alert on unexpected behavior
    #[test]
    fn test_anomaly_detection() {
        let mut crystal = TimeCrystal::new("system_load", TemporalPeriod::Daily);

        // Normal pattern: low at night, high during day
        for _ in 0..10 {
            for hour in 0..24 {
                let normal_value = if hour >= 9 && hour < 17 { 0.8 } else { 0.1 };
                let timestamp = Utc::now().with_hour(hour).unwrap().with_minute(0).unwrap();
                crystal.record(timestamp, normal_value);
            }
        }

        // Test anomaly: high load at 3am
        let night_time = Utc::now().with_hour(3).unwrap();
        assert!(crystal.is_anomalous(&night_time, 0.9, 0.3));

        // Normal: high load at 10am
        let day_time = Utc::now().with_hour(10).unwrap();
        assert!(!crystal.is_anomalous(&day_time, 0.8, 0.3));
    }

    /// Test: Temporal memory for multiple patterns.
    ///
    /// What happens:
    /// 1. Track multiple independent patterns
    /// 2. Each pattern has its own period
    /// 3. Query predictions for each
    #[test]
    fn test_temporal_memory() {
        let mut memory = TemporalMemory::new();

        // Track different patterns
        memory.record("coding", TemporalPeriod::Daily, 1.0);
        memory.record("meetings", TemporalPeriod::Daily, 0.5);
        memory.record("deploys", TemporalPeriod::Weekly, 1.0);

        assert_eq!(memory.len(), 3);
        assert!(memory.predict("coding").is_some());
        assert!(memory.predict("unknown").is_none());

        let patterns = memory.list_patterns();
        assert!(patterns.contains(&"coding"));
        assert!(patterns.contains(&"meetings"));
        assert!(patterns.contains(&"deploys"));
    }
}
