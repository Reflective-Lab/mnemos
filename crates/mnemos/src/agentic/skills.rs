//! Skill Library - Consolidated Successful Patterns
//!
//! Implements a skill library where agents can:
//! 1. Consolidate successful action patterns into reusable skills
//! 2. Retrieve relevant skills when facing similar tasks
//! 3. Build up a repertoire of proven approaches
//!
//! Based on the "Voyager" paper's skill library concept.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A learned skill that can be reused.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Unique identifier.
    pub id: Uuid,

    /// Skill category/domain.
    pub category: String,

    /// Human-readable name.
    pub name: String,

    /// Description of what this skill does.
    pub description: String,

    /// Component patterns that make up this skill.
    pub patterns: Vec<SkillPattern>,

    /// Preconditions for using this skill.
    pub preconditions: Vec<String>,

    /// What this skill produces/achieves.
    pub postconditions: Vec<String>,

    /// Success rate when applied (0.0 to 1.0).
    pub success_rate: f32,

    /// How many times this skill has been used.
    pub usage_count: u64,

    /// When this skill was created.
    pub created_at: DateTime<Utc>,

    /// When this skill was last used.
    pub last_used: DateTime<Utc>,

    /// Tags for categorization.
    pub tags: Vec<String>,
}

impl Skill {
    /// Create a new skill.
    pub fn new(
        category: impl Into<String>,
        name: impl Into<String>,
        patterns: Vec<SkillPattern>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            category: category.into(),
            name: name.into(),
            description: String::new(),
            patterns,
            preconditions: Vec::new(),
            postconditions: Vec::new(),
            success_rate: 1.0,
            usage_count: 0,
            created_at: now,
            last_used: now,
            tags: Vec::new(),
        }
    }

    /// Add a description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Add preconditions.
    pub fn with_preconditions(mut self, preconditions: Vec<String>) -> Self {
        self.preconditions = preconditions;
        self
    }

    /// Add postconditions.
    pub fn with_postconditions(mut self, postconditions: Vec<String>) -> Self {
        self.postconditions = postconditions;
        self
    }

    /// Set success rate.
    pub fn with_success_rate(mut self, rate: f32) -> Self {
        self.success_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set usage count.
    pub fn with_usage_count(mut self, count: u64) -> Self {
        self.usage_count = count;
        self
    }

    /// Add tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Record a usage of this skill.
    pub fn record_usage(&mut self, succeeded: bool) {
        self.usage_count += 1;
        self.last_used = Utc::now();

        // Update success rate with exponential moving average
        let alpha = 0.1;
        let outcome = if succeeded { 1.0 } else { 0.0 };
        self.success_rate = alpha * outcome + (1.0 - alpha) * self.success_rate;
    }

    /// Get the executable code/template.
    pub fn to_code(&self) -> String {
        self.patterns
            .iter()
            .map(|p| format!("// {}\n{}", p.name, p.template))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

/// A component pattern within a skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillPattern {
    /// Pattern name.
    pub name: String,

    /// Code template or instruction.
    pub template: String,

    /// Parameters that can be filled in.
    pub parameters: Vec<PatternParameter>,

    /// Example usage.
    pub example: Option<String>,
}

impl SkillPattern {
    /// Create a new pattern.
    pub fn new(name: impl Into<String>, template: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            template: template.into(),
            parameters: Vec::new(),
            example: None,
        }
    }

    /// Add parameters.
    pub fn with_parameters(mut self, params: Vec<PatternParameter>) -> Self {
        self.parameters = params;
        self
    }

    /// Add example.
    pub fn with_example(mut self, example: impl Into<String>) -> Self {
        self.example = Some(example.into());
        self
    }
}

/// A parameter in a skill pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternParameter {
    /// Parameter name.
    pub name: String,

    /// Parameter type.
    pub param_type: String,

    /// Description.
    pub description: String,

    /// Default value if any.
    pub default: Option<String>,
}

/// Library of learned skills.
pub struct SkillLibrary {
    skills: Vec<Skill>,
}

impl SkillLibrary {
    /// Create a new skill library.
    pub fn new() -> Self {
        Self { skills: Vec::new() }
    }

    /// Add a skill to the library.
    pub fn add_skill(&mut self, skill: Skill) {
        // Check if similar skill exists (by name and category)
        if let Some(existing) = self
            .skills
            .iter_mut()
            .find(|s| s.name == skill.name && s.category == skill.category)
        {
            // Merge: keep the one with higher success rate
            if skill.success_rate > existing.success_rate {
                *existing = skill;
            }
        } else {
            self.skills.push(skill);
        }
    }

    /// Find skills by category.
    pub fn find_by_category(&self, category: &str) -> Vec<&Skill> {
        self.skills
            .iter()
            .filter(|s| s.category == category)
            .collect()
    }

    /// Find skills by tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<&Skill> {
        self.skills
            .iter()
            .filter(|s| s.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Find skills matching a description (keyword search).
    pub fn search(&self, query: &str, limit: usize) -> Vec<&Skill> {
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(f32, &Skill)> = self
            .skills
            .iter()
            .map(|s| {
                let name_lower = s.name.to_lowercase();
                let desc_lower = s.description.to_lowercase();
                let cat_lower = s.category.to_lowercase();

                let score: f32 = keywords
                    .iter()
                    .map(|k| {
                        let mut kw_score = 0.0;
                        if name_lower.contains(k) {
                            kw_score += 2.0;
                        }
                        if desc_lower.contains(k) {
                            kw_score += 1.0;
                        }
                        if cat_lower.contains(k) {
                            kw_score += 1.5;
                        }
                        kw_score
                    })
                    .sum();

                // Boost by success rate and usage
                let adjusted = score
                    * (0.5 + s.success_rate)
                    * (1.0 + (s.usage_count as f32).ln().max(0.0) * 0.1);

                (adjusted, s)
            })
            .filter(|(score, _)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().take(limit).map(|(_, s)| s).collect()
    }

    /// Get most used skills.
    pub fn most_used(&self, limit: usize) -> Vec<&Skill> {
        let mut skills: Vec<_> = self.skills.iter().collect();
        skills.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
        skills.into_iter().take(limit).collect()
    }

    /// Get highest success rate skills.
    pub fn most_reliable(&self, limit: usize) -> Vec<&Skill> {
        let mut skills: Vec<_> = self.skills.iter().filter(|s| s.usage_count >= 3).collect();
        skills.sort_by(|a, b| {
            b.success_rate
                .partial_cmp(&a.success_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        skills.into_iter().take(limit).collect()
    }

    /// Total skill count.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

impl Default for SkillLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: Creating and using a skill.
    ///
    /// What happens:
    /// 1. Create a skill with code patterns
    /// 2. Add preconditions (when to use)
    /// 3. Add postconditions (what it achieves)
    /// 4. Track usage statistics
    #[test]
    fn test_skill_creation() {
        let skill = Skill::new(
            "error_handling",
            "Rust Result Pattern",
            vec![
                SkillPattern::new(
                    "define_error",
                    "#[derive(Debug, thiserror::Error)]\nenum AppError { ... }",
                ),
                SkillPattern::new(
                    "use_result",
                    "fn process() -> Result<Output, AppError> { ... }",
                ),
                SkillPattern::new("propagate", "let value = operation()?;"),
            ],
        )
        .with_description("Standard Rust error handling with custom error types")
        .with_preconditions(vec![
            "Function can fail".to_string(),
            "Need to propagate errors".to_string(),
        ])
        .with_postconditions(vec![
            "Errors are properly typed".to_string(),
            "Caller can handle or propagate".to_string(),
        ])
        .with_tags(vec!["rust".to_string(), "errors".to_string()]);

        assert_eq!(skill.patterns.len(), 3);
        assert_eq!(skill.preconditions.len(), 2);
        assert_eq!(skill.postconditions.len(), 2);

        // Get executable code
        let code = skill.to_code();
        assert!(code.contains("define_error"));
        assert!(code.contains("Result<Output, AppError>"));
    }

    /// Test: Skill library search.
    ///
    /// What happens:
    /// 1. Add multiple skills to library
    /// 2. Search by keyword
    /// 3. Results ranked by relevance and success rate
    #[test]
    fn test_skill_search() {
        let mut library = SkillLibrary::new();

        library.add_skill(
            Skill::new("testing", "Unit Test Pattern", vec![])
                .with_description("Writing unit tests with assertions")
                .with_success_rate(0.9)
                .with_usage_count(50),
        );

        library.add_skill(
            Skill::new("testing", "Integration Test Pattern", vec![])
                .with_description("Testing API endpoints end-to-end")
                .with_success_rate(0.85)
                .with_usage_count(20),
        );

        library.add_skill(
            Skill::new("deployment", "Docker Build Pattern", vec![])
                .with_description("Building Docker containers")
                .with_success_rate(0.95)
                .with_usage_count(30),
        );

        // Search for testing skills
        let results = library.search("test assertions", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].name, "Unit Test Pattern");

        // Search by category
        let testing = library.find_by_category("testing");
        assert_eq!(testing.len(), 2);
    }

    /// Test: Usage tracking and success rate.
    ///
    /// What happens:
    /// 1. Use a skill multiple times
    /// 2. Track success/failure outcomes
    /// 3. Success rate adjusts with exponential moving average
    #[test]
    fn test_usage_tracking() {
        let mut skill = Skill::new("math", "Division Pattern", vec![]).with_success_rate(1.0);

        // Use successfully 3 times
        skill.record_usage(true);
        skill.record_usage(true);
        skill.record_usage(true);
        assert!(skill.success_rate > 0.95);

        // One failure
        skill.record_usage(false);
        assert!(skill.success_rate < 1.0);
        assert!(skill.success_rate > 0.8);

        assert_eq!(skill.usage_count, 4);
    }
}
