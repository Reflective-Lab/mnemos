//! Knowledge routing and classification system.
//!
//! This module implements the Case vs Background knowledge classification system
//! as described in the design document. It provides:
//!
//! - Classification of incoming knowledge as Case (foreground) or Background (indirect)
//! - Rule-based routing using source paths, categories, tags, and content patterns
//! - Relevance scoring using the unified ranking formula
//!
//! # Example
//!
//! ```rust
//! use mnemos::ingest::{
//!     KnowledgeRouter, RoutingRule, RoutingCondition, KnowledgeTypeHint,
//!     AccessPattern, Permanence,
//! };
//! use std::collections::HashMap;
//! use std::path::Path;
//!
//! let mut router = KnowledgeRouter::new();
//!
//! // Route project files as case knowledge
//! router.add_rule(RoutingRule {
//!     condition: RoutingCondition::SourcePath("projects/**/*".to_string()),
//!     knowledge_type: KnowledgeTypeHint::Case {
//!         context: "active-project".to_string(),
//!         access_pattern: AccessPattern::ActiveUse,
//!     },
//! });
//!
//! // Route reference docs as background knowledge
//! router.add_rule(RoutingRule {
//!     condition: RoutingCondition::Category("reference".to_string()),
//!     knowledge_type: KnowledgeTypeHint::Background {
//!         domain: "documentation".to_string(),
//!         permanence: Permanence::Versioned,
//!     },
//! });
//!
//! let metadata = HashMap::new();
//! let knowledge = router.classify(
//!     Path::new("projects/my-app/README.md"),
//!     "Project documentation...",
//!     &metadata,
//! );
//! ```

use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;

/// Knowledge directly relevant to the current task or case.
///
/// Case knowledge has high relevance, is recently accessed, and is explicitly
/// linked to the current working context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseKnowledge {
    /// Unique identifier for this knowledge entry.
    pub entry_id: Uuid,

    /// Project or case identifier this knowledge belongs to.
    pub case_context: String,

    /// Time-based decay factor (0.0 to 1.0).
    /// Higher values indicate more recent/relevant knowledge.
    pub relevance_decay: f32,

    /// Manually linked entry IDs.
    pub explicit_links: Vec<Uuid>,

    /// Current access pattern for this knowledge.
    pub access_pattern: AccessPattern,

    /// When this knowledge was last accessed.
    pub last_accessed: DateTime<Utc>,
}

impl CaseKnowledge {
    /// Create new case knowledge with default values.
    pub fn new(entry_id: Uuid, case_context: impl Into<String>) -> Self {
        Self {
            entry_id,
            case_context: case_context.into(),
            relevance_decay: 1.0,
            explicit_links: Vec::new(),
            access_pattern: AccessPattern::ActiveUse,
            last_accessed: Utc::now(),
        }
    }

    /// Set the access pattern.
    pub fn with_access_pattern(mut self, pattern: AccessPattern) -> Self {
        self.access_pattern = pattern;
        self
    }

    /// Add an explicit link to another entry.
    pub fn with_link(mut self, linked_id: Uuid) -> Self {
        self.explicit_links.push(linked_id);
        self
    }

    /// Update the relevance decay based on time since last access.
    ///
    /// Uses exponential decay: `decay = exp(-days / half_life)`
    pub fn update_decay(&mut self, half_life_days: f32) {
        let days_since = (Utc::now() - self.last_accessed).num_hours() as f32 / 24.0;
        self.relevance_decay = (-days_since / half_life_days).exp();
    }

    /// Record an access, resetting the decay factor.
    pub fn record_access(&mut self) {
        self.last_accessed = Utc::now();
        self.relevance_decay = 1.0;
    }
}

/// Access pattern indicating how the knowledge is currently being used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AccessPattern {
    /// Currently being actively referenced.
    ActiveUse,

    /// Used in the last session but not currently active.
    #[default]
    RecentHistory,

    /// From a completed case, kept for reference.
    Archived,
}

impl AccessPattern {
    /// Get the boost factor for this access pattern.
    pub fn boost_factor(&self) -> f32 {
        match self {
            AccessPattern::ActiveUse => 2.0,
            AccessPattern::RecentHistory => 1.5,
            AccessPattern::Archived => 0.5,
        }
    }
}

/// Contextual knowledge that supports understanding but isn't directly actionable.
///
/// Background knowledge provides context, general reference material,
/// and supports case knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundKnowledge {
    /// Unique identifier for this knowledge entry.
    pub entry_id: Uuid,

    /// Knowledge domain (e.g., "rust", "architecture", "devops").
    pub domain: String,

    /// How stable/permanent this knowledge is.
    pub permanence: Permanence,

    /// IDs of case knowledge entries this supports.
    pub supports: Vec<Uuid>,

    /// When this knowledge was last verified as current.
    pub last_verified: DateTime<Utc>,

    /// Version information if applicable.
    pub version: Option<String>,
}

impl BackgroundKnowledge {
    /// Create new background knowledge with default values.
    pub fn new(entry_id: Uuid, domain: impl Into<String>) -> Self {
        Self {
            entry_id,
            domain: domain.into(),
            permanence: Permanence::Evergreen,
            supports: Vec::new(),
            last_verified: Utc::now(),
            version: None,
        }
    }

    /// Set the permanence level.
    pub fn with_permanence(mut self, permanence: Permanence) -> Self {
        self.permanence = permanence;
        self
    }

    /// Set the version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Add a supported case knowledge entry.
    pub fn with_support(mut self, case_id: Uuid) -> Self {
        self.supports.push(case_id);
        self
    }

    /// Check if this knowledge is still valid based on permanence.
    pub fn is_valid(&self) -> bool {
        !matches!(self.permanence, Permanence::Deprecated)
    }
}

/// How stable or permanent the knowledge is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Permanence {
    /// Always valid knowledge (math, physics, fundamental concepts).
    #[default]
    Evergreen,

    /// Valid for a specific version of software/specification.
    Versioned,

    /// Valid for a specific time period.
    Temporal,

    /// Outdated but kept for historical reference.
    Deprecated,
}

impl Permanence {
    /// Get the support factor for this permanence level.
    pub fn support_factor(&self) -> f32 {
        match self {
            Permanence::Evergreen => 1.0,
            Permanence::Versioned => 0.8,
            Permanence::Temporal => 0.5,
            Permanence::Deprecated => 0.1,
        }
    }
}

/// Classified knowledge type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KnowledgeType {
    /// Direct, actionable knowledge for current tasks.
    Case(CaseKnowledge),

    /// Contextual knowledge that supports understanding.
    Background(BackgroundKnowledge),
}

impl KnowledgeType {
    /// Get the entry ID regardless of type.
    pub fn entry_id(&self) -> Uuid {
        match self {
            KnowledgeType::Case(k) => k.entry_id,
            KnowledgeType::Background(k) => k.entry_id,
        }
    }

    /// Check if this is case knowledge.
    pub fn is_case(&self) -> bool {
        matches!(self, KnowledgeType::Case(_))
    }

    /// Check if this is background knowledge.
    pub fn is_background(&self) -> bool {
        matches!(self, KnowledgeType::Background(_))
    }

    /// Get as case knowledge if applicable.
    pub fn as_case(&self) -> Option<&CaseKnowledge> {
        match self {
            KnowledgeType::Case(k) => Some(k),
            KnowledgeType::Background(_) => None,
        }
    }

    /// Get as background knowledge if applicable.
    pub fn as_background(&self) -> Option<&BackgroundKnowledge> {
        match self {
            KnowledgeType::Case(_) => None,
            KnowledgeType::Background(k) => Some(k),
        }
    }
}

/// Hint for what type of knowledge to create during routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KnowledgeTypeHint {
    /// Create case knowledge with the given context.
    Case {
        /// The case/project context.
        context: String,
        /// Initial access pattern.
        access_pattern: AccessPattern,
    },

    /// Create background knowledge with the given domain.
    Background {
        /// The knowledge domain.
        domain: String,
        /// Permanence level.
        permanence: Permanence,
    },
}

impl Default for KnowledgeTypeHint {
    fn default() -> Self {
        Self::Background {
            domain: "general".to_string(),
            permanence: Permanence::Evergreen,
        }
    }
}

/// Condition for matching knowledge to route.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingCondition {
    /// Match source path against a glob pattern.
    SourcePath(String),

    /// Match exact category.
    Category(String),

    /// Match if entry has this tag.
    Tag(String),

    /// Match content against a regex pattern.
    ContentMatch(String),

    /// Match metadata key-value pair.
    Metadata(String, String),

    /// Combine multiple conditions with AND.
    All(Vec<RoutingCondition>),

    /// Combine multiple conditions with OR.
    Any(Vec<RoutingCondition>),
}

impl RoutingCondition {
    /// Check if this condition matches the given input.
    pub fn matches(
        &self,
        source_path: &Path,
        content: &str,
        metadata: &HashMap<String, String>,
    ) -> bool {
        match self {
            RoutingCondition::SourcePath(pattern) => {
                glob_match(pattern, &source_path.to_string_lossy())
            }
            RoutingCondition::Category(category) => {
                metadata.get("category").is_some_and(|c| c == category)
            }
            RoutingCondition::Tag(tag) => metadata
                .get("tags")
                .is_some_and(|tags| tags.split(',').any(|t| t.trim() == tag)),
            RoutingCondition::ContentMatch(pattern) => {
                Regex::new(pattern).is_ok_and(|re| re.is_match(content))
            }
            RoutingCondition::Metadata(key, value) => metadata.get(key).is_some_and(|v| v == value),
            RoutingCondition::All(conditions) => conditions
                .iter()
                .all(|c| c.matches(source_path, content, metadata)),
            RoutingCondition::Any(conditions) => conditions
                .iter()
                .any(|c| c.matches(source_path, content, metadata)),
        }
    }
}

/// Simple glob pattern matching.
///
/// Supports `*` (any characters except `/`) and `**` (any characters including `/`).
fn glob_match(pattern: &str, path: &str) -> bool {
    let regex_pattern = pattern
        .replace('.', r"\.")
        // Handle **/ to match zero or more path components
        .replace("**/", "\x00")
        // Handle ** at the end to match remaining path
        .replace("**", "\x01")
        .replace('*', "[^/]*")
        // **/ can match empty or any path ending with /
        .replace('\x00', "([^/]+/)*")
        // ** matches any characters
        .replace('\x01', ".*");

    Regex::new(&format!("^{regex_pattern}$")).is_ok_and(|re| re.is_match(path))
}

/// A routing rule that maps conditions to knowledge types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    /// Condition that must match for this rule to apply.
    pub condition: RoutingCondition,

    /// The knowledge type to assign when this rule matches.
    pub knowledge_type: KnowledgeTypeHint,
}

impl RoutingRule {
    /// Create a new routing rule.
    pub fn new(condition: RoutingCondition, knowledge_type: KnowledgeTypeHint) -> Self {
        Self {
            condition,
            knowledge_type,
        }
    }
}

/// Router for classifying incoming knowledge.
///
/// The router applies rules in order, using the first matching rule.
/// If no rules match, the default type is used.
#[derive(Debug, Clone)]
pub struct KnowledgeRouter {
    /// Ordered list of routing rules.
    rules: Vec<RoutingRule>,

    /// Default knowledge type when no rules match.
    default_type: KnowledgeTypeHint,

    /// Scoring weights for relevance calculation.
    scoring_weights: ScoringWeights,
}

/// Weights used in the relevance scoring formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// Weight for base similarity score.
    pub source_weight: f32,

    /// Boost factor for case knowledge matching context.
    pub context_boost: f32,

    /// Factor for background knowledge support.
    pub support_factor: f32,

    /// Half-life in days for recency decay.
    pub decay_half_life_days: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            source_weight: 1.0,
            context_boost: 2.0,
            support_factor: 0.5,
            decay_half_life_days: 7.0,
        }
    }
}

impl KnowledgeRouter {
    /// Create a new router with default settings.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            default_type: KnowledgeTypeHint::default(),
            scoring_weights: ScoringWeights::default(),
        }
    }

    /// Create a router with a custom default type.
    pub fn with_default(default_type: KnowledgeTypeHint) -> Self {
        Self {
            rules: Vec::new(),
            default_type,
            scoring_weights: ScoringWeights::default(),
        }
    }

    /// Set custom scoring weights.
    pub fn with_scoring_weights(mut self, weights: ScoringWeights) -> Self {
        self.scoring_weights = weights;
        self
    }

    /// Add a routing rule.
    ///
    /// Rules are evaluated in the order they are added.
    pub fn add_rule(&mut self, rule: RoutingRule) {
        self.rules.push(rule);
    }

    /// Add multiple routing rules at once.
    pub fn add_rules(&mut self, rules: impl IntoIterator<Item = RoutingRule>) {
        self.rules.extend(rules);
    }

    /// Classify knowledge based on source path, content, and metadata.
    ///
    /// Evaluates rules in order and returns the first matching type.
    /// If no rules match, returns the default type.
    pub fn classify(
        &self,
        source_path: &Path,
        content: &str,
        metadata: &HashMap<String, String>,
    ) -> KnowledgeType {
        let entry_id = Uuid::new_v4();

        let hint = self
            .rules
            .iter()
            .find(|rule| rule.condition.matches(source_path, content, metadata))
            .map_or(&self.default_type, |rule| &rule.knowledge_type);

        self.create_knowledge(entry_id, hint)
    }

    /// Classify with a specific entry ID.
    pub fn classify_with_id(
        &self,
        entry_id: Uuid,
        source_path: &Path,
        content: &str,
        metadata: &HashMap<String, String>,
    ) -> KnowledgeType {
        let hint = self
            .rules
            .iter()
            .find(|rule| rule.condition.matches(source_path, content, metadata))
            .map_or(&self.default_type, |rule| &rule.knowledge_type);

        self.create_knowledge(entry_id, hint)
    }

    /// Create knowledge from a hint.
    fn create_knowledge(&self, entry_id: Uuid, hint: &KnowledgeTypeHint) -> KnowledgeType {
        match hint {
            KnowledgeTypeHint::Case {
                context,
                access_pattern,
            } => KnowledgeType::Case(
                CaseKnowledge::new(entry_id, context).with_access_pattern(*access_pattern),
            ),
            KnowledgeTypeHint::Background { domain, permanence } => KnowledgeType::Background(
                BackgroundKnowledge::new(entry_id, domain).with_permanence(*permanence),
            ),
        }
    }

    /// Compute the relevance score for a knowledge entry.
    ///
    /// Uses the unified ranking formula:
    /// ```text
    /// final_score = (
    ///     base_similarity * source_weight +
    ///     case_relevance * context_boost +
    ///     background_relevance * support_factor +
    ///     recency_score * decay_factor
    /// )
    /// ```
    ///
    /// # Arguments
    ///
    /// * `knowledge` - The classified knowledge to score
    /// * `base_similarity` - Base similarity score from vector search (0.0 to 1.0)
    /// * `query_context` - Optional current case context for boosting
    ///
    /// # Returns
    ///
    /// The final relevance score.
    pub fn compute_relevance_score(
        &self,
        knowledge: &KnowledgeType,
        base_similarity: f32,
        query_context: Option<&str>,
    ) -> f32 {
        let weights = &self.scoring_weights;

        match knowledge {
            KnowledgeType::Case(case) => {
                // Case relevance is boosted if context matches
                let context_matches = query_context.is_some_and(|ctx| ctx == case.case_context);

                let context_multiplier = if context_matches {
                    weights.context_boost
                } else {
                    1.0
                };

                let access_boost = case.access_pattern.boost_factor();

                // Final score for case knowledge
                base_similarity * weights.source_weight
                    + context_multiplier * access_boost
                    + case.relevance_decay * access_boost
            }
            KnowledgeType::Background(bg) => {
                // Background knowledge gets support factor applied
                let permanence_factor = bg.permanence.support_factor();

                // Bonus if this background knowledge supports active case knowledge
                let support_bonus = if bg.supports.is_empty() {
                    0.0
                } else {
                    0.2 * bg.supports.len() as f32
                };

                // Final score for background knowledge
                base_similarity * weights.source_weight
                    + permanence_factor * weights.support_factor
                    + support_bonus
            }
        }
    }

    /// Get all rules.
    pub fn rules(&self) -> &[RoutingRule] {
        &self.rules
    }

    /// Get the default knowledge type hint.
    pub fn default_type(&self) -> &KnowledgeTypeHint {
        &self.default_type
    }

    /// Get the scoring weights.
    pub fn scoring_weights(&self) -> &ScoringWeights {
        &self.scoring_weights
    }
}

impl Default for KnowledgeRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_knowledge_creation() {
        let id = Uuid::new_v4();
        let case = CaseKnowledge::new(id, "my-project")
            .with_access_pattern(AccessPattern::ActiveUse)
            .with_link(Uuid::new_v4());

        assert_eq!(case.entry_id, id);
        assert_eq!(case.case_context, "my-project");
        assert_eq!(case.access_pattern, AccessPattern::ActiveUse);
        assert_eq!(case.explicit_links.len(), 1);
        assert!((case.relevance_decay - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_case_knowledge_decay() {
        let id = Uuid::new_v4();
        let mut case = CaseKnowledge::new(id, "test");

        // Simulate time passing by setting last_accessed to the past
        case.last_accessed = Utc::now() - chrono::Duration::days(7);
        case.update_decay(7.0);

        // After one half-life, decay should be approximately 0.37 (1/e)
        assert!(case.relevance_decay < 0.5);
        assert!(case.relevance_decay > 0.3);
    }

    #[test]
    fn test_background_knowledge_creation() {
        let id = Uuid::new_v4();
        let bg = BackgroundKnowledge::new(id, "rust")
            .with_permanence(Permanence::Versioned)
            .with_version("1.75")
            .with_support(Uuid::new_v4());

        assert_eq!(bg.entry_id, id);
        assert_eq!(bg.domain, "rust");
        assert_eq!(bg.permanence, Permanence::Versioned);
        assert_eq!(bg.version, Some("1.75".to_string()));
        assert_eq!(bg.supports.len(), 1);
        assert!(bg.is_valid());
    }

    #[test]
    fn test_background_knowledge_deprecated() {
        let id = Uuid::new_v4();
        let bg = BackgroundKnowledge::new(id, "legacy").with_permanence(Permanence::Deprecated);

        assert!(!bg.is_valid());
    }

    #[test]
    fn test_knowledge_type_accessors() {
        let case_id = Uuid::new_v4();
        let bg_id = Uuid::new_v4();

        let case = KnowledgeType::Case(CaseKnowledge::new(case_id, "test"));
        let bg = KnowledgeType::Background(BackgroundKnowledge::new(bg_id, "test"));

        assert!(case.is_case());
        assert!(!case.is_background());
        assert_eq!(case.entry_id(), case_id);
        assert!(case.as_case().is_some());
        assert!(case.as_background().is_none());

        assert!(!bg.is_case());
        assert!(bg.is_background());
        assert_eq!(bg.entry_id(), bg_id);
        assert!(bg.as_case().is_none());
        assert!(bg.as_background().is_some());
    }

    #[test]
    fn test_access_pattern_boost() {
        assert!((AccessPattern::ActiveUse.boost_factor() - 2.0).abs() < f32::EPSILON);
        assert!((AccessPattern::RecentHistory.boost_factor() - 1.5).abs() < f32::EPSILON);
        assert!((AccessPattern::Archived.boost_factor() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_permanence_support_factor() {
        assert!((Permanence::Evergreen.support_factor() - 1.0).abs() < f32::EPSILON);
        assert!((Permanence::Versioned.support_factor() - 0.8).abs() < f32::EPSILON);
        assert!((Permanence::Temporal.support_factor() - 0.5).abs() < f32::EPSILON);
        assert!((Permanence::Deprecated.support_factor() - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_glob_matching() {
        assert!(glob_match("*.rs", "main.rs"));
        assert!(!glob_match("*.rs", "main.txt"));
        assert!(glob_match("src/*.rs", "src/lib.rs"));
        assert!(!glob_match("src/*.rs", "src/foo/lib.rs"));
        assert!(glob_match("src/**/*.rs", "src/foo/bar/lib.rs"));
        assert!(glob_match("projects/**/*", "projects/my-app/README.md"));
    }

    #[test]
    fn test_routing_condition_source_path() {
        let condition = RoutingCondition::SourcePath("projects/**/*".to_string());
        let metadata = HashMap::new();

        assert!(condition.matches(Path::new("projects/my-app/src/main.rs"), "", &metadata));
        assert!(!condition.matches(Path::new("docs/README.md"), "", &metadata));
    }

    #[test]
    fn test_routing_condition_category() {
        let condition = RoutingCondition::Category("reference".to_string());
        let mut metadata = HashMap::new();

        assert!(!condition.matches(Path::new("test.md"), "", &metadata));

        metadata.insert("category".to_string(), "reference".to_string());
        assert!(condition.matches(Path::new("test.md"), "", &metadata));
    }

    #[test]
    fn test_routing_condition_tag() {
        let condition = RoutingCondition::Tag("rust".to_string());
        let mut metadata = HashMap::new();

        assert!(!condition.matches(Path::new("test.md"), "", &metadata));

        metadata.insert("tags".to_string(), "programming, rust, systems".to_string());
        assert!(condition.matches(Path::new("test.md"), "", &metadata));
    }

    #[test]
    fn test_routing_condition_content_match() {
        let condition = RoutingCondition::ContentMatch(r"TODO:|FIXME:".to_string());
        let metadata = HashMap::new();

        assert!(condition.matches(Path::new("test.rs"), "// TODO: implement this", &metadata));
        assert!(!condition.matches(Path::new("test.rs"), "// This is done", &metadata));
    }

    #[test]
    fn test_routing_condition_metadata() {
        let condition = RoutingCondition::Metadata("status".to_string(), "active".to_string());
        let mut metadata = HashMap::new();

        assert!(!condition.matches(Path::new("test.md"), "", &metadata));

        metadata.insert("status".to_string(), "active".to_string());
        assert!(condition.matches(Path::new("test.md"), "", &metadata));
    }

    #[test]
    fn test_routing_condition_all() {
        let condition = RoutingCondition::All(vec![
            RoutingCondition::SourcePath("src/**/*.rs".to_string()),
            RoutingCondition::Category("code".to_string()),
        ]);

        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), "code".to_string());

        assert!(condition.matches(Path::new("src/lib.rs"), "", &metadata));
        assert!(!condition.matches(Path::new("docs/README.md"), "", &metadata));

        let mut wrong_category = HashMap::new();
        wrong_category.insert("category".to_string(), "docs".to_string());
        assert!(!condition.matches(Path::new("src/lib.rs"), "", &wrong_category));
    }

    #[test]
    fn test_routing_condition_any() {
        let condition = RoutingCondition::Any(vec![
            RoutingCondition::Tag("important".to_string()),
            RoutingCondition::Tag("urgent".to_string()),
        ]);

        let mut metadata1 = HashMap::new();
        metadata1.insert("tags".to_string(), "important".to_string());
        assert!(condition.matches(Path::new("test.md"), "", &metadata1));

        let mut metadata2 = HashMap::new();
        metadata2.insert("tags".to_string(), "urgent".to_string());
        assert!(condition.matches(Path::new("test.md"), "", &metadata2));

        let metadata3 = HashMap::new();
        assert!(!condition.matches(Path::new("test.md"), "", &metadata3));
    }

    #[test]
    fn test_router_classify_with_rule() {
        let mut router = KnowledgeRouter::new();

        router.add_rule(RoutingRule {
            condition: RoutingCondition::SourcePath("projects/**/*".to_string()),
            knowledge_type: KnowledgeTypeHint::Case {
                context: "active-project".to_string(),
                access_pattern: AccessPattern::ActiveUse,
            },
        });

        let metadata = HashMap::new();
        let knowledge = router.classify(
            Path::new("projects/my-app/README.md"),
            "Project documentation",
            &metadata,
        );

        assert!(knowledge.is_case());
        let case = knowledge.as_case().unwrap();
        assert_eq!(case.case_context, "active-project");
        assert_eq!(case.access_pattern, AccessPattern::ActiveUse);
    }

    #[test]
    fn test_router_classify_default() {
        let router = KnowledgeRouter::with_default(KnowledgeTypeHint::Background {
            domain: "general".to_string(),
            permanence: Permanence::Evergreen,
        });

        let metadata = HashMap::new();
        let knowledge =
            router.classify(Path::new("some/random/file.txt"), "Some content", &metadata);

        assert!(knowledge.is_background());
        let bg = knowledge.as_background().unwrap();
        assert_eq!(bg.domain, "general");
        assert_eq!(bg.permanence, Permanence::Evergreen);
    }

    #[test]
    fn test_router_rule_priority() {
        let mut router = KnowledgeRouter::new();

        // More specific rule first
        router.add_rule(RoutingRule {
            condition: RoutingCondition::SourcePath("projects/urgent/**/*".to_string()),
            knowledge_type: KnowledgeTypeHint::Case {
                context: "urgent".to_string(),
                access_pattern: AccessPattern::ActiveUse,
            },
        });

        // General rule second
        router.add_rule(RoutingRule {
            condition: RoutingCondition::SourcePath("projects/**/*".to_string()),
            knowledge_type: KnowledgeTypeHint::Case {
                context: "normal".to_string(),
                access_pattern: AccessPattern::RecentHistory,
            },
        });

        let metadata = HashMap::new();

        // Should match first rule
        let urgent = router.classify(Path::new("projects/urgent/fix.rs"), "", &metadata);
        assert_eq!(urgent.as_case().unwrap().case_context, "urgent");

        // Should match second rule
        let normal = router.classify(Path::new("projects/other/main.rs"), "", &metadata);
        assert_eq!(normal.as_case().unwrap().case_context, "normal");
    }

    #[test]
    fn test_compute_relevance_score_case_matching_context() {
        let router = KnowledgeRouter::new();

        let case = KnowledgeType::Case(
            CaseKnowledge::new(Uuid::new_v4(), "my-project")
                .with_access_pattern(AccessPattern::ActiveUse),
        );

        // Score with matching context should be higher
        let score_match = router.compute_relevance_score(&case, 0.8, Some("my-project"));
        let score_no_match = router.compute_relevance_score(&case, 0.8, Some("other-project"));
        let score_no_context = router.compute_relevance_score(&case, 0.8, None);

        assert!(score_match > score_no_match);
        assert!(score_match > score_no_context);
    }

    #[test]
    fn test_compute_relevance_score_background() {
        let router = KnowledgeRouter::new();

        let bg = KnowledgeType::Background(
            BackgroundKnowledge::new(Uuid::new_v4(), "rust").with_permanence(Permanence::Evergreen),
        );

        let score = router.compute_relevance_score(&bg, 0.8, Some("any-context"));

        // Background knowledge gets base score + support factor
        assert!(score > 0.8);
    }

    #[test]
    fn test_compute_relevance_score_background_with_supports() {
        let router = KnowledgeRouter::new();

        let bg_no_supports =
            KnowledgeType::Background(BackgroundKnowledge::new(Uuid::new_v4(), "rust"));

        let bg_with_supports = KnowledgeType::Background(
            BackgroundKnowledge::new(Uuid::new_v4(), "rust")
                .with_support(Uuid::new_v4())
                .with_support(Uuid::new_v4()),
        );

        let score_no = router.compute_relevance_score(&bg_no_supports, 0.8, None);
        let score_with = router.compute_relevance_score(&bg_with_supports, 0.8, None);

        // More supports = higher score
        assert!(score_with > score_no);
    }

    #[test]
    fn test_router_with_custom_weights() {
        let weights = ScoringWeights {
            source_weight: 2.0,
            context_boost: 3.0,
            support_factor: 0.8,
            decay_half_life_days: 14.0,
        };

        let router = KnowledgeRouter::new().with_scoring_weights(weights);

        assert!((router.scoring_weights().source_weight - 2.0).abs() < f32::EPSILON);
        assert!((router.scoring_weights().context_boost - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_complex_routing_scenario() {
        let mut router = KnowledgeRouter::new();

        // Active project code
        router.add_rule(RoutingRule::new(
            RoutingCondition::All(vec![
                RoutingCondition::SourcePath("src/**/*.rs".to_string()),
                RoutingCondition::Metadata("status".to_string(), "active".to_string()),
            ]),
            KnowledgeTypeHint::Case {
                context: "current-sprint".to_string(),
                access_pattern: AccessPattern::ActiveUse,
            },
        ));

        // Reference documentation
        router.add_rule(RoutingRule::new(
            RoutingCondition::Any(vec![
                RoutingCondition::Category("reference".to_string()),
                RoutingCondition::Tag("documentation".to_string()),
            ]),
            KnowledgeTypeHint::Background {
                domain: "documentation".to_string(),
                permanence: Permanence::Versioned,
            },
        ));

        // Test active code
        let mut metadata1 = HashMap::new();
        metadata1.insert("status".to_string(), "active".to_string());

        let code = router.classify(Path::new("src/lib.rs"), "pub fn main() {}", &metadata1);
        assert!(code.is_case());
        assert_eq!(code.as_case().unwrap().case_context, "current-sprint");

        // Test reference doc
        let mut metadata2 = HashMap::new();
        metadata2.insert("category".to_string(), "reference".to_string());

        let doc = router.classify(Path::new("docs/api.md"), "# API Reference", &metadata2);
        assert!(doc.is_background());
        assert_eq!(doc.as_background().unwrap().domain, "documentation");

        // Test unmatched file - should use default
        let other = router.classify(Path::new("random.txt"), "some content", &HashMap::new());
        assert!(other.is_background()); // Default is background
    }
}
