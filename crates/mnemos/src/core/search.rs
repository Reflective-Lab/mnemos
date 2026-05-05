//! Search types and options.

use super::KnowledgeEntry;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A search result from the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// The matching entry.
    pub entry: KnowledgeEntry,

    /// Similarity score (0.0 to 1.0, higher is more similar).
    pub similarity: f32,

    /// Learned relevance boost applied.
    pub relevance_boost: f32,

    /// Final combined score.
    pub score: f32,

    /// Distance in vector space.
    pub distance: f32,
}

impl SearchResult {
    /// Create a new search result.
    pub fn new(entry: KnowledgeEntry, similarity: f32, distance: f32) -> Self {
        let relevance_boost = entry.learned_relevance;
        let score = similarity * relevance_boost;

        Self {
            entry,
            similarity,
            relevance_boost,
            score,
            distance,
        }
    }

    /// Get the entry ID.
    pub fn id(&self) -> Uuid {
        self.entry.id
    }
}

/// Options for configuring search behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
    /// Maximum number of results to return.
    pub limit: usize,

    /// Minimum similarity threshold (0.0 to 1.0).
    pub min_similarity: f32,

    /// Filter by category.
    pub category: Option<String>,

    /// Filter by tags (any match).
    pub tags: Vec<String>,

    /// Apply learned relevance boosting.
    pub use_learning: bool,

    /// Include related entries in results.
    pub include_related: bool,

    /// Diversity factor for MMR (Maximal Marginal Relevance).
    pub diversity: f32,

    /// Use hybrid search (combine vector + keyword).
    pub hybrid: bool,

    /// Keyword weight for hybrid search (0.0 to 1.0).
    pub keyword_weight: f32,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            min_similarity: 0.0,
            category: None,
            tags: Vec::new(),
            use_learning: true,
            include_related: false,
            diversity: 0.0,
            hybrid: false,
            keyword_weight: 0.3,
        }
    }
}

impl SearchOptions {
    /// Create new search options with a result limit.
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            ..Default::default()
        }
    }

    /// Set minimum similarity threshold.
    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = threshold.clamp(0.0, 1.0);
        self
    }

    /// Filter by category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Filter by tags.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    /// Disable learning-based relevance boosting.
    pub fn without_learning(mut self) -> Self {
        self.use_learning = false;
        self
    }

    /// Include related entries.
    pub fn with_related(mut self) -> Self {
        self.include_related = true;
        self
    }

    /// Set diversity factor for MMR.
    pub fn with_diversity(mut self, factor: f32) -> Self {
        self.diversity = factor.clamp(0.0, 1.0);
        self
    }

    /// Enable hybrid search.
    pub fn hybrid(mut self, keyword_weight: f32) -> Self {
        self.hybrid = true;
        self.keyword_weight = keyword_weight.clamp(0.0, 1.0);
        self
    }
}

/// Filter criteria for entries.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct Filter {
    /// Required categories (any match).
    pub categories: Vec<String>,

    /// Required tags (any match).
    pub tags: Vec<String>,

    /// Minimum access count.
    pub min_access_count: Option<u64>,

    /// Metadata key-value filters.
    pub metadata: Vec<(String, String)>,
}

#[allow(dead_code)]
impl Filter {
    /// Create an empty filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add category filter.
    pub fn category(mut self, category: impl Into<String>) -> Self {
        self.categories.push(category.into());
        self
    }

    /// Add tag filter.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set minimum access count.
    pub fn min_access(mut self, count: u64) -> Self {
        self.min_access_count = Some(count);
        self
    }

    /// Add metadata filter.
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    /// Check if an entry matches this filter.
    pub fn matches(&self, entry: &KnowledgeEntry) -> bool {
        // Check categories
        if !self.categories.is_empty() {
            if let Some(cat) = &entry.category {
                if !self.categories.iter().any(|c| c == cat) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check tags
        if !self.tags.is_empty()
            && !self
                .tags
                .iter()
                .any(|t| entry.tags.iter().any(|et| et == t))
        {
            return false;
        }

        // Check access count
        if let Some(min) = self.min_access_count {
            if entry.access_count < min {
                return false;
            }
        }

        // Check metadata
        for (key, value) in &self.metadata {
            if entry.metadata.get(key) != Some(value.as_str()) {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_options_builder() {
        let opts = SearchOptions::new(5)
            .with_min_similarity(0.5)
            .with_category("Programming")
            .with_tags(["rust", "tutorial"])
            .with_diversity(0.3)
            .hybrid(0.4);

        assert_eq!(opts.limit, 5);
        assert!((opts.min_similarity - 0.5).abs() < f32::EPSILON);
        assert_eq!(opts.category, Some("Programming".to_string()));
        assert_eq!(opts.tags, vec!["rust", "tutorial"]);
        assert!((opts.diversity - 0.3).abs() < f32::EPSILON);
        assert!(opts.hybrid);
        assert!((opts.keyword_weight - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_filter_matching() {
        let entry = KnowledgeEntry::new("Test", "Content")
            .with_category("Programming")
            .with_tags(["rust", "testing"]);

        let filter = Filter::new().category("Programming").tag("rust");

        assert!(filter.matches(&entry));

        let non_matching_filter = Filter::new().category("Other");
        assert!(!non_matching_filter.matches(&entry));
    }
}
