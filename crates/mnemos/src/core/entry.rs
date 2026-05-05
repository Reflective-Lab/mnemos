//! Knowledge entry types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A single entry in the knowledge base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeEntry {
    /// Unique identifier for the entry.
    pub id: Uuid,

    /// Title or short summary.
    pub title: String,

    /// Full content of the entry.
    pub content: String,

    /// Category or type classification.
    pub category: Option<String>,

    /// Tags for categorization.
    pub tags: Vec<String>,

    /// Source URL or reference.
    pub source: Option<String>,

    /// Custom metadata.
    pub metadata: Metadata,

    /// Creation timestamp.
    pub created_at: DateTime<Utc>,

    /// Last update timestamp.
    pub updated_at: DateTime<Utc>,

    /// Access count for learning.
    pub access_count: u64,

    /// Relevance score from learning.
    pub learned_relevance: f32,

    /// Related entry IDs (knowledge graph).
    pub related_entries: Vec<Uuid>,
}

impl KnowledgeEntry {
    /// Create a new knowledge entry.
    pub fn new(title: impl Into<String>, content: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            title: title.into(),
            content: content.into(),
            category: None,
            tags: Vec::new(),
            source: None,
            metadata: Metadata::new(),
            created_at: now,
            updated_at: now,
            access_count: 0,
            learned_relevance: 1.0,
            related_entries: Vec::new(),
        }
    }

    /// Set the category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Add tags.
    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    /// Set the source.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Add a related entry.
    pub fn with_related(mut self, related_id: Uuid) -> Self {
        self.related_entries.push(related_id);
        self
    }

    /// Get the combined text for embedding.
    pub fn embedding_text(&self) -> String {
        let mut parts = vec![self.title.clone(), self.content.clone()];

        if let Some(category) = &self.category {
            parts.push(category.clone());
        }

        if !self.tags.is_empty() {
            parts.push(self.tags.join(" "));
        }

        parts.join(" ")
    }

    /// Record an access and update relevance.
    pub fn record_access(&mut self, relevance_boost: f32) {
        self.access_count += 1;
        self.learned_relevance = (self.learned_relevance + relevance_boost) / 2.0;
        self.updated_at = Utc::now();
    }
}

/// Custom metadata for knowledge entries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metadata {
    data: HashMap<String, String>,
}

impl Metadata {
    /// Create empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a key-value pair.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.data.insert(key.into(), value.into());
    }

    /// Get a value by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(String::as_str)
    }

    /// Remove a key.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.data.remove(key)
    }

    /// Iterate over all key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.data.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_creation() {
        let entry = KnowledgeEntry::new("Test Title", "Test content")
            .with_category("Testing")
            .with_tags(["rust", "testing"])
            .with_source("https://example.com")
            .with_metadata("author", "test");

        assert_eq!(entry.title, "Test Title");
        assert_eq!(entry.content, "Test content");
        assert_eq!(entry.category, Some("Testing".to_string()));
        assert_eq!(entry.tags, vec!["rust", "testing"]);
        assert_eq!(entry.source, Some("https://example.com".to_string()));
        assert_eq!(entry.metadata.get("author"), Some("test"));
    }

    #[test]
    fn test_embedding_text() {
        let entry = KnowledgeEntry::new("Rust Guide", "A guide to Rust programming")
            .with_category("Programming")
            .with_tags(["rust", "guide"]);

        let text = entry.embedding_text();
        assert!(text.contains("Rust Guide"));
        assert!(text.contains("A guide to Rust programming"));
        assert!(text.contains("Programming"));
        assert!(text.contains("rust guide"));
    }

    #[test]
    fn test_access_recording() {
        let mut entry = KnowledgeEntry::new("Test", "Content");
        let initial_relevance = entry.learned_relevance;

        entry.record_access(1.5);

        assert_eq!(entry.access_count, 1);
        assert!((entry.learned_relevance - (initial_relevance + 1.5) / 2.0).abs() < f32::EPSILON);
    }
}
