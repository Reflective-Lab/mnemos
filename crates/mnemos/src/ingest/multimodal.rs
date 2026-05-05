//! Multi-modal embedding descriptors.
//!
//! This module provides storage/indexing descriptors for modality-specific
//! vectors without requiring a concrete embedding backend at this stage.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported content modalities for embedding/indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Text content.
    Text,
    /// Image content.
    Image,
    /// Audio content.
    Audio,
    /// Video content.
    Video,
}

impl Modality {
    /// Stable string name used in IDs and metadata.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
        }
    }
}

/// Optional temporal span describing a source interval for audio/video vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalSpan {
    /// Start timestamp in milliseconds.
    pub start_ms: u64,
    /// End timestamp in milliseconds.
    pub end_ms: u64,
}

impl TemporalSpan {
    /// Returns `true` when the span is non-empty.
    pub fn is_valid(&self) -> bool {
        self.end_ms > self.start_ms
    }
}

/// Location of a vector in a storage/index backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingLocation {
    /// Opaque vector identifier in the vector index/store.
    pub vector_id: String,
    /// Embedding model name or alias.
    pub model: String,
    /// Vector dimensionality.
    pub dimensions: usize,
}

impl EmbeddingLocation {
    /// Create a vector location descriptor.
    pub fn new(vector_id: impl Into<String>, model: impl Into<String>, dimensions: usize) -> Self {
        Self {
            vector_id: vector_id.into(),
            model: model.into(),
            dimensions,
        }
    }
}

/// Descriptor connecting a modality-specific vector to source content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingDescriptor {
    /// Modality this vector represents.
    pub modality: Modality,
    /// Logical source content ID (entry/chunk/asset identifier).
    pub source_content_id: String,
    /// Source sub-part identifier (e.g., block/chunk/frame/segment).
    pub part_id: Option<String>,
    /// Optional time span for audio/video-derived vectors.
    pub temporal_span: Option<TemporalSpan>,
    /// Where the vector is stored.
    pub location: EmbeddingLocation,
    /// Additional modality-specific metadata.
    pub metadata: HashMap<String, String>,
}

impl EmbeddingDescriptor {
    /// Create a new embedding descriptor.
    pub fn new(
        modality: Modality,
        source_content_id: impl Into<String>,
        location: EmbeddingLocation,
    ) -> Self {
        Self {
            modality,
            source_content_id: source_content_id.into(),
            part_id: None,
            temporal_span: None,
            location,
            metadata: HashMap::new(),
        }
    }

    /// Set the source part ID.
    pub fn with_part_id(mut self, part_id: impl Into<String>) -> Self {
        self.part_id = Some(part_id.into());
        self
    }

    /// Set the temporal span.
    pub fn with_temporal_span(mut self, temporal_span: TemporalSpan) -> Self {
        self.temporal_span = Some(temporal_span);
        self
    }

    /// Deterministic key for upsert/idempotent indexing.
    pub fn descriptor_key(&self) -> String {
        let part = self.part_id.as_deref().unwrap_or("root");
        match self.temporal_span {
            Some(span) => format!(
                "{}:{}:{}:{}-{}",
                self.modality.as_str(),
                self.source_content_id,
                part,
                span.start_ms,
                span.end_ms
            ),
            None => format!(
                "{}:{}:{}",
                self.modality.as_str(),
                self.source_content_id,
                part
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_key_includes_temporal_span_when_present() {
        let descriptor = EmbeddingDescriptor::new(
            Modality::Audio,
            "entry-1",
            EmbeddingLocation::new("vec-1", "mock-audio", 128),
        )
        .with_part_id("seg-3")
        .with_temporal_span(TemporalSpan {
            start_ms: 1_000,
            end_ms: 2_500,
        });

        assert_eq!(
            descriptor.descriptor_key(),
            "audio:entry-1:seg-3:1000-2500".to_string()
        );
    }

    #[test]
    fn temporal_span_validation_rejects_empty_spans() {
        assert!(
            !TemporalSpan {
                start_ms: 10,
                end_ms: 10
            }
            .is_valid()
        );
        assert!(
            TemporalSpan {
                start_ms: 10,
                end_ms: 11
            }
            .is_valid()
        );
    }
}
