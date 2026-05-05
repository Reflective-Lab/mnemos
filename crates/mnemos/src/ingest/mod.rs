//! Document ingestion module for extracting content from various file formats.
//!
//! This module provides parsers and ingesters for different document types,
//! converting them into structured data suitable for the knowledge base.
//!
//! # Supported Formats
//!
//! - **Markdown** (`markdown.rs`): Parse markdown files with YAML front-matter,
//!   chunk by headings, and preserve code blocks with language tags.
//! - PDF extraction has moved to `organism-intelligence`.
//!
//! # Knowledge Classification
//!
//! The module also includes a routing system for classifying knowledge:
//!
//! - **Case Knowledge**: Direct, actionable knowledge relevant to the current task.
//!   Has high relevance, is recently accessed, and is explicitly linked to current context.
//!
//! - **Background Knowledge**: Contextual knowledge that supports understanding.
//!   Provides context, general reference material, and supports case knowledge.
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
//! // Create a router with rules
//! let mut router = KnowledgeRouter::new();
//!
//! // Route project files as case knowledge
//! router.add_rule(RoutingRule::new(
//!     RoutingCondition::SourcePath("projects/**/*".to_string()),
//!     KnowledgeTypeHint::Case {
//!         context: "active-project".to_string(),
//!         access_pattern: AccessPattern::ActiveUse,
//!     },
//! ));
//!
//! // Classify incoming knowledge
//! let metadata = HashMap::new();
//! let knowledge = router.classify(
//!     Path::new("projects/my-app/README.md"),
//!     "Project documentation...",
//!     &metadata,
//! );
//! ```

mod markdown;
mod multimodal;
mod rich_media;
pub mod routing;
mod source;

pub use markdown::{ChunkType, IngesterConfig, MarkdownChunk, MarkdownDocument, MarkdownIngester};
pub use multimodal::{EmbeddingDescriptor, EmbeddingLocation, Modality, TemporalSpan};
pub use rich_media::{
    MediaIngestRequest, MediaKind, TranscriptChunkPolicy, TranscriptDocument, TranscriptSegment,
    TranscriptionBackend, TranscriptionEngine, TranscriptionRequest,
};
pub use source::{SourceKind, SourceProvenance};

pub use routing::{
    AccessPattern, BackgroundKnowledge, CaseKnowledge, KnowledgeRouter, KnowledgeType,
    KnowledgeTypeHint, Permanence, RoutingCondition, RoutingRule, ScoringWeights,
};
