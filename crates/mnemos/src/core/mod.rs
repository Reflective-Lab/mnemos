//! Core types and knowledge base implementation.

mod entry;
mod knowledge_base;
mod search;

pub use entry::{KnowledgeEntry, Metadata};
pub use knowledge_base::{KnowledgeBase, KnowledgeBaseConfig};
pub use search::{SearchOptions, SearchResult};
