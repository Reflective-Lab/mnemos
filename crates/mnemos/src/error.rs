//! Error types for the knowledge base.

use thiserror::Error;

/// Result type alias using the crate's Error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in the knowledge base.
#[derive(Error, Debug)]
pub enum Error {
    /// Entry not found in the knowledge base.
    #[error("Entry not found: {0}")]
    NotFound(String),

    /// Invalid embedding dimension.
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// The dimension the operation required.
        expected: usize,
        /// The dimension that was actually provided.
        actual: usize,
    },

    /// Storage backend error.
    #[error("Storage error: {0}")]
    Storage(String),

    /// Embedding generation error.
    #[error("Embedding error: {0}")]
    Embedding(String),

    /// Learning engine error.
    #[error("Learning error: {0}")]
    Learning(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(#[from] bincode::Error),

    /// JSON serialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    Config(String),

    /// Index corruption detected.
    #[error("Index corruption: {0}")]
    IndexCorruption(String),

    /// Concurrent access conflict.
    #[error("Concurrent access conflict: {0}")]
    ConcurrencyConflict(String),

    /// Ingest/parsing error.
    #[error("Ingest error: {0}")]
    Ingest(String),
}

impl Error {
    /// Create a storage error.
    pub fn storage(msg: impl Into<String>) -> Self {
        Self::Storage(msg.into())
    }

    /// Create an embedding error.
    pub fn embedding(msg: impl Into<String>) -> Self {
        Self::Embedding(msg.into())
    }

    /// Create a learning error.
    pub fn learning(msg: impl Into<String>) -> Self {
        Self::Learning(msg.into())
    }

    /// Create a not found error.
    pub fn not_found(id: impl Into<String>) -> Self {
        Self::NotFound(id.into())
    }

    /// Create a config error.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    /// Create an ingest error.
    pub fn ingest(msg: impl Into<String>) -> Self {
        Self::Ingest(msg.into())
    }
}
