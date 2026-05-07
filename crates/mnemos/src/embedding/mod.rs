//! Embedding generation for text vectorization.
//!
//! Supports multiple embedding backends:
//! - Hash-based (default, for testing and offline use)
//! - OpenAI API (production, high-quality embeddings)
//!
//! # Choosing a Backend
//!
//! | Backend | Quality | Speed | Cost | Offline |
//! |---------|---------|-------|------|---------|
//! | Hash    | Low     | Fast  | Free | Yes     |
//! | OpenAI  | High    | Medium| Paid | No      |
//!
//! # Example
//! ```ignore
//! use mnemos::embedding::EmbeddingEngine;
//!
//! // Development/testing: use hash embeddings
//! let engine = EmbeddingEngine::new(256);
//!
//! // Production: use OpenAI (reads OPENAI_API_KEY)
//! let engine = EmbeddingEngine::from_env()?;
//!
//! // Production with explicit key
//! let engine = EmbeddingEngine::with_openai("sk-...", None);
//! ```

mod openai;

pub use openai::{OpenAIConfig, OpenAIEmbedding, OpenAIModel, UsageSnapshot, UsageStats};

use crate::error::{Error, Result};
use std::any::Any;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Embedding provider trait for different backends.
#[async_trait::async_trait]
pub trait EmbeddingProvider: Any + Send + Sync {
    /// Generate embedding for text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts in batch.
    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.embed(text).await?);
        }
        Ok(embeddings)
    }

    /// Get embedding dimensions.
    fn dimensions(&self) -> usize;

    /// Downcast support for provider-specific fast paths.
    fn as_any(&self) -> &dyn Any;
}

/// Embedding engine for converting text to vectors.
///
/// Wraps different embedding providers with a unified interface.
pub struct EmbeddingEngine {
    provider: Box<dyn EmbeddingProvider>,
}

impl EmbeddingEngine {
    /// Create a new embedding engine with hash-based embeddings.
    ///
    /// Use this for development, testing, or offline scenarios.
    /// Hash embeddings are fast and free but lower quality.
    pub fn new(dimensions: usize) -> Self {
        Self {
            provider: Box::new(HashEmbedding::new(dimensions)),
        }
    }

    /// Create from environment variables.
    ///
    /// Reads OPENAI_API_KEY. Falls back to hash embeddings if not set.
    pub fn from_env() -> Self {
        if let Ok(provider) = OpenAIEmbedding::from_env() { Self {
            provider: Box::new(provider),
        } } else {
            tracing::warn!("OPENAI_API_KEY not set, falling back to hash embeddings");
            Self::new(1536) // Match OpenAI default dimensions
        }
    }

    /// Create from environment, returning error if not configured.
    pub fn from_env_required() -> Result<Self> {
        let provider = OpenAIEmbedding::from_env()?;
        Ok(Self {
            provider: Box::new(provider),
        })
    }

    /// Create with OpenAI embeddings.
    pub fn with_openai(api_key: impl Into<String>, model: Option<String>) -> Self {
        Self {
            provider: Box::new(OpenAIEmbedding::new(api_key, model)),
        }
    }

    /// Create with OpenAI using custom configuration.
    pub fn with_openai_config(api_key: impl Into<String>, config: OpenAIConfig) -> Self {
        Self {
            provider: Box::new(OpenAIEmbedding::with_config(api_key, config)),
        }
    }

    /// Create with a custom provider.
    pub fn with_provider(provider: Box<dyn EmbeddingProvider>) -> Self {
        Self { provider }
    }

    /// Get the embedding dimensions.
    pub fn dimensions(&self) -> usize {
        self.provider.dimensions()
    }

    /// Generate an embedding for the given text.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.provider.embed(text).await
    }

    /// Generate embeddings for multiple texts.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.provider.embed_batch(texts).await
    }

    /// Compute similarity between two embeddings.
    pub fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

/// Hash-based embedding for testing and development.
pub struct HashEmbedding {
    dimensions: usize,
}

impl HashEmbedding {
    /// Create a new hash embedding engine.
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }

    fn embed_sync(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::embedding("Cannot embed empty text"));
        }

        let mut embedding = vec![0.0f32; self.dimensions];
        let normalized_text = text.to_lowercase();

        // Hash individual words
        for word in normalized_text.split_whitespace() {
            self.add_word_embedding(&mut embedding, word, 1.0);
        }

        // Hash bigrams for context
        let words: Vec<&str> = normalized_text.split_whitespace().collect();
        for window in words.windows(2) {
            let bigram = format!("{} {}", window[0], window[1]);
            self.add_word_embedding(&mut embedding, &bigram, 0.5);
        }

        // Hash trigrams for more context
        for window in words.windows(3) {
            let trigram = format!("{} {} {}", window[0], window[1], window[2]);
            self.add_word_embedding(&mut embedding, &trigram, 0.3);
        }

        // Character-level features for typo tolerance
        for word in &words {
            for char_ngram in word.as_bytes().windows(3) {
                let hash = self.hash_bytes(char_ngram);
                let idx = (hash as usize) % self.dimensions;
                embedding[idx] += 0.1;
            }
        }

        // Normalize to unit length
        self.normalize(&mut embedding);

        Ok(embedding)
    }

    fn add_word_embedding(&self, embedding: &mut [f32], text: &str, weight: f32) {
        let hash = self.hash_text(text);
        for i in 0..8 {
            let idx = ((hash.wrapping_add(i * 0x9e37_79b9)) as usize) % self.dimensions;
            let sign = if (hash >> i) & 1 == 0 { 1.0 } else { -1.0 };
            embedding[idx] += sign * weight;
        }
    }

    fn hash_text(&self, text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    fn hash_bytes(&self, bytes: &[u8]) -> u64 {
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        hasher.finish()
    }

    fn normalize(&self, embedding: &mut [f32]) {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in embedding.iter_mut() {
                *x /= norm;
            }
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for HashEmbedding {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_sync(text)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embedding_dimensions() {
        let engine = EmbeddingEngine::new(128);
        let embedding = engine.embed("test text").await.unwrap();
        assert_eq!(embedding.len(), 128);
    }

    #[tokio::test]
    async fn test_embedding_consistency() {
        let engine = EmbeddingEngine::new(64);
        let emb1 = engine.embed("hello world").await.unwrap();
        let emb2 = engine.embed("hello world").await.unwrap();
        assert_eq!(emb1, emb2);
    }

    #[tokio::test]
    async fn test_embedding_similarity() {
        let engine = EmbeddingEngine::new(128);

        let emb1 = engine.embed("rust programming language").await.unwrap();
        let emb2 = engine.embed("rust programming").await.unwrap();
        let emb3 = engine.embed("cooking recipes").await.unwrap();

        let sim_similar = engine.similarity(&emb1, &emb2);
        let sim_different = engine.similarity(&emb1, &emb3);

        assert!(sim_similar > sim_different);
    }

    #[tokio::test]
    async fn test_normalized_embeddings() {
        let engine = EmbeddingEngine::new(256);
        let embedding = engine.embed("some text here").await.unwrap();

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_empty_text_error() {
        let engine = EmbeddingEngine::new(64);
        assert!(engine.embed("").await.is_err());
    }
}
