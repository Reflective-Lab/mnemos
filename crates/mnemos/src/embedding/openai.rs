//! OpenAI embedding provider.
//!
//! Uses the OpenAI API to generate high-quality text embeddings.
//! Supports text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002.
//!
//! # Features
//! - Automatic retries with exponential backoff
//! - In-memory caching to avoid re-embedding
//! - Rate limiting to respect API limits
//! - Token usage tracking
//! - Environment variable support for API key
//!
//! # Example
//! ```ignore
//! use mnemos::embedding::OpenAIEmbedding;
//!
//! // From environment variable OPENAI_API_KEY
//! let provider = OpenAIEmbedding::from_env()?;
//!
//! // Or with explicit key
//! let provider = OpenAIEmbedding::new("sk-...", None);
//!
//! let embedding = provider.embed("Hello, world!").await?;
//! ```

use super::EmbeddingProvider;
use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, warn};

/// OpenAI embedding models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpenAIModel {
    /// text-embedding-3-small: 1536 dimensions, lowest cost
    #[default]
    TextEmbedding3Small,
    /// text-embedding-3-large: 3072 dimensions, highest quality
    TextEmbedding3Large,
    /// text-embedding-ada-002: 1536 dimensions, legacy model
    TextEmbeddingAda002,
}

impl OpenAIModel {
    /// Get the model string for API requests.
    pub fn as_str(&self) -> &'static str {
        match self {
            OpenAIModel::TextEmbedding3Small => "text-embedding-3-small",
            OpenAIModel::TextEmbedding3Large => "text-embedding-3-large",
            OpenAIModel::TextEmbeddingAda002 => "text-embedding-ada-002",
        }
    }

    /// Get default dimensions for this model.
    pub fn default_dimensions(&self) -> usize {
        match self {
            OpenAIModel::TextEmbedding3Small | OpenAIModel::TextEmbeddingAda002 => 1536,
            OpenAIModel::TextEmbedding3Large => 3072,
        }
    }

    /// Whether this model supports custom dimensions.
    pub fn supports_custom_dimensions(&self) -> bool {
        matches!(
            self,
            OpenAIModel::TextEmbedding3Small | OpenAIModel::TextEmbedding3Large
        )
    }

    /// Parse from string.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "text-embedding-3-small" => Some(OpenAIModel::TextEmbedding3Small),
            "text-embedding-3-large" => Some(OpenAIModel::TextEmbedding3Large),
            "text-embedding-ada-002" => Some(OpenAIModel::TextEmbeddingAda002),
            _ => None,
        }
    }
}

/// Configuration for the OpenAI embedding provider.
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// The model to use.
    pub model: OpenAIModel,
    /// Custom dimensions (only for v3 models).
    pub dimensions: Option<usize>,
    /// Maximum retries for failed requests.
    pub max_retries: u32,
    /// Base delay for exponential backoff (milliseconds).
    pub retry_base_delay_ms: u64,
    /// Maximum concurrent requests.
    pub max_concurrent_requests: usize,
    /// Cache capacity (number of embeddings to cache).
    pub cache_capacity: usize,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// Custom API base URL (for proxies or Azure).
    pub api_base: Option<String>,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            model: OpenAIModel::default(),
            dimensions: None,
            max_retries: 3,
            retry_base_delay_ms: 1000,
            max_concurrent_requests: 10,
            cache_capacity: 10_000,
            timeout_secs: 30,
            api_base: None,
        }
    }
}

/// Token usage statistics.
#[derive(Debug, Default)]
pub struct UsageStats {
    /// Total prompt tokens used.
    pub prompt_tokens: AtomicU64,
    /// Total requests made.
    pub requests: AtomicU64,
    /// Cache hits.
    pub cache_hits: AtomicU64,
    /// Cache misses.
    pub cache_misses: AtomicU64,
    /// Failed requests.
    pub failures: AtomicU64,
}

impl UsageStats {
    /// Get a snapshot of current stats.
    pub fn snapshot(&self) -> UsageSnapshot {
        UsageSnapshot {
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            requests: self.requests.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            failures: self.failures.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of usage statistics.
#[derive(Debug, Clone)]
pub struct UsageSnapshot {
    /// Total prompt tokens used.
    pub prompt_tokens: u64,
    /// Total API requests made.
    pub requests: u64,
    /// Number of cache hits.
    pub cache_hits: u64,
    /// Number of cache misses.
    pub cache_misses: u64,
    /// Number of failed requests.
    pub failures: u64,
}

impl UsageSnapshot {
    /// Calculate cache hit rate.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }

    /// Estimate cost in USD (approximate, based on public pricing).
    pub fn estimated_cost_usd(&self, model: OpenAIModel) -> f64 {
        let cost_per_million = match model {
            OpenAIModel::TextEmbedding3Small => 0.02,
            OpenAIModel::TextEmbedding3Large => 0.13,
            OpenAIModel::TextEmbeddingAda002 => 0.10,
        };
        (self.prompt_tokens as f64 / 1_000_000.0) * cost_per_million
    }
}

/// Cached embedding entry.
struct CacheEntry {
    embedding: Vec<f32>,
    created_at: Instant,
}

/// LRU-ish cache for embeddings.
struct EmbeddingCache {
    entries: HashMap<String, CacheEntry>,
    capacity: usize,
    ttl: Duration,
}

impl EmbeddingCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            ttl: Duration::from_secs(3600), // 1 hour TTL
        }
    }

    fn get(&self, key: &str) -> Option<Vec<f32>> {
        self.entries.get(key).and_then(|entry| {
            if entry.created_at.elapsed() < self.ttl {
                Some(entry.embedding.clone())
            } else {
                None
            }
        })
    }

    fn insert(&mut self, key: String, embedding: Vec<f32>) {
        // Simple eviction: remove expired entries if at capacity
        if self.entries.len() >= self.capacity {
            self.evict_expired();
        }

        // If still at capacity, remove oldest
        if self.entries.len() >= self.capacity
            && let Some(oldest_key) = self
                .entries
                .iter()
                .min_by_key(|(_, v)| v.created_at)
                .map(|(k, _)| k.clone())
        {
            self.entries.remove(&oldest_key);
        }

        self.entries.insert(
            key,
            CacheEntry {
                embedding,
                created_at: Instant::now(),
            },
        );
    }

    fn evict_expired(&mut self) {
        self.entries
            .retain(|_, entry| entry.created_at.elapsed() < self.ttl);
    }
}

/// OpenAI embedding provider with production features.
pub struct OpenAIEmbedding {
    api_key: String,
    config: OpenAIConfig,
    client: reqwest::Client,
    cache: Arc<RwLock<EmbeddingCache>>,
    semaphore: Arc<Semaphore>,
    stats: Arc<UsageStats>,
    effective_dimensions: usize,
}

impl OpenAIEmbedding {
    /// Create a new OpenAI embedding provider.
    pub fn new(api_key: impl Into<String>, model: Option<String>) -> Self {
        let mut config = OpenAIConfig::default();
        if let Some(model_str) = model
            && let Some(model) = OpenAIModel::parse(&model_str)
        {
            config.model = model;
        }
        Self::with_config(api_key, config)
    }

    /// Create from OPENAI_API_KEY environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| Error::embedding("OPENAI_API_KEY environment variable not set"))?;
        Ok(Self::new(api_key, None))
    }

    /// Create with custom configuration.
    pub fn with_config(api_key: impl Into<String>, config: OpenAIConfig) -> Self {
        let effective_dimensions = config
            .dimensions
            .unwrap_or_else(|| config.model.default_dimensions());

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_max_idle_per_host(config.max_concurrent_requests)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            api_key: api_key.into(),
            effective_dimensions,
            cache: Arc::new(RwLock::new(EmbeddingCache::new(config.cache_capacity))),
            semaphore: Arc::new(Semaphore::new(config.max_concurrent_requests)),
            stats: Arc::new(UsageStats::default()),
            client,
            config,
        }
    }

    /// Set custom dimensions (for text-embedding-3-* models).
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        if self.config.model.supports_custom_dimensions() {
            self.config.dimensions = Some(dimensions);
            self.effective_dimensions = dimensions;
        }
        self
    }

    /// Get usage statistics.
    pub fn stats(&self) -> UsageSnapshot {
        self.stats.snapshot()
    }

    /// Get the API base URL.
    fn api_url(&self) -> String {
        self.config
            .api_base
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
            + "/embeddings"
    }

    /// Compute cache key for text.
    fn cache_key(&self, text: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.config.model.as_str().hash(&mut hasher);
        self.effective_dimensions.hash(&mut hasher);
        text.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Execute request with retries.
    async fn execute_with_retry(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let mut last_error = None;

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                let delay = self.config.retry_base_delay_ms * 2u64.pow(attempt - 1);
                debug!(attempt, delay_ms = delay, "Retrying after delay");
                tokio::time::sleep(Duration::from_millis(delay)).await;
            }

            // Acquire semaphore permit for rate limiting
            let _permit = self
                .semaphore
                .acquire()
                .await
                .map_err(|_| Error::embedding("Semaphore closed"))?;

            self.stats.requests.fetch_add(1, Ordering::Relaxed);

            match self.execute_request(request).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    warn!(attempt, error = %e, "Request failed");
                    self.stats.failures.fetch_add(1, Ordering::Relaxed);

                    // Don't retry on certain errors
                    if e.to_string().contains("invalid_api_key")
                        || e.to_string().contains("insufficient_quota")
                    {
                        return Err(e);
                    }

                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| Error::embedding("Unknown error")))
    }

    /// Execute a single request.
    async fn execute_request(&self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let response = self
            .client
            .post(self.api_url())
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| Error::embedding(format!("Request failed: {}", e)))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| Error::embedding(format!("Failed to read response: {}", e)))?;

        if !status.is_success() {
            let error: std::result::Result<ErrorResponse, _> = serde_json::from_str(&body);
            return Err(match error {
                Ok(e) => Error::embedding(format!("OpenAI API error: {}", e.error.message)),
                Err(_) => Error::embedding(format!("API error ({}): {}", status, body)),
            });
        }

        serde_json::from_str(&body)
            .map_err(|e| Error::embedding(format!("Failed to parse response: {}", e)))
    }

    /// Embed texts, using cache where possible.
    async fn embed_with_cache(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        // Check cache first
        {
            let cache = self.cache.read().await;
            for (i, text) in texts.iter().enumerate() {
                let key = self.cache_key(text);
                if let Some(embedding) = cache.get(&key) {
                    results[i] = Some(embedding);
                    self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
                } else {
                    uncached_indices.push(i);
                    uncached_texts.push(*text);
                    self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // Fetch uncached embeddings
        if !uncached_texts.is_empty() {
            debug!(
                count = uncached_texts.len(),
                cached = texts.len() - uncached_texts.len(),
                "Fetching embeddings from API"
            );

            let request = EmbeddingRequest {
                model: self.config.model.as_str().to_string(),
                input: uncached_texts
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect(),
                dimensions: if self.config.model.supports_custom_dimensions() {
                    Some(self.effective_dimensions)
                } else {
                    None
                },
            };

            let response = self.execute_with_retry(&request).await?;

            // Track token usage
            self.stats
                .prompt_tokens
                .fetch_add(response.usage.prompt_tokens as u64, Ordering::Relaxed);

            // Sort by index to maintain order
            let mut data = response.data;
            data.sort_by_key(|d| d.index);

            // Update cache and results
            {
                let mut cache = self.cache.write().await;
                for (data_idx, embedding_data) in data.into_iter().enumerate() {
                    let original_idx = uncached_indices[data_idx];
                    let text = uncached_texts[data_idx];
                    let key = self.cache_key(text);

                    cache.insert(key, embedding_data.embedding.clone());
                    results[original_idx] = Some(embedding_data.embedding);
                }
            }
        }

        // Unwrap all results (should all be Some now)
        results
            .into_iter()
            .enumerate()
            .map(|(i, opt)| {
                opt.ok_or_else(|| Error::embedding(format!("Missing embedding for index {}", i)))
            })
            .collect()
    }
}

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
    #[allow(dead_code)]
    model: String,
    usage: Usage,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: usize,
    #[allow(dead_code)]
    total_tokens: usize,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: ApiError,
}

#[derive(Deserialize)]
struct ApiError {
    message: String,
    #[allow(dead_code)]
    #[serde(rename = "type")]
    error_type: String,
}

#[async_trait::async_trait]
impl EmbeddingProvider for OpenAIEmbedding {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let embeddings = self.embed_with_cache(&[text]).await?;
        embeddings
            .into_iter()
            .next()
            .ok_or_else(|| Error::embedding("No embedding returned"))
    }

    async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        // OpenAI has a limit of ~8000 tokens per batch
        // Split into chunks of 100 texts to be safe
        const BATCH_SIZE: usize = 100;

        if texts.len() <= BATCH_SIZE {
            return self.embed_with_cache(texts).await;
        }

        let mut all_embeddings = Vec::with_capacity(texts.len());

        for chunk in texts.chunks(BATCH_SIZE) {
            let embeddings = self.embed_with_cache(chunk).await?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    fn dimensions(&self) -> usize {
        self.effective_dimensions
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_properties() {
        assert_eq!(OpenAIModel::TextEmbedding3Small.default_dimensions(), 1536);
        assert_eq!(OpenAIModel::TextEmbedding3Large.default_dimensions(), 3072);
        assert!(OpenAIModel::TextEmbedding3Small.supports_custom_dimensions());
        assert!(!OpenAIModel::TextEmbeddingAda002.supports_custom_dimensions());
    }

    #[test]
    fn test_model_parsing() {
        assert_eq!(
            OpenAIModel::parse("text-embedding-3-small"),
            Some(OpenAIModel::TextEmbedding3Small)
        );
        assert_eq!(OpenAIModel::parse("unknown-model"), None);
    }

    #[test]
    fn test_openai_dimensions() {
        let provider = OpenAIEmbedding::new("test-key", None);
        assert_eq!(provider.dimensions(), 1536);

        let provider = OpenAIEmbedding::new("test-key", Some("text-embedding-3-large".to_string()));
        assert_eq!(provider.dimensions(), 3072);
    }

    #[test]
    fn test_custom_dimensions() {
        let provider = OpenAIEmbedding::new("test-key", Some("text-embedding-3-small".to_string()))
            .with_dimensions(512);
        assert_eq!(provider.dimensions(), 512);

        // Ada doesn't support custom dimensions
        let provider = OpenAIEmbedding::new("test-key", Some("text-embedding-ada-002".to_string()))
            .with_dimensions(512);
        assert_eq!(provider.dimensions(), 1536); // Unchanged
    }

    #[test]
    fn test_config_defaults() {
        let config = OpenAIConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.max_concurrent_requests, 10);
        assert_eq!(config.cache_capacity, 10_000);
    }

    #[test]
    fn test_usage_stats() {
        let stats = UsageStats::default();
        stats.prompt_tokens.fetch_add(1000, Ordering::Relaxed);
        stats.cache_hits.fetch_add(80, Ordering::Relaxed);
        stats.cache_misses.fetch_add(20, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.prompt_tokens, 1000);
        assert!((snapshot.cache_hit_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_cost_estimation() {
        let snapshot = UsageSnapshot {
            prompt_tokens: 1_000_000,
            requests: 100,
            cache_hits: 50,
            cache_misses: 50,
            failures: 0,
        };

        let cost_small = snapshot.estimated_cost_usd(OpenAIModel::TextEmbedding3Small);
        let cost_large = snapshot.estimated_cost_usd(OpenAIModel::TextEmbedding3Large);

        assert!((cost_small - 0.02).abs() < 0.001);
        assert!((cost_large - 0.13).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_cache_operations() {
        let mut cache = EmbeddingCache::new(3);

        cache.insert("key1".to_string(), vec![1.0, 2.0, 3.0]);
        cache.insert("key2".to_string(), vec![4.0, 5.0, 6.0]);

        assert_eq!(cache.get("key1"), Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(cache.get("key2"), Some(vec![4.0, 5.0, 6.0]));
        assert_eq!(cache.get("key3"), None);

        // Test capacity eviction
        cache.insert("key3".to_string(), vec![7.0, 8.0, 9.0]);
        cache.insert("key4".to_string(), vec![10.0, 11.0, 12.0]);

        // One of the older keys should be evicted
        assert_eq!(cache.entries.len(), 3);
    }

    #[test]
    fn test_cache_key_consistency() {
        let provider = OpenAIEmbedding::new("test-key", None);

        let key1 = provider.cache_key("hello world");
        let key2 = provider.cache_key("hello world");
        let key3 = provider.cache_key("different text");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}
