//! # mnemos
//!
//! A self-learning knowledgebase built on ruvector that gets smarter the more you use it.
//!
//! ## Features
//!
//! - **Vector Storage**: High-performance HNSW-based vector indexing
//! - **Self-Learning**: Adaptive query understanding using GNN-inspired learning
//! - **Knowledge Graph**: Semantic relationships between knowledge entries
//! - **Hybrid Search**: Combine vector similarity with BM25 lexical recall using
//!   Reciprocal Rank Fusion
//! - **gRPC Interface**: High-performance RPC for service integration
//! - **MCP Server**: Model Context Protocol for Claude Desktop
//! - **Suggestor Adapters**: Knowledge retrieval and persistence inside the
//!   convergence loop
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use mnemos::{KnowledgeBase, KnowledgeEntry};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let kb = KnowledgeBase::open("./knowledge.db").await?;
//!
//!     // Add knowledge
//!     kb.add_entry(KnowledgeEntry::new(
//!         "Rust Memory Safety",
//!         "Rust ensures memory safety through ownership and borrowing rules...",
//!     )).await?;
//!
//!     // Search with learning
//!     let results = kb.search_simple("memory management in rust", 5).await?;
//!
//!     // Search with hybrid vector + BM25 rank fusion
//!     let _hybrid = kb.search(
//!         "BM25 and RRF",
//!         mnemos::SearchOptions::new(5).hybrid(0.3),
//!     ).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    mnemos                        │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │
//! │  │   CLI   │  │  gRPC   │  │   MCP   │  │  Library API    │ │
//! │  │         │  │ Server  │  │ Server  │  │                 │ │
//! │  └────┬────┘  └────┬────┘  └────┬────┘  └────────┬────────┘ │
//! │       │            │            │                │          │
//! │       └────────────┴────────────┴────────────────┘          │
//! │                           │                                  │
//! │  ┌────────────────────────┴───────────────────────────────┐ │
//! │  │                   KnowledgeBase                        │ │
//! │  │  ┌─────────────┐  ┌───────────────┐  ┌──────────────┐ │ │
//! │  │  │  Embedding  │  │   Learning    │  │   Storage    │ │ │
//! │  │  │   Engine    │  │    Engine     │  │   Backend    │ │ │
//! │  │  │  (Hash/ML)  │  │  (GNN-style)  │  │  (Bincode)   │ │ │
//! │  │  └─────────────┘  └───────────────┘  └──────────────┘ │ │
//! │  └────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod agentic;
pub mod core;
pub mod embedding;
pub mod error;
pub mod ingest;
pub mod learning;
pub(crate) mod math;
pub mod provenance;
pub mod storage;
pub mod suggestor;

#[cfg(feature = "grpc")]
pub mod grpc;

// Re-exports
pub use crate::agentic::{
    // Core agent memory
    AgenticDB,
    AgenticStats,
    CausalEdge,
    // Causal memory
    CausalMemory,
    CausalNode,
    Critique,
    CritiqueType,
    DriftDetector,
    Experience,
    ExperienceWindow,
    FewShotLearner,
    Hyperedge,
    // Learning sessions
    LearningSession,
    LearningStrategy,
    // Meta-learning
    MetaLearner,
    // Online/continual learning
    OnlineLearner,
    ParameterSnapshot,
    // Reflexion (self-critique)
    ReflexionEpisode,
    ReflexionMemory,
    Reward,
    SessionTurn,
    // Skills
    Skill,
    SkillLibrary,
    SkillPattern,
    TaskFeatures,
    TemporalMemory,
    TemporalOccurrence,
    TemporalPeriod,
    // Temporal patterns (time crystals)
    TimeCrystal,
};
pub use crate::core::{
    KnowledgeBase, KnowledgeBaseConfig, KnowledgeEntry, SearchOptions, SearchResult,
};
pub use crate::embedding::EmbeddingEngine;
pub use crate::error::{Error, Result};
pub use crate::learning::LearningEngine;
pub use crate::provenance::{MNEMOS_PROVENANCE, Mnemos};
pub use crate::storage::StorageBackend;
pub use crate::suggestor::{
    KnowledgeHitPayload, KnowledgeRetrievalSuggestor, KnowledgeStoreSuggestor,
};
