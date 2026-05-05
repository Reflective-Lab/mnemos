//! gRPC client for the knowledge base.

use super::knowledge_service_client::KnowledgeServiceClient as ProtoClient;
use super::*;
use crate::error::{Error, Result};

use tonic::transport::Channel;

/// gRPC client for interacting with the knowledge base server.
pub struct KnowledgeClient {
    client: ProtoClient<Channel>,
}

impl KnowledgeClient {
    /// Connect to a knowledge base server.
    pub async fn connect(addr: impl Into<String>) -> Result<Self> {
        let addr = addr.into();
        let client = ProtoClient::connect(addr)
            .await
            .map_err(|e| Error::storage(format!("Failed to connect: {}", e)))?;

        Ok(Self { client })
    }

    /// Add a new knowledge entry.
    pub async fn add_entry(
        &mut self,
        title: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<String> {
        let request = AddEntryRequest {
            title: title.into(),
            content: content.into(),
            category: None,
            tags: Vec::new(),
            source: None,
            metadata: std::collections::HashMap::new(),
        };

        let response = self
            .client
            .add_entry(request)
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        if response.success {
            Ok(response.id)
        } else {
            Err(Error::storage(
                response
                    .error
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ))
        }
    }

    /// Add entry with full options.
    pub async fn add_entry_full(&mut self, request: AddEntryRequest) -> Result<String> {
        let response = self
            .client
            .add_entry(request)
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        if response.success {
            Ok(response.id)
        } else {
            Err(Error::storage(
                response
                    .error
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ))
        }
    }

    /// Add multiple entries in batch.
    pub async fn add_entries(&mut self, entries: Vec<AddEntryRequest>) -> Result<Vec<String>> {
        let request = AddEntriesRequest { entries };

        let response = self
            .client
            .add_entries(request)
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        if response.success {
            Ok(response.ids)
        } else {
            Err(Error::storage(
                response
                    .error
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ))
        }
    }

    /// Get an entry by ID.
    pub async fn get_entry(&mut self, id: impl Into<String>) -> Result<Option<KnowledgeEntry>> {
        let request = GetEntryRequest { id: id.into() };

        let response = self
            .client
            .get_entry(request)
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        if response.found {
            Ok(response.entry)
        } else {
            Ok(None)
        }
    }

    /// Update an existing entry.
    pub async fn update_entry(&mut self, request: UpdateEntryRequest) -> Result<()> {
        let response = self
            .client
            .update_entry(request)
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        if response.success {
            Ok(())
        } else {
            Err(Error::storage(
                response
                    .error
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ))
        }
    }

    /// Delete an entry.
    pub async fn delete_entry(&mut self, id: impl Into<String>) -> Result<()> {
        let request = DeleteEntryRequest { id: id.into() };

        let response = self
            .client
            .delete_entry(request)
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        if response.success {
            Ok(())
        } else {
            Err(Error::storage(
                response
                    .error
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ))
        }
    }

    /// Search for similar entries.
    pub async fn search(
        &mut self,
        query: impl Into<String>,
        limit: u32,
    ) -> Result<Vec<SearchResult>> {
        let request = SearchRequest {
            query: query.into(),
            limit,
            min_similarity: 0.0,
            category: None,
            tags: Vec::new(),
            use_learning: true,
            include_related: false,
            diversity: 0.0,
            hybrid: false,
            keyword_weight: 0.3,
        };

        let response = self
            .client
            .search(request)
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        Ok(response.results)
    }

    /// Search with full options.
    pub async fn search_full(&mut self, request: SearchRequest) -> Result<SearchResponse> {
        let response = self
            .client
            .search(request)
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        Ok(response)
    }

    /// Record feedback on a search result.
    pub async fn record_feedback(
        &mut self,
        entry_id: impl Into<String>,
        positive: bool,
    ) -> Result<()> {
        let request = FeedbackRequest {
            entry_id: entry_id.into(),
            positive,
            query_context: None,
        };

        let response = self
            .client
            .record_feedback(request)
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        if response.success {
            Ok(())
        } else {
            Err(Error::storage("Failed to record feedback"))
        }
    }

    /// Get related entries.
    pub async fn get_related(
        &mut self,
        id: impl Into<String>,
        limit: u32,
    ) -> Result<Vec<KnowledgeEntry>> {
        let request = GetRelatedRequest {
            id: id.into(),
            limit,
        };

        let response = self
            .client
            .get_related(request)
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        Ok(response.entries)
    }

    /// Link two entries as related.
    pub async fn link_entries(
        &mut self,
        id1: impl Into<String>,
        id2: impl Into<String>,
    ) -> Result<()> {
        let request = LinkEntriesRequest {
            id1: id1.into(),
            id2: id2.into(),
        };

        let response = self
            .client
            .link_entries(request)
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        if response.success {
            Ok(())
        } else {
            Err(Error::storage(
                response
                    .error
                    .unwrap_or_else(|| "Unknown error".to_string()),
            ))
        }
    }

    /// Get knowledge base statistics.
    pub async fn get_stats(&mut self) -> Result<GetStatsResponse> {
        let response = self
            .client
            .get_stats(GetStatsRequest {})
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        Ok(response)
    }

    /// Health check.
    pub async fn health(&mut self) -> Result<HealthResponse> {
        let response = self
            .client
            .health(HealthRequest {})
            .await
            .map_err(|e| Error::storage(e.to_string()))?
            .into_inner();

        Ok(response)
    }
}
